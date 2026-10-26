"""Stateless FastAPI service around the async SpeechLM engine.

The service owns the GPUs and holds **no dialogue state** — every ``/generate``
call carries the full wire-form prefix. Multi-hop reuse across calls is pure
vLLM prefix caching (keyed by the stable audio uuids), never a session (the
plan's stateless requirement; also the shape verl's rollout wraps). Audio is the
only server-side state, and it is an idempotent content-addressed *cache*, not
per-request state: register once, reference by ``(audio_id, start, end)`` for
every hop.

Endpoints:
- ``GET  /health``       — liveness + registered-audio count.
- ``GET  /models``       — engine configuration.
- ``POST /audio``        — register a shared-FS path → ``audio_id``.
- ``POST /audio/upload`` — register uploaded bytes → ``audio_id`` (spooled).
- ``POST /generate``     — one assistant segment (wire prompt + audio refs).
- ``POST /score``        — teacher-forced ``prompt_logprobs`` (alignment/verl).

``/generate`` mirrors ``LocalEngine`` exactly once the audio refs are resolved
to waveforms, so a served decode equals the offline decode. The audio-seed count
in the wire prompt must equal the number of audio segments (each audio is one
``id 0`` seed the plugin expands); a mismatch is a 422 with the expected count.

Run (vllm env), single dense replica on one GPU::

    CUDA_VISIBLE_DEVICES=0 envs/vllm/bin/python -m espnet2.speechlm.vllm.vllm_server.service \\
        --model artifacts/model_0.8b_step_50000 --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field
from starlette.requests import Request
from fastapi import UploadFile

from ..vllm_client import constants as C
from ..vllm_client.protocol import GenerateRequest
from .async_engine import AsyncSpeechLMEngine
from .audio_registry import AudioRegistry

# Module-level singletons populated at startup (one engine per process).
_ENGINE: AsyncSpeechLMEngine | None = None
_REGISTRY: AudioRegistry | None = None
_CONFIG: dict = {}


class _ClientDisconnected(RuntimeError):
    """The HTTP peer disappeared while an engine request was in flight."""


async def _await_disconnect_aware(
    request: Request, operation, *, poll_seconds: float = 0.1
):
    """Await an engine operation and cancel it promptly on HTTP disconnect.

    Uvicorn does not cancel a running FastAPI handler when its client goes
    away.  Cancelling the task here is important: vLLM's async generator
    catches generator cancellation and aborts the corresponding scheduler
    request, releasing its KV-cache allocation instead of decoding an answer
    that no client can receive.

    The ``finally`` also propagates cancellation in the other direction.  If
    the ASGI server cancels this handler during shutdown, its vLLM request is
    cancelled as well.
    """
    task = asyncio.create_task(operation)
    try:
        while True:
            if task.done():
                return task.result()
            disconnected = await request.is_disconnected()
            # Prefer an already-completed engine result over a disconnect that
            # raced with its final token.
            if task.done():
                return task.result()
            if disconnected:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                raise _ClientDisconnected("HTTP client disconnected")
            done, _ = await asyncio.wait({task}, timeout=poll_seconds)
            if done:
                return task.result()
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


# Request/response schemas. These MUST live at module scope: with
# ``from __future__ import annotations`` the endpoint annotations are lazy
# strings, and FastAPI resolves them via ``get_type_hints`` against the
# function's module globals. Nested (local) models are invisible there, so
# FastAPI would silently demote a body param to a query param (422 "field
# required" on ["query","body"]).
class SegmentRef(BaseModel):
    audio_id: str
    start_sample: int = 0
    end_sample: int | None = None


class Sampling(BaseModel):
    max_tokens: int = 4096
    min_tokens: int = 1
    temperature: float = 0.0
    top_k: int = -1
    mode: str = "content"  # content | choice | none
    repetition_penalty: float = Field(default=1.0, gt=0, allow_inf_nan=False)


class PathBody(BaseModel):
    path: str


class GenerateBody(BaseModel):
    prompt_token_ids: list[int]
    audio_segments: list[SegmentRef] = Field(default_factory=list)
    sampling: Sampling = Field(default_factory=Sampling)


class GenerateResponse(BaseModel):
    token_ids: list[int]
    terminator: int | None
    finish_reason: str


class ScoreBody(BaseModel):
    prompt_token_ids: list[int]
    audio_segments: list[SegmentRef] = Field(default_factory=list)
    top_logprobs: int = 32
    # Return only the last `tail` positions (0 = all). The engine still scores
    # the full sequence; this only trims the JSON payload — a 93k-token deep-
    # context probe would otherwise ship tens of MB of logprobs per call.
    tail: int = 0


class ScoreResponse(BaseModel):
    # token_id keyed by str (JSON object keys are strings) -> logprob.
    prompt_logprobs: list[dict[str, float] | None]
    num_positions: int = 0


def _build_app(engine: AsyncSpeechLMEngine, registry: AudioRegistry, config: dict):
    from fastapi import FastAPI, HTTPException

    io_threads = config.get("io_threads", 0)
    io_pool = ThreadPoolExecutor(max_workers=io_threads, thread_name_prefix="longcap-audio") if io_threads > 0 else None

    @contextlib.asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            if io_pool is not None:
                io_pool.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="LongCap audio captioner", version="0.1.0", lifespan=lifespan)

    def _resolve_segments(segs: list[SegmentRef]):
        """Registry refs -> (waveforms, uuids). Raises HTTPException on miss."""
        waves, uuids = [], []
        for s in segs:
            try:
                wav = registry.slice_16k(s.audio_id, s.start_sample, s.end_sample)
            except KeyError:
                raise HTTPException(
                    status_code=404, detail=f"unknown audio_id {s.audio_id!r}"
                )
            waves.append((wav, C.AUDIO_SAMPLE_RATE))
            uuids.append(f"{s.audio_id}:{s.start_sample}:{s.end_sample}")
        return waves, uuids

    def _check_seeds(prompt_token_ids, n_segments):
        n_seed = sum(1 for t in prompt_token_ids if t == C.AUDIO_PLACEHOLDER)
        if n_seed != n_segments:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "audio seed / segment count mismatch",
                    "audio_seeds_in_prompt": n_seed,
                    "audio_segments": n_segments,
                },
            )

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": config["model"],
                "num_registered": registry.num_registered,
                "slice_cache": {"hits": registry.slice_hits,
                                "misses": registry.slice_misses,
                                "max_entries": registry._slice_cache_max}}

    @app.get("/models")
    async def models():
        return config

    @app.post("/audio")
    async def register_audio(body: PathBody):
        if not config.get("allow_audio_paths", False):
            raise HTTPException(status_code=403, detail="Use /audio/upload; shared-path access is disabled")
        try:
            return registry.register_path(body.path)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"no such file {body.path!r}")

    @app.post("/audio/upload")
    async def upload_audio(file: UploadFile):
        data = await file.read()
        suffix = os.path.splitext(file.filename or "")[1] or ".wav"
        return registry.register_bytes(data, suffix=suffix)

    async def _resolve_segments_async(segs: list[SegmentRef]):
        """Run segment resolution off the event loop when --io-threads > 0.

        slice_16k does disk reads + torchaudio sinc resampling — measured tens
        of seconds per deep multi-hop request when the slice LRU thrashes —
        and running it inline stalls every other in-flight dialogue.
        """
        if io_pool is not None:
            return await asyncio.get_running_loop().run_in_executor(io_pool, _resolve_segments, segs)
        return _resolve_segments(segs)

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: Request, body: GenerateBody):
        _check_seeds(body.prompt_token_ids, len(body.audio_segments))
        waves, uuids = await _resolve_segments_async(body.audio_segments)
        req = GenerateRequest(
            prompt_token_ids=list(body.prompt_token_ids),
            audio=waves,
            audio_uuids=uuids or None,
            max_tokens=body.sampling.max_tokens,
            min_tokens=body.sampling.min_tokens,
            temperature=body.sampling.temperature,
            top_k=body.sampling.top_k,
            mode=body.sampling.mode,
            repetition_penalty=body.sampling.repetition_penalty,
        )
        try:
            res = await _await_disconnect_aware(request, engine.generate(req))
        except _ClientDisconnected as exc:
            # nginx convention for a client-closed request.  The response is
            # normally unobservable because the peer is already gone, but the
            # explicit status is useful to ASGI tests and access logs.
            raise HTTPException(status_code=499, detail=str(exc)) from exc
        return GenerateResponse(
            token_ids=res.token_ids,
            terminator=res.terminator,
            finish_reason=res.finish_reason,
        )

    @app.post("/score", response_model=ScoreResponse)
    async def score(request: Request, body: ScoreBody):
        _check_seeds(body.prompt_token_ids, len(body.audio_segments))
        waves, uuids = await _resolve_segments_async(body.audio_segments)
        req = GenerateRequest(
            prompt_token_ids=list(body.prompt_token_ids),
            audio=waves,
            audio_uuids=uuids or None,
            mode="content",
        )
        try:
            res = await _await_disconnect_aware(
                request, engine.score(req, body.top_logprobs)
            )
        except _ClientDisconnected as exc:
            raise HTTPException(status_code=499, detail=str(exc)) from exc
        rows = [
            None if pos is None else {str(tid): lp for tid, lp in pos.items()}
            for pos in res.prompt_logprobs
        ]
        total = len(rows)
        if body.tail and body.tail > 0:
            rows = rows[-body.tail:]
        return ScoreResponse(prompt_logprobs=rows, num_positions=total)

    return app


def build_app_from_args(args) -> "object":
    """Construct engine + registry + FastAPI app (used by __main__ and tests)."""
    from pathlib import Path
    from longcap.identity import checkpoint_identity
    from longcap.server_identity import serving_runtime_identity
    checkpoint = checkpoint_identity(Path(args.model))
    global _ENGINE, _REGISTRY, _CONFIG
    if getattr(args, "torch_num_threads", 0) > 0:
        # The API process does CPU featurization (torchaudio resample + mel).
        # torch's default intra-op pool (= all cores) thrashes when several
        # service replicas share one node; a modest cap is faster in practice.
        import torch
        torch.set_num_threads(args.torch_num_threads)
    engine = AsyncSpeechLMEngine(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_audio_per_prompt=args.max_audio_per_prompt,
        enable_prefix_caching=args.apc,
        enforce_eager=args.enforce_eager,
        dtype=args.dtype,
        seed=args.seed,
        max_num_batched_tokens=getattr(args, "max_num_batched_tokens", None),
        mamba_ssm_cache_dtype=getattr(args, "mamba_ssm_cache_dtype", "auto"),
        mm_processor_cache_gb=getattr(args, "mm_processor_cache_gb", None),
        async_scheduling=(True if getattr(args, "async_scheduling", False) else None),
        stats_jsonl=getattr(args, "stats_jsonl", None),
        yarn_factor=getattr(args, "yarn_factor", None),
        yarn_original_max_position_embeddings=getattr(
            args, "yarn_original_max_position_embeddings", None
        ),
        enable_expert_parallel=getattr(args, "enable_expert_parallel", False),
        moe_backend=getattr(args, "moe_backend", "auto"),
    )
    registry = AudioRegistry(
        spool_dir=args.spool_dir,
        slice_cache_max=getattr(args, "slice_cache_max", 1024),
    )
    config = {
        "model": "LongCap-35B-A3B",
        "checkpoint": checkpoint,
        "runtime": serving_runtime_identity(),
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
        "allow_audio_paths": getattr(args, "allow_audio_paths", False),
        "max_audio_per_prompt": args.max_audio_per_prompt,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "enable_prefix_caching": args.apc,
        "dtype": args.dtype,
        "max_num_batched_tokens": getattr(args, "max_num_batched_tokens", None),
        "enable_expert_parallel": getattr(args, "enable_expert_parallel", False),
        "moe_backend": getattr(args, "moe_backend", "auto"),
        "mamba_ssm_cache_dtype": getattr(args, "mamba_ssm_cache_dtype", "auto"),
        "slice_cache_max": getattr(args, "slice_cache_max", 1024),
        "io_threads": getattr(args, "io_threads", 4),
        "torch_num_threads": getattr(args, "torch_num_threads", 0),
        "mm_processor_cache_gb": getattr(args, "mm_processor_cache_gb", None),
        "async_scheduling": getattr(args, "async_scheduling", False),
        "yarn_factor": (
            engine.yarn_rope_parameters.get("factor")
            if engine.yarn_rope_parameters else None
        ),
        "yarn_original_max_position_embeddings": (
            engine.yarn_rope_parameters.get("original_max_position_embeddings")
            if engine.yarn_rope_parameters else None
        ),
        "yarn_scaled_max_model_len": (
            int(
                engine.yarn_rope_parameters["original_max_position_embeddings"]
                * engine.yarn_rope_parameters["factor"]
            )
            if engine.yarn_rope_parameters
            else None
        ),
        "yarn_rope_parameters": engine.yarn_rope_parameters,
    }
    _ENGINE, _REGISTRY, _CONFIG = engine, registry, config
    return _build_app(engine, registry, config)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF model dir (exported)")
    ap.add_argument("--allow-audio-paths", action="store_true", default=False)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--spool-dir", default=None,
                    help="dir for uploaded-audio spool (required for /audio/upload)")
    # Defaults below are the measured-best single-GPU serving config for the
    # long-audio multi-hop job at 128-way concurrency (2026-08-27 study,
    # espnet2/speechlm/vllm/THROUGHPUT.md): 218x realtime steady on 1x H100
    # with the fp8-block 35B export. Every non-vLLM-default value was isolated
    # experimentally; see the .md for the attribution table.
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92,
                    help="0.85 wastes ~5.6 GB of KV pool on an 80 GB card "
                    "(measured part of a +27%% defaults->tuned gain)")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-num-seqs", type=int, default=128,
                    help="64 pins running requests at ~55 and queues the rest "
                    "at 128-way concurrency (166x -> 211x realtime measured)")
    ap.add_argument("--max-audio-per-prompt", type=int, default=64)
    ap.add_argument("--enable-prefix-caching", dest="apc",
                    action="store_true", default=True)
    ap.add_argument("--no-prefix-caching", dest="apc", action="store_false")
    ap.add_argument("--enforce-eager", action="store_true", default=False)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    # -- performance / numerics knobs (defaults preserve historical behavior) --
    ap.add_argument("--max-num-batched-tokens", type=int, default=16384,
                    help="chunked-prefill token budget per scheduler step; "
                    "also caps the audio-encoder budget. THE biggest serving "
                    "lever: vLLM's async default 2048 halves throughput on the "
                    "long-audio job (103x vs 203x realtime measured — 17k-token "
                    "history re-prefills get chopped into 8+ steps and the "
                    "encoder admits ~5 clips/step)")
    ap.add_argument("--mamba-ssm-cache-dtype", default="float32",
                    choices=["auto", "float32"],
                    help="GDN/SSM recurrent-state cache dtype. float32 matches "
                    "training (config mamba_ssm_dtype); 'auto' resolves to the "
                    "model dtype (bf16) for our custom arch because vLLM's "
                    "Qwen3.5 hook keys on the architecture name. bf16 halves "
                    "the per-seq state (61->31 MiB on the 35B) and buys +19%% "
                    "at 256-way concurrency, at the cost of near-tie decode "
                    "flips vs training")
    ap.add_argument("--slice-cache-max", type=int, default=16384,
                    help="entries in the registry's decoded-16k-slice LRU; must "
                    "cover concurrency x clips_per_dialogue or every hop "
                    "re-resamples its history (measured at the old 1024 "
                    "default: 46-61%% hit rate at 128-way = 24-34k redundant "
                    "42 ms sinc resamples per eval run). ~2 MB RAM per entry "
                    "worst-case; /health reports hit/miss counters")
    ap.add_argument("--io-threads", type=int, default=32,
                    help=">0: resolve audio slices off the event loop "
                    "(asyncio.to_thread) so featurization does not stall "
                    "concurrent requests; 0 restores the old inline behavior")
    ap.add_argument("--torch-num-threads", type=int, default=16,
                    help=">0: cap torch intra-op threads in the API process "
                    "(featurization); avoids multi-replica CPU thrash")
    ap.add_argument("--mm-processor-cache-gb", type=float, default=32.0,
                    help="vLLM multimodal processor cache size in GiB (vLLM's "
                    "own default is 4). Sized in cached mel items (~1.5 MB / "
                    "30 s clip); the live set at 256-way concurrency is "
                    "~4.8 GB and movie-length dialogues need far more, so 4 "
                    "GiB sits exactly at the thrash boundary. RAM is reserved "
                    "lazily (LRU cap), not preallocated")
    ap.add_argument("--async-scheduling", action="store_true", default=False,
                    help="enable vLLM async scheduling (overlaps the CPU "
                    "scheduler step with GPU execution)")
    ap.add_argument("--stats-jsonl", default=None,
                    help="append engine scheduler/cache stats as JSON lines "
                    "to this path every ~5s (see stats_logger.py)")
    ap.add_argument("--yarn-factor", type=float, default=None,
                    help="enable YaRN RoPE scaling by this factor; omitted "
                         "keeps the checkpoint's native RoPE unchanged")
    ap.add_argument("--yarn-original-max-position-embeddings", type=int,
                    default=None,
                    help="YaRN's trained-context length; default: derive from "
                         "text_config.max_position_embeddings")
    ap.add_argument("--enable-expert-parallel", action="store_true", default=False,
                    help="MoE: shard experts across the TP ranks (EP) instead of "
                         "slicing every expert; fuller-width expert GEMMs")
    ap.add_argument("--moe-backend", default="auto",
                    help="vLLM fused-MoE kernel backend (auto|triton|deepgemm|"
                         "flashinfer_cutlass|...); auto = vLLM's priority list")
    args = ap.parse_args(argv)

    import uvicorn

    app = build_app_from_args(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

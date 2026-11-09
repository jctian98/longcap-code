"""Async vLLM engine wrapper — the service's compute surface.

``engine.LocalEngine`` wraps the *synchronous* ``vllm.LLM`` (offline eval, verl
in-process). This module wraps the *async* ``vllm.v1.engine.AsyncLLM`` so the
FastAPI service can serve many concurrent multi-hop dialogues off one engine
(vLLM continuous-batches the concurrent requests). It is deliberately a thin
mirror of ``LocalEngine``: identical prompt construction (``_to_prompt``),
identical sampling params (``stop_token_ids=[EOS,EOT]``, ``detokenize=False``,
``extra_args={"speechlm_mode": mode}``), identical finish-reason resolution
(``_resolve``) — so a request served here is byte-equivalent to the same request
run offline. Those helpers are reused verbatim from ``LocalEngine`` (its static
methods) rather than re-implemented, keeping a single source of truth.

Two entry points:
- ``generate(req)`` — one assistant segment; returns a ``GenerateResult``.
- ``score(req, top_logprobs)`` — teacher-forced ``prompt_logprobs`` over the
  supplied sequence (max_tokens=1, greedy, full recompute → deterministic). The
  alignment-harness primitive and the verl logprob-recompute primitive.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass

from ..vllm_client import constants as C
from ..vllm_client.protocol import GenerateRequest, GenerateResult
from .engine import (
    LocalEngine,
    _LOGITS_PROCESSOR,
    validate_yarn_max_model_len,
    yarn_hf_overrides,
)


@dataclass
class ScoreResult:
    """Teacher-forced scoring result.

    ``prompt_logprobs[i]`` is ``{token_id: logprob}`` for prompt position ``i``
    (position 0 is ``None`` — no logprob for the first token), mirroring vLLM's
    ``RequestOutput.prompt_logprobs``.
    """

    prompt_logprobs: list[dict[int, float] | None]


class AsyncSpeechLMEngine:
    """Async, stateless wrapper over ``AsyncLLM`` for SpeechLM generation."""

    def __init__(
        self,
        model_dir: str,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 16384,
        max_num_seqs: int = 64,
        max_audio_per_prompt: int = 64,
        enable_prefix_caching: bool = True,
        enforce_eager: bool = False,
        dtype: str = "bfloat16",
        max_logprobs: int = 32,
        seed: int = 0,
        max_num_batched_tokens: int | None = None,
        mamba_ssm_cache_dtype: str = "auto",
        mm_processor_cache_gb: float | None = None,
        async_scheduling: bool | None = None,
        stats_jsonl: str | None = None,
        yarn_factor: float | None = None,
        yarn_original_max_position_embeddings: int | None = None,
        enable_expert_parallel: bool = False,
        moe_backend: str = "auto",
    ):
        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        self.model_dir = model_dir
        self.enable_prefix_caching = enable_prefix_caching
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.max_logprobs = max_logprobs
        self._ids = itertools.count()
        self._pid = os.getpid()

        # mamba_ssm_cache_dtype: vLLM's Qwen3.5 config hook promotes "auto" to
        # the checkpoint's mamba_ssm_dtype (float32), but the hook is keyed on
        # the architecture name and never fires for our custom
        # "SpeechLMForConditionalGeneration" — "auto" then silently resolves to
        # the model dtype (bf16), diverging from training. Pass "float32" to
        # restore the training-time SSM state dtype.
        extra: dict = {}
        if max_num_batched_tokens is not None:
            extra["max_num_batched_tokens"] = max_num_batched_tokens
        hf_overrides = yarn_hf_overrides(
            model_dir, yarn_factor, yarn_original_max_position_embeddings
        )
        validate_yarn_max_model_len(max_model_len, hf_overrides)
        self.yarn_rope_parameters = (
            dict(hf_overrides["text_config"]["rope_parameters"])
            if hf_overrides
            else None
        )
        if hf_overrides:
            extra["hf_overrides"] = hf_overrides
        # mm_processor_cache_gb: vLLM's default (4 GiB) holds ~2.7k cached mel
        # items; a high-concurrency long-dialogue load re-references every past
        # clip per hop and needs concurrency x clips_per_dialogue entries or the
        # cache thrashes back to O(N^2) CPU featurization.
        if mm_processor_cache_gb is not None:
            extra["mm_processor_cache_gb"] = mm_processor_cache_gb
        # async_scheduling: overlap the CPU scheduler step with GPU execution.
        if async_scheduling is not None:
            extra["async_scheduling"] = async_scheduling
        # vLLM fuses the tensor-parallel all-reduce with RMSNorm only when nvcc
        # is on PATH, so two-GPU output would depend on whether a CUDA toolkit
        # is installed. Keep it off so every installation runs the same kernels.
        compilation_config = {"pass_config": {"fuse_allreduce_rms": False}}
        args = AsyncEngineArgs(
            model=model_dir,
            tokenizer=model_dir,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            enable_prefix_caching=enable_prefix_caching,
            enable_chunked_prefill=True,
            limit_mm_per_prompt={"audio": max_audio_per_prompt},
            max_logprobs=max_logprobs,
            enforce_eager=enforce_eager,
            logits_processors=[_LOGITS_PROCESSOR],
            trust_remote_code=True,
            seed=seed,
            mamba_ssm_cache_dtype=mamba_ssm_cache_dtype,
            enable_expert_parallel=enable_expert_parallel,
            moe_backend=moe_backend,
            compilation_config=compilation_config,
            **extra,
        )
        # Machine-readable scheduler/cache stats (see stats_logger.py). The
        # default LoggingStatLogger prints via the vllm logger, which uvicorn's
        # logging setup can swallow; the JSONL logger is benchmark/monitoring
        # infrastructure and does not touch the request path.
        stat_loggers = None
        if stats_jsonl:
            from .stats_logger import make_jsonl_stat_logger_factory

            stat_loggers = [make_jsonl_stat_logger_factory(stats_jsonl)]
        self.engine = AsyncLLM.from_engine_args(args, stat_loggers=stat_loggers)

    def is_healthy(self) -> bool:
        """False once the vLLM engine core has failed."""
        return not getattr(self.engine, "errored", False)

    def _request_id(self, tag: str) -> str:
        return f"{tag}-{self._pid}-{next(self._ids)}"

    def _sampling_params(self, req: GenerateRequest):
        # Mirror of LocalEngine._sampling_params (kept in sync deliberately;
        # both paths must produce byte-identical decodes).
        from vllm import SamplingParams

        return SamplingParams(
            temperature=req.temperature,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            max_tokens=req.max_tokens,
            min_tokens=req.min_tokens,
            stop_token_ids=[C.EOS, C.EOT],
            detokenize=False,
            extra_args={"speechlm_mode": req.mode},
        )

    async def generate(self, req: GenerateRequest) -> GenerateResult:
        """Generate one assistant segment for a wire-form request."""
        prompt = LocalEngine._to_prompt(req)
        sp = self._sampling_params(req)
        final = None
        async for out in self.engine.generate(
            prompt, sp, self._request_id("gen")
        ):
            final = out
        if final is None:
            raise RuntimeError("engine produced no output")
        return LocalEngine._resolve(final)

    async def score(
        self, req: GenerateRequest, top_logprobs: int
    ) -> ScoreResult:
        """Teacher-forced prompt_logprobs over ``req.prompt_token_ids``.

        Full prompt recompute (vLLM forces this when prompt_logprobs is set), so
        the result is deterministic and independent of prefix caching — the
        alignment/verl scoring primitive.
        """
        from vllm import SamplingParams

        k = min(top_logprobs, self.max_logprobs)
        prompt = LocalEngine._to_prompt(req)
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            min_tokens=1,
            prompt_logprobs=k,
            detokenize=False,
            extra_args={"speechlm_mode": req.mode},
        )
        final = None
        async for out in self.engine.generate(
            prompt, sp, self._request_id("score")
        ):
            final = out
        if final is None:
            raise RuntimeError("engine produced no output")
        rows: list[dict[int, float] | None] = []
        for pos in final.prompt_logprobs or []:
            if pos is None:
                rows.append(None)
            else:
                rows.append({int(tid): float(lp.logprob) for tid, lp in pos.items()})
        return ScoreResult(prompt_logprobs=rows)

"""Resumable single-recording and batch audio inference over HTTP."""
from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from espnet2.speechlm.vllm.vllm_client.client import CaptionerClient, build_driver
from .identity import checkpoint_identity, require_matching_model
from .processing.common import read_json, unique_rows, write_json
from .processing.convert import convert_predictions
from .processing.prepare import SYSTEM_PROMPTS, verify_prepared
from .transport import JournalEngine, UploadAudioBook


def _source_identity() -> str:
    digest = hashlib.sha256()
    roots = [Path(__file__).parent,
             Path(__file__).parents[1] / "espnet2/speechlm/vllm/vllm_client"]
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def _joined_rounds(record: dict) -> list:
    """Retain original segments in records; join only for per-chunk conversion."""
    rounds = [[] for _ in range(record.get("num_rounds", 0))]
    for index, message in zip(record.get("message_round_indices", []), record["messages"]):
        rounds[index].append(message[2][0])
    return [["assistant", "text", ["\n".join(parts)]] for parts in rounds]


def _retry_summary(records, field):
    values = [(key, retry) for key, rec in records.items() for retry in rec.get(field, [])]
    return {
        "num_examples": len({key for key, _ in values}), "num_segments": len(values),
        "num_retries": sum(len(r["attempts"]) - 1 for _, r in values),
        "num_exhausted": sum(bool(r["exhausted"]) for _, r in values),
        "retry_output_tokens": sum(a["output_tokens"] for _, r in values for a in r["attempts"][1:]),
    }


def _run_inference(
    run: Path, model: Path, service: str, *, concurrency: int = 1,
    max_tokens: int = 8192, timeout: float = 1800, max_retried_chunks: int = 20,
    future_start_guard: bool = False, tokenizer=None, task: str = "caption",
) -> tuple[dict, int]:
    if concurrency < 1 or max_tokens < 1 or timeout <= 0 or max_retried_chunks < 0:
        raise ValueError("Invalid concurrency, token budget, timeout or retry limit")
    if task not in SYSTEM_PROMPTS or (task == "diarization" and future_start_guard):
        raise ValueError("Invalid task or caption-only conversion option")
    prepared = verify_prepared(run)
    if prepared["identity"]["system_prompt"] != SYSTEM_PROMPTS[task]:
        raise ValueError("Prepared task differs; use a fresh output directory")
    caption = task == "caption"
    identity = checkpoint_identity(model)
    client = CaptionerClient(service, timeout=timeout)
    try:
        server = client.models()
    finally:
        client.close()
    require_matching_model(identity, server)
    if server.get("max_audio_per_prompt", 0) < prepared["max_audio_per_prompt"]:
        raise ValueError("Server audio-item limit is below this recording's chunk count")
    # A replica may move to a different URL/path without changing its identity.
    server_identity = {k: v for k, v in server.items() if k not in {"model", "checkpoint"}}
    server_identity["checkpoint_fingerprint"] = identity["fingerprint"]
    spec = {
        "version": 1, "input_sha256": prepared["output_sha256"],
        "source_sha256": _source_identity(), "server": server_identity,
        "max_tokens": max_tokens, "max_retried_chunks": max_retried_chunks,
        "retry_penalties": [1.0, 1.05, 1.10, 1.15] if caption else [1.0],
        "task": task,
        "future_start_guard": future_start_guard,
    }
    manifest = run / "raw/run_identity.json"
    if manifest.exists() and read_json(manifest) != spec:
        raise ValueError("Model, inputs, code or decoding settings changed; use a fresh output directory")
    write_json(manifest, spec)
    examples = unique_rows(run / "inputs/chunks.jsonl")
    dialogues = unique_rows(run / "inputs/dialogue.jsonl", "example_id")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)

    def one(key):
        directory = run / "raw/records" / hashlib.sha256(key.encode()).hexdigest()
        record_path = directory / "result.json"
        if record_path.exists():
            record = read_json(record_path)
            if record["example_id"] != key:
                raise ValueError("Saved record identity mismatch")
            return key, record, True
        started = time.monotonic()
        client = CaptionerClient(service, timeout=timeout)
        book = UploadAudioBook(client)
        driver = build_driver(client, tokenizer, audiobook=book, max_tokens=max_tokens,
                              max_workers=1, loop_retry=caption, caption_format_retry=caption,
                              max_retried_chunks=max_retried_chunks)
        http_engine = driver.engine
        driver.engine = JournalEngine(http_engine, directory / "attempts", driver.prompt.decode_text)
        try:
            result = driver.run([(key, dialogues[key]["messages"])])[key]
            record = {**asdict(result), "error": None}
        except Exception as error:
            # The durable journal retains selected and rejected responses even
            # if no whole-recording result was produced.
            record = {"example_id": key, "error": {"type": type(error).__name__, "message": str(error)},
                      "messages": list(driver.partial.get(key, [])), "num_rounds": None,
                      "num_segments": len(driver.partial.get(key, [])),
                      "truncated_hops": None, "partial": True}
        finally:
            http_engine.close()
            client.close()
        record["wall_seconds"] = time.monotonic() - started
        write_json(record_path, record)
        return key, record, False

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        finished = list(pool.map(one, examples))
    records = {key: record for key, record, _ in finished}
    error_ids = [key for key, r in records.items() if r.get("error") is not None]
    early_ids = [key for key, r in records.items() if r.get("early_stop")]
    truncated_ids = [key for key, r in records.items() if r.get("truncated_hops", 0)]
    stats = {
        "task": task,
        "num_examples": len(examples), "num_written": len(records),
        "num_errors": len(error_ids), "error_ids": error_ids,
        "error_details": [{"key": key, "error": records[key]["error"]} for key in error_ids],
        "num_early_stopped": len(early_ids), "early_stopped_ids": early_ids,
        "early_stop_details": [{"key": key, **records[key]["early_stop"]} for key in early_ids],
        "num_truncated": len(truncated_ids), "truncated_ids": truncated_ids,
        "num_resumed": sum(resumed for _, _, resumed in finished),
        "loop_retry_summary": _retry_summary(records, "loop_retry_attempts"),
        "caption_retry_summary": _retry_summary(records, "caption_retry_attempts"),
        "checkpoint": identity, "max_tokens": max_tokens,
        "max_retried_chunks": max_retried_chunks,
    }
    write_json(run / "raw/client.json", stats)
    # This export preserves the converter's historical one-turn-per-chunk shape.
    # Original segment boundaries and every sampled token remain in raw/records.
    replies = {key: _joined_rounds(record) if not record.get("error") else []
               for key, record in records.items()}
    write_json(run / "raw/decode/inference_rank0/results.json", replies)
    if error_ids or early_ids:
        stats["prediction_export"] = "withheld: request errors or early-stopped prefixes"
        write_json(run / "raw/client.json", stats)
        return stats, 1 if error_ids else 2
    if caption:
        conversion = convert_predictions(run, future_start_guard=future_start_guard)
    else:
        from .processing.diarization import convert_diarization
        conversion = convert_diarization(run)
    stats["conversion"] = conversion
    incomplete = bool(truncated_ids) or any(k in conversion["statuses"] for k in ("partial", "parse_error"))
    exhausted = stats["caption_retry_summary"]["num_exhausted"]
    return stats, 2 if incomplete or exhausted else 0


def run_caption(
    run: Path, model: Path, service: str, *, concurrency: int = 1,
    max_tokens: int = 8192, timeout: float = 1800, max_retried_chunks: int = 20,
    future_start_guard: bool = False, tokenizer=None,
) -> tuple[dict, int]:
    return _run_inference(
        run, model, service, concurrency=concurrency, max_tokens=max_tokens,
        timeout=timeout, max_retried_chunks=max_retried_chunks,
        future_start_guard=future_start_guard, tokenizer=tokenizer,
    )


def run_diarization(
    run: Path, model: Path, service: str, *, concurrency: int = 1,
    max_tokens: int = 32768, timeout: float = 1800, tokenizer=None,
) -> tuple[dict, int]:
    """Use the native diarization prompt and decoding policy, without caption retries."""
    return _run_inference(
        run, model, service, concurrency=concurrency, max_tokens=max_tokens,
        timeout=timeout, max_retried_chunks=0, tokenizer=tokenizer, task="diarization",
    )

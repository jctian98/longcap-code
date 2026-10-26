"""Uploads and crash-safe request replay for the CPU captioning client."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from espnet2.speechlm.vllm.vllm_client.audio import AudioSegment
from espnet2.speechlm.vllm.vllm_client.protocol import GenerateResult
from .processing.common import read_json, write_json


class UploadAudioBook:
    """Register local WAV chunks by bytes; never expose client paths to server."""

    def __init__(self, client):
        self.client = client
        self.metadata = {}
        self.ref_by_uuid = {}

    def resolver(self, content):
        if isinstance(content, AudioSegment):
            path, start, end = content.path, content.start_sample, content.end_sample
        else:
            path, start, end = str(content), 0, None
        if path is None:
            raise ValueError("Upload audio needs a local file")
        if path not in self.metadata:
            file = Path(path)
            self.metadata[path] = self.client.register_bytes(file.read_bytes(), file.name)
        meta = self.metadata[path]
        end = meta["num_frames"] if end is None else end
        segment = AudioSegment(meta["audio_id"], path, start, end, meta["sample_rate"])
        self.ref_by_uuid[segment.uuid()] = {
            "audio_id": meta["audio_id"], "start_sample": start, "end_sample": end,
        }
        return segment, np.zeros(0, dtype=np.float32)


class JournalEngine:
    """Persist each response before the driver consumes it, then replay on resume.

    Replaying token IDs reconstructs accepted history and the exact consumed
    format/loop retry budget. Request errors are retained and terminal for this
    run; a fresh output directory is required for a deliberate fresh attempt.
    """

    def __init__(self, engine, directory: Path, decode_text=None):
        self.engine, self.directory, self.index = engine, directory, 0
        self.decode_text = decode_text
        directory.mkdir(parents=True, exist_ok=True)

    def generate(self, requests):
        if len(requests) != 1:
            raise ValueError("A request journal belongs to one recording")
        req = requests[0]
        request = {
            "prompt_token_ids": req.prompt_token_ids, "audio_uuids": req.audio_uuids,
            "max_tokens": req.max_tokens, "min_tokens": req.min_tokens,
            "temperature": req.temperature, "top_k": req.top_k,
            "mode": req.mode, "repetition_penalty": req.repetition_penalty,
        }
        digest = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        path = self.directory / f"{self.index:08d}.json"
        if path.exists():
            record = read_json(path)
            if record["request_sha256"] != digest:
                raise ValueError("Resumed request differs from saved token/audio history")
        else:
            record = {"request_sha256": digest, "request_index": self.index,
                      "prompt_tokens": len(req.prompt_token_ids),
                      "audio_items": len(req.audio_uuids or []),
                      "repetition_penalty": req.repetition_penalty}
            try:
                result = self.engine.generate(requests)[0]
                record["response"] = {
                    "token_ids": result.token_ids, "terminator": result.terminator,
                    "finish_reason": result.finish_reason,
                }
                if self.decode_text is not None:
                    record["text"] = self.decode_text(result.token_ids)
            except Exception as error:
                record["error"] = {"type": type(error).__name__, "message": str(error)}
                write_json(path, record)
                raise
            write_json(path, record)
        self.index += 1
        if "error" in record:
            raise RuntimeError(f"Saved request failure: {record['error']}; use a fresh output directory to retry")
        return [GenerateResult(**record["response"])]

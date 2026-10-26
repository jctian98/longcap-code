"""Portable model identity from the published SHA256SUMS manifest."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

from .processing.common import file_sha256, read_json

REQUIRED_METADATA = {
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "model.safetensors.index.json",
    "audio_processor/preprocessor_config.json",
}


def checkpoint_identity(model: Path, *, verify_weights: bool = False) -> dict:
    """Verify metadata and manifest coverage; optionally hash every payload.

    Fast checks verify every non-weight file and require every weight shard.
    Full checks should be run after downloading/copying the release. The digest
    names the manifest's contents, so it is independent of the mount path.
    """
    model = model.resolve()
    entries = {}
    for line in (model / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        if not match:
            raise ValueError("Invalid SHA256SUMS line")
        digest, name = match.groups()
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name:
            raise ValueError(f"Unsafe model manifest path: {name}")
        if name in entries:
            raise ValueError(f"Duplicate model manifest path: {name}")
        target = model / name
        if target.is_symlink() or not target.resolve().is_relative_to(model):
            raise ValueError(f"Model manifest escapes model directory: {name}")
        if not target.is_file():
            raise ValueError(f"Missing model file: {name}")
        entries[name] = digest.lower()
    if not REQUIRED_METADATA.issubset(entries):
        raise ValueError(f"SHA256SUMS missing metadata: {sorted(REQUIRED_METADATA - entries.keys())}")
    index = read_json(model / "model.safetensors.index.json")
    shards = set(index.get("weight_map", {}).values())
    if not shards or not shards.issubset(entries):
        raise ValueError("SHA256SUMS must cover every indexed weight shard")
    for name, digest in entries.items():
        if verify_weights or name not in shards:
            if file_sha256(model / name) != digest:
                raise ValueError(f"Model checksum mismatch: {name}")
    config = read_json(model / "config.json")
    if "SpeechLMForConditionalGeneration" not in config.get("architectures", []):
        raise ValueError("Expected SpeechLMForConditionalGeneration checkpoint")
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "scheme": "sha256-manifest-v1",
        "fingerprint": hashlib.sha256(canonical).hexdigest(),
        "files": len(entries), "weight_shards": len(shards),
        "metadata_verified": True,
        "all_payloads_verified": verify_weights,
    }


def require_matching_model(local: dict, service: dict) -> None:
    remote = service.get("checkpoint", {})
    if remote.get("scheme") != local["scheme"] or remote.get("fingerprint") != local["fingerprint"]:
        raise ValueError("Service checkpoint differs from the local model manifest")

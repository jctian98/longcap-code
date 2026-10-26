"""Stable serving-code and dependency identity for client resume validation."""
import hashlib
import importlib.metadata
import platform
from pathlib import Path


def serving_runtime_identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for directory in (root / "longcap", root / "espnet2/speechlm/vllm"):
        for path in sorted(directory.rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    versions = {}
    for package in ("torch", "torchaudio", "vllm", "transformers", "tokenizers",
                    "flashinfer-python", "triton", "numpy", "soundfile", "fastapi", "uvicorn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"protocol": "longcap-http-v1", "source_sha256": digest.hexdigest(),
            "python": platform.python_version(), "packages": versions}

"""Small file helpers shared by the captioner workflow."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator



def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique_object)


def unique_object(pairs: list[tuple]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON field {key!r}")
        value[key] = item
    return value


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line, object_pairs_hook=unique_object)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def unique_rows(path: Path, field: str = "key") -> dict[str, dict]:
    rows = {}
    for row in read_jsonl(path):
        key = row.get(field)
        if not isinstance(key, str) or not key:
            raise ValueError(f"{path}: expected a nonempty {field}")
        if key in rows:
            raise ValueError(f"{path}: duplicate {field} {key!r}")
        rows[key] = row
    if not rows:
        raise ValueError(f"{path}: no examples")
    return rows


def write_json(path: Path, value) -> None:
    write_text(
        path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def write_jsonl(path: Path, rows) -> None:
    write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
        ),
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def run_lock(run_directory: Path):
    """Prevent preparation, inference, and conversion from racing in one run."""
    run_directory.mkdir(parents=True, exist_ok=True)
    with (run_directory / ".workflow.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another workflow owns {run_directory}") from error
        yield

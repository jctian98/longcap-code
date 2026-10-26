"""Build audio-only, history-preserving captioning or diarization dialogues."""

from __future__ import annotations

import hashlib
import math
import multiprocessing
import random
from pathlib import Path

from .common import file_sha256, read_json, unique_rows, write_json, write_jsonl

SYSTEM_PROMPT = "Conduct audio caption"
SYSTEM_PROMPTS = {"caption": SYSTEM_PROMPT, "diarization": "Conduct audio diarization"}
CHUNK_LENGTHS = (8, 16, 24, 32)


def chunk_bounds(duration: float, key: str, seed: int) -> list[tuple[float, float]]:
    """Use the existing training tiler and its prior evaluation seed convention.

    This is the pure tiling routine from local/data_format_v2/segment_dialogue.py
    (also unchanged in v3), specialized to the established 8/32/8 settings.
    Keeping this small routine here avoids importing the parquet curation stack.
    """
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"{key}: duration must be finite and positive")
    generator = random.Random(f"{seed}:{key}")
    bounds = []
    start = 0.0
    while duration - start > CHUNK_LENGTHS[-1]:
        end = start + generator.choice(CHUNK_LENGTHS)
        bounds.append((start, end))
        start = end
    bounds.append((start, duration))
    return bounds


def _prepare_example(arguments: tuple[dict, str, int]) -> dict:
    # Read one chunk at a time, so an hour-long recording does not require an
    # hour-long waveform allocation in every preparation worker.
    import soundfile

    row, output_directory, seed = arguments
    key, duration = row["key"], row["duration"]
    destination = Path(output_directory) / hashlib.sha256(key.encode()).hexdigest()
    destination.mkdir(parents=True, exist_ok=True)
    chunks = []
    with soundfile.SoundFile(row["audio_path"]) as source:
        sample_rate = source.samplerate
        audio_duration = len(source) / sample_rate
        if duration > audio_duration + 1 / sample_rate:
            raise ValueError(f"{key}: manifest duration exceeds the source audio")
        duration = min(duration, audio_duration)
        for index, (start, end) in enumerate(chunk_bounds(duration, key, seed)):
            first_frame, last_frame = int(start * sample_rate), int(end * sample_rate)
            last_frame = min(last_frame, len(source))
            if last_frame - first_frame < int(0.1 * sample_rate):
                continue  # The prior converter also discards sub-100 ms tails.
            source.seek(first_frame)
            audio = source.read(
                last_frame - first_frame, dtype="float32", always_2d=True
            )
            chunk_path = destination / f"turn_{index:04d}.wav"
            soundfile.write(
                chunk_path, audio.mean(axis=1), sample_rate, subtype="PCM_16"
            )
            chunks.append(
                {
                    "start": start,
                    "end": end,
                    "audio_path": str(chunk_path),
                    "sha256": file_sha256(chunk_path),
                }
            )
    if not chunks:
        raise ValueError(f"{key}: no audio chunks survived preparation")
    return {
        "key": key,
        "duration": duration,
        "subset": row.get("subset"),
        "audio_path": row["audio_path"],
        "source_sha256": file_sha256(Path(row["audio_path"])),
        "audio_duration": audio_duration,
        "sample_rate": sample_rate,
        "chunks": chunks,
        "discarded_tail_seconds": duration - chunks[-1]["end"],
    }


def verify_prepared(run_directory: Path) -> dict:
    inputs = run_directory / "inputs"
    report = read_json(inputs / "preparation.json")
    for filename, digest in report["output_sha256"].items():
        if file_sha256(inputs / filename) != digest:
            raise ValueError(f"Prepared input changed: {inputs / filename}")
    examples = unique_rows(inputs / "chunks.jsonl")
    for row in examples.values():
        if row.get("source_sha256") and file_sha256(Path(row["audio_path"])) != row["source_sha256"]:
            raise ValueError(f"Source audio changed: {row['audio_path']}")
        for chunk in row["chunks"]:
            if file_sha256(Path(chunk["audio_path"])) != chunk["sha256"]:
                raise ValueError(f"Prepared audio changed: {chunk['audio_path']}")
    return report


def prepare_inputs(
    manifest_path: Path,
    run_directory: Path,
    *,
    seed: int = 42,
    workers: int = 16,
    task: str = "caption",
) -> dict:
    if task not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown inference task: {task}")
    system_prompt = SYSTEM_PROMPTS[task]
    if workers < 1:
        raise ValueError("workers must be positive")
    manifest_path, run_directory = manifest_path.resolve(), run_directory.resolve()
    inputs = run_directory / "inputs"
    identity = {
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "seed": seed,
        "system_prompt": system_prompt,
        "chunk_lengths": list(CHUNK_LENGTHS),
        "audio_view": "manifest-duration-prefix/native-rate/mono/pcm16",
        "preparer_sha256": file_sha256(Path(__file__)),
    }
    if (inputs / "preparation.json").exists():
        report = verify_prepared(run_directory)
        if report["identity"] != identity:
            raise ValueError("Preparation settings changed; use a new run directory")
        return report
    identity_path = inputs / "identity.json"
    if identity_path.exists():
        if read_json(identity_path) != identity:
            raise ValueError(
                "Partial preparation settings changed; use a new run directory"
            )
    else:
        for directory in (
            inputs,
            run_directory / "raw",
            run_directory / "predictions",
            run_directory / "results",
        ):
            if directory.exists() and any(directory.iterdir()):
                raise ValueError(
                    f"Unrecognized existing run files: {directory}; use a new run directory"
                )

    rows = []
    for key, source in unique_rows(manifest_path).items():
        # Select only inference fields. In particular, do not copy benchmark
        # events, captions, or curation metadata into any model request.
        audio_path = Path(source["audio_path"]).expanduser()
        if not audio_path.is_absolute():
            audio_path = manifest_path.parent / audio_path
        if not audio_path.is_file():
            raise ValueError(f"{key}: missing audio {audio_path}")
        duration = source["duration"]
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise ValueError(f"{key}: duration must be a number")
        chunk_bounds(duration, key, seed)  # Validate before launching workers.
        rows.append(
            {
                "key": key,
                "audio_path": str(audio_path.resolve()),
                "duration": float(duration),
                "subset": source.get("subset"),
            }
        )
    write_json(identity_path, identity)
    tasks = [(row, str(inputs / "audio"), seed) for row in rows]
    if workers == 1:
        examples = list(map(_prepare_example, tasks))
    else:
        with multiprocessing.get_context("spawn").Pool(
            min(workers, len(tasks))
        ) as pool:
            examples = list(pool.imap(_prepare_example, tasks, chunksize=1))

    dialogues = []
    for row in examples:
        messages = [["system", "text", system_prompt]]
        for chunk in row["chunks"]:
            messages.extend(
                [
                    ["user", "audio", chunk["audio_path"]],
                    ["assistant", "text", ""],
                ]
            )
        dialogues.append({"example_id": row["key"], "messages": messages})
    write_jsonl(inputs / "dialogue.jsonl", dialogues)
    write_jsonl(inputs / "chunks.jsonl", examples)
    report = {
        "identity": identity,
        "num_examples": len(examples),
        "num_chunks": sum(len(row["chunks"]) for row in examples),
        "max_audio_per_prompt": max(len(row["chunks"]) for row in examples),
        "duration_hours": sum(row["duration"] for row in examples) / 3600,
        "output_sha256": {
            name: file_sha256(inputs / name)
            for name in ("dialogue.jsonl", "chunks.jsonl")
        },
    }
    write_json(inputs / "preparation.json", report)
    return report

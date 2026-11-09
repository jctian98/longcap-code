"""Convert native diarization replies to speaker-attributed transcript segments.

The transcript-first and legacy speaker-first grammars follow the native
diarization evaluation parser. No caption deduplication or text repair is used.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from .common import read_json, unique_rows, write_json, write_jsonl
from .convert import _assistant_text
from .prepare import verify_prepared

POLICY_VERSION = "native_diarization_v1"
TIMESTAMP = r"-?\d+(?:(?:\.\d+)|(?::\d+){3})?"
SPEAKER = r"spk\s*\d+|speaker\s*\d+|说话人\s*[A-Za-z0-9]+|bgn|background_noise"
TRANSCRIPT_FIRST = re.compile(
    rf"<\s*(?P<speaker>{SPEAKER})\s*>\s*"
    rf"<\s*(?P<start>{TIMESTAMP})\s*,\s*(?P<end>{TIMESTAMP})\s*>",
    re.IGNORECASE,
)
SPEAKER_FIRST = re.compile(
    # Possessive whitespace keeps matching linear on long runs of spaces.
    rf"(?P<speaker>{SPEAKER})\s*,\s*+(?:[^,\[\]\n]*,\s*+)?"
    rf"\[\s*(?P<start>{TIMESTAMP})\s*,\s*(?P<end>{TIMESTAMP})\s*\]\s*:\s*",
    re.IGNORECASE,
)
PREFIX = re.compile(
    r"^\s*the\s+rich\s+caption\s+from\s+[-\d.]+\s+second\s+to\s+[-\d.]+"
    r"\s+second\s+is\s*:\s*", re.IGNORECASE,
)
WRAPPERS = re.compile(
    r"^\s*(?:(?:start|end)\s+time\s*:\s*[^\n]*|<Empty>)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _seconds(value: str) -> float:
    if ":" not in value:
        return float(value)
    hours, minutes, seconds, centiseconds = map(int, value.split(":"))
    if min(hours, minutes, seconds, centiseconds) < 0 or minutes >= 60 or seconds >= 60 or centiseconds >= 100:
        raise ValueError("invalid clock timestamp")
    return hours * 3600 + minutes * 60 + seconds + centiseconds / 100


def _speaker(value: str) -> str:
    value = value.strip()
    if value.lower() in {"bgn", "background_noise"}:
        return "background_noise"
    return re.sub(r"^(?:speaker|spk|说话人)\s*", "", value, flags=re.IGNORECASE)


def parse_diarization_turn(text: str, chunk_start: float) -> tuple[list[dict], list[dict]]:
    """Parse one audio turn; native timestamps are relative to its chunk start.

    Preserve segment order and overlapping speakers. The native scorer clamps
    recording starts to zero and excludes the background-noise pseudo-speaker.
    Unparsed text and invalid timestamps remain visible in diagnostics/raw data.
    """
    body = WRAPPERS.sub("", PREFIX.sub("", text, count=1))
    transcript_matches = list(TRANSCRIPT_FIRST.finditer(body))
    speaker_matches = list(SPEAKER_FIRST.finditer(body))
    transcript_first = len(transcript_matches) > len(speaker_matches)
    matches = transcript_matches if transcript_first else speaker_matches
    segments, issues = [], []
    if not matches:
        if body.strip():
            issues.append({"reason": "unparsed_text", "text": body.strip()})
        return segments, issues
    previous = 0
    if not transcript_first and body[:matches[0].start()].strip():
        issues.append({"reason": "unparsed_text", "text": body[:matches[0].start()].strip()})
    for index, match in enumerate(matches):
        if transcript_first:
            utterance = body[previous:match.start()].strip()
        else:
            end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
            utterance = body[match.end():end].strip()
        previous = match.end()
        try:
            start = max(0.0, _seconds(match["start"]) + chunk_start)
            end = _seconds(match["end"]) + chunk_start
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                raise ValueError("nonfinite or nonpositive segment interval")
        except (ValueError, OverflowError) as error:
            issues.append({"segment": index, "reason": "invalid_timestamp", "detail": str(error)})
            continue
        speaker = _speaker(match["speaker"])
        if speaker != "background_noise":
            segments.append({"start": start, "end": end, "speaker": speaker, "text": utterance})
    if transcript_first and body[previous:].strip():
        issues.append({"reason": "unparsed_text", "text": body[previous:].strip()})
    return segments, issues


def convert_diarization(run_directory: Path) -> dict:
    """Export complete native-task runs without invoking the caption converter."""
    prepared = verify_prepared(run_directory)
    if prepared["identity"]["system_prompt"] != "Conduct audio diarization":
        raise ValueError("Diarization conversion requires diarization inputs")
    previous = run_directory / "predictions/conversion_summary.json"
    if previous.exists() and read_json(previous).get("postprocessing_policy") != POLICY_VERSION:
        raise ValueError("Postprocessing policy changed; use a fresh run copy without predictions")
    examples = unique_rows(run_directory / "inputs/chunks.jsonl")
    statistics = read_json(run_directory / "raw/client.json")
    if statistics.get("num_errors") != 0 or statistics.get("error_ids"):
        raise ValueError("Inference contains request errors; conversion is withheld")
    if statistics.get("num_early_stopped") or statistics.get("early_stopped_ids"):
        raise ValueError("Inference contains early-stopped prefixes; conversion is withheld")
    if statistics.get("num_written") != len(examples) or statistics.get("num_examples") != len(examples):
        raise ValueError("Client statistics do not cover every recording key")
    results = {}
    for shard in sorted((run_directory / "raw/decode").glob("inference_rank*/results.json")):
        values = read_json(shard)
        if not isinstance(values, dict) or results.keys() & values.keys():
            raise ValueError("Invalid or duplicate raw result keys")
        results.update(values)
    if results.keys() != examples.keys():
        raise ValueError("Raw result coverage differs from prepared recordings")
    truncated = set(statistics.get("truncated_ids", []))
    if truncated - examples.keys() or statistics.get("num_truncated", 0) != len(truncated):
        raise ValueError("Client truncation statistics have inconsistent key coverage")
    predictions, diagnostics = [], []
    for key, example in examples.items():
        messages = results[key]
        if not isinstance(messages, list) or len(messages) != len(example["chunks"]):
            raise ValueError(f"{key}: assistant turn count differs from saved audio chunks")
        segments, issues = [], []
        for turn, (message, chunk) in enumerate(zip(messages, example["chunks"])):
            parsed, turn_issues = parse_diarization_turn(_assistant_text(message, key), chunk["start"])
            segments.extend(parsed)
            issues.extend({"turn": turn, **issue} for issue in turn_issues)
        predictions.append({"key": key, "duration": example["duration"], "segments": segments})
        status = "partial" if issues or key in truncated else "ok"
        if not segments and key not in truncated:
            status = "parse_error" if issues else "empty"
        diagnostics.append({"key": key, "status": status, "truncated": key in truncated,
                            "num_segments": len(segments), "issues": issues})
    destination = run_directory / "predictions"
    write_jsonl(destination / "diarization.jsonl", predictions)
    write_jsonl(destination / "conversion.jsonl", diagnostics)
    report = {
        "task": "diarization", "num_examples": len(predictions),
        "num_segments": sum(len(row["segments"]) for row in predictions),
        "num_truncated": len(truncated),
        "postprocessing_policy": POLICY_VERSION,
        "statuses": dict(Counter(row["status"] for row in diagnostics)),
    }
    write_json(destination / "conversion_summary.json", report)
    return report

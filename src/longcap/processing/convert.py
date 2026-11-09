"""Convert saved captioner turns into the model-independent scorer grammar."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from .common import (
    file_sha256,
    read_json,
    unique_object,
    unique_rows,
    write_json,
    write_jsonl,
)
from espnet2.speechlm.vllm.vllm_client.caption_format import split_lineage
from .prepare import verify_prepared
from .deduplicate import POLICY_VERSION, deduplicate_events, inspect_chunk
from .future_start_guard import POLICY_VERSION as POSTPROCESSING_POLICY, filter_future_starts

NUMBER = r"-?\d+(?:\.\d+)?"
EVENT_LINE = re.compile(rf"^\s*(\{{.*\}})\s*<\s*({NUMBER})\s*,\s*({NUMBER})\s*>\s*$")
CLOCK_LINE = re.compile(
    r"^\s*(\{.*\})\s*<\s*(\d+:\d{2}:\d{2}:\d{2})\s*,\s*"
    r"(\d+:\d{2}:\d{2}:\d{2})\s*>\s*$"
)
ATTRIBUTE_PREFIX = re.compile(r"^[\d.]+\s*")
METADATA_FIELDS = frozenset({"ancestor", "ancestors", "event_index"})


def _clock_seconds(value: str) -> float:
    hours, minutes, seconds, centiseconds = map(int, value.split(":"))
    if minutes >= 60 or seconds >= 60 or centiseconds >= 100:
        raise ValueError("invalid clock timestamp")
    return hours * 3600 + minutes * 60 + seconds + centiseconds / 100


def parse_turn(text: str, chunk_start: float) -> tuple[list[dict], list[dict]]:
    """Preserve explicit events and record every unusable line without repair."""
    events, issues = [], []
    for line_number, source in enumerate(text.splitlines(), 1):
        line = source.strip()
        if (
            not line
            or line == "<Empty>"
            or line.startswith(("start time:", "end time:"))
        ):
            continue
        # Ancestry metadata is ignored, matching the retry-time format check.
        line = split_lineage(line)
        if line is None:
            continue
        relative = EVENT_LINE.match(line)
        match = relative or CLOCK_LINE.match(line)
        if match is None:
            issues.append(
                {"line": line_number, "reason": "unparsed_line", "text": source}
            )
            continue
        try:
            attributes = json.loads(match[1], object_pairs_hook=unique_object)
            if not isinstance(attributes, dict):
                raise ValueError("attributes must be an object")
            attributes = {name: value for name, value in attributes.items()
                          if ATTRIBUTE_PREFIX.sub("", name) not in METADATA_FIELDS}
            # Reject non-JSON NaN/Infinity values without rewriting the event.
            json.dumps(attributes, allow_nan=False)
            if relative:
                start, end = (
                    float(match[2]) + chunk_start,
                    float(match[3]) + chunk_start,
                )
            else:
                # Retain the prior converter's v2 absolute-clock compatibility.
                start, end = _clock_seconds(match[2]), _clock_seconds(match[3])
            if not math.isfinite(start) or not math.isfinite(end):
                raise ValueError("nonfinite timestamp")
        except (ValueError, TypeError) as error:
            issues.append(
                {
                    "line": line_number,
                    "reason": "invalid_event",
                    "detail": str(error),
                    "text": source,
                }
            )
            continue
        normalized_names = [ATTRIBUTE_PREFIX.sub("", name) for name in attributes]
        if len(normalized_names) != len(set(normalized_names)):
            # Keep all original fields when removing prefixes would erase a
            # distinct attribute. The shared scorer also recognizes prefixes.
            issues.append({"line": line_number, "reason": "attribute_name_collision",
                           "warning": True})
            normalized = attributes
        else:
            normalized = dict(zip(normalized_names, attributes.values()))
        # Timing normalization belongs to the shared scorer. Keep negative,
        # reversed and out-of-duration predictions here as observed.
        events.append({"start": start, "end": end, "attributes": normalized})
    return events, issues


def _assistant_text(message, key: str) -> str:
    if (
        not isinstance(message, list)
        or len(message) != 3
        or message[:2] != ["assistant", "text"]
    ):
        raise ValueError(f"{key}: malformed assistant message")
    content = message[2]
    if isinstance(content, list) and len(content) == 1:
        content = content[0]
    if not isinstance(content, str):
        raise ValueError(f"{key}: expected exactly one assistant text")
    return content


def convert_predictions(run_directory: Path, *, future_start_guard: bool = False) -> dict:
    policy = POSTPROCESSING_POLICY if future_start_guard else POLICY_VERSION
    previous = run_directory / "predictions/conversion_summary.json"
    if previous.exists():
        saved = read_json(previous)
        if saved.get("postprocessing_policy", saved.get("deduplication_policy")) != policy:
            raise ValueError("Postprocessing policy changed; use a fresh run copy without predictions")
    prepared = verify_prepared(run_directory)
    if prepared["identity"]["system_prompt"] != "Conduct audio caption":
        raise ValueError("Caption conversion requires caption inputs; use longcap convert for diarization")
    examples = unique_rows(run_directory / "inputs/chunks.jsonl")
    raw_directory = run_directory / "raw"
    statistics = read_json(raw_directory / "client.json")
    # Recordings stopped at the retried-chunk limit export their decoded prefix
    # and are marked partial.
    early_stopped = set(statistics.get("early_stopped_ids", []))
    if statistics.get("num_errors") != 0 or statistics.get("error_ids"):
        raise ValueError(
            "Inference contains request errors; resume query before conversion"
        )
    if statistics.get("num_written") != len(examples) or statistics.get(
        "num_examples"
    ) != len(examples):
        raise ValueError("Client statistics do not cover every benchmark key")
    shards = sorted((raw_directory / "decode").glob("inference_rank*/results.json"))
    results = {}
    for shard in shards:
        values = read_json(shard)
        if not isinstance(values, dict):
            raise ValueError(f"{shard}: expected a result object")
        duplicate_keys = results.keys() & values.keys()
        if duplicate_keys:
            raise ValueError(
                f"Duplicate raw keys across shards: {sorted(duplicate_keys)[:5]}"
            )
        results.update(values)
    missing, unexpected = (
        examples.keys() - results.keys(),
        results.keys() - examples.keys(),
    )
    if missing or unexpected:
        raise ValueError(
            f"Raw coverage mismatch: missing={sorted(missing)[:5]}, unexpected={sorted(unexpected)[:5]}"
        )

    predictions, diagnostics, deduplication, chunk_diagnostics = [], [], [], []
    future_start_removals = []
    parsed_event_count = 0
    truncated = set(statistics.get("truncated_ids", []))
    if truncated - examples.keys() or statistics.get("num_truncated", 0) != len(
        truncated
    ):
        raise ValueError("Client truncation statistics have inconsistent key coverage")
    for key, example in examples.items():
        messages = results[key]
        expected = len(example["chunks"])
        if not isinstance(messages, list) or not (
            len(messages) == expected or (key in early_stopped and len(messages) < expected)
        ):
            raise ValueError(
                f"{key}: assistant turn count differs from saved audio chunks"
            )
        events, issues, suspicious_turns = [], [], []
        if len(messages) < expected:
            issues.append({"reason": "early_stop", "decoded_chunks": len(messages),
                           "total_chunks": expected})
        for turn_index, (message, chunk) in enumerate(zip(messages, example["chunks"])):
            parsed, turn_issues = parse_turn(
                _assistant_text(message, key), chunk["start"]
            )
            chunk_report = {"key": key, "turn": turn_index,
                            **inspect_chunk(parsed, chunk["start"], chunk["end"])}
            chunk_diagnostics.append(chunk_report)
            if chunk_report["warning_reasons"]:
                suspicious_turns.append(turn_index)
            parsed_event_count += len(parsed)
            causal_events, removed = (filter_future_starts(parsed, chunk["end"])
                                      if future_start_guard else (parsed, []))
            future_start_removals.extend({"key": key, "turn": turn_index,
                "chunk_start": chunk["start"], **item} for item in removed)
            events.extend(causal_events)
            issues.extend({"turn": turn_index, **issue} for issue in turn_issues)
        prediction = {"key": key, "duration": example["duration"], "events": events}
        if example.get("subset") is not None:
            prediction["subset"] = example["subset"]
        retained, duplicate_groups = deduplicate_events(events)
        predictions.append({**prediction, "events": retained})
        deduplication.append({"key": key, "original_events": len(events), "retained_events": len(retained), "groups": duplicate_groups})
        problems = [issue for issue in issues if not issue.get("warning")]
        status = "partial" if problems or key in truncated else "ok"
        if not events and key not in truncated:
            status = "parse_error" if problems else "empty"
        diagnostics.append(
            {
                "key": key,
                "status": status,
                "truncated": key in truncated,
                "num_events": len(events),
                "suspicious_turns": suspicious_turns,
                "issues": issues,
            }
        )
    destination = run_directory / "predictions"
    write_jsonl(destination / "predictions.jsonl", predictions)
    write_json(destination / "deduplication.json", {"policy": POLICY_VERSION, "records": deduplication})
    write_jsonl(destination / "conversion.jsonl", diagnostics)
    write_jsonl(destination / "chunk_diagnostics.jsonl", chunk_diagnostics)
    write_jsonl(destination / "future_start_removals.jsonl", future_start_removals)
    report = {
        "num_examples": len(predictions),
        "num_events": sum(len(row["events"]) for row in predictions),
        "num_events_before_deduplication": sum(row["original_events"] for row in deduplication),
        "deduplication_policy": POLICY_VERSION,
        "postprocessing_policy": POSTPROCESSING_POLICY if future_start_guard else POLICY_VERSION,
        "num_events_before_time_guard": parsed_event_count,
        "num_future_start_removed": len(future_start_removals),
        "future_start_guard_sha256": file_sha256(Path(__file__).with_name("future_start_guard.py")),
        "statuses": dict(Counter(row["status"] for row in diagnostics)),
        "num_truncated": len(truncated),
        "num_suspicious_chunks": sum(bool(row["warning_reasons"]) for row in chunk_diagnostics),
        "chunk_warning_counts": dict(Counter(
            reason for row in chunk_diagnostics for reason in row["warning_reasons"]
        )),
        "loop_retry_summary": statistics.get("loop_retry_summary"),
        **({"caption_retry_summary": statistics["caption_retry_summary"]}
           if "caption_retry_summary" in statistics else {}),
        "raw_sha256": {
            str(path.relative_to(raw_directory)): file_sha256(path) for path in shards
        },
        "client_sha256": file_sha256(raw_directory / "client.json"),
        "converter_sha256": file_sha256(Path(__file__)),
        "prediction_sha256": file_sha256(destination / "predictions.jsonl"),
    }
    write_json(destination / "conversion_summary.json", report)
    return report

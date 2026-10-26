"""Remove repeated, overlapping copies of the same native caption event."""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from .common import file_sha256, read_jsonl, write_json, write_jsonl

LEGACY_POLICY_VERSION = "identical_attributes_overlap_v1"
POLICY_VERSION = "speech_caption_insensitive_overlap_v2"
HIGH_EVENT_COUNT = 20
REPEATED_SPEECH_COUNT = 6


def _overlap_signature(attributes: dict, policy: str) -> str:
    # A caption-only description is not enough evidence to merge events. Keep
    # speech identity, transcript, and every structured attribute unchanged.
    transcript = attributes.get("文字转写") or attributes.get("transcript")
    if (policy == POLICY_VERSION and attributes.get("type") == "speech"
            and isinstance(transcript, str) and transcript.strip()):
        attributes = {key: value for key, value in attributes.items() if key != "caption"}
    return json.dumps(attributes, sort_keys=True, ensure_ascii=False, allow_nan=False)


def deduplicate_events(
    events: list[dict], *, policy: str = POLICY_VERSION,
) -> tuple[list[dict], list[dict]]:
    """Keep the first source event in each connected overlapping duplicate group.

    Speech with a nonempty transcript may differ only in its free-text caption.
    All other attributes, including speaker and transcript, must match exactly.
    Other event types and speech without a transcript require full equality.
    Disjoint or merely touching occurrences stay distinct. Timestamps and the
    retained attributes are never averaged, extended, repaired, or clipped.
    """
    if policy not in (POLICY_VERSION, LEGACY_POLICY_VERSION):
        raise ValueError(f"Unknown deduplication policy: {policy}")
    by_attributes = defaultdict(list)
    full_signatures = []
    exact, removed, groups = {}, set(), []
    for index, event in enumerate(events):
        signature = json.dumps(event["attributes"], sort_keys=True, ensure_ascii=False, allow_nan=False)
        full_signatures.append(signature)
        start, end = event["start"], event["end"]
        identity = (signature, start, end)
        if identity in exact:
            first = exact[identity]
            removed.add(index)
            groups.append({"kept_index": first, "removed_indices": [index], "reason": "exact_duplicate"})
            continue
        exact[identity] = index
        if (isinstance(start, (int, float)) and isinstance(end, (int, float))
                and not isinstance(start, bool) and not isinstance(end, bool)
                and math.isfinite(start) and math.isfinite(end) and start < end):
            by_attributes[_overlap_signature(event["attributes"], policy)].append((start, end, index))

    for values in by_attributes.values():
        component, right = [], None

        def finish():
            if len(component) > 1:
                first = min(component)
                duplicates = sorted(i for i in component if i != first)
                removed.update(duplicates)
                differs_only_in_caption = len({full_signatures[i] for i in component}) > 1
                reason = ("speech_attributes_except_caption_overlapping_intervals"
                          if differs_only_in_caption
                          else "identical_attributes_overlapping_intervals")
                groups.append({"kept_index": first, "removed_indices": duplicates,
                               "reason": reason})

        for start, end, index in sorted(values):
            if component and start >= right:
                finish()
                component = []
            component.append(index)
            right = max(right, end) if len(component) > 1 else end
        finish()

    # An exact duplicate's first copy may itself belong to an overlap group.
    parents = {index: group["kept_index"] for group in groups for index in group["removed_indices"]}
    for group in groups:
        while group["kept_index"] in parents:
            group["kept_index"] = parents[group["kept_index"]]
    return [event for index, event in enumerate(events) if index not in removed], groups


def inspect_chunk(events: list[dict], start: float, end: float) -> dict:
    """Expose unusually large replies without discarding dense valid events.

    Twenty events is a review threshold, not a cap or evidence of invalidity.
    Duration, repeated overlapping copies and postprocessing counts let callers
    distinguish real event density from a looping response. Inspect raw parsed
    events, before deduplication can hide the warning.
    """
    retained, groups = deduplicate_events(events)
    copies = defaultdict(int)
    for group in groups:
        copies[group["kept_index"]] += len(group["removed_indices"])
    largest = 1 + max(copies.values(), default=0) if events else 0
    reasons = []
    if len(events) >= HIGH_EVENT_COUNT:
        reasons.append("high_event_count")
    if largest >= 4:
        reasons.append("repeated_overlapping_events")
    transcripts = Counter()
    for event in events:
        attributes = event["attributes"]
        transcript = attributes.get("文字转写") or attributes.get("transcript")
        if attributes.get("type") == "speech" and isinstance(transcript, str) and transcript.strip():
            transcripts[transcript] += 1
    most_repeated, repetitions = transcripts.most_common(1)[0] if transcripts else (None, 0)
    if repetitions >= REPEATED_SPEECH_COUNT:
        reasons.append("repeated_speech_text")
    duration = end - start
    return {
        "start_seconds": start, "end_seconds": end,
        "num_events": len(events), "num_retained_events": len(retained),
        "events_per_second": len(events) / duration if duration > 0 else None,
        "largest_duplicate_group": largest,
        "max_transcript_repetitions": repetitions,
        "most_repeated_transcript": most_repeated if repetitions >= REPEATED_SPEECH_COUNT else None,
        "high_event_count_threshold": HIGH_EVENT_COUNT,
        "repeated_speech_count_threshold": REPEATED_SPEECH_COUNT,
        "warning_reasons": reasons,
    }


def deduplicate_file(source: Path, destination: Path) -> dict:
    """Create a separate scoring export while preserving saved inference bytes."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists() or destination.with_suffix(".deduplication.json").exists():
        raise ValueError("Deduplicated predictions require a fresh separate destination")
    rows, audit = [], []
    original_count = 0
    for row in read_jsonl(source):
        original_count += len(row["events"])
        events, groups = deduplicate_events(row["events"])
        rows.append({**row, "events": events})
        audit.append({"key": row["key"], "original_events": len(row["events"]),
                      "retained_events": len(events), "groups": groups})
    write_jsonl(destination, rows)
    receipt = {"policy": POLICY_VERSION, "source": str(source),
               "source_sha256": file_sha256(source), "prediction_sha256": file_sha256(destination),
               "records": len(rows), "original_events": original_count,
               "retained_events": sum(len(row["events"]) for row in rows), "audit": audit}
    write_json(destination.with_suffix(".deduplication.json"), receipt)
    return receipt

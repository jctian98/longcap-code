"""Validate native caption event lines without judging their audio content."""

import json
import math
import re

POLICY_VERSION = "caption_format_with_ancestor_metadata_v2"
_NUMBER = r"-?\d+(?:\.\d+)?"
_EVENT = re.compile(rf"^(\{{.*\}})\s*<\s*({_NUMBER})\s*,\s*({_NUMBER})\s*>$")
_CLOCK = re.compile(r"^(\{.*\})\s*<\s*(\d+:\d{2}:\d{2}:\d{2})\s*,\s*(\d+:\d{2}:\d{2}:\d{2})\s*>$")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


_METADATA_FIELDS = frozenset({"ancestor", "ancestors", "event_index"})
_ATTRIBUTE_PREFIX = re.compile(r"^[\d.]+\s*")
_STAMP = rf"(?:{_NUMBER}|\d+:\d{{2}}:\d{{2}}:\d{{2}})"
_TIMING = re.compile(rf"^\s*(<\s*{_STAMP}\s*,\s*{_STAMP}\s*>)\s*(.*)$")


_LINEAGE_START = re.compile(r'^\{\s*"(?:[\d.]+\s*)?(?:ancestor|ancestors|event_index)"')


def is_lineage(suffix: str) -> bool:
    """True for a trailing object that holds only event-ancestry metadata.

    A truncated suffix counts when it opens with a metadata field. Anything
    else after the timestamps, such as a second event, is not lineage.
    """
    suffix = suffix.strip()
    if not suffix.startswith("{"):
        return False
    try:
        value = json.loads(suffix)
    except (ValueError, RecursionError):
        return bool(_LINEAGE_START.match(suffix))
    return (isinstance(value, dict) and bool(value)
            and all(_ATTRIBUTE_PREFIX.sub("", key) in _METADATA_FIELDS for key in value))


def split_lineage(line: str) -> str | None:
    """Return an event line without a lineage suffix; None for a metadata-only line.

    Lines without metadata are returned unchanged.
    """
    stripped = line.lstrip()
    try:
        value, end = json.JSONDecoder().raw_decode(stripped)
    except (ValueError, RecursionError):
        return line
    if not isinstance(value, dict):
        return line
    rest = stripped[end:]
    if not rest.strip():
        if value and all(_ATTRIBUTE_PREFIX.sub("", key) in _METADATA_FIELDS for key in value):
            return None
        return line
    match = _TIMING.match(rest)
    if match and match[2] and is_lineage(match[2]):
        return stripped[:end] + " " + match[1]
    return line


def _caption_object(pairs):
    return _unique_object([(key, value) for key, value in pairs
                           if _ATTRIBUTE_PREFIX.sub("", key) not in _METADATA_FIELDS])


def strip_ancestor_metadata(text: str) -> str:
    """Exclude lineage from validation, leaving the actual model reply untouched.

    Locate timing after the first decoded attribute object, never inside a quoted
    caption. A trailing metadata-only object, possibly truncated, is ignored;
    any other trailing text, such as a second event on the same line, stays
    visible to validators.
    """
    decoder = json.JSONDecoder(object_pairs_hook=_caption_object)
    output = []
    for source in text.splitlines():
        line = source.lstrip()
        candidate = source
        try:
            attributes, end = decoder.raw_decode(line)
            match = _TIMING.match(line[end:])
            if isinstance(attributes, dict) and match and (not match[2] or is_lineage(match[2])):
                candidate = json.dumps(attributes, ensure_ascii=False) + " " + match[1]
            elif attributes == {} and not line[end:].strip():
                original = json.loads(line)
                if original and all(_ATTRIBUTE_PREFIX.sub("", k) in _METADATA_FIELDS for k in original):
                    candidate = ""
        except (ValueError, TypeError, OverflowError, RecursionError):
            pass
        output.append(candidate)
    return "\n".join(output)


def caption_format_error(text: str) -> dict | None:
    """Return the first contract failure; explicit <Empty> is valid.

    One caption object plus external timestamps is required per event line.
    Optional ancestor/event-index metadata is ignored. Signed
    relative times and the converter's legacy absolute clocks are accepted.
    No missing punctuation, fields, events, or timestamps are inferred. Empty
    attributes, overlaps, reversed times and semantic errors are not retried.
    """
    text = strip_ancestor_metadata(text)
    meaningful = []
    for number, source in enumerate(text.splitlines(), 1):
        line = source.strip()
        if not line or line.startswith(("start time:", "end time:")):
            continue
        meaningful.append((number, line))
    if not meaningful:
        return {"reason": "blank_output"}
    if all(line == "<Empty>" for _, line in meaningful):
        return None
    if not any(line.startswith("{") for _, line in meaningful):
        return {"reason": "no_json", "line": meaningful[0][0]}
    for number, line in meaningful:
        if line == "<Empty>":
            continue
        relative = _EVENT.fullmatch(line)
        match = relative or _CLOCK.fullmatch(line)
        if match is None:
            return {"reason": "invalid_event_line", "line": number}
        try:
            attributes = json.loads(match[1], object_pairs_hook=_unique_object)
            if not isinstance(attributes, dict):
                raise ValueError("event attributes must be an object")
            json.dumps(attributes, allow_nan=False)
            if relative:
                if not all(math.isfinite(float(v)) for v in match.group(2, 3)):
                    raise ValueError("nonfinite timestamp")
            else:
                for stamp in match.group(2, 3):
                    hours, minutes, seconds, centiseconds = map(int, stamp.split(":"))
                    if minutes >= 60 or seconds >= 60 or centiseconds >= 100:
                        raise ValueError("invalid clock timestamp")
                    if not math.isfinite(hours * 3600 + minutes * 60 + seconds + centiseconds / 100):
                        raise ValueError("nonfinite timestamp")
        except (ValueError, TypeError, OverflowError, RecursionError) as error:
            return {"reason": "invalid_json_or_timestamp", "line": number,
                    "detail": str(error)}
    return None

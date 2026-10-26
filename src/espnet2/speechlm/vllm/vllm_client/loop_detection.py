"""Conservative textual loop evidence for retrying a caption response.

This does not judge acoustic correctness or treat a long response as a loop.
"""

import json
import re
from collections import Counter, defaultdict

from .caption_format import strip_ancestor_metadata

POLICY_VERSION = "mechanical_loop_v1"

_VALUE = re.compile(r':\s*"((?:[^"\\]|\\.)*)(?:"|$)')
_TANDEM = re.compile(r"(.{1,160}?)\1{15,}", re.DOTALL)
_EVENT = re.compile(
    r"^\s*(\{.*\})\s*<\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*>\s*$"
)
_PREFIX = re.compile(r"^[\d.]+\s*")


def detect_loop(text: str) -> str | None:
    """Return conservative mechanical-loop evidence, including in unfinished JSON.

    Repetition of schema keys and ordinary repeated dialogue do not suffice.
    Event-time signs and ordering alone never trigger a retry.
    """
    text = strip_ancestor_metadata(text)
    for value in _VALUE.finditer(text):
        if len(value[1]) < 512:
            continue
        for repeated in _TANDEM.finditer(value[1]):
            if len(repeated[0]) >= 512 and any(c.isalnum() for c in repeated[1]):
                return "repeated_string_value"

    timed = Counter()
    by_attributes = defaultdict(list)
    for line in text.splitlines():
        match = _EVENT.match(line)
        if not match:
            continue
        try:
            attributes = json.loads(match[1])
        except json.JSONDecodeError:
            continue
        attributes = {_PREFIX.sub("", k): v for k, v in attributes.items()}
        # Captions often paraphrase the same attributes. Keep caption-only events.
        signature = {k: v for k, v in attributes.items() if k != "caption"}
        signature = json.dumps(signature or attributes, sort_keys=True, ensure_ascii=False)
        start, end = float(match[2]), float(match[3])
        timed[signature, start, end] += 1
        if timed[signature, start, end] >= 4:
            return "repeated_timed_event"
        by_attributes[signature].append((start, end))

    # Twelve identical events with an exact constant time translation are strong
    # mechanical evidence. Do not confuse a start backstep with end ordering.
    for events in by_attributes.values():
        previous_step = None
        run = 1
        for previous, current in zip(events, events[1:]):
            start_step = round(current[0] - previous[0], 6)
            end_step = round(current[1] - previous[1], 6)
            if start_step > 0 and start_step == end_step:
                run = run + 1 if start_step == previous_step else 2
                previous_step = start_step
                if run >= 12:
                    return "uniform_repeated_events"
            else:
                previous_step = None
                run = 1
    return None

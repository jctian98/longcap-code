import json

import pytest

from espnet2.speechlm.vllm.vllm_client.loop_detection import detect_loop


def event(text, start, end, **attributes):
    attributes = {"2.1 文字转写": text, **attributes}
    return json.dumps(attributes, ensure_ascii=False) + f" <{start}, {end}>"


@pytest.mark.parametrize("unit", ["啦", "Ooh ", "a little bit of a ", "그..."])
def test_runaway_unclosed_string(unit):
    assert detect_loop('{"2.1 文字转写": "' + unit * 600) == "repeated_string_value"


def test_complete_value_can_loop_too():
    assert detect_loop(event("Oh " * 500, 0, 1)) == "repeated_string_value"


def test_short_song_refrain_is_not_a_loop():
    assert detect_loop(event("哦" * 16, -0.5, 4)) is None


def test_long_unique_text_is_not_a_loop():
    assert detect_loop(event(" ".join(f"item{i}" for i in range(2000)), 0, 32)) is None


def test_repeated_schema_keys_are_not_content_loops():
    text = "\n".join(event(f"utterance {i}", i, i + 0.5) for i in range(200))
    assert detect_loop(text) is None


def test_exact_event_loop():
    text = "\n".join(event("same utterance", -0.5, 2) for _ in range(4))
    assert detect_loop(text) == "repeated_timed_event"


def test_two_duplicate_events_are_not_a_runaway_loop():
    assert detect_loop("\n".join([event("yes", 0, 1)] * 2)) is None


def test_mechanically_advancing_events():
    text = "\n".join(event("观众笑声", i * 0.5, (i + 1) * 0.5) for i in range(30))
    assert detect_loop(text) == "uniform_repeated_events"


def test_ordinary_repeated_dialogue_has_individual_timing():
    starts = [0.0, 1.1, 2.8, 4.6, 5.9, 7.2, 8.6, 10.3, 11.6, 13.0]
    text = "\n".join(event("Vicky Lester.", s, s + 0.7) for s in starts)
    assert detect_loop(text) is None


def test_nine_uniform_repetitions_can_be_real_dialogue():
    # The reference has several crew members repeating "Ready, Mr. Burke".
    # The prediction regularizes their timing; that alone is not a clear loop.
    text = "\n".join(
        event("Ready, Mr. Burke.", 10.92 + 1.5 * i, 12.42 + 1.5 * i)
        for i in range(9)
    )
    assert detect_loop(text) is None


def test_negative_starts_and_end_ordered_overlap_are_normal():
    text = "\n".join(event(f"utterance {i}", -i, i + 1) for i in range(30))
    assert detect_loop(text) is None


def test_timestamp_backsteps_alone_are_not_loop_evidence():
    text = "\n".join(event(f"utterance {i}", i % 3, i % 3 + 1) for i in range(30))
    assert detect_loop(text) is None


def test_incomplete_normal_json_is_not_a_loop():
    assert detect_loop(event("first", 0, 1) + '\n{"2.1 文字转写": "second') is None

"""Repetition cleanup preserves first-source events and distinct occurrences."""
import json

import pytest

from longcap.processing.deduplicate import (
    LEGACY_POLICY_VERSION, deduplicate_events, deduplicate_file, inspect_chunk,
)


def event(start=10, end=11, **attributes):
    return {"start": start, "end": end,
            "attributes": {"type": "speech", "speaker": "S1", "caption": "Five dollars or two days.", **attributes}}


def test_backwards_expanding_106_event_run_keeps_first_and_other_utterance():
    # Long-form audit: one different utterance, then identical attributes with
    # overlapping/backwards-expanding timestamps. Retain source-order timing.
    first = event(5600.84, 5602.14)
    distinct = event(5599, 5601, caption="The preceding utterance.")
    repeats = [first] + [event(5600.84 - i, 5602.14 + i) for i in range(1, 106)]
    retained, groups = deduplicate_events([distinct] + repeats)
    assert retained == [distinct, first]
    assert sum(len(g["removed_indices"]) for g in groups) == 105
    assert all(g["kept_index"] == 1 for g in groups)
    assert deduplicate_events(retained) == (retained, [])


def test_distinct_speakers_types_and_attributes_are_preserved():
    events = [event(), event(speaker="S2"), event(type="music"),
              event(caption="Different"), event(emotion="happy")]
    assert deduplicate_events(events) == (events, [])


def test_disjoint_and_touching_occurrences_survive():
    events = [event(0, 1), event(1, 2), event(3, 4)]
    assert deduplicate_events(events) == (events, [])


def test_exact_duplicates_resolve_to_retained_overlap_representative():
    events = [event(2, 4), event(1, 3), event(1, 3), event(10, 11)]
    retained, groups = deduplicate_events(events)
    assert retained == [events[0], events[-1]]
    assert {g["kept_index"] for g in groups} == {0}


def test_export_preserves_original_bytes_and_refuses_overwrite(tmp_path):
    source, target = tmp_path / "original.jsonl", tmp_path / "deduplicated.jsonl"
    original = json.dumps({"key": "x", "events": [event(), event()]}) + "\n"
    source.write_text(original)
    report = deduplicate_file(source, target)
    assert source.read_text() == original
    assert report["original_events"] == 2 and report["retained_events"] == 1
    assert target.with_suffix(".deduplication.json").exists()
    for destination in (source, target):
        with pytest.raises(ValueError, match="fresh separate"):
            deduplicate_file(source, destination)


def test_existing_receipt_is_never_overwritten(tmp_path):
    source, target = tmp_path / "source.jsonl", tmp_path / "output.jsonl"
    source.write_text(json.dumps({"key": "x", "events": [event()]}) + "\n")
    target.with_suffix(".deduplication.json").write_text("existing evidence")
    with pytest.raises(ValueError, match="fresh separate"):
        deduplicate_file(source, target)
    assert not target.exists()


def test_numbered_captions_do_not_hide_same_speech_at_same_time():
    events = [event(4970.54, 4976.06, transcript="I'm glad to see you.",
                    caption=f"Oliver repeats his greeting for the {i}th time.")
              for i in range(27, 57)]
    original = json.dumps(events)
    retained, groups = deduplicate_events(events)
    assert retained == [events[0]]
    assert groups == [{"kept_index": 0, "removed_indices": list(range(1, 30)),
                       "reason": "speech_attributes_except_caption_overlapping_intervals"}]
    assert json.dumps(events) == original
    assert deduplicate_events(retained) == (retained, [])
    assert deduplicate_events(events, policy=LEGACY_POLICY_VERSION) == (events, [])


@pytest.mark.parametrize("difference", [
    {"speaker": "S2"}, {"transcript": "A different greeting."},
    {"emotion": "happy"}, {"type": "music"},
])
def test_caption_relaxation_keeps_every_other_attribute_distinction(difference):
    events = [event(transcript="hello", caption="first"),
              event(**{"transcript": "hello", "caption": "second", **difference})]
    assert deduplicate_events(events) == (events, [])


@pytest.mark.parametrize("attributes", [
    {}, {"transcript": ""}, {"transcript": "  "},
    {"type": "music", "transcript": "hello"},
    {"type": "sound_event", "transcript": "hello"},
])
def test_missing_text_and_non_speech_require_full_caption_equality(attributes):
    events = [event(caption="first", **attributes), event(caption="second", **attributes)]
    assert deduplicate_events(events) == (events, [])


@pytest.mark.parametrize("bounds", [[(0, 1), (1, 2)], [(0, 1), (3, 4)],
                                  [(1, 1), (1, 1)], [(2, 1), (2, 1)]])
def test_caption_relaxation_requires_positive_overlap(bounds):
    events = [event(a, b, transcript="hello", caption=str(i))
              for i, (a, b) in enumerate(bounds)]
    assert deduplicate_events(events) == (events, [])


def test_chinese_transcript_and_mixed_exact_duplicates_have_stable_receipt():
    events = [event(2, 4, 文字转写="你好", caption="first"),
              event(1, 3, 文字转写="你好", caption="second"),
              event(1, 3, 文字转写="你好", caption="second")]
    retained, groups = deduplicate_events(events)
    assert retained == events[:1]
    assert all(g["kept_index"] == 0 for g in groups)
    assert sorted(i for g in groups for i in g["removed_indices"]) == [1, 2]


def test_high_event_count_is_a_warning_not_an_event_cap():
    events = [event(i / 2, (i + 1) / 2, transcript=f"word {i}") for i in range(30)]
    report = inspect_chunk(events, 0, 16)
    assert report["warning_reasons"] == ["high_event_count"]
    assert report["num_events"] == report["num_retained_events"] == 30
    assert report["events_per_second"] == 30 / 16
    assert deduplicate_events(events) == (events, [])


def test_chunk_warning_survives_removal_of_numbered_repetitions():
    events = [event(transcript="hello", caption=str(i)) for i in range(30)]
    report = inspect_chunk(events, 0, 16)
    assert report["warning_reasons"] == ["high_event_count", "repeated_overlapping_events", "repeated_speech_text"]
    assert report["num_events"] == report["largest_duplicate_group"] == 30
    assert report["num_retained_events"] == 1


def test_repeated_speech_with_disjoint_times_is_reported_but_preserved():
    events = [event(i, i + 0.5, transcript="Goodbye.", caption=str(i)) for i in range(19)]
    report = inspect_chunk(events, 0, 24)
    assert report["warning_reasons"] == ["repeated_speech_text"]
    assert report["max_transcript_repetitions"] == 19
    assert report["most_repeated_transcript"] == "Goodbye."
    assert report["num_events"] == report["num_retained_events"] == 19
    assert deduplicate_events(events) == (events, [])

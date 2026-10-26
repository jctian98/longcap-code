"""Reject events that begin beyond the audio heard in their generating turn.

Applies uniformly to every event type, before overlap deduplication. The 1 s
tolerance accommodates small clock errors. Events beginning within the heard
audio retain their original ends; negative starts and cross-chunk overlaps are
preserved. This is inference postprocessing, not a training correction.
"""
import math

POLICY_VERSION = 'speech_overlap_v2_plus_future_start_v1'
TOLERANCE_SECONDS = 1.0


def filter_future_starts(events, chunk_end, tolerance=TOLERANCE_SECONDS):
    if not math.isfinite(chunk_end) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Finite audio end and nonnegative tolerance required')
    kept, removed = [], []
    for index, event in enumerate(events):
        start = event['start']
        if isinstance(start, (int, float)) and math.isfinite(start) and start >= chunk_end + tolerance:
            removed.append({'index': index, 'event': event, 'audio_end_seconds': chunk_end,
                            'seconds_beyond_heard_audio': start - chunk_end,
                            'reason': 'event_starts_after_available_audio'})
        else:
            kept.append(event)
    return kept, removed

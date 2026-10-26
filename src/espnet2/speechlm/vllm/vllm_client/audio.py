"""Client-side audio math: placeholder lengths without touching the audio.

The reference pipeline (``ContinuousAudioIO.preprocess``) is, per segment:
    native-SR slice -> first channel -> torchaudio resample to 16 kHz
    -> mel frames = resampled_samples // 160
    -> LLM positions = qwen3_omni_output_length(mel_frames)

Everything here is exact integer arithmetic replicating that pipeline, so the
client can size placeholder spans (and budget context) from sample counts
alone. Gate G2 validates these formulas against the reference implementation
over every eval-set segment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def resampled_num_samples(num_samples: int, orig_sr: int, new_sr: int = 16000) -> int:
    """Output length of ``torchaudio.functional.resample``.

    torchaudio computes ``ceil(new_freq * length / orig_freq)`` on the
    gcd-reduced frequencies (float math in torch; exact integer ceil-div here
    is equivalent for realistic lengths and validated in tests).
    """
    if orig_sr == new_sr:
        return num_samples
    gcd = math.gcd(int(orig_sr), int(new_sr))
    orig, new = orig_sr // gcd, new_sr // gcd
    return -(-num_samples * new // orig)  # ceil division


def mel_frames(num_samples_16k: int, hop_length: int = 160) -> int:
    """Mel frame count fed to the encoder (``wav.shape[0] // hop_length``)."""
    return num_samples_16k // hop_length


def qwen3_omni_output_length(frame_length: int) -> int:
    """LLM positions produced for ``frame_length`` mel frames.

    Copied from ``ContinuousAudioIO.find_length`` (Qwen3-Omni branch), itself
    copied from HF ``modeling_qwen3_omni_moe.py``. Python floor division
    matches the reference for the ``frame_length % 100 == 0`` edge (yields 0
    for the remainder term). ~13 positions per second of audio.
    """
    leave = frame_length % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (frame_length // 100) * 13


def placeholder_length(num_samples: int, orig_sr: int) -> int:
    """LLM placeholder positions for a native-SR audio slice."""
    n16k = resampled_num_samples(num_samples, orig_sr)
    return qwen3_omni_output_length(mel_frames(n16k))


@dataclass(frozen=True)
class AudioSegment:
    """A reference to a slice of a registered (or local) audio.

    ``start_sample``/``end_sample`` are in the *native* sample rate, matching
    the reference pipeline which slices before resampling. ``audio_id`` is a
    service registry id; ``path`` may be used instead for local/in-process
    use. ``end_sample`` may be None meaning end-of-file.
    """

    audio_id: str | None
    path: str | None
    start_sample: int
    end_sample: int | None
    sample_rate: int

    @property
    def num_samples(self) -> int:
        if self.end_sample is None:
            raise ValueError("end_sample must be resolved before length math")
        return self.end_sample - self.start_sample

    def placeholder_length(self) -> int:
        return placeholder_length(self.num_samples, self.sample_rate)

    def uuid(self) -> str:
        """Deterministic multimodal identity for all vLLM caches."""
        source = self.audio_id or self.path
        return f"{source}:{self.start_sample}:{self.end_sample}"

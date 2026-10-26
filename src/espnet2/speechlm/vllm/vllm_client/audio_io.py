"""Client-side audio materialization: file path -> 16 kHz mono waveform.

``audio.py`` is pure integer math (placeholder lengths) with no I/O. This module
holds the small amount of actual file reading + resampling the offline driver
needs, so the math module stays dependency-free. The recipe mirrors the
reference ``ContinuousAudioIO.preprocess`` exactly (native-SR slice -> channel 0
-> resample to 16 kHz) and is the same loader gate G4 validated.

The service resolves audio server-side from its registry instead; the driver
only needs *some* resolver that yields ``(AudioSegment, wav16k)``, so this
filesystem resolver is injectable and replaceable.
"""

from __future__ import annotations

import numpy as np

from .audio import AudioSegment

TARGET_SR = 16000


def load_wave_16k_mono(
    path: str, start_sample: int = 0, end_sample: int | None = None
) -> np.ndarray:
    """Channel 0 of ``path[start:end]`` (native SR), resampled to 16 kHz.

    Slices in the *native* sample rate before resampling, matching the
    reference pipeline. ``end_sample=None`` means end-of-file.
    """
    import soundfile as sf

    wav, sr = sf.read(
        path, dtype="float32", always_2d=True, start=start_sample, stop=end_sample
    )  # [samples, ch]
    wav = wav.T[0]  # channel 0
    if sr != TARGET_SR:
        import torch
        import torchaudio

        wav = torchaudio.functional.resample(
            torch.from_numpy(wav), sr, TARGET_SR
        ).numpy()
    return np.ascontiguousarray(wav, dtype=np.float32)


def resolve_audio(content) -> tuple[AudioSegment, np.ndarray]:
    """Turn a dialogue audio ``content`` into ``(AudioSegment, wav16k)``.

    ``content`` is either a filesystem path (whole file is one segment) or an
    already-built :class:`AudioSegment` (with a readable ``path``).
    """
    import soundfile as sf

    if isinstance(content, AudioSegment):
        seg = content
        if seg.path is None:
            raise ValueError("AudioSegment must carry a readable path for the driver")
        wav = load_wave_16k_mono(seg.path, seg.start_sample, seg.end_sample)
        return seg, wav

    info = sf.info(content)
    seg = AudioSegment(
        audio_id=None,
        path=content,
        start_sample=0,
        end_sample=info.frames,
        sample_rate=info.samplerate,
    )
    wav = load_wave_16k_mono(content)
    return seg, wav

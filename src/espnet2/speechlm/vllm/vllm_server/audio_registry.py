"""Server-side content-addressed audio registry.

The service resolves audio *by reference* — a request carries
``(audio_id, start_sample, end_sample)`` triples, not waveforms — so a 2 h
recording is uploaded (or path-registered) **once** and then sliced lazily per
hop. This is the "zero re-upload" property gate G8c checks.

Design:
- ``register_path`` / ``register_bytes`` return a stable ``audio_id`` =
  ``sha256`` of the file *content* (content-addressed → idempotent: the same
  audio always maps to the same id, registering twice is a no-op). Path
  registration is memoized by realpath so a 2 h file is hashed at most once.
- Bytes uploads are spooled to a disk directory keyed by ``audio_id`` (so the
  process holds no large buffers); path registrations reference the shared-FS
  path in place (no copy).
- ``slice_16k(audio_id, start, end)`` reads only the requested native-SR window
  from disk (``soundfile`` seeks), takes channel 0, and resamples to 16 kHz via
  the exact same recipe as the offline loader (``audio_io.load_wave_16k_mono``),
  so a server-resolved slice is bit-identical to the driver's local resolve.
  Memory is bounded by construction (only the slice is materialized); a small
  LRU caches recently decoded slices for repeated identical windows.

Bounded memory: metadata is tiny (path + sr + frame count per id); the LRU
holds at most ``slice_cache_max`` decoded slices. Nothing scales with audio
duration except the single slice being read.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..vllm_client.audio_io import load_wave_16k_mono

_HASH_CHUNK = 1 << 20  # 1 MiB


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class _Entry:
    path: str
    sample_rate: int
    num_frames: int


class AudioRegistry:
    """Idempotent, content-addressed audio cache with lazy 16 kHz slicing."""

    def __init__(self, spool_dir: str | None = None, slice_cache_max: int = 1024):
        # slice_cache_max sizing: a multi-hop dialogue re-references EVERY past
        # clip on EVERY hop, so the cache must hold the whole dialogue's clips
        # or the hit rate collapses to ~0 once the dialogue outgrows it (an
        # 80-min movie is 233 clips ≈ 300 MB at 16 kHz f32 — cheap vs the
        # 44.1k→16k sinc resample it avoids, measured ~O(N²) per dialogue).
        self._meta: dict[str, _Entry] = {}
        self._by_path: dict[str, str] = {}  # realpath -> audio_id (dedupe)
        self._spool = Path(spool_dir) if spool_dir else None
        if self._spool is not None:
            self._spool.mkdir(parents=True, exist_ok=True)
        self._slice_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._slice_cache_max = slice_cache_max
        self._lock = threading.Lock()
        # Duplicate-work visibility: a miss = one disk read + sinc resample.
        # Under high dialogue concurrency the live slice set is
        # concurrency x clips_per_dialogue; if that exceeds slice_cache_max
        # the LRU thrashes and every hop re-resamples its whole history.
        self.slice_hits = 0
        self.slice_misses = 0

    # -- registration -------------------------------------------------------
    def register_path(self, path: str) -> dict:
        """Register an on-disk (shared-FS) audio file in place; no copy."""
        import soundfile as sf

        rp = os.path.realpath(path)
        if not os.path.isfile(rp):
            raise FileNotFoundError(path)
        with self._lock:
            aid = self._by_path.get(rp)
        if aid is None:
            aid = _sha256_file(rp)
            info = sf.info(rp)
            with self._lock:
                self._meta[aid] = _Entry(rp, info.samplerate, info.frames)
                self._by_path[rp] = aid
        return self._describe(aid)

    def register_bytes(self, data: bytes, suffix: str = ".wav") -> dict:
        """Register uploaded audio bytes; spooled to disk by content id."""
        import soundfile as sf

        if self._spool is None:
            raise RuntimeError("registry has no spool_dir; cannot accept uploads")
        aid = hashlib.sha256(data).hexdigest()
        with self._lock:
            known = aid in self._meta
        if not known:
            dest = self._spool / f"{aid}{suffix}"
            if not dest.exists():
                tmp = dest.with_suffix(dest.suffix + ".tmp")
                tmp.write_bytes(data)
                os.replace(tmp, dest)
            info = sf.info(str(dest))
            with self._lock:
                self._meta[aid] = _Entry(str(dest), info.samplerate, info.frames)
                self._by_path[os.path.realpath(dest)] = aid
        return self._describe(aid)

    # -- resolution ---------------------------------------------------------
    def slice_16k(
        self, audio_id: str, start_sample: int = 0, end_sample: int | None = None
    ) -> np.ndarray:
        """Channel-0 16 kHz waveform for a native-SR window of ``audio_id``.

        Reads only the requested window from disk (bounded memory). Raises
        ``KeyError`` if the id is unknown so the service can map it to 404.
        """
        entry = self._meta.get(audio_id)
        if entry is None:
            raise KeyError(audio_id)
        key = (audio_id, start_sample, end_sample)
        with self._lock:
            hit = self._slice_cache.get(key)
            if hit is not None:
                self._slice_cache.move_to_end(key)
                self.slice_hits += 1
                return hit
            self.slice_misses += 1
        wav = load_wave_16k_mono(entry.path, start_sample, end_sample)
        with self._lock:
            self._slice_cache[key] = wav
            self._slice_cache.move_to_end(key)
            while len(self._slice_cache) > self._slice_cache_max:
                self._slice_cache.popitem(last=False)
        return wav

    def get(self, audio_id: str) -> _Entry | None:
        return self._meta.get(audio_id)

    def _describe(self, audio_id: str) -> dict:
        e = self._meta[audio_id]
        return {
            "audio_id": audio_id,
            "sample_rate": e.sample_rate,
            "num_frames": e.num_frames,
        }

    @property
    def num_registered(self) -> int:
        return len(self._meta)

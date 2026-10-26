"""HTTP client for the captioner service — the multi-hop rollout surface.

The whole point of the split is that the **grammar lives in one place**. This
client does *not* re-implement the multi-hop control flow; it reuses
``MultiHopDriver`` (the G5/G6/G7-proven core) verbatim over an HTTP transport by
supplying two small adapters:

- ``_AudioBook.resolver`` — a drop-in ``AudioResolver``. It registers each audio
  with the service **once** (memoized by path), and returns an ``AudioSegment``
  carrying the registry ``audio_id`` plus an *empty* waveform. The driver stores
  the segment's ``uuid()`` (= ``f"{audio_id}:{start}:{end}"``) as usual; the
  empty waveform is never shipped (the server slices from its registry), so
  audio is uploaded once and referenced by id for every hop (zero re-upload).
- ``HttpEngine`` — duck-typed to ``LocalEngine`` (``generate(list) -> list``).
  It turns each ``GenerateRequest``'s ``audio_uuids`` back into registry refs via
  the book's ``uuid -> ref`` map and POSTs ``/generate``. A batched driver round
  (one pending segment per active dialogue) is fanned out concurrently, so the
  server continuous-batches exactly the requests the offline engine would.

Because the driver, prompt builder, constants, and audio math are all shared
with the offline path, a service decode is the same computation as the offline
decode — only the transport differs.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .audio import AudioSegment
from .dialogue import ExampleResult, MultiHopDriver
from .prompt import PromptBuilder
from .protocol import GenerateRequest, GenerateResult

_EMPTY_WAV = np.zeros(0, dtype=np.float32)


class CaptionerClient:
    """Thin HTTP client over the service endpoints.

    Transport robustness: uvicorn closes keep-alive connections idle longer
    than its ``timeout_keep_alive`` (default 5 s). A multi-hop client naturally
    idles between rounds (wire building, other dialogues' turns), so a pooled
    connection can be dead by the next request and the POST fails with
    ``ReadError: Connection reset by peer`` — measured at ~40% of dialogues on
    the zh diar set (2026-08-05). Every endpoint here is idempotent (the audio
    registry is content-addressed; ``/generate``/``/score`` are deterministic
    at temperature 0), so ``_post`` retries dead-connection errors on a fresh
    connection instead of surfacing them (timeouts and HTTP status errors
    still raise immediately).
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 600.0):
        import httpx

        self._httpx = httpx
        self._c = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        # Instrumentation (gate G8c "registered once"): count register calls.
        self.num_register_calls = 0

    def _post(self, url: str, *, retries: int = 3, **kwargs):
        """POST with retry on connection-level failures (idempotent endpoints).

        Retries only dead-connection errors (the keep-alive reset family) — a
        TimeoutException is NOT retried: with the 600 s client timeout a retry
        would triple the wait and re-enqueue an expensive request on a server
        that is already struggling. HTTP 4xx/5xx raise immediately, as before.
        """
        import time as _time

        if retries < 1:
            raise ValueError(f"retries must be >= 1, got {retries}")
        retriable = (
            self._httpx.ConnectError,
            self._httpx.ReadError,
            self._httpx.WriteError,
            self._httpx.RemoteProtocolError,
        )
        for attempt in range(retries):
            try:
                r = self._c.post(url, **kwargs)
                r.raise_for_status()
                return r
            except retriable:
                if attempt == retries - 1:
                    raise
                _time.sleep(0.2 * (attempt + 1))

    # -- endpoint wrappers --------------------------------------------------
    def health(self) -> dict:
        return self._c.get("/health").json()

    def models(self) -> dict:
        return self._c.get("/models").json()

    def register_path(self, path: str) -> dict:
        self.num_register_calls += 1
        return self._post("/audio", json={"path": path}).json()

    def register_bytes(self, data: bytes, filename: str = "audio.wav") -> dict:
        self.num_register_calls += 1
        return self._post("/audio/upload", files={"file": (filename, data)}).json()

    def generate(
        self, prompt_token_ids, audio_segments, sampling: dict
    ) -> dict:
        return self._post("/generate", json={
            "prompt_token_ids": list(prompt_token_ids),
            "audio_segments": list(audio_segments),
            "sampling": sampling,
        }).json()

    def score(self, prompt_token_ids, audio_segments, top_logprobs: int = 32) -> dict:
        return self._post("/score", json={
            "prompt_token_ids": list(prompt_token_ids),
            "audio_segments": list(audio_segments),
            "top_logprobs": top_logprobs,
        }).json()

    def close(self) -> None:
        self._c.close()


class _AudioBook:
    """Registers a dialogue's audio once and maps uuids back to registry refs."""

    def __init__(self, client: CaptionerClient):
        self._client = client
        self._meta_by_path: dict[str, dict] = {}  # path -> register response
        self.ref_by_uuid: dict[str, dict] = {}

    def resolver(self, content):
        """``AudioResolver``: content (path or AudioSegment) -> (segment, wav).

        The waveform is empty on purpose — the server slices from its registry.
        """
        if isinstance(content, AudioSegment):
            path = content.path
            start = content.start_sample
            end = content.end_sample
        else:
            path = content
            start, end = 0, None
        if path is None:
            raise ValueError("client audio content must carry a path to register")

        meta = self._meta_by_path.get(path)
        if meta is None:
            meta = self._client.register_path(path)
            self._meta_by_path[path] = meta
        if end is None:
            end = meta["num_frames"]

        seg = AudioSegment(
            audio_id=meta["audio_id"],
            path=path,
            start_sample=start,
            end_sample=end,
            sample_rate=meta["sample_rate"],
        )
        self.ref_by_uuid[seg.uuid()] = {
            "audio_id": meta["audio_id"],
            "start_sample": start,
            "end_sample": end,
        }
        return seg, _EMPTY_WAV


class HttpEngine:
    """``LocalEngine``-shaped adapter that runs generation over HTTP."""

    def __init__(
        self, client: CaptionerClient, audiobook: _AudioBook, max_workers: int = 64
    ):
        self._client = client
        self._book = audiobook
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def _one(self, req: GenerateRequest) -> GenerateResult:
        segs = [self._book.ref_by_uuid[u] for u in (req.audio_uuids or [])]
        resp = self._client.generate(
            req.prompt_token_ids,
            segs,
            {
                "max_tokens": req.max_tokens,
                "min_tokens": req.min_tokens,
                "temperature": req.temperature,
                "top_k": req.top_k,
                "mode": req.mode,
                "repetition_penalty": req.repetition_penalty,
            },
        )
        return GenerateResult(
            token_ids=list(resp["token_ids"]),
            terminator=resp["terminator"],
            finish_reason=resp["finish_reason"],
        )

    def generate(self, requests: list[GenerateRequest]) -> list[GenerateResult]:
        if not requests:
            return []
        futs = [self._pool.submit(self._one, r) for r in requests]
        return [f.result() for f in futs]

    def close(self) -> None:
        self._pool.shutdown(wait=False)


def build_driver(
    client: CaptionerClient,
    tokenizer,
    *,
    max_tokens: int = 4096,
    max_workers: int = 64,
    audiobook: _AudioBook | None = None,
    loop_retry: bool = False,
    caption_format_retry: bool = False,
    max_retried_chunks: int = 0,
) -> MultiHopDriver:
    """A ``MultiHopDriver`` wired to the HTTP service (grammar reused verbatim)."""
    book = audiobook or _AudioBook(client)
    engine = HttpEngine(client, book, max_workers=max_workers)
    return MultiHopDriver(
        engine, PromptBuilder(tokenizer),
        audio_resolver=book.resolver, max_tokens=max_tokens, loop_retry=loop_retry,
        caption_format_retry=caption_format_retry,
        max_retried_chunks=max_retried_chunks,
    )


def decode_dialogues(
    client: CaptionerClient,
    tokenizer,
    items: list[tuple[str, list]],
    *,
    max_tokens: int = 4096,
    max_workers: int = 64,
) -> dict[str, ExampleResult]:
    """Decode ``[(example_id, messages)]`` concurrently against the service."""
    driver = build_driver(
        client, tokenizer, max_tokens=max_tokens, max_workers=max_workers
    )
    return driver.run(items)

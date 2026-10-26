"""Wire protocol shared by the pure-CPU client and the GPU server.

These are the request/result dataclasses that cross the client/server boundary
(both the in-process ``LocalEngine`` path and the HTTP path go through the same
shapes). They live in ``vllm_client`` — the pure-CPU side — because the client
*owns* the wire contract; the server imports them. Nothing here imports vLLM, so
the client stays installable and testable without CUDA.

``EngineLike`` is the structural type the multi-hop driver (:mod:`.dialogue`)
depends on: any object exposing ``generate(list[GenerateRequest]) ->
list[GenerateResult]`` satisfies it. Both the in-process
``vllm_server.engine.LocalEngine`` and the HTTP ``client.HttpEngine`` conform,
so the driver never references a server-side class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class GenerateRequest:
    """One stateless generation call (one assistant segment)."""

    prompt_token_ids: list[int]
    # (waveform, sample_rate) per audio item, in prompt/message order. The
    # waveform should already be 16 kHz mono (the reference target); the
    # plugin's MultiModalDataParser is then a no-op resample.
    audio: list[tuple[np.ndarray, int]] = field(default_factory=list)
    audio_uuids: list[str] | None = None
    max_tokens: int = 4096
    min_tokens: int = 1
    temperature: float = 0.0
    top_k: int = -1
    mode: str = "content"  # content | choice | none
    repetition_penalty: float = 1.0


@dataclass
class GenerateResult:
    """Result of one generation call."""

    token_ids: list[int]          # content only (terminator stripped)
    terminator: int | None        # 2 (eos), 3 (eot), or None on length
    finish_reason: str            # "eos" | "eot" | "length"


class EngineLike(Protocol):
    """Structural type for anything the multi-hop driver can drive.

    Satisfied by both ``vllm_server.engine.LocalEngine`` (in-process vLLM) and
    ``client.HttpEngine`` (HTTP front). Keeps :mod:`.dialogue` engine-agnostic
    and free of any server-side import.
    """

    def generate(self, requests: list[GenerateRequest]) -> list[GenerateResult]:
        ...

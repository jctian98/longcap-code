"""Multi-hop decode driver — the stateless client mirror of the torch reference.

The torch reference (``ParallelLLM.inference`` / ``inference_round``,
parallel.py) carries one hybrid cache across the whole dialogue and, per round:
prefills the run of non-assistant messages, then loops assistant *segments* —
emit ``<assistant>``, pick the modality (here enforced to text), generate
content greedily, stop on ``<eos>``/``<eot>``; ``<eos>`` ends the round,
``<eot>`` opens another segment in the same round. Round boundaries come from
where assistant messages sit in the input; the number of segments per round is
decided by the model (the input assistant slots are ignored during generation,
exactly as the reference ignores its ``references`` in production).

This driver reproduces that control flow *statelessly*: it maintains a growing
**wire-form** prefix (one id-0 seed per audio, expanded server-side) plus the
16 kHz waveforms and their multimodal uuids, and calls the engine once per
assistant segment. The generated content tokens and the model-emitted
terminator are appended verbatim, so the running wire sequence stays identical
to what the reference would serialize. Reuse across hops is pure APC (keyed by
the stable uuids); correctness never depends on it (gate G5).

Output is the reference ``results.json`` shape: assistant-only messages, one per
segment, content as a one-element list — so the existing scorers run unchanged.

Batching: :meth:`run` drives many dialogues concurrently by advancing each as a
coroutine and issuing one batched engine call per lockstep round (one pending
segment per still-active dialogue). Solo == one-item batch, which makes
solo-vs-batched invariance (gate G5) a direct comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

from . import constants as C
from .audio import AudioSegment
from .audio_io import resolve_audio
from .caption_format import caption_format_error
from .loop_detection import detect_loop
from .prompt import PromptBuilder
from .protocol import EngineLike, GenerateRequest

# content -> (AudioSegment, wav16k). Injectable so the service can resolve from
# its registry instead of the filesystem.
AudioResolver = Callable[[object], "tuple[AudioSegment, np.ndarray]"]
LOOP_RETRY_PENALTIES = (1.0, 1.05, 1.10, 1.15)


@dataclass
class ExampleResult:
    """Decode result for one dialogue, in reference ``results.json`` shape."""

    example_id: str
    messages: list  # [["assistant", "text", [text]], ...]
    message_round_indices: list[int] = field(default_factory=list)
    num_rounds: int = 0
    num_segments: int = 0
    truncated_hops: int = 0  # selected segments that hit the length cap
    loop_retry_attempts: list[dict] = field(default_factory=list)
    caption_retry_attempts: list[dict] = field(default_factory=list)
    retried_chunks: int = 0
    early_stop: dict | None = None


class MultiHopDriver:
    """Drives multi-hop captioner decoding over an in-process engine."""

    def __init__(
        self,
        engine: EngineLike,
        prompt_builder: PromptBuilder,
        *,
        audio_resolver: AudioResolver = resolve_audio,
        max_tokens: int = 4096,
        max_segments_per_round: int = 64,
        loop_retry: bool = False,
        caption_format_retry: bool = False,
        max_retried_chunks: int = 0,
    ):
        self.engine = engine
        self.prompt = prompt_builder
        self.resolve_audio = audio_resolver
        self.max_tokens = max_tokens
        self.max_segments_per_round = max_segments_per_round
        self.loop_retry = loop_retry
        if caption_format_retry and not loop_retry:
            raise ValueError("caption_format_retry requires loop_retry")
        self.caption_format_retry = caption_format_retry
        if max_retried_chunks < 0 or (max_retried_chunks and not loop_retry):
            raise ValueError("max_retried_chunks must be nonnegative and requires loop_retry")
        self.max_retried_chunks = max_retried_chunks
        # outputs of dialogues still in flight, so a caller can salvage the rounds
        # decoded before an engine error (e.g. context overflow) aborts the run
        self.partial: dict[str, list] = {}

    # -- wire-prefix serialization -----------------------------------------
    @staticmethod
    def _terminators(messages: Sequence[tuple]) -> list[int]:
        # Depends only on roles (assistant contents may be empty placeholders).
        return PromptBuilder._terminators(messages)

    def _append_prefill(
        self, role, modality, content, terminator, include_bos, state
    ) -> None:
        """Append one non-assistant message to the growing wire prefix."""
        wire, waves, uuids = state["wire"], state["waves"], state["uuids"]
        if include_bos:
            wire.append(C.BOS)
        wire.append(C.ROLE_TOKENS[role])
        wire.append(C.MODALITY_TOKENS[modality])
        if modality == "text":
            wire.extend(self.prompt.encode_text(content))
        elif modality == "audio":
            seg, wav16k = self.resolve_audio(content)
            wire.append(C.AUDIO_PLACEHOLDER)  # single seed; server expands
            waves.append((wav16k, 16000))
            uuids.append(seg.uuid())
        else:
            raise ValueError(f"unsupported modality {modality!r}")
        wire.append(terminator)

    # -- per-dialogue coroutine --------------------------------------------
    def _drive(self, example_id: str, messages: Sequence[tuple]):
        """Coroutine: yields GenerateRequest, is sent GenerateResult back.

        Returns an :class:`ExampleResult` via ``StopIteration.value``.
        """
        if not messages:
            raise ValueError(f"{example_id}: empty dialogue")
        terms = self._terminators(messages)
        state = {"wire": [], "waves": [], "uuids": []}
        outputs: list = []
        message_round_indices: list[int] = []
        self.partial[example_id] = outputs
        num_rounds = num_segments = truncated = 0
        loop_retry_attempts = []
        caption_retry_attempts = []
        retried_rounds: set[int] = set()
        early_stop = None
        expected_rounds = sum(
            message[0] != "assistant" and (index == 0 or messages[index - 1][0] == "assistant")
            for index, message in enumerate(messages)
        )

        i = 0
        while i < len(messages):
            start = i
            while i < len(messages) and messages[i][0] != "assistant":
                role, modality, content = messages[i]
                self._append_prefill(
                    role, modality, content, terms[i], include_bos=(i == 0),
                    state=state,
                )
                i += 1
            if i == start:
                raise ValueError(
                    f"{example_id}: round must start with a system/user message; "
                    f"found assistant at index {i}"
                )
            num_rounds += 1

            # Generate assistant segments until the model emits <eos>.
            for _seg in range(self.max_segments_per_round):
                prompt_ids = state["wire"] + [C.ASSISTANT, C.TEXT]
                req = GenerateRequest(
                    prompt_token_ids=prompt_ids,
                    audio=list(state["waves"]),
                    audio_uuids=list(state["uuids"]),
                    max_tokens=self.max_tokens,
                    min_tokens=1,
                    mode="content",
                )
                attempts = []
                penalties = LOOP_RETRY_PENALTIES if self.loop_retry else (1.0,)
                for attempt_index, penalty in enumerate(penalties):
                    # No rejected output enters the next attempt's prompt/history.
                    result = yield replace(req, repetition_penalty=penalty)
                    text = self.prompt.decode_text(result.token_ids)
                    reason = detect_loop(text) if self.loop_retry else None
                    format_error = (caption_format_error(text)
                                    if self.caption_format_retry else None)
                    if self.loop_retry:
                        attempts.append({
                            "attempt_index": attempt_index,
                            "penalty": penalty,
                            "text": text,
                            "output_tokens": len(result.token_ids),
                            "generated_tokens": (len(result.token_ids)
                                                 + int(result.terminator is not None)),
                            "max_tokens": req.max_tokens,
                            "finish_reason": result.finish_reason,
                            "terminator": result.terminator,
                            "loop_reason": reason,
                            **({"format_error": format_error}
                               if self.caption_format_retry else {}),
                        })
                    if reason is None and format_error is None:
                        break
                if attempts and (attempts[0]["loop_reason"] is not None
                                 or attempts[0].get("format_error") is not None):
                    retry_record = {
                        "round_index": num_rounds - 1,
                        "segment_index": _seg,
                        "attempts": attempts,
                        "selected_attempt_index": len(attempts) - 1,
                        "exhausted": reason is not None or format_error is not None,
                        "total_output_tokens": sum(a["output_tokens"] for a in attempts),
                        "total_attempt_max_tokens": len(attempts) * req.max_tokens,
                    }
                    if attempts[0]["loop_reason"] is not None:
                        loop_retry_attempts.append(retry_record)
                    if self.caption_format_retry:
                        caption_retry_attempts.append(retry_record)
                    # Multiple attempts/segments/reasons in one audio chunk count once.
                    if len(attempts) > 1:
                        retried_rounds.add(num_rounds - 1)
                num_segments += 1
                outputs.append(["assistant", "text", [text]])
                message_round_indices.append(num_rounds - 1)

                # Commit <assistant><text> content [terminator] to the prefix.
                committed = [C.ASSISTANT, C.TEXT] + list(result.token_ids)
                if result.finish_reason == "length":
                    truncated += 1
                    state["wire"].extend(committed)  # no terminator emitted
                    break  # contain the failure; end this round
                state["wire"].extend(committed + [result.terminator])
                if result.finish_reason == "eos":
                    break
                # "eot": another assistant segment continues the same round.
            else:
                # The segment cap ended the round without an EOS. Preserve all
                # replies while exposing that the generated round is incomplete.
                truncated += 1

            # Skip over the input's assistant placeholder slots for this round.
            while i < len(messages) and messages[i][0] == "assistant":
                i += 1
            # Keep the selected reply and every retry from the threshold chunk.
            # Stop before resolving/prefilling any later audio. Finishing the
            # final input chunk at the limit is still a complete recording.
            if (i < len(messages) and self.max_retried_chunks
                    and len(retried_rounds) >= self.max_retried_chunks):
                early_stop = {
                    "reason": "max_retried_chunks",
                    "limit": self.max_retried_chunks,
                    "retried_chunks": len(retried_rounds),
                    "retried_chunk_indices": sorted(retried_rounds),
                    "decoded_chunks": num_rounds,
                    "expected_chunks": expected_rounds,
                    "remaining_chunks": expected_rounds - num_rounds,
                    "stop_after_chunk_index": num_rounds - 1,
                }
                break

        self.partial.pop(example_id, None)
        return ExampleResult(
            example_id=example_id,
            messages=outputs,
            message_round_indices=message_round_indices,
            num_rounds=num_rounds,
            num_segments=num_segments,
            truncated_hops=truncated,
            loop_retry_attempts=loop_retry_attempts,
            caption_retry_attempts=caption_retry_attempts,
            retried_chunks=len(retried_rounds),
            early_stop=early_stop,
        )

    # -- public API ---------------------------------------------------------
    def run(self, items: Sequence[tuple]) -> dict[str, ExampleResult]:
        """Decode a batch of dialogues concurrently.

        ``items`` is ``[(example_id, messages), ...]``. Returns
        ``{example_id: ExampleResult}``. Passing a single item runs it solo.
        """
        coros = {eid: self._drive(eid, msgs) for eid, msgs in items}
        results: dict[str, ExampleResult] = {}
        pending: dict[str, GenerateRequest] = {}
        for eid, co in coros.items():
            try:
                pending[eid] = next(co)  # prime to first yield
            except StopIteration as stop:
                results[eid] = stop.value

        while pending:
            eids = list(pending.keys())
            reqs = [pending[e] for e in eids]
            batch = self.engine.generate(reqs)
            pending = {}
            for e, res in zip(eids, batch):
                try:
                    pending[e] = coros[e].send(res)
                except StopIteration as stop:
                    results[e] = stop.value
        return results

    @staticmethod
    def to_results_json(results: dict[str, ExampleResult]) -> dict:
        """Reference-format mapping ``{example_id: [assistant messages]}``."""
        return {eid: r.messages for eid, r in results.items()}

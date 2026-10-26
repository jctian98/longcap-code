"""Client-side prompt construction: dialogue -> SpeechLM token ids.

Bit-exact mirror of the reference serializer
``SpeechLMInferencePreprocessor.process_inference_dialogue`` /
``process_inference_message`` (espnet2/speechlm/model/speechlm/speechlm_job.py)
for the captioner setup (single stream, continuous-audio input, text output).

A message is ``(role, modality, content)`` where ``modality in {text, audio}``.
Per message the reference emits::

    [<|bos|> if first] <|role|> <|modality|> <content...> <|terminator|>

- terminator is ``<|eot|>`` iff the next message has the same role, else
  ``<|eos|>`` (last message -> ``<|eos|>``).
- text content = HF tokenizer ids + 256 (``add_special_tokens=True``, which is
  a no-op for Qwen).
- audio content = ``AUDIO_PLACEHOLDER`` repeated ``placeholder_length`` times.

The builder produces a *round* decomposition matching the reference: a round is
one prefill (a run of consecutive non-assistant messages) followed by the
assistant turn(s) to be generated. Decoding drives one round at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from . import constants as C
from .audio import AudioSegment


@dataclass
class PromptRound:
    """One decode round: a prefill span plus its audio placeholder spans."""

    prefill_ids: list[int]
    # (start_position_within_prefill, length, segment) for each audio item
    audio_spans: list[tuple[int, int, AudioSegment]] = field(default_factory=list)


class PromptBuilder:
    """Turns dialogues into SpeechLM token ids using an HF tokenizer.

    Stateless w.r.t. dialogues; holds only the tokenizer. Text detokenization
    (ids - 256 -> HF decode) is provided for reading generated captions.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    # -- text ---------------------------------------------------------------
    def encode_text(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text, truncation=False, add_special_tokens=True)
        return [i + C.TEXT_START for i in ids]

    def decode_text(self, ids: Sequence[int]) -> str:
        hf_ids = [i - C.TEXT_START for i in ids if i >= C.TEXT_START]
        return self.tokenizer.decode(
            hf_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

    # -- message serialization ---------------------------------------------
    @staticmethod
    def _terminators(messages: Sequence[tuple]) -> list[int]:
        out = []
        for i, msg in enumerate(messages):
            same_next = i + 1 < len(messages) and messages[i + 1][0] == msg[0]
            out.append(C.EOT if same_next else C.EOS)
        return out

    def _serialize_message(
        self, role: str, modality: str, content, terminator: int, include_bos: bool
    ) -> tuple[list[int], tuple[int, int, AudioSegment] | None]:
        ids: list[int] = []
        if include_bos:
            ids.append(C.BOS)
        content_start = int(include_bos) + 2  # after role + modality tokens
        ids.append(C.ROLE_TOKENS[role])
        ids.append(C.MODALITY_TOKENS[modality])

        audio_span = None
        if modality == "text":
            ids.extend(self.encode_text(content))
        elif modality == "audio":
            if not isinstance(content, AudioSegment):
                raise TypeError("audio content must be an AudioSegment")
            length = content.placeholder_length()
            ids.extend([C.AUDIO_PLACEHOLDER] * length)
            audio_span = (content_start, length, content)
        else:
            raise ValueError(f"unsupported modality {modality!r}")

        ids.append(terminator)
        return ids, audio_span

    # -- dialogue -> rounds -------------------------------------------------
    def build_rounds(self, messages: Sequence[tuple]) -> list[PromptRound]:
        """Group messages into decode rounds (mirrors the reference)."""
        if not messages:
            raise ValueError("dialogue must contain at least one message")
        terms = self._terminators(messages)

        rounds: list[PromptRound] = []
        i = 0
        while i < len(messages):
            start = i
            prefill_ids: list[int] = []
            audio_spans: list[tuple[int, int, AudioSegment]] = []
            while i < len(messages) and messages[i][0] != "assistant":
                role, modality, content = messages[i]
                seg_ids, span = self._serialize_message(
                    role, modality, content, terms[i], include_bos=(i == 0)
                )
                if span is not None:
                    off, length, seg = span
                    audio_spans.append((len(prefill_ids) + off, length, seg))
                prefill_ids.extend(seg_ids)
                i += 1
            if i == start:
                raise ValueError(
                    "each round must start with a system/user message; "
                    f"found assistant at index {i}"
                )
            rounds.append(PromptRound(prefill_ids, audio_spans))
            while i < len(messages) and messages[i][0] == "assistant":
                i += 1
        return rounds

    def build_flat(self, messages: Sequence[tuple]) -> tuple[list[int], list]:
        """Full expanded token id sequence + audio spans (for parity checks).

        Serializes *every* message including assistant text (used by the G1
        gate to compare against the reference training preprocessor, and by
        teacher-forced scoring). Returns (ids, [(abs_start, length, seg), ...]).
        """
        terms = self._terminators(messages)
        ids: list[int] = []
        spans: list[tuple[int, int, AudioSegment]] = []
        for i, (role, modality, content) in enumerate(messages):
            seg_ids, span = self._serialize_message(
                role, modality, content, terms[i], include_bos=(i == 0)
            )
            if span is not None:
                off, length, seg = span
                spans.append((len(ids) + off, length, seg))
            ids.extend(seg_ids)
        return ids, spans

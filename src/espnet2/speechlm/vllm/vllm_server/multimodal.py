"""Multimodal processing for the SpeechLM captioner vLLM plugin.

The captioner's continuous-audio input is serialized, per LLM position, to the
placeholder id 0 (``<|pad|>``); the embeddings at those positions are later
overwritten by ``adaptor(audio_tower(mel))`` (see ``model.py``). This module
tells vLLM how to (a) turn a raw waveform into the encoder's mel input, (b)
size the placeholder span, and (c) generate dummy inputs for memory profiling.

Why we override ``_call_hf_processor`` and force ``_hf_processor_applies_updates
=> False``: there is no packaged HF *processor* that emits our id-0 placeholder
run from text (the base tokenizer can never produce id 0). So instead:

- ``_call_hf_processor`` computes mel features itself (exact reference recipe,
  ``ContinuousAudioIO.preprocess`` Qwen3-Omni branch) and synthesizes
  ``input_ids = [[0] * prompt.count(SENTINEL)]`` — one id-0 seed per audio item,
  derived purely from the *string* prompt (which is passed through verbatim and
  never tokenized in the text+mm path). In the mm-only and token-ids runtime
  paths this ``input_ids`` is discarded by vLLM anyway.
- ``_hf_processor_applies_updates`` returns False so the ``PromptReplacement``
  (``[0] -> [0] * out_len``) below runs in *every* processing path, expanding
  each single id-0 seed to the audio's true placeholder length.

This design was validated against all four vLLM processing entry paths
(profiling with prompt-update enabled/disabled, and the token-ids runtime path).
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from transformers import BatchFeature, WhisperFeatureExtractor

from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.config.multimodal import BaseDummyOptions

from ..vllm_client.audio import qwen3_omni_output_length

from . import config as _cfg

# The audio placeholder token id (``<|pad|>`` == 0). One per LLM position of
# audio; embeddings overwritten by the tower+adaptor in the model wrapper.
AUDIO_PLACEHOLDER_ID = 0

# A sentinel substring, one per audio item, used only to size ``input_ids`` in
# the text+mm profiling path. It is never tokenized (we own ``_call_hf_processor``
# and read the raw string), so its exact value is irrelevant as long as it does
# not appear elsewhere in a dummy/profiling prompt.
SENTINEL = "<|speechlm_audio|>"

# Bound on a single audio item's mel frames for memory profiling / encoder
# budget, mirroring the Qwen2-Audio/Qwen3-Omni convention (cap at 30 s).
_PROFILE_CHUNK_SECONDS = 30

# model_path -> WhisperFeatureExtractor (immutable once loaded; one per model).
_FEATURE_EXTRACTOR_CACHE: dict[str, WhisperFeatureExtractor] = {}


class SpeechLMProcessingInfo(BaseProcessingInfo):
    """Static info: feature extractor, mm limits, per-item token budget."""

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # Multi-hop dialogues carry an unbounded number of audio segments; the
        # engine's ``limit_mm_per_prompt`` caps it per request.
        return {"audio": None}

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        # The feature extractor is packaged next to the checkpoint (export
        # writes ``audio_processor/preprocessor_config.json``). Loading it
        # directly avoids constructing a full multimodal HF processor for a
        # model_type (qwen3_5) whose HF processor expects vision inputs.
        # Memoized per model path: this is called on EVERY mel extraction and
        # every data-parser construction, and a from_pretrained disk read per
        # request is measurable on the serving hot path.
        model_path = self.ctx.model_config.model
        fe = _FEATURE_EXTRACTOR_CACHE.get(model_path)
        if fe is None:
            fe = WhisperFeatureExtractor.from_pretrained(
                os.path.join(model_path, "audio_processor")
            )
            assert isinstance(fe, WhisperFeatureExtractor)
            _FEATURE_EXTRACTOR_CACHE[model_path] = fe
        return fe

    def get_data_parser(self) -> MultiModalDataParser:
        # Resample every incoming waveform to 16 kHz mono before it reaches
        # ``_call_hf_processor`` — matching the reference pipeline, which slices
        # at native SR then resamples to 16 kHz.
        fe = self.get_feature_extractor()
        return MultiModalDataParser(target_sr=fe.sampling_rate, target_channels=1)

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int] | None = None,
    ) -> Mapping[str, int]:
        mm_counts = mm_counts or {}
        if mm_counts.get("audio", 0) <= 0:
            return {}
        fe = self.get_feature_extractor()
        chunk_seconds = min(int(getattr(fe, "chunk_length", 30)), _PROFILE_CHUNK_SECONDS)
        max_samples = int(chunk_seconds * fe.sampling_rate)
        max_mel_frames = max_samples // fe.hop_length
        return {"audio": qwen3_omni_output_length(max_mel_frames)}


class SpeechLMDummyInputsBuilder(BaseDummyInputsBuilder[SpeechLMProcessingInfo]):
    """Profiling inputs: one sentinel + one dummy waveform per audio item."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return SENTINEL * mm_counts.get("audio", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ):
        num_audios = mm_counts.get("audio", 0)
        fe = self.info.get_feature_extractor()
        chunk_seconds = min(int(getattr(fe, "chunk_length", 30)), _PROFILE_CHUNK_SECONDS)
        length = int(chunk_seconds * fe.sampling_rate)
        audio_overrides = (mm_options or {}).get("audio")
        return {
            "audio": self._get_dummy_audios(
                length=length, num_audios=num_audios, overrides=audio_overrides
            ),
        }


class SpeechLMMultiModalProcessor(BaseMultiModalProcessor[SpeechLMProcessingInfo]):
    """Mel extraction + placeholder-span expansion for continuous audio."""

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        fe = self.info.get_feature_extractor()
        audios = mm_data.get("audios", []) or []

        input_features: list[torch.Tensor] = []
        audio_feature_lengths: list[int] = []
        for wav in audios:
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            # Reference recipe (ContinuousAudioIO.preprocess, Qwen3-Omni branch):
            # keep every real mel frame (before_length), no whisper 30 s padding.
            before_length = wav.shape[0] // fe.hop_length
            out = fe(
                [wav],
                sampling_rate=fe.sampling_rate,
                truncation=False,
                do_normalize=True,
                max_length=wav.shape[0],
                return_attention_mask=True,
                return_tensors="np",
            )
            feat = out["input_features"][0][:, :before_length]  # [num_mel_bins, T]
            input_features.append(torch.from_numpy(np.ascontiguousarray(feat)))
            audio_feature_lengths.append(int(before_length))

        # One id-0 seed per audio item; expanded to true length by the prompt
        # update below. Derived from the raw prompt string (never tokenized).
        num_seed = prompt.count(SENTINEL) if isinstance(prompt, str) else 0
        data: dict[str, object] = {
            "input_ids": [[AUDIO_PLACEHOLDER_ID] * num_seed],
        }
        if input_features:
            # [num_mel_bins, sum(audio_feature_lengths)] — sliced per item by
            # ``flat_from_sizes`` in ``_get_mm_fields_config``.
            data["input_features"] = torch.cat(input_features, dim=1)
            data["audio_feature_lengths"] = torch.tensor(
                audio_feature_lengths, dtype=torch.long
            )
        return BatchFeature(data=data)

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        # Our ``_call_hf_processor`` emits only a single id-0 seed per item; the
        # PromptReplacement below always performs the real expansion.
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        audio_feature_lengths = hf_inputs.get(
            "audio_feature_lengths", torch.empty(0, dtype=torch.long)
        )
        return dict(
            input_features=MultiModalFieldConfig.flat_from_sizes(
                "audio", audio_feature_lengths, dim=1
            ),
            audio_feature_lengths=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        out_mm_data = out_mm_kwargs.get_data()
        audio_feature_lengths = out_mm_data.get("audio_feature_lengths")
        if audio_feature_lengths is None:
            audio_output_lengths: list[int] = []
        else:
            audio_output_lengths = [
                qwen3_omni_output_length(int(length))
                for length in audio_feature_lengths
            ]

        def get_replacement(item_idx: int):
            num_features = audio_output_lengths[item_idx]
            if num_features == 0:
                raise ValueError(
                    f"audio item {item_idx} is too short to occupy any LLM "
                    "position; check the segment bounds"
                )
            return [AUDIO_PLACEHOLDER_ID] * num_features

        return [
            PromptReplacement(
                modality="audio",
                target=[AUDIO_PLACEHOLDER_ID],
                replacement=get_replacement,
            )
        ]

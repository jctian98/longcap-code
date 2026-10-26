"""vLLM model wrapper for the SpeechLM audio captioner.

``SpeechLMForConditionalGeneration`` glues three pieces together:

- a Qwen3-Omni **audio tower** (``Qwen3OmniMoeAudioEncoder``) + a linear
  **adaptor** (encoder ``output_dim`` -> LM ``hidden_size``) that turn mel
  features into per-position audio embeddings, and
- a Qwen3.5 **text backbone** (dense ``Qwen3_5ForCausalLM`` or MoE
  ``Qwen3_5MoeForCausalLM``, chosen by ``text_config.model_type``) that runs
  the hybrid (Gated-DeltaNet + full-attention) transformer and the LM head.

The continuous-audio positions serialize to placeholder id 0; the base
``SupportsMultiModal.embed_input_ids`` embeds text ids then scatters our audio
embeddings into those positions (id 0 is in-vocab, so no OOV masking needed).

Weight loading mirrors the export layout exactly (names already match vLLM):
``language_model.*`` is delegated to the backbone's own loader (which fuses
GDN + qkv via its ``packed_modules_mapping``); ``audio_tower.*`` is delegated
to the encoder's own ``load_weights`` (``AutoWeightsLoader`` recurses into any
submodule that defines one), which applies the q/k/v->qkv ``WeightsMapper``
*scoped to the tower subtree* (crucial — the LM's full-attention layers also
have ``self_attn.q_proj`` and must NOT be rewritten by the audio mapper);
``adaptor.*`` and ``stream_emb.*`` load as ordinary params.

``stream_emb`` is loaded for completeness but is a no-op at inference: the
captioner is single-stream, and the reference always zeroes stream 0.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    IsHybrid,
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForCausalLMBase,
    Qwen3_5MoeForCausalLM,
)
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder,
    _get_feat_extract_output_lengths,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY

from . import config as _cfg
from .multimodal import (
    SpeechLMDummyInputsBuilder,
    SpeechLMMultiModalProcessor,
    SpeechLMProcessingInfo,
)

# vLLM architecture string -> backbone class.
_TEXT_BACKBONE_CLASSES = {
    "Qwen3_5ForCausalLM": Qwen3_5ForCausalLM,
    "Qwen3_5MoeForCausalLM": Qwen3_5MoeForCausalLM,
}


@MULTIMODAL_REGISTRY.register_processor(
    SpeechLMMultiModalProcessor,
    info=SpeechLMProcessingInfo,
    dummy_inputs=SpeechLMDummyInputsBuilder,
)
class SpeechLMForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsMRoPE, IsHybrid
):
    supports_multimodal_pruning = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # Protocols have no __init__; initialize nn.Module explicitly.
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model_config = vllm_config.model_config

        meta = _cfg.speechlm_meta(config)
        self.num_stream = int(meta["num_stream"])
        hidden_size = int(meta["hidden_size"])
        adaptor_in = int(meta["adaptor_in_features"])
        adaptor_out = int(meta["adaptor_out_features"])

        # Audio tower (skipped in --mm-encoder-only / audio-limit-0 modes).
        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_tower = Qwen3OmniMoeAudioEncoder(
                _cfg.audio_encoder_config(config),
                prefix=maybe_prefix(prefix, "audio_tower"),
            )

        # Text backbone (dense or MoE), dispatched on text_config.model_type.
        arch = _cfg.text_backbone_arch(config)
        backbone_cls = _TEXT_BACKBONE_CLASSES[arch]
        with self._mark_language_model(vllm_config):
            self.language_model: Qwen3_5ForCausalLMBase = backbone_cls(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        # Peripherals live in the root unit (always built).
        self.adaptor = nn.Linear(adaptor_in, adaptor_out, bias=True)
        # Single-stream: loaded for parity, never applied (reference zeroes it).
        self.stream_emb = nn.Embedding(self.num_stream, hidden_size)

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    # ---- multimodal -------------------------------------------------------

    def _process_audio_input(
        self,
        input_features: torch.Tensor,
        audio_feature_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # [num_mel_bins, sum(frames)] mel -> tower -> [sum(positions), out_dim].
        audio_output_lengths = _get_feat_extract_output_lengths(audio_feature_lengths)
        audio_features = self.audio_tower(
            input_features.to(self.audio_tower.dtype),
            feature_lens=audio_feature_lengths,
            aftercnn_lens=audio_output_lengths,
        )
        # adaptor: encoder out_dim -> LM hidden_size, per position.
        audio_embeds = self.adaptor(audio_features.to(self.adaptor.weight.dtype))
        return tuple(audio_embeds.split(audio_output_lengths.tolist()))

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        input_features = kwargs.get("input_features")
        audio_feature_lengths = kwargs.get("audio_feature_lengths")
        if input_features is None or audio_feature_lengths is None:
            return []
        # vLLM may pass a leading batch dim of 1 for the batched scalar field.
        audio_feature_lengths = torch.as_tensor(audio_feature_lengths).reshape(-1)
        if input_features.dim() == 3:
            # [1, num_mel_bins, total] -> [num_mel_bins, total]
            input_features = input_features.squeeze(0)
        return self._process_audio_input(input_features, audio_feature_lengths)

    # ---- forward / logits -------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        if intermediate_tensors is not None:
            inputs_embeds = None
        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    # ---- M-RoPE -----------------------------------------------------------

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list,
    ) -> tuple[torch.Tensor, int]:
        # The captioner uses 1-D temporal positions for every token (audio is
        # serialized in time). With identical T/H/W indices, interleaved M-RoPE
        # collapses to standard RoPE — matching the reference torch decoder,
        # which uses plain sequential positions.
        seq_len = len(input_tokens)
        positions = torch.arange(seq_len, dtype=torch.long)
        llm_positions = positions.unsqueeze(0).expand(3, seq_len).clone()
        return llm_positions, 0

    # ---- weight loading ---------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # No top-level mapper: export names already match vLLM names. Each
        # subtree loader handles its own fusion (audio q/k/v->qkv in the tower
        # subclass; GDN + qkv in the backbone). mtp.* is never exported but is
        # skipped defensively.
        loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])
        return loader.load_weights(weights)

    # ---- IsHybrid (delegate to the Qwen3.5 backbone semantics) ------------

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateDtypeCalculator,
        )

        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateShapeCalculator,
        )

        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateCopyFuncCalculator,
        )

        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

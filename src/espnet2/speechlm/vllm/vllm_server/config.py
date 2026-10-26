"""Config accessors for the SpeechLM vLLM plugin.

The exported HF ``config.json`` (see ``export/export_hf.py``) is a
``model_type: qwen3_5`` config carrying:

- ``text_config``  — the Qwen3.5 (dense or MoE) hybrid text backbone config,
  with ``model_type`` in {``qwen3_5_text``, ``qwen3_5_moe_text``} and the
  multimodal vocab size (248576).
- ``audio_config`` — the Qwen3-Omni audio-encoder config (``output_dim`` etc.).
- ``speechlm``     — a small metadata block (token protocol + adaptor dims):
  ``num_special_tokens, text_start, audio_placeholder_id, eos/eot/bos/pad
  token ids, hidden_size, adaptor_in_features, adaptor_out_features,
  num_stream, audio_sample_rate, audio_hop_length``.

These helpers isolate the "read a field from the HF config" logic so the model
and processor modules stay readable. No vLLM import here — cheap to unit-test.
"""

from __future__ import annotations

from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
)

# text backbone architecture dispatch, keyed by text_config.model_type. The
# value is the vLLM CausalLM architecture string; the class is imported lazily
# in model.py to avoid importing MoE machinery when only the dense path is used.
TEXT_BACKBONE_ARCH_BY_MODEL_TYPE = {
    "qwen3_5_text": "Qwen3_5ForCausalLM",
    "qwen3_5_moe_text": "Qwen3_5MoeForCausalLM",
}


def speechlm_meta(hf_config) -> dict:
    """The ``speechlm`` metadata block as a plain dict.

    vLLM stores unknown top-level config keys verbatim; depending on the
    transformers version this may be a dict or a namespace-like object.
    """
    meta = getattr(hf_config, "speechlm", None)
    if meta is None:
        raise ValueError(
            "config.json is missing the 'speechlm' block; re-export with "
            "export/export_hf.py"
        )
    if isinstance(meta, dict):
        return meta
    return dict(vars(meta))


def text_config(hf_config):
    """The Qwen3.5 text backbone sub-config (has .model_type, .vocab_size...)."""
    tc = getattr(hf_config, "text_config", None)
    if tc is None:
        raise ValueError("config.json is missing 'text_config'")
    return tc


def text_backbone_arch(hf_config) -> str:
    """vLLM architecture string for the text backbone (dense or MoE)."""
    mt = text_config(hf_config).model_type
    try:
        return TEXT_BACKBONE_ARCH_BY_MODEL_TYPE[mt]
    except KeyError:
        raise ValueError(
            f"unsupported text_config.model_type {mt!r}; expected one of "
            f"{sorted(TEXT_BACKBONE_ARCH_BY_MODEL_TYPE)}"
        )


def audio_encoder_config(hf_config) -> Qwen3OmniMoeAudioEncoderConfig:
    """A proper ``Qwen3OmniMoeAudioEncoderConfig`` (the tower needs attr access)."""
    ac = getattr(hf_config, "audio_config", None)
    if ac is None:
        raise ValueError("config.json is missing 'audio_config'")
    if isinstance(ac, Qwen3OmniMoeAudioEncoderConfig):
        return ac
    if isinstance(ac, dict):
        return Qwen3OmniMoeAudioEncoderConfig(**ac)
    # transformers sometimes nests it as a generic config namespace.
    return Qwen3OmniMoeAudioEncoderConfig(**dict(vars(ac)))


def stack_factor(hf_config) -> int:
    """Frame-stacking factor before the adaptor.

    ``adaptor_in_features == audio output_dim * stack_factor``. For the current
    captioner the adaptor is 2048->1024 and output_dim is 2048, so the factor
    is 1 (no stacking). Kept generic so a future model that stacks frames works
    without a code change.
    """
    meta = speechlm_meta(hf_config)
    in_features = int(meta["adaptor_in_features"])
    out_dim = int(audio_encoder_config(hf_config).output_dim)
    if in_features % out_dim != 0:
        raise ValueError(
            f"adaptor_in_features ({in_features}) is not a multiple of audio "
            f"output_dim ({out_dim}); cannot infer stack factor"
        )
    return in_features // out_dim

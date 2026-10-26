"""Constrained-decoding logits processor for SpeechLM generation.

The reference torch decoder (``ParallelLLM.inference_segment`` /
``prepare_inference`` in ``espnet2/speechlm/model/speechlm/lm/parallel.py``)
constrains each generated token to a modality-specific set:

- **content** (assistant text): allowed = the text interval
  ``[text_start, text_end)`` plus the two terminators ``{eos, eot}``. This is
  the mask used for every caption token. Terminators are additionally banned
  on the very first step (``min_step=1``); that first-step ban is reproduced by
  vLLM's builtin ``min_tokens`` (with ``min_tokens=1, stop_token_ids=[eos,
  eot]``) and is NOT re-implemented here.
- **choice** (modality selection after ``<eot>``): allowed = a small set of
  modality special tokens (``{<text>,<audio>,<image>,<video>,<toolcall>}`` by
  default, overridable per request). A single constrained step picks the next
  segment's modality.
- **none**: no masking (unconstrained), for debugging / free generation.

The mode is per request via ``SamplingParams.extra_args``:
``{"speechlm_mode": "content"|"choice"|"none", "speechlm_choice_ids": [...]}``.
Default (no extra_args) is ``content`` — the common case.

Why preserve-then-restore for the terminators (content mode): we clone the
``{eos, eot}`` logits *before* masking the surrounding intervals and write them
back *after*. Builtin ``min_tokens`` runs before custom processors, so at step 0
the cloned terminator logits are already ``-inf`` and get restored as ``-inf``
(stay banned); at step >= 1 they are the model's real logits and are restored
intact. The clone/restore is therefore correct regardless of ``min_tokens``.
"""

from __future__ import annotations

import torch

from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from . import config as _cfg

# Modality special-token ids (see vllm_client.constants): <text>=7, <audio>=8,
# <image>=9, <video>=10, <toolcall>=11. Duplicated here as a plain default so
# the plugin's logits module does not import the client package at build time.
_DEFAULT_CHOICE_IDS = (7, 8, 9, 10, 11)

NEG_INF = float("-inf")


def _content_mask_fn(text_start: int, text_end: int, terminals: tuple[int, ...]):
    """Return a per-request callable masking to the text interval + terminals."""
    term_index = list(terminals)

    def apply(_output_ids, logits: torch.Tensor) -> torch.Tensor:
        # logits is the 1-D vocab vector for a single request.
        saved = logits[term_index].clone()
        logits[:text_start] = NEG_INF
        logits[text_end:] = NEG_INF
        logits[term_index] = saved
        return logits

    return apply


def _choice_mask_fn(allowed_ids: tuple[int, ...]):
    """Return a per-request callable allowing only ``allowed_ids``."""
    allowed = list(allowed_ids)

    def apply(_output_ids, logits: torch.Tensor) -> torch.Tensor:
        saved = logits[allowed].clone()
        logits[:] = NEG_INF
        logits[allowed] = saved
        return logits

    return apply


class SpeechLMConstraintLogitsProcessor(AdapterLogitsProcessor):
    """Per-request modality-constrained masking (content / choice / none)."""

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ):
        super().__init__(vllm_config, device, is_pin_memory)
        hf_config = vllm_config.model_config.hf_config
        meta = _cfg.speechlm_meta(hf_config)
        self.text_start = int(meta["text_start"])
        # text interval is [text_start, vocab_size); vocab_size lives on the
        # text backbone config.
        self.text_end = int(_cfg.text_config(hf_config).vocab_size)
        self.eos = int(meta["eos_token_id"])
        self.eot = int(meta["eot_token_id"])
        self.default_choice_ids = _DEFAULT_CHOICE_IDS

    def is_argmax_invariant(self) -> bool:
        # Masking bans tokens outside the modality set, so it can change which
        # token is the argmax (the unconstrained max may be banned). Not
        # invariant — vLLM must run us even under greedy sampling.
        return False

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        # This processor owns token masking; a per-request allowed_token_ids
        # would silently fight it. Reject it up front.
        if sampling_params.allowed_token_ids:
            raise ValueError(
                "allowed_token_ids is not supported with the SpeechLM "
                "constraint logits processor; use extra_args['speechlm_mode']"
            )
        extra = sampling_params.extra_args or {}
        mode = extra.get("speechlm_mode", "content")
        if mode not in ("content", "choice", "none"):
            raise ValueError(f"unknown speechlm_mode {mode!r}")

    def new_req_logits_processor(self, params: SamplingParams):
        extra = params.extra_args or {}
        mode = extra.get("speechlm_mode", "content")
        if mode == "none":
            return None
        if mode == "content":
            return _content_mask_fn(self.text_start, self.text_end, (self.eos, self.eot))
        # choice
        ids = extra.get("speechlm_choice_ids") or self.default_choice_ids
        return _choice_mask_fn(tuple(int(i) for i in ids))

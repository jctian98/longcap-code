"""In-process vLLM engine wrapper for the SpeechLM captioner.

This is the server-side compute surface the multi-hop driver (``dialogue.py``)
and the offline eval runner drive directly — no HTTP. It is also the shape the
future FastAPI service (``service.py``) and the verl rollout wrap: a stateless
``generate`` that takes wire-form ``prompt_token_ids`` + audio waveforms +
sampling knobs and returns generated token ids with a resolved finish reason.

The request carries the *wire form* prompt (one ``id 0`` seed per audio, which
the plugin's ``PromptReplacement`` expands to the reference placeholder length)
and the pre-16 kHz-mono waveforms in message order. ``audio_uuids`` keys vLLM's
prefix / mm / encoder caches so identical audio prefixes across hops hit cache
(the reuse the multi-hop driver relies on); correctness never depends on it
(gate G5 proves APC-on == APC-off).

Finish-reason resolution (see ``vllm/v1/core/sched/utils.py:check_stop``):
- ``<eos>`` (id 2, the packaged ``eos_token_id``) stops with ``stop_reason
  is None``  -> ``"eos"``.
- ``<eot>`` (id 3, only in ``stop_token_ids``) stops with ``stop_reason == 3``
  -> ``"eot"``.
- otherwise ``finish_reason == "length"`` -> ``"length"``.

The terminator id is returned separately and stripped from ``token_ids`` if
vLLM left it in the output, so callers always get pure content ids.
"""

from __future__ import annotations

import math

import numpy as np

from ..vllm_client import constants as C
from ..vllm_client.protocol import GenerateRequest, GenerateResult

_LOGITS_PROCESSOR = (
    "espnet2.speechlm.vllm.vllm_server.logits:SpeechLMConstraintLogitsProcessor"
)


def yarn_hf_overrides(
    model_dir: str,
    factor: float | None,
    original_max_position_embeddings: int | None = None,
) -> dict:
    """Build a non-mutating Qwen3.5 M-RoPE YaRN config override.

    vLLM's generic ``hf_overrides`` replaces dict-valued config fields rather
    than merging them.  Preserve every checkpoint-specific M-RoPE setting and
    change only the scaling fields required by YaRN.  Returning an empty dict
    when ``factor`` is unset keeps the historical engine path unchanged.
    """
    if factor is None:
        return {}
    if not math.isfinite(factor) or factor <= 1.0:
        raise ValueError(f"yarn_factor must be greater than 1, got {factor}")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    rope_parameters = dict(getattr(text_config, "rope_parameters", None) or {})
    original = (
        original_max_position_embeddings
        if original_max_position_embeddings is not None
        else getattr(text_config, "max_position_embeddings", None)
    )
    if not isinstance(original, int) or original < 1:
        raise ValueError(
            "YaRN requires a positive original_max_position_embeddings; "
            "pass yarn_original_max_position_embeddings explicitly"
        )
    rope_parameters.update(
        rope_type="yarn",
        factor=float(factor),
        original_max_position_embeddings=original,
    )
    return {"text_config": {"rope_parameters": rope_parameters}}


def validate_yarn_max_model_len(
    max_model_len: int,
    hf_overrides: dict,
) -> None:
    """Fail early when the requested context exceeds YaRN's scaled range."""
    if not hf_overrides:
        return
    rope = hf_overrides["text_config"]["rope_parameters"]
    scaled_max = int(rope["original_max_position_embeddings"] * rope["factor"])
    if max_model_len > scaled_max:
        raise ValueError(
            f"max_model_len ({max_model_len}) exceeds the YaRN-scaled maximum "
            f"({scaled_max} = {rope['original_max_position_embeddings']} * "
            f"{rope['factor']})"
        )


class LocalEngine:
    """Thin, stateless wrapper over ``vllm.LLM`` for SpeechLM generation."""

    def __init__(
        self,
        model_dir: str,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 16384,
        max_num_seqs: int = 64,
        max_audio_per_prompt: int = 64,
        enable_prefix_caching: bool = True,
        enforce_eager: bool = False,
        dtype: str = "bfloat16",
        seed: int = 0,
        max_num_batched_tokens: int | None = None,
        mamba_ssm_cache_dtype: str = "auto",
        yarn_factor: float | None = None,
        yarn_original_max_position_embeddings: int | None = None,
        enable_expert_parallel: bool = False,
        moe_backend: str = "auto",
    ):
        from vllm import LLM

        self.model_dir = model_dir
        self.enable_prefix_caching = enable_prefix_caching
        # mamba_ssm_cache_dtype: "auto" resolves to bf16 for our custom arch
        # (vLLM's Qwen3.5 fp32-promotion hook keys on the architecture name and
        # never fires for SpeechLMForConditionalGeneration); float32 matches
        # training. See async_engine.py for the full story.
        extra: dict = {}
        if max_num_batched_tokens is not None:
            extra["max_num_batched_tokens"] = max_num_batched_tokens
        hf_overrides = yarn_hf_overrides(
            model_dir, yarn_factor, yarn_original_max_position_embeddings
        )
        validate_yarn_max_model_len(max_model_len, hf_overrides)
        if hf_overrides:
            extra["hf_overrides"] = hf_overrides
        self.llm = LLM(
            model=model_dir,
            tokenizer=model_dir,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            enable_prefix_caching=enable_prefix_caching,
            enable_chunked_prefill=True,
            limit_mm_per_prompt={"audio": max_audio_per_prompt},
            max_logprobs=32,
            enforce_eager=enforce_eager,
            logits_processors=[_LOGITS_PROCESSOR],
            trust_remote_code=True,
            seed=seed,
            mamba_ssm_cache_dtype=mamba_ssm_cache_dtype,
            enable_expert_parallel=enable_expert_parallel,
            moe_backend=moe_backend,
            **extra,
        )

    def _sampling_params(self, req: GenerateRequest):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=req.temperature,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            max_tokens=req.max_tokens,
            min_tokens=req.min_tokens,
            stop_token_ids=[C.EOS, C.EOT],
            detokenize=False,
            extra_args={"speechlm_mode": req.mode},
        )

    @staticmethod
    def _to_prompt(req: GenerateRequest) -> dict:
        prompt: dict = {"prompt_token_ids": list(req.prompt_token_ids)}
        if req.audio:
            prompt["multi_modal_data"] = {
                "audio": [(np.asarray(w, dtype=np.float32), sr) for w, sr in req.audio]
            }
            if req.audio_uuids is not None:
                if len(req.audio_uuids) != len(req.audio):
                    raise ValueError("audio_uuids length must match audio length")
                prompt["multi_modal_uuids"] = {"audio": list(req.audio_uuids)}
        return prompt

    @staticmethod
    def _resolve(out) -> GenerateResult:
        comp = out.outputs[0]
        token_ids = list(comp.token_ids)
        if comp.finish_reason == "length":
            return GenerateResult(token_ids, None, "length")
        # finished by a token stop. stop_reason == 3 -> <eot>; None -> <eos>.
        terminator = C.EOT if comp.stop_reason == C.EOT else C.EOS
        # vLLM may leave the stopping token in token_ids; strip one trailing
        # terminator so callers always receive pure content.
        if token_ids and token_ids[-1] in (C.EOS, C.EOT):
            terminator = token_ids[-1]
            token_ids = token_ids[:-1]
        reason = "eot" if terminator == C.EOT else "eos"
        return GenerateResult(token_ids, terminator, reason)

    def generate(self, requests: list[GenerateRequest]) -> list[GenerateResult]:
        """Run a batch of independent generation calls."""
        if not requests:
            return []
        prompts = [self._to_prompt(r) for r in requests]
        sps = [self._sampling_params(r) for r in requests]
        outs = self.llm.generate(prompts, sps)
        return [self._resolve(o) for o in outs]

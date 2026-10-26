"""SpeechLM captioner vLLM server plugin (Qwen3.5 dense + MoE backbones).

This package is the GPU/server side of the inference stack. It requires vLLM.
It registers an ``nn.Module`` wrapper
(:class:`espnet2.speechlm.vllm.vllm_server.model.SpeechLMForConditionalGeneration`)
that pairs the Qwen3-Omni audio tower + adaptor with a Qwen3.5 hybrid text
backbone, plus the multimodal processor and the constrained-decoding logits
processor. It imports the wire protocol and token/prompt logic from the pure-CPU
client (:mod:`espnet2.speechlm.vllm.vllm_client`); the dependency is one-way.

- :mod:`.config`       — HF-config accessors (no vLLM import)
- :mod:`.multimodal`   — mel extraction + placeholder-span expansion
- :mod:`.model`        — the model wrapper (registered arch)
- :mod:`.logits`       — content/choice constrained decoding
- :mod:`.plugin`       — ``register()`` general-plugin entry point
- :mod:`.engine`       — in-process ``LocalEngine`` (verl-facing surface)
- :mod:`.async_engine` — AsyncLLM wrapper for the FastAPI service
- :mod:`.audio_registry` — content-addressed native-SR audio cache
- :mod:`.service`      — stateless FastAPI front end
- :mod:`.eval_runner`  — offline sharded eval driver
- :mod:`.service_eval_client` — multiprocessing HTTP eval client (pure CPU, no vLLM import)
- :mod:`.export_hf`    — DCP checkpoint -> HF model dir export tool
"""

__version__ = "0.2.0"

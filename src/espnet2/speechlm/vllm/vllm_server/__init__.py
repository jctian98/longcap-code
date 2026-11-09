"""LongCap inference server: the vLLM model, multimodal plugin, and HTTP service.

This package requires vLLM. It registers the LongCap model
(:class:`espnet2.speechlm.vllm.vllm_server.model.SpeechLMForConditionalGeneration`),
which pairs the Qwen3-Omni audio encoder and adaptor with a Qwen3.5 language model,
together with its multimodal processor and decoding logits processor. Token and
prompt handling come from the CPU-only client package
(:mod:`espnet2.speechlm.vllm.vllm_client`); the dependency is one-way.

- :mod:`.config`         — model-config accessors (no vLLM import)
- :mod:`.multimodal`     — audio features and placeholder expansion
- :mod:`.model`          — the registered model
- :mod:`.logits`         — constrained decoding
- :mod:`.plugin`         — vLLM general-plugin entry point
- :mod:`.engine`         — in-process engine
- :mod:`.async_engine`   — AsyncLLM wrapper used by the service
- :mod:`.audio_registry` — content-addressed audio cache
- :mod:`.service`        — FastAPI server
"""

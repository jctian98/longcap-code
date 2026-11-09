"""vLLM general-plugin entry point for the SpeechLM captioner.

Registered in this package's ``pyproject.toml``:

    [project.entry-points."vllm.general_plugins"]
    longcap_speechlm_captioner = "espnet2.speechlm.vllm.vllm_server.plugin:register"

Why an installed entry point (and not a bare ``register()`` call in the parent
process): the ``vllm.general_plugins`` group is the *only* mechanism vLLM
guarantees to run inside the EngineCore process (and every worker) that actually
builds the model — it is invoked in ``EngineCore.__init__`` and
``WorkerBase.init_worker``. ``ModelRegistry`` is a per-process in-memory dict, so
a registration done only in the parent is inherited *only* under ``fork`` and is
silently lost under ``spawn`` (which vLLM auto-selects the moment CUDA is
initialized in the parent). Keeping this entry point installed makes
registration robust regardless of the multiprocessing start method.

We register the architecture in *lazy string form* so importing this module (and
thus running the plugin) does NOT import torch/vLLM model code until the
architecture is actually instantiated — keeping plugin discovery cheap and
avoiding import-order hazards. ``register()`` is idempotent, so calling it
explicitly (e.g. from a smoke test, before ``LLM(...)`` under fork) is harmless.

The constrained-decoding logits processor is NOT registered here; it is passed
per-engine via the ``logits_processors`` engine argument
(``"espnet2.speechlm.vllm.vllm_server.logits:SpeechLMConstraintLogitsProcessor"``),
which is the documented public surface for custom logits processors.
"""

from __future__ import annotations

ARCH_NAME = "SpeechLMForConditionalGeneration"
_MODEL_PATH = (
    "espnet2.speechlm.vllm.vllm_server.model:SpeechLMForConditionalGeneration"
)


def register() -> None:
    from vllm import ModelRegistry

    if ARCH_NAME not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCH_NAME, _MODEL_PATH)

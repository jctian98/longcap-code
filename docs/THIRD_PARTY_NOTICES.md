# Third-party components

LongCap's source is provided under Apache-2.0. Installed dependencies retain their upstream licenses; the dependency lock records exact versions and hashes.

| Component | Upstream | License |
| --- | --- | --- |
| ESPnet (the `espnet2.speechlm` modules are derived from it) | https://github.com/espnet/espnet | Apache-2.0 |
| vLLM | https://github.com/vllm-project/vllm | Apache-2.0 |
| Transformers/tokenizers | https://github.com/huggingface/transformers and https://github.com/huggingface/tokenizers | Apache-2.0 |
| PyTorch/torchaudio | https://github.com/pytorch/pytorch and https://github.com/pytorch/audio | BSD-style licenses; see installed distributions |
| Qwen3.5 text backbone | https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base | Apache-2.0 |
| Qwen3-Omni audio encoder | https://github.com/QwenLM/Qwen3-Omni | Apache-2.0 |

Model weights are distributed separately with their own LICENSE, NOTICE and upstream license copies. The benchmark dataset is released under Apache-2.0 with its own LICENSE; source audio and video are not part of the release. NVIDIA CUDA libraries and official container base images retain their supplier terms; no CUDA license is replaced by the LongCap source license.

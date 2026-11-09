<h1 align="center">LongCap</h1>

<p align="center"><b>Hour-Scale Timestamped Audio Captioning via Chunk-wise Processing</b></p>

<p align="center">
Jinchuan Tian, Sen Chen, Wenbin Huang, Chao Weng, Zhengwei Huang, Jiatong Shi, Yichen Lu, Haoran Wang, Shinji Watanabe, and Cong Zhou
</p>

<p align="center">
Paper (arXiv, coming soon) · <a href="https://jctian98.github.io/longcap/">Project page</a> · <a href="docs/quickstart.md">Quickstart</a> · <a href="https://huggingface.co/JinchuanTian/LongCap-35B-A3B-FP8">Model</a> · <a href="https://huggingface.co/datasets/JinchuanTian/LongCapBench-TAC-Long">Benchmark</a>
</p>

<p align="center">
<a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
<img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-blue">
</p>

**LongCap** describes what happens in a recording, from a short clip to hours of audio, as a timeline of events. This repository contains its inference code: a vLLM server, a CPU client, and a Docker image.

## What LongCap does

<p align="center"><img src="docs/assets/tac_format.png" width="100%" alt="LongCap output: a timeline of speech, music, and sound events"></p>

- **Timestamped audio captioning.** LongCap finds every speech, music, and sound event in a recording, gives its start and end times, and describes it with structured attributes, such as the transcript and speaker traits for speech, the instruments and mood for music, or the source and setting for a sound, plus a short caption. Overlapping events are described separately.
- **Speech recognition with speaker diarization.** LongCap transcribes speech and labels who speaks when.
- **Long recordings.** Both tasks work on short clips and on hours-long recordings such as full films and shows.

Attribute names and values are written in Chinese, while transcripts keep the spoken language; [`attribute_keys.json`](https://huggingface.co/JinchuanTian/LongCap-35B-A3B-FP8/blob/main/attribute_keys.json) gives the English name of every key.

## How to use it

LongCap runs as a GPU server in Docker and a CPU client. Follow the [Quickstart](docs/quickstart.md) to download a checkpoint and start the server, then run either task:

```bash
# Timestamped audio captioning
longcap caption --audio recording.wav --key my_recording \
  --model ./models/LongCap-35B-A3B-FP8 --service http://127.0.0.1:20000 \
  --output ./outputs/caption

# Speech recognition with speaker diarization
longcap diarize --audio recording.wav --key my_recording \
  --model ./models/LongCap-35B-A3B-FP8 --service http://127.0.0.1:20000 \
  --output ./outputs/diarization
```

Captioning writes timestamped events with their attributes to `outputs/caption/predictions/predictions.jsonl`; diarization writes timestamped segments with speaker labels and text to `outputs/diarization/predictions/diarization.jsonl`. The [inference guide](docs/inference.md) covers batches, resume, and output files, and [Docker](docs/DOCKER.md) covers container details.

| Release | Description |
| --- | --- |
| [LongCap-35B-A3B-FP8](https://huggingface.co/JinchuanTian/LongCap-35B-A3B-FP8) | Model weights in FP8; recommended, runs on a single GPU |
| [LongCap-35B-A3B](https://huggingface.co/JinchuanTian/LongCap-35B-A3B) | Model weights in BF16 |
| [LongCapBench-TAC-Long](https://huggingface.co/datasets/JinchuanTian/LongCapBench-TAC-Long) | Human-annotated long-form benchmark |

## Disclaimer

LongCap is a research release, intended for research use. It is not a product or a hosted service and has not been validated for production use. Like other AI models, it can make mistakes: it may miss or invent events, give wrong timestamps, mistranscribe speech, or mislabel speakers. Attributes such as a speaker's gender, age, or emotion are inferred from audio and may be wrong or biased. Review outputs before relying on them, and do not use them to identify people or to make decisions about them. You are responsible for how you use the model and its outputs, including having the right to process the audio you provide. The software and weights are provided "as is", without warranty of any kind, under the Apache License 2.0.

## Citation

```bibtex
@misc{tian2026longcap,
  title = {LongCap: Hour-Scale Timestamped Audio Captioning via Chunk-wise Processing},
  author = {Tian, Jinchuan and Chen, Sen and Huang, Wenbin and Weng, Chao and Huang, Zhengwei and Shi, Jiatong and Lu, Yichen and Wang, Haoran and Watanabe, Shinji and Zhou, Cong},
  year = {2026},
  howpublished = {Preprint},
  url = {https://jctian98.github.io/longcap/}
}
```

## Contact

Jinchuan Tian and Shinji Watanabe, Carnegie Mellon University: [jinchuat@andrew.cmu.edu](mailto:jinchuat@andrew.cmu.edu), [swatanab@andrew.cmu.edu](mailto:swatanab@andrew.cmu.edu). For bugs and usage questions, please open a [GitHub issue](https://github.com/jctian98/longcap-code/issues).

## Acknowledgments and license

We thank Xinying Hu, Shuangshuang He, and Zhuoqun Liu for their expert annotation work, and Anuttacon for its support of this work. The inference code is built on [ESPnet](https://github.com/espnet/espnet) and [vLLM](https://github.com/vllm-project/vllm). The code, model weights, and benchmark are released under the [Apache License 2.0](LICENSE); see [third-party notices](docs/THIRD_PARTY_NOTICES.md).

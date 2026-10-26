# LongCap: Hour-Scale Timestamped Audio Captioning via Chunk-wise Processing

**Jinchuan Tian, Sen Chen, Wenbin Huang, Chao Weng, Zhengwei Huang, Jiatong Shi, Yichen Lu, Haoran Wang, Shinji Watanabe, and Cong Zhou**

[Project page](https://jctian98.github.io/longcap/) · [Paper and release assets](https://github.com/jctian98/longcap-code/releases) · [Model](https://huggingface.co/JinchuanTian/LongCap-35B-A3B) · [Long benchmark](https://huggingface.co/datasets/JinchuanTian/LongCap-Long)

Official inference code for **LongCap**, a framework for describing audio events with timestamps across recordings from seconds to hours. LongCap interleaves audio chunks and structured text, retaining earlier context while expressing event times relative to each chunk.

The released **LongCap-35B-A3B** checkpoint supports two tasks with the same model and vLLM server:

| Task | Output | Command |
| --- | --- | --- |
| Timestamped audio captioning | Speech, music, and environmental events with timestamps and descriptive attributes | `longcap caption` |
| Speech recognition and diarization | Transcripts with speaker labels and timestamps | `longcap diarize` |

This repository includes a CPU client, the custom vLLM model and server, resumable inference, and Docker support. The checkpoint is the paper's 35B-A3B V4 model at stage-2 step 3000, before flywheel training. Native event-ancestor prediction is not supported.

## Results

Captioning results for the released checkpoint:

| Evaluation | TER ↓ | GER ↓ | tcpCapSim ↑ |
| --- | ---: | ---: | ---: |
| Short, 1,500 clips | 13.01 | 15.93 | 64.59 |
| Long, 4 recordings | 44.84 | 11.46 | 50.85 |
| Long, 45 ten-minute windows | 41.30 | 10.24 | 50.87 |

Values are percentages from the saved evaluation, using `speech_caption_insensitive_overlap_v2` conversion. All examples are included, including partial/truncated outputs in three full recordings and two windows, and two empty short outputs. These are historical checkpoint results; they are not scores from a new container run. The [model card](https://huggingface.co/JinchuanTian/LongCap-35B-A3B) also reports the checkpoint's speech recognition and diarization results, with full values in its `evaluation.json`.

The [long benchmark](https://huggingface.co/datasets/JinchuanTian/LongCap-Long) contains four recordings with 6,520 reference events and the saved model predictions. It distributes source URLs and annotations. Audio and video are not bundled. See [reproducibility and limitations](VALIDATION.md) for inference checks and long-recording coverage.

## Getting started

Use Python 3.12 on Linux. The client runs on CPU; the server targets one H100 80GB with CUDA 13.0 and tensor parallel size 1.

```bash
git clone https://github.com/jctian98/longcap-code.git
cd longcap-code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements/client.lock
python -m pip install --no-deps .

hf download JinchuanTian/LongCap-35B-A3B --local-dir ./models/LongCap-35B-A3B
longcap verify-model --model ./models/LongCap-35B-A3B
```

The model occupies approximately 38.43 GB across 21 weight shards. Download the complete directory, including `SHA256SUMS`. Use `hf auth login` if repository access requires authentication. For reproducible downloads, select the model revision in the release manifest with `hf download --revision`.

### Start the server

Install Docker and NVIDIA Container Toolkit, then launch the inference image:

```bash
mkdir -p cache
docker run --rm --gpus '"device=0"' --shm-size=16g \
  -p 127.0.0.1:20000:20000 \
  -v "$PWD/models/LongCap-35B-A3B:/models/longcap-35b-a3b:ro" \
  -v "$PWD/cache:/cache" \
  ghcr.io/jctian98/longcap:0.1.0-rc.2
```

Use `docker login ghcr.io` if package access requires authentication. Wait for `http://127.0.0.1:20000/health` to return success. [Docker details](docs/DOCKER.md) cover driver requirements, running the client in a container, and building the image locally.

### Audio captioning

```bash
longcap caption --audio /absolute/path/to/example.wav --key example \
  --model ./models/LongCap-35B-A3B \
  --service http://127.0.0.1:20000 --output ./outputs/captions
```

### Speech recognition and diarization

```bash
longcap diarize --audio /absolute/path/to/example.wav --key example \
  --model ./models/LongCap-35B-A3B \
  --service http://127.0.0.1:20000 --output ./outputs/diarization
```

Both tasks use the same server and model. Use separate output directories for each task. The diarization prompt requests transcripts, speaker labels, and timestamps; it does not infer real-world speaker identities.

See [inference and output formats](docs/inference.md) for batch manifests, task-specific outputs, decoding, and resume. Repeat an identical command to resume its saved request journal. A changed input, task, model, implementation, or decoding configuration requires a fresh output directory. Inspect `raw/client.json` for coverage and errors; exit status 2 indicates incomplete output or exhausted retries.

## Citation

If you use LongCap, please cite:

```bibtex
@misc{tian2026longcap,
  title = {LongCap: Hour-Scale Timestamped Audio Captioning via Chunk-wise Processing},
  author = {Tian, Jinchuan and Chen, Sen and Huang, Wenbin and Weng, Chao and Huang, Zhengwei and Shi, Jiatong and Lu, Yichen and Wang, Haoran and Watanabe, Shinji and Zhou, Cong},
  year = {2026},
  howpublished = {Preprint},
  url = {https://jctian98.github.io/longcap/}
}
```

## License and contributions

Code is licensed under [Apache-2.0](LICENSE); see [third-party notices](docs/THIRD_PARTY_NOTICES.md). The model and benchmark include their own license files. Linked source media and the paper retain separate terms. See [CONTRIBUTING.md](CONTRIBUTING.md) for development and tests.

# LongCap: Hour-Scale Timestamped Audio Captioning via Chunk-wise Processing

**Jinchuan Tian, Sen Chen, Wenbin Huang, Chao Weng, Zhengwei Huang, Jiatong Shi, Yichen Lu, Haoran Wang, Shinji Watanabe, and Cong Zhou**

[Project page](https://jctian98.github.io/longcap/) · [Paper and release assets](https://github.com/jctian98/longcap-code/releases) · [Model, FP8](https://huggingface.co/JinchuanTian/LongCap-35B-A3B-FP8) · [Model, BF16](https://huggingface.co/JinchuanTian/LongCap-35B-A3B) · [Long benchmark](https://huggingface.co/datasets/JinchuanTian/LongCap-Long)

Official inference code for **LongCap**, a framework for describing audio events with timestamps across recordings from seconds to hours. LongCap interleaves audio chunks and structured text, retaining earlier context while expressing event times relative to each chunk.

The released **LongCap-35B-A3B** checkpoint comes in FP8 and BF16 and supports two tasks with the same model and vLLM server:

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

Values are percentages from the saved evaluation, using `speech_caption_insensitive_overlap_v2` conversion. All examples are included, including partial/truncated outputs in three full recordings and two windows, and two empty short outputs. These are historical results of the FP8 checkpoint; they are not scores from a new container run. Speech recognition and diarization results for the same checkpoint are in the model repositories' `evaluation.json`.

The [long benchmark](https://huggingface.co/datasets/JinchuanTian/LongCap-Long) contains four recordings with 6,520 reference events and the saved model predictions. It distributes source URLs and annotations. Audio and video are not bundled. See [reproducibility and limitations](VALIDATION.md) for inference checks and long-recording coverage.

## Quickstart

LongCap runs as a GPU server (a custom vLLM build, shipped as a Docker image) and a CPU client that sends audio to it. You need Linux x86-64, Docker with the NVIDIA Container Toolkit, an NVIDIA driver compatible with CUDA 13.0, and Python 3.12 for the client.

### 1. Choose a checkpoint

| Checkpoint | Precision | Download | GPUs |
| --- | --- | ---: | --- |
| [LongCap-35B-A3B-FP8](https://huggingface.co/JinchuanTian/LongCap-35B-A3B-FP8) | FP8 MoE experts, BF16 elsewhere | 38.4 GB | 1x H100 80GB |
| [LongCap-35B-A3B](https://huggingface.co/JinchuanTian/LongCap-35B-A3B) | BF16 | 70.6 GB | 2x H100 80GB |

All reported scores use FP8, and the two precisions produce nearly identical output ([validation](VALIDATION.md#bf16-checkpoint)). BF16 does not fit on one 80 GB GPU: its weights take 65.9 GiB, leaving too little memory for the KV cache.

### 2. Install the client and download the model

```bash
git clone https://github.com/jctian98/longcap-code.git
cd longcap-code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements/client.lock
python -m pip install --no-deps .

MODEL=LongCap-35B-A3B-FP8   # or LongCap-35B-A3B for BF16
hf download JinchuanTian/$MODEL --local-dir ./models/$MODEL
longcap verify-model --model ./models/$MODEL
```

`verify-model` hashes every runtime file listed in the model's `SHA256SUMS`; keep that file with the download. Use `hf auth login` if repository access requires authentication.

### 3. Start the server

**FP8, one GPU:**

```bash
mkdir -p cache
docker run --rm --gpus '"device=0"' --shm-size=16g -p 127.0.0.1:20000:20000 \
  -v "$PWD/models/$MODEL:/models/longcap-35b-a3b:ro" \
  -v "$PWD/cache:/cache" \
  ghcr.io/jctian98/longcap:0.1.0-rc.2
```

**BF16, two GPUs.** `longcap serve` runs on one GPU, so this starts the server module directly with the same `long` profile settings plus `--tensor-parallel-size 2`:

```bash
mkdir -p cache
docker run --rm --gpus '"device=0,1"' --shm-size=16g -p 127.0.0.1:20000:20000 \
  -v "$PWD/models/$MODEL:/models/longcap-35b-a3b:ro" \
  -v "$PWD/cache:/cache" \
  --entrypoint python ghcr.io/jctian98/longcap:0.1.0-rc.2 \
  -m espnet2.speechlm.vllm.vllm_server.service \
  --model /models/longcap-35b-a3b --host 0.0.0.0 --port 20000 \
  --spool-dir /cache/audio-spool --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 --max-model-len 1048576 \
  --yarn-factor 4 --yarn-original-max-position-embeddings 262144 \
  --max-num-seqs 4 --max-num-batched-tokens 16384 --mamba-ssm-cache-dtype float32 \
  --slice-cache-max 2048 --io-threads 4 --torch-num-threads 4 \
  --mm-processor-cache-gb 4 --max-audio-per-prompt 1024
```

Loading the weights and compiling kernels takes a few minutes; the server is ready when `http://127.0.0.1:20000/health` returns success. Use `docker login ghcr.io` if package access requires authentication. [Docker details](docs/DOCKER.md) cover driver requirements, running the client in a container, and building the image locally.

### 4. Caption or diarize a recording

```bash
# Timestamped audio captioning
longcap caption --audio /absolute/path/to/example.wav --key example \
  --model ./models/$MODEL --service http://127.0.0.1:20000 \
  --output ./outputs/caption

# Speech recognition with speaker diarization
longcap diarize --audio /absolute/path/to/example.wav --key example \
  --model ./models/$MODEL --service http://127.0.0.1:20000 \
  --output ./outputs/diarization
```

Captions are written to `outputs/caption/predictions/predictions.jsonl`, one line per recording, with a list of events (`start`, `end`, `attributes`). Diarization is written to `outputs/diarization/predictions/diarization.jsonl`, with segments (`start`, `end`, `speaker`, `text`). Times are seconds from the start of the recording. Speaker labels distinguish voices within a recording; they do not identify real people. Attribute names and descriptive values are in Chinese, and the model repository's `attribute_keys.json` gives their English names.

Use `--manifest` instead of `--audio` for a batch of recordings, and separate output directories for each task. Rerun an identical command to resume an interrupted run; a changed input, task, model, implementation, or decoding configuration requires a fresh output directory. Inspect `raw/client.json` for coverage and errors; exit status 2 indicates incomplete output or exhausted retries. See [inference and output formats](docs/inference.md) for details.

### Serving profiles

| Profile | Context | Use for |
| --- | ---: | --- |
| `long` (image default) | 1,048,576 tokens with YaRN | Recordings longer than a few minutes, and diarization |
| `short` | 131,072 tokens | Captioning short clips, with higher batch concurrency |

Use the `long` profile for diarization. In our release checks, diarization with the `short` profile stopped transcribing after the first minute of a five-minute excerpt with both precisions, while the `long` profile transcribed all of it.

To serve FP8 with the `short` profile, append `serve --model /models/longcap-35b-a3b --profile short --host 0.0.0.0 --port 20000` to the FP8 command. For BF16, change `--max-model-len` to `131072`, drop the two `--yarn` options, and set `--max-num-seqs 128 --slice-cache-max 16384 --io-threads 8 --mm-processor-cache-gb 8 --max-audio-per-prompt 64`.

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

# LongCap

LongCap produces timestamped audio captions for recordings from seconds to hours. This repository contains the Python client, custom vLLM model implementation, inference server, and Docker recipe for **LongCap-35B-A3B V4, stage-2 step 3000**: the checkpoint used for the paper's main captioning experiments before flywheel training. Native event-ancestor prediction is not supported by this checkpoint.

The [model](https://huggingface.co/JinchuanTian/LongCap-35B-A3B), [benchmark](https://huggingface.co/datasets/JinchuanTian/LongCap-Long), [project page](https://jctian98.github.io/longcap/), and paper are distributed separately. The benchmark and demo include original media URLs and our labels; they do not distribute audio or video. Model/data repositories require authorized access during private staging.

## Quick start

Use Python 3.12 on Linux. Install the CPU client in a virtual environment from this repository:

```bash
git clone https://github.com/jctian98/longcap-code.git
cd longcap-code
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --require-hashes -r requirements/client.lock
python -m pip install --no-deps .
hf auth login
hf download JinchuanTian/LongCap-35B-A3B --local-dir /absolute/path/to/LongCap-35B-A3B
longcap verify-model --model /absolute/path/to/LongCap-35B-A3B
```

The complete model directory, including `SHA256SUMS`, is required. Its 21 weight shards occupy approximately 38.43 GB. The client does not require PyTorch or a GPU. A client and server may mount the same verified model files at different paths.

Use `--revision` with the model commit recorded in the release manifest for reproducible downloads. Authentication is required while the repository is private.

Build and run the GPU server from the repository root:

```bash
docker build -f docker/Dockerfile -t longcap:0.1.0 .
docker run --rm --gpus '"device=0"' --shm-size=16g \
  -p 127.0.0.1:20000:20000 \
  -v /absolute/path/to/LongCap-35B-A3B:/models/longcap-35b-a3b:ro \
  -v /absolute/path/to/cache:/cache \
  longcap:0.1.0
```

The Docker recipe starts from an official PyTorch image pinned by digest, installs hash-locked dependencies in an isolated environment, and loads weights from the external model mount. The initial hardware target is one H100 80GB, CUDA 13.0, and tensor parallel size 1. Wait for `http://127.0.0.1:20000/health` to return success before captioning.

```bash
longcap caption --audio /absolute/path/to/example.wav --key example \
  --model /absolute/path/to/LongCap-35B-A3B \
  --service http://127.0.0.1:20000 --output ./outputs/example
```

Read timestamped events in `outputs/example/predictions/predictions.jsonl` and processing status in `outputs/example/raw/client.json`. Run the identical command again to resume from its durable request journal. A new input, model, server implementation, or decoding configuration requires a fresh output directory. Exit status 2 means partial/truncated output or exhausted retries; inspect the saved status before using predictions.

See [inference and output details](docs/inference.md) for batch manifests, chunking, profiles, retry behavior, and conversion. See [Docker setup](docs/DOCKER.md) for host requirements, model verification, client containers, and dependency-lock maintenance.

## Paper results and validation

Historical V4 results use `speech_caption_insensitive_overlap_v2` conversion. The release converter exactly reproduces all 1,549 saved prediction records: 1,500 short clips, four full recordings, and 45 ten-minute windows. This is conversion parity; historical scores are not measurements of a fresh release decode.

| Evaluation | TER ↓ | GER ↓ | tcpCapSim ↑ |
| --- | ---: | ---: | ---: |
| Short, 1,500 clips | 13.01 | 15.93 | 64.59 |
| Long, 4 recordings | 44.84 | 11.46 | 50.85 |
| Long, 45 windows | 41.30 | 10.24 | 50.87 |

Values are percentages from the saved scoring2-v5 evaluation. All inputs are included. Three full recordings and two windows contain partial/truncated output; two short clips have empty output. Consult the model's `evaluation.json` for exact scores and coverage. The newer optional `--future-start-guard` conversion policy changes predictions and must not be reported with these historical metrics.

The CPU test suite covers model and server identity, upload transport, input-label isolation, retry/resume behavior, partial-output reporting, and conversion. Run it with `python -m pip install '.[test]'` followed by `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`. Container-build and GPU validation status are recorded separately in `VALIDATION.md`.

## Citation and license

Please cite *LongCap: Hour-Scale Timestamped Audio Captioning via Chunk-wise Processing* by Jinchuan Tian and colleagues. Author details are in [CITATION.cff](CITATION.cff). An arXiv identifier will be added after submission.

Code is provided under [Apache-2.0](LICENSE). See [third-party notices](docs/THIRD_PARTY_NOTICES.md). Model weights and benchmark annotations carry their separately supplied licenses; this code license does not cover the paper or linked source media.

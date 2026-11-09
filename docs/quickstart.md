# Quickstart

LongCap runs as a GPU server and a client that sends audio to it. Install both in your own Python environment (recommended), or run the server from the Docker image. Both ways use the same model files and the same client commands.

## 1. Choose a checkpoint

| Checkpoint | Precision | Download | GPUs |
| --- | --- | ---: | --- |
| [LongCap-35B-A3B-FP8](https://huggingface.co/JinchuanTian/LongCap-35B-A3B-FP8) (recommended) | FP8 MoE experts, BF16 elsewhere | 38.4 GB | 1x H100 80GB |
| [LongCap-35B-A3B](https://huggingface.co/JinchuanTian/LongCap-35B-A3B) | BF16 | 70.6 GB | 2x H100 80GB |

The two precisions produce closely matching output ([validation](../VALIDATION.md)). BF16 does not fit on a single 80 GB GPU.

## 2. Install

### Option A: Python environment (recommended)

You need Linux x86-64, an NVIDIA driver compatible with CUDA 13.0, Python 3.12 with `venv` and its development headers, and a C/C++ compiler, which Triton uses to build GPU kernels on first use. On Ubuntu 24.04, `sudo apt-get install python3.12-venv python3.12-dev gcc g++` installs all of these.

```bash
git clone https://github.com/jctian98/longcap-code.git
cd longcap-code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: --require-hashes -r requirements/bootstrap.lock
python -m pip install --only-binary=:all: --require-hashes -r requirements/server.lock
python -m pip install --no-build-isolation --no-deps .
```

The lock files pin every package with its hash, including PyTorch 2.11 for CUDA 13.0, vLLM 0.26.0, and the CUDA 13.0 libraries and compiler, so no separate CUDA toolkit is needed. The download is several gigabytes.

### Option B: Docker

You need Linux x86-64, Docker with the NVIDIA Container Toolkit, and an NVIDIA driver compatible with CUDA 13.0. The image contains the server; install the client, which needs no GPU, in a Python 3.12 environment:

```bash
docker pull ghcr.io/jctian98/longcap:0.1.0-rc.4

git clone https://github.com/jctian98/longcap-code.git
cd longcap-code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: --require-hashes -r requirements/client.lock
python -m pip install --no-deps .
```

The client-only install also works on any machine that sends audio to a LongCap server elsewhere.

## 3. Download the model

```bash
MODEL=LongCap-35B-A3B-FP8   # or LongCap-35B-A3B for BF16
hf download JinchuanTian/$MODEL --local-dir ./models/$MODEL
longcap verify-model --model ./models/$MODEL
```

`verify-model` checks every runtime file listed in the model's `SHA256SUMS`.

## 4. Start the server

**Option A, Python environment:**

```bash
nohup longcap serve --model ./models/$MODEL > server.log 2>&1 &                          # FP8, one GPU
nohup longcap serve --model ./models/$MODEL --tensor-parallel-size 2 > server.log 2>&1 &  # BF16, two GPUs
```

Use `CUDA_VISIBLE_DEVICES` to choose the GPUs. Uploaded audio and server logs go to `service-work/`.

**Option B, Docker:**

```bash
mkdir -p cache
# FP8, one GPU
docker run -d --name longcap --gpus '"device=0"' --shm-size=16g -p 127.0.0.1:20000:20000 \
  -v "$PWD/models/$MODEL:/models/longcap-35b-a3b:ro" -v "$PWD/cache:/cache" \
  ghcr.io/jctian98/longcap:0.1.0-rc.4

# BF16, two GPUs
docker run -d --name longcap --gpus '"device=0,1"' --shm-size=16g -p 127.0.0.1:20000:20000 \
  -v "$PWD/models/$MODEL:/models/longcap-35b-a3b:ro" -v "$PWD/cache:/cache" \
  ghcr.io/jctian98/longcap:0.1.0-rc.4 \
  serve --model /models/longcap-35b-a3b --host 0.0.0.0 --port 20000 \
    --work-dir /cache/service --tensor-parallel-size 2
```

Loading takes a few minutes. Wait until the server answers:

```bash
until curl -sf http://127.0.0.1:20000/health > /dev/null; do sleep 10; done
```

Follow the server log with `tail -f server.log` (Option A) or `docker logs -f longcap` (Option B).

## 5. Caption a recording

```bash
longcap caption --audio /path/to/recording.wav \
  --model ./models/$MODEL --output ./outputs/caption
```

Captions are written to `outputs/caption/predictions/predictions.jsonl`, one line per recording. Each event has start and end times in seconds and its attributes:

```json
{"key": "recording", "duration": 96.0, "events": [{"start": 0.0, "end": 59.0, "attributes": {"type": "music", "caption": "..."}}]}
```

## 6. Diarize a recording

```bash
longcap diarize --audio /path/to/recording.wav \
  --model ./models/$MODEL --output ./outputs/diarization
```

Transcripts are written to `outputs/diarization/predictions/diarization.jsonl`, with each segment's times, speaker label, and text:

```json
{"key": "recording", "duration": 96.0, "segments": [{"start": 0.2, "end": 1.1, "speaker": "0", "text": "..."}]}
```

Speaker labels distinguish voices within a recording; they do not identify real people.

## Tips

- **Audio formats.** The client reads WAV, FLAC, and other formats supported by libsndfile. Convert video or MP3 first: `ffmpeg -i input.mp4 -vn -ac 1 recording.wav`.
- **Batches.** Use `--manifest` instead of `--audio` to process many recordings; see the [inference guide](inference.md#inputs).
- **Exit status.** 0 means every recording finished. 2 means some output is incomplete; `predictions/conversion.jsonl` says which. 1 means a request failed or the server was unreachable.
- **Resume.** Rerun the same command after an interruption or a server restart; finished work is kept. Use a new output directory for a different task, input, or model.
- **Short clips.** For many short clips, start the server with `--profile short`, which serves more requests at once, and pass `--concurrency` to the client. With Docker, give the full `serve ...` command as in the BF16 example. Keep the default `long` profile for diarization.
- **More.** See the [inference guide](inference.md) for all options and [Docker](DOCKER.md) for container details.

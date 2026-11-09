# Quickstart

LongCap runs as a GPU server, shipped as a Docker image, and a CPU client that sends audio to it. You need Linux x86-64, Docker with the NVIDIA Container Toolkit, an NVIDIA driver compatible with CUDA 13.0, and Python 3.12 for the client.

## 1. Choose a checkpoint

| Checkpoint | Precision | Download | GPUs |
| --- | --- | ---: | --- |
| [LongCap-35B-A3B-FP8](https://huggingface.co/JinchuanTian/LongCap-35B-A3B-FP8) (recommended) | FP8 MoE experts, BF16 elsewhere | 38.4 GB | 1x H100 80GB |
| [LongCap-35B-A3B](https://huggingface.co/JinchuanTian/LongCap-35B-A3B) | BF16 | 70.6 GB | 2x H100 80GB |

The two precisions produce closely matching output ([validation](../VALIDATION.md)). BF16 does not fit on a single 80 GB GPU.

## 2. Install the client and download the model

```bash
git clone https://github.com/jctian98/longcap-code.git
cd longcap-code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: --require-hashes -r requirements/client.lock
python -m pip install --no-deps .

MODEL=LongCap-35B-A3B-FP8   # or LongCap-35B-A3B for BF16
hf download JinchuanTian/$MODEL --local-dir ./models/$MODEL
longcap verify-model --model ./models/$MODEL
```

`verify-model` checks every file against the model's `SHA256SUMS`. Use `hf auth login` if repository access requires authentication.

## 3. Start the server

**FP8, one GPU:**

```bash
mkdir -p cache
docker run -d --name longcap --gpus '"device=0"' --shm-size=16g -p 127.0.0.1:20000:20000 \
  -v "$PWD/models/$MODEL:/models/longcap-35b-a3b:ro" -v "$PWD/cache:/cache" \
  ghcr.io/jctian98/longcap:0.1.0-rc.3
```

**BF16, two GPUs:** add the second GPU and split the model across both:

```bash
mkdir -p cache
docker run -d --name longcap --gpus '"device=0,1"' --shm-size=16g -p 127.0.0.1:20000:20000 \
  -v "$PWD/models/$MODEL:/models/longcap-35b-a3b:ro" -v "$PWD/cache:/cache" \
  ghcr.io/jctian98/longcap:0.1.0-rc.3 \
  serve --model /models/longcap-35b-a3b --host 0.0.0.0 --port 20000 \
    --work-dir /cache/service --tensor-parallel-size 2
```

Loading takes a few minutes. Wait until the server answers:

```bash
until curl -sf http://127.0.0.1:20000/health > /dev/null; do sleep 10; done
```

`docker logs -f longcap` shows its progress, and `docker stop longcap` stops it. Use `docker login ghcr.io` if the image requires authentication.

## 4. Caption a recording

```bash
longcap caption --audio /path/to/recording.wav \
  --model ./models/$MODEL --output ./outputs/caption
```

Captions are written to `outputs/caption/predictions/predictions.jsonl`, one line per recording. Each event has start and end times in seconds and its attributes:

```json
{"key": "recording", "duration": 96.0, "events": [{"start": 0.0, "end": 59.0, "attributes": {"type": "music", "caption": "..."}}]}
```

## 5. Diarize a recording

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
- **Short clips.** For many short clips, the `short` profile serves more requests at once: start the server with the `serve ...` arguments shown for BF16, adding `--profile short` (and dropping `--tensor-parallel-size 2` for FP8), then pass `--concurrency` to the client. Keep the default `long` profile for diarization.
- **More.** See the [inference guide](inference.md) for all options and [Docker](DOCKER.md) to run the client in a container or build the image.

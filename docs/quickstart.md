# Quickstart

LongCap runs as a GPU server (a custom vLLM build, shipped as a Docker image) and a CPU client that sends audio to it. You need Linux x86-64, Docker with the NVIDIA Container Toolkit, an NVIDIA driver compatible with CUDA 13.0, and Python 3.12 for the client.

## 1. Choose a checkpoint

| Checkpoint | Precision | Download | GPUs |
| --- | --- | ---: | --- |
| [LongCap-35B-A3B-FP8](https://huggingface.co/JinchuanTian/LongCap-35B-A3B-FP8) (recommended) | FP8 MoE experts, BF16 elsewhere | 38.4 GB | 1x H100 80GB |
| [LongCap-35B-A3B](https://huggingface.co/JinchuanTian/LongCap-35B-A3B) | BF16 | 70.6 GB | 2x H100 80GB |

The two precisions produce nearly identical output ([validation](../VALIDATION.md#bf16-checkpoint)). BF16 does not fit on a single 80 GB GPU.

## 2. Install the client and download the model

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

`verify-model` checks every file against the model's `SHA256SUMS`. Use `hf auth login` if repository access requires authentication.

## 3. Start the server

**FP8, one GPU:**

```bash
mkdir -p cache
docker run --rm --gpus '"device=0"' --shm-size=16g -p 127.0.0.1:20000:20000 \
  -v "$PWD/models/$MODEL:/models/longcap-35b-a3b:ro" \
  -v "$PWD/cache:/cache" \
  ghcr.io/jctian98/longcap:0.1.0-rc.2
```

**BF16, two GPUs.** `longcap serve` runs on one GPU, so this starts the server module directly with the same settings plus `--tensor-parallel-size 2`:

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

Loading takes a few minutes; the server is ready when `http://127.0.0.1:20000/health` returns success. Use `docker login ghcr.io` if the image requires authentication.

## 4. Caption a recording

```bash
longcap caption --audio /path/to/recording.wav --key my_recording \
  --model ./models/$MODEL --service http://127.0.0.1:20000 \
  --output ./outputs/caption
```

Captions are written to `outputs/caption/predictions/predictions.jsonl`, one line per recording. Each event has start and end times in seconds and its attributes:

```json
{"key": "my_recording", "duration": 96.0, "events": [{"start": 0.0, "end": 59.0, "attributes": {"type": "music", "caption": "..."}}]}
```

## 5. Diarize a recording

```bash
longcap diarize --audio /path/to/recording.wav --key my_recording \
  --model ./models/$MODEL --service http://127.0.0.1:20000 \
  --output ./outputs/diarization
```

Transcripts are written to `outputs/diarization/predictions/diarization.jsonl`, with each segment's times, speaker label, and text:

```json
{"key": "my_recording", "duration": 96.0, "segments": [{"start": 0.2, "end": 1.1, "speaker": "0", "text": "..."}]}
```

Speaker labels distinguish voices within a recording; they do not identify real people.

## Tips

- **Batches.** Use `--manifest` instead of `--audio` to process many recordings; see the [inference guide](inference.md).
- **Resume.** Rerun the same command to continue an interrupted run. Use a new output directory for a different task, input, or model.
- **Check the result.** `raw/client.json` in the output directory reports coverage and errors; exit status 2 means some output is incomplete.
- **Serving profiles.** The image serves the `long` profile, which handles long recordings and should be used for diarization. For short clips, the `short` profile allows more concurrent requests: for FP8, append `serve --model /models/longcap-35b-a3b --profile short --host 0.0.0.0 --port 20000` to the FP8 command; for BF16, change `--max-model-len` to `131072`, drop the two `--yarn` options, and set `--max-num-seqs 128 --slice-cache-max 16384 --io-threads 8 --mm-processor-cache-gb 8 --max-audio-per-prompt 64`.
- **Containers.** To run the client inside the image or build the image yourself, see [Docker](DOCKER.md).

# LongCap with Docker

The LongCap image provides the vLLM server and both inference commands: audio captioning and speech diarization with transcription. Model weights are downloaded separately and mounted read-only.

## Requirements

- Linux x86-64, Docker, and NVIDIA Container Toolkit.
- An NVIDIA driver compatible with CUDA 13.0. The prior H100 checks used driver 580.159.03.
- One H100 80GB at tensor parallel size 1 is the initial hardware target. Prior tests used an 8-CPU, 128-GB RAM host; minimum host memory was not measured.

The base is the official PyTorch image `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel`, pinned by digest in the [Dockerfile](../docker/Dockerfile). The environment uses Python 3.12, PyTorch/torchaudio 2.11.0, vLLM 0.26.0, and Transformers 5.14.1. The image contains hash-locked dependencies and the LongCap implementation; it does not include model weights, input audio, or credentials. See [validation and limitations](../VALIDATION.md) for the scope of GPU testing.

## Download the model and image

```bash
hf download JinchuanTian/LongCap-35B-A3B --local-dir ./models/LongCap-35B-A3B
docker pull ghcr.io/jctian98/longcap:0.1.0-rc.2
```

Use `hf auth login` and `docker login ghcr.io` when repository/package access requires authentication. Immutable model revisions and image digests are listed in the [release manifest](https://github.com/jctian98/longcap-code/releases).

Verify the complete model directory:

```bash
docker run --rm \
  -v "$PWD/models/LongCap-35B-A3B:/models/longcap-35b-a3b:ro" \
  ghcr.io/jctian98/longcap:0.1.0-rc.2 \
  verify-model --model /models/longcap-35b-a3b
```

## Start the server

```bash
mkdir -p cache
docker run --rm --gpus '"device=0"' --shm-size=16g \
  -p 127.0.0.1:20000:20000 \
  -v "$PWD/models/LongCap-35B-A3B:/models/longcap-35b-a3b:ro" \
  -v "$PWD/cache:/cache" \
  ghcr.io/jctian98/longcap:0.1.0-rc.2
```

Model loading and CUDA kernel compilation may take several minutes. Wait for `http://127.0.0.1:20000/health` to return success. The cache mount stores compiled kernels; inference needs no external model download.

The default service uses the long profile, with a configured 1M-token context. For shorter inputs, append `serve --model /models/longcap-35b-a3b --profile short --host 0.0.0.0 --port 20000`. Context size does not guarantee caption completeness or unlimited recording length. See [inference settings](inference.md).

## Run either task

The CPU client can run on the host as shown in the [README](../README.md), or in another container. Put an input recording at `inputs/example.wav` and create the output directory:

```bash
mkdir -p outputs
```

Audio captioning:

```bash
docker run --rm --network host \
  -v "$PWD/models/LongCap-35B-A3B:/models/longcap-35b-a3b:ro" \
  -v "$PWD/inputs:/inputs:ro" \
  -v "$PWD/outputs:/output" \
  ghcr.io/jctian98/longcap:0.1.0-rc.2 \
  caption --audio /inputs/example.wav --key example \
    --model /models/longcap-35b-a3b \
    --service http://127.0.0.1:20000 --output /output/caption
```

Speech diarization and transcription:

```bash
docker run --rm --network host \
  -v "$PWD/models/LongCap-35B-A3B:/models/longcap-35b-a3b:ro" \
  -v "$PWD/inputs:/inputs:ro" \
  -v "$PWD/outputs:/output" \
  ghcr.io/jctian98/longcap:0.1.0-rc.2 \
  diarize --audio /inputs/example.wav --key example \
    --model /models/longcap-35b-a3b \
    --service http://127.0.0.1:20000 --output /output/diarization
```

Both clients use the same server. `--network host` is for Linux. The client uploads audio chunks, so the server does not need access to the input mount. Use separate output directories for different tasks and inspect each run's `raw/client.json` for coverage and errors. The [inference guide](inference.md) describes outputs and resume behavior.

The service is intended for local or trusted deployment. Remote deployments need an authenticated gateway. Uploaded chunks remain in the server's work directory until removed after shutdown.

## Build from source

From the repository root:

```bash
docker build -f docker/Dockerfile -t longcap:local .
```

Use `longcap:local` in place of the registry image in the examples. The build installs dependencies from `requirements/bootstrap.lock` and `requirements/server.lock` into an isolated environment. Building the CUDA development image needs substantial temporary disk space; the repository's optional container workflow checks for at least 80 GiB on its runner.

# LongCap with Docker

The image `ghcr.io/jctian98/longcap:0.1.0-rc.4` is an alternative to installing LongCap in your own Python environment. It contains the server and the `longcap` command line with the same pinned dependencies. Model weights are downloaded separately and mounted read-only. The [quickstart](quickstart.md) covers both ways; this page covers container details.

## Requirements

- Linux x86-64, Docker, and the NVIDIA Container Toolkit.
- An NVIDIA driver compatible with CUDA 13.0.
- One H100 80GB for the FP8 checkpoint, or two for the BF16 checkpoint.

The image is built on the official PyTorch image `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel`, pinned by digest in the [Dockerfile](../docker/Dockerfile), with Python 3.12, vLLM 0.26.0, and hash-locked dependencies. It contains no model weights, audio, or credentials.

## Start the server

```bash
mkdir -p cache
docker run -d --name longcap --gpus '"device=0"' --shm-size=16g -p 127.0.0.1:20000:20000 \
  -v "$PWD/models/LongCap-35B-A3B-FP8:/models/longcap-35b-a3b:ro" -v "$PWD/cache:/cache" \
  ghcr.io/jctian98/longcap:0.1.0-rc.4
```

Without arguments, the image runs `longcap serve --model /models/longcap-35b-a3b --profile long --host 0.0.0.0 --port 20000 --work-dir /cache/service`. Arguments after the image name replace this whole command, so pass a complete `serve ...` command to change it. For the BF16 checkpoint on two GPUs, use `--gpus '"device=0,1"'` and `serve --model /models/longcap-35b-a3b --host 0.0.0.0 --port 20000 --work-dir /cache/service --tensor-parallel-size 2`, as in the [quickstart](quickstart.md#4-start-the-server).

The `/cache` mount keeps uploaded audio chunks, server logs, and compiled kernels, so later starts are faster. Uploaded chunks stay there until you delete them. Files in `cache/` are owned by root because the container runs as root.

The server has no authentication. Publish its port only on `127.0.0.1`, as above, or put an authenticated proxy in front of it.

## Run the client in a container

The client can also run from the image instead of a local install. Put recordings in `inputs/`:

```bash
mkdir -p inputs outputs
docker run --rm --network host \
  -v "$PWD/models/LongCap-35B-A3B-FP8:/models/longcap-35b-a3b:ro" \
  -v "$PWD/inputs:/inputs:ro" -v "$PWD/outputs:/output" \
  ghcr.io/jctian98/longcap:0.1.0-rc.4 \
  caption --audio /inputs/recording.wav \
    --model /models/longcap-35b-a3b --output /output/caption
```

Replace `caption` with `diarize` for speech recognition with speaker diarization. `--network host` lets the container reach the server on `127.0.0.1` and works on Linux. The client uploads audio to the server, so the server needs no access to `inputs/`.

## Build the image

From the repository root:

```bash
docker build -f docker/Dockerfile -t longcap:local .
```

Use `longcap:local` in place of the published image. The build needs about 80 GB of free disk space.

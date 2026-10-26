# Docker inference

Private release candidates use `ghcr.io/jctian98/longcap`. Authenticate with a GitHub account that has package access, then pull the candidate:

```bash
docker login ghcr.io
docker pull ghcr.io/jctian98/longcap:0.1.0-rc.1
docker tag ghcr.io/jctian98/longcap:0.1.0-rc.1 longcap:0.1.0
```

The local alias above works with the commands below. For reproducible runs, use the immutable image digest and model revision recorded in the release manifest. Candidate build and validation receipts are attached to the draft GitHub Release; the earlier scoped GPU evidence in `VALIDATION.md` does not qualify a new digest automatically.

To build locally instead, run this from the code repository root:

```bash
docker build -f docker/Dockerfile -t longcap:0.1.0 .
```

The Dockerfile pins the official PyTorch `2.11.0-cuda13.0-cudnn9-devel` image by digest and installs the complete hash-locked Python dependencies in `requirements/server.lock`. A hash-pinned pip bootstrap in `requirements/bootstrap.lock` creates the isolated environment without relying on the base image's absent `ensurepip` module. It targets Linux x86-64 with Python 3.12. Use an NVIDIA host driver compatible with CUDA 13.0 and NVIDIA Container Toolkit. The validated GPU setup is one H100 80GB at TP=1, driver 580.159.03, with an 8-CPU / 128-GB host allocation. This is the tested allocation, not a measured minimum host-memory requirement. Other GPU configurations require their own validation.

Download or copy the separately distributed model directory, including `SHA256SUMS`, then verify it:

```bash
docker run --rm \
  -v /absolute/path/to/model:/models/longcap-35b-a3b:ro \
  longcap:0.1.0 verify-model --model /models/longcap-35b-a3b
```

Start the full-recording service:

```bash
docker run --rm --gpus '"device=0"' --shm-size=16g \
  -p 127.0.0.1:20000:20000 \
  -v /absolute/path/to/model:/models/longcap-35b-a3b:ro \
  -v /absolute/path/to/cache:/cache \
  longcap:0.1.0
```

Model initialization and CUDA kernel compilation may take several minutes. Check `/health` before sending a request. To select the shorter context profile, append `serve --model /models/longcap-35b-a3b --profile short --host 0.0.0.0 --port 20000`.

Run the CPU client on the host, or in a separate container with no GPU:

```bash
docker run --rm --network host \
  -v /absolute/path/to/model:/models/longcap-35b-a3b:ro \
  -v /absolute/path/to/your-inputs:/inputs:ro \
  -v /absolute/path/to/output:/output \
  longcap:0.1.0 caption --audio /inputs/example.wav \
    --model /models/longcap-35b-a3b \
    --service http://127.0.0.1:20000 --output /output/run
```

`--network host` in the client example is for Linux. The client uploads chunks, so the server does not need the client input mount. Shared-path registration is disabled by default. The service is intended for local/trusted deployment; expose it through your own authenticated gateway for remote users.

Weights, audio and credentials are outside image layers. No external model download is required at inference time; the complete tokenizer and audio processor are in the model mount. The cache mount holds locally compiled kernels. Full-recording context uses the configured 1M-token/YaRN profile; context limits, output truncation and retries are still possible and are reported in results.

Rebuild the lock deliberately when changing dependencies:

```bash
uv pip compile requirements/server.in --python-version 3.12 \
  --python-platform x86_64-manylinux_2_28 --generate-hashes \
  --no-header --no-annotate --emit-index-url \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match -o requirements/server.lock
```

Consult the accompanying release validation report for actual build and GPU test status; the historical checkpoint smoke test is separate from validation of this container.

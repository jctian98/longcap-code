# Contributing to LongCap

Use Python 3.12 and open pull requests against `main`. Include a focused description of the changed behavior and the checks you ran. Keep checkpoint identity, saved prediction failures, and benchmark annotation structures explicit when changing inference or evaluation code.

## Run the CPU tests

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: --require-hashes -r requirements/client.lock
python -m pip install --no-deps .
python -m pip install pytest==8.1.1 fastapi==0.136.3 python-multipart==0.0.32
python -m pip check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -B -m pytest -q
```

FastAPI and multipart support let the existing ASGI upload and request tests run with a fake engine. The suite uses synthetic fixtures and requires neither model weights nor a GPU. CI runs these commands for pushes and pull requests to `main`; external pytest plugins are disabled so unrelated installed plugins cannot affect the run.

## Build and publish a container

The **Build and publish container** workflow runs only through `workflow_dispatch`. Select the source branch or tag, provide a version such as `0.1.0-rc.1`, and supply the runner labels as a JSON string. For example, use `["self-hosted", "linux", "x64", "longcap-build"]` for a dedicated runner, or `["your-large-hosted-runner-label"]` for a larger GitHub-hosted runner configured for this repository.

The runner must be Linux x64 with a working local Docker daemon and at least **80 GiB free** on both the workspace and Docker storage filesystems. Standard GitHub-hosted runners may not have enough free storage. The preflight checks available space before registry login or image building; the CUDA development base and installed dependencies need substantial temporary storage. Self-hosted runners also need a current GitHub Actions runner compatible with the pinned actions.

The workflow builds `docker/Dockerfile` for `linux/amd64` from the checked-out source and publishes `ghcr.io/jctian98/longcap:<version>` and `ghcr.io/jctian98/longcap:sha-<commit>`. It refuses to publish if either tag already exists or its absence cannot be confirmed. OCI labels record the code repository, source commit, version, and license. The resulting digest appears in the workflow summary and the `publish` job output; use that digest for reproducible deployments.

Publishing uses the repository's `GITHUB_TOKEN` with `contents: read` and `packages: write`. For an existing GHCR package, grant this repository Actions access in the package settings. The workflow does not change repository or package visibility. Ordinary commits and pull requests run tests only. Action references are pinned to verified official release commits; update each SHA and its version comment together after checking the upstream release.

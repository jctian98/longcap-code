# Contributing to LongCap

We welcome improvements to inference, documentation, and reproducibility. Open an issue for a bug or proposed change, or submit a focused pull request against `main`. Describe the behavior, include a small reproduction when useful, and report the checks you ran.

## Development setup

Use Python 3.12. The CPU tests use synthetic fixtures and do not require model weights or a GPU:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: --require-hashes -r requirements/client.lock
python -m pip install --no-deps -e .
python -m pip install pytest==8.1.1 fastapi==0.136.3 python-multipart==0.0.32
python -m pip check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -B -m pytest -q
```

CI runs the CPU suite on pushes and pull requests. GPU checks and known limitations are documented in [VALIDATION.md](VALIDATION.md).

## Inference and evaluation changes

Keep task prompts, checkpoint identity, timestamp conventions, and output schemas explicit. Preserve raw model replies and report incomplete outputs. When a parser or decoding policy changes predictions, identify the policy and evaluate it separately from the paper's saved results.

The dependency locks pin the supported client and server environments. Include corresponding lock updates and relevant validation when changing dependencies. Build the server locally with the command in [Docker inference](docs/DOCKER.md).

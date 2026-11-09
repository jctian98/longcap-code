---
name: Bug report
about: Report a problem with installation, serving, or inference
labels: bug
---

**What happened**
A clear description of the problem and what you expected instead.

**How to reproduce**
The exact commands you ran (server and client), with paths shortened if needed.

**Environment**
- Checkpoint: LongCap-35B-A3B-FP8 or LongCap-35B-A3B (BF16)
- Docker image tag or `pip show longcap` version
- GPU model and count, NVIDIA driver version
- Serving profile: `short` or `long`

**Logs**
The server log around the error, and `raw/client.json` from the output directory if the client ran.
Please remove anything private, such as file paths or transcripts, before posting.

# Validation and known limitations

This page records how this release was tested. It describes software checks; benchmark scores are reported on the model cards and in the paper.

## Tests

- **CPU test suite.** 96 tests, run with the commands in [CONTRIBUTING.md](CONTRIBUTING.md). They need no GPU or model weights and cover audio preparation, the HTTP server API with a stub engine, request journaling and resume, retries, output conversion, and both tasks.
- **Image build.** The published image `ghcr.io/jctian98/longcap:0.1.0-rc.4` installs the hash-locked dependencies, then runs `pip check`, the CLI, and a check that the vLLM plugin is registered.
- **GPU runs.** On NVIDIA H100 80GB GPUs, this release ran in both setups from the [quickstart](docs/quickstart.md), and the first three checks below passed in each:
  - **Python environment:** the quickstart's install commands in a new virtual environment on Ubuntu 24.04, with no CUDA toolkit installed.
  - **Docker:** an image built from the same base image and the previous release's dependency locks, which differ only in CUDA compiler packages that the image does not use.

| Setup | Server start | Checks |
| --- | --- | --- |
| FP8, one GPU, `longcap serve` | 2 to 3 minutes | 30 short clips captioned, a 10-minute film excerpt captioned, three recordings up to 5 minutes diarized; all complete |
| FP8, server killed mid-batch | | The interrupted run exited with status 1; after a restart, the same command finished all 30 clips and reused the 4 already done |
| BF16, two GPUs, `longcap serve --tensor-parallel-size 2` | About 2 minutes | The same captioning and diarization inputs; all complete |
| BF16, one GPU | Fails to start | The weights take 65.9 GiB, leaving too little memory for the KV cache |

## Python environment and Docker

Both setups run the same GPU kernels: `longcap serve` turns off the vLLM features that depend on whether a CUDA toolkit is installed, and in a Python environment it points `CUDA_HOME` at the CUDA 13.0 compiler installed from the lock ([details](docs/inference.md#server)). With either precision, their outputs are as close as two runs of the same setup: 99% of caption events on the 30 short clips pair one-to-one in time, the captions of the 10-minute excerpt are identical, and diarization transcripts are 98% to 100% similar.

## FP8 and BF16

With the same inputs and serving profile, the two precisions produce closely matching output. On 30 short clips, 95% to 98% of caption events pair one-to-one in time (median temporal overlap 1.0) and paired transcripts are 98% to 99% similar. Running the same precision twice, or switching serving profiles, varies the output by a similar amount, because batched GPU decoding is not bitwise deterministic. Diarization transcripts of three recordings are 80% to 100% similar across precisions; differences concentrate in hard passages such as overlapping speech over music.

## Known limitations

- **Diarization needs the `long` profile.** With `short`, diarization of a five-minute recording stopped after about a minute, with both precisions.
- **Long recordings can stop early.** In an earlier run over four recordings of 94 to 117 minutes, one film stopped after about 92 minutes because too many chunks needed retries. The client now exports the decoded part and marks it `partial`; `--max-retried-chunks 0` removes the limit.
- **Output is not bitwise reproducible** across batch sizes, concurrency, or hardware, as noted above.
- **Untested:** GPUs other than H100 80GB, tensor parallel sizes other than 1 and 2, Linux distributions other than Ubuntu 24.04 for the Python environment, and the published image's exact digest on a GPU.
- **Scores are not re-measured.** The scores on the model cards come from the evaluation pipeline used for the paper, before this release's client changes.

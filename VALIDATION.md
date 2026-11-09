# Reproducibility and known limitations

LongCap reports historical checkpoint evaluations separately from software and container checks. The captioning and diarization scores in the model card are saved results for the released V4 weights. The checks below establish specific implementation behavior; they do not establish semantic completeness or bitwise GPU generation reproducibility.

| Check | Result |
| --- | --- |
| CPU test suite | 88 passed, 0 skipped; includes ASGI uploads, generation dispatch, server lifecycle, model identity, retries/resume, caption conversion, and native diarization task routing/export |
| Saved V4 caption conversion parity | Exact agreement on 1,549 records: 1,500 Short, 4 full Long, and 45 Long-window records |
| Native diarization parser parity | Exact agreement with the original evaluator on 451 saved English/Chinese recordings, 9,789 turns, and 36,065 speech segments |
| Copied model | All 21 shards independently rehashed after copying; safetensor headers, index keys, tensor offsets, and payload coverage verified; full checksums also passed on both GPU workers |
| CPU package installation | Final wheel installed in a clean environment; CLI, dependency consistency, tokenizer round-trip, model identity, and stereo-audio preparation passed without installing or importing PyTorch/vLLM |
| Official-base container build | Passed isolated installation from hash-locked dependencies, dependency consistency, CLI startup, pinned versions, CUDA 13.0, plugin registration, and image export |
| Earlier image GPU smoke | Passed 16-second and 96-second inputs under the long profile and completed-record resume on one H100 80GB; request journal unchanged |
| Four full recordings | Full-coverage check failed: three recordings reached the end; the English movie stopped at the default retry limit after 277/333 chunks. Raw records were retained; final batch prediction export was withheld. |

## Container and GPU checks

Registry images use the pinned official PyTorch `2.11.0-cuda13.0-cudnn9-devel` base and hash-locked dependencies. The earlier H100-tested image used a checksummed source archive in place of the Dockerfile's local `COPY` operations; its remaining build instructions and source bytes were preserved. This build difference is why its GPU results are reported separately.

The earlier image tested on H100 has digest `sha256:244371f2678aa6bbd4257d56d69c0fc66e9f38014d44481267b4af35cb6a2a12`. Registry images are distributed at `ghcr.io/jctian98/longcap`; their source revisions, digests, and build checks accompany each GitHub Release. The source Dockerfile uses ordinary `COPY` instructions. The H100 results below belong to that earlier image and do not qualify later registry digests, including the client update that adds `longcap diarize`.

The actual-image GPU test used Python 3.12.3, PyTorch/torchaudio 2.11.0 with CUDA 13.0, vLLM 0.26.0, Transformers 5.14.1, one H100 80GB, and driver 580.159.03. The host allocation was 8 CPUs and 128 GB RAM; minimum host RAM was not measured. All 34 installed Python source files matched the captioning implementation at the time of that test. The later diarization client extension is covered by separate CPU tests; it has not been run on H100 in this release. Peak sampled GPU memory was 72,381 MiB, including cache reservations; this is not a minimum-memory claim. Other GPU types and tensor-parallel configurations are unverified.

The 16-second input produced no events (`empty`); the 96-second multi-chunk input produced one music event (`ok`). These exact predictions matched the preceding isolated-wheel smoke run. All test commands exited successfully and completed-record resume left its request journal unchanged. This establishes execution, export, and resume behavior, not semantic caption completeness. Interrupted-resume behavior is covered by CPU tests, not this GPU smoke. No registry release image has been qualified on all four full recordings.

## Full-duration and historical comparisons

The separate full-recording run used the same installed release Python source and core model-runtime versions, with an existing dependency environment and Python 3.12.13. Its transitive package versions differ from the freshly resolved container lock. These results are distinct from the actual-image smoke and do not establish exact container decode parity.

The four-recording stage finished in 179.1 minutes with concurrency 4 on one H100 80GB. Peak sampled GPU memory was 72,747 MiB. All four raw result records were retained, with no request errors. Full recording coverage failed because the English movie reached the default limit of 20 distinct retried chunks. The CLI exited with status 2; the validation driver marked the coverage check failed. The checkpoint and retry limit were unchanged in this test.

| Recording | Decoded / expected chunks | Audio coverage | Selected truncated chunks | Exhausted caption-format retry sequences |
| --- | ---: | --- | ---: | ---: |
| English movie, A Star Is Born | 277 / 333 | About 91.7 / 110.7 minutes; stopped at retry limit | 4 | 4 |
| Chinese movie, Personal Tailor | 365 / 365 | Reached the end, 117.1 minutes | 6 | 6 |
| English Oscars | 288 / 288 | Reached the end, 93.6 minutes | 0 | 0 |
| Chinese Roast! | 317 / 317 | Reached the end, 101.5 minutes | 0 | 1 |

The run used 80 extra generation attempts across 39 retried chunks. Caption-format and repetition checks share attempts, so their separate counters must not be added. Reaching the end establishes audio coverage, not caption completeness or semantic correctness.

As documented in `docs/inference.md`, an early-stopped recording withholds the batch's complete-benchmark prediction export. The raw replies and `raw/client.json` exist; final prediction JSONL and conversion summaries were not produced. There are therefore no final converted event counts, conversion statuses, or new semantic scores for this run. No diagnostic conversion was substituted for that missing export. The default retry limit remains 20; this test does not qualify uninterrupted coverage of all four recordings under that default.

The retained historical full-Long scores include partial/truncated output for three of four recordings. The windowed scores include two partial/truncated windows, and two Short clips have empty output. Default conversion parity establishes that the same saved raw responses produce the same published-format predictions. It does not establish bitwise GPU generation reproducibility or new benchmark scores. The current client also has an explicit operational retry-chunk limit, documented in `docs/inference.md`, that is separate from the historical scored decodes.

## BF16 checkpoint

Tested on 2026-11-08 with this release's source and dependency versions on two H100 80GB GPUs. `verify-model` passed for both checkpoints.

| Configuration | Result |
| --- | --- |
| BF16, one H100 80GB | Does not start: the weights take 65.9 GiB, leaving 0.85 GiB of KV cache against 2.64 GiB needed for a 131,072-token context |
| BF16, two H100 80GB, tensor parallel size 2 | Both profiles start (144 s for `long`, 159 s for `short`); every request completed |
| FP8, one H100 80GB | Both profiles start; every request completed |

With the same inputs and serving profile, BF16 and FP8 outputs on 30 LongCapBench-TAC-Short clips match closely: 95.5% to 98.1% of events pair one-to-one at temporal IoU of at least 0.5 (median IoU 1.0), and paired speech transcripts are 98% to 99% similar. Switching profiles at one precision changes the output by a similar amount (94.1% to 96.7% of events paired). Diarization transcripts of three recordings are 80% to 100% similar across precisions.

Diarization of a 300-second awards-ceremony excerpt stopped after about 57 seconds with the `short` profile, for both precisions, while the `long` profile transcribed the whole excerpt. Use the `long` profile for diarization; the cause was not investigated.

These checks ran the release source on PYTHONPATH inside the earlier validated image, not the published `0.1.0-rc.2` image. They compare outputs between precisions and are not new benchmark scores.

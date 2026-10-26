# Inference with LongCap 35B-A3B

This package serves the stage-2 V4 step-3000 checkpoint before flywheel training.
It produces timestamped audio captions; native event-ancestor prediction is not
a supported feature of this checkpoint. Use the released FP8 model, its tokenizer
and audio processor together with this custom vLLM plugin.

## Install and verify the model

The Docker instructions in the main README are the primary server setup. A CPU
machine can install only the client:

```bash
python -m pip install .
longcap verify-model --model /path/to/LongCap-35B-A3B
```

Python 3.12 is required. The client needs NumPy, SoundFile, HTTPX, Transformers
and tokenizers; it does not install or import PyTorch or vLLM. Server dependencies
are the `server` extra. Use the supplied dependency locks and CUDA 13.0 PyTorch
wheel source when installing that extra. Arbitrary vLLM upgrades may change the
multimodal and hybrid-model APIs this implementation uses.

Keep `SHA256SUMS` with the model. `verify-model` independently hashes every file.
Normal client/server startup verifies metadata checksums and manifest coverage,
then compares the manifest fingerprint across machines. This avoids requiring
the client and server to mount their model directories at the same path. Normal
startup does not rehash the weight payloads; run full verification after every
download or copy, and keep the verified model read-only while serving.

## Start the server

```bash
longcap serve --model /models/longcap-35b-a3b --profile short \
  --host 127.0.0.1 --port 20000 --work-dir /work/service
```

Use `--host 0.0.0.0` inside a container and publish its port to host loopback for
local usage. Wait for `GET /health` to return success. Uploads are spooled under
the work directory. Shared server-filesystem path registration is disabled by
default; trusted local deployments may explicitly enable `--allow-audio-paths`.
The packaged client always uploads local WAV chunks, so it needs no shared
filesystem. The service is intended for a trusted inference deployment; the
package does not provide a public authentication or hosting layer.

Uploaded chunks remain in the spool directory until you remove them after
shutting down the server. There is no automatic retention policy or deletion
endpoint; allocate disk space for the audio you process.

The initial hardware target is one H100 80GB with tensor parallel size 1.
The default profiles preserve the documented server settings:

| Setting | Short | Long |
| --- | ---: | ---: |
| Context tokens | 131072 | 1048576 |
| Maximum sequences | 128 | 4 |
| Maximum audio items per prompt | 64 | 1024 |
| YaRN factor / original context | none | 4 / 262144 |
| SSM cache dtype | float32 | float32 |
| GPU memory utilization | 0.90 | 0.90 |

Start the client at concurrency one. Use `--profile long` for full recordings.
Both prior audio and generated captions consume context. The context setting is
not a guarantee that an arbitrary recording will fit in GPU memory. The client
checks the audio-item limit before sending generation requests; raise the server's
`--max-audio-per-prompt` if a dialogue has more chunks. `--max-num-seqs` can reduce
concurrency pressure. TP=8 is incompatible with the audio tower's 20 heads.

## Caption a recording or batch

```bash
longcap caption --audio /input/example.wav --key example \
  --model /local/models/longcap-35b-a3b \
  --service http://127.0.0.1:20000 --output /output/example
```

For a batch, supply a JSONL manifest instead of `--audio`:

```json
{"key":"recording_001","audio_path":"audio/recording_001.wav","duration":73.5}
```

```bash
longcap caption --manifest /input/manifest.jsonl \
  --model /local/models/longcap-35b-a3b \
  --service http://127.0.0.1:20000 --output /output/batch --concurrency 1
```

Relative audio paths resolve against the manifest's directory. Duration is in
seconds and must not exceed the source. Reference events and labels in a manifest
are ignored when constructing prompts. `--prepare-only` prepares and validates
inputs without contacting a server. Runtime input chunks and raw outputs stay in
your chosen local output directory; they are not distribution assets.

Audio is averaged to mono and written as PCM16 chunks at its native sample rate;
the server resamples each chunk to 16 kHz. The deterministic chunk schedule uses
8, 16, 24 or 32 seconds, with seed 42 and the recording key. Tails shorter than
100 ms are discarded and reported. Do not change benchmark keys when reproducing
the established schedule. A whole recording is one continuing dialogue, with
the exact prompt `Conduct audio caption`, all prior audio and accepted generated
token IDs. The supplied prompt builder handles the SpeechLM token offset;
ordinary OpenAI messages or the base Qwen chat template are not interchangeable.

## Outputs, failures and resume

```text
output/
  inputs/                         # audio chunks, offsets, hashes, dialogue
  raw/run_identity.json           # model, input, code and decoding fingerprints
  raw/records/<key-hash>/
    attempts/00000000.json ...    # request hashes, response text/token IDs/errors
    result.json                  # original segments, retry history and status
  raw/decode/inference_rank0/results.json
  raw/client.json                # coverage, truncation, retries and failures
  predictions/predictions.jsonl
  predictions/conversion.jsonl
  predictions/conversion_summary.json
  predictions/deduplication.json
  predictions/chunk_diagnostics.jsonl
```

The prediction schema is one object per recording:

```json
{"key":"example","duration":12.0,"events":[{"start":0.2,"end":1.1,"attributes":{"type":"speech","transcript":"Hello."}}]}
```

This illustrates syntax, not a measured model prediction. Attribute names and
values are generated by the model. Timestamps are absolute recording seconds.
Native replies contain an attributes JSON object followed by `<start, end>`
pairs, or `<Empty>`. The converter adds the chunk offset once; it retains negative
relative starts, overlapping events and observed timestamp errors. Raw segments
remain unchanged. Multiple EOT-delimited segments for one chunk are joined only
for event parsing, with their original boundaries retained in `result.json`.

Run the identical command again to resume. Completed recording records are
reused; interrupted recordings rebuild their dialogue from the durable request
journal, replaying saved token IDs without resending completed generation calls
or resetting format/repetition budgets. Audio may be uploaded again after a
restart; content-derived server IDs preserve request identity. A process can be
interrupted before its response is durably written; that unrecorded request may
need to be sent again. Changed inputs, model, code or decoding settings require
a fresh output directory.

Format and mechanical repetition checks share four attempts per assistant
segment, with penalties 1.0, 1.05, 1.10 and 1.15. Rejected replies never enter
history; exhausted retries retain the final observed response. Generation uses
greedy decoding and an 8129-token cap. The newer operational default stops a
recording after 20 distinct chunks use retries, marking its prefix incomplete.
Use `--max-retried-chunks 0` only when intentionally disabling that stop for a
fresh run. This operational limit is distinct from the historical scored V4
decodes; the historical results are not new measurements of this release.

Exit status 0 means an export completed without detected partial/error status;
it does not establish semantic correctness. Status 2 indicates partial/truncated
output, exhausted retries or the retry-chunk stop. Status 1 indicates a request,
identity, input or execution failure. Inspect `raw/client.json` and conversion
diagnostics in every case. Request errors and deliberate early stops are terminal
for that output directory: restarting does not silently spend another budget.
Their raw attempts and replies remain available, and a complete-benchmark
prediction file is withheld. A deliberate fresh attempt needs a new output
directory. Other recordings in the batch continue independently.

## Conversion and paper comparability

Default conversion uses `speech_caption_insensitive_overlap_v2`, matching the
saved V4 results. It removes connected, positively overlapping duplicates;
speech with a nonempty transcript may differ only in free-text `caption`.
All other attributes must match. Non-speech and transcript-free events require
complete attribute equality. It preserves the first event and its timestamps;
disjoint and merely touching events remain separate. Warnings alone never
authorize deletion.

The release parser/deduplicator was replayed against all saved V4 outputs:
1,500 Short records, four full Long recordings and 45 Long-10min windows, with
exact agreement for every prediction record. This verifies conversion parity,
not GPU decode parity. The retained full-Long scores include three recordings
with partial/truncated replies; the windowed scores include two such windows.

```bash
longcap convert --run /output/existing-complete-run
```

`--future-start-guard` is an explicit optional newer policy that removes events
starting at least one second after their generating chunk's end. It records
`speech_overlap_v2_plus_future_start_v1` and removal receipts; results can differ
from paper scores. It cannot overwrite an existing export under a different
policy. Use a fresh run, or a separate copy of the saved `inputs/` and `raw/`
directories without `predictions/`, when comparing policies. Historical metrics
must not be attached to changed predictions.

## Development checks

```bash
python -m pip install '.[test]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Tests cover CPU dependency isolation, upload requests, model identity across
mount paths, input-label isolation, chunking, persistent retry replay, event
parsing and overlap deduplication. Clean image/GPU validation and supported
hardware measurements are reported separately by the release validation record.

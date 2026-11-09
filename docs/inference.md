# Inference guide

This guide covers the `longcap` commands in detail. For installation, in a Python environment or with Docker, and a first run, follow the [quickstart](quickstart.md).

| Command | Purpose |
| --- | --- |
| `longcap serve` | Start the GPU server |
| `longcap caption` | Timestamped audio captioning |
| `longcap diarize` | Speech recognition with speaker diarization |
| `longcap convert` | Convert the saved replies of a finished run again |
| `longcap verify-model` | Check a model directory against its `SHA256SUMS` |

Every command has `--help`, and `longcap --version` prints the installed version.

## Inputs

`--audio` takes one file in any format that [libsndfile](https://libsndfile.github.io/libsndfile/) reads, such as WAV or FLAC. Convert other formats first, for example `ffmpeg -i input.mp4 -vn -ac 1 recording.wav`. The recording ID is the file name without its extension unless you pass `--key`.

`--manifest` takes a JSONL file with one recording per line:

```json
{"key": "recording_001", "audio_path": "audio/recording_001.wav", "duration": 73.5}
```

| Field | Meaning |
| --- | --- |
| `key` | Unique recording ID |
| `audio_path` | Audio file; relative paths resolve against the manifest's directory |
| `duration` | Seconds to process, at most the file's length; a shorter value processes a prefix |

Other fields are ignored. The client averages stereo to mono and cuts each recording into chunks of 8, 16, 24, or 32 seconds; the server resamples them to 16 kHz. The chunk schedule is deterministic for a given key. `--prepare-only` cuts the chunks without contacting a server.

## Server

```bash
longcap serve --model /models/longcap-35b-a3b --host 127.0.0.1 --port 20000 --work-dir service-work
```

| Profile | Context | Concurrent requests | Audio chunks per request | Use for |
| --- | ---: | ---: | ---: | --- |
| `long` (default) | 1,048,576 tokens | 4 | 1,024 | Long recordings and diarization |
| `short` | 131,072 tokens | 128 | 64 | Captioning short clips at high throughput |

Use `long` for diarization: with `short`, diarization of a five-minute recording stopped after about a minute in our checks. With `short`, pass `--concurrency` to the client to use the extra capacity. `--max-num-seqs` and `--max-audio-per-prompt` override the profile values.

`--tensor-parallel-size 2` splits the model across two GPUs, which the BF16 checkpoint needs. The FP8 checkpoint runs on one H100 80GB. Tensor parallel size 8 is not supported.

`longcap serve` runs the same GPU kernels in a Python environment and in the Docker image. Unless `CUDA_HOME` or `CUDA_PATH` is set, it uses the CUDA 13.0 compiler installed from `server.lock`; a toolkit you set yourself should be CUDA 13.0. It turns off two vLLM features that depend on whether a CUDA toolkit is installed: the FlashInfer top-k/top-p sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`), which greedy decoding does not use, and the fusion of the tensor-parallel all-reduce with RMSNorm.

The server keeps uploaded chunks under `--work-dir` until you delete them after shutdown; allocate disk space for the audio you process. It has no authentication and is meant for local or trusted use: keep it on `127.0.0.1`, or put an authenticated proxy in front of it. `--allow-audio-paths` lets clients register files by server-side path and should only be used on a trusted machine.

## Decoding

Decoding is greedy. `caption` allows 8,192 output tokens per chunk and `diarize` 32,768; `--max-tokens` changes this.

For captioning, a reply that repeats itself or breaks the output format is generated again with a repetition penalty of 1.05, then 1.10, then 1.15. Rejected replies never enter the dialogue. Once 20 chunks of a recording have needed retries, captioning stops that recording and exports what it decoded so far; `--max-retried-chunks 0` removes this limit. Diarization has no retries.

## Outputs

```text
OUTPUT/
  inputs/                             audio chunks and the prepared dialogue
  raw/client.json                     coverage, retries, and errors of the run
  raw/records/<id>/                   every request and reply, used to resume
  predictions/predictions.jsonl       captioning results
  predictions/diarization.jsonl       diarization results
  predictions/conversion.jsonl        per-recording status and parsing issues
  predictions/conversion_summary.json totals for the run
```

Captioning writes one line per recording. Event times are seconds from the start of the recording, and events may overlap:

```json
{"key": "recording_001", "duration": 73.5, "events": [{"start": 0.2, "end": 1.1, "attributes": {"type": "speech", "caption": "..."}}]}
```

Diarization writes segments with a speaker label and text. Speaker labels distinguish voices within one recording:

```json
{"key": "recording_001", "duration": 73.5, "segments": [{"start": 0.2, "end": 1.1, "speaker": "0", "text": "..."}]}
```

Each recording's status in `conversion.jsonl` is `ok`, `partial` (some replies were truncated or could not be parsed, or captioning stopped early), `empty`, or `parse_error`. A caption event repeated with the same attributes and overlapping in time is kept once (repeated speech may differ in its free-text caption); nothing else is repaired. Captioning also writes `deduplication.json`, `chunk_diagnostics.jsonl`, and `future_start_removals.jsonl` next to the predictions.

## Exit status and resume

| Exit status | Meaning | What to do |
| ---: | --- | --- |
| 0 | Every recording finished | Use the predictions |
| 2 | Some output is incomplete; see `conversion.jsonl` | Use the predictions with care, or rerun those recordings in a new output directory |
| 1 | A request failed, the service could not be reached, or an input was invalid | Fix the cause and rerun the same command |

Rerunning the same command resumes the run: finished recordings are kept, interrupted ones continue from their saved replies, and failed ones are retried. After a connection failure or server error, recordings that have not started are left for the rerun. Changing the input, model, client version, decoding options, or server settings requires a new output directory. A run directory can be copied or moved.

## Convert again

```bash
longcap convert --run OUTPUT
```

`convert` rebuilds the predictions from the saved replies. With `--future-start-guard`, captioning drops events whose start lies after the end of the chunk that produced them. Use it on a copy of the run, because the conversion settings are recorded with the predictions.

## Model files

`longcap verify-model --model DIR` hashes every runtime file listed in `SHA256SUMS`. The client and server compare these checksums at startup, so both must use the same checkpoint, even when the model directory is mounted at different paths.

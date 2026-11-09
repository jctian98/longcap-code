"""Public command line; importing it never imports torch or vLLM."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def serve_arguments(args) -> list[str]:
    long = args.profile == "long"
    options = [
        "--model", str(args.model), "--host", args.host, "--port", str(args.port),
        "--spool-dir", str(args.work_dir / "audio-spool"),
        "--stats-jsonl", str(args.work_dir / "engine.jsonl"),
        "--tensor-parallel-size", str(args.tensor_parallel_size), "--gpu-memory-utilization", "0.90",
        "--max-model-len", "1048576" if long else "131072",
        "--max-num-seqs", str(args.max_num_seqs or (4 if long else 128)),
        "--max-num-batched-tokens", "16384", "--mamba-ssm-cache-dtype", "float32",
        "--slice-cache-max", "2048" if long else "16384",
        "--io-threads", "4" if long else "8", "--torch-num-threads", "4",
        "--mm-processor-cache-gb", "4" if long else "8",
        "--max-audio-per-prompt", str(args.max_audio_per_prompt or (1024 if long else 64)),
    ]
    if long:
        options += ["--yarn-factor", "4", "--yarn-original-max-position-embeddings", "262144"]
    if args.allow_audio_paths:
        options += ["--allow-audio-paths"]
    return options


def serve_environment() -> None:
    """Make a pip-installed server behave like the Docker image.

    LongCap decodes greedily, so vLLM's FlashInfer top-k/top-p sampler never
    affects its output; turning it off avoids compiling it with nvcc at startup.
    Optional vLLM kernels such as DeepGEMM look for a CUDA toolkit when they are
    imported; without CUDA_HOME or CUDA_PATH, point them at the CUDA 13.0
    toolkit installed from server.lock. The Docker image sets its own CUDA_HOME.
    """
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return
    from importlib.util import find_spec
    spec = find_spec("nvidia")
    for base in (spec.submodule_search_locations or []) if spec else []:
        candidate = Path(base) / "cu13"
        if (candidate / "bin" / "nvcc").is_file():
            os.environ["CUDA_HOME"] = str(candidate)
            return


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults only for options that have a meaningful one."""

    def _get_help_string(self, action):
        if action.required or action.default in (None, False):
            return action.help
        return super()._get_help_string(action)


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("longcap")
    except PackageNotFoundError:
        return "unknown"


def main(argv=None) -> int:
    formatter = _HelpFormatter
    parser = argparse.ArgumentParser(
        prog="longcap", description="LongCap: timestamped audio captioning and speech diarization")
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="Start the LongCap vLLM server", formatter_class=formatter)
    serve.add_argument("--model", type=Path, required=True, help="Model directory")
    serve.add_argument("--profile", choices=["short", "long"], default="long",
                       help="long: 1M-token context for long recordings and diarization; "
                            "short: 128k-token context with more concurrent requests")
    serve.add_argument("--tensor-parallel-size", type=int, default=1,
                       help="Number of GPUs to split the model across; use 2 for the BF16 checkpoint")
    serve.add_argument("--host", default="127.0.0.1", help="Address to listen on")
    serve.add_argument("--port", type=int, default=20000, help="Port to listen on")
    serve.add_argument("--work-dir", type=Path, default=Path("service-work"),
                       help="Directory for uploaded audio and server logs")
    serve.add_argument("--max-num-seqs", type=int,
                       help="Maximum concurrent requests (profile default: 4 long, 128 short)")
    serve.add_argument("--max-audio-per-prompt", type=int,
                       help="Maximum audio chunks per request (profile default: 1024 long, 64 short)")
    serve.add_argument("--allow-audio-paths", action="store_true",
                       help="Let clients register server-side file paths; only for trusted local use")

    verify = commands.add_parser("verify-model", help="Check every model file against SHA256SUMS",
                                 formatter_class=formatter)
    verify.add_argument("--model", type=Path, required=True, help="Model directory")

    for name, description, token_budget in (
        ("caption", "Caption audio: timestamped speech, music, and sound events", 8192),
        ("diarize", "Transcribe speech with speaker labels and timestamps", 32768),
    ):
        command = commands.add_parser(name, help=description, formatter_class=formatter)
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--audio", type=Path, help="One audio file")
        source.add_argument("--manifest", type=Path,
                            help="JSONL file with one recording per line: key, audio_path, duration")
        command.add_argument("--key", help="Recording ID for --audio (default: the file name without extension)")
        command.add_argument("--model", type=Path, required=True,
                             help="Local copy of the model the server runs; used for its tokenizer and checksums")
        command.add_argument("--service", default="http://127.0.0.1:20000", help="Server URL")
        command.add_argument("--output", type=Path, required=True, help="Output directory for this run")
        command.add_argument("--concurrency", type=int, default=1, help="Recordings processed in parallel")
        command.add_argument("--max-tokens", type=int, default=token_budget, help="Output token limit per chunk")
        command.add_argument("--http-timeout", type=float, default=1800,
                             help="Seconds to wait for one server response")
        command.add_argument("--prepare-only", action="store_true",
                             help="Only cut the audio into chunks; do not contact the server")
        if name == "caption":
            command.add_argument("--max-retried-chunks", type=int, default=20,
                                 help="Stop a recording after this many chunks needed retries; 0 disables the limit")
            command.add_argument("--future-start-guard", action="store_true",
                                 help="Drop events that start after the end of their chunk")

    convert = commands.add_parser("convert", help="Convert a finished run's raw replies again",
                                  formatter_class=formatter)
    convert.add_argument("--run", type=Path, required=True, help="Output directory of a previous run")
    convert.add_argument("--future-start-guard", action="store_true",
                         help="Drop events that start after the end of their chunk")
    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            args.work_dir.mkdir(parents=True, exist_ok=True)
            serve_environment()
            from espnet2.speechlm.vllm.vllm_server.service import main as service_main
            return service_main(serve_arguments(args))
        if args.command == "verify-model":
            from .identity import checkpoint_identity
            print(json.dumps(checkpoint_identity(args.model, verify_weights=True), indent=2))
            return 0
        from .processing.common import file_sha256, run_lock, write_jsonl
        if args.command == "convert":
            from .processing.convert import convert_predictions
            from .processing.common import read_json
            if not (args.run / "inputs/preparation.json").is_file():
                raise ValueError(f"{args.run} is not a LongCap run directory")
            with run_lock(args.run):
                prepared = read_json(args.run / "inputs/preparation.json")
                if prepared["identity"]["system_prompt"] == "Conduct audio diarization":
                    if args.future_start_guard:
                        raise ValueError("--future-start-guard applies only to captioning")
                    from .processing.diarization import convert_diarization
                    report = convert_diarization(args.run)
                else:
                    report = convert_predictions(args.run, future_start_guard=args.future_start_guard)
                print(json.dumps(report, indent=2))
            incomplete = report["num_truncated"] or any(k in report["statuses"] for k in ("partial", "parse_error"))
            exhausted = (report.get("caption_retry_summary") or {}).get("num_exhausted", 0)
            return 2 if incomplete or exhausted else 0
        from .processing.prepare import prepare_inputs
        from .runner import run_caption, run_diarization
        run = args.output.resolve()
        with run_lock(run):
            manifest = args.manifest
            if args.audio:
                import soundfile
                audio = args.audio.resolve()
                if not audio.is_file():
                    raise ValueError(f"audio file not found: {args.audio}")
                try:
                    info = soundfile.info(audio)
                except RuntimeError as error:
                    raise ValueError(f"cannot read {args.audio} as audio; convert it to WAV or FLAC "
                                     f"first, for example with ffmpeg ({error})") from error
                row = {"key": args.key or audio.stem, "audio_path": str(audio),
                       "duration": info.frames / info.samplerate,
                       "source_sha256": file_sha256(audio)}
                manifest = run / "audio_manifest.jsonl"
                if manifest.exists() and json.loads(manifest.read_text()) != row:
                    raise ValueError(f"{args.output} was used for a different audio file or key; "
                                     "use a new output directory")
                write_jsonl(manifest, [row])
            task = "diarization" if args.command == "diarize" else "caption"
            report = prepare_inputs(manifest, run, workers=1, task=task)
            if args.prepare_only:
                print(json.dumps(report, indent=2))
                return 0
            options = {"concurrency": args.concurrency, "max_tokens": args.max_tokens,
                       "timeout": args.http_timeout}
            if task == "diarization":
                report, code = run_diarization(run, args.model, args.service, **options)
            else:
                report, code = run_caption(
                    run, args.model, args.service, **options,
                    max_retried_chunks=args.max_retried_chunks,
                    future_start_guard=args.future_start_guard,
                )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return code
    except (ValueError, RuntimeError, OSError) as error:
        print(f"longcap: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        import httpx
        if isinstance(error, httpx.HTTPError):
            print(f"longcap: request to the LongCap service failed: {error}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())

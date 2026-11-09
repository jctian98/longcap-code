"""Public command line; importing it never imports torch or vLLM."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def serve_arguments(args) -> list[str]:
    long = args.profile == "long"
    options = [
        "--model", str(args.model), "--host", args.host, "--port", str(args.port),
        "--spool-dir", str(args.work_dir / "audio-spool"),
        "--stats-jsonl", str(args.work_dir / "engine.jsonl"),
        "--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.90",
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="LongCap 35B-A3B audio captioning and speech diarization")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Start the custom vLLM server")
    serve.add_argument("--model", type=Path, required=True)
    serve.add_argument("--profile", choices=["short", "long"], default="short")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=20000)
    serve.add_argument("--work-dir", type=Path, default=Path("service-work"))
    serve.add_argument("--max-num-seqs", type=int)
    serve.add_argument("--max-audio-per-prompt", type=int)
    serve.add_argument("--allow-audio-paths", action="store_true",
                       help="Enable shared-path registration for a trusted local deployment")
    verify = commands.add_parser("verify-model", help="Verify every published model checksum")
    verify.add_argument("--model", type=Path, required=True)
    for name, description, token_budget in (
        ("caption", "Caption local audio with the upload client", 8192),
        ("diarize", "Transcribe local audio with speaker labels and timestamps", 32768),
    ):
        command = commands.add_parser(name, help=description)
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--audio", type=Path)
        source.add_argument("--manifest", type=Path)
        command.add_argument("--key", default="example")
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--service", default="http://127.0.0.1:20000")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--concurrency", type=int, default=1)
        command.add_argument("--max-tokens", type=int, default=token_budget)
        command.add_argument("--http-timeout", type=float, default=1800)
        command.add_argument("--prepare-only", action="store_true")
        if name == "caption":
            command.add_argument("--max-retried-chunks", type=int, default=20)
            command.add_argument("--future-start-guard", action="store_true",
                                 help="Opt into a newer timestamp filter; changes the paper's conversion policy")
    convert = commands.add_parser("convert", help="Convert a complete saved raw run")
    convert.add_argument("--run", type=Path, required=True)
    convert.add_argument("--future-start-guard", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            args.work_dir.mkdir(parents=True, exist_ok=True)
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
                info = soundfile.info(audio)
                row = {"key": args.key, "audio_path": str(audio),
                       "duration": info.frames / info.samplerate,
                       "source_sha256": file_sha256(audio)}
                manifest = run / "audio_manifest.jsonl"
                if manifest.exists() and json.loads(manifest.read_text()) != row:
                    raise ValueError("Source audio changed; use a fresh output directory")
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


if __name__ == "__main__":
    raise SystemExit(main())

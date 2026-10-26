"""Native-task protocol, timestamp conversion, and resumable HTTP inference."""
import hashlib
import io
import json

import pytest
import soundfile as sf

from longcap import runner
from longcap.cli import main
from longcap.identity import checkpoint_identity
from longcap.processing.common import read_json, write_json
from longcap.processing.convert import convert_predictions
from longcap.processing.diarization import convert_diarization, parse_diarization_turn
from longcap.processing.prepare import prepare_inputs
from test_release_client import PROMPT, Tokenizer, model_fixture, prepared_fixture


def test_native_transcript_offsets_overlap_and_unparsed_tail():
    text = ("Hello. <spk0> <-0.5, 1.2>\n"
            "Hi! <spk1> <0.2, 1.0>\n"
            "[noise] <bgn> <1.0, 1.1>\n"
            "unfinished words")
    segments, issues = parse_diarization_turn(text, 16)
    assert segments == [
        {"start": 15.5, "end": 17.2, "speaker": "0", "text": "Hello."},
        {"start": 16.2, "end": 17.0, "speaker": "1", "text": "Hi!"},
    ]
    assert issues == [{"reason": "unparsed_text", "text": "unfinished words"}]
    assert parse_diarization_turn("hello <spk0> <-1, 2>", 0)[0][0]["start"] == 0


def test_native_duplicates_are_retained_and_bad_clock_reported():
    text = "Hello <spk0> <0, 2>\nHello <spk0> <0, 2>"
    segments, issues = parse_diarization_turn(text, 8)
    assert len(segments) == 2 and segments[0] == segments[1] and not issues
    segments, issues = parse_diarization_turn("text <spk0> <00:99:00:00, 00:99:01:00>", 0)
    assert not segments and issues[0]["reason"] == "invalid_timestamp"
    assert parse_diarization_turn("<Empty>\n", 0) == ([], [])


def test_legacy_native_format_and_clock_wrappers():
    segments, issues = parse_diarization_turn(
        "start time: 00:00:08:00\n"
        "speaker0, None, [0.0, 1.5] : Hello\n"
        "speaker1, None, [1.0, 2.0] : yes\n"
        "end time: 00:00:16:00", 8,
    )
    assert [(x["speaker"], x["start"], x["end"], x["text"]) for x in segments] == [
        ("0", 8, 9.5, "Hello"), ("1", 9, 10, "yes"),
    ]
    assert not issues


def test_diarize_cli_prepares_native_prompt_and_refuses_task_reuse(tmp_path):
    source = tmp_path / "input"
    run = prepared_fixture(source, seconds=2, task="diarization")
    assert main(["diarize", "--manifest", str(source / "manifest.jsonl"),
                 "--model", str(tmp_path / "not-needed-for-preparation"),
                 "--output", str(run), "--prepare-only"]) == 0
    dialogue = json.loads((run / "inputs/dialogue.jsonl").read_text())
    assert dialogue["messages"][0] == ["system", "text", "Conduct audio diarization"]
    assert "REFERENCE" not in str(dialogue)
    with pytest.raises(ValueError, match="Preparation settings changed"):
        prepare_inputs(source / "manifest.jsonl", run, workers=1)
    with pytest.raises(ValueError, match="Caption conversion requires"):
        convert_predictions(run)


def test_diarization_upload_protocol_export_and_terminal_resume(tmp_path, monkeypatch):
    model = model_fixture(tmp_path / "model")
    run = prepared_fixture(tmp_path / "input", seconds=40, task="diarization")
    identity = checkpoint_identity(model)
    chunks = json.loads((run / "inputs/chunks.jsonl").read_text())["chunks"]

    class Client:
        requests = []
        uploads = []
        def __init__(self, *args, **kwargs): pass
        def models(self): return {"checkpoint": identity, "model": "/server-model", "max_audio_per_prompt": 64}
        def close(self): pass
        def register_path(self, path): pytest.fail("Client must upload local chunks")
        def register_bytes(self, data, filename):
            self.uploads.append(data)
            info = sf.info(io.BytesIO(data))
            return {"audio_id": hashlib.sha256(data).hexdigest(), "num_frames": info.frames,
                    "sample_rate": info.samplerate}
        def generate(self, prompt, audio, sampling):
            self.requests.append((prompt, audio, sampling))
            return {"token_ids": PROMPT.encode_text("Hello. <spk0> <0.0, 0.5>"),
                    "terminator": 2, "finish_reason": "eos"}

    monkeypatch.setattr(runner, "CaptionerClient", Client)
    report, code = runner.run_diarization(run, model, "http://fake", tokenizer=Tokenizer())
    assert code == 0 and report["task"] == "diarization"
    assert len(Client.requests) == len(chunks)
    assert all(x[2]["max_tokens"] == 32768 and x[2]["repetition_penalty"] == 1 for x in Client.requests)
    assert report["caption_retry_summary"]["num_retries"] == report["loop_retry_summary"]["num_retries"] == 0
    assert "Conduct audio diarization" in PROMPT.decode_text(Client.requests[0][0])
    output = json.loads((run / "predictions/diarization.jsonl").read_text())
    assert [segment["start"] for segment in output["segments"]] == [chunk["start"] for chunk in chunks]
    assert all(segment["speaker"] == "0" for segment in output["segments"])
    assert not (run / "predictions/predictions.jsonl").exists()
    assert not (run / "predictions/deduplication.json").exists()
    assert main(["convert", "--run", str(run)]) == 0
    replay, code = runner.run_diarization(run, model, "http://relocated", tokenizer=Tokenizer())
    assert code == 0 and replay["num_resumed"] == 1 and len(Client.requests) == len(chunks)
    with pytest.raises(ValueError, match="Prepared task differs"):
        runner.run_caption(run, model, "http://fake", tokenizer=Tokenizer())


def test_diarization_truncation_and_conversion_failures_remain_visible(tmp_path):
    run = prepared_fixture(tmp_path / "input", task="diarization")
    stats = {"task": "diarization", "num_errors": 0, "num_written": 1, "num_examples": 1,
             "num_truncated": 1, "truncated_ids": ["sample"]}
    write_json(run / "raw/client.json", stats)
    write_json(run / "raw/decode/inference_rank0/results.json", {
        "sample": [["assistant", "text", ["Hello. <spk0> <0, 0.5>\nunfinished"]]],
    })
    assert main(["convert", "--run", str(run)]) == 2
    report = read_json(run / "predictions/conversion_summary.json")
    assert report["num_segments"] == 1 and report["statuses"] == {"partial": 1}
    assert main(["convert", "--run", str(run), "--future-start-guard"]) == 1
    stats["num_errors"] = 1
    write_json(run / "raw/client.json", stats)
    with pytest.raises(ValueError, match="request errors"):
        convert_diarization(run)

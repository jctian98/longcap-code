"""CPU integration checks for uploads, model identity and durable inference."""
import hashlib
import io
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from espnet2.speechlm.vllm.vllm_client.dialogue import MultiHopDriver
from espnet2.speechlm.vllm.vllm_client.prompt import PromptBuilder
from espnet2.speechlm.vllm.vllm_client.protocol import GenerateResult
from longcap import runner
from longcap.identity import checkpoint_identity, require_matching_model
from longcap.processing.common import read_json, write_json, write_jsonl
from longcap.processing.convert import convert_predictions
from longcap.processing.prepare import prepare_inputs, verify_prepared
from longcap.transport import JournalEngine


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, tokens, **kwargs):
        return bytes(tokens).decode()


PROMPT = PromptBuilder(Tokenizer())
VALID = '{"type":"speech","transcript":"hello"} <0, 1>'


def model_fixture(path):
    path.mkdir()
    files = {
        "config.json": {"architectures": ["SpeechLMForConditionalGeneration"]},
        "generation_config.json": {"eos_token_id": 2},
        "tokenizer.json": {"fixture": True}, "tokenizer_config.json": {},
        "model.safetensors.index.json": {"weight_map": {"weight": "model.safetensors"}},
        "audio_processor/preprocessor_config.json": {"sampling_rate": 16000},
    }
    for name, contents in files.items():
        write_json(path / name, contents)
    (path / "model.safetensors").write_bytes(b"test fixture weights")
    paths = sorted(p for p in path.rglob("*") if p.is_file())
    (path / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(path).as_posix()}\n"
        for p in paths
    ))
    return path


def prepared_fixture(path, seconds=1, key="sample", task="caption"):
    path.mkdir()
    audio = path / "input.wav"
    sf.write(audio, np.zeros((int(seconds * 16000), 2)), 16000)
    manifest = path / "manifest.jsonl"
    write_jsonl(manifest, [{"key": key, "audio_path": str(audio), "duration": seconds,
                           "events": [{"caption": "THIS REFERENCE MUST NEVER ENTER A PROMPT"}]}])
    run = path / "run"
    prepare_inputs(manifest, run, workers=1, task=task)
    return run


def test_model_identity_ignores_mount_path_but_checks_metadata(tmp_path):
    import shutil
    first = model_fixture(tmp_path / "first")
    second = tmp_path / "another-container-mount"
    shutil.copytree(first, second)
    a, b = checkpoint_identity(first), checkpoint_identity(second)
    assert a == b
    require_matching_model(a, {"checkpoint": b, "model": "/different/server/path"})
    write_json(second / "config.json", {"architectures": ["wrong"]})
    with pytest.raises(ValueError, match="checksum mismatch"):
        checkpoint_identity(second)
    (first / "model.safetensors").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum mismatch"):
        checkpoint_identity(first, verify_weights=True)


def test_model_manifest_rejects_escape_and_missing_shards(tmp_path):
    model = model_fixture(tmp_path / "model")
    sums = model / "SHA256SUMS"
    original = sums.read_text()
    sums.write_text(original + "a" * 64 + "  ../escape\n")
    with pytest.raises(ValueError, match="Unsafe"):
        checkpoint_identity(model)
    sums.write_text("\n".join(line for line in original.splitlines() if not line.endswith("  model.safetensors")) + "\n")
    with pytest.raises(ValueError, match="every indexed"):
        checkpoint_identity(model)


def test_preparation_isolates_labels_and_detects_original_audio_change(tmp_path):
    run = prepared_fixture(tmp_path / "input", seconds=47.5)
    dialogues = (run / "inputs/dialogue.jsonl").read_text()
    assert "REFERENCE" not in dialogues and "events" not in dialogues
    rows = [json.loads(line) for line in (run / "inputs/chunks.jsonl").read_text().splitlines()]
    assert len(rows[0]["chunks"]) >= 2
    assert rows[0]["chunks"][0]["start"] == 0
    assert rows[0]["chunks"][-1]["end"] == 47.5
    assert not Path(rows[0]["chunks"][0]["audio_path"]).is_absolute()
    sf.write(Path(rows[0]["audio_path"]), np.ones((760000, 2)), 16000)
    verify_prepared(run)  # Inference and conversion need only the prepared chunks.
    with pytest.raises(ValueError, match="Source audio changed"):
        verify_prepared(run, check_source=True)


class ScriptedEngine:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def generate(self, requests):
        self.requests.extend(requests)
        value = next(self.replies)
        if isinstance(value, BaseException):
            raise value
        text, finish = value
        return [GenerateResult(PROMPT.encode_text(text), {"eos": 2, "eot": 3, "length": None}[finish], finish)]


def test_interrupted_journal_replays_retries_without_respending_budget(tmp_path):
    messages = [("user", "text", "first"), ("assistant", "text", ""),
                ("user", "text", "second"), ("assistant", "text", "")]
    first_engine = ScriptedEngine([("bad", "eos"), (VALID, "eos"), KeyboardInterrupt()])
    first = MultiHopDriver(JournalEngine(first_engine, tmp_path), PROMPT,
                           loop_retry=True, caption_format_retry=True)
    with pytest.raises(KeyboardInterrupt):
        first.run([("sample", messages)])
    assert len(list(tmp_path.glob("*.json"))) == 2
    next_engine = ScriptedEngine([("<Empty>", "eos")])
    second = MultiHopDriver(JournalEngine(next_engine, tmp_path), PROMPT,
                            loop_retry=True, caption_format_retry=True)
    result = second.run([("sample", messages)])["sample"]
    assert len(next_engine.requests) == 1
    assert result.caption_retry_attempts[0]["selected_attempt_index"] == 1
    assert PROMPT.decode_text(next_engine.requests[0].prompt_token_ids) == "first" + VALID + "second"
    assert "bad" not in PROMPT.decode_text(next_engine.requests[0].prompt_token_ids)
    assert result.message_round_indices == [0, 1]


def test_failed_request_is_resent_on_resume_and_changed_request_is_refused(tmp_path):
    from espnet2.speechlm.vllm.vllm_client.protocol import GenerateRequest
    request = GenerateRequest([1, 5, 7, 300])
    first = JournalEngine(ScriptedEngine([RuntimeError("network unavailable")]), tmp_path)
    with pytest.raises(RuntimeError, match="network"):
        first.generate([request])
    assert list(tmp_path.glob("*.json")) == []
    next_engine = ScriptedEngine([(VALID, "eos")])
    assert JournalEngine(next_engine, tmp_path).generate([request])[0].finish_reason == "eos"
    assert len(next_engine.requests) == 1
    request.prompt_token_ids.append(301)
    with pytest.raises(ValueError, match="differs"):
        JournalEngine(ScriptedEngine([]), tmp_path).generate([request])


def test_eot_segments_retain_round_mapping():
    driver = MultiHopDriver(ScriptedEngine([(VALID, "eot"), ("<Empty>", "eos")]), PROMPT)
    result = driver.run([("x", [("user", "text", "one"), ("assistant", "text", "")])])["x"]
    assert result.message_round_indices == [0, 0]
    assert runner._joined_rounds(asdict(result)) == [["assistant", "text", [VALID + "\n<Empty>"]]]


def test_eot_segment_cap_is_reported_as_truncation():
    driver = MultiHopDriver(ScriptedEngine([(VALID, "eot"), (VALID, "eot")]),
                            PROMPT, max_segments_per_round=2)
    result = driver.run([("x", [("user", "text", "one"), ("assistant", "text", "")])])["x"]
    assert result.truncated_hops == 1
    assert result.message_round_indices == [0, 0]
    assert len(result.messages) == 2


def test_truncated_empty_output_stays_partial_and_convert_exits_nonzero(tmp_path):
    from longcap.cli import main
    run = prepared_fixture(tmp_path / "input")
    write_json(run / "raw/client.json", {"num_errors": 0, "num_written": 1, "num_examples": 1,
                                        "num_truncated": 1, "truncated_ids": ["sample"]})
    write_json(run / "raw/decode/inference_rank0/results.json", {
        "sample": [["assistant", "text", ["<Empty>"]]],
    })
    assert convert_predictions(run)["statuses"] == {"partial": 1}
    assert main(["convert", "--run", str(run)]) == 2


def test_end_to_end_upload_retry_conversion_and_terminal_resume(tmp_path, monkeypatch):
    model = model_fixture(tmp_path / "model")
    run = prepared_fixture(tmp_path / "input")
    identity = checkpoint_identity(model)

    class FakeClient:
        uploads = []
        requests = []

        def __init__(self, *args, **kwargs): pass
        def models(self): return {"checkpoint": identity, "model": "/remote/mount", "max_audio_per_prompt": 64}
        def close(self): pass
        def register_path(self, path): pytest.fail("Remote client attempted a shared path")
        def register_bytes(self, data, filename):
            self.uploads.append((data, filename))
            info = sf.info(io.BytesIO(data))
            return {"audio_id": hashlib.sha256(data).hexdigest(), "num_frames": info.frames,
                    "sample_rate": info.samplerate}
        def generate(self, prompt, audio, sampling):
            self.requests.append((prompt, audio, sampling))
            text = "transcript only" if len(self.requests) == 1 else VALID
            return {"token_ids": PROMPT.encode_text(text), "terminator": 2, "finish_reason": "eos"}

    monkeypatch.setattr(runner, "CaptionerClient", FakeClient)
    report, code = runner.run_caption(run, model, "http://fake", tokenizer=Tokenizer())
    assert code == 0 and report["caption_retry_summary"]["num_retries"] == 1
    assert len(FakeClient.uploads) == 1
    assert [x[2]["repetition_penalty"] for x in FakeClient.requests] == [1.0, 1.05]
    assert "REFERENCE" not in PROMPT.decode_text(FakeClient.requests[0][0])
    predictions = json.loads((run / "predictions/predictions.jsonl").read_text())
    assert predictions["events"][0]["attributes"]["transcript"] == "hello"
    replay, code = runner.run_caption(run, model, "http://another-port", tokenizer=Tokenizer())
    assert code == 0 and replay["num_resumed"] == 1
    assert len(FakeClient.requests) == 2 and len(FakeClient.uploads) == 1


def test_default_converter_keeps_future_events_and_opt_in_is_versioned(tmp_path):
    import shutil
    run = prepared_fixture(tmp_path / "input")
    write_json(run / "raw/client.json", {"num_errors": 0, "num_written": 1, "num_examples": 1,
                                        "num_truncated": 0, "truncated_ids": []})
    write_json(run / "raw/decode/inference_rank0/results.json", {
        "sample": [["assistant", "text", ['{"caption":"future"} <3, 4>']]],
    })
    standard = convert_predictions(run)
    assert standard["num_events"] == 1
    assert standard["postprocessing_policy"] == "speech_caption_insensitive_overlap_v2"
    with pytest.raises(ValueError, match="Postprocessing policy changed"):
        convert_predictions(run, future_start_guard=True)
    alternate = tmp_path / "alternate"
    shutil.copytree(run, alternate)
    shutil.rmtree(alternate / "predictions")
    filtered = convert_predictions(alternate, future_start_guard=True)
    assert filtered["num_events"] == 0 and filtered["num_future_start_removed"] == 1
    assert filtered["postprocessing_policy"] == "speech_overlap_v2_plus_future_start_v1"


def test_retry_chunk_limit_stops_before_loading_next_chunk_and_keeps_budget():
    engine = ScriptedEngine([("invalid", "eos"), (VALID, "eos")])
    resolved = []
    from espnet2.speechlm.vllm.vllm_client.audio import AudioSegment
    def resolve(index):
        resolved.append(index)
        return AudioSegment(str(index), None, 0, 16000, 16000), np.zeros(0)
    driver = MultiHopDriver(engine, PROMPT, audio_resolver=resolve,
                            loop_retry=True, caption_format_retry=True, max_retried_chunks=1)
    result = driver.run([("x", [("user", "audio", 0), ("assistant", "text", ""),
                                ("user", "audio", 1), ("assistant", "text", "")])])["x"]
    assert resolved == [0]
    assert result.early_stop["remaining_chunks"] == 1
    assert result.caption_retry_attempts[0]["selected_attempt_index"] == 1
    assert result.messages == [["assistant", "text", [VALID]]]


def test_cpu_cli_imports_without_torch_or_vllm():
    script = '''
import sys
class BlockGPU:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'torchaudio', 'vllm'}:
            raise RuntimeError('GPU dependency imported: ' + fullname)
sys.meta_path.insert(0, BlockGPU())
from longcap import cli, runner
from longcap.transport import UploadAudioBook
assert 'torch' not in sys.modules and 'vllm' not in sys.modules
'''
    subprocess.run([sys.executable, "-c", script], check=True, env=os.environ.copy())


def test_batch_resumes_after_service_outage(tmp_path, monkeypatch):
    import httpx
    model = model_fixture(tmp_path / "model")
    identity = checkpoint_identity(model)
    rows = []
    for key in ("a", "b", "c"):
        path = tmp_path / f"{key}.wav"
        sf.write(path, np.zeros(16000), 16000)
        rows.append({"key": key, "audio_path": str(path), "duration": 1.0})
    write_jsonl(tmp_path / "manifest.jsonl", rows)
    run = tmp_path / "run"
    prepare_inputs(tmp_path / "manifest.jsonl", run, workers=1)
    state = {"down": False, "generations": 0, "outage": True}

    class Client:
        def __init__(self, *args, **kwargs): pass
        def models(self): return {"checkpoint": identity, "model": "x", "max_audio_per_prompt": 64}
        def close(self): pass
        def register_bytes(self, data, filename):
            if state["down"]:
                raise httpx.ConnectError("Connection refused")
            info = sf.info(io.BytesIO(data))
            return {"audio_id": hashlib.sha256(data).hexdigest(), "num_frames": info.frames,
                    "sample_rate": info.samplerate}
        def generate(self, prompt, audio, sampling):
            state["generations"] += 1
            state["down"] = state["outage"]  # The server goes away after its first reply.
            return {"token_ids": PROMPT.encode_text(VALID), "terminator": 2, "finish_reason": "eos"}

    monkeypatch.setattr(runner, "CaptionerClient", Client)
    report, code = runner.run_caption(run, model, "http://fake", tokenizer=Tokenizer())
    assert code == 1 and report["error_ids"] == ["b"] and report["not_started_ids"] == ["c"]
    state.update(down=False, generations=0, outage=False)  # The server is back.
    report, code = runner.run_caption(run, model, "http://fake", tokenizer=Tokenizer())
    assert code == 0 and report["num_resumed"] == 1 and state["generations"] == 2
    assert len((run / "predictions/predictions.jsonl").read_text().splitlines()) == 3


def test_copied_run_checks_and_uses_its_own_chunks(tmp_path):
    import shutil
    from longcap.cli import main
    run = prepared_fixture(tmp_path / "original")
    write_json(run / "raw/client.json", {"num_errors": 0, "num_written": 1, "num_examples": 1,
                                         "num_truncated": 0, "truncated_ids": []})
    write_json(run / "raw/decode/inference_rank0/results.json", {"sample": [["assistant", "text", [VALID]]]})
    copy = tmp_path / "copy"
    shutil.copytree(run, copy)
    shutil.rmtree(run)
    assert main(["convert", "--run", str(copy)]) == 0
    chunk = json.loads((copy / "inputs/chunks.jsonl").read_text())["chunks"][0]
    (copy / chunk["audio_path"]).write_bytes(b"changed")
    assert main(["convert", "--run", str(copy)]) == 1


def test_format_check_and_converter_agree_on_metadata_and_extra_events():
    from espnet2.speechlm.vllm.vllm_client.caption_format import caption_format_error
    from longcap.processing.convert import parse_turn
    two_events = '{"type":"speech","transcript":"hi"} <0, 1> {"type":"speech","transcript":"yo"} <1, 2>'
    assert caption_format_error(two_events) is not None
    lineage = '{"type":"speech","transcript":"hi"} <0, 1> {"ancestor": 0}'
    assert caption_format_error(lineage) is None
    events, issues = parse_turn(lineage, 10.0)
    assert issues == [] and events == [{"start": 10.0, "end": 11.0,
                                        "attributes": {"type": "speech", "transcript": "hi"}}]
    inline = '{"event_index": 3, "type":"speech","transcript":"hi"} <0, 1>'
    assert caption_format_error(inline) is None
    events, issues = parse_turn(inline, 0.0)
    assert issues == [] and "event_index" not in events[0]["attributes"]
    metadata_line = VALID + '\n{"ancestors": [0]}'
    assert caption_format_error(metadata_line) is None
    events, issues = parse_turn(metadata_line, 0.0)
    assert issues == [] and len(events) == 1
    plain, _ = parse_turn(VALID, 5.0)
    assert plain[0]["attributes"] == {"type": "speech", "transcript": "hello"}


def test_early_stopped_recording_exports_partial_prefix(tmp_path):
    from longcap.cli import main
    run = prepared_fixture(tmp_path / "input", seconds=47.5)
    chunks = json.loads((run / "inputs/chunks.jsonl").read_text())["chunks"]
    assert len(chunks) >= 2
    write_json(run / "raw/client.json", {"num_errors": 0, "num_written": 1, "num_examples": 1,
                                         "num_truncated": 0, "truncated_ids": [],
                                         "num_early_stopped": 1, "early_stopped_ids": ["sample"]})
    write_json(run / "raw/decode/inference_rank0/results.json", {"sample": [["assistant", "text", [VALID]]]})
    assert main(["convert", "--run", str(run)]) == 2
    status = json.loads((run / "predictions/conversion.jsonl").read_text())
    assert status["status"] == "partial" and status["issues"][0]["reason"] == "early_stop"
    assert json.loads((run / "predictions/predictions.jsonl").read_text())["events"]


def test_cli_reports_unreachable_service_without_traceback(tmp_path, capsys):
    from longcap.cli import main
    model = model_fixture(tmp_path / "model")
    audio = tmp_path / "clip.wav"
    sf.write(audio, np.zeros(16000), 16000)
    code = main(["caption", "--audio", str(audio), "--model", str(model),
                 "--service", "http://127.0.0.1:9", "--output", str(tmp_path / "out")])
    error = capsys.readouterr().err
    assert code == 1 and "cannot reach the LongCap service at http://127.0.0.1:9" in error
    assert "Traceback" not in error
    missing = main(["caption", "--audio", str(tmp_path / "nope.wav"), "--model", str(model),
                    "--output", str(tmp_path / "out2")])
    assert missing == 1 and "audio file not found" in capsys.readouterr().err


def test_manifest_without_duration_is_a_clear_error(tmp_path):
    audio = tmp_path / "clip.wav"
    sf.write(audio, np.zeros(16000), 16000)
    write_jsonl(tmp_path / "manifest.jsonl", [{"key": "x", "audio_path": str(audio)}])
    with pytest.raises(ValueError, match="missing duration"):
        prepare_inputs(tmp_path / "manifest.jsonl", tmp_path / "run", workers=1)

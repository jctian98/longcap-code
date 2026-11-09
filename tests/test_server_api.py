"""Real ASGI/HTTP upload contract with a fake engine; no GPU is used."""
import asyncio
import hashlib
import io
import threading
import time

import numpy as np
import pytest
import soundfile

pytest.importorskip("fastapi", reason="server API test requires the server extra")
import httpx

from espnet2.speechlm.vllm.vllm_client.protocol import GenerateResult
from espnet2.speechlm.vllm.vllm_server.audio_registry import AudioRegistry
from espnet2.speechlm.vllm.vllm_server.service import _build_app


def test_upload_generate_and_disabled_file_access(tmp_path):
    class Engine:
        requests = []
        async def generate(self, request):
            self.requests.append(request)
            return GenerateResult([300, 301], 2, "eos")
    engine = Engine()
    registry = AudioRegistry(str(tmp_path))
    config = {"model": "LongCap-35B-A3B", "io_threads": 0, "allow_audio_paths": False,
              "max_audio_per_prompt": 64, "checkpoint": {"fingerprint": "test"}}
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_build_app(engine, registry, config)), base_url="http://test") as client:
            assert (await client.get("/health")).json()["status"] == "ok"
            assert (await client.get("/models")).json()["checkpoint"]["fingerprint"] == "test"
            assert (await client.post("/audio", json={"path": "/private/server/file.wav"})).status_code == 403
            buffer = io.BytesIO()
            soundfile.write(buffer, np.zeros(16000), 16000, format="WAV", subtype="PCM_16")
            data = buffer.getvalue()
            reply = await client.post("/audio/upload", files={"file": ("chunk.wav", data, "audio/wav")})
            assert reply.status_code == 200
            meta = reply.json()
            assert meta["audio_id"] == hashlib.sha256(data).hexdigest()
            generated = await client.post("/generate", json={
                "prompt_token_ids": [1, 5, 8, 0, 2, 6, 7],
                "audio_segments": [{"audio_id": meta["audio_id"], "start_sample": 0, "end_sample": 16000}],
                "sampling": {"max_tokens": 8192, "repetition_penalty": 1.05},
            })
            assert generated.status_code == 200
            assert generated.json() == {"token_ids": [300, 301], "terminator": 2, "finish_reason": "eos"}
            assert engine.requests[0].repetition_penalty == 1.05
            assert engine.requests[0].audio[0][0].shape == (16000,)
            bad = await client.post("/generate", json={"prompt_token_ids": [1, 0], "audio_segments": []})
            assert bad.status_code == 422
    asyncio.run(scenario())


def test_audio_resolution_respects_worker_limit_and_shutdown():
    class Registry:
        active = peak = 0
        lock = threading.Lock()
        def slice_16k(self, *args):
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                time.sleep(0.03)
                return np.zeros(16)
            finally:
                with self.lock:
                    self.active -= 1
    class Engine:
        async def generate(self, request):
            return GenerateResult([300], 2, "eos")
    registry = Registry()
    app = _build_app(Engine(), registry, {"model": "test", "io_threads": 2})
    async def scenario():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                replies = await asyncio.gather(*(client.post("/generate", json={
                    "prompt_token_ids": [1, 5, 8, 0, 2, 6, 7],
                    "audio_segments": [{"audio_id": "fixture", "start_sample": 0, "end_sample": 16}],
                }) for _ in range(6)))
                assert all(reply.status_code == 200 for reply in replies)
    asyncio.run(scenario())
    assert registry.peak == 2 and registry.active == 0

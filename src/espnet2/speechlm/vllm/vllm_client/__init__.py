"""Pure-CPU client for the SpeechLM audio captioner (no vLLM dependency).

Everything here runs without CUDA or vLLM: special tokens, the system prompt,
the dialogue grammar, placeholder-length math, tokenization/detokenization, the
multi-hop driver, the wire protocol, and the HTTP client. The server
(``espnet2.speechlm.vllm.vllm_server``) imports *from* this package; the
dependency never points the other way. That one-way rule is the guardrail that
keeps token/prompt logic client-side and lets the client be installed and tested
on a CPU-only box.

Modules:
    constants  — token protocol (specials, vocab offsets)
    audio      — placeholder-length math + AudioSegment (deterministic, CPU)
    audio_io   — waveform loading / segment resolution (soundfile, CPU)
    prompt     — dialogue -> SpeechLM token ids (mirror of the torch reference)
    dialogue   — stateless multi-hop decode driver (engine-agnostic)
    protocol   — GenerateRequest / GenerateResult / EngineLike (wire contract)
    client     — HTTP client + HttpEngine + CaptionerClient
"""

__version__ = "0.2.0"

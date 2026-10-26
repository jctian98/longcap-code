"""SpeechLM vocabulary constants.

Mirrors ``SpeechLMJobTemplate._build_vocabulary`` (espnet2/speechlm/model/
speechlm/speechlm_job.py): 256 reserved special-token slots followed by the
HF text vocabulary, so a SpeechLM text token id is the HF token id + 256.

The captioner configs have a single stream and no discrete-audio IO, so the
full vocabulary is exactly ``NUM_SPECIAL_TOKENS + hf_vocab_size``.
"""

NUM_SPECIAL_TOKENS = 256

PAD = 0
BOS = 1
EOS = 2
EOT = 3
SYSTEM = 4
USER = 5
ASSISTANT = 6
TEXT = 7
AUDIO = 8
IMAGE = 9
VIDEO = 10
TOOLCALL = 11

ROLE_TOKENS = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}
MODALITY_TOKENS = {"text": TEXT, "audio": AUDIO, "image": IMAGE, "video": VIDEO}

TEXT_START = NUM_SPECIAL_TOKENS  # first text token id (HF id 0)

# The audio placeholder token. Continuous-audio content serializes to id 0
# (<|pad|>) per LLM position; the embedding at those positions is overwritten
# by adaptor(encoder(mel)), so the id itself is inert.
AUDIO_PLACEHOLDER = PAD

TERMINATORS = (EOS, EOT)

# Qwen3-Omni audio front-end (feature extractor) constants used by the
# placeholder-length formula. Values come from the encoder's
# preprocessor_config.json and are asserted against it at model-load time.
AUDIO_SAMPLE_RATE = 16000
AUDIO_HOP_LENGTH = 160

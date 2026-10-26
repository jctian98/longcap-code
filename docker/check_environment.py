"""Check the pinned runtime and plugin without initializing a GPU."""
import importlib.metadata as metadata
import json
import sys

EXPECTED = {
    'vllm': '0.26.0', 'torch': '2.11.0+cu130',
    'torchaudio': '2.11.0+cu130', 'transformers': '5.14.1',
    'tokenizers': '0.22.2', 'flashinfer-python': '0.6.14',
}
observed = {name: metadata.version(name) for name in EXPECTED}
assert observed == EXPECTED, {'expected': EXPECTED, 'observed': observed}
assert sys.version_info[:2] == (3, 12), sys.version
plugins = [p.name for p in metadata.entry_points(group='vllm.general_plugins')]
assert 'longcap_speechlm_captioner' in plugins, plugins
import torch
assert torch.version.cuda == '13.0', torch.version.cuda
print(json.dumps({'python': sys.version.split()[0], 'packages': observed,
                  'torch_cuda': torch.version.cuda, 'plugin_registered': True}))

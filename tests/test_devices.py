"""Device routing is simulated; actual transformer pooling is checked separately."""
from contextlib import contextmanager
import sys
import pytest
import torch
from interventions.local_model import LocalModel, model_device


def test_cpu_fallback_and_explicit_cuda_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert model_device(torch, 'auto') == ('cpu', torch.float32)
    assert model_device(torch, 'cpu') == ('cpu', torch.float32)
    with pytest.raises(RuntimeError, match='Enable a GPU runtime'):
        model_device(torch, 'cuda')
    # Failure precedes model/tokenizer downloads, even with an invalid model ID.
    with pytest.raises(RuntimeError, match='CUDA was requested'):
        LocalModel('not-a-model', device='cuda')


@pytest.mark.parametrize('requested,index,capability,supported,dtype', [
    ('auto', 0, 8, True, torch.bfloat16),
    ('cuda', 0, 7, True, torch.float16),  # T4: no native BF16, even if emulated
    ('cuda', 0, 8, False, torch.float16),
    ('cuda:1', 1, 8, True, torch.bfloat16),
])
def test_cuda_selection_and_precision(monkeypatch, requested, index, capability, supported, dtype):
    active = []
    @contextmanager
    def select_device(value):
        active.append(value)
        yield
        active.pop()
    def get_capability():
        assert active == [index]
        return capability, 0
    def bf16_supported():
        assert active == [index]
        return supported
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'device', select_device)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', get_capability)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', bf16_supported)
    assert model_device(torch, requested) == (f'cuda:{index}', dtype)
    assert not active


def test_invalid_gpu_index_and_unsupported_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    with pytest.raises(ValueError, match='does not exist'):
        model_device(torch, 'cuda:1')
    with pytest.raises(ValueError, match='Use auto'):
        model_device(torch, 'meta')


@pytest.mark.parametrize('override,expected', [(None, 'cpu'), ('cuda:1', 'cuda:1')])
def test_builder_cli_device_override(monkeypatch, override, expected):
    import data.build as builder
    config = {'model': {'device': 'cpu'}}
    seen = []
    monkeypatch.setattr(builder, 'read_config', lambda path: config)
    monkeypatch.setattr(builder, 'build', seen.append)
    argv = ['data.build', '--config', 'config.yaml']
    if override:
        argv += ['--device', override]
    monkeypatch.setattr(sys, 'argv', argv)
    builder.main()
    assert seen[0]['model']['device'] == expected

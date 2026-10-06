"""Tiny real transformer, random weights; mechanism checks, no quality claims."""
import numpy as np
import pytest


def tiny_backend():
    torch = pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from interventions.local_model import LocalModel
    torch.manual_seed(42)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2)
    backend = LocalModel.__new__(LocalModel)
    backend.torch = torch
    backend.device = 'cpu'
    backend.model = Qwen2ForCausalLM(config).eval()
    backend.layers = backend.model.model.layers
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {'input_ids': [ord(c) % 31 for c in text]}
    backend.tokenizer = Tokenizer()
    backend.prompt = lambda row: 'prompt'
    return backend


def test_features_capture_answer_only_and_cleanup():
    backend = tiny_backend()
    before = [len(layer._forward_hooks) for layer in backend.layers]
    mean, last = backend.features({}, 'ABC')
    assert [len(layer._forward_hooks) for layer in backend.layers] == before
    assert mean.shape == last.shape == (3, 16)
    ids = backend.torch.tensor([[ord(c) % 31 for c in 'promptABC']])
    with backend.torch.no_grad():
        states = backend.model(input_ids=ids, output_hidden_states=True).hidden_states
    # First two decoder block outputs are returned before the final norm.
    assert np.allclose(mean[0], states[1][0, 6:].mean(0).numpy(), atol=1e-6)
    assert np.allclose(last[1], states[2][0, -1].numpy(), atol=1e-6)


def test_steering_changes_activations_and_always_removes_hook():
    backend = tiny_backend()
    layer = backend.layers[-1]
    captured = []
    ids = backend.torch.tensor([[1, 2, 3]])
    def capture(module, args, output):
        captured.append((output[0] if isinstance(output, tuple) else output).detach().clone())
    handle = layer.register_forward_hook(capture)
    with backend.torch.no_grad():
        backend.model(input_ids=ids)
    handle.remove()
    with backend.steering(np.ones(16), strength=.5):
        handle = layer.register_forward_hook(capture)
        with backend.torch.no_grad():
            backend.model(input_ids=ids)
        handle.remove()
    assert backend.torch.allclose(captured[0][:, :-1], captured[1][:, :-1])
    assert backend.torch.allclose(captured[0][:, -1] + .5, captured[1][:, -1])
    with pytest.raises(RuntimeError):
        with backend.steering(np.ones(16)):
            raise RuntimeError('deliberate failure')
    assert not layer._forward_hooks

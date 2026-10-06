import numpy as np
from eval.cache import CachedLocalModel


def test_greedy_reuse_preserves_stochastic_seed_and_direction():
    class Backend:
        identity = {'model_id': 'fake'}
        def generate(self, row, **kwargs):
            return str(kwargs)
    cached = CachedLocalModel(Backend(), capacity=4)
    row = {'question': 'Q', 'family': 'factual'}
    cached.generate(row, seed=1)
    cached.generate(row, seed=2)
    assert cached.counts['generate']['computed'] == 1
    cached.generate(row, seed=1, sample=True)
    cached.generate(row, seed=2, sample=True)
    assert cached.counts['generate']['computed'] == 3
    cached.generate(row, direction=np.ones(3))
    cached.generate(row, direction=np.zeros(3))
    assert cached.counts['generate']['computed'] == 5

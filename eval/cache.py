"""Reuse identical frozen-model calls across evaluation arms.

Budgets still count requested operations per policy. Cache reuse saves evaluation
wall time and never transfers labels or outcomes into generation.
"""
import hashlib
from collections import OrderedDict
import numpy as np


def freeze(value):
    if isinstance(value, dict):
        return tuple((k, freeze(v)) for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    if isinstance(value, np.ndarray):
        return (value.shape, str(value.dtype), hashlib.sha256(value.tobytes()).hexdigest())
    return value


class CachedLocalModel:
    def __init__(self, backend, capacity=512):
        self.backend = backend
        self.identity = backend.identity
        self.capacity = capacity
        self.caches = {name: OrderedDict() for name in ('generate', 'features', 'fluency')}
        self.counts = {name: {'requested': 0, 'computed': 0} for name in self.caches}

    def _call(self, name, args, kwargs, cache_kwargs=None):
        cache = self.caches[name]
        key = freeze((args, kwargs if cache_kwargs is None else cache_kwargs))
        self.counts[name]['requested'] += 1
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        result = getattr(self.backend, name)(*args, **kwargs)
        self.counts[name]['computed'] += 1
        cache[key] = result
        if len(cache) > self.capacity:
            cache.popitem(last=False)
        return result

    def generate(self, row, **kwargs):
        cache_kwargs = dict(kwargs)
        if not kwargs.get('sample', False):
            cache_kwargs.pop('seed', None)
        return self._call('generate', (row,), kwargs, cache_kwargs)

    def features(self, row, answer):
        return self._call('features', (row, answer), {})

    def fluency(self, row, answer):
        return self._call('fluency', (row, answer), {})

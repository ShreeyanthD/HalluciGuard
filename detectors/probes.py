import hashlib
import re
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


class LinearProbe:
    """Regularized logistic probe; optional exponentiated-gradient group DRO."""
    def __init__(self, seed=0, dro=False, steps=300, lr=0.05, l2=0.01, eta=0.1):
        self.seed, self.dro, self.steps, self.lr, self.l2, self.eta = seed, dro, steps, lr, l2, eta

    def fit(self, x, y, groups):
        x, y, groups = np.asarray(x, dtype=float), np.asarray(y), np.asarray(groups)
        if set(y.tolist()) != {0, 1}:
            raise ValueError('Probe training requires both correctness classes')
        self.mean = x.mean(0)
        self.scale = np.maximum(x.std(0), 1e-5)
        z = (x - self.mean) / self.scale
        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(0, 0.001, z.shape[1])
        self.b = 0.0
        names = sorted(set(groups.tolist()))
        masks = [groups == g for g in names]
        # Bound the logistic gradient step using source-training covariance.
        # Flattened residual streams can have tens of thousands of coordinates.
        batches = [z[m] for m in masks] if self.dro else [z]
        curvature = max(float(np.linalg.eigvalsh(batch @ batch.T / len(batch))[-1])
                        for batch in batches)
        self.effective_lr = min(self.lr, 1 / (0.25 * (curvature + 1) + self.l2))
        q = np.ones(len(names)) / len(names)
        for _ in range(self.steps):
            logits = z @ self.w + self.b
            error = sigmoid(logits) - y
            if self.dro:
                losses = np.logaddexp(0, logits) - y * logits
                q *= np.exp(np.clip(self.eta * np.array([losses[m].mean() for m in masks]), -20, 20))
                q /= q.sum()
                weights = np.zeros(len(y))
                for weight, mask in zip(q, masks):
                    weights[mask] = weight / mask.sum()
            else:
                weights = np.full(len(y), 1 / len(y))
            self.w -= self.effective_lr * (z.T @ (weights * error) + self.l2 * self.w)
            self.b -= self.effective_lr * np.sum(weights * error)
        self.group_weights = dict(zip(names, q.tolist()))
        return self

    def predict(self, x):
        return sigmoid(((np.asarray(x) - self.mean) / self.scale) @ self.w + self.b)


def text_features(rows, dimensions=512):
    x = np.zeros((len(rows), dimensions))
    for i, row in enumerate(rows):
        for prefix, text in (('q:', row['question']), ('a:', row['answer'])):
            for token in re.findall(r'\w+', text.casefold()):
                key = int.from_bytes(hashlib.sha256((prefix + token).encode()).digest()[:8], 'big')
                x[i, key % dimensions] += 1
        x[i] /= max(1, np.linalg.norm(x[i]))
    return x


def drift_features(h):
    # Inspired transition baseline, not an exact DRIFT reproduction.
    delta = np.diff(h, axis=1)
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    cosine = np.sum(h[:, 1:] * h[:, :-1], -1, keepdims=True) / np.maximum(
        np.linalg.norm(h[:, 1:], axis=-1, keepdims=True) *
        np.linalg.norm(h[:, :-1], axis=-1, keepdims=True), 1e-8)
    return np.concatenate([delta, norm, cosine], -1).reshape(len(h), -1)


def make_features(name, rows, mean, last, layer=-1):
    if name in {'linear', 'single_dataset', 'pooled'}:
        return mean[:, layer, :]
    if name in {'group_dro', 'cross_layer_erm'}:
        return mean.reshape(len(mean), -1)
    if name == 'trajectory':
        return drift_features(last)
    if name == 'text_only':
        return text_features(rows)
    if name == 'clap_style':
        return last
    raise ValueError(f'Unknown detector {name}')

"""Engineered offline fixture. Never use as evidence of model performance."""
import argparse
from pathlib import Path
import numpy as np
from research_utils import write_json, write_jsonl


def create(path='artifacts/smoke_data'):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    rows, means, lasts = [], [], []
    for family in ('factual', 'math', 'code'):
        for split in ('train', 'calibration', 'test'):
            for i in range(12):
                label = i % 2
                ident = f'{family}-{split}-{i}'
                rows.append({'id': ident, 'question': f'{family} fixture question {ident}',
                             'family': family, 'dataset': f'{family}_fixture', 'split': split,
                             'group_id': ident, 'answer': 'candidate', 'label': label,
                             'outcome': 'INCORRECT' if label else 'CORRECT'})
                h = rng.normal(size=(4, 8))
                h[:, 0] += label * 1.5
                means.append(h)
                lasts.append(h + rng.normal(0, .1, h.shape))
    write_jsonl(path / 'records.jsonl', rows)
    np.savez_compressed(path / 'features.npz', mean=np.stack(means), last=np.stack(lasts))
    write_json(path / 'manifest.json', {'kind': 'engineered_fixture', 'seed': 42,
               'n': len(rows), 'warning': 'Artificial label-correlated features; plumbing validation only'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/smoke_data')
    create(parser.parse_args().output)

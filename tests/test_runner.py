import json
import numpy as np
import pytest
from eval.smoke import create
from eval.run import run
from research_utils import read_jsonl, write_jsonl, write_json


def test_detection_runner_source_only_and_fixture_guard(tmp_path):
    data = tmp_path / 'data'
    create(data)
    cfg = {'experiment': 'detection', 'data': str(data), 'output': str(tmp_path/'out'),
           'seeds': [1,2,3], 'detectors': ['pooled', 'group_dro'], 'steps': 20, 'bootstrap': 30}
    with pytest.raises(ValueError, match='allow_fixture'):
        run(cfg)
    cfg['allow_fixture'] = True
    report = run(cfg)
    assert len(report['results']) == 18 and not report['skipped']
    for result in report['results']:
        assert result['family'] not in result['thresholds']
    for path in (tmp_path/'out'/'probes').glob('*.json'):
        meta = json.loads(path.read_text())
        heldout = path.name.split('-')[0]
        assert all(not ident.startswith(heldout+'-') for ident in meta['train_ids'] + meta['calibration_ids'])


def test_model_reference_runs_when_probe_cannot_fit(tmp_path, monkeypatch):
    data = tmp_path/'data'
    create(data)
    rows = read_jsonl(data/'records.jsonl')
    for row in rows:
        row['label'], row['outcome'] = 1, 'INCORRECT'
        row['gold'] = ['right'] if row['family'] == 'factual' else '4'
        row['answer'] = 'wrong' if row['family'] != 'math' else '5'
        if row['family'] == 'code':
            row['tests'] = ['assert add(1,2)==3']
    write_jsonl(data/'records.jsonl', rows)
    identity = {'model_id': 'stub', 'revision': '1', 'feature_location': 'block',
                'pooling': 'mean', 'max_new_tokens': 1}
    write_json(data/'manifest.json', {'kind': 'model_generations', 'model': identity})
    class Backend:
        def __init__(self, **kwargs):
            self.identity = identity
        def fluency(self, row, answer):
            return 1.0
    import eval.run as runner
    monkeypatch.setattr(runner, 'LocalModel', Backend)
    cfg = {'experiment': 'mitigation', 'data': str(data), 'output': str(tmp_path/'out'),
           'model': {}, 'seeds': [1,2,3], 'detectors': ['group_dro'], 'steps': 20,
           'bootstrap': 30, 'test_limit': 2,
           'ablations': ['blackboard','base_model_alone','strongest_model_alone']}
    report = run(cfg)
    assert len(report['results']) == 9
    assert all(r['ablation'] == 'base_model_alone' and r['final']['accuracy'] == 0
               for r in report['results'])
    assert len(report['skipped']) == 18


def test_raw_detection_auroc_survives_unavailable_calibration(tmp_path):
    data = tmp_path/'data'
    create(data)
    rows = read_jsonl(data/'records.jsonl')
    for row in rows:
        if row['split'] == 'calibration':
            row['label'], row['outcome'] = 1, 'INCORRECT'
    write_jsonl(data/'records.jsonl', rows)
    cfg = {'experiment': 'detection', 'data': str(data), 'output': str(tmp_path/'out'),
           'seeds': [1,2,3], 'detectors': ['group_dro'], 'steps': 20, 'bootstrap': 30,
           'allow_fixture': True}
    report = run(cfg)
    assert len(report['results']) == 9 and not report['skipped']
    assert len(report['calibration_unavailable']) == 9
    assert all(r['thresholds'] is None and r['ood']['auroc'] is not None for r in report['results'])

import json
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from blackboard.research import Blackboard, calibrate, choose, route
from data.hygiene import assign_groups, fold
from data.labelers import grade, is_abstention, number
from detectors.probes import LinearProbe, drift_features, make_features, text_features
from eval.metrics import auroc, bootstrap, paired_reduction, outcome_metrics
from interventions.pipeline import Mitigator, public_row
from interventions.verification import run_tests


def test_auroc_ties_and_single_class():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 100)
    scores = rng.integers(0, 5, 100) / 5
    assert auroc(y, scores) == pytest.approx(roc_auc_score(y, scores))
    assert auroc([0, 0], [.1, .2]) is None
    result = bootstrap(y, scores, [str(i // 2) for i in range(100)], 50, 1)
    assert 0 <= result['ci95'][0] <= result['ci95'][1] <= 1


def test_labels_do_not_accept_embedded_gold_or_hedged_fabrication():
    row = {'id': 'q', 'family': 'factual', 'gold': ['Paris']}
    assert grade(row, 'The Paris.') == 'CORRECT'
    assert grade(row, 'Paris or London') == 'INCORRECT'
    assert grade(row, 'Unknown') == 'ABSTAINED'
    assert grade(row, "I don't know, but it is London") == 'INCORRECT'
    assert not is_abstention('Unknown Pleasures')
    assert number('steps 3 and 7') is None
    assert number('work 3 and 7. Final answer: 10') == 10
    assert number('#### 1,250') == 1250
    assert number(r'\boxed{1/2}') == number('0.5')
    assert grade({'id': 'm', 'family': 'math', 'gold': '42'}, 'Final answer: 42') == 'CORRECT'


def test_code_tests_correct_wrong_timeout_restricted():
    tests = ['assert add(1, 2) == 3', 'assert add(-1, 1) == 0']
    assert run_tests('def add(a, b):\n    return a+b', tests)['passed']
    assert not run_tests('def add(a, b):\n    return a-b', tests)['passed']
    assert not run_tests('import os\ndef add(a,b): return 3', tests)['passed']
    assert not run_tests('while True: pass', tests, timeout=.1)['passed']
    with pytest.raises(SyntaxError):
        run_tests('x=1', ['assert ???'])


def test_groups_transitively_join_entities_and_paraphrases():
    rows = [{'id': 'a', 'question': 'Who wrote Hamlet?', 'entities': ['Shakespeare']},
            {'id': 'b', 'question': 'Hamlet author?', 'entities': ['Shakespeare'], 'paraphrase_group': 'p'},
            {'id': 'c', 'question': 'Which writer?', 'paraphrase_group': 'p'},
            {'id': 'd', 'question': 'Compute nine times nine'}]
    assign_groups(rows)
    assert rows[0]['group_id'] == rows[1]['group_id'] == rows[2]['group_id']
    assert rows[3]['group_id'] != rows[0]['group_id']


def test_calibration_and_budget_change_dispatch():
    thresholds = calibrate([.1, .2, .3, .9], [0, 0, 0, 1], 0)
    assert thresholds['global'] == .9
    board = Blackboard('factual', {'d': .9}, thresholds, 1)
    assert choose(board) == 'steering'
    assert board.reserve('steering', 1)
    assert not board.reserve('steering_again', 1)
    assert board.compute_used == 1
    board = Blackboard('math', {'d': .9}, thresholds, 2)
    assert choose(board) == 'resampling'
    board.detector_scores['d'] = .1
    assert choose(board) == 'pass'
    board.abstained = True
    assert choose(board, True) == 'pass'
    assert board.final_decision == 'preserve_abstention'
    assert route('Implement a Python function') == 'code'
    assert route('How many apples are left?') == 'math'


def test_source_threshold_disables_gate_without_infinity():
    result = calibrate([1., 1.], [0, 1], 0)
    assert result['global'] > 1
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError):
        calibrate([.5], [1])


def test_dro_training_and_source_only_normalization():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 8))
    y = (x[:, 0] > 0).astype(int)
    probe = LinearProbe(seed=1, dro=True).fit(x, y, ['a']*20 + ['b']*20)
    assert auroc(y, probe.predict(x)) > .9
    before = probe.mean.copy()
    probe.predict(np.ones((2, 8))*1000)
    assert np.array_equal(before, probe.mean)
    assert sum(probe.group_weights.values()) == pytest.approx(1)


class FakeBackend:
    def __init__(self):
        self.calls = []
    def generate(self, row, **kwargs):
        assert 'gold' not in row and 'tests' not in row and 'label' not in row
        self.calls.append(kwargs)
        return 'Unknown' if kwargs.get('direction') is not None else 'Final answer: 4'


def test_blackboard_ablation_and_no_gold_leakage():
    backend = FakeBackend()
    mit = Mitigator(backend, lambda r, a: .5, {'global': .7}, np.ones(8), budget=8)
    row = {'id': 'x', 'family': 'math', 'question': 'Calculate 2+2', 'gold': '4', 'tests': ['assert False']}
    final, state = mit.run(row, 'Final answer: 5', .9, 1)
    assert final == 'Final answer: 4'
    assert state['task_family'] == 'math' and state['compute_used'] == 3
    final, state = mit.run(row, 'Final answer: 5', .9, 1, 'no_blackboard')
    assert final == 'Unknown' and state['final_decision'] == 'fixed_steering'
    final, state = mit.run(row, 'Final answer: 5', .1, 1)
    assert final == 'Final answer: 5' and state['compute_used'] == 0
    final, state = mit.run(row, 'Unknown', .99, 1, 'no_gating')
    assert final == 'Unknown' and state['compute_used'] == 0
    assert 'gold' not in public_row(row)


def test_zero_budget_and_ablation_validation():
    mit = Mitigator(FakeBackend(), lambda r,a: .5, {'global': .7}, np.ones(8), budget=0)
    row = {'id': 'f', 'family': 'factual', 'question': 'Who wrote Hamlet?'}
    final, state = mit.run(row, 'Marlowe', .9, 1)
    assert final == 'Marlowe' and state['final_decision'] == 'pass_budget_exhausted'
    with pytest.raises(ValueError, match='strongest'):
        mit.run(row, 'Marlowe', .9, 1, 'strongest_model_alone')
    with pytest.raises(ValueError, match='ablation'):
        mit.run(row, 'Marlowe', .9, 1, 'nonsense')


def test_fold_excludes_heldout_labels_and_overlap(tmp_path):
    from eval.smoke import create
    from research_utils import read_jsonl
    create(tmp_path)
    rows = read_jsonl(tmp_path / 'records.jsonl')
    train, cal, test, idtest = fold(rows, 'factual')
    assert all(rows[i]['family'] != 'factual' for i in train + cal + idtest)
    assert all(rows[i]['family'] == 'factual' for i in test)
    rows[train[0]]['group_id'] = rows[test[0]]['group_id']
    train2, _, _, _ = fold(rows, 'factual')
    assert train[0] not in train2


def test_metric_accounting_and_paired_interval():
    base = ['INCORRECT', 'CORRECT', 'INCORRECT', 'ABSTAINED']
    final = ['CORRECT', 'CORRECT', 'ABSTAINED', 'ABSTAINED']
    metrics = outcome_metrics(final)
    assert metrics['accuracy'] + metrics['abstention_rate'] + metrics['error_rate'] == 1
    reduction = paired_reduction(base, final, ['a','b','c','d'], 100, 1)
    assert reduction['absolute_error_reduction'] == .5


def test_attention_probe_trains_and_predicts():
    pytest.importorskip('torch')
    from detectors.attention import AttentionProbe
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 4, 8)).astype(np.float32)
    y = (x[:, :, 0].mean(1) > 0).astype(int)
    probe = AttentionProbe(seed=1, steps=20).fit(x, y, ['a']*20)
    scores = probe.predict(x)
    assert scores.shape == (20,) and np.isfinite(scores).all()

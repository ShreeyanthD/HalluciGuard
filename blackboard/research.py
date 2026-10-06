"""Research blackboard: typed decision state, independent of API agents."""
from dataclasses import dataclass, field, asdict
import math
import re
import numpy as np


@dataclass
class Blackboard:
    task_family: str
    detector_scores: dict
    calibrated_thresholds: dict
    compute_budget: int
    interventions_tried: list = field(default_factory=list)
    compute_used: int = 0
    final_decision: str = 'pending'
    risk: float = 0.0
    abstained: bool = False

    def reserve(self, action, cost):
        if cost < 0:
            raise ValueError('Cost must be nonnegative')
        if action in self.interventions_tried or self.compute_used + cost > self.compute_budget:
            self.final_decision = 'pass_budget_exhausted'
            return False
        self.interventions_tried.append(action)
        self.compute_used += cost
        self.final_decision = action
        return True

    def snapshot(self):
        return asdict(self)


def route(question, task_hint=None):
    if task_hint is not None:
        if task_hint not in {'factual', 'math', 'code'}:
            raise ValueError('Invalid task hint')
        return task_hint
    if re.search(r'(?i)\b(python|function|implement|program|code)\b|def \w+\(', question):
        return 'code'
    if re.search(r'(?i)\b(calculate|equation|solve|how many|sum|product|percent)\b|\d+\s*[+*/=]\s*\d+', question):
        return 'math'
    return 'factual'


def calibrate(scores, labels, max_false_positive_rate=0.1):
    """Source-calibration empirical FPR bound. No held-out family labels."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    if not 0 <= max_false_positive_rate <= 1:
        raise ValueError('FPR target must be between zero and one')
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Scores must be finite probabilities')
    correct = scores[labels == 0]
    if not len(correct):
        raise ValueError('Calibration requires correct source answers')
    # A threshold just above one disables gating without infinities in JSON.
    for threshold in sorted(set(scores.tolist()) | {1.000001}):
        if float(np.mean(correct >= threshold)) <= max_false_positive_rate:
            return {'global': float(threshold), 'source_correct_n': len(correct),
                    'empirical_source_fpr': float(np.mean(correct >= threshold)),
                    'target_fpr': max_false_positive_rate,
                    'heldout_threshold_policy': 'global_source_only'}
    raise RuntimeError('Unreachable calibration state')


def choose(board, bypass_gate=False):
    if board.compute_budget < 0:
        raise ValueError('Budget must be nonnegative')
    if not board.detector_scores:
        raise ValueError('Missing detector scores')
    values = list(board.detector_scores.values())
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in values):
        raise ValueError('Detector scores must be finite probabilities')
    board.risk = max(values)
    if board.abstained:
        board.final_decision = 'preserve_abstention'
        return 'pass'
    threshold = board.calibrated_thresholds.get(board.task_family,
                                               board.calibrated_thresholds['global'])
    if not bypass_gate and board.risk < threshold:
        board.final_decision = 'pass_low_risk'
        return 'pass'
    return 'steering' if board.task_family == 'factual' else 'resampling'

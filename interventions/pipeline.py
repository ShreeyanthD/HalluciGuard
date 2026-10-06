"""Budgeted same-model mitigation. Gold answers are never accepted here."""
from collections import Counter
import numpy as np
from blackboard.research import Blackboard, route, choose
from data.labelers import is_abstention, number, normalize
from interventions.verification import run_tests

MODES = {'blackboard', 'no_blackboard', 'always_on_steering', 'no_gating',
         'detector_only', 'base_model_alone', 'strongest_model_alone'}


def fit_direction(backend, train_rows, train_mean, layer=-1, limit=32):
    # Verbal-abstention contrast from SOURCE TRAINING correct completions.
    # This is a hypothesis about a direction, not a guarantee of truthfulness.
    selected = [i for i, r in enumerate(train_rows) if r['outcome'] == 'CORRECT'][:limit]
    if not selected:
        raise ValueError('Need correct source training completions for steering contrast')
    deltas = []
    for i in selected:
        safe = public_row(train_rows[i])
        abstention, _ = backend.features(safe, 'Unknown')
        deltas.append(abstention[layer] - train_mean[i, layer])
    return np.mean(deltas, axis=0), [train_rows[i]['id'] for i in selected]


def public_row(row):
    # Private correctness tests, golds, dataset label and split never reach an
    # intervention. Public tests must be independently supplied in the task.
    return {k: row[k] for k in ('id', 'question', 'family', 'public_tests') if k in row}


class Mitigator:
    def __init__(self, backend, score_answer, thresholds, direction=None,
                 layer=-1, strength=1.0, budget=5, samples=3,
                 use_task_hints=False, strongest_backend=None):
        self.backend, self.score_answer, self.thresholds = backend, score_answer, thresholds
        self.direction, self.layer, self.strength = direction, layer, strength
        self.budget, self.samples, self.use_task_hints = budget, samples, use_task_hints
        if budget < 0 or samples < 0:
            raise ValueError('Budget and sample count must be nonnegative')
        self.strongest_backend = strongest_backend

    def run(self, row, answer, score, seed, mode='blackboard'):
        if mode not in MODES:
            raise ValueError(f'Unknown ablation {mode}')
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Score must be a finite probability')
        safe = public_row(row)
        family = route(row['question'], row['family'] if self.use_task_hints else None)
        board = Blackboard(family, {'probe': score}, self.thresholds, self.budget,
                           abstained=is_abstention(answer), risk=score)
        if mode in {'detector_only', 'base_model_alone'}:
            board.final_decision = mode
            return answer, board.snapshot()
        if mode == 'strongest_model_alone':
            if self.strongest_backend is None:
                raise ValueError('strongest_model_alone needs an explicit strongest_model config')
            final = self.strongest_backend.generate(safe, seed=seed)
            board.final_decision = mode
            board.compute_used = 1
            return final, board.snapshot()
        if mode == 'no_blackboard':
            # Fixed pipeline: global gate then steering for every task. No
            # arbiter/task-specific dispatch or iterative state decisions.
            if board.abstained or score < self.thresholds['global'] or self.budget < 1:
                board.final_decision = 'fixed_pass'
                return answer, board.snapshot()
            final = self._steer(safe, seed)
            board.final_decision = 'fixed_steering'
            board.compute_used = 1
            board.interventions_tried = ['fixed_steering']
            return final, board.snapshot()
        if mode == 'always_on_steering':
            if board.abstained:
                board.final_decision = 'preserve_abstention'
            action = 'pass' if board.abstained else 'steering'
        else:
            action = choose(board, bypass_gate=mode == 'no_gating')
        if action == 'pass':
            return answer, board.snapshot()
        if action == 'steering':
            if not board.reserve(action, 1):
                return answer, board.snapshot()
            return self._steer(safe, seed), board.snapshot()
        candidates = [answer]
        for i in range(self.samples):
            if not board.reserve(f'resample_{i}', 1):
                break
            candidates.append(self.backend.generate(safe, seed=seed+i, sample=True))
        if len(candidates) == 1:
            return answer, board.snapshot()
        if family == 'code' and safe.get('public_tests'):
            for i, candidate in enumerate(candidates):
                if not board.reserve(f'public_verify_{i}', 1):
                    break
                if run_tests(candidate, safe['public_tests'])['passed']:
                    board.final_decision = 'public_tests_passed'
                    return candidate, board.snapshot()
        elif family == 'math':
            keys = [number(c) for c in candidates]
            counts = Counter(k for k in keys if k is not None)
            if counts:
                winning = counts.most_common(1)[0][0]
                board.final_decision = 'self_consistency'
                return candidates[keys.index(winning)], board.snapshot()
        # Same-model verification without a gold-answer oracle.
        if board.reserve('verification_pass', 1):
            candidates.append(self.backend.generate(safe, seed=seed+self.samples,
                                                    revision_answer=answer))
        costs = len(candidates)
        if board.reserve('candidate_scoring', costs):
            scores = [self.score_answer(safe, c) for c in candidates]
            board.final_decision = 'lowest_detected_risk'
            return candidates[int(np.argmin(scores))], board.snapshot()
        # Consistent output is still not proof of correctness.
        keys = [normalize(c) for c in candidates]
        winning = Counter(keys).most_common(1)[0][0]
        board.final_decision = 'self_consistency_text'
        return candidates[keys.index(winning)], board.snapshot()

    def _steer(self, row, seed):
        if self.direction is None:
            raise ValueError('Steering requires a source-trained activation direction')
        return self.backend.generate(row, seed=seed, direction=self.direction,
                                     layer=self.layer, strength=self.strength)

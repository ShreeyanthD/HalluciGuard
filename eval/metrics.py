import numpy as np


def auroc(labels, scores):
    y, scores = np.asarray(labels), np.asarray(scores)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite detector scores')
    positive, negative = scores[y == 1], scores[y == 0]
    if not len(positive) or not len(negative):
        return None
    # Rank sum with ties; O(n log n), no pairwise matrix.
    order = np.argsort(scores, kind='stable')
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[y == 1].sum() - len(positive)*(len(positive)+1)/2) /
                 (len(positive)*len(negative)))


def bootstrap(labels, scores, groups, iterations=1000, seed=0):
    y, scores, groups = np.asarray(labels), np.asarray(scores), np.asarray(groups)
    rng = np.random.default_rng(seed)
    names = np.unique(groups)
    members = [np.flatnonzero(groups == name) for name in names]
    values = []
    for _ in range(iterations):
        idx = np.concatenate([members[j] for j in rng.integers(0, len(names), len(names))])
        value = auroc(y[idx], scores[idx])
        if value is not None:
            values.append(value)
    return {'auroc': auroc(y, scores), 'ci95': np.quantile(values, [.025, .975]).tolist() if values else None,
            'bootstrap_valid': len(values), 'n': len(y), 'positive_n': int(y.sum()),
            'bootstrap_invalid': iterations - len(values),
            'ci_warning': 'Tiny sample: percentile CI can be degenerate and unreliable' if len(y) < 30 else None,
            'unit': 'connected_question_group'}


def paired_reduction(base, final, groups, iterations=1000, seed=0):
    delta = (np.asarray(base) == 'INCORRECT').astype(float) - (np.asarray(final) == 'INCORRECT').astype(float)
    groups = np.asarray(groups)
    names = np.unique(groups)
    members = [np.flatnonzero(groups == name) for name in names]
    rng = np.random.default_rng(seed)
    values = [float(delta[np.concatenate([members[j] for j in rng.integers(0, len(names), len(names))])].mean())
              for _ in range(iterations)]
    return {'absolute_error_reduction': float(delta.mean()),
            'ci95': np.quantile(values, [.025, .975]).tolist(), 'unit': 'paired_connected_question_group'}


def outcome_metrics(outcomes, nlls=None):
    outcomes = np.asarray(outcomes)
    correct = int(np.sum(outcomes == 'CORRECT'))
    abstained = int(np.sum(outcomes == 'ABSTAINED'))
    incorrect = int(np.sum(outcomes == 'INCORRECT'))
    n = len(outcomes)
    values = [x for x in (nlls or []) if x is not None]
    return {'n': n, 'accuracy': correct/n, 'abstention_rate': abstained/n,
            'error_rate': incorrect/n, 'coverage': (n-abstained)/n,
            'selective_accuracy': correct/(n-abstained) if n > abstained else None,
            'mean_answer_nll': float(np.mean(values)) if values else None,
            'fluency_metric': 'base_model_teacher_forced_token_nll_proxy' if values else 'not_measured'}

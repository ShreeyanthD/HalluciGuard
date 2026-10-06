"""python -m eval.run --config configs/detect_loto.yaml"""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from blackboard.research import calibrate
from data.hygiene import fold
from data.labelers import grade
from detectors.probes import LinearProbe, make_features
from detectors.attention import AttentionProbe
from eval.metrics import bootstrap, outcome_metrics, paired_reduction
from eval.cache import CachedLocalModel
from interventions.local_model import LocalModel
from interventions.pipeline import Mitigator, fit_direction, public_row
from research_utils import read_config, read_jsonl, write_json, write_jsonl, provenance


def fit_detector(name, features, rows, train, seed, config):
    train = [i for i in train if rows[i]['label'] is not None]
    if name == 'linear':
        # Legacy single-family probe: source family fixed before evaluation.
        family = config.get('linear_source_family', sorted({rows[i]['family'] for i in train})[0])
        train = [i for i in train if rows[i]['family'] == family]
    if name == 'single_dataset':
        dataset = config.get('single_dataset', sorted({rows[i]['dataset'] for i in train})[0])
        train = [i for i in train if rows[i]['dataset'] == dataset]
    if len({rows[i]['label'] for i in train}) < 2:
        raise ValueError(f'{name}: source training split does not contain both classes')
    kwargs = {'seed': seed, 'steps': config.get('steps', 300)}
    if name == 'clap_style':
        probe = AttentionProbe(seed=seed, steps=config.get('attention_steps', 100))
    else:
        probe = LinearProbe(dro=name == 'group_dro', **kwargs)
    probe.fit(features[train], [rows[i]['label'] for i in train], [rows[i]['family'] for i in train])
    return probe, train


def persist_probe(path, probe):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(probe, AttentionProbe):
        import torch
        torch.save(probe.net.state_dict(), str(path) + '.pt')
        np.savez_compressed(str(path) + '.npz', mean=probe.mean, scale=probe.scale)
    else:
        np.savez_compressed(str(path) + '.npz', mean=probe.mean, scale=probe.scale, w=probe.w, b=probe.b)


def run(config):
    dataset = Path(config['data'])
    rows = read_jsonl(dataset / 'records.jsonl')
    import json
    manifest = json.loads((dataset / 'manifest.json').read_text())
    if manifest.get('complete') is False:
        raise ValueError('Dataset generation is incomplete; finish building before evaluation')
    cache = np.load(dataset / 'features.npz', allow_pickle=False)
    mean, last = cache['mean'], cache['last']
    if mean.shape != last.shape or mean.ndim != 3 or len(mean) != len(rows):
        raise ValueError('Feature cache must align with rows as [example, layer, hidden]')
    if not np.isfinite(mean).all() or not np.isfinite(last).all():
        raise ValueError('Nonfinite feature cache')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate record IDs')
    for row in rows:
        if row['label'] not in {None, 0, 1}:
            raise ValueError('Expected labels 0/1/null')
    if len(set(config['seeds'])) < 3:
        raise ValueError('Research protocol requires at least three distinct probe seeds')
    kind = manifest['kind']
    if kind != 'model_generations' and not config.get('allow_fixture', False):
        raise ValueError('Non-model data requires explicit allow_fixture: true')
    if config['experiment'] == 'mitigation' and kind != 'model_generations':
        raise ValueError('Synthetic features cannot evaluate real mitigation')
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    results, predictions, decisions, skips, calibration_notes = [], [], [], [], []
    backend = None
    strongest = None
    if config['experiment'] == 'mitigation':
        backend = CachedLocalModel(LocalModel(**config['model']))
        if any(backend.identity[k] != manifest['model'][k]
               for k in ('model_id', 'revision', 'feature_location', 'pooling', 'max_new_tokens')):
            raise ValueError('Mitigation model or feature protocol does not match the generation cache')
        if config.get('strongest_model'):
            strongest = CachedLocalModel(LocalModel(**config['strongest_model']))
    for family in config.get('families', ['factual', 'math', 'code']):
        train, cal, test, idtest = fold(rows, family)
        if config.get('test_limit') is not None:
            if config['test_limit'] < 1:
                raise ValueError('test_limit must be positive')
            # Fixed prefix for compute-limited pilots, selected without labels.
            test = test[:config['test_limit']]
        for seed in config['seeds']:
            if backend is not None:
                # Model-only references remain measurable even if no probe fits.
                base_labels = [rows[i]['outcome'] for i in test]
                base_nlls = [backend.fluency(public_row(rows[i]), rows[i]['answer']) for i in test]
                for mode in ('base_model_alone', 'strongest_model_alone'):
                    if mode not in config['ablations']:
                        continue
                    if mode == 'strongest_model_alone' and strongest is None:
                        skips.append({'family': family, 'seed': seed, 'ablation': mode,
                                      'reason': 'No explicitly configured strongest model'})
                        continue
                    answers = ([rows[i]['answer'] for i in test] if mode == 'base_model_alone' else
                               [strongest.generate(public_row(rows[i]), seed=seed*100000+j) for j, i in enumerate(test)])
                    labels = [grade(rows[i], a) for i, a in zip(test, answers)]
                    nlls = [backend.fluency(public_row(rows[i]), a) for i, a in zip(test, answers)]
                    results.append({'family': family, 'seed': seed, 'detector': 'none', 'ablation': mode,
                                    'base': outcome_metrics(base_labels, base_nlls),
                                    'final': outcome_metrics(labels, nlls),
                                    'mean_additional_operations': float(mode == 'strongest_model_alone'),
                                    **paired_reduction(base_labels, labels,
                                        [rows[i]['group_id'] for i in test], config.get('bootstrap', 1000), seed)})
                    decisions.extend({'id': rows[i]['id'], 'family': family, 'seed': seed,
                                      'detector': 'none', 'ablation': mode, 'base': rows[i]['answer'],
                                      'final': a, 'base_outcome': rows[i]['outcome'], 'final_outcome': label,
                                      'blackboard': None} for i, a, label in zip(test, answers, labels))
            names = config.get('detectors', ['group_dro'])
            for name in names:
                x = make_features(name, rows, mean, last, config.get('layer', -1))
                try:
                    probe, fit_indices = fit_detector(name, x, rows, train, seed, config)
                except ValueError as exc:
                    skips.append({'family': family, 'seed': seed, 'detector': name, 'reason': str(exc)})
                    continue
                calibration_error = None
                try:
                    cal_indices = [i for i in cal if rows[i]['label'] is not None]
                    detector_idtest = list(idtest)
                    if name in {'linear', 'single_dataset'}:
                        field = 'family' if name == 'linear' else 'dataset'
                        fit_domains = {rows[i][field] for i in fit_indices}
                        cal_indices = [i for i in cal_indices if rows[i][field] in fit_domains]
                        detector_idtest = [i for i in idtest if rows[i][field] in fit_domains]
                    threshold = calibrate(probe.predict(x[cal_indices]),
                                          [rows[i]['label'] for i in cal_indices],
                                          config.get('max_fpr', 0.1))
                    threshold['source_family_calibration'] = {}
                    for source_family in sorted({rows[i]['family'] for i in cal_indices}):
                        subset = [i for i in cal_indices if rows[i]['family'] == source_family]
                        if any(rows[i]['label'] == 0 for i in subset):
                            local = calibrate(probe.predict(x[subset]), [rows[i]['label'] for i in subset],
                                              config.get('max_fpr', 0.1))
                            threshold[source_family] = local['global']
                            threshold['source_family_calibration'][source_family] = local
                except ValueError as exc:
                    threshold = None
                    calibration_error = str(exc)
                    calibration_notes.append({'family': family, 'seed': seed,
                        'detector': name + ' calibration', 'reason': str(exc)})
                stem = output / 'probes' / f'{family}-{seed}-{name}'
                persist_probe(stem, probe)
                write_json(str(stem) + '.json', {'train_ids': [rows[i]['id'] for i in fit_indices],
                           'calibration_ids': [rows[i]['id'] for i in cal_indices],
                           'thresholds': threshold, 'kind': kind,
                           'calibration_error': calibration_error,
                           'feature_shape': list(x.shape[1:]), 'seed': seed, 'detector': name,
                           'layer': config.get('layer', -1),
                           'group_weights': getattr(probe, 'group_weights', None)})
                if config['experiment'] == 'detection':
                    stats = {}
                    for label, indices in (('ood', test), ('id', detector_idtest)):
                        indices = [i for i in indices if rows[i]['label'] is not None]
                        if not indices:
                            stats[label] = {'auroc': None, 'ci95': None, 'n': 0}
                            continue
                        scores = probe.predict(x[indices])
                        stats[label] = bootstrap([rows[i]['label'] for i in indices], scores,
                                                 [rows[i]['group_id'] for i in indices],
                                                 config.get('bootstrap', 1000), seed)
                        predictions.extend({'id': rows[i]['id'], 'group_id': rows[i]['group_id'],
                                            'heldout': family, 'split': label, 'seed': seed,
                                            'detector': name, 'label': rows[i]['label'], 'score': float(score)}
                                           for i, score in zip(indices, scores))
                    gap = (stats['id']['auroc'] - stats['ood']['auroc']
                           if stats['id']['auroc'] is not None and stats['ood']['auroc'] is not None else None)
                    results.append({'family': family, 'seed': seed, 'detector': name,
                                    **stats, 'id_minus_ood': gap, 'thresholds': threshold})
                elif config['experiment'] == 'mitigation':
                    direction, direction_ids = fit_direction(backend, [rows[i] for i in train], mean[train],
                                                             config.get('steering_layer', -1),
                                                             config.get('direction_limit', 32))
                    np.save(str(stem) + '-direction.npy', direction)
                    write_json(str(stem) + '-direction.json', {'train_ids': direction_ids,
                               'layer': config.get('steering_layer', -1),
                               'strength': config.get('steering_strength', 1.0)})
                    def score_answer(row, answer):
                        m, l = backend.features(row, answer)
                        xx = make_features(name, [{**row, 'answer': answer}], m[None], l[None], config.get('layer', -1))
                        return float(probe.predict(xx)[0])
                    # This placeholder is used only by ungated policies; gated
                    # policies are skipped below when calibration is unavailable.
                    policy_thresholds = threshold or {'global': 1.000001, 'calibration_available': False}
                    mitigator = Mitigator(backend, score_answer, policy_thresholds, direction,
                        layer=config.get('steering_layer', -1), strength=config.get('steering_strength', 1.0),
                        budget=config.get('budget', 5), samples=config.get('samples', 3),
                        use_task_hints=config.get('use_task_hints', False), strongest_backend=strongest)
                    base_outcomes = [rows[i]['outcome'] for i in test]
                    groups = [rows[i]['group_id'] for i in test]
                    base_nll = [backend.fluency(public_row(rows[i]), rows[i]['answer']) for i in test]
                    for mode in config['ablations']:
                        if mode in {'base_model_alone', 'strongest_model_alone'}:
                            continue
                        if threshold is None and mode in {'blackboard', 'no_blackboard'}:
                            skips.append({'family': family, 'seed': seed, 'detector': name,
                                'ablation': mode, 'reason': 'Gated policy unavailable: ' + calibration_error})
                            continue
                        outcomes, nlls, used = [], [], []
                        for j, i in enumerate(test):
                            initial_score = float(probe.predict(x[i:i+1])[0])
                            final, state = mitigator.run(rows[i], rows[i]['answer'], initial_score,
                                                        seed*100000 + j, mode)
                            outcome = grade(rows[i], final)
                            outcomes.append(outcome)
                            nlls.append(backend.fluency(public_row(rows[i]), final))
                            used.append(state['compute_used'])
                            decisions.append({'id': rows[i]['id'], 'family': family, 'seed': seed,
                                              'detector': name, 'ablation': mode, 'base': rows[i]['answer'],
                                              'final': final, 'base_outcome': rows[i]['outcome'],
                                              'final_outcome': outcome, 'blackboard': state})
                        results.append({'family': family, 'seed': seed, 'detector': name, 'ablation': mode,
                                        'base': outcome_metrics(base_outcomes, base_nll),
                                        'final': outcome_metrics(outcomes, nlls),
                                        'mean_additional_operations': float(np.mean(used)),
                                        **paired_reduction(base_outcomes, outcomes, groups,
                                                           config.get('bootstrap', 1000), seed)})
                else:
                    raise ValueError('Experiment must be detection or mitigation')
                print(f"{family} seed={seed} detector={name} finished", flush=True)
    metadata = {**provenance(config, [dataset/'records.jsonl', dataset/'features.npz', dataset/'manifest.json']),
                'kind': kind, 'data_manifest': manifest, 'results': results, 'skipped': skips,
                'calibration_unavailable': calibration_notes,
                'counts': dict(Counter(r['family'] + ':' + r['outcome'] for r in rows)),
                'seeds_note': 'Probe/intervention seeds; cached initial generations use data.seed',
                'model_cache_calls': backend.counts if backend is not None else None,
                'compute_note': 'Budgets count requested operations; identical frozen-model calls may be reused across arms',
                'ci_note': 'Per-seed 95% group-bootstrap intervals; no claim to estimate across-seed uncertainty'}
    write_json(output / 'metrics.json', metadata)
    write_jsonl(output / 'predictions.jsonl', predictions)
    write_jsonl(output / 'decisions.jsonl', decisions)
    write_table(output / 'results.md', config['experiment'], kind, results, skips + calibration_notes, config)
    print(f'Saved {len(results)} result rows; {len(skips)} explicit skips to {output}', flush=True)
    return metadata


def fmt(value):
    return 'N/A' if value is None else f'{value:.3f}'


def write_table(path, experiment, kind, rows, skips, config=None):
    lines = ['# HalluciGuard results', '', f'Data kind: **{kind}**.', '']
    if kind != 'model_generations':
        lines += ['**Engineered fixture smoke test only. These are not LLM research results.**', '']
    lines += ['Intervals are per-seed, question-group bootstrap 95% CIs. Single-class AUROC is N/A.', '']
    lines += ['Seeds share cached initial generations. These intervals do not estimate across-seed uncertainty.', '']
    lines += ['On tiny samples percentile intervals can collapse to a point; this does not imply population certainty. '
              'AUROC bootstrap draws with only one class are skipped and counted in metrics.json.', '']
    if config and config.get('test_limit'):
        lines += [f"**Compute-limited pilot: at most {config['test_limit']} test questions per held-out family. "
                  'This cannot establish generalization or intervention effectiveness.**', '']
    if experiment == 'detection':
        lines += ['| Held out | Seed | Detector | ID AUROC | OOD AUROC [CI] | ID−OOD | OOD n |',
                  '|---|---:|---|---:|---|---:|---:|']
        for r in rows:
            ci = r['ood']['ci95']
            interval = f'[{fmt(ci[0])}, {fmt(ci[1])}]' if ci else 'N/A'
            lines.append(f"| {r['family']} | {r['seed']} | {r['detector']} | {fmt(r['id']['auroc'])} | "
                         f"{fmt(r['ood']['auroc'])} {interval} | {fmt(r['id_minus_ood'])} | {r['ood']['n']} |")
    else:
        lines += ['| Held out | Seed | Detector | Ablation | n | Accuracy | Abstention | Error | Answer NLL | Error reduction [CI] |',
                  '|---|---:|---|---|---:|---:|---:|---:|---:|---|']
        for r in rows:
            final = r['final']
            ci = r['ci95']
            lines.append(f"| {r['family']} | {r['seed']} | {r['detector']} | {r['ablation']} | {final['n']} | {fmt(final['accuracy'])} | "
                         f"{fmt(final['abstention_rate'])} | {fmt(final['error_rate'])} | "
                         f"{fmt(final['mean_answer_nll'])} | {fmt(r['absolute_error_reduction'])} "
                         f"[{fmt(ci[0])}, {fmt(ci[1])}] |")
    if skips:
        lines += ['', '## Unavailable measurements', '']
        lines += [f"- {r.get('family')}, seed {r.get('seed')}, "
                  f"{r.get('detector', '')} {r.get('ablation', '')}: {r['reason']}" for r in skips]
    Path(path).write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(read_config(args.config))


if __name__ == '__main__':
    main()

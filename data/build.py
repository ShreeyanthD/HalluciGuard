"""python -m data.build --config configs/data.yaml"""
import argparse
import ast
from pathlib import Path
import numpy as np
from data.hygiene import assign_groups, assign_splits
from data.labelers import grade
from interventions.local_model import LocalModel
from research_utils import read_config, read_jsonl, write_json, write_jsonl, provenance


def source_rows(source):
    if 'path' in source:
        rows = read_jsonl(source['path'])
    else:
        from datasets import load_dataset
        rows = load_dataset(source['dataset'], source.get('subset'), split=source['split'],
                            revision=source.get('revision'))
    output = []
    for i, raw in enumerate(rows):
        if i >= source.get('limit', len(rows)):
            break
        family = source['family']
        adapter = source.get('adapter', 'canonical')
        if adapter == 'truthfulqa':
            row = {'question': raw['question'], 'gold': raw['correct_answers']}
        elif adapter == 'gsm8k':
            row = {'question': raw['question'], 'gold': raw['answer']}
        elif adapter == 'mbpp':
            prompt = raw.get('prompt', raw.get('text'))
            functions = [n for n in ast.parse(raw['code']).body if isinstance(n, ast.FunctionDef)]
            # Export only the ABI, never reference implementation or expected
            # test results. MBPP natural-language prompts omit the function name.
            if functions:
                names = {f.name: f for f in functions}
                test_calls = [n.func.id for text in raw['test_list'] for n in ast.walk(ast.parse(text))
                              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                              and n.func.id in names]
                fn = names[test_calls[0]] if test_calls else functions[-1]
                prompt += f'\nRequired function signature: {fn.name}({ast.unparse(fn.args)}).'
            tests = list(raw.get('test_imports', [])) + list(raw['test_list'])
            row = {'question': prompt, 'tests': tests, 'gold': raw['code']}
        elif adapter == 'canonical':
            row = dict(raw)
        else:
            raise ValueError(f'Unknown adapter {adapter}')
        row.update(id=str(row.get('id', f"{source.get('dataset', source.get('path'))}:{i}")),
                   family=family, dataset=source.get('name', source.get('dataset', source.get('path'))))
        if not row.get('question') or family not in {'factual', 'math', 'code'}:
            raise ValueError('Every input needs a question and a valid family')
        if family != 'code' and row.get('gold') is None:
            raise ValueError('Missing gold answer')
        if family == 'code' and not row.get('tests'):
            raise ValueError('Missing correctness tests')
        output.append(row)
    return output


def build(config):
    rows = [r for source in config['sources'] for r in source_rows(source)]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate example IDs')
    assign_groups(rows, config.get('paraphrase_jaccard', 0.85))
    assign_splits(rows, config['seed'])
    backend = LocalModel(**config['model'])
    out = Path(config['output'])
    out.mkdir(parents=True, exist_ok=True)
    paths = [s['path'] for s in config['sources'] if 'path' in s]
    metadata = {**provenance(config, paths), 'kind': 'model_generations',
                'model': backend.identity, 'n': len(rows),
                'hygiene': 'entity/paraphrase annotations plus exact/lexical connected grouping'}
    means, lasts = [], []
    for i, row in enumerate(rows):
        answer = backend.generate(row, seed=config['seed'] + i)
        mean, last = backend.features(row, answer)
        row.update(answer=answer, outcome=grade(row, answer))
        # Abstentions are excluded from detector training and detection AUROC.
        row['label'] = None if row['outcome'] == 'ABSTAINED' else int(row['outcome'] == 'INCORRECT')
        means.append(mean)
        lasts.append(last)
        print(f"[{i+1}/{len(rows)}] {row['family']} {row['outcome']}", flush=True)
        if (i + 1) % config.get('save_every', 10) == 0:
            write_jsonl(out / 'records.jsonl', rows[:i+1])
            np.savez_compressed(out / 'features.npz', mean=np.stack(means), last=np.stack(lasts))
            write_json(out / 'manifest.json', {**metadata, 'complete': False, 'generated': i+1})
    write_jsonl(out / 'records.jsonl', rows)
    np.savez_compressed(out / 'features.npz', mean=np.stack(means), last=np.stack(lasts))
    write_json(out / 'manifest.json', {**metadata, 'complete': True, 'generated': len(rows)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    build(read_config(args.config))


if __name__ == '__main__':
    main()

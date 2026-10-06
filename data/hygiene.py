"""Connected groups for exact matches, supplied entities, and paraphrases."""
import hashlib
import re
import numpy as np


def question_key(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def assign_groups(rows, similarity=0.85):
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        parent[root(b)] = root(a)
    seen = {}
    tokens = []
    for i, row in enumerate(rows):
        keys = [('question', question_key(row['question']))]
        keys += [('entity', question_key(str(e))) for e in row.get('entities', [])]
        for field in ('paraphrase_group', 'group_id'):
            if row.get(field):
                keys.append((field, str(row[field])))
        for key in keys:
            if key in seen:
                union(i, seen[key])
            seen[key] = i
        tokens.append(set(question_key(row['question']).split()))
    # Lexical paraphrase screening. Semantic paraphrases need supplied group
    # annotations; this heuristic is deliberately not advertised as complete.
    for i in range(len(rows)):
        for j in range(i):
            if tokens[i] and len(tokens[i] & tokens[j]) / len(tokens[i] | tokens[j]) >= similarity:
                union(i, j)
    members = {}
    for i, row in enumerate(rows):
        members.setdefault(root(i), []).append(str(row['id']))
    ids = {k: hashlib.sha256('|'.join(sorted(v)).encode()).hexdigest()[:16] for k, v in members.items()}
    for i, row in enumerate(rows):
        row['group_id'] = ids[root(i)]
    return rows


def assign_splits(rows, seed, train_fraction=0.6, calibration_fraction=0.2):
    if not 0 < train_fraction < 1 or not 0 < calibration_fraction < 1 - train_fraction:
        raise ValueError('Invalid split fractions')
    rng = np.random.default_rng(seed)
    groups = sorted({r['group_id'] for r in rows})
    rng.shuffle(groups)
    n_train = int(len(groups) * train_fraction)
    n_cal = int(len(groups) * calibration_fraction)
    mapping = {g: ('train' if i < n_train else 'calibration' if i < n_train + n_cal else 'test')
               for i, g in enumerate(groups)}
    for row in rows:
        row['split'] = mapping[row['group_id']]
    return rows


def fold(rows, heldout):
    # Entire held-out family is inaccessible to fitting and calibration.
    test = [i for i, r in enumerate(rows) if r['family'] == heldout and r['split'] == 'test']
    heldout_groups = {r['group_id'] for r in rows if r['family'] == heldout}
    train = [i for i, r in enumerate(rows) if r['family'] != heldout
             and r['split'] == 'train' and r['group_id'] not in heldout_groups]
    cal = [i for i, r in enumerate(rows) if r['family'] != heldout
           and r['split'] == 'calibration' and r['group_id'] not in heldout_groups]
    idtest = [i for i, r in enumerate(rows) if r['family'] != heldout
              and r['split'] == 'test' and r['group_id'] not in heldout_groups]
    for a, b in ((train, cal), (train, test), (cal, test), (train, idtest), (cal, idtest)):
        if {rows[i]['group_id'] for i in a} & {rows[i]['group_id'] for i in b}:
            raise ValueError('Group leakage between splits')
    if not train or not cal or not test or not idtest:
        raise ValueError(f'Empty split in fold {heldout}; supply more independent groups')
    return train, cal, test, idtest

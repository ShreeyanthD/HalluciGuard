#!/usr/bin/env python3
"""Build closed-book eval files for eval_pipeline.py from PopQA and TriviaQA.

  python prepare_eval_data.py --dataset popqa    --n 1000 --out eval_data/popqa_1000.jsonl
  python prepare_eval_data.py --dataset triviaqa --n 1000 --out eval_data/triviaqa_1000.jsonl

Then:
  python eval_pipeline.py --dataset eval_data/popqa_1000.jsonl --probe probe/probe.joblib ...

Output rows: {"question", "answers": [aliases...], "group", "s_pop"?, "prop"?}
  PopQA   : group = popularity tercile of the subject (head / torso / tail), computed over the
            FULL PopQA test set, then sampled equally from each tercile (seeded).
  TriviaQA: group = "triviaqa" (validation split of unfiltered.nocontext; train is not used).

Sources: HuggingFace `akariasai/PopQA` (split test) and `mandarjoshi/trivia_qa`
(config unfiltered.nocontext, split validation). Use --popqa_file / --triviaqa_file to read a
local json/jsonl/csv export instead (offline machines).
"""
import argparse
import csv
import json
import random
import re
from pathlib import Path


def read_local(path):
    p = Path(path)
    if p.suffix.lower() == ".csv":
        with open(p, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    text = p.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).strip()


def clean_aliases(aliases, min_len=2):
    """Dedupe, drop empties and 1-char aliases (they cause false 'correct' matches)."""
    seen, out = set(), []
    for a in aliases:
        a = str(a).strip()
        k = norm(a)
        if len(k) < min_len or k in seen:
            continue
        seen.add(k)
        out.append(a)
    return out


def parse_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            x = json.loads(v)
            return x if isinstance(x, list) else [x]
        except json.JSONDecodeError:
            return [v]
    return []


def load_popqa(path=None):
    if path:
        return read_local(path)
    from datasets import load_dataset
    return list(load_dataset("akariasai/PopQA", split="test"))


def load_triviaqa(path=None):
    if path:
        return read_local(path)
    from datasets import load_dataset
    return list(load_dataset("mandarjoshi/trivia_qa", "unfiltered.nocontext", split="validation"))


def build_popqa(rows, n, seed):
    items = []
    for r in rows:
        ans = clean_aliases(parse_list(r.get("possible_answers")))
        q = str(r.get("question", "")).strip()
        if not q or not ans:
            continue
        items.append({"question": q, "answers": ans, "s_pop": float(r["s_pop"]),
                      "prop": r.get("prop", ""), "popqa_id": r.get("id", "")})
    # tercile cut points over the full set (not the sample)
    pops = sorted(x["s_pop"] for x in items)
    lo, hi = pops[len(pops) // 3], pops[2 * len(pops) // 3]
    for x in items:
        x["group"] = "tail" if x["s_pop"] < lo else ("torso" if x["s_pop"] < hi else "head")
    rng = random.Random(seed)
    per = n // 3
    out = []
    for g in ("head", "torso", "tail"):
        pool = [x for x in items if x["group"] == g]
        rng.shuffle(pool)
        out += pool[:per]
    rng.shuffle(out)
    return out


def build_triviaqa(rows, n, seed):
    items = []
    for r in rows:
        a = r.get("answer")
        if isinstance(a, str):
            try:
                a = json.loads(a)
            except json.JSONDecodeError:
                a = {"value": a, "aliases": []}
        a = a or {}
        ans = clean_aliases([a.get("value", "")] + list(a.get("aliases", [])) + list(a.get("normalized_aliases", [])))
        q = str(r.get("question", "")).strip()
        if q and ans:
            items.append({"question": q, "answers": ans, "group": "triviaqa"})
    random.Random(seed).shuffle(items)
    return items[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["popqa", "triviaqa"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--popqa_file")
    ap.add_argument("--triviaqa_file")
    args = ap.parse_args()

    if args.dataset == "popqa":
        out = build_popqa(load_popqa(args.popqa_file), args.n, args.seed)
    else:
        out = build_triviaqa(load_triviaqa(args.triviaqa_file), args.n, args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    groups = {}
    for x in out:
        groups[x["group"]] = groups.get(x["group"], 0) + 1
    print(f"wrote {len(out)} rows -> {args.out}  groups={groups}")


if __name__ == "__main__":
    main()

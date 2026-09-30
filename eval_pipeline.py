"""
eval_pipeline.py — measure how much the Blackboard reduces hallucinations.

Two arms, same answers, same questions:

    WITHOUT blackboard : the base model's raw answer, used as-is.
    WITH blackboard    : the same answer after blackboard_core.process_response()
                         (probe score -> threshold gate -> abstention check ->
                         claim extraction -> memory/retrieval/verify -> correct
                         or abstain).

Every answer in each arm is graded as one of:
    CORRECT       matches the gold answer
    ABSTAINED     "Unknown" / "I don't know" / hedge (not a wrong answer)
    HALLUCINATED  answered, and it is wrong

Headline numbers:
    hallucination_rate = HALLUCINATED / all questions
    (also reported: accuracy, abstention rate, hallucination rate among
    answered questions, and a paired-bootstrap 95% CI on the reduction)

Two stages, so the GPU part and the API part can run on different machines:

    Stage 1  generate + probe-score   (needs a CUDA GPU, Qwen, probe.joblib)
    Stage 2  run the blackboard       (needs GROQ_API_KEY, no GPU)

Usage
-----
  # everything in one go (GPU box with GROQ_API_KEY set):
  python eval_pipeline.py --dataset HaluEval-main/data/qa_data.json \
      --probe probe/probe.joblib --limit 200

  # stage 1 only (e.g. on Colab), then stage 2 anywhere:
  python eval_pipeline.py --dataset qa_data.json --probe probe.joblib \
      --generate_only --out_dir eval_runs/exp1
  python eval_pipeline.py --answers_file eval_runs/exp1/generations.jsonl \
      --out_dir eval_runs/exp1

  # no probe / no GPU: send EVERY answer through the blackboard
  python eval_pipeline.py --answers_file my_answers.jsonl --threshold 0

Dataset formats (--format auto detects by file contents)
  halueval : JSON/JSONL with question, right_answer, knowledge
  generic  : JSON/JSONL/CSV with question + answer (or gold / right_answer;
             a list of aliases is fine) + optional knowledge/context.
             Override column names with --q_field / --a_field / --k_field.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent

CORRECT, ABSTAINED, HALLUCINATED = "CORRECT", "ABSTAINED", "HALLUCINATED"
LABELS = (CORRECT, ABSTAINED, HALLUCINATED)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _read_rows(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if p.suffix.lower() == ".csv":
        with open(p, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":  # one JSON array
        return json.loads(text)
    rows = []  # JSONL
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _first(row: Dict[str, Any], names: List[str]) -> Any:
    for n in names:
        if n and n in row and row[n] not in (None, ""):
            return row[n]
    return None


def parse_indices(spec: Optional[str]) -> Optional[List[int]]:
    """'0,5,10-14' -> [0, 5, 10, 11, 12, 13, 14]. Order is kept, duplicates dropped."""
    if not spec:
        return None
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return list(dict.fromkeys(out)) or None


def load_dataset(path: str, fmt: str = "auto", q_field=None, a_field=None, k_field=None,
                 limit: Optional[int] = None, seed: int = 0, shuffle: bool = False,
                 start: int = 0, indices: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Selection (all indexes are 0-based row numbers in the ORIGINAL file):
         indices=[3, 17, 42]  -> exactly those rows, in that order
         start=100, limit=50  -> rows 100..149
         shuffle=True         -> random sample of `limit` rows (seeded)
       Each record keeps its original row number in rec["id"]."""
    rows = _read_rows(path)
    n_rows = len(rows)

    # original index of each row (a stage-1 file stores it in "id"; raw files use position)
    orig = []
    for i, row in enumerate(rows):
        rid = row.get("id", i) if isinstance(row, dict) else i
        try:
            orig.append(int(rid))
        except (TypeError, ValueError):
            orig.append(i)

    if indices is not None:
        pos = {o: i for i, o in enumerate(orig)}
        missing = [k for k in indices if k not in pos]
        if missing:
            raise IndexError(f"Indexes not in the file (it has {n_rows} rows): {missing[:10]}")
        order = [pos[k] for k in indices]
    else:
        order = list(range(n_rows))[max(0, start):]
        if shuffle:
            random.Random(seed).shuffle(order)

    out = []
    for i in order:
        row = rows[i]
        q = _first(row, [q_field, "question", "prompt", "query"])
        gold = _first(row, [a_field, "right_answer", "gold", "answer", "answers", "target"])
        know = _first(row, [k_field, "knowledge", "context", "evidence"])
        if fmt == "halueval":
            q = row.get("question")
            gold = row.get("right_answer")
            know = row.get("knowledge")
        if q is None or gold is None:
            continue
        golds = gold if isinstance(gold, list) else [gold]
        rec = {
            "id": orig[i],
            "question": str(q).strip(),
            "gold": [str(g).strip() for g in golds if str(g).strip()],
            "knowledge": str(know).strip() if know else "",
        }
        if row.get("hallucinated_answer"):
            rec["hallucinated_answer"] = str(row["hallucinated_answer"]).strip()
        # If the file already holds model answers (stage-1 output), keep them.
        if row.get("answer") is not None and "gold" in row:
            rec["answer"] = str(row["answer"])
        if row.get("prob_hallucinated") is not None:
            rec["prob_hallucinated"] = float(row["prob_hallucinated"])
        out.append(rec)
        if indices is None and limit and len(out) >= limit:
            break
    if not out:
        raise ValueError("No usable rows found (need a question and a gold answer per row).")
    return out


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

_ARTICLES = {"a", "an", "the"}

_ABSTAIN_EXACT = {"unknown", "idk", "unsure", "unanswerable", "n/a", "none of the above"}

_ABSTAIN_PHRASES = (
    "i don't know", "i do not know", "i'm not sure", "i am not sure",
    "not certain", "uncertain", "cannot determine", "can't determine",
    "unable to answer", "unable to verify", "no information available",
    "insufficient information", "don't have enough information",
    "do not have enough information", "i have no information",
    "i cannot say", "i can't say", "cannot confirm", "can't confirm",
    "couldn't be confirmed", "could not be confirmed", "couldn't be verified",
    "could not be verified", "not able to verify", "don't have reliable",
    "do not have reliable", "no reliable information", "i have no data",
    "i don't have access", "i do not have access", "not able to confirm",
)


def _tokens(s: str) -> List[str]:
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return [t for t in s.split() if t not in _ARTICLES]


def _contains(hay: List[str], needle: List[str]) -> bool:
    n = len(needle)
    if n == 0 or n > len(hay):
        return False
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def is_exact_abstention(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return True
    return re.sub(r"[^\w/\s']", "", t).strip() in _ABSTAIN_EXACT


def matches_gold(answer: str, golds: List[str], mode: str = "lenient") -> bool:
    a = _tokens(answer)
    if not a:
        return False
    for g in golds:
        gt = _tokens(g)
        if not gt:
            continue
        if a == gt or _contains(a, gt):
            return True
        # lenient: a one-word answer like "Einstein" is fine for "Albert Einstein"
        if mode == "lenient" and _contains(gt, a):
            return True
    return False


def grade(answer: str, golds: List[str], mode: str = "lenient", judge=None) -> str:
    if is_exact_abstention(answer):
        return ABSTAINED
    if matches_gold(answer, golds, mode):
        return CORRECT
    low = answer.lower()
    if any(p in low for p in _ABSTAIN_PHRASES):
        return ABSTAINED
    if judge is not None:
        try:
            if judge(answer, golds):
                return CORRECT
        except Exception:
            pass
    return HALLUCINATED


def make_llm_judge(llm_caller):
    """Optional: ask an LLM whether a non-matching answer is actually
    equivalent to the gold answer (synonyms, abbreviations, rewording)."""
    def judge(answer: str, golds: List[str]) -> bool:
        prompt = (
            "Is the RESPONSE saying the same thing as the GOLD answer "
            "(same entity/fact; synonyms and abbreviations count)? "
            "Reply with exactly YES or NO.\n\n"
            f"GOLD: {' | '.join(golds)}\nRESPONSE: {answer}"
        )
        return llm_caller(prompt, temperature=0.0).strip().upper().startswith("YES")
    return judge


# ---------------------------------------------------------------------------
# Stage 1: generate + probe score (GPU)
# ---------------------------------------------------------------------------

def run_generation_stage(records, args) -> List[Dict[str, Any]]:
    import joblib
    import torch

    sys.path.insert(0, str(_ROOT / "probe"))
    from llama_features import QwenResidualFeatureExtractor

    scaler = clf = feature_dim = None
    if args.probe:
        bundle = joblib.load(args.probe)
        scaler, clf, feature_dim = bundle["scaler"], bundle["classifier"], bundle.get("feature_dim")

    print(f"[stage 1] loading {args.model_id} (layer {args.layer})...")
    extractor = QwenResidualFeatureExtractor(model_id=args.model_id, layer=args.layer)
    print(f"[stage 1] system prompt: {extractor.default_system_prompt!r}")

    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        gens = extractor.batch_generate_and_extract_features(
            [r["question"] for r in batch], max_new_tokens=args.max_new_tokens
        )
        for r, g in zip(batch, gens):
            r["answer"] = g["answer_text"]
            if clf is not None:
                X = g["feature_vector"].cpu().to(torch.float32).numpy()[None, :]
                if feature_dim is not None and X.shape[1] != feature_dim:
                    raise ValueError(f"Probe dim {feature_dim} != feature dim {X.shape[1]} (check --layer).")
                r["prob_hallucinated"] = float(clf.predict_proba(scaler.transform(X))[:, 1][0])
        print(f"[stage 1] {min(start + args.batch_size, len(records))}/{len(records)}")
    return records


# ---------------------------------------------------------------------------
# Stage 2: blackboard
# ---------------------------------------------------------------------------

def run_blackboard_stage(records, args):
    # Isolated store so the eval never touches your real halluciguard_chroma.
    # Must be set BEFORE blackboard_core is imported (it opens Chroma at import).
    os.environ["CHROMA_PERSIST_DIRECTORY"] = str(Path(args.chroma_dir).resolve())
    bb_dir = str(_ROOT / "blackboard")
    if bb_dir not in sys.path:
        sys.path.insert(0, bb_dir)  # use blackboard/blackboard_core.py, not the stale root copy
    import blackboard_core as bc

    bc.HALLUCINATION_RISK_THRESHOLD = args.threshold
    if not args.enable_web:
        bc.orchestrator.web_search_agent = None  # keep the eval closed-book + local KB only

    bc.reset_memory_collection()
    bc.reset_knowledge_collection()
    if not args.no_knowledge:
        seen, docs, ids = set(), [], []
        for r in records:
            k = r.get("knowledge", "")
            if k and k not in seen:
                seen.add(k)
                docs.append(k)
                ids.append(f"kb-{len(ids)}")
        if docs:
            for s in range(0, len(docs), 200):
                bc.add_knowledge_documents(docs[s:s + 200], ids=ids[s:s + 200],
                                           metadatas=[{"source": "eval_dataset"}] * len(docs[s:s + 200]))
        print(f"[stage 2] knowledge base: {len(docs)} documents loaded")
    else:
        print("[stage 2] knowledge base: EMPTY (--no_knowledge)")

    results = []
    t_all = time.time()
    for i, r in enumerate(records):
        raw = r["answer"]
        score = r.get("prob_hallucinated")
        if score is None:
            score = args.default_score
        row = {"status": None, "final_response": raw, "evidence_source": None,
               "verdict": None, "error": None, "seconds": 0.0}
        t0 = time.time()
        if not raw.strip():
            row["status"] = "EMPTY_ANSWER"
        else:
            try:
                if args.isolate_memory:
                    bc.reset_memory_collection()
                out = bc.process_response(prompt=r["question"], response=raw, confidence_score=score)
                row["status"] = out["status"]
                row["final_response"] = out["final_response"]
                row["evidence_source"] = out.get("evidence_source")
                vr = out.get("verification_result") or {}
                row["verdict"] = vr.get("verdict")
            except Exception as exc:  # one bad API call must not kill the run
                row["status"] = "ERROR"
                row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        row["seconds"] = round(time.time() - t0, 2)
        results.append(row)
        if (i + 1) % 10 == 0 or i + 1 == len(records):
            print(f"[stage 2] {i + 1}/{len(records)}  ({time.time() - t_all:.0f}s)")
    return results, bc


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def arm_metrics(labels: List[str]) -> Dict[str, Any]:
    n = len(labels)
    c = Counter(labels)
    answered = c[CORRECT] + c[HALLUCINATED]
    return {
        "n": n,
        "correct": c[CORRECT],
        "abstained": c[ABSTAINED],
        "hallucinated": c[HALLUCINATED],
        "hallucination_rate": c[HALLUCINATED] / n if n else 0.0,
        "accuracy": c[CORRECT] / n if n else 0.0,
        "abstention_rate": c[ABSTAINED] / n if n else 0.0,
        "hallucination_rate_among_answered": c[HALLUCINATED] / answered if answered else 0.0,
    }


def paired_bootstrap(base: List[str], bb: List[str], iters: int = 2000, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    b = np.array([x == HALLUCINATED for x in base], dtype=float)
    w = np.array([x == HALLUCINATED for x in bb], dtype=float)
    n = len(b)
    idx = rng.integers(0, n, size=(iters, n))
    diffs = b[idx].mean(axis=1) - w[idx].mean(axis=1)  # positive = blackboard helps
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"abs_reduction_ci95_low": float(lo), "abs_reduction_ci95_high": float(hi)}


def build_report(records, rows, base_labels, bb_labels, args) -> Dict[str, Any]:
    base = arm_metrics(base_labels)
    bb = arm_metrics(bb_labels)
    abs_red = base["hallucination_rate"] - bb["hallucination_rate"]
    rel_red = abs_red / base["hallucination_rate"] if base["hallucination_rate"] else 0.0

    transitions = Counter(f"{a} -> {b}" for a, b in zip(base_labels, bb_labels))
    by_status = Counter(r["status"] for r in rows)
    triggered = sum(1 for r in rows if r["status"] not in ("SKIPPED_LOW_RISK", "EMPTY_ANSWER", "ERROR"))

    return {
        "config": {
            "n_questions": len(records),
            "threshold": args.threshold,
            "default_score_when_no_probe": args.default_score,
            "knowledge_base": "empty" if args.no_knowledge else "dataset knowledge (pooled)",
            "web_search": bool(args.enable_web),
            "isolate_memory": bool(args.isolate_memory),
            "match_mode": args.match,
            "llm_judge": bool(args.llm_judge),
        },
        "without_blackboard": base,
        "with_blackboard": bb,
        "absolute_hallucination_reduction": abs_red,
        "relative_hallucination_reduction": rel_red,
        **paired_bootstrap(base_labels, bb_labels, seed=args.seed),
        "blackboard_triggered": triggered,
        "status_counts": dict(by_status),
        "transitions_baseline_to_blackboard": dict(transitions),
        "harmed": transitions.get(f"{CORRECT} -> {HALLUCINATED}", 0),
        "lost_to_over_abstention": transitions.get(f"{CORRECT} -> {ABSTAINED}", 0),
        "errors": by_status.get("ERROR", 0),
    }


def format_summary(rep: Dict[str, Any]) -> str:
    b, w = rep["without_blackboard"], rep["with_blackboard"]
    pct = lambda x: f"{100 * x:5.1f}%"
    lines = [
        f"# Blackboard evaluation  (n={b['n']})",
        "",
        "| metric | without blackboard | with blackboard |",
        "|---|---|---|",
        f"| hallucination rate (wrong / all) | {pct(b['hallucination_rate'])} | {pct(w['hallucination_rate'])} |",
        f"| accuracy (correct / all) | {pct(b['accuracy'])} | {pct(w['accuracy'])} |",
        f"| abstention rate | {pct(b['abstention_rate'])} | {pct(w['abstention_rate'])} |",
        f"| hallucination rate among answered | {pct(b['hallucination_rate_among_answered'])} | {pct(w['hallucination_rate_among_answered'])} |",
        f"| counts (correct / abstained / wrong) | {b['correct']} / {b['abstained']} / {b['hallucinated']} | {w['correct']} / {w['abstained']} / {w['hallucinated']} |",
        "",
        f"**Hallucination reduction: {100 * rep['absolute_hallucination_reduction']:.1f} points "
        f"({100 * rep['relative_hallucination_reduction']:.1f}% relative)**, "
        f"95% CI on the point drop: [{100 * rep['abs_reduction_ci95_low']:.1f}, {100 * rep['abs_reduction_ci95_high']:.1f}]",
        "",
        f"Blackboard ran on {rep['blackboard_triggered']} of {b['n']} answers. "
        f"Made worse (correct -> wrong): {rep['harmed']}. "
        f"Lost to over-abstaining (correct -> abstained): {rep['lost_to_over_abstention']}. "
        f"API errors: {rep['errors']}.",
        "",
        "Status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(rep["status_counts"].items())),
        "",
        "Transitions (without -> with blackboard):",
    ]
    for k, v in sorted(rep["transitions_baseline_to_blackboard"].items()):
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("input")
    src.add_argument("--dataset", help="QA file (HaluEval qa_data.json, or generic json/jsonl/csv)")
    src.add_argument("--answers_file", help="Stage-1 output (generations.jsonl) — skips generation")
    src.add_argument("--format", default="auto", choices=["auto", "halueval", "generic"])
    src.add_argument("--q_field"); src.add_argument("--a_field"); src.add_argument("--k_field")
    src.add_argument("--limit", type=int, default=None, help="how many questions to use")
    src.add_argument("--start", type=int, default=0, help="first row to use (0-based), with --limit")
    src.add_argument("--indices", default=None,
                     help="exact 0-based row numbers, e.g. '0,5,12' or '10-19,42' (overrides --start/--limit/--shuffle)")
    src.add_argument("--indices_file", default=None, help="text file with indexes (commas/newlines/ranges)")
    src.add_argument("--shuffle", action="store_true")
    src.add_argument("--seed", type=int, default=0)

    gen = ap.add_argument_group("stage 1 (generation, GPU)")
    gen.add_argument("--probe", help="probe .joblib (gives prob_hallucinated for the threshold gate)")
    gen.add_argument("--layer", type=int, default=20)
    gen.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    gen.add_argument("--batch_size", type=int, default=8)
    gen.add_argument("--max_new_tokens", type=int, default=16)
    gen.add_argument("--generate_only", action="store_true")

    bbg = ap.add_argument_group("stage 2 (blackboard)")
    bbg.add_argument("--threshold", type=float, default=0.70,
                     help="risk threshold; 0 = send every answer through the blackboard")
    bbg.add_argument("--default_score", type=float, default=1.0,
                     help="risk score used when a record has no probe score (1.0 = always flagged)")
    bbg.add_argument("--chroma_dir", default="./eval_chroma", help="isolated ChromaDB dir for the eval")
    bbg.add_argument("--no_knowledge", action="store_true", help="ablation: empty knowledge base")
    bbg.add_argument("--enable_web", action="store_true", help="allow Tavily web-search fallback")
    bbg.add_argument("--isolate_memory", action="store_true",
                     help="clear episodic memory before every question (no cross-question leakage)")

    gr = ap.add_argument_group("grading / output")
    gr.add_argument("--match", default="lenient", choices=["lenient", "strict"])
    gr.add_argument("--llm_judge", action="store_true", help="LLM check for answers that don't string-match gold")
    gr.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    if not args.dataset and not args.answers_file:
        ap.error("Provide --dataset or --answers_file.")

    out_dir = Path(args.out_dir or f"eval_runs/run_{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load / generate -------------------------------------------------
    spec = args.indices
    if args.indices_file:
        spec = (spec + "," if spec else "") + ",".join(Path(args.indices_file).read_text().split())
    indices = parse_indices(spec)

    if args.answers_file:
        records = load_dataset(args.answers_file, "generic", limit=args.limit, seed=args.seed,
                               shuffle=args.shuffle, start=args.start, indices=indices)
        if any("answer" not in r for r in records):
            ap.error("--answers_file rows need an 'answer' field (use stage-1 output, or add model answers).")
        print(f"Loaded {len(records)} records with answers from {args.answers_file}")
    else:
        records = load_dataset(args.dataset, args.format, args.q_field, args.a_field, args.k_field,
                               args.limit, args.seed, args.shuffle, start=args.start, indices=indices)
        print(f"Loaded {len(records)} questions from {args.dataset}")
        records = run_generation_stage(records, args)
        with open(out_dir / "generations.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved generations -> {out_dir / 'generations.jsonl'}")
        if args.generate_only:
            return

    print("Using dataset rows:", [r["id"] for r in records])

    # ---- blackboard ------------------------------------------------------
    rows, bc = run_blackboard_stage(records, args)

    # ---- grade both arms -------------------------------------------------
    judge = make_llm_judge(bc.groq_caller) if args.llm_judge else None
    base_labels = [grade(r["answer"], r["gold"], args.match, judge) for r in records]
    bb_labels = [grade(x["final_response"], r["gold"], args.match, judge) for r, x in zip(records, rows)]

    report = build_report(records, rows, base_labels, bb_labels, args)
    summary = format_summary(report)

    # ---- save ------------------------------------------------------------
    with open(out_dir / "per_item.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_index", "question", "gold", "halueval_hallucinated_answer", "prob_hallucinated", "raw_answer", "raw_label",
                    "status", "verdict", "evidence_source", "final_response", "final_label", "error"])
        for r, x, bl, wl in zip(records, rows, base_labels, bb_labels):
            w.writerow([r["id"], r["question"], " | ".join(r["gold"]), r.get("hallucinated_answer", ""), r.get("prob_hallucinated", ""),
                        r["answer"], bl, x["status"], x["verdict"] or "", x["evidence_source"] or "",
                        x["final_response"], wl, x["error"] or ""])
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(out_dir / "summary.md", "w") as f:
        f.write(summary + "\n")

    print("\n" + summary)
    print(f"\nSaved -> {out_dir}/ (metrics.json, per_item.csv, summary.md)")


if __name__ == "__main__":
    main()

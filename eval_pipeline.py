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
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent

CORRECT, ABSTAINED, HALLUCINATED = "CORRECT", "ABSTAINED", "HALLUCINATED"
PROCESSING_ERROR = "PROCESSING_ERROR"
LABELS = (CORRECT, ABSTAINED, HALLUCINATED, PROCESSING_ERROR)


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


LOAD_STATS: Dict[str, int] = {}


def norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(q).lower())).strip()


def load_dataset(path: str, fmt: str = "auto", q_field=None, a_field=None, k_field=None,
                 limit: Optional[int] = None, seed: int = 0, shuffle: bool = False,
                 start: int = 0, indices: Optional[List[int]] = None,
                 exclude_ids: Optional[set] = None, exclude_questions: Optional[set] = None) -> List[Dict[str, Any]]:
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
    seen_q: set = set()
    LOAD_STATS.update(excluded_probe_train=0, duplicates_dropped=0)
    exclude_ids = exclude_ids or set()
    exclude_questions = set(exclude_questions or set())
    # a row that duplicates a probe-training question is also training data
    for i, row in enumerate(rows):
        if orig[i] in exclude_ids and isinstance(row, dict):
            q0 = _first(row, [q_field, "question", "prompt", "query"])
            if q0 is not None:
                exclude_questions.add(norm_q(q0))
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
        if orig[i] in exclude_ids or norm_q(q) in exclude_questions:
            LOAD_STATS["excluded_probe_train"] += 1   # probe saw this question in training
            continue
        if norm_q(q) in seen_q:
            LOAD_STATS["duplicates_dropped"] += 1
            continue
        seen_q.add(norm_q(q))
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
# Statistics helpers
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> List[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p-value on discordant pairs (b = baseline-only wrong,
    c = other-arm-only wrong)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def auroc(scores: List[float], y: List[int]) -> Optional[float]:
    """Mann-Whitney AUROC; None if only one class is present."""
    pos = [s for s, t in zip(scores, y) if t == 1]
    neg = [s for s, t in zip(scores, y) if t == 0]
    if not pos or not neg:
        return None
    sc = np.array(scores, dtype=float)
    order = sc.argsort()
    ranks = np.empty(len(sc))
    i = 0
    while i < len(sc):               # average ranks over ties
        j = i
        while j + 1 < len(sc) and sc[order[j + 1]] == sc[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    yy = np.array(y)
    r_pos = ranks[yy == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def probe_threshold_report(records, base_labels, threshold: float) -> Optional[Dict[str, Any]]:
    """Precision/recall of 'flag if score >= threshold' for catching wrong raw answers.
    Positive class = HALLUCINATED; abstained raw answers are excluded."""
    pairs = [(r["prob_hallucinated"], l == HALLUCINATED) for r, l in zip(records, base_labels)
             if r.get("prob_hallucinated") is not None and l != ABSTAINED]
    if not pairs:
        return None
    tp = sum(1 for sc, y in pairs if sc >= threshold and y)
    fp = sum(1 for sc, y in pairs if sc >= threshold and not y)
    fn = sum(1 for sc, y in pairs if sc < threshold and y)
    tn = sum(1 for sc, y in pairs if sc < threshold and not y)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1,
            "precision_ci95": wilson_ci(tp, tp + fp) if tp + fp else None,
            "recall_ci95": wilson_ci(tp, tp + fn) if tp + fn else None,
            "base_wrong_rate": (tp + fn) / len(pairs), "flagged_rate": (tp + fp) / len(pairs)}


def probe_auroc_report(records, base_labels, iters: int = 2000, seed: int = 0) -> Optional[Dict[str, Any]]:
    """Does the probe score separate wrong from correct raw answers? Abstentions
    are excluded (the probe score is not meaningful for them)."""
    pairs = [(r["prob_hallucinated"], 1 if l == HALLUCINATED else 0)
             for r, l in zip(records, base_labels)
             if r.get("prob_hallucinated") is not None and l != ABSTAINED]
    if len(pairs) < 10:
        return None
    sc, y = [p[0] for p in pairs], [p[1] for p in pairs]
    point = auroc(sc, y)
    if point is None:
        return None
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(iters):
        idx = rng.integers(0, len(sc), len(sc))
        a = auroc([sc[i] for i in idx], [y[i] for i in idx])
        if a is not None:
            boots.append(a)
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (None, None)
    return {"n": len(sc), "n_wrong": int(sum(y)), "auroc": point,
            "ci95_low": None if lo is None else float(lo),
            "ci95_high": None if hi is None else float(hi)}


# ---------------------------------------------------------------------------
# Probe-contamination guard
# ---------------------------------------------------------------------------

def probe_exclusions(args) -> Dict[str, Any]:
    """Questions/rows the probe was TRAINED on must not be evaluated.
    Sources, in order: the probe bundle's own `train_questions` (probes trained
    with the updated train_probe.py), then --probe_train_indices for legacy
    probes (e.g. rows 0-99 for the original Dataset_builder run)."""
    ids: set = set(parse_indices(args.probe_train_indices) or [])
    questions: set = set()
    source = "probe_train_indices" if ids else None
    if args.probe:
        import joblib
        bundle = joblib.load(args.probe)
        tq = bundle.get("train_questions")
        if tq:
            questions = {norm_q(q) for q in tq}
            source = "probe bundle train_questions"
    return {"ids": ids, "questions": questions, "source": source}


def enforce_probe_guard(args, ex: Dict[str, Any], uses_probe_scores: bool) -> None:
    if not uses_probe_scores or ex["source"] or args.allow_train_overlap:
        return
    raise SystemExit(
        "REFUSING TO RUN: this evaluation uses probe scores but nothing says which questions the "
        "probe was trained on, so train/test overlap cannot be ruled out.\n"
        "  - retrain with the updated probe/train_probe.py (records train_questions in the bundle), or\n"
        "  - pass --probe_train_indices (e.g. 0-99 for the original Dataset_builder run), or\n"
        "  - pass --allow_train_overlap to run anyway (results are then NOT paper-clean).")


# ---------------------------------------------------------------------------
# Rate-limit handling shared by every worker thread (Groq 429s)
# ---------------------------------------------------------------------------

class RateLimit:
    """When ANY worker gets a 429, ALL workers pause for the server-suggested time.
    A per-day quota (long wait) stops further API calls instead of burning retries."""
    import threading as _th
    _lock = _th.Lock()
    until = 0.0                     # monotonic time before which no call should start
    exhausted: Optional[str] = None  # message if a daily quota is used up

    @staticmethod
    def parse_wait(msg: str) -> Optional[float]:
        """'Please try again in 7m12.5s' / '560ms' / '1.2s' -> seconds."""
        m = re.search(r"try again in\s+((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)", msg)
        if not m:
            return None
        mult = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
        return sum(float(n) * mult[u] for n, u in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", m.group(1)))

    @classmethod
    def note(cls, exc: Exception) -> bool:
        """Register a failure; returns True if it was a rate limit (the shared pause is then set)."""
        msg = str(exc)
        if "429" not in msg and "rate limit" not in msg.lower():
            return False
        parsed = cls.parse_wait(msg)
        wait = 20.0 if parsed is None else parsed
        per_day = bool(re.search(r"per day|\(TPD\)|\(RPD\)", msg))
        with cls._lock:
            # a daily limit with a long (or unknown) reset time will not clear during this run
            if per_day and (parsed is None or wait > 900):
                if cls.exhausted is None:
                    cls.exhausted = msg[:400]
                    print(f"\n[RATE LIMIT] DAILY QUOTA EXHAUSTED (retry in ~{wait/60:.0f} min). Stopping API calls; "
                          "finished work is kept. Rerun later with --reuse_stage2 --retry_errors.\n{0}".format(msg[:300]))
            else:
                cls.until = max(cls.until, time.monotonic() + min(wait, 300.0) * 1.1 + 1.0)
        return True

    @classmethod
    def wait(cls) -> None:
        while True:
            d = cls.until - time.monotonic()
            if d <= 0:
                return
            time.sleep(min(d, 5.0))


# ---------------------------------------------------------------------------
# Control arm: the strong LLM answering closed-book, with no blackboard
# ---------------------------------------------------------------------------

CONTROL_PROMPT = (
    "Answer the question in a single word or short phrase only. Do not use full sentences, "
    "explanations, or punctuation. If you do not know the answer or are not sure, answer with "
    "the single word: Unknown\n\nQuestion: {q}")


def run_llm_only_control(records, llm_caller, workers: int = 1, retries: int = 0,
                         checkpoint: Optional[Path] = None) -> List[str]:
    """Same model family that powers the blackboard agents, asked the question
    directly. If the blackboard doesn't beat this, its gain is just 'a bigger
    model answered' rather than the architecture.
    Parallel, retried, and resumable: answers already in `checkpoint` (same question, not an error) are reused."""
    import threading
    key = lambda i, r: (r.get("id", i), r["question"])
    done: Dict[Any, str] = {}
    if checkpoint and Path(checkpoint).exists():
        for l in open(checkpoint, encoding="utf-8"):
            if l.strip():
                x = json.loads(l)
                if not x["answer"].startswith("[CONTROL_ERROR"):
                    done[(x["id"], x["question"])] = x["answer"]
    out: List[Optional[str]] = [done.get(key(i, r)) for i, r in enumerate(records)]
    todo = [i for i, a in enumerate(out) if a is None]
    if done:
        print(f"[control] resuming: {len(records) - len(todo)} answers reused, {len(todo)} to go")
    lock = threading.Lock()
    ck = open(checkpoint, "a", encoding="utf-8") if checkpoint else None

    def one(i):
        r = records[i]
        ans = None
        for attempt in range(1 + max(0, retries)):
            if RateLimit.exhausted:
                ans = "[CONTROL_ERROR QUOTA_EXHAUSTED]"
                break
            RateLimit.wait()
            try:
                ans = llm_caller(CONTROL_PROMPT.format(q=r["question"]), temperature=0.0).strip()
                break
            except Exception as exc:
                ans = f"[CONTROL_ERROR {type(exc).__name__}]"
                if not RateLimit.note(exc) and attempt < retries:
                    time.sleep(min(30.0, 2.0 ** attempt))
        with lock:
            out[i] = ans
            if ck and not ans.startswith("[CONTROL_ERROR"):
                ck.write(json.dumps({"id": r.get("id", i), "question": r["question"], "answer": ans}, ensure_ascii=False) + "\n")
                ck.flush()

    n_done = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for _ in ex.map(one, todo):
                n_done += 1
                if n_done % 10 == 0 or n_done == len(todo):
                    print(f"[control] {n_done}/{len(todo)}")
    finally:
        if ck:
            ck.close()
    return out  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Preflight: measure real API cost/limits on a few items BEFORE the long run
# ---------------------------------------------------------------------------

def run_preflight(records, args) -> None:
    import copy
    bc = load_blackboard_module(args)
    usage = {"calls": 0, "tokens": 0}
    comp = bc._groq_client.chat.completions
    orig_create = comp.create

    def counting(*a, **k):
        r = orig_create(*a, **k)
        usage["calls"] += 1
        u = getattr(r, "usage", None)
        usage["tokens"] += int(getattr(u, "total_tokens", 0) or 0)
        return r

    comp.create = counting
    try:
        def routed(r):
            sc = r.get("prob_hallucinated")
            return (sc if sc is not None else args.default_score) >= args.threshold or \
                grade(r["answer"], r["gold"], "lenient") == ABSTAINED
        todo = [r for r in records if routed(r)]
        n = min(max(1, args.preflight), len(todo))
        print(f"[preflight] {len(records)} questions, {len(todo)} will go through the blackboard; testing {n}")
        sub = copy.copy(args)
        sub.workers, sub.retries, sub.fallback_on_error = 1, 1, False
        t0 = time.time()
        rows, _ = run_blackboard_stage(todo[:n], sub)
        secs = max(1e-6, time.time() - t0)
        errs = [x for x in rows if x["status"] == "ERROR"]
        if usage["calls"] == 0 and len(errs) < n:
            print("[preflight] WARNING: no LLM calls were counted, so token estimates below are NOT valid.")
        tok_item = usage["tokens"] / n
        print(f"\n[preflight] blackboard: {usage['calls']/n:.1f} LLM calls and {tok_item:,.0f} tokens per item, {secs/n:.1f}s per item")
        print(f"[preflight] {len(errs)}/{n} items failed")
        for x in errs[:3]:
            print("   error:", (x.get("error") or "")[:400])

        if len(errs) == n:
            print("[preflight] EVERY test item failed, so no estimate is possible. Fix the API problem shown above first "
                  "(429 = rate/quota limit, 401 = bad key, connection = no internet), then rerun --preflight.")
            return
        before = usage["tokens"]
        m = min(5, len(records))
        for r in records[:m]:
            try:
                bc.groq_caller(CONTROL_PROMPT.format(q=r["question"]), temperature=0.0)
            except Exception as exc:
                print("   control error:", str(exc)[:300])
        tok_ctrl = (usage["tokens"] - before) / m

        total = len(todo) * tok_item + (len(records) * tok_ctrl if args.llm_only_control else 0)
        print(f"[preflight] control arm: {tok_ctrl:,.0f} tokens per question")
        print(f"[preflight] PROJECTED full run: {total:,.0f} tokens"
              f" ({len(todo) * tok_item:,.0f} blackboard"
              + (f" + {len(records) * tok_ctrl:,.0f} control" if args.llm_only_control else "") + ")")
        hdr = {}
        try:
            raw = comp.with_raw_response.create(model=bc.GROQ_MODEL, max_tokens=1,
                                                messages=[{"role": "user", "content": "hi"}])
            hdr = {k.lower(): v for k, v in raw.headers.items() if k.lower().startswith("x-ratelimit")}
        except Exception as exc:
            print("[preflight] could not read rate-limit headers:", str(exc)[:200])
        for k, v in sorted(hdr.items()):
            print(f"   {k}: {v}")
        try:
            tpm = float(hdr.get("x-ratelimit-limit-tokens", ""))
            rpd = float(hdr.get("x-ratelimit-remaining-requests", ""))
            rate = tok_item / (secs / n) * 60            # tokens/min used by ONE worker
            w = max(1, int(0.6 * tpm / rate))
            print(f"[preflight] one worker uses ~{rate:,.0f} tokens/min; with a {tpm:,.0f} TPM limit use --workers {w} or fewer")
            calls_needed = len(todo) * usage["calls"] / n + (len(records) if args.llm_only_control else 0)
            print(f"[preflight] requests/day: ~{calls_needed:,.0f} needed, {rpd:,.0f} remaining -> "
                  + ("OK" if calls_needed < 0.8 * rpd else "WILL NOT FIT TODAY"))
        except ValueError:
            pass
        print("[preflight] The daily TOKEN limit is not in the headers. Open console.groq.com/settings/limits, find "
              f"{bc.GROQ_MODEL} 'tokens per day', and compare it with the projected total above. "
              "If projected > 80% of it, split the run over days with --reuse_stage2 --retry_errors.")
    finally:
        comp.create = orig_create


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------

def _sha256(path: Optional[str]) -> Optional[str]:
    if not path or not Path(path).exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*cmd) -> Optional[str]:
    try:
        return subprocess.check_output(["git", *cmd], cwd=_ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def write_manifest(out_dir: Path, args, records, bc, guard) -> None:
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for mod in ("torch", "transformers", "sklearn", "chromadb"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            pass
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "args": {k: v for k, v in vars(args).items()},
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "blackboard_core_file": getattr(bc, "__file__", None),
        "groq_model": getattr(bc, "GROQ_MODEL", None),
        "generator_model": args.model_id, "probe_layer": args.layer,
        "sha256": {"dataset": _sha256(args.dataset), "answers_file": _sha256(args.answers_file),
                   "probe": _sha256(args.probe)},
        "probe_contamination_check": guard["source"] or ("NOT CHECKED (allow_train_overlap)" if args.allow_train_overlap else "n/a (no probe scores)"),
        "excluded_as_probe_train": LOAD_STATS.get("excluded_probe_train", 0),
        "duplicate_questions_dropped": LOAD_STATS.get("duplicates_dropped", 0),
        "row_ids_evaluated": [r["id"] for r in records],
        "versions": versions,
    }
    with open(out_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)


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

def load_blackboard_module(args):
    # Isolated store so the eval never touches your real halluciguard_chroma.
    # Must be set BEFORE blackboard_core is imported (it opens Chroma at import).
    os.environ["CHROMA_PERSIST_DIRECTORY"] = str(Path(args.chroma_dir).resolve())
    bb_dir = str(_ROOT / "blackboard")
    if bb_dir not in sys.path:
        sys.path.insert(0, bb_dir)  # use blackboard/blackboard_core.py, not the stale root copy
    import blackboard_core as bc
    if Path(bc.__file__).resolve().parent != Path(bb_dir).resolve():
        raise RuntimeError(f"Imported the wrong blackboard_core: {bc.__file__} (expected {bb_dir}/)")
    return bc


STAGE2_KEYS = ("threshold", "default_score", "no_knowledge", "enable_web", "isolate_memory", "fallback_on_error")


def save_stage2_rows(path: Path, records, rows, args) -> None:
    """Checkpoint right after stage 2 so a crash/kill in grading or the control arm never costs a rerun."""
    meta = {k: getattr(args, k, None) for k in STAGE2_KEYS}
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": meta}) + "\n")
        for r, x in zip(records, rows):
            f.write(json.dumps({"id": r["id"], "question": r["question"], **x}, ensure_ascii=False) + "\n")


def load_stage2_rows(path: Path, records, args):
    lines = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    meta, saved = lines[0]["_meta"], lines[1:]
    cur = {k: getattr(args, k, None) for k in STAGE2_KEYS}
    if meta != cur:
        raise SystemExit(f"--reuse_stage2: saved rows were made with different settings.\n  saved:   {meta}\n  current: {cur}")
    if [(x["id"], x["question"]) for x in saved] != [(r["id"], r["question"]) for r in records]:
        raise SystemExit("--reuse_stage2: saved rows do not match the questions selected for this run.")
    return [{k: v for k, v in x.items() if k not in ("id", "question")} for x in saved]


def run_blackboard_stage(records, args):
    bc = load_blackboard_module(args)

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
        print(f"[stage 2] knowledge base: {len(docs)} ORACLE documents loaded (gold knowledge from the eval items). "
              "Report this as an upper bound / oracle-retrieval result, NOT as a closed-book result.")
    else:
        print("[stage 2] knowledge base: EMPTY (closed-book; default)")

    # ---- parallel stage 2 ------------------------------------------------
    # Each worker thread gets its own Blackboard + Orchestrator (the module-level
    # one holds a single shared workspace). Agents themselves are stateless and
    # take the blackboard as an argument, so they are shared.
    class _NoMemory:
        """Used with --isolate_memory: every question must see an empty episodic memory,
        which is not expressible with a shared store under concurrency."""
        def check_memory(self, blackboard):
            res = {"match_found": False, "matches": [], "best_match": None}
            blackboard.write("memory_result", res, author="NoMemory")
            return res
        def log_verified_claim(self, blackboard):
            return None

    memory_agent = _NoMemory() if args.isolate_memory else bc.memory_agent
    local = threading.local()

    def _components():
        if not hasattr(local, "orch"):
            o = _components_base
            local.orch = bc.Orchestrator(
                blackboard=bc.Blackboard(), memory_agent=memory_agent,
                retrieval_agent=o.retrieval_agent, verifier_agent=o.verifier_agent,
                correction_agent=o.correction_agent,
                max_verification_rounds=o.max_verification_rounds,
                web_search_agent=o.web_search_agent, answer_agent=o.answer_agent,
            )
        return local.orch

    # Works with ANY blackboard_core: process_response() looks up the module-level
    # `orchestrator` at call time, so swap in a proxy that forwards every attribute
    # to this thread's own Orchestrator. No signature changes needed.
    class _OrchProxy:
        def __getattr__(self, name):
            return getattr(_components(), name)

    _components_base = bc.orchestrator
    bc.orchestrator = _OrchProxy()
    print(f"[stage 2] using {bc.__file__}")

    def _one(r):
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
            attempts = 1 + max(0, getattr(args, "retries", 0))
            for attempt in range(attempts):
                if RateLimit.exhausted:
                    row["error"] = "QUOTA_EXHAUSTED: " + RateLimit.exhausted[:200]
                    break
                RateLimit.wait()
                try:
                    out = bc.process_response(prompt=r["question"], response=raw, confidence_score=score)
                    row["status"] = out["status"]
                    row["final_response"] = out["final_response"]
                    row["evidence_source"] = out.get("evidence_source")
                    vr = out.get("verification_result") or {}
                    row["verdict"] = vr.get("verdict")
                    row["error"] = None
                    break
                except Exception as exc:  # one bad API call must not kill the run
                    row["error"] = f"{type(exc).__name__}: {exc}"[:300]
                    if not RateLimit.note(exc) and attempt + 1 < attempts:
                        time.sleep(min(30.0, 2.0 ** attempt))
            row["attempts"] = attempt + 1
            if row["status"] is None:
                # persistent failure: either count it as a failure, or fall back to the original answer
                row["status"] = "ERROR_FALLBACK" if getattr(args, "fallback_on_error", False) else "ERROR"
                row["final_response"] = raw
        row["seconds"] = round(time.time() - t0, 2)
        return row

    workers = max(1, args.workers)
    print(f"[stage 2] {len(records)} questions, {workers} workers")
    results = [None] * len(records)
    done = 0
    t_all = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, r): i for i, r in enumerate(records)}
        from concurrent.futures import as_completed
        for f in as_completed(futs):
            results[futs[f]] = f.result()
            done += 1
            if done % 10 == 0 or done == len(records):
                print(f"[stage 2] {done}/{len(records)}  ({time.time() - t_all:.0f}s)")
    return results, bc


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def arm_metrics(labels: List[str]) -> Dict[str, Any]:
    n = len(labels)
    c = Counter(labels)
    answered = c[CORRECT] + c[HALLUCINATED]
    failed = c[HALLUCINATED] + c[PROCESSING_ERROR]
    return {
        "n": n,
        "correct": c[CORRECT],
        "abstained": c[ABSTAINED],
        "hallucinated": c[HALLUCINATED],
        "processing_errors": c[PROCESSING_ERROR],
        "hallucination_rate": c[HALLUCINATED] / n if n else 0.0,
        "failure_rate": failed / n if n else 0.0,
        "accuracy": c[CORRECT] / n if n else 0.0,
        "abstention_rate": c[ABSTAINED] / n if n else 0.0,
        "hallucination_rate_among_answered": c[HALLUCINATED] / answered if answered else 0.0,
    }


def paired_bootstrap(base: List[str], bb: List[str], iters: int = 2000, seed: int = 0) -> Dict[str, float]:
    if len(base) != len(bb):
        raise ValueError("Paired bootstrap requires equal-length outcome lists.")
    if not base:
        return {
            "abs_reduction_ci95_low": 0.0,
            "abs_reduction_ci95_high": 0.0,
            "abs_failure_reduction_ci95_low": 0.0,
            "abs_failure_reduction_ci95_high": 0.0,
        }
    rng = np.random.default_rng(seed)
    b = np.array([x == HALLUCINATED for x in base], dtype=float)
    w = np.array([x == HALLUCINATED for x in bb], dtype=float)
    b_fail = np.array([x in (HALLUCINATED, PROCESSING_ERROR) for x in base], dtype=float)
    w_fail = np.array([x in (HALLUCINATED, PROCESSING_ERROR) for x in bb], dtype=float)
    n = len(b)
    idx = rng.integers(0, n, size=(iters, n))
    diffs = b[idx].mean(axis=1) - w[idx].mean(axis=1)  # positive = blackboard helps
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    failure_diffs = b_fail[idx].mean(axis=1) - w_fail[idx].mean(axis=1)
    failure_lo, failure_hi = np.percentile(failure_diffs, [2.5, 97.5])
    return {
        "abs_reduction_ci95_low": float(lo),
        "abs_reduction_ci95_high": float(hi),
        "abs_failure_reduction_ci95_low": float(failure_lo),
        "abs_failure_reduction_ci95_high": float(failure_hi),
    }


def build_report(records, rows, base_labels, bb_labels, args, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base = arm_metrics(base_labels)
    bb = arm_metrics(bb_labels)
    abs_red = base["hallucination_rate"] - bb["hallucination_rate"]
    rel_red = abs_red / base["hallucination_rate"] if base["hallucination_rate"] else 0.0

    transitions = Counter(f"{a} -> {b}" for a, b in zip(base_labels, bb_labels))
    by_status = Counter(r["status"] for r in rows)
    triggered = sum(1 for r in rows if r["status"] not in ("SKIPPED_LOW_RISK", "EMPTY_ANSWER", "ERROR", "ERROR_FALLBACK"))

    n = len(base_labels)
    # Processing errors are NOT fixes: the paired hallucination test only uses pairs where the blackboard produced an answer.
    ok_pairs = [(a, b) for a, b in zip(base_labels, bb_labels) if a != PROCESSING_ERROR and b != PROCESSING_ERROR]
    b_only = sum(1 for a, b in ok_pairs if a == HALLUCINATED and b != HALLUCINATED)
    w_only = sum(1 for a, b in ok_pairs if a != HALLUCINATED and b == HALLUCINATED)
    fail = lambda x: x in (HALLUCINATED, PROCESSING_ERROR)
    f_fixed = sum(1 for a, b in zip(base_labels, bb_labels) if fail(a) and not fail(b))
    f_broken = sum(1 for a, b in zip(base_labels, bb_labels) if not fail(a) and fail(b))
    extra = extra or {}
    rep = {
        "config": {
            "n_questions": len(records),
            "threshold": args.threshold,
            "default_score_when_no_probe": args.default_score,
            "knowledge_base": "none (closed-book)" if args.no_knowledge else "ORACLE: gold knowledge from eval items (upper bound, not closed-book)",
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
        "fallbacks_to_original_answer": by_status.get("ERROR_FALLBACK", 0),
        "wilson_ci95_hallucination_rate": {
            "without_blackboard": wilson_ci(base["hallucinated"], n),
            "with_blackboard": wilson_ci(bb["hallucinated"], n),
        },
        "wilson_ci95_failure_rate": {
            "without_blackboard": wilson_ci(round(base["failure_rate"] * n), n),
            "with_blackboard": wilson_ci(round(bb["failure_rate"] * n), n),
        },
        "mcnemar_exact_p_failure": mcnemar_exact(f_fixed, f_broken),
        "paired_fixed_vs_broken_failure": {"fixed": f_fixed, "broken": f_broken},
        "paired_pairs_used_for_hallucination_test": len(ok_pairs),
        "mcnemar_exact_p_hallucination": mcnemar_exact(b_only, w_only),
        "paired_fixed_vs_broken": {"fixed": b_only, "broken": w_only},
        **extra,
    }
    return rep


def format_summary(rep: Dict[str, Any]) -> str:
    b, w = rep["without_blackboard"], rep["with_blackboard"]
    pct = lambda x: f"{100 * x:5.1f}%"
    lines = [
        f"# Blackboard evaluation  (n={b['n']})",
        "",
        "| metric | without blackboard | with blackboard |",
        "|---|---|---|",
        f"| hallucination rate (wrong / all) | {pct(b['hallucination_rate'])} | {pct(w['hallucination_rate'])} |",
        f"| failure rate (wrong or processing error / all) | {pct(b['failure_rate'])} | {pct(w['failure_rate'])} |",
        f"| accuracy (correct / all) | {pct(b['accuracy'])} | {pct(w['accuracy'])} |",
        f"| abstention rate | {pct(b['abstention_rate'])} | {pct(w['abstention_rate'])} |",
        f"| hallucination rate among answered | {pct(b['hallucination_rate_among_answered'])} | {pct(w['hallucination_rate_among_answered'])} |",
        f"| counts (correct / abstained / wrong / errors) | {b['correct']} / {b['abstained']} / {b['hallucinated']} / {b['processing_errors']} | {w['correct']} / {w['abstained']} / {w['hallucinated']} / {w['processing_errors']} |",
        "",
        f"**Hallucination-rate reduction: {100 * rep['absolute_hallucination_reduction']:.1f} points "
        f"({100 * rep['relative_hallucination_reduction']:.1f}% relative)**, "
        f"95% paired-bootstrap CI: [{100 * rep['abs_reduction_ci95_low']:.1f}, {100 * rep['abs_reduction_ci95_high']:.1f}] points.",
        f"Failure-rate reduction (errors counted as failures): "
        f"{100 * (b['failure_rate'] - w['failure_rate']):.1f} points, "
        f"95% paired-bootstrap CI: [{100 * rep['abs_failure_reduction_ci95_low']:.1f}, "
        f"{100 * rep['abs_failure_reduction_ci95_high']:.1f}] points.",
        "",
        f"Blackboard ran on {rep['blackboard_triggered']} of {b['n']} answers. "
        f"Made worse (correct -> wrong): {rep['harmed']}. "
        f"Lost to over-abstaining (correct -> abstained): {rep['lost_to_over_abstention']}. "
        f"API errors: {rep['errors']}. Fell back to the original answer after retries: {rep['fallbacks_to_original_answer']}.",
        "",
        "Status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(rep["status_counts"].items())),
        "",
        "Transitions (without -> with blackboard):",
    ]
    for k, v in sorted(rep["transitions_baseline_to_blackboard"].items()):
        lines.append(f"- {k}: {v}")
    lo0, hi0 = rep["wilson_ci95_hallucination_rate"]["without_blackboard"]
    lo1, hi1 = rep["wilson_ci95_hallucination_rate"]["with_blackboard"]
    lines += ["",
              f"Wilson 95% CI on hallucination rate: without [{100*lo0:.1f}, {100*hi0:.1f}], with [{100*lo1:.1f}, {100*hi1:.1f}]. "
              f"Paired fixed/broken (error-free pairs only, n={rep['paired_pairs_used_for_hallucination_test']}) = "
              f"{rep['paired_fixed_vs_broken']['fixed']}/{rep['paired_fixed_vs_broken']['broken']}, "
              f"exact McNemar p = {rep['mcnemar_exact_p_hallucination']:.4f}. "
              f"Failure basis (errors count as failures): {rep['paired_fixed_vs_broken_failure']['fixed']}/"
              f"{rep['paired_fixed_vs_broken_failure']['broken']}, p = {rep['mcnemar_exact_p_failure']:.4f}.",
              f"Knowledge base: {rep['config']['knowledge_base']}."]
    if rep.get("strict_grading"):
        sg = rep["strict_grading"]
        lines += ["", "Grading sensitivity (strict match: answer must equal gold after normalisation):",
                  f"- hallucination rate without / with blackboard: {pct(sg['without_blackboard']['hallucination_rate'])} / {pct(sg['with_blackboard']['hallucination_rate'])}"]
    if rep.get("probe_detection"):
        pd_ = rep["probe_detection"]
        lines += ["", f"Probe detection on held-out eval answers: AUROC {pd_['auroc']:.3f} "
                      f"(95% CI {pd_['ci95_low']:.3f}-{pd_['ci95_high']:.3f}; n={pd_['n']}, wrong={pd_['n_wrong']})."]
    if rep.get("probe_detection", {}).get("at_threshold"):
        t = rep["probe_detection"]["at_threshold"]
        f = lambda v: "n/a" if v is None else f"{v:.3f}"
        ci = t["precision_ci95"]
        lines += [f"Probe at threshold {t['threshold']:.2f}: precision {f(t['precision'])}"
                  + (f" (95% CI {ci[0]:.3f}-{ci[1]:.3f})" if ci else "")
                  + f", recall {f(t['recall'])}, F1 {f(t['f1'])}; flags {100*t['flagged_rate']:.1f}% of answers "
                  f"(TP {t['tp']}, FP {t['fp']}, FN {t['fn']}, TN {t['tn']})."]
    if rep.get("complete_case"):
        cc = rep["complete_case"]
        lines += ["", f"Complete-case (only the {cc['n']} items where the blackboard ran; {cc['excluded']} errored/fell back excluded): "
                      f"hallucination rate {pct(cc['without_blackboard']['hallucination_rate'])} -> {pct(cc['with_blackboard']['hallucination_rate'])}, "
                      f"abstention {pct(cc['without_blackboard']['abstention_rate'])} -> {pct(cc['with_blackboard']['abstention_rate'])}, "
                      f"accuracy {pct(cc['without_blackboard']['accuracy'])} -> {pct(cc['with_blackboard']['accuracy'])}."]
    if rep.get("llm_only_control"):
        c = rep["llm_only_control"]
        if c.get("control_errors"):
            lines += ["", f"WARNING: {c['control_errors']} control calls failed after retries; the comparison below is NOT valid. "
                          "Rerun the same command: finished control answers are resumed from control_partial.jsonl."]
        lines += ["", "Control: strong LLM answering closed-book, no blackboard",
                  f"- hallucination rate {pct(c['metrics']['hallucination_rate'])}, accuracy {pct(c['metrics']['accuracy'])}, "
                  f"abstention {pct(c['metrics']['abstention_rate'])}",
                  f"- blackboard minus control hallucination rate: {100*c['blackboard_minus_control_hallucination']:+.1f} points "
                  f"(95% CI {100*c['ci95_low']:+.1f} to {100*c['ci95_high']:+.1f}); negative = blackboard better",
                  f"- same on failure rate (errors count as failures): {100*c['blackboard_minus_control_failure']:+.1f} points "
                  f"(95% CI {100*c['failure_ci95_low']:+.1f} to {100*c['failure_ci95_high']:+.1f})"]
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
    bbg.add_argument("--oracle_knowledge", action="store_true",
                     help="load the dataset's gold 'knowledge' into the KB. ORACLE setting: leaks the answer "
                          "source, report only as an upper bound. Default (off) is closed-book.")
    bbg.add_argument("--no_knowledge", action="store_true", help="deprecated: closed-book is now the default")
    bbg.add_argument("--enable_web", action="store_true", help="allow Tavily web-search fallback")
    bbg.add_argument("--isolate_memory", action="store_true",
                     help="every question sees an empty episodic memory (no cross-question leakage)")
    bbg.add_argument("--workers", type=int, default=8,
                     help="parallel questions in stage 2 (lower it if Groq returns 429s)")

    bbg.add_argument("--retries", type=int, default=6,
                     help="retry a failed blackboard call this many times (exponential backoff) before giving up")
    bbg.add_argument("--fallback_on_error", action="store_true",
                     help="if a call still fails after retries, keep the ORIGINAL answer (status ERROR_FALLBACK). "
                          "Fallbacks are counted and reported separately, plus a complete-case analysis on items where "
                          "the blackboard really ran. Without this flag such items are PROCESSING_ERROR failures.")

    bbg.add_argument("--preflight", type=int, default=0, metavar="N",
                     help="test N items, print tokens/item, projected total and Groq limits, then EXIT without writing results")
    bbg.add_argument("--retry_errors", action="store_true",
                     help="with --reuse_stage2: rerun ONLY items whose saved status is ERROR / ERROR_FALLBACK, keep the rest")
    bbg.add_argument("--reuse_stage2", action="store_true",
                     help="skip stage 2 and reuse <out_dir>/bb_rows.jsonl (refuses if settings/questions differ)")

    gd = ap.add_argument_group("leakage guards / controls")
    gd.add_argument("--probe_train_indices", default=None,
                    help="rows the probe was trained on (legacy probes), e.g. '0-99'; excluded from eval")
    gd.add_argument("--allow_train_overlap", action="store_true",
                    help="run even if train/test overlap cannot be ruled out (NOT paper-clean)")
    gd.add_argument("--llm_only_control", action="store_true",
                    help="also grade the strong LLM answering closed-book with no blackboard (needs GROQ_API_KEY)")

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

    args.no_knowledge = not args.oracle_knowledge
    ex = probe_exclusions(args)

    if args.answers_file:
        records = load_dataset(args.answers_file, "generic", limit=args.limit, seed=args.seed,
                               shuffle=args.shuffle, start=args.start, indices=indices,
                               exclude_ids=ex["ids"], exclude_questions=ex["questions"])
        if any("answer" not in r for r in records):
            ap.error("--answers_file rows need an 'answer' field (use stage-1 output, or add model answers).")
        print(f"Loaded {len(records)} records with answers from {args.answers_file}")
    else:
        records = load_dataset(args.dataset, args.format, args.q_field, args.a_field, args.k_field,
                               args.limit, args.seed, args.shuffle, start=args.start, indices=indices,
                               exclude_ids=ex["ids"], exclude_questions=ex["questions"])
        print(f"Loaded {len(records)} questions from {args.dataset}")
        records = run_generation_stage(records, args)
        with open(out_dir / "generations.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved generations -> {out_dir / 'generations.jsonl'}")
        if args.generate_only:
            return

    if args.preflight:
        enforce_probe_guard(args, ex, bool(args.probe) or any(r.get("prob_hallucinated") is not None for r in records))
        run_preflight(records, args)
        raise SystemExit(0)
    enforce_probe_guard(args, ex, bool(args.probe) or any(r.get("prob_hallucinated") is not None for r in records))
    print(f"Excluded {LOAD_STATS.get('excluded_probe_train', 0)} probe-training rows; "
          f"dropped {LOAD_STATS.get('duplicates_dropped', 0)} duplicate questions.")
    print("Using dataset rows:", [r["id"] for r in records])

    # ---- blackboard ------------------------------------------------------
    stage2_path = out_dir / "bb_rows.jsonl"
    if args.reuse_stage2 and stage2_path.exists():
        rows = load_stage2_rows(stage2_path, records, args)
        bc = load_blackboard_module(args)
        print(f"[stage 2] reused {len(rows)} saved rows from {stage2_path}")
        if args.retry_errors:
            bad = [i for i, x in enumerate(rows) if x["status"] in ("ERROR", "ERROR_FALLBACK")]
            print(f"[stage 2] --retry_errors: rerunning {len(bad)} failed items")
            if bad:
                new_rows, _ = run_blackboard_stage([records[i] for i in bad], args)
                for i, nr in zip(bad, new_rows):
                    rows[i] = nr
                save_stage2_rows(stage2_path, records, rows, args)
    else:
        rows, bc = run_blackboard_stage(records, args)
        save_stage2_rows(stage2_path, records, rows, args)
        print(f"[stage 2] checkpoint saved -> {stage2_path} (rerun with --reuse_stage2 to skip it)")

    # ---- grade both arms -------------------------------------------------
    judge = make_llm_judge(bc.groq_caller) if args.llm_judge else None
    def _pmap(fn, items):
        with ThreadPoolExecutor(max_workers=max(1, args.workers) if judge else 1) as ex:
            return list(ex.map(fn, items))

    base_labels = _pmap(lambda r: grade(r["answer"], r["gold"], args.match, judge), records)
    bb_labels = _pmap(
        lambda rx: PROCESSING_ERROR if rx[1]["status"] == "ERROR"
        else grade(rx[1]["final_response"], rx[0]["gold"], args.match, judge),
        list(zip(records, rows)),
    )

    extra: Dict[str, Any] = {}
    strict_base = [grade(r["answer"], r["gold"], "strict", judge) for r in records]
    strict_bb = [PROCESSING_ERROR if x["status"] == "ERROR" else grade(x["final_response"], r["gold"], "strict", judge)
                 for r, x in zip(records, rows)]
    extra["strict_grading"] = {"without_blackboard": arm_metrics(strict_base), "with_blackboard": arm_metrics(strict_bb)}
    pa = probe_auroc_report(records, base_labels, seed=args.seed)
    if pa:
        extra["probe_detection"] = pa
        pt = probe_threshold_report(records, base_labels, args.threshold)
        if pt:
            extra["probe_detection"]["at_threshold"] = pt
    # complete-case analysis: only items where the blackboard actually ran to completion
    cc = [i for i, x in enumerate(rows) if x["status"] not in ("ERROR", "ERROR_FALLBACK")]
    if len(cc) < len(rows):
        extra["complete_case"] = {"n": len(cc), "excluded": len(rows) - len(cc),
                                  "without_blackboard": arm_metrics([base_labels[i] for i in cc]),
                                  "with_blackboard": arm_metrics([bb_labels[i] for i in cc])}
    ctrl_answers = ctrl_labels = None
    if args.llm_only_control:
        ctrl_answers = run_llm_only_control(records, bc.groq_caller, workers=args.workers, retries=args.retries,
                                            checkpoint=out_dir / "control_partial.jsonl")
        ctrl_labels = [PROCESSING_ERROR if a.startswith("[CONTROL_ERROR") else grade(a, r["gold"], args.match, judge)
                       for r, a in zip(records, ctrl_answers)]
        d = paired_bootstrap(ctrl_labels, bb_labels, seed=args.seed)   # positive = blackboard better than control
        cm, bm = arm_metrics(ctrl_labels), arm_metrics(bb_labels)
        extra["llm_only_control"] = {
            "metrics": cm,
            "control_errors": sum(1 for a in ctrl_answers if a.startswith("[CONTROL_ERROR")),
            # blackboard minus control; negative = blackboard is better
            "blackboard_minus_control_hallucination": bm["hallucination_rate"] - cm["hallucination_rate"],
            "ci95_low": -d["abs_reduction_ci95_high"], "ci95_high": -d["abs_reduction_ci95_low"],
            "blackboard_minus_control_failure": bm["failure_rate"] - cm["failure_rate"],
            "failure_ci95_low": -d["abs_failure_reduction_ci95_high"], "failure_ci95_high": -d["abs_failure_reduction_ci95_low"],
        }
        with open(out_dir / "control_answers.jsonl", "w", encoding="utf-8") as f:
            for r, a, l in zip(records, ctrl_answers, ctrl_labels):
                f.write(json.dumps({"id": r["id"], "answer": a, "label": l}, ensure_ascii=False) + "\n")

    report = build_report(records, rows, base_labels, bb_labels, args, extra)
    write_manifest(out_dir, args, records, bc, ex)
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

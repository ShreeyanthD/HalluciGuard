"""
eval_selftest.py — offline smoke test for eval_pipeline.py.

No GPU, no API key, no model downloads. It swaps in:
  * a fake LLM that reads the evidence in each prompt and answers the way a
    careful verifier would, and
  * a tiny hashed bag-of-words embedder for ChromaDB.

It checks the grading rules and runs the real stage-2 code path
(process_response -> Orchestrator -> agents) end to end on
eval_data/sample_qa.jsonl. It does NOT measure real model quality.

    python eval_selftest.py
"""

import zlib
import json
import os
import re
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("GROQ_API_KEY", "selftest")
os.environ["CHROMA_PERSIST_DIRECTORY"] = tempfile.mkdtemp(prefix="selftest_chroma_")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "blackboard"))

import numpy as np
import chromadb

import eval_pipeline as ev


# ---- fake embedder so Chroma needs no model download ----------------------
class HashEF:
    def __init__(self): pass
    def name(self): return "hash_ef"
    def __call__(self, input):
        out = []
        for text in input:
            v = np.zeros(256, dtype=np.float32)
            for tok in re.findall(r"\w+", text.lower()):
                v[zlib.crc32(tok.encode()) % 256] += 1.0
            n = np.linalg.norm(v) or 1.0
            out.append(v / n)
        return out
    def embed_query(self, input): return self(input)
    @staticmethod
    def build_from_config(config): return HashEF()
    def get_config(self): return {}
    @staticmethod
    def is_legacy(): return False
    def default_space(self): return "cosine"
    def supported_spaces(self): return ["cosine", "l2", "ip"]


_orig_client = chromadb.PersistentClient
def _patched_client(*a, **kw):
    c = _orig_client(*a, **kw)
    og = c.get_or_create_collection
    c.get_or_create_collection = lambda **k: og(embedding_function=HashEF(), **k)
    return c
chromadb.PersistentClient = _patched_client

import blackboard_core as bc  # noqa: E402  (after the patch on purpose)


# ---- fake LLM --------------------------------------------------------------
def _between(text, tag):
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.S)
    return m.group(1) if m else ""

def fake_llm(prompt, **kw):
    if "ClaimExtractor" in prompt:
        q, a = _between(prompt, "USER_PROMPT"), _between(prompt, "ASSISTANT_RESPONSE")
        return json.dumps({"claim": f"{q} {a}", "flagged_span": a, "reason": "test"})
    if "VerifierAgent" in prompt:
        claim = _between(prompt, "CLAIM")
        ev = _between(prompt, "EVIDENCE")
        ids = re.findall(r"\[Evidence id=([^\]]+)\]", ev)
        if not ids:
            return json.dumps({"verdict": "INSUFFICIENT", "explanation": "no evidence",
                               "supporting_evidence_ids": [], "confidence": 0.2})
        ans = claim.split()[-1].lower()
        ok = ans in ev.lower()
        return json.dumps({"verdict": "SUPPORTED" if ok else "CONTRADICTED",
                           "explanation": "rule-based", "supporting_evidence_ids": ids[:1], "confidence": 0.9})
    if "operating in ABSTAIN mode" in prompt:
        return json.dumps({"abstained_response": "I'm not sure about that.", "correction_summary": "abstained"})
    if "CorrectionAgent" in prompt:
        ev = _between(prompt, "RETRIEVED_EVIDENCE")
        text = re.sub(r"\[Evidence id=[^\]]+\]\s*", "", ev).split("\n\n")[0]
        return json.dumps({"corrected_response": text, "correction_summary": "from evidence"})
    if "AnswerAgent" in prompt:
        ev = _between(prompt, "EVIDENCE")
        ids = re.findall(r"\[Evidence id=([^\]]+)\]", ev)
        if not ids:
            return json.dumps({"verdict": "INSUFFICIENT", "answer": "", "explanation": "", "supporting_evidence_ids": []})
        text = re.sub(r"\[Evidence id=[^\]]+\]\s*Text:\s*", "", ev).split("\n\n")[0]
        return json.dumps({"verdict": "ANSWER_FOUND", "answer": text, "explanation": "", "supporting_evidence_ids": ids[:1]})
    return json.dumps({"is_abstention": False, "explanation": "test"})

for agent in (bc.verifier_agent, bc.correction_agent, bc.claim_extractor,
              bc.abstention_detector, bc.answer_agent):
    agent.llm_caller = fake_llm
bc.abstention_detector.use_llm_fallback = True


# ---- 1. grading rules ------------------------------------------------------
def test_grading():
    g = ev.grade
    assert g("Paris", ["Paris"]) == ev.CORRECT
    assert g("The Paris.", ["paris"]) == ev.CORRECT
    assert g("Einstein", ["Albert Einstein"]) == ev.CORRECT            # lenient
    assert g("Einstein", ["Albert Einstein"], "strict") == ev.HALLUCINATED
    assert g("Unknown", ["Paris"]) == ev.ABSTAINED
    assert g("unknown.", ["Paris"]) == ev.ABSTAINED
    assert g("", ["Paris"]) == ev.ABSTAINED
    assert g("I'm not sure about that.", ["Paris"]) == ev.ABSTAINED
    assert g("London", ["Paris"]) == ev.HALLUCINATED
    assert g("Unknown", ["Unknown"]) == ev.ABSTAINED                   # abstention wins on the literal token
    # the detector in blackboard_core agrees that a bare "Unknown" is an abstention
    assert bc.abstention_detector._heuristic_match("Unknown")
    assert bc.abstention_detector._heuristic_match("Unknown.")
    assert not bc.abstention_detector._heuristic_match("Unknown Pleasures")
    print("ok  grading + abstention detection")


def test_processing_error_accounting():
    treated = [ev.CORRECT, ev.PROCESSING_ERROR]
    metrics = ev.arm_metrics(treated)
    intervals = ev.paired_bootstrap(
        [ev.CORRECT, ev.HALLUCINATED], treated, iters=100, seed=3
    )
    assert metrics["processing_errors"] == 1
    assert metrics["hallucination_rate"] == 0.0
    assert metrics["failure_rate"] == 0.5
    assert "abs_failure_reduction_ci95_low" in intervals
    assert "abs_failure_reduction_ci95_high" in intervals
    print("ok  processing errors counted separately from hallucinations")


# ---- 2. end-to-end stage 2 -------------------------------------------------
def test_stage2():
    recs = ev.load_dataset(str(ROOT / "eval_data" / "sample_qa.jsonl"), "halueval")
    wrong = {"Who wrote the play Hamlet?": "Marlowe",
             "What is the capital of Australia?": "Sydney",
             "Which planet is known as the Red Planet?": "Venus",
             "In which city is the Eiffel Tower located?": "Lyon",
             "Who discovered penicillin?": "Unknown",          # honest abstention from the base model
             "What is the largest planet in our solar system?": "Saturn"}
    for i, r in enumerate(recs):
        r["answer"] = wrong.get(r["question"], r["gold"][0])   # rest already correct
        r["prob_hallucinated"] = 0.9 if r["question"] in wrong else 0.1
    args = Namespace(threshold=0.70, default_score=1.0, chroma_dir=os.environ["CHROMA_PERSIST_DIRECTORY"],
                     no_knowledge=False, oracle_knowledge=True, enable_web=False, isolate_memory=False, match="lenient",
                     llm_judge=False, seed=0, workers=1)
    rows, _ = ev.run_blackboard_stage(recs, args)
    base = [ev.grade(r["answer"], r["gold"]) for r in recs]
    bb = [ev.grade(x["final_response"], r["gold"]) for r, x in zip(recs, rows)]
    rep = ev.build_report(recs, rows, base, bb, args)
    print(ev.format_summary(rep))

    assert rep["without_blackboard"]["hallucinated"] == 5, rep["without_blackboard"]
    # 2 of the 5 wrong answers legitimately survive here, and both are mock
    # artifacts: the fake verifier calls "Sydney" SUPPORTED because the
    # Canberra doc mentions Sydney, and the crude hash embedder retrieves the
    # Jupiter doc for the Red Planet question. A real verifier + embedder
    # should do better; this test only checks the plumbing and accounting.
    assert rep["with_blackboard"]["hallucinated"] <= 2, rep["with_blackboard"]
    assert rep["harmed"] == 0 and rep["errors"] == 0
    assert rep["status_counts"].get("SKIPPED_LOW_RISK") == 6
    assert rep["status_counts"].get("ABSTENTION_RESOLVED") == 1
    assert rep["absolute_hallucination_reduction"] > 0
    print("ok  stage 2 end to end")

    # empty knowledge base: blackboard can't verify anything -> should abstain, not invent
    args.no_knowledge = True
    rows2, _ = ev.run_blackboard_stage(recs, args)
    bb2 = [ev.grade(x["final_response"], r["gold"]) for r, x in zip(recs, rows2)]
    assert bb2.count(ev.HALLUCINATED) == 0, bb2
    assert bb2.count(ev.ABSTAINED) >= 5
    print("ok  empty-KB ablation abstains instead of hallucinating")


def test_stats_and_guards():
    assert abs(ev.auroc([.9, .8, .2, .1], [1, 1, 0, 0]) - 1.0) < 1e-9
    assert abs(ev.auroc([.5, .5, .5, .5], [1, 0, 1, 0]) - 0.5) < 1e-9
    lo, hi = ev.wilson_ci(5, 10); assert 0.2 < lo < 0.5 < hi < 0.8
    assert ev.mcnemar_exact(0, 0) == 1.0 and ev.mcnemar_exact(10, 0) < 0.01
    # probe-training rows and duplicate questions are excluded
    import json as _j, tempfile as _t
    rows = [{"question": f"Q{i}?", "right_answer": "a", "knowledge": "k"} for i in range(10)]
    rows.append({"question": "Q3?", "right_answer": "a", "knowledge": "k"})   # duplicate
    f = _t.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    f.write("\n".join(_j.dumps(r) for r in rows)); f.close()
    recs = ev.load_dataset(f.name, "halueval", exclude_ids=set(range(0, 5)))
    assert [r["id"] for r in recs] == [5, 6, 7, 8, 9], [r["id"] for r in recs]
    assert ev.LOAD_STATS["excluded_probe_train"] == 6
    recs = ev.load_dataset(f.name, "halueval", exclude_questions={ev.norm_q("Q1?")})
    assert "Q1?" not in [r["question"] for r in recs] and ev.LOAD_STATS["duplicates_dropped"] == 1
    # guard refuses unknown-overlap probe runs
    ns = Namespace(allow_train_overlap=False)
    try:
        ev.enforce_probe_guard(ns, {"source": None}, True); raise AssertionError("guard did not fire")
    except SystemExit:
        pass
    ev.enforce_probe_guard(ns, {"source": None}, False)           # no probe scores -> fine
    ev.enforce_probe_guard(ns, {"source": "x"}, True)             # overlap source known -> fine
    # control arm uses the supplied caller
    out = ev.run_llm_only_control([{"question": "Who wrote Hamlet?"}], lambda p, temperature=0: "Shakespeare")
    assert out == ["Shakespeare"]
    # processing errors must not be counted as hallucination fixes
    _A = lambda: Namespace(no_knowledge=True, enable_web=False, isolate_memory=False, llm_judge=False, match="lenient", threshold=.7, default_score=1.0, seed=0, probe_train_indices=None)
    recs2 = [{"id": i, "question": f"q{i}", "gold": ["a"], "answer": "wrong", "prob_hallucinated": .9} for i in range(4)]
    rows2 = [{"status": "ERROR", "verdict": "", "final_response": "", "evidence_source": "", "error": "x"} for _ in range(4)]
    rep2 = ev.build_report(recs2, rows2, [ev.HALLUCINATED] * 4, [ev.PROCESSING_ERROR] * 4, _A())
    assert rep2["paired_fixed_vs_broken"] == {"fixed": 0, "broken": 0}, rep2["paired_fixed_vs_broken"]
    assert rep2["paired_pairs_used_for_hallucination_test"] == 0
    print("ok  stats, probe-overlap guard, dedupe, control arm, errors-not-fixes")


def test_preflight():
    import io, contextlib
    recs = ev.load_dataset(str(ROOT / "eval_data" / "sample_qa.jsonl"), "halueval")
    for r in recs:
        r["answer"] = "WRONG"; r["prob_hallucinated"] = 0.9
    a = Namespace(preflight=3, threshold=0.7, default_score=1.0, chroma_dir=os.environ["CHROMA_PERSIST_DIRECTORY"],
                  no_knowledge=True, oracle_knowledge=False, enable_web=False, isolate_memory=False, match="lenient",
                  llm_judge=False, seed=0, workers=4, retries=0, fallback_on_error=True, llm_only_control=True)
    import blackboard_core as bc0
    class _U: total_tokens = 100
    class _R: usage = _U()
    class _Comp:
        def create(self, *a, **k): return _R()
        class with_raw_response:
            @staticmethod
            def create(*a, **k):
                class H:
                    headers = {"x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-requests": "500"}
                return H()
    fake_comp = _Comp()
    real_client = bc0._groq_client
    class _Chat: completions = fake_comp
    class _Client: chat = _Chat()
    real_pr, real_gc = bc0.process_response, bc0.groq_caller
    def fake_pr(prompt, response, confidence_score):
        fake_comp.create(); return {"status": "BLACKBOARD_PROCESSED", "final_response": "fixed", "verification_result": {}}
    bc0._groq_client, bc0.process_response = _Client(), fake_pr
    bc0.groq_caller = lambda p, temperature=0: fake_comp.create() and "x"
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            ev.run_preflight(recs, a)
    finally:
        bc0._groq_client, bc0.process_response, bc0.groq_caller = real_client, real_pr, real_gc
    out = buf.getvalue()
    assert "100 tokens per item" in out and "PROJECTED full run" in out and "--workers" in out, out
    print("ok  preflight")


def test_rate_limit_handling():
    P = ev.RateLimit.parse_wait
    assert P("Please try again in 7m12.5s. Need more") == 432.5
    assert abs(P("... try again in 560ms") - 0.56) < 1e-9 and P("try again in 1.2s") == 1.2 and P("nothing") is None
    ev.RateLimit.until, ev.RateLimit.exhausted = 0.0, None
    assert ev.RateLimit.note(RuntimeError("Error code: 429 - rate limit ... Please try again in 3s")) is True
    assert 3.0 < ev.RateLimit.until - ev.time.monotonic() < 6.0          # shared pause set (3s * 1.1 + 1)
    assert ev.RateLimit.note(ValueError("boom")) is False
    ev.RateLimit.until = 0.0
    ev.RateLimit.note(RuntimeError("429 Rate limit reached ... on tokens per day (TPD): Limit 200000 ... Please try again in 40m3s"))
    assert ev.RateLimit.exhausted and "tokens per day" in ev.RateLimit.exhausted
    # once exhausted, no API calls are made and items are marked as errors
    n = {"c": 0}
    def caller(prompt, temperature=0):
        n["c"] += 1; return "x"
    out = ev.run_llm_only_control([{"id": 1, "question": "q"}], caller, workers=1, retries=3)
    assert n["c"] == 0 and out[0].startswith("[CONTROL_ERROR")
    ev.RateLimit.until, ev.RateLimit.exhausted = 0.0, None
    # a worker that gets a 429 pauses, then succeeds on retry
    state = {"c": 0}
    def flaky429(prompt, temperature=0):
        state["c"] += 1
        if state["c"] == 1:
            raise RuntimeError("Error code: 429 ... try again in 0.01s")
        return "ok"
    real = ev.time.sleep
    out = ev.run_llm_only_control([{"id": 1, "question": "q"}], flaky429, workers=1, retries=2)
    assert out == ["ok"] and state["c"] == 2
    ev.RateLimit.until, ev.RateLimit.exhausted = 0.0, None
    print("ok  rate-limit parse, shared pause, daily-quota stop")


def test_control_parallel_resume_and_checkpoint():
    import tempfile, threading, time as _t
    d = Path(tempfile.mkdtemp())
    recs = [{"id": i, "question": f"Q{i}?"} for i in range(20)]
    calls, lock, fails = [], threading.Lock(), {"Q3?": 1}
    def caller(prompt, temperature=0):
        q = prompt.rsplit("Question: ", 1)[1]
        with lock:
            calls.append(q)
            if fails.get(q, 0) > 0:
                fails[q] -= 1
                raise RuntimeError("transient 500")
        _t.sleep(0.05)
        return "ans-" + q
    ev.RateLimit.until, ev.RateLimit.exhausted = 0.0, None
    ev_sleep, ev.time.sleep = ev.time.sleep, lambda s: None
    try:
        t0 = _t.time()
        out = ev.run_llm_only_control(recs, caller, workers=10, retries=2, checkpoint=d / "c.jsonl")
        assert out == ["ans-" + r["question"] for r in recs] and _t.time() - t0 < 0.5   # parallel, retried
        calls.clear()
        out2 = ev.run_llm_only_control(recs, caller, workers=4, retries=0, checkpoint=d / "c.jsonl")
        assert out2 == out and calls == []                                               # fully resumed, no API calls
        (d / "c.jsonl").write_text("".join(l for l in open(d / "c.jsonl") if '"Q5?"' not in l and '"Q6?"' not in l))
        ev.run_llm_only_control(recs, caller, workers=4, retries=0, checkpoint=d / "c.jsonl")
        assert sorted(calls) == ["Q5?", "Q6?"]                                           # only the missing ones
    finally:
        ev.time.sleep = ev_sleep
    # stage-2 checkpoint round trip, and refusal on mismatched settings / questions
    rows = [{"status": "BLACKBOARD_PROCESSED", "final_response": "x", "verdict": None, "error": None} for _ in recs]
    a = Namespace(threshold=.7, default_score=1.0, no_knowledge=True, enable_web=False, isolate_memory=False, fallback_on_error=True, retries=3)
    ev.save_stage2_rows(d / "bb.jsonl", recs, rows, a)
    assert ev.load_stage2_rows(d / "bb.jsonl", recs, a) == rows
    for bad_args, bad_recs in ((Namespace(**{**vars(a), "threshold": .5}), recs), (a, recs[:-1])):
        try:
            ev.load_stage2_rows(d / "bb.jsonl", bad_recs, bad_args); raise AssertionError("should refuse")
        except SystemExit:
            pass
    print("ok  control parallel + resume, stage-2 checkpoint/reuse")


def test_retry_and_fallback():
    recs = ev.load_dataset(str(ROOT / "eval_data" / "sample_qa.jsonl"), "halueval")
    for r in recs:
        r["answer"] = "WRONG"
        r["prob_hallucinated"] = 0.9
    bad = {recs[0]["question"], recs[1]["question"]}
    calls = {}
    def flaky(prompt, response, confidence_score):
        calls[prompt] = calls.get(prompt, 0) + 1
        if prompt in bad:
            raise RuntimeError("transient 500")
        if prompt == recs[2]["question"] and calls[prompt] < 3:
            raise RuntimeError("503")                      # succeeds on 3rd attempt
        return {"status": "BLACKBOARD_PROCESSED", "final_response": "fixed", "verification_result": {}}
    import blackboard_core as bc0
    real, real_sleep = bc0.process_response, ev.time.sleep
    bc0.process_response, ev.time.sleep = flaky, lambda s: None
    try:
        base = dict(threshold=0.7, default_score=1.0, chroma_dir=os.environ["CHROMA_PERSIST_DIRECTORY"], no_knowledge=True,
                    oracle_knowledge=False, enable_web=False, isolate_memory=False, match="lenient", llm_judge=False,
                    seed=0, workers=1, retries=2)
        rows, _ = ev.run_blackboard_stage(recs, Namespace(**base, fallback_on_error=True))
        assert rows[0]["status"] == rows[1]["status"] == "ERROR_FALLBACK" and rows[0]["final_response"] == "WRONG"
        assert rows[2]["status"] == "BLACKBOARD_PROCESSED" and rows[2]["attempts"] == 3
        assert calls[recs[0]["question"]] == 3                                  # 1 try + 2 retries
        rows2, _ = ev.run_blackboard_stage(recs, Namespace(**base, fallback_on_error=False))
        assert rows2[0]["status"] == "ERROR"
    finally:
        bc0.process_response, ev.time.sleep = real, real_sleep
    base_l = [ev.grade(r["answer"], r["gold"]) for r in recs]
    bb_l = [ev.PROCESSING_ERROR if x["status"] == "ERROR" else ev.grade(x["final_response"], r["gold"]) for r, x in zip(recs, rows)]
    rep = ev.build_report(recs, rows, base_l, bb_l, Namespace(**base, fallback_on_error=True))
    assert rep["fallbacks_to_original_answer"] == 2 and rep["errors"] == 0
    # probe precision/recall: all answers wrong and all flagged -> precision 1, recall 1
    pt = ev.probe_threshold_report(recs, base_l, 0.7)
    assert pt["precision"] == 1.0 and pt["recall"] == 1.0 and pt["fp"] == 0
    pt2 = ev.probe_threshold_report([{"prob_hallucinated": s} for s in (.9, .8, .2, .1)],
                                    [ev.HALLUCINATED, ev.CORRECT, ev.HALLUCINATED, ev.CORRECT], 0.5)
    assert (pt2["tp"], pt2["fp"], pt2["fn"], pt2["tn"]) == (1, 1, 1, 1) and abs(pt2["f1"] - 0.5) < 1e-9
    print("ok  retries, fallback-on-error, probe precision/recall")


if __name__ == "__main__":
    test_grading()
    test_processing_error_accounting()
    test_stats_and_guards()
    test_stage2()
    test_retry_and_fallback()
    test_control_parallel_resume_and_checkpoint()
    test_rate_limit_handling()
    test_preflight()
    print("\nAll self-tests passed.")

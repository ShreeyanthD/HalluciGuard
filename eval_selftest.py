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

import json
import hashlib
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
                slot = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:8], 'big') % 256
                v[slot] += 1.0
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
                     no_knowledge=False, enable_web=False, isolate_memory=False, match="lenient",
                     llm_judge=False, seed=0)
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


if __name__ == "__main__":
    test_grading()
    test_stage2()
    print("\nAll self-tests passed.")

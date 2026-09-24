"""
Applies Task 3 (3.1 - 3.4) to blackboard_imp2.ipynb and adds a mock test cell.

Usage:  python3 patch_task3.py [path/to/blackboard_imp2.ipynb]

Safety: every edit must match EXACTLY ONE place, otherwise nothing is written.
"""
import json
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "blackboard_imp2.ipynb"

raw = open(PATH, encoding="utf-8").read()
nb = json.loads(raw)

# keep the file's original JSON indent so the git diff stays small
indent = 1
for candidate in (1, 2, 4):
    if json.dumps(nb, indent=candidate, ensure_ascii=False) + "\n" == raw:
        indent = candidate
        break

EDITS = []

# ---------------- 3.1  save gate ----------------
EDITS.append((
"3.1a signature",
'''    ) -> str:
        """Store the completed verification outcome in episodic memory."""''',
'''    ) -> Optional[str]:
        """Store a verification outcome only if it is grounded. Returns record id, or None if skipped."""''',
))

EDITS.append((
"3.1b gate",
'''        if not claim:
            raise ValueError("Cannot log memory without a claim.")

        record_id = str(uuid.uuid4())''',
'''        if not claim:
            raise ValueError("Cannot log memory without a claim.")

        verification_result = blackboard.read("verification_result", {}) or {}
        evidence_ids = verification_result.get("supporting_evidence_ids", []) or []
        correction_result = blackboard.read("correction_result", None)

        grounded = (
            (verdict == "SUPPORTED" and len(evidence_ids) > 0)
            or (
                verdict == "CONTRADICTED"
                and len(evidence_ids) > 0
                and bool(correction_result)
            )
        )

        if not grounded:
            blackboard.write(
                "memory_write_skipped",
                f"ungrounded (verdict={verdict}, evidence_ids={len(evidence_ids)})",
                author=self.__class__.__name__,
            )
            return None

        record_id = str(uuid.uuid4())''',
))

# ---------------- 3.2  stamp metadata ----------------
EDITS.append((
"3.2 metadata",
'''            "confidence_score": float(confidence_score),
            "verified_at": utc_now_iso(),
        }''',
'''            "confidence_score": float(confidence_score),
            "grounded": True,
            "evidence_ids": metadata_safe(evidence_ids),
            "verifier_confidence": float(blackboard.read("verifier_confidence", 0.0)),
            "verified_at": utc_now_iso(),
        }''',
))

# ---------------- 3.3  reuse check ----------------
EDITS.append((
"3.3 reuse check",
'''        if verdict not in VerifierAgent.VALID_VERDICTS:
            return None

        return {
            "verdict": verdict,
            "explanation": (
                "Reused a sufficiently similar past verification result. "
                + str(metadata.get("explanation", ""))
            ).strip(),
            "supporting_evidence_ids": [],
            "confidence": 1.0,''',
'''        if verdict not in {"SUPPORTED", "CONTRADICTED"}:
            return None

        if metadata.get("grounded") is not True:
            return None

        try:
            evidence_ids = json.loads(metadata.get("evidence_ids", "[]"))
        except (TypeError, ValueError):
            return None

        if not isinstance(evidence_ids, list) or not evidence_ids:
            return None

        stored_confidence = metadata.get("verifier_confidence")

        if isinstance(stored_confidence, bool) or not isinstance(
            stored_confidence, (int, float)
        ):
            return None

        return {
            "verdict": verdict,
            "explanation": (
                "Reused a sufficiently similar past verification result. "
                + str(metadata.get("explanation", ""))
            ).strip(),
            "supporting_evidence_ids": evidence_ids,
            "confidence": max(0.0, min(1.0, float(stored_confidence))),''',
))

# ---------------- 3.4  two small fixes in run() ----------------
EDITS.append((
"3.4a no old-response replay",
'''        if verification_source == "episodic_memory" and verification_result.get("final_response"):''',
'''        if (
            verification_source == "episodic_memory"
            and verdict == "CONTRADICTED"
            and verification_result.get("final_response")
        ):''',
))

EDITS.append((
"3.4b no re-saving replays",
'''        memory_record_id = self.memory_agent.log_verified_claim(
            self.blackboard
        )''',
'''        if verification_source == "episodic_memory":
            memory_record_id = None
        else:
            memory_record_id = self.memory_agent.log_verified_claim(
                self.blackboard
            )''',
))

TEST_CELL = '''# ---- Task 3 mock tests (no Gemini, no real ChromaDB) ----
class FakeCollection:
    def __init__(self):
        self.rows = []
    def count(self):
        return len(self.rows)
    def add(self, ids, documents, metadatas):
        for i, d, m in zip(ids, documents, metadatas):
            self.rows.append((i, d, m))
    def query(self, query_texts, n_results, include):
        q = query_texts[0]
        rows = sorted(self.rows, key=lambda r: 0.0 if r[1] == q else 1.0)[:n_results]
        return {
            "ids": [[r[0] for r in rows]],
            "documents": [[r[1] for r in rows]],
            "metadatas": [[r[2] for r in rows]],
            "distances": [[0.0 if r[1] == q else 1.0 for r in rows]],
        }

class FakeRetrieval:
    top_k = 5
    def retrieve(self, blackboard, top_k=None):
        blackboard.write(
            "retrieved_evidence",
            [{"id": "doc-1", "text": "Fleming discovered penicillin.",
              "metadata": {}, "distance": 0.1}],
            author="FakeRetrieval",
        )

class FakeVerifier:
    def __init__(self, result):
        self.result = result
        self.calls = 0
    def verify(self, blackboard):
        self.calls += 1
        r = self.result
        blackboard.update(
            {
                "verification_result": r,
                "verification_verdict": r["verdict"],
                "verification_explanation": r["explanation"],
                "verifier_confidence": r["confidence"],
            },
            author="FakeVerifier",
        )
        return r

class FakeCorrection:
    def correct(self, blackboard):
        r = {"corrected_response": "Corrected text.", "correction_summary": "fixed"}
        blackboard.update(
            {"correction_result": r, "final_response": r["corrected_response"]},
            author="FakeCorrection",
        )
        return r

def make_orch(verifier, mem):
    return Orchestrator(
        Blackboard(),
        MemoryAgent(collection=mem, similarity_distance_threshold=0.35, top_k=3),
        FakeRetrieval(),
        verifier,
        FakeCorrection(),
        max_verification_rounds=2,
    )

def run_one(orch, claim="c1"):
    return orch.run(prompt="p", response="original text", claim=claim, confidence_score=0.9)

# Test 1: INSUFFICIENT -> nothing written
mem = FakeCollection()
v = FakeVerifier({"verdict": "INSUFFICIENT", "explanation": "no docs",
                  "supporting_evidence_ids": [], "confidence": 0.4})
out = run_one(make_orch(v, mem))
print("T1", mem.count(), out["memory_record_id"])
assert mem.count() == 0 and out["memory_record_id"] is None

# Test 2: grounded CONTRADICTED -> written once, reused, confidence != 1.0
mem = FakeCollection()
v1 = FakeVerifier({"verdict": "CONTRADICTED", "explanation": "conflicts",
                   "supporting_evidence_ids": ["doc-1"], "confidence": 0.9})
run_one(make_orch(v1, mem))
assert mem.count() == 1
v2 = FakeVerifier({"verdict": "INSUFFICIENT", "explanation": "x",
                   "supporting_evidence_ids": [], "confidence": 0.0})
out = run_one(make_orch(v2, mem))
print("T2", out["verification_source"], out["verification_result"]["confidence"], v2.calls, mem.count())
assert out["verification_source"] == "episodic_memory"
assert out["verification_result"]["confidence"] == 0.9
assert v2.calls == 0 and mem.count() == 1

# Test 3: old entry without `grounded` -> not reused
mem = FakeCollection()
mem.add(ids=["old"], documents=["c1"], metadatas=[{
    "verdict": "CONTRADICTED", "explanation": "x",
    "final_response": "Old fabricated text", "confidence_score": 0.9,
    "verified_at": "t"}])
v3 = FakeVerifier({"verdict": "INSUFFICIENT", "explanation": "x",
                   "supporting_evidence_ids": [], "confidence": 0.0})
out = run_one(make_orch(v3, mem))
print("T3", out["verification_source"], v3.calls)
assert out["verification_source"] == "retrieval_and_gemini" and v3.calls >= 1

# Test 4: SUPPORTED but no evidence ids -> not written
mem = FakeCollection()
v4 = FakeVerifier({"verdict": "SUPPORTED", "explanation": "x",
                   "supporting_evidence_ids": [], "confidence": 0.9})
out = run_one(make_orch(v4, mem))
print("T4", mem.count())
assert mem.count() == 0

print("ALL TASK 3 TESTS PASSED")
'''


def src_of(cell):
    return "".join(cell["source"])


# ---- dry run: check every edit matches exactly once before writing anything ----
plan = []
for name, find, repl in EDITS:
    hits = [
        i for i, c in enumerate(nb["cells"])
        if c["cell_type"] == "code" and src_of(c).count(find) == 1
    ]
    total = sum(src_of(c).count(find) for c in nb["cells"] if c["cell_type"] == "code")
    if len(hits) != 1 or total != 1:
        sys.exit(f"STOP: edit '{name}' matched {total} places (need exactly 1). Nothing written.")
    plan.append((hits[0], name, find, repl))

if any("Task 3 mock tests" in src_of(c) for c in nb["cells"]):
    sys.exit("STOP: test cell already present. Looks like this was already applied.")

for cell_idx, name, find, repl in plan:
    new_src = src_of(nb["cells"][cell_idx]).replace(find, repl)
    nb["cells"][cell_idx]["source"] = new_src.splitlines(keepends=True)
    print(f"applied {name} in cell {cell_idx}")

# test cell: reuse the trailing empty cell if there is one, else append
last = nb["cells"][-1]
if last["cell_type"] == "code" and not src_of(last).strip():
    last["source"] = TEST_CELL.splitlines(keepends=True)
    print(f"test cell written into empty last cell ({len(nb['cells']) - 1})")
else:
    nb["cells"].append({
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": TEST_CELL.splitlines(keepends=True),
    })
    print("test cell appended as new last cell")

open(PATH, "w", encoding="utf-8").write(
    json.dumps(nb, indent=indent, ensure_ascii=False) + "\n"
)
print("done. now run: git diff --stat")

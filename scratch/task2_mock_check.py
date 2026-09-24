import sys

class FakeBlackboard:
    def __init__(self, claim):
        self._data = {"claim": claim}
    def read(self, key, default=None):
        return self._data.get(key, default)

class FakeCorrectionAgent:
    def __init__(self):
        self.called = False
    def correct(self, blackboard):
        self.called = True
        return {"corrected_response": "CORRECTED"}

def run_case(verdict, evidence_ids):
    bb = FakeBlackboard("test claim")
    correction_agent = FakeCorrectionAgent()
    response = "ORIGINAL"
    verification_result = {"supporting_evidence_ids": evidence_ids}
    if verdict == "SUPPORTED":
        final_response = response
    elif verdict == "CONTRADICTED" and verification_result.get("supporting_evidence_ids"):
        correction_result = correction_agent.correct(bb)
        final_response = correction_result["corrected_response"]
    else:
        claim_text = bb.read("claim", "test claim")
        final_response = f"{response}\n\n[Note: the claim \"{claim_text}\" could not be verified against available evidence and has not been changed or corrected.]"
    return correction_agent.called, final_response

cases = [
    ("SUPPORTED", []),
    ("CONTRADICTED", ["evidence-1"]),
    ("CONTRADICTED", []),
    ("INSUFFICIENT", []),
]
expected_called = [False, True, False, False]

all_pass = True
for (verdict, ids), expect in zip(cases, expected_called):
    called, resp = run_case(verdict, ids)
    ok = called == expect
    all_pass = all_pass and ok
    print(f"{verdict} evidence={ids}: correct() called={called} expected={expect} PASS={ok}")

print("ALL PASS" if all_pass else "FAILURE")
sys.exit(0 if all_pass else 1)

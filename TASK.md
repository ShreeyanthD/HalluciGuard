Work in this session yourself. Do not delegate to subagents. Do not ask me questions.

Scope: exactly one text replacement inside Orchestrator.run(). Nothing else. Do not touch CorrectionAgent, VerifierAgent, or any other method.

Find this exact block inside Orchestrator.run() (the notebook cell containing "class Orchestrator"):

        elif verdict == "SUPPORTED":
            final_response = response
            self.blackboard.write(
                "final_response",
                final_response,
                author=self.__class__.__name__,
            )
        else:
            correction_result = self.correction_agent.correct(
                self.blackboard
            )
            final_response = correction_result["corrected_response"]

Replace it with exactly this:

        elif verdict == "SUPPORTED":
            final_response = response
            self.blackboard.write(
                "final_response",
                final_response,
                author=self.__class__.__name__,
            )
        elif verdict == "CONTRADICTED" and verification_result.get("supporting_evidence_ids"):
            correction_result = self.correction_agent.correct(
                self.blackboard
            )
            final_response = correction_result["corrected_response"]
        else:
            claim_text = self.blackboard.read("claim", claim)
            final_response = (
                f"{response}\n\n"
                f"[Note: the claim \"{claim_text}\" could not be verified "
                f"against available evidence and has not been changed or "
                f"corrected.]"
            )
            self.blackboard.write(
                "final_response",
                final_response,
                author=self.__class__.__name__,
            )

Apply this now. Do not stop after locating it — write the change to blackboard_imp2.ipynb.

If the exact block is not found verbatim (e.g. whitespace differs), paste the actual current text you found instead of guessing a replacement, and stop there — do not improvise.

If the edit tool itself errors, paste the full exact error message. Do not report "unconfirmed" with no detail — either it worked, or here is the exact reason it didn't.

After a successful edit, write scratch/task2_mock_check.py (do not commit) with this exact content:

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

Run it: python scratch/task2_mock_check.py
Paste the exact real stdout.

Do not commit anything yet. Do not attempt any real Gemini call.

End: paste the exact diff applied to Orchestrator.run(), and the exact real stdout from running the script.

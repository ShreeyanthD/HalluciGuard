Work in this session yourself. Do not delegate to subagents. Do not ask me questions.

Scope: a mocked dry run only, no real Gemini calls, no quota usage. Do not modify VerifierAgent.verify() itself.

1. Write a standalone script scratch/task1_mock_check.py (do not commit it).
2. In it: create a Blackboard(), write a test claim, and write retrieved_evidence directly as a list of two dicts with fields id/text/metadata/distance — e.g. ids "real-evidence-1" and "real-evidence-2" — bypassing RetrievalAgent entirely (no live calls needed for this check).
3. Create a VerifierAgent instance passing a mock gemini_caller function instead of the real one — a plain function that ignores its arguments and returns this fixed JSON string: '{"verdict": "SUPPORTED", "explanation": "test", "supporting_evidence_ids": ["real-evidence-1", "some-fake-id-not-in-evidence"], "confidence": 0.9}'
4. Call verify() with this mocked agent and the blackboard from step 2.
5. Print the result's supporting_evidence_ids.
6. Actually run the script and paste the real output. Expected correct output: supporting_evidence_ids should contain only "real-evidence-1" — "some-fake-id-not-in-evidence" must be filtered out. If it isn't filtered out, that's a real bug, report it exactly, don't fix it yourself yet.
7. If and only if the output is correct: git add -A && git commit -m "task 1: real evidence ids in verifier, mock-verified" (do not include scratch/ in the commit). Paste git log --oneline -n 3 after.

End: paste the exact script, the exact real stdout, and the git log output.

End: paste the raw output of `git log --oneline -n 3`, and state plainly whether the codebase is notebook-only or already has extracted .py files.

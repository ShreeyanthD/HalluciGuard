This should be the final task because adding Markdown cells changes notebook indexes.
Clean and regenerate the saved execution state in blackboard_imp2.ipynb.

Phase 1:
- Clear outputs from every code cell.
- Set every execution_count to null.
- Verify that no cell source changed.

Phase 2:
- Execute the notebook in dependency order through component initialization.
- Identify cells by content and notebook order rather than relying only on old numeric indexes.
- Do not run Gemini calls.
- Run the Paris verification test.
- Run all four executable demonstrations.
- Execute the SAE helper-definition cell before its demonstration.

Paris verification:

_r = orchestrator.run(
    prompt="p",
    response="Paris is the capital of France.",
    claim="Paris is the capital of France.",
    confidence_score=0.9,
)
print(_r["verification_result"])
print(
    "evidence label check:",
    "[Evidence id=" in str(_r.get("blackboard", {})),
)

Expected:
- Verdict is SUPPORTED.
- supporting_evidence_ids is non-empty.
- Evidence-label check is True.
- No exceptions occur.

Requirements:
- Show the complete Paris verification output.
- Show the complete output of each of the four executable demonstrations.
- Report exceptions honestly.
- Save consistent outputs and execution counters.
- Verify that executed code-cell counters increase monotonically.
- Verify that no source code changed during execution.

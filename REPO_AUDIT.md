# Repository audit

Audited 2026-10-06, branch `feature/blackboard_fix`, starting commit `cac4d54`.

| Area | Existing behavior | Research change |
|---|---|---|
| Generation | Qwen2.5-7B, CUDA-only, one-word factual prompt | Separate frozen CPU/CUDA local backend with task prompts and hooks on every decoder block |
| Features | Answer-span means from layer 20 | All-block answer means and last-answer-token features; embedding and final normalization excluded |
| Labels | Interactive manual HaluEval labels | Local JSONL and TruthfulQA/GSM8K/MBPP adapters; strict factual/numeric match and bounded Python tests |
| Detection | Scaler and logistic regression | Single-family/dataset, pooled, all-layer ERM, group-DRO, CLAP-style, trajectory, and text baselines |
| Blackboard | Shared retrieval/verification state using Groq, Chroma, optional Tavily | Separate typed research state chooses same-model interventions by estimated task, risk, and remaining budget |
| Evaluation | Factual answer comparison, gold-context retrieval, paired bootstrap | Leave-one-family-out; source-only calibration; group bootstrap; three seeds and explicit skips |
| Reproduction | Legacy CLI and API routes | New `data.build`, `eval.run` and offline smoke commands with configs and manifests |

The original implementation remains available. The new research CLI imports
`blackboard.research`, avoiding the legacy module's API-key and Chroma
initialization at import time. The duplicate root and package blackboard cores
are different; legacy entry points select them differently. They are not used
for the new research results.

The existing `eval_data/sample_qa.jsonl` **is present**. The legacy self-test's
hashed embedder used Python's process-randomized `hash`, making retrieval vary
between runs. Validation details and repairs are recorded in `VALIDATION.md`.

## Abstention routing

The old README says honest abstentions pass through. Actual package code checks
abstention before its risk gate and always tries `resolve_abstention`, potentially
replacing Unknown with a retrieved answer even at low risk. This is a meaningful
documentation/code mismatch. The legacy behavior is preserved; the new research
pipeline preserves standalone abstentions and grades them separately from errors.
Mixed hedges plus claims are still graded for correctness.

## Evaluation concerns

The old factual evaluator pools dataset knowledge, including evaluation-item
knowledge, into its retrieval corpus. That evaluates access to reference evidence,
not the proposed no-external-model intervention. The new mitigation API accepts
only the question, family hint when explicitly enabled, and separately supplied
public tests. Gold answers and private correctness tests never enter it.

The old model's one-word instruction would truncate many math/code answers.
Existing logistic probe artifacts also lack model revision and feature-location
checks. New manifests record those protocols, and mitigation rejects incompatible
generation caches. No old probe artifact is relabeled as a cross-task detector.

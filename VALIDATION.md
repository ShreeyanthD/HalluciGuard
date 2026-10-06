# Validation and research status

Validated locally on 2026-10-06, using CPU only. No GPU was visible. See
`requirements-tested.txt` for the installed environment (Python 3.11.15,
PyTorch 2.14.1+cpu, Transformers 5.18.0, NumPy 2.4.6).

## Automated and mechanism checks

- All 20 targeted tests pass. They check correctness labels, mixed hedges,
  numeric parsing, bounded code tests, entity/paraphrase grouping, source-only
  folds/calibration/scaling, budget decisions, no-gold mitigation inputs,
  AUROC against scikit-learn, bootstrap accounting, attention training, dataset
  adapters, unavailable measurements, caching, and real transformer hooks.
- A tiny random-weight Qwen transformer checks exact answer-span pooling,
  activation changes at the chosen token, and hook removal after an exception.
  This is a mechanism test, not a quality benchmark.
- The downloaded pinned Qwen2.5-0.5B-Instruct also passed real generation,
  `[24, 896]` feature extraction, finite NLL, steering calls and cleanup checks.
- `python eval_selftest.py` passes both legacy mocked end-to-end arms, including
  the empty-knowledge abstention check. Python's randomized token hash was
  replaced with SHA-256. Its mocked result numbers are not model research results.
- `python -m eval.smoke` plus `configs/smoke.yaml` produced all 72 combinations
  (eight detectors × three held-out families × three seeds), with no missing
  measurements. These results use engineered label-correlated features and are
  marked `engineered_fixture` throughout.

## Real generation pilot

Pinned Qwen2.5-0.5B-Instruct generated 108 answers: the first 36 questions each
from pinned TruthfulQA generation, GSM8K train, and sanitized MBPP test. The
protocol used seed 42, greedy generation, CPU and a 96-token response limit.
This config was chosen before inspecting generated correctness labels. Cached
initial generations are shared across the three probe/intervention seeds.

| Family | Questions | Strictly correct | Abstained | Incorrect |
|---|---:|---:|---:|---:|
| Factual | 36 | 0 | 4 | 32 |
| Math | 36 | 1 | 0 | 35 |
| Code | 36 | 7 | 0 | 29 |
| Total | 108 | 8 | 4 | 96 |

These are outcomes under the automatic operational labelers, not manually
adjudicated factual truthfulness. Strict alias matching can reject valid
paraphrases, and the short token limit can truncate math/code completions.
No pilot labels, split seed or thresholds were changed to obtain better scores.

## Detection pilot

The runner fitted 69 of the 72 requested detector/family/seed combinations.
Three single-family linear probes for held-out code cannot fit because their
chosen factual training source has only one class. Calibration is unavailable
for every fitted probe: the source calibration partitions contain no correct
answers. Raw AUROC is evaluated independently of threshold calibration.

Factual and math OOD test partitions are single-class, giving N/A AUROC. Code has
only four test examples, two correct and two incorrect. Several detectors rank
these four perfectly, but their degenerate percentile intervals **do not imply
population certainty or useful transfer**. No fold supplies a statistically
useful estimate of the proposed ID-to-OOD gap. See
[the full detection table](artifacts/pilot_detection/results.md) and
[metrics/provenance](artifacts/pilot_detection/metrics.json).

## Mitigation pilot

The fixed compute limit selects the first two test questions per held-out
family, without using their labels, across seeds 11, 22 and 33. Gated blackboard
and fixed-gate policies require the unavailable source threshold and are
explicitly skipped. Model-only, detector-only, always-on steering and no-gating
are separately evaluated where their prerequisites are present. The strongest
model reference is unavailable because no stronger model was configured.

The final run status and actual policy outcomes are recorded in
`artifacts/pilot_mitigation/metrics.json`, `decisions.jsonl` and `results.md`.
The run completed 36 result rows, with 27 explicitly unavailable measurements
(18 gated policies and nine strongest-model references). The always-on steering
arms produced no recognized standalone abstentions and no correct answers in
this tiny selection. The no-gating arm corrected one math and one code answer
under seed 33; the corresponding reduction intervals span zero to one. These
isolated outcomes do not establish a reliable effect.
These two-question arms are mechanism pilots and cannot establish intervention
benefit. Budgets count requested additional operations; identical frozen-model
calls are reused across arms to reduce CPU evaluation time. Teacher-forced
answer NLL is a fluency proxy, not a human rating.

## Dataset adapter audit

Schemas and numeric gold parsing were checked for the full default inputs,
200 examples in each family. 195 of 200 MBPP reference programs pass the defined
restricted runner. Five reference cases are flagged: four need imports excluded
by the runner (`sys`, `cmath`, `array`) and one exceeds its CPU limit. The tests
are still structurally valid; other implementations may pass. Treat this as an
execution-environment limitation when interpreting full code results. Details
are in `artifacts/adapter_validation.json`. No 600-question generation study was
run here.

## Work still needed for a research claim

A larger model and adequate generation budget, enough independent examples of
both correctness classes in every source calibration and test split, manually
audited factual labels and entity/paraphrase groups, stronger exact-method
baselines, and an explicitly configured strongest-model reference are still
needed. Neither robust cross-task generalization nor reduced hallucinations is
established by this pilot. Research code and runnable configs are delivered;
statistically adequate gated experiments remain unverified.

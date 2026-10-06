# HalluciGuard research notes

Literature checked 2026-10-06. This is an implementation and experiment protocol,
not evidence that cross-task hallucination detection has been solved.

## Problem and question

Can source-task training of an internal-state detector reduce the detection gap
on an entirely unseen task family, and can source-calibrated risk drive a
same-model intervention that lowers answer errors without trading away accuracy
or fluency? Here factual mistakes, mathematical errors, and failing programs are
operationalized as **answer incorrectness**. Those categories do not all coincide
with the narrower definition of fabricated factual claims.

## Checked evidence and closest work

| Source | What the paper supports | Implication |
|---|---|---|
| [Dubanowska et al., Findings EMNLP 2025](https://arxiv.org/html/2509.19372v1) ([ACL record](https://aclanthology.org/2025.findings-emnlp.952/)) | RAGTruth aggregate gains can reflect task/label correlations; analyzed cross-distribution methods are close to random | Report per-family outcomes and text artifact controls; this is evidence about the studied methods, not every future detector |
| [Liu et al., Universal Truthfulness Hyperplane](https://arxiv.org/html/2407.08582v3) | Table 1 gives Probe-LR accuracy 82.28% in distribution and 54.44% mean OOD: a 27.84-point gap; Probe-MM drops 26.37 points. Diverse training across more than 40 datasets improves transfer | The README's “about 25 points” is approximate and refers to accuracy, not AUROC; diversified probe training already exists |
| [Mrykhin and Malykh, PEP](https://arxiv.org/html/2608.08024v1) | Table 6 evaluates training on two datasets and testing the third. Neither PEP nor its linear probe transfers robustly across TriviaQA, MedQA, GSM8K | The leave-one-dataset experiment is already established; our task-family protocol alone is not a new contribution |
| [Suresh et al., CLAP](https://arxiv.org/html/2509.09700v1) | Projects the full layer sequence, prepends CLS, and trains a Transformer encoder; reports improvements including OOD comparisons | All-layer probing and attention across layers are not novel; use the CLAP-style architectural baseline |
| [Hussain and Kantarcioglu, PARALLAX / DRIFT](https://arxiv.org/html/2605.17028v1) | DRIFT uses upper-layer pairwise differences, cosine similarities and norms, then a supervised probe; distinguishes benchmark artifacts from live-generation detection | The adjacent-layer trajectory baseline is explicitly inspired by this method, not an exact reproduction. “DRIFT” is an ambiguous acronym; this reference identifies the intended transition probe |
| [CASAL](https://arxiv.org/html/2510.02324v1) | Contrasts known/unknown activation groups for steering and amortizes the intervention into weights | Contrastive abstention steering is established; this implementation does not implement CASAL weight amortization |
| [Ji et al., EMNLP 2025](https://arxiv.org/html/2503.14477v2) ([ACL record](https://aclanthology.org/2025.emnlp-main.187/)) | Identifies a verbal-uncertainty feature and intervenes at inference time; verbal and semantic uncertainty differ | A direction that encourages Unknown is not automatically a truthfulness direction; report correctness and abstention together |
| [ICR Probe](https://arxiv.org/abs/2507.16488) | Studies cross-layer hidden-state evolution for hallucination detection | Layer trajectories alone are not a novelty claim |
| [USteer](https://arxiv.org/html/2609.38962v1) | Recent reasoning-time steering uses gradients of an internal confidence measure | Recheck this recent overlap before claiming detector-guided reasoning mitigation is new |

The search used the cited identifiers and searches for cross-layer probing,
group-DRO/task-invariant detection and uncertainty-guided steering. This is a
bounded literature check, not a guarantee that no similar combination exists.

## Implemented hypothesis and comparisons

The hypothesis is a regularized logistic detector on flattened all-block answer
means, trained by group-DRO over source families. Group weights are updated by
exponentiated source-group losses. Scaling is fitted on source training data only.
An all-layer ERM comparison isolates the objective from simply adding layers.
Group-DRO does **not** mathematically guarantee family invariance or OOD success.
The objective is based on [Sagawa et al., group-DRO](https://arxiv.org/abs/1911.08731),
which also emphasizes regularization for worst-group generalization. Applying an
established objective is not itself a novelty claim.

The CLAP-style baseline uses a trainable down-projection, CLS token, learned
layer positions, one Transformer block and binary loss on last-answer-token
activations. It differs from the paper's EOS token and training recipe. The
trajectory baseline uses adjacent-block differences, norms and cosine features,
rather than DRIFT's selected upper-layer pairs. These are architectural controls;
published baseline numbers cannot be claimed from these implementations.

The blackboard holds estimated family, scores, source-calibrated thresholds,
attempted actions and remaining operations. Low risk passes through. Factual
risk triggers a forward-hook steering regeneration using a source-training
Unknown-versus-correct-completion contrast. Math uses sampled numeric-answer
agreement. Code may use independently supplied public tests, same-model revision,
and detector selection when budget allows. None uses a stronger external model.

The fixed-pipeline ablation uses the same global gate and steering on all tasks.
Always-on steering bypasses the gate and task selection. No-gating retains task
dispatch while bypassing risk. Detector-only and base-model-alone pass through.
Strongest-model-alone is a separate upper reference requiring an explicitly
chosen model; missing configuration is recorded as an unavailable measurement.
CASAL and Ji-style full reproduction are future external baselines, not reported
as completed experiments.

## Protocol and limitations

- Three distinct probe seeds. Initial generations are cached with a fixed data
  seed; this is not three independent generator datasets. Per-seed group-bootstrap
  intervals do not estimate across-seed or pretraining uncertainty.
- Held-out family labels are never used for fitting, normalization, direction
  learning, or calibration. Its training/calibration partitions remain unused.
  Connected groups shared with that entire family are excluded from source data.
- Exact/lexical matches and supplied entity/paraphrase groups are connected before
  splitting. Unannotated semantic paraphrases and shared entities can remain;
  annotate and audit them before claiming leakage-free publication results.
- Source calibration controls empirical false-positive rate on correct source
  responses, optionally by source family. The global threshold is transferred
  to an unseen family without target labels. It is
  neither probability calibration nor an OOD error guarantee.
  With no correct source-calibration examples, raw AUROC can still be measured,
  but gated policies are explicitly unavailable; ungated ablations can run.
- Standalone abstentions are excluded from detection labels/AUROC and reported
  as mitigation abstentions. Empty or unparseable non-abstaining outputs count
  as errors. Single-class tests yield N/A AUROC; missing source classes are explicit
  skips instead of invented scores.
- Strict factual alias matching can count semantically correct paraphrases wrong,
  especially long-form TruthfulQA. Audit ambiguous labels manually before research
  claims. Numeric parsing only accepts a bare number or explicit final marker.
- The code runner is restricted, timed, and resource-limited; it is not a secure
  sandbox. Run untrusted datasets/completions in a container without networking.
  Restricted imports and missing tests can affect scores. Private tests are only
  grading tools; they cannot be used to select mitigation candidates.
- Routing is a heuristic, so incorrect task estimates may select poor interventions.
  Oracle task hints are a distinct configurable condition, never silently enabled.
- Answer NLL under the base model is a fluency proxy, not a human fluency score.
  Compute budgets count additional generation, test and scoring operations rather
  than tokens/FLOPs; initial generation, direction training, and evaluation NLL
  are logged protocol costs outside the per-answer intervention budget.
- Small CPU model pilots cannot establish useful transfer for larger models.
  More than three families, model scales, dataset revisions, sample counts and
  precise baseline reproductions are needed before a publication claim.
  Tiny-sample bootstrap intervals may collapse to a point. Single-class AUROC
  draws are skipped and counted; the reported interval is conditional on valid
  draws and does not remove the need for adequate test-set size.

## Novelty status

No confirmed novelty or effectiveness claim is made. The candidate contribution
to test is the **combination** of source-family robust detector training, a strict
unseen-family protocol, and budgeted task-dependent mitigation. Blackboard state
is an implementation choice, not by itself a research contribution. Evidence
must show benefit over pooled/all-layer ERM, architectural baselines, fixed
policies, and the properly configured strongest-model reference.

## Result status

See `VALIDATION.md` for actual runs. Engineered smoke-fixture AUROC is only a
plumbing test. Real model pilots, when recorded, are separate and retain their
sample sizes, single-class failures, confidence intervals and limitations.

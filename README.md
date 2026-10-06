# HalluciGuard

Cross-task answer-error detection and blackboard-driven mitigation using a frozen
local open-weights model. **Research hypothesis, not a proven reliability system.**
Work is on `feature/blackboard_fix`; `main` is not modified.

The repository contains a separate research pipeline alongside the original
Groq/Chroma API demo. Read [REPO_AUDIT.md](REPO_AUDIT.md),
[RESEARCH_NOTES.md](RESEARCH_NOTES.md), and [VALIDATION.md](VALIDATION.md)
for scope, verified prior work, and actual run status.

## Reproduction

Use Python 3.11 or 3.12. The default model is pinned Qwen2.5-0.5B-Instruct.
The default data config selects a CUDA GPU when available and otherwise uses
CPU. Research commands need no API keys or legacy server dependencies.

```bash
git clone -b feature/blackboard_fix https://github.com/ShreeyanthD/HalluciGuard.git
cd HalluciGuard
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-research.txt
python -m data.build --config configs/data.yaml
python -m eval.run --config configs/detect_loto.yaml
python -m eval.run --config configs/mitigate_loto.yaml
```

The clone command identifies the source branch. Local changes delivered with
this task must first be applied to that checkout; they are not automatically
published to GitHub. Alternatively use the delivered checkout/archive directly.

The requested `pip install -r requirements.txt` also includes the new research
requirements. For CPU-only PyTorch, install its CPU wheel from the official
PyTorch index to avoid unnecessary CUDA packages.

The default data config uses 200 questions each from pinned TruthfulQA, GSM8K and
sanitized MBPP. A first run downloads the model and datasets. Generation is
greedy; detector/intervention seeds are `[11, 22, 33]`. A full CPU run can be slow.
For a smaller pilot use `pilot_data.yaml`, `pilot_detect.yaml`, and
`pilot_mitigate.yaml`. Small samples can produce undefined AUROCs or untrainable
folds; these are explicitly reported.

### GPU data generation

Enable a GPU runtime in your notebook (for example, Colab's runtime settings)
and use CUDA-enabled PyTorch. Force GPU use with:

```python
!python -m data.build --config configs/data.yaml --device cuda
```

In a terminal, omit `!`. Use `--device cuda:1` to choose another GPU,
`--device auto` for GPU with CPU fallback, or `--device cpu` to force CPU.
An explicit CUDA request fails clearly if CUDA is unavailable. The command
prints the resolved device and precision and records both in its manifest.
CUDA uses BF16 on supported GPUs and FP16 on older GPUs such as T4; CPU uses
FP32. Feature summaries are pooled on the model device before transfer to CPU.
Reduced precision can change generated answers; regenerate features and refit
detectors for the GPU run when comparing results. Dataset loading, grading,
and saving still run on CPU. GPU speed depends on hardware and response length;
no CUDA speedup was benchmarked on the CPU-only validation host.

## Offline validation

```bash
python -m pytest -q tests
python -m eval.smoke
python -m eval.run --config configs/smoke.yaml
```

The smoke run constructs artificial, label-correlated features. Its output is
marked `engineered_fixture` and **cannot support claims about LLM performance**.
It validates eight detectors across three held-out families and three seeds.

## Architecture

```text
question → frozen local LM + all-block hooks → probe scores
                                             ↓
        Blackboard: estimated family, calibrated thresholds, scores,
                    actions attempted, remaining operation budget
                                             ↓
        low risk → pass
        factual risk → activation steering and regeneration
        math/code risk → resampling, agreement or verification
```

The arbiter reads and updates decision state; this blackboard is not a message
log. Mitigation uses the original local model. Existing honest standalone
abstentions remain abstentions. Steering changes decoder activations with a
removable forward hook and does not update weights.

| Directory | Contents |
|---|---|
| `configs/` | Full research, pilot, and fixture configs |
| `data/` | Dataset adapters, automatic correctness labels, connected-group splitting |
| `detectors/` | Single-family/dataset and pooled linear probes, all-layer ERM, group-DRO, CLAP-style attention, transition and text controls |
| `blackboard/research.py` | Decision state, heuristic router, source-only thresholds, arbiter |
| `interventions/` | Local model/hooks, steering contrast, resampling/revision, bounded Python tests |
| `eval/` | Leave-one-family-out runner, grouped bootstrap, tables, fixture builder |
| `tests/` | Label, leakage, calibration, budget, intervention, metric and transformer mechanism checks |

## Data and label contract

Custom sources are JSONL records. Point a `sources` entry at `path` and set its
`family` and optional `name`. No gold answer is shown during initial generation.

```json
{"id":"q1","question":"What is the capital of France?","gold":["Paris"],"entities":["France"]}
{"id":"m1","question":"Calculate 2 + 2","gold":"4","paraphrase_group":"addition-example"}
{"id":"c1","question":"Implement add(a, b) in Python.","tests":["assert add(2,3)==5"],"public_tests":["assert add(0,0)==0"]}
```

Factual answers use strict normalized alias matching; math uses numeric equality
after an explicit final marker or a bare number; code uses private correctness
tests. MBPP's required function signature is extracted as interface metadata;
implementation and expected outputs are not shown. Its `test_imports` are
retained for grading. Independent `public_tests` may be used for mitigation, but
private `tests` never are. Restricted subprocess testing is not a secure sandbox;
isolate untrusted code at the OS/container level.

Standalone abstentions get `label: null` and are excluded from detector AUROC.
Incorrect answers get label 1, correct answers label 0. Empty responses count as
incorrect. Exact questions, lexical near-duplicates, and supplied entity and
paraphrase groups share a split. Semantic/entity annotations need manual review
before publication; default datasets lack exhaustive annotations.

## Evaluation outputs

The builder saves `records.jsonl`, `features.npz` and a manifest. Evaluations save
configs, Git revision/dirty state, input hashes, split IDs, detector weights,
thresholds, predictions or mitigation decisions, `metrics.json`, and `results.md`.
Mitigation rejects a model revision or feature protocol mismatch.

The entire held-out family is excluded from training, normalization, direction
fitting and calibration. Its test partition is the OOD evaluation; source-family
test partitions provide an ID reference. Thresholds use source calibration only,
with a global fallback for unseen families. Probe seeds share cached initial
generations. Confidence intervals resample connected question groups per seed.
If source calibration has no correct answers, raw AUROC is still reported when
defined, gated policies are unavailable, and ungated controls may still run.
Bootstrap intervals from tiny samples can be misleading or degenerate.

Mitigation reports accuracy, error rate, abstention rate, coverage, selective
accuracy, paired error reduction with CI, and base-model answer NLL as a fluency
proxy. Operation counts enforce intervention budgets. NLL is not human fluency,
and operation counts are not equivalent to tokens or FLOPs.

Ablations include fixed pipeline/no blackboard, always-on steering, no gating,
detector only, base model alone, and strongest model alone. Set `strongest_model`
explicitly to enable the latter; otherwise its measurement is recorded as
unavailable. CASAL and Ji full-method reproductions are not implemented.

## Legacy demo

The API/retrieval pipeline remains in `pipeline.py`, `probe/`, and
`blackboard/blackboard_core.py`; setup is in [LEGACY_README.md](LEGACY_README.md).
It uses Groq agents, Chroma evidence and optional web search, so its evaluation
answers a different question. The legacy self-test uses mocks:

```bash
pip install -r requirements.txt
python eval_selftest.py
```

See the audit for its old abstention-description/routing mismatch. None of its
mocked metrics are research evidence.

## Deliverable status

- Repository audit and verified research notes: delivered.
- Three-family adapters and labelers: delivered.
- Detector baselines and group-DRO hypothesis: delivered.
- Blackboard-driven local mitigation and ablation runner: delivered.
- Offline and real-model mechanism validation: see `VALIDATION.md`.
- Real results: see recorded pilot outputs; full statistically adequate evaluation
  and a configured strongest-model reference remain separate work.
- Proven cross-task generalization or mitigation benefit: **not established**.

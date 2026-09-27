# Merged Hallucination Detection + Mitigation Pipeline

This merges two previously separate systems into one pipeline:

1. **`probe/`** — a residual-stream probe (Qwen2.5 layer 20 hidden states ->
   `StandardScaler` + `LogisticRegression`) that looks at a generated answer
   and outputs `prob_hallucinated` in `[0, 1]`.
2. **`blackboard/`** — the Blackboard-architecture mitigation system
   (`BlackBoard-arch`). If a response's risk score is at/above a threshold
   (default `0.70`), a sequence of agents — `ClaimExtractor` ->
   `MemoryAgent` -> `RetrievalAgent` -> `VerifierAgent` -> `CorrectionAgent`
   — extracts the riskiest claim, checks episodic memory, retrieves
   evidence from a knowledge base, verifies the claim, and corrects or
   hedges the response if it's unsupported.

Nothing inside `probe/` or `blackboard/` was changed — the two systems
already spoke the same language (a `[0, 1]` risk score in, a response out),
so `pipeline.py` is just the wiring between them.

## How a request flows end-to-end

```
question
   │
   ▼
QwenResidualFeatureExtractor.batch_generate_and_extract_features()   [probe/llama_features.py]
   │  -> generated answer + mean-pooled layer-20 residual-stream vector
   ▼
scaler.transform() -> classifier.predict_proba()                     [trained probe .joblib]
   │  -> prob_hallucinated  (0.0–1.0)
   ▼
blackboard_core.process_response(prompt, response, prob_hallucinated) [blackboard/blackboard_core.py]
   │
   ├─ prob_hallucinated < 0.70  ─────────────────────────► response returned unchanged
   │
   └─ prob_hallucinated ≥ 0.70
        ├─ ClaimExtractor   – picks the single riskiest claim
        ├─ MemoryAgent      – checks episodic memory for a past verdict
        ├─ RetrievalAgent   – (if no memory hit) pulls top-k evidence docs
        ├─ VerifierAgent    – SUPPORTED / CONTRADICTED / INSUFFICIENT
        └─ CorrectionAgent  – rewrites (if contradicted) or hedges (if
                              insufficient); passes through unchanged if
                              supported
   │
   ▼
final_response  (+ full trace: extracted claim, verdict, evidence, etc.)
```

## Project structure

```
.
├── pipeline.py            # NEW — the merge point (see below)
├── unified_server.py       # NEW — one FastAPI service exposing /ask, /score_and_mitigate, /analyze
├── probe/
│   ├── llama_features.py   # Qwen2.5 residual-stream feature extractor
│   ├── Dataset_builder.py  # manual-labeling dataset builder (HaluEval)
│   ├── train_probe.py      # trains scaler + LogisticRegression -> probe.joblib
│   └── infer_probe.py      # standalone probe inference (unchanged, still works on its own)
├── blackboard/
│   ├── blackboard_core.py  # Agents, Orchestrator, Blackboard, ChromaDB, process_response()
│   ├── server.py           # original standalone Blackboard-only FastAPI app (unchanged)
│   └── static/index.html   # live Blackboard trace demo UI
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Environment variables:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | Yes | — | Powers the Verifier, Correction, and Claim Extraction agents. |
| `PROBE_PATH` | Yes (for `/ask`) | — | Path to a trained probe `.joblib` from `probe/train_probe.py`. |
| `PROBE_LAYER` | No | `20` | Must match the layer the probe was trained on. |
| `QWEN_MODEL_ID` | No | `Qwen/Qwen2.5-7B-Instruct` | Model the feature extractor loads. |
| `CHROMA_PERSIST_DIRECTORY` | No | `./halluciguard_chroma` | Blackboard's ChromaDB store (knowledge + memory). |
| `GROQ_MODEL` | No | `openai/gpt-oss-120b` | Groq model for the Blackboard agents. |

A CUDA GPU is required for the probe half (`QwenResidualFeatureExtractor`
refuses to fall back to CPU by design).

## 1. Train the probe (one-time, if you don't already have `probe.joblib`)

```bash
cd probe
python Dataset_builder.py --local_path HaluEval-main/data --max_questions 100 \
    --output_path halueval_manual_features.pt
python train_probe.py --data halueval_manual_features.pt --output probe.joblib
```

## 2. Run the merged pipeline

### Command line

```bash
export GROQ_API_KEY=your_key_here
python pipeline.py --probe probe/probe.joblib \
    --questions "What year was the Eiffel Tower completed?" "Who wrote Hamlet?"
```

Each question is generated, scored, and — if flagged — run through the
Blackboard. Output shows the risk score, verdict, and final response.

### As a service

```bash
export GROQ_API_KEY=your_key_here
export PROBE_PATH=probe/probe.joblib
uvicorn unified_server:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/ask \
    -H "Content-Type: application/json" \
    -d '{"question": "What year was the Eiffel Tower completed?"}'
```

`GET /health` reports whether the probe is loaded, plus the Blackboard's
knowledge/memory doc counts and active threshold.

## 3. Already have an answer from elsewhere?

Use `pipeline.py`'s `run_on_qa(question, answer)` (or the
`/score_and_mitigate` endpoint) to score and mitigate a `(question, answer)`
pair without generating a new answer — e.g. if a different model produced
the response and you just want this pipeline to risk-score and ground it.

## Notes

- `probe/infer_probe.py` still works standalone (score a `.pt` file of
  precomputed features, or run questions through the probe with no
  Blackboard involved) — nothing there was changed.
- `blackboard/server.py` still works standalone too (bring your own
  `confidence_score`) — also unchanged. `unified_server.py`'s `/analyze`
  route is the same call, just hosted alongside the new `/ask` route.
- The risk threshold (`blackboard_core.HALLUCINATION_RISK_THRESHOLD`,
  default `0.70`) is shared by both halves once merged: it's the same
  number the probe's score is compared against. Override it per-run with
  `pipeline.py --threshold` or `HallucinationMitigationPipeline(...,
  override_threshold=...)`.

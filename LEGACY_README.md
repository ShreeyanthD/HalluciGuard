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
   ├─ prob_hallucinated < 0.70  ─────────────────────────► response returned unchanged (status=SKIPPED_LOW_RISK)
   │
   └─ prob_hallucinated ≥ 0.70
        ├─ AbstentionDetector  – is this already an honest "I don't know"? ─► if yes: status=ABSTENTION,
        │                                                                      returned unchanged, Blackboard skipped
        └─ if not an abstention:
             ├─ ClaimExtractor   – picks the single riskiest claim
             ├─ MemoryAgent      – checks episodic memory for a past verdict
             ├─ RetrievalAgent   – (if no memory hit) pulls top-k evidence docs, up to
             │                     MAX_VERIFICATION_ROUNDS rounds
             ├─ WebSearchAgent   – (if still INSUFFICIENT after local rounds, and enabled)
             │                     live web search, adds results as evidence, re-verifies once
             ├─ VerifierAgent    – SUPPORTED / CONTRADICTED / INSUFFICIENT
             └─ CorrectionAgent  – .correct() rewrites a CONTRADICTED claim using evidence;
                                   .abstain() honestly hedges if still INSUFFICIENT; passes
                                   through unchanged if SUPPORTED
   │
   ▼
final_response  (+ full trace: extracted claim, verdict, evidence, abstention_detection, etc.)
```

### Three behaviors worth calling out explicitly

1. **Insufficient evidence → an honest abstention, not a bracketed note.**
   Previously, an unresolved claim was left in the response with `[Note: ... could
   not be verified ...]` appended. Now `CorrectionAgent.abstain()` rewrites the
   response so it naturally says the claim couldn't be confirmed, without
   inventing a replacement fact.

2. **Insufficient local evidence → try a live web search before giving up.**
   If the knowledge base's retrieval rounds all come back `INSUFFICIENT`,
   `WebSearchAgent` (Tavily) does one live search for the claim, adds the
   results as evidence, and gives `VerifierAgent` one more pass. Only if
   that *still* comes back `INSUFFICIENT` does it fall through to the
   abstention above. This is opt-in — see `TAVILY_API_KEY` below; if unset,
   this step is skipped automatically and behavior falls straight through
   to (1).

3. **An abstention the model already gave shouldn't be relabeled a hallucination.**
   The probe's risk score reflects how *unusual* a response's activations
   look — an honest "I don't know" can score high simply because it's an
   atypical pattern, not because anything was fabricated. `AbstentionDetector`
   runs before claim extraction: a keyword check first, an LLM classifier
   fallback for paraphrased abstentions it might miss. If the response is
   already an abstention, `process_response()` returns `status="ABSTENTION"`
   and leaves it untouched — the Blackboard never runs on it, so it can't
   accidentally "correct" an honest hedge into a confident invented answer.

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
| `TAVILY_API_KEY` | No | — | Enables the web-search fallback (behavior 2 above). Unset = skipped automatically, no error. |
| `WEB_SEARCH_RESULTS` | No | `5` | Max web results pulled into evidence per fallback search. |

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
knowledge/memory doc counts, active threshold, and whether the web-search
fallback is enabled.

### Frontend

`blackboard/static/index.html` (served at `/` by `unified_server.py`) now
has two modes, toggled at the top of the input panel:

- **Bring your own answer** — the original demo: type a prompt, a
  response, and a confidence score, hits `/analyze`.
- **Ask the model** — new: type only a question. This calls `/ask_trace`,
  which runs the full merged pipeline (Qwen generates, the probe scores
  it, the Blackboard verifies if flagged) and renders the same stage-by-
  stage trace, plus the generated answer and its risk score up top. If
  the response turns out to already be an abstention, or gets corrected
  after a web search, that's shown as a distinct step in the trace rather
  than folded into the generic "Correct" stage.

## 3. Already have an answer from elsewhere?

Use `pipeline.py`'s `run_on_qa(question, answer)` (or the
`/score_and_mitigate` endpoint) to score and mitigate a `(question, answer)`
pair without generating a new answer — e.g. if a different model produced
the response and you just want this pipeline to risk-score and ground it.

## Notes

- Every result (`pipeline.py` output, `/ask`, `/score_and_mitigate`, `/analyze`)
  now carries a `status` of `SKIPPED_LOW_RISK`, `ABSTENTION`, or
  `BLACKBOARD_PROCESSED`, plus an `abstention_detection` field (`None` when
  skipped for low risk; `{"is_abstention": ..., "method": "heuristic"|"llm",
  "explanation": ...}` otherwise).
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

## Evaluating the Blackboard (with vs. without)

`eval_pipeline.py` runs the same questions through two arms — the base
model's raw answer (**without** blackboard) and the answer after
`process_response()` (**with** blackboard) — and grades each as
`CORRECT`, `ABSTAINED` or `HALLUCINATED`. Headline metric:
`hallucination_rate = wrong answers / all questions`, plus a paired-bootstrap
95% CI on the drop.

```bash
# full run (GPU box, GROQ_API_KEY set)
python eval_pipeline.py --dataset HaluEval-main/data/qa_data.json \
    --probe probe/probe.joblib --limit 200

# split it: generate on Colab, run the blackboard anywhere
python eval_pipeline.py --dataset qa_data.json --probe probe.joblib --generate_only --out_dir eval_runs/exp1
python eval_pipeline.py --answers_file eval_runs/exp1/generations.jsonl --out_dir eval_runs/exp1

# no probe: push every answer through the blackboard
python eval_pipeline.py --answers_file eval_runs/exp1/generations.jsonl --threshold 0

python eval_selftest.py   # offline plumbing test (no GPU / API key)
```

Outputs go to `out_dir`: `metrics.json`, `per_item.csv` (every question, raw
vs. final answer, labels, status), `summary.md`. The eval uses its own
ChromaDB (`--chroma_dir`, default `./eval_chroma`) so your real store is never
touched. The knowledge base is built from the dataset's `knowledge` field
(pooled across all questions); `--no_knowledge` runs the empty-KB ablation and
`--isolate_memory` stops verdicts leaking between questions.

**Abstention:** the base model's system prompt now allows the single word
`Unknown` instead of forcing an answer, and `AbstentionDetector` treats a bare
`Unknown` / `IDK` as an abstention.

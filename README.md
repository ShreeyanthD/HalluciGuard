# BlackBoard-arch

**BlackBoard-arch** is a Blackboard-architecture pipeline for mitigating LLM hallucinations. When an external hallucination-risk detector (e.g. a sparse autoencoder, or SAE) flags a response as risky, BlackBoard-arch extracts the specific claim at fault, checks episodic memory for a past verdict, retrieves supporting evidence from a knowledge base, verifies the claim against that evidence, and — if the claim is unsupported — corrects or hedges the response before it reaches the user.

It ships as a small FastAPI service with a live "Blackboard trace" frontend that visualizes each stage of the pipeline for a submitted prompt/response pair.

## How it works

Responses only enter the pipeline if their hallucination-risk score is at or above a threshold (`HALLUCINATION_RISK_THRESHOLD`, default `0.70`). Below that, the response is returned unchanged.

For flagged responses, a shared **Blackboard** workspace is used by a sequence of agents:

1. **ClaimExtractor** — identifies the single most important, independently verifiable factual claim in the response (the one most likely responsible for the risk score).
2. **MemoryAgent** — checks episodic memory (a ChromaDB collection of previously verified claims) for a similar past verdict. A close-enough match short-circuits retrieval and verification.
3. **RetrievalAgent** — if no memory hit, retrieves the top-k relevant documents from a persistent knowledge base (also ChromaDB) as evidence.
4. **VerifierAgent** — judges the claim against the retrieved evidence, returning a verdict of `SUPPORTED`, `CONTRADICTED`, or `INSUFFICIENT`.
5. **CorrectionAgent** — if the claim is contradicted, rewrites the response to be grounded in the evidence. If evidence is insufficient, a hedge is appended instead of inventing a correction. Supported claims pass through unchanged.
6. Grounded, evidence-backed verdicts are written back to episodic memory so future occurrences of the same (or a very similar) claim can be resolved instantly.

Every read/write to the Blackboard is logged with a timestamp and author, giving a full audit trail (`blackboard_history`) of how a response was processed.

## Project structure

```
.
├── blackboard_core.py      # Agents, Orchestrator, Blackboard, ChromaDB setup, process_response()
├── blackboard_imp2.ipynb   # Original notebook implementation
├── server.py               # FastAPI app: /analyze, /health, and static frontend host
├── static/index.html       # Live Blackboard trace demo UI
├── BlackBoard-arch_chroma/    # Persistent ChromaDB store (knowledge + memory collections)
└── requirements.txt
```

## Requirements

- Python 3.10+
- A [Groq](https://console.groq.com/) API key (used by the Verifier, Correction, and Claim Extraction agents)
- Optionally, a Gemini API key if you want to swap the LLM provider (see [Configuration](#configuration))

Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup

1. Set your Groq API key (required — the server fails fast at startup without it):

   ```bash
   export GROQ_API_KEY=your_key_here        # Linux/macOS
   $env:GROQ_API_KEY="your_key_here"         # Windows PowerShell
   ```

2. (Optional) Point the server at an existing seeded ChromaDB directory, if it's not the default `./BlackBoard-arch_chroma`:

   ```bash
   export CHROMA_PERSIST_DIRECTORY=/path/to/your/BlackBoard-arch_chroma
   ```

3. Run the server:

   ```bash
   uvicorn server:app --reload --port 8000
   ```

4. Open [http://localhost:8000](http://localhost:8000) to use the live Blackboard trace demo.

## API

### `POST /analyze`

Runs a prompt/response pair through the pipeline.

**Request body:**

```json
{
  "prompt": "What year was the Eiffel Tower completed?",
  "response": "The Eiffel Tower was completed in 1887.",
  "confidence_score": 0.85
}
```

- `confidence_score` is the external hallucination-risk score (0.0–1.0), typically produced upstream by an SAE or similar detector.

**Response:** a stage-by-stage trace object showing whether the pipeline was skipped (low risk) or run, and if run, the memory check, retrieved evidence, verification verdict, and correction/hedge/pass-through result.

### `GET /health`

Returns service status plus the current document counts in the knowledge and memory collections, and the active risk threshold.

```json
{
  "status": "ok",
  "knowledge_docs": 20,
  "memory_docs": 4,
  "threshold": 0.7
}
```

## Configuration

Key environment variables:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | Yes | — | Powers the Verifier, Correction, and Claim Extraction agents via Groq. |
| `GROQ_MODEL` | No | `openai/gpt-oss-120b` | Groq model to use. |
| `CHROMA_PERSIST_DIRECTORY` | No | `./BlackBoard-arch_chroma` | Path to the persistent ChromaDB store for the knowledge and memory collections. |
| `BlackBoard-arch_KEY` | No | — | Gemini API key, if you want an alternate LLM provider available. |
| `GEMINI_MODEL` | No | `gemini-3.6-flash` | Gemini model, used only if `BlackBoard-arch_KEY` is set. |

Other pipeline constants (`RETRIEVAL_TOP_K`, `MEMORY_TOP_K`, `MAX_VERIFICATION_ROUNDS`, similarity thresholds) are set in `blackboard_core.py`.

## Managing the knowledge base

`blackboard_core.py` exposes helper functions for populating and clearing ChromaDB collections:

- `add_knowledge_documents(documents, metadatas=None, ids=None)` — add evidence documents to the knowledge base.
- `reset_knowledge_collection()` — clear all knowledge documents.
- `reset_memory_collection()` — clear all episodic memory (previously verified claims).

## Notes

- `server.py` is an adaptation of the pipeline logic originally developed in `blackboard_imp2.ipynb`; no pipeline behavior was changed in the port — see the `CHANGES FROM THE NOTEBOOK` note at the bottom of `blackboard_core.py` for the (non-logic) differences.
- The `BlackBoard-arch_chroma/` directory contains a working ChromaDB store. If you're pushing this repo publicly, consider whether you want to commit it as-is, ship it empty, or add it to `.gitignore` and let it be recreated on first run.

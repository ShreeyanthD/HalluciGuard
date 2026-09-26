# HalluciGuard — System Workflow

## What this is
A backend (already built and tested) that runs your real Blackboard pipeline
and returns a stage-by-stage trace. The frontend's only job is to collect
input, call it, and visualize the trace as it comes back.

## Files already built (do not rewrite these)
- `blackboard_core.py` — your actual notebook pipeline (Blackboard,
  MemoryAgent, RetrievalAgent, VerifierAgent, CorrectionAgent, Orchestrator),
  adapted to run as a server module. Logic unchanged from the notebook.
- `server.py` — FastAPI app. Exposes `/analyze` and `/health`, serves the
  frontend from `static/`.
- `requirements.txt`

## Request flow
```
User fills form (prompt, response, confidence_score)
        │
        ▼
POST /analyze  ──────────────────────────────►  server.py
        │                                         │
        │                                calls blackboard_core.process_response()
        │                                         │
        │                          score < 0.70?  │  score ≥ 0.70?
        │                          skip pipeline   │  run full pipeline:
        │                                         │   Memory → Retrieve → Verify → Correct
        │                                         │
        │◄──────────── JSON trace ────────────────┘
        ▼
Frontend renders the trace stage by stage
```

## API contract

### `POST /analyze`
Request:
```json
{"prompt": "string", "response": "string", "confidence_score": 0.0}
```

Response — below threshold:
```json
{"skipped": true, "confidence_score": 0.2, "threshold": 0.7, "final_response": "..."}
```

Response — pipeline ran (this is what the frontend visualizes):
```json
{
  "skipped": false,
  "confidence_score": 0.9,
  "threshold": 0.7,
  "extracted_claim": "string",
  "flagged_span": "string",
  "extraction_reason": "string",
  "memory": {"hit": false, "detail": "string"},
  "retrieve": {
    "evidence": [{"id": "ev_014", "text": "string", "distance": 0.05}],
    "reused_from_memory": false
  },
  "verify": {
    "verdict": "SUPPORTED | CONTRADICTED | INSUFFICIENT",
    "explanation": "string",
    "supporting_evidence_ids": ["ev_014"],
    "confidence": 0.95,
    "source": "retrieval_and_gemini | episodic_memory"
  },
  "correct": {
    "action": "unchanged | corrected | hedged | corrected (from memory)",
    "summary": "string",
    "output": "string"
  },
  "final_response": "string"
}
```
On error, `/analyze` returns HTTP 400 (bad input) or 502 (pipeline/model
failure) with `{"detail": "message"}`.

### `GET /health`
```json
{"status": "ok", "knowledge_docs": 20, "memory_docs": 3, "threshold": 0.7}
```

## Frontend requirements (build this)
1. **Form**: prompt (text), response (textarea), confidence_score (slider
   0–1). A few one-click presets that pre-fill known-good example claims are
   expected — hardcode 2–3 in the JS.
2. **On submit**: `POST /analyze`, show a loading state (this can take several
   seconds — it's calling a real LLM), then render the 4 stages from the
   response: Memory → Retrieve → Verify → Correct. Verdict needs to be
   visually distinct per value (SUPPORTED / CONTRADICTED / INSUFFICIENT) —
   this is a functional signal, not decoration.
3. **Handle `skipped: true`** as its own state (routing skipped the pipeline
   entirely) — don't show empty stage cards for it.
4. **Handle errors** (non-200 response) — show the `detail` message, don't
   fail silently or show a blank panel.
5. **`/health` on load** — show whether the backend is reachable and how many
   knowledge/memory docs are loaded, so a failed connection is obvious before
   the user tries to run anything.
6. Everything is same-origin (`server.py` serves the frontend itself from
   `static/`) — no CORS config needed on the frontend side, just call
   relative paths (`/analyze`, `/health`).

## Explainability content to include somewhere on the page
Static, not interactive — just needs to be present:
- Architecture: Base LLM → Detector (teammate, SAE/Jacobian, out of scope
  here) → confidence_score → Blackboard (this system). Below 0.70, the
  Blackboard never runs.
- The 4 Blackboard stages, one line each: Memory (reuse a grounded past
  verdict), Retrieve (query the knowledge base), Verify (claim vs. evidence →
  verdict + evidence ids), Correct (fix / hedge / leave unchanged, by verdict).

## Known constraints
- Pipeline calls Groq per request — no true streaming, results arrive as one
  JSON payload. Stage-by-stage "live" feel should come from revealing stages
  with a short delay after the response lands, not from the backend
  streaming partial results.
- Needs `GROQ_API_KEY` set before starting, and the existing
  `halluciguard_chroma/` folder in the working directory (or
  `CHROMA_PERSIST_DIRECTORY` pointed at it) so the 20 seeded documents are
  found. Without it, retrieval returns no evidence and everything resolves to
  INSUFFICIENT.

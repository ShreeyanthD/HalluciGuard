"""
server.py — HalluciGuard Blackboard API + frontend host.

Run:
    export GROQ_API_KEY=your_key_here
    export CHROMA_PERSIST_DIRECTORY=/path/to/your/halluciguard_chroma   # optional
    uvicorn server:app --reload --port 8000

Then open http://localhost:8000
"""

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import blackboard_core as bc
from trace import build_trace

app = FastAPI(title="HalluciGuard Blackboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    prompt: str
    response: str
    confidence_score: float


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    try:
        result = bc.process_response(
            prompt=req.prompt,
            response=req.response,
            confidence_score=req.confidence_score,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        # Groq/Chroma failures land here — return a clear message instead of
        # a bare 500 with no context, since this may run live on stage.
        raise HTTPException(status_code=502, detail=f"Pipeline error: {exc}")

    return build_trace(result)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "knowledge_docs": bc.knowledge_collection.count(),
        "memory_docs": bc.memory_collection.count(),
        "threshold": bc.HALLUCINATION_RISK_THRESHOLD,
        "web_search_fallback_enabled": bc.web_search_agent.enabled,
    }


# Serve the frontend. Must be mounted last — routes above take priority.
app.mount("/", StaticFiles(directory="static", html=True), name="static")

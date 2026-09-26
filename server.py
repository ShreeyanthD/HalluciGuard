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


def build_trace(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Turn process_response()'s raw return value into the stage-by-stage shape
    the frontend renders. Presentation only — no pipeline logic lives here.
    """
    if result["status"] == "SKIPPED_LOW_RISK":
        return {
            "skipped": True,
            "confidence_score": result["confidence_score"],
            "threshold": result["threshold"],
            "final_response": result["final_response"],
        }

    orch = result["orchestrator_result"]
    bb = orch["blackboard"]
    memory_result = orch["memory_result"]
    verification_source = orch["verification_source"]
    verification = result["verification_result"] or {}
    correction = result["correction_result"]

    memory_stage = {
        "hit": bool(memory_result.get("match_found")),
        "detail": (
            f"reused a similar past verdict (distance={memory_result['best_match']['distance']:.3f})"
            if memory_result.get("match_found")
            else "no cached verdict for this claim"
        ),
    }

    # Retrieval only actually ran this request if the memory shortcut wasn't taken.
    retrieved = bb.get("retrieved_evidence", []) if verification_source != "episodic_memory" else []
    retrieve_stage = {
        "evidence": [
            {"id": e["id"], "text": e["text"], "distance": e.get("distance")}
            for e in retrieved
        ],
        "reused_from_memory": verification_source == "episodic_memory",
    }

    verify_stage = {
        "verdict": verification.get("verdict"),
        "explanation": verification.get("explanation"),
        "supporting_evidence_ids": verification.get("supporting_evidence_ids", []),
        "confidence": verification.get("confidence"),
        "source": verification_source,
    }

    verdict = verify_stage["verdict"]
    if correction:
        correct_stage = {
            "action": "corrected",
            "summary": correction.get("correction_summary"),
            "output": correction.get("corrected_response"),
        }
    elif verdict == "SUPPORTED":
        correct_stage = {
            "action": "unchanged",
            "summary": "claim matched cited evidence",
            "output": result["final_response"],
        }
    elif verdict == "CONTRADICTED" and verification_source == "episodic_memory":
        correct_stage = {
            "action": "corrected (from memory)",
            "summary": "reused a previously corrected response",
            "output": result["final_response"],
        }
    else:
        correct_stage = {
            "action": "hedged",
            "summary": "insufficient evidence — hedge note appended, nothing invented",
            "output": result["final_response"],
        }

    return {
        "skipped": False,
        "confidence_score": result["confidence_score"],
        "threshold": result["threshold"],
        "extracted_claim": result["extracted_claim"],
        "flagged_span": result["flagged_span"],
        "extraction_reason": result["extraction_reason"],
        "memory": memory_stage,
        "retrieve": retrieve_stage,
        "verify": verify_stage,
        "correct": correct_stage,
        "final_response": result["final_response"],
    }


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
    }


# Serve the frontend. Must be mounted last — routes above take priority.
app.mount("/", StaticFiles(directory="static", html=True), name="static")

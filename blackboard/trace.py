"""
trace.py — turns blackboard_core.process_response()'s raw return value into
the stage-by-stage shape blackboard/static/index.html renders.

Presentation only — no pipeline logic lives here. Shared by server.py
(Blackboard-only) and unified_server.py (merged pipeline), so both expose
identical trace shapes to the frontend.
"""

from typing import Any, Dict


def _cited_sources(evidence, cited_ids):
    cited = set(cited_ids or [])
    picked = [e for e in evidence if e["id"] in cited] or [
        e for e in evidence if (e.get("metadata") or {}).get("url")
    ]
    out = []
    for e in picked:
        md = e.get("metadata") or {}
        url = md.get("url") or (e["id"][4:] if str(e["id"]).startswith("web:") else "")
        out.append({"id": e["id"], "url": url, "title": md.get("title") or url or md.get("source") or e["id"]})
    return out

def build_trace(result: Dict[str, Any]) -> Dict[str, Any]:
    if result["status"] == "SKIPPED_LOW_RISK":
        return {
            "skipped": True,
            "abstained": False,
            "confidence_score": result["confidence_score"],
            "threshold": result["threshold"],
            "final_response": result["final_response"],
        }

    if result["status"] in ("ABSTENTION", "ABSTENTION_RESOLVED"):
        ad = result.get("abstention_detection") or {}
        orch = result.get("orchestrator_result") or {}
        bb = orch.get("blackboard") or {}
        answer = result.get("verification_result") or {}
        return {
            "skipped": False,
            "abstained": True,
            "resolved": result["status"] == "ABSTENTION_RESOLVED",
            "original_response": result.get("original_response"),
            "sources": _cited_sources(bb.get("retrieved_evidence", []),
                                      answer.get("supporting_evidence_ids", [])),
            "web_search_used": bool(bb.get("web_search_used")),
            "answer_explanation": answer.get("explanation"),
            "supporting_evidence_ids": answer.get("supporting_evidence_ids", []),
            "confidence_score": result["confidence_score"],
            "threshold": result["threshold"],
            "final_response": result["final_response"],
            "abstention_method": ad.get("method"),
            "abstention_explanation": ad.get("explanation"),
        }

    orch = result["orchestrator_result"]
    bb = orch["blackboard"]
    memory_result = orch.get("memory_result") or {}
    verification_source = orch.get("verification_source", "retrieval_and_gemini")
    verification = result["verification_result"] or {}
    correction = result["correction_result"] or {}

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
            {
                "id": e["id"],
                "text": e["text"],
                "distance": e.get("distance"),
                # Web-search hits carry a url/title/source in metadata;
                # knowledge-base hits generally don't. The frontend uses
                # this to show a link instead of the full scraped text.
                "url": (e.get("metadata") or {}).get("url"),
                "title": (e.get("metadata") or {}).get("title"),
                "source": (e.get("metadata") or {}).get("source"),
            }
            for e in retrieved
        ],
        "reused_from_memory": verification_source == "episodic_memory",
        "web_search_used": bool(bb.get("web_search_used")),
    }

    verify_stage = {
        "verdict": verification.get("verdict"),
        "explanation": verification.get("explanation"),
        "supporting_evidence_ids": verification.get("supporting_evidence_ids", []),
        "confidence": verification.get("confidence"),
        "source": verification_source,
    }

    verdict = verify_stage["verdict"]
    if correction and correction.get("mode") == "abstain":
        correct_stage = {
            "action": "abstained",
            "summary": correction.get("correction_summary"),
            "output": correction.get("corrected_response"),
        }
    elif correction:
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
            "action": "unchanged",
            "summary": "",
            "output": result["final_response"],
        }

    return {
        "skipped": False,
        "abstained": False,
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

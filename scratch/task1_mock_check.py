import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def call_gemini_with_retry(*args, **kwargs):
    raise RuntimeError("The real Gemini caller must not be used by this check.")


notebook_path = Path(__file__).resolve().parents[1] / "blackboard_imp2.ipynb"
notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
namespace = globals()

for target in ("def parse_json_object", "class Blackboard", "class VerifierAgent"):
    cell = next(
        cell
        for cell in notebook["cells"]
        if target in "".join(cell.get("source", []))
    )
    exec("".join(cell["source"]), namespace)


def mock_gemini_caller(*args, **kwargs):
    return (
        '{"verdict": "SUPPORTED", "explanation": "test", '
        '"supporting_evidence_ids": ["real-evidence-1", '
        '"some-fake-id-not-in-evidence"], "confidence": 0.9}'
    )


blackboard = Blackboard()
blackboard.write("claim", "This is a test claim.")
blackboard.write(
    "retrieved_evidence",
    [
        {
            "id": "real-evidence-1",
            "text": "Evidence supporting the test claim.",
            "metadata": {"source": "mock"},
            "distance": 0.1,
        },
        {
            "id": "real-evidence-2",
            "text": "Additional evidence for the test claim.",
            "metadata": {"source": "mock"},
            "distance": 0.2,
        },
    ],
)

result = VerifierAgent(gemini_caller=mock_gemini_caller).verify(blackboard)
print(result["supporting_evidence_ids"])

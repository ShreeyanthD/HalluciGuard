"""
pipeline.py — the merge point between the two systems.

    probe/          -> QwenResidualFeatureExtractor + a trained scaler/
                        LogisticRegression probe. Given a question (and,
                        optionally, a candidate answer), it produces
                        prob_hallucinated in [0, 1].

    blackboard/     -> blackboard_core.process_response(prompt, response,
                        confidence_score). Below HALLUCINATION_RISK_THRESHOLD
                        the response passes through unchanged; at/above it,
                        the Blackboard agents (ClaimExtractor -> MemoryAgent
                        -> RetrievalAgent -> VerifierAgent -> CorrectionAgent)
                        verify the flagged claim against a knowledge base and
                        correct or hedge the response.

This file is the wiring: run the probe, feed its score into
process_response() as confidence_score, and return the final,
possibly-corrected response plus the full trace from both stages.

No logic inside probe/ or blackboard/ was changed to make this work — the
two systems already agreed on the contract (a 0-1 risk score in, a response
out), so the merge is purely compositional.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
for _sub in ("probe", "blackboard"):
    _p = str(_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llama_features import QwenResidualFeatureExtractor  # probe/llama_features.py
import blackboard_core as bc  # blackboard/blackboard_core.py


class HallucinationMitigationPipeline:
    """End-to-end: question -> answer -> risk score -> (maybe) grounded correction.

    Loads the Qwen feature extractor once (GPU) and the trained probe
    (scaler + classifier) once, then reuses both across calls.
    """

    def __init__(
        self,
        probe_path: str,
        layer: int = 20,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        override_threshold: Optional[float] = None,
    ):
        bundle = joblib.load(probe_path)
        self.scaler = bundle["scaler"]
        self.classifier = bundle["classifier"]
        self.feature_dim = bundle.get("feature_dim")

        self.extractor = QwenResidualFeatureExtractor(model_id=model_id, layer=layer)

        # Let the CLI/API override the Blackboard's default risk threshold
        # (bc.HALLUCINATION_RISK_THRESHOLD, default 0.70) without editing
        # blackboard_core.py.
        if override_threshold is not None:
            bc.HALLUCINATION_RISK_THRESHOLD = override_threshold

    # -- scoring -----------------------------------------------------------

    def _score(self, feature_vector: torch.Tensor) -> float:
        X = feature_vector.cpu().to(torch.float32).numpy()[None, :]
        if self.feature_dim is not None and X.shape[1] != self.feature_dim:
            raise ValueError(
                f"Feature dim mismatch: probe was trained on dim {self.feature_dim}, "
                f"got dim {X.shape[1]}. Is --layer the same one the probe was trained on?"
            )
        X_s = self.scaler.transform(X)
        prob_hallucinated = float(self.classifier.predict_proba(X_s)[:, 1][0])
        return prob_hallucinated

    # -- the two entry points ------------------------------------------------

    def run(self, question: str, max_new_tokens: int = 32) -> Dict[str, Any]:
        """Generate an answer with Qwen, score it, and run it through the
        Blackboard pipeline. Use this when you don't already have a response
        (the common case — the probe drives generation itself)."""
        gen = self.extractor.batch_generate_and_extract_features(
            [question], max_new_tokens=max_new_tokens
        )[0]
        return self._finish(question, gen["answer_text"], gen["feature_vector"])

    def run_on_qa(self, question: str, answer: str) -> Dict[str, Any]:
        """Score and mitigate an existing (question, answer) pair — e.g. a
        response produced by a different model upstream — without generating
        a new answer."""
        feat = self.extractor.batch_extract_features_for_qa_pairs([question], [answer])[0]
        return self._finish(question, answer, feat["feature_vector"])

    def _finish(self, question: str, answer: str, feature_vector: torch.Tensor) -> Dict[str, Any]:
        prob_hallucinated = self._score(feature_vector)

        bb_result = bc.process_response(
            prompt=question,
            response=answer,
            confidence_score=prob_hallucinated,
        )

        return {
            "question": question,
            "generated_answer": answer,
            "prob_hallucinated": round(prob_hallucinated, 4),
            "threshold": bb_result["threshold"],
            "blackboard_triggered": bb_result["pipeline_triggered"],
            "extracted_claim": bb_result.get("extracted_claim"),
            "verification_result": bb_result.get("verification_result"),
            "correction_result": bb_result.get("correction_result"),
            "final_response": bb_result["final_response"],
            "blackboard_raw": bb_result,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_result(r: Dict[str, Any]) -> None:
    tag = "HALLUCINATED" if r["prob_hallucinated"] >= r["threshold"] else "low-risk"
    print(f"\n[{tag:>12}] p(hallucinated)={r['prob_hallucinated']:.3f} (threshold={r['threshold']})")
    print(f"  Q: {r['question']}")
    print(f"  A (generated): {r['generated_answer']}")
    if r["blackboard_triggered"]:
        print(f"  Claim checked: {r['extracted_claim']}")
        v = r["verification_result"] or {}
        print(f"  Verdict: {v.get('verdict')}  ({v.get('explanation', '')[:120]})")
        print(f"  Final response: {r['final_response']}")
    else:
        print("  Blackboard skipped (below risk threshold) — response passed through unchanged.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, help="Path to trained probe .joblib (from probe/train_probe.py)")
    parser.add_argument("--questions", nargs="*", default=None)
    parser.add_argument("--questions_file", default=None, help="One question per line")
    parser.add_argument("--layer", type=int, default=20, help="Must match the layer the probe was trained on")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Override the Blackboard's default risk threshold (0.70)")
    parser.add_argument("--output", default=None, help="Optional path to write JSON results")
    args = parser.parse_args()

    questions: List[str] = list(args.questions) if args.questions else []
    if args.questions_file:
        with open(args.questions_file) as f:
            questions += [line.strip() for line in f if line.strip()]
    if not questions:
        parser.error("Provide --questions or --questions_file.")

    pipeline = HallucinationMitigationPipeline(
        probe_path=args.probe,
        layer=args.layer,
        model_id=args.model_id,
        override_threshold=args.threshold,
    )

    results = []
    for q in questions:
        r = pipeline.run(q)
        _print_result(r)
        results.append(r)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()

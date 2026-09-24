Edit ONLY MemoryAgent.log_verified_claim. Two exact replacements.

FIND:
    ) -> str:
        """Store the completed verification outcome in episodic memory."""
REPLACE WITH:
    ) -> Optional[str]:
        """Store a verification outcome only if it is grounded. Returns record id, or None if skipped."""

FIND:
        if not claim:
            raise ValueError("Cannot log memory without a claim.")

        record_id = str(uuid.uuid4())
REPLACE WITH:
        if not claim:
            raise ValueError("Cannot log memory without a claim.")

        verification_result = blackboard.read("verification_result", {}) or {}
        evidence_ids = verification_result.get("supporting_evidence_ids", []) or []
        correction_result = blackboard.read("correction_result", None)

        grounded = (
            (verdict == "SUPPORTED" and len(evidence_ids) > 0)
            or (
                verdict == "CONTRADICTED"
                and len(evidence_ids) > 0
                and bool(correction_result)
            )
        )

        if not grounded:
            blackboard.write(
                "memory_write_skipped",
                f"ungrounded (verdict={verdict}, evidence_ids={len(evidence_ids)})",
                author=self.__class__.__name__,
            )
            return None

        record_id = str(uuid.uuid4())

Show the diff. Do not run any Gemini calls.

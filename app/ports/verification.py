"""Canonical answer-verification boundary used by RAG mode adapters."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.contracts import EvidenceChunk, VerificationSummary


@runtime_checkable
class VerificationPort(Protocol):
    """Verify an answer against the exact evidence returned to the caller."""

    def verify(
        self,
        *,
        answer: str,
        evidence: Sequence[EvidenceChunk],
    ) -> VerificationSummary: ...

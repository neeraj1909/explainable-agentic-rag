"""Canonical retrieval boundary used by RAG mode adapters."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.contracts import EvidenceChunk


@runtime_checkable
class RetrievalPort(Protocol):
    """Retrieve canonical evidence without exposing vector-store types."""

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> Sequence[EvidenceChunk]: ...

"""Adapters from evaluation questions to the canonical RAG service."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from app.bootstrap import build_default_rag_service
from app.contracts import RAGMode, RAGRequest, RAGResponse
from app.services.rag_service import RAGService


COMPARISON_MODES = (
    RAGMode.two_step,
    RAGMode.agentic,
    RAGMode.graph,
    RAGMode.multi_agent,
)


class EvaluationSystem(Protocol):
    """Answer evaluation questions for one declared RAG mode."""

    mode: RAGMode

    def answer(self, question: str) -> RAGResponse: ...


@dataclass(frozen=True, slots=True)
class RAGSystemAdapter:
    """Submit one mode's evaluation requests through the application service."""

    mode: RAGMode
    service: RAGService
    top_k: int
    max_retries: int | None = None

    def answer(self, question: str) -> RAGResponse:
        return self.service.answer(
            RAGRequest(
                query=question,
                mode=self.mode,
                top_k=self.top_k,
                max_retries=self.max_retries,
            )
        )


def build_systems(
    *,
    modes: Iterable[RAGMode],
    top_k: int,
    max_retries: int | None = None,
    service: RAGService | None = None,
) -> Mapping[RAGMode, EvaluationSystem]:
    """Build ordered mode adapters that share one set of runtime dependencies."""

    selected_modes = tuple(dict.fromkeys(modes))
    if not selected_modes:
        raise ValueError("at least one evaluation mode must be selected")

    unsupported = [
        mode.value for mode in selected_modes if mode not in COMPARISON_MODES
    ]
    if unsupported:
        raise ValueError(
            "evaluation supports only local-PDF modes; unsupported: "
            f"{', '.join(unsupported)}"
        )

    resolved_service = service or build_default_rag_service(
        modes=selected_modes,
        top_k=top_k,
    )
    return {
        mode: RAGSystemAdapter(
            mode=mode,
            service=resolved_service,
            top_k=top_k,
            max_retries=max_retries,
        )
        for mode in selected_modes
    }

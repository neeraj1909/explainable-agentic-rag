"""Composition root for the canonical RAG application service."""

from __future__ import annotations

from collections.abc import Mapping
from uuid import uuid4

from opentelemetry import trace

from app.contracts import RAGMode, RAGRequest
from app.services.rag_service import (
    ModeHandler,
    RAGService,
    RunContext,
    RunContextProvider,
)


def build_rag_service(
    *,
    mode_handlers: Mapping[RAGMode, ModeHandler],
    run_context_provider: RunContextProvider | None = None,
) -> RAGService:
    """Compose a service without constructing mode dependencies in callers."""

    return RAGService(
        mode_handlers=mode_handlers,
        run_context_provider=run_context_provider or create_run_context,
    )


def create_run_context(request: RAGRequest) -> RunContext:
    """Create a run identity and capture the active OpenTelemetry trace, if any."""

    del request
    span_context = trace.get_current_span().get_span_context()
    trace_id = (
        f"{span_context.trace_id:032x}"
        if span_context.is_valid and span_context.trace_id
        else None
    )
    return RunContext(run_id=str(uuid4()), trace_id=trace_id)

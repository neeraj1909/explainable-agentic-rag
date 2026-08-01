"""Composition root for the canonical RAG application service."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
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


def build_default_rag_service(
    *,
    modes: Iterable[RAGMode],
    top_k: int,
    max_results: int = 5,
    checkpoint_file: Path | None = None,
    run_context_provider: RunContextProvider | None = None,
    llm: Any | None = None,
    retriever: Any | None = None,
) -> RAGService:
    """Compose selected production mode adapters and share expensive resources."""

    selected_modes = tuple(dict.fromkeys(modes))
    if not selected_modes:
        raise ValueError("at least one RAG mode must be selected")

    from langgraph.checkpoint.memory import InMemorySaver

    from app.config import get_llm_client
    from app.graphs.agentic_rag_graph import build_rag_graph, build_rag_graph_nodes
    from app.graphs.multi_agent_graph import (
        build_multi_agent_graph_nodes,
        build_multi_agent_rag_graph,
        build_persistent_checkpointer,
    )
    from app.main import build_agent as build_research_agent
    from app.rag.agentic_rag import build_agentic_mode
    from app.rag.mode_adapters import (
        GraphModeAdapter,
        MultiAgentModeAdapter,
        ResearchAssistantModeAdapter,
    )
    from app.rag.retriever import build_attributed_retriever
    from app.rag.two_step_rag import build_two_step_mode

    local_modes = {
        RAGMode.two_step,
        RAGMode.agentic,
        RAGMode.graph,
        RAGMode.multi_agent,
    }
    needs_llm = any(mode in local_modes for mode in selected_modes) or (
        RAGMode.research_assistant in selected_modes
    )
    resolved_llm = llm if llm is not None else (get_llm_client() if needs_llm else None)
    needs_retriever = any(mode in local_modes for mode in selected_modes)
    resolved_retriever = retriever
    if resolved_retriever is None and needs_retriever:
        resolved_retriever = build_attributed_retriever(k=top_k)

    handlers: dict[RAGMode, ModeHandler] = {}

    if RAGMode.two_step in selected_modes:
        mode = build_two_step_mode(
            k=top_k,
            retriever=resolved_retriever,
            llm=resolved_llm,
        )
        handlers[RAGMode.two_step] = ModeHandler(answer=mode.answer)

    if RAGMode.agentic in selected_modes:
        mode = build_agentic_mode(
            k=top_k,
            retriever=resolved_retriever,
            llm=resolved_llm,
        )
        handlers[RAGMode.agentic] = ModeHandler(answer=mode.answer)

    if RAGMode.graph in selected_modes:
        nodes = build_rag_graph_nodes(
            k=top_k,
            retriever=resolved_retriever,
            llm=resolved_llm,
        )
        mode = GraphModeAdapter(
            build_rag_graph(nodes=nodes, checkpointer=InMemorySaver())
        )
        handlers[RAGMode.graph] = ModeHandler(
            answer=mode.answer,
            stream=mode.stream,
        )

    if RAGMode.multi_agent in selected_modes:
        nodes = build_multi_agent_graph_nodes(
            k=top_k,
            retriever=resolved_retriever,
            llm=resolved_llm,
        )
        checkpointer = (
            build_persistent_checkpointer(checkpoint_file)
            if checkpoint_file is not None
            else build_persistent_checkpointer()
        )
        mode = MultiAgentModeAdapter(
            build_multi_agent_rag_graph(nodes=nodes, checkpointer=checkpointer)
        )
        handlers[RAGMode.multi_agent] = ModeHandler(
            answer=mode.answer,
            stream=mode.stream,
        )

    if RAGMode.research_assistant in selected_modes:
        mode = ResearchAssistantModeAdapter(
            build_research_agent(max_results=max_results, llm=resolved_llm)
        )
        handlers[RAGMode.research_assistant] = ModeHandler(
            answer=mode.answer,
            stream=mode.stream,
        )

    return build_rag_service(
        mode_handlers=handlers,
        run_context_provider=run_context_provider,
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

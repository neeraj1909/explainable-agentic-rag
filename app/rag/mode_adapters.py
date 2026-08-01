"""Adapters from current LangChain/LangGraph modes to canonical contracts."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.contracts import (
    ClaimStatus,
    ClaimVerification,
    ConfidenceEstimate,
    EvidenceChunk,
    ProgressEvent,
    RAGMode,
    RAGRequest,
    RAGResponse,
    RouteStep,
    RunMetrics,
    ScoreProvenance,
    VerificationStatus,
    VerificationSummary,
    adapt_agent_response,
)
from app.schemas import AgentResponse, NextAction
from app.services.rag_service import RunContext, StreamItem


@dataclass(slots=True)
class TwoStepModeAdapter:
    """Run fixed retrieve-then-generate RAG behind the canonical boundary."""

    retriever: Any
    answer_chain: Any

    def answer(self, request: RAGRequest, context: RunContext) -> RAGResponse:
        started = time.perf_counter()
        docs = list(self.retriever.invoke(request.query))
        if request.top_k is not None:
            docs = docs[: request.top_k]
        evidence = evidence_from_items(docs)
        answer = _require_answer(
            self.answer_chain.invoke(
                {
                    "question": request.query,
                    "context": format_evidence_context(evidence),
                }
            )
        )

        return RAGResponse(
            mode=RAGMode.two_step,
            answer=answer,
            explanation="Retrieved one evidence set, then generated the answer.",
            evidence=evidence,
            metrics=RunMetrics(
                latency_ms=_elapsed_ms(started),
                model_calls=1,
                retrieval_calls=1,
            ),
        )


@dataclass(slots=True)
class AgenticModeAdapter:
    """Translate a LangChain tool-using agent result into a canonical response."""

    agent: Any

    def answer(self, request: RAGRequest, context: RunContext) -> RAGResponse:
        started = time.perf_counter()
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": request.query}]}
        )
        messages = _message_payloads(result)
        answer = _require_answer(_final_ai_answer(messages))
        tool_calls = _tool_calls(messages)
        evidence = _agentic_evidence(messages)
        route_history = [
            RouteStep(
                step=index,
                agent="agentic_rag",
                decision=call.get("name") or "tool_call",
                reason="The model selected this tool while answering the request.",
                called_by="agentic_rag",
            )
            for index, call in enumerate(tool_calls, start=1)
        ]

        return RAGResponse(
            mode=RAGMode.agentic,
            answer=answer,
            explanation=(
                "The model decided whether and how often to retrieve evidence."
            ),
            evidence=evidence,
            metrics=RunMetrics(
                latency_ms=_elapsed_ms(started),
                retrieval_calls=sum(
                    call.get("name") == "retrieve_documents" for call in tool_calls
                ),
                tool_calls=len(tool_calls),
            ),
            route_history=route_history,
        )


@dataclass(slots=True)
class GraphModeAdapter:
    """Run the single LangGraph workflow behind the canonical boundary."""

    graph: Any

    def answer(self, request: RAGRequest, context: RunContext) -> RAGResponse:
        started = time.perf_counter()
        result = self.graph.invoke(
            _single_graph_inputs(request),
            _graph_config(request, context),
        )
        final = _graph_final(result)
        return response_from_graph_final(
            final,
            mode=RAGMode.graph,
            latency_ms=_elapsed_ms(started),
        )

    def stream(
        self,
        request: RAGRequest,
        context: RunContext,
    ) -> Iterator[StreamItem]:
        yield from _stream_graph(
            graph=self.graph,
            inputs=_single_graph_inputs(request),
            config=_graph_config(request, context),
            mode=RAGMode.graph,
        )


@dataclass(slots=True)
class MultiAgentModeAdapter:
    """Run the orchestrator-led graph behind the canonical boundary."""

    graph: Any

    def answer(self, request: RAGRequest, context: RunContext) -> RAGResponse:
        started = time.perf_counter()
        try:
            result = self.graph.invoke(
                _multi_agent_inputs(request),
                _graph_config(request, context),
            )
            final = _graph_final(result)
            return response_from_graph_final(
                final,
                mode=RAGMode.multi_agent,
                latency_ms=_elapsed_ms(started),
            )
        finally:
            _sync_checkpointer(self.graph)

    def stream(
        self,
        request: RAGRequest,
        context: RunContext,
    ) -> Iterator[StreamItem]:
        try:
            yield from _stream_graph(
                graph=self.graph,
                inputs=_multi_agent_inputs(request),
                config=_graph_config(request, context),
                mode=RAGMode.multi_agent,
            )
        finally:
            _sync_checkpointer(self.graph)


@dataclass(slots=True)
class ResearchAssistantModeAdapter:
    """Adapt the existing structured research agent without changing its schema."""

    agent: Any

    def answer(self, request: RAGRequest, context: RunContext) -> RAGResponse:
        started = time.perf_counter()
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": request.query}]}
        )
        payload = _as_mapping(result)
        structured_response = AgentResponse.model_validate(
            payload.get("structured_response")
        )
        return adapt_agent_response(
            structured_response,
            run_id=context.run_id,
            trace_id=context.trace_id,
            corpus_version=context.corpus_version,
            index_version=context.index_version,
            metrics=RunMetrics(
                latency_ms=_elapsed_ms(started),
                model_calls=1,
            ),
        )

    def stream(
        self,
        request: RAGRequest,
        context: RunContext,
    ) -> Iterator[StreamItem]:
        started = time.perf_counter()
        structured_response: AgentResponse | None = None
        seen_tool_calls: set[str] = set()

        for stream_mode, chunk in self.agent.stream(
            {"messages": [{"role": "user", "content": request.query}]},
            stream_mode=["updates", "custom"],
        ):
            chunk_payload = _as_mapping(chunk)
            if stream_mode == "custom":
                yield ProgressEvent(
                    event=_optional_text(chunk_payload.get("event")) or "progress",
                    message=_optional_text(chunk_payload.get("message"))
                    or "Research assistant progress.",
                    mode=RAGMode.research_assistant,
                    data=_json_mapping(chunk_payload.get("data")),
                )
                continue

            if stream_mode != "updates":
                continue
            for update in chunk_payload.values():
                update_payload = _as_mapping(update)
                for message in list(update_payload.get("messages") or []):
                    for call in list(_as_mapping(message).get("tool_calls") or []):
                        call_payload = _as_mapping(call)
                        name = _optional_text(call_payload.get("name")) or "tool"
                        identity = _optional_text(call_payload.get("id")) or repr(
                            call_payload
                        )
                        if identity in seen_tool_calls:
                            continue
                        seen_tool_calls.add(identity)
                        yield ProgressEvent(
                            event="tool_selected",
                            message=f"Selected {name}.",
                            mode=RAGMode.research_assistant,
                            agent="research_assistant",
                            decision=name,
                        )

                candidate = update_payload.get("structured_response")
                if candidate is not None:
                    structured_response = AgentResponse.model_validate(candidate)

        if structured_response is None:
            raise RuntimeError(
                "research-assistant stream ended without a structured response"
            )
        yield adapt_agent_response(
            structured_response,
            run_id=context.run_id,
            trace_id=context.trace_id,
            corpus_version=context.corpus_version,
            index_version=context.index_version,
            metrics=RunMetrics(
                latency_ms=_elapsed_ms(started),
                model_calls=1,
                tool_calls=len(seen_tool_calls),
            ),
        )


def evidence_from_items(items: Sequence[Any]) -> list[EvidenceChunk]:
    """Convert documents or source mappings and remove duplicate chunks."""

    evidence: list[EvidenceChunk] = []
    seen_chunk_ids: set[str] = set()
    for item in items:
        chunk = evidence_from_item(item)
        if chunk.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.chunk_id)
        evidence.append(chunk)
    return evidence


def evidence_from_item(item: Any) -> EvidenceChunk:
    payload = _as_mapping(item)
    metadata = _as_mapping(payload.get("metadata"))
    values = {**metadata, **payload}

    content = _optional_text(
        values.get("content")
        or values.get("page_content")
        or getattr(item, "page_content", None)
    )
    source = _optional_text(values.get("source"))
    title = _optional_text(values.get("title"))
    page = _non_negative_int(values.get("page"))
    raw_chunk_id = _optional_text(values.get("chunk_id") or getattr(item, "id", None))
    identity = f"{source or ''}\0{page!s}\0{content or ''}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    chunk_id = raw_chunk_id or f"chunk:sha256:{digest}"
    document_id = (
        _optional_text(values.get("document_id"))
        or source
        or f"document:sha256:{digest}"
    )

    scores: list[ScoreProvenance] = []
    retriever_score = _finite_float(values.get("retriever_score"))
    if retriever_score is not None:
        scores.append(
            ScoreProvenance(
                name="retriever",
                value=retriever_score,
                rank=_positive_int(values.get("retriever_rank")),
                method="vectorstore_raw_score",
            )
        )
    reranker_score = _finite_float(values.get("reranker_score"))
    if reranker_score is not None:
        scores.append(
            ScoreProvenance(
                name="reranker",
                value=reranker_score,
                rank=_positive_int(values.get("selected_rank")),
                higher_is_better=True,
                method="embedding_cosine_similarity",
            )
        )

    return EvidenceChunk(
        document_id=document_id,
        chunk_id=chunk_id,
        source=source,
        title=title,
        page=page,
        content=content,
        scores=scores,
        selected_rank=_positive_int(values.get("selected_rank")),
        reason_selected=_optional_text(values.get("reason_selected")),
    )


def format_evidence_context(evidence: Sequence[EvidenceChunk]) -> str:
    """Format canonical evidence for the existing generation prompts."""

    return "\n\n".join(
        (
            f"[source={chunk.source} chunk={chunk.chunk_id} page={chunk.page} "
            f"selected_rank={chunk.selected_rank} "
            f"reason_selected={chunk.reason_selected}]\n{chunk.content or ''}"
        )
        for chunk in evidence
    )


def response_from_graph_final(
    final: Mapping[str, Any],
    *,
    mode: RAGMode,
    latency_ms: float | None = None,
) -> RAGResponse:
    """Translate either graph's final state into one canonical response."""

    evidence = evidence_from_items(list(final.get("sources") or []))
    verification = _verification_from_graph(final, evidence)
    retry_count = _non_negative_int(final.get("retry_count")) or 0
    needs_human_review = bool(final.get("needs_human_review"))
    return RAGResponse(
        mode=mode,
        answer=_require_answer(final.get("answer")),
        explanation=_optional_text(final.get("explanation")),
        evidence=evidence,
        verification=verification,
        confidence=ConfidenceEstimate(
            score=verification.score,
            method=("verification_score" if verification.score is not None else None),
            calibrated=False,
        ),
        metrics=RunMetrics(
            latency_ms=latency_ms,
            retrieval_calls=(retry_count + 1 if evidence else 0),
            retry_count=retry_count,
        ),
        route_history=_route_history(final.get("route_history")),
        next_action=(
            NextAction.human_review
            if needs_human_review
            else NextAction.no_follow_up_needed
            if verification.verified
            else None
        ),
    )


def _verification_from_graph(
    final: Mapping[str, Any],
    evidence: Sequence[EvidenceChunk],
) -> VerificationSummary:
    score = _normalized_score(final.get("faithfulness_score"))
    unsupported_claims = [
        text
        for claim in list(final.get("unsupported_claims") or [])
        if (text := _optional_text(claim)) is not None
    ]
    verified = bool(final.get("verified")) and bool(evidence)
    if verified:
        status = VerificationStatus.verified
    elif score is not None or unsupported_claims:
        status = VerificationStatus.failed
    else:
        status = VerificationStatus.not_run

    claims = [
        ClaimVerification(
            claim_id=_stable_claim_id(claim),
            claim=claim,
            status=ClaimStatus.unsupported,
            reason="The graph verifier marked this claim as unsupported.",
        )
        for claim in unsupported_claims
    ]
    return VerificationSummary(
        status=status,
        verified=verified,
        score=score,
        method=_optional_text(final.get("verification_method"))
        or ("token_overlap_stub" if score is not None else None),
        claims=claims,
        unsupported_claims=unsupported_claims,
    )


def _stream_graph(
    *,
    graph: Any,
    inputs: dict[str, Any],
    config: dict[str, Any],
    mode: RAGMode,
) -> Iterator[StreamItem]:
    started = time.perf_counter()
    state = dict(inputs)
    for update in graph.stream(inputs, config=config, stream_mode="updates"):
        update_payload = _as_mapping(update)
        node_name = next(iter(update_payload), "graph")
        node_payload = _as_mapping(update_payload.get(node_name))
        state.update(node_payload)
        if node_name == "__interrupt__":
            state["__interrupt__"] = update_payload.get(node_name)
        route = _latest_route(node_payload)
        yield ProgressEvent(
            event="agent_step" if route else "node_update",
            message=(
                f"{route.get('agent', node_name)} -> "
                f"{route.get('decision', 'completed')}"
                if route
                else f"{node_name} completed."
            ),
            mode=mode,
            agent=_optional_text(route.get("agent")) if route else node_name,
            decision=_optional_text(route.get("decision")) if route else None,
            reason=_optional_text(route.get("reason")) if route else None,
            retry_count=_non_negative_int(
                route.get("retry_count") if route else node_payload.get("retry_count")
            ),
            verified=_optional_bool(
                route.get("verified") if route else node_payload.get("verified")
            ),
            verification_score=_normalized_score(
                route.get("faithfulness_score")
                if route
                else node_payload.get("faithfulness_score")
            ),
            data={"node": node_name},
        )

        if node_name == "finalize" and node_payload.get("final") is not None:
            yield response_from_graph_final(
                _as_mapping(node_payload["final"]),
                mode=mode,
                latency_ms=_elapsed_ms(started),
            )
            return

    if state.get("__interrupt__") is not None and state.get("answer"):
        yield response_from_graph_final(
            _graph_final(state),
            mode=mode,
            latency_ms=_elapsed_ms(started),
        )
        return

    raise RuntimeError(f"{mode.value} graph stream ended without a final response")


def _single_graph_inputs(request: RAGRequest) -> dict[str, Any]:
    return {
        "question": request.query,
        "retry_count": 0,
        "max_retries": request.max_retries if request.max_retries is not None else 2,
    }


def _multi_agent_inputs(request: RAGRequest) -> dict[str, Any]:
    return {
        "question": request.query,
        "query": "",
        "retry_count": 0,
        "max_retries": request.max_retries if request.max_retries is not None else 2,
        "route_history": [],
    }


def _graph_config(request: RAGRequest, context: RunContext) -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": request.thread_id or context.run_id,
        }
    }


def _graph_final(result: Any) -> Mapping[str, Any]:
    payload = _as_mapping(result)
    final = payload.get("final")
    if final is not None:
        return _as_mapping(final)
    if payload.get("__interrupt__") is not None and payload.get("answer"):
        return {
            "answer": payload["answer"],
            "explanation": payload.get("human_review_reason")
            or "The draft requires human review before approval.",
            "sources": list(payload.get("docs") or []),
            "faithfulness_score": payload.get("faithfulness_score"),
            "unsupported_claims": list(payload.get("unsupported_claims") or []),
            "verified": False,
            "retry_count": payload.get("retry_count", 0),
            "needs_human_review": True,
            "route_history": list(payload.get("route_history") or []),
        }
    raise RuntimeError("graph execution ended without a final response")


def _message_payloads(result: Any) -> list[dict[str, Any]]:
    payload = _as_mapping(result)
    return [_as_mapping(message) for message in list(payload.get("messages") or [])]


def _final_ai_answer(messages: Sequence[Mapping[str, Any]]) -> str | None:
    for message in reversed(messages):
        if message.get("type") not in {"ai", "assistant"}:
            continue
        content = _content_text(message.get("content"))
        if content:
            return content
    return None


def _tool_calls(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in messages:
        calls.extend(
            _as_mapping(call) for call in list(message.get("tool_calls") or [])
        )
    return calls


def _agentic_evidence(messages: Sequence[Mapping[str, Any]]) -> list[EvidenceChunk]:
    sources: list[Any] = []
    for message in messages:
        if message.get("type") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            sources.extend(list(payload.get("results") or []))
    return evidence_from_items(sources)


def _route_history(value: Any) -> list[RouteStep]:
    routes: list[RouteStep] = []
    for index, item in enumerate(list(value or []), start=1):
        payload = _as_mapping(item)
        routes.append(
            RouteStep(
                step=_positive_int(payload.get("step")) or index,
                agent=_optional_text(payload.get("agent")) or "graph",
                decision=_optional_text(payload.get("decision")) or "completed",
                reason=_optional_text(payload.get("reason")) or "Graph step completed.",
                called_by=_optional_text(payload.get("called_by")),
                retry_count=_non_negative_int(payload.get("retry_count")),
                verified=_optional_bool(payload.get("verified")),
                verification_score=_normalized_score(
                    payload.get("verification_score")
                    if payload.get("verification_score") is not None
                    else payload.get("faithfulness_score")
                ),
            )
        )
    return routes


def _latest_route(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    history = list(payload.get("route_history") or [])
    return _as_mapping(history[-1]) if history else {}


def _sync_checkpointer(graph: Any) -> None:
    checkpointer = getattr(graph, "checkpointer", None)
    for attribute in ("storage", "writes", "blobs"):
        store = getattr(checkpointer, attribute, None)
        sync = getattr(store, "sync", None)
        if callable(sync):
            sync()


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    return {}


def _json_mapping(value: Any) -> dict[str, Any]:
    payload = _as_mapping(value)
    return json.loads(json.dumps(payload, default=str))


def _content_text(value: Any) -> str | None:
    if isinstance(value, str):
        return _optional_text(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        parts = []
        for block in value:
            payload = _as_mapping(block)
            text = _optional_text(payload.get("text"))
            if text:
                parts.append(text)
        return "\n".join(parts) or None
    return None


def _require_answer(value: Any) -> str:
    answer = _optional_text(value)
    if answer is None:
        raise RuntimeError("RAG mode completed without a final answer")
    return answer


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _normalized_score(value: Any) -> float | None:
    result = _finite_float(value)
    return result if result is not None and 0.0 <= result <= 1.0 else None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _stable_claim_id(claim: str) -> str:
    digest = hashlib.sha256(claim.encode("utf-8")).hexdigest()
    return f"claim:sha256:{digest}"


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000

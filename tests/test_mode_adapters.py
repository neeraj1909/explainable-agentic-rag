import json
from collections.abc import Iterator

import pytest
from langchain_core.documents import Document

from app.bootstrap import build_rag_service
from app.contracts import ProgressEvent, RAGMode, RAGRequest, RAGResponse
from app.rag.mode_adapters import (
    AgenticModeAdapter,
    GraphModeAdapter,
    MultiAgentModeAdapter,
    ResearchAssistantModeAdapter,
    TwoStepModeAdapter,
)
from app.schemas import AgentResponse, NextAction, SourceUsed
from app.services.rag_service import ModeHandler, RunContext


class FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def invoke(self, query: str) -> list[Document]:
        self.queries.append(query)
        return [
            Document(
                page_content="Canonical evidence for the shared fake answer.",
                metadata={
                    "source": "docs/example.pdf",
                    "chunk_id": "chunk-1",
                    "page": 2,
                    "retriever_score": 0.25,
                    "retriever_rank": 3,
                    "reranker_score": 0.91,
                    "selected_rank": 1,
                    "reason_selected": "Best fake result.",
                },
            )
        ]


class FakeAnswerChain:
    def __init__(self, answer: str = "Shared fake answer.") -> None:
        self.answer = answer
        self.invocations: list[dict[str, str]] = []

    def invoke(self, payload: dict[str, str]) -> str:
        self.invocations.append(payload)
        return self.answer


class FakeAgenticAgent:
    def invoke(self, payload: dict) -> dict:
        query = payload["messages"][0]["content"]
        return {
            "messages": [
                {
                    "type": "ai",
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "retrieve_documents",
                            "args": {"query": query, "k": 3},
                        }
                    ],
                },
                {
                    "type": "tool",
                    "name": "retrieve_documents",
                    "content": json.dumps(
                        {
                            "query": query,
                            "retrieved_count": 1,
                            "results": [
                                {
                                    "source": "docs/example.pdf",
                                    "chunk_id": "chunk-1",
                                    "page": 2,
                                    "content": (
                                        "Canonical evidence for the shared fake answer."
                                    ),
                                    "retriever_score": 0.25,
                                    "retriever_rank": 3,
                                    "selected_rank": 1,
                                    "reason_selected": "Tool-selected evidence.",
                                }
                            ],
                        }
                    ),
                },
                {
                    "type": "ai",
                    "content": "Shared fake answer.",
                    "tool_calls": [],
                },
            ]
        }


class FakeGraph:
    def __init__(self, final: dict, updates: list[dict] | None = None) -> None:
        self.final = final
        self.updates = updates or [{"finalize": {"final": final}}]
        self.invocations: list[tuple[dict, dict]] = []

    def invoke(self, inputs: dict, config: dict) -> dict:
        self.invocations.append((inputs, config))
        return {"final": self.final}

    def stream(
        self,
        inputs: dict,
        *,
        config: dict,
        stream_mode: str,
    ) -> Iterator[dict]:
        self.invocations.append((inputs, config))
        assert stream_mode == "updates"
        yield from self.updates


class FakeResearchAgent:
    @staticmethod
    def _response() -> AgentResponse:
        return AgentResponse(
            answer="Shared fake answer.",
            confidence=0.8,
            sources_used=[
                SourceUsed(
                    title="Example paper",
                    url="https://example.test/paper",
                    reason_used="Supports the fake answer.",
                )
            ],
            unsupported_claims=[],
            next_action=NextAction.no_follow_up_needed,
        )

    def invoke(self, payload: dict) -> dict:
        return {"structured_response": self._response()}

    def stream(
        self,
        payload: dict,
        *,
        stream_mode: list[str],
    ) -> Iterator[tuple[str, dict]]:
        assert payload["messages"][0]["content"] == "Use the same fake evidence."
        assert stream_mode == ["updates", "custom"]
        yield (
            "custom",
            {
                "event": "retrieval_started",
                "message": "Retrieving fake research.",
                "data": {"source": "arxiv"},
            },
        )
        yield (
            "updates",
            {
                "agent": {
                    "messages": [
                        {
                            "type": "ai",
                            "tool_calls": [
                                {"id": "tool-1", "name": "search_papers", "args": {}}
                            ],
                        }
                    ],
                    "structured_response": self._response(),
                }
            },
        )


class FakeInterruptedGraph:
    def invoke(self, inputs: dict, config: dict) -> dict:
        return {
            **inputs,
            "answer": "Draft requiring review.",
            "docs": FakeRetriever().invoke(inputs["question"]),
            "faithfulness_score": 0.2,
            "unsupported_claims": ["Draft requiring review."],
            "verified": False,
            "__interrupt__": [{"reason": "Verification failed."}],
        }

    def stream(
        self,
        inputs: dict,
        *,
        config: dict,
        stream_mode: str,
    ) -> Iterator[dict]:
        docs = FakeRetriever().invoke(inputs["question"])
        yield {"retrieve": {"docs": docs}}
        yield {"generate_answer": {"answer": "Draft requiring review."}}
        yield {
            "verify_claims": {
                "faithfulness_score": 0.2,
                "unsupported_claims": ["Draft requiring review."],
                "verified": False,
            }
        }
        yield {"__interrupt__": ({"reason": "Verification failed."},)}


def fixed_context(request: RAGRequest) -> RunContext:
    return RunContext(run_id=f"run-{request.mode.value}", trace_id="1" * 32)


def graph_final() -> dict:
    return {
        "answer": "Shared fake answer.",
        "sources": [
            {
                "source": "docs/example.pdf",
                "chunk_id": "chunk-1",
                "page": 2,
                "content": "Canonical evidence for the shared fake answer.",
                "retriever_score": 0.25,
                "retriever_rank": 3,
                "selected_rank": 1,
                "reason_selected": "Graph-selected evidence.",
            }
        ],
        "faithfulness_score": 0.9,
        "unsupported_claims": [],
        "verified": True,
        "retry_count": 0,
    }


def multi_agent_final() -> dict:
    return {
        **graph_final(),
        "explanation": "Specialists retrieved, answered, and verified.",
        "verification_method": "fake_verifier",
        "route_history": [
            {
                "step": 1,
                "agent": "retriever_agent",
                "decision": "evidence_retrieved",
                "reason": "Retrieved one fake chunk.",
                "called_by": "orchestrator",
            }
        ],
        "needs_human_review": False,
    }


def build_cross_mode_service():
    two_step = TwoStepModeAdapter(FakeRetriever(), FakeAnswerChain())
    agentic = AgenticModeAdapter(FakeAgenticAgent())
    graph = GraphModeAdapter(FakeGraph(graph_final()))
    multi_agent = MultiAgentModeAdapter(FakeGraph(multi_agent_final()))
    research = ResearchAssistantModeAdapter(FakeResearchAgent())

    return build_rag_service(
        mode_handlers={
            RAGMode.two_step: ModeHandler(answer=two_step.answer),
            RAGMode.agentic: ModeHandler(answer=agentic.answer),
            RAGMode.graph: ModeHandler(answer=graph.answer, stream=graph.stream),
            RAGMode.multi_agent: ModeHandler(
                answer=multi_agent.answer,
                stream=multi_agent.stream,
            ),
            RAGMode.research_assistant: ModeHandler(
                answer=research.answer,
                stream=research.stream,
            ),
        },
        run_context_provider=fixed_context,
    )


@pytest.mark.parametrize(
    "mode",
    [
        RAGMode.two_step,
        RAGMode.agentic,
        RAGMode.graph,
        RAGMode.multi_agent,
        RAGMode.research_assistant,
    ],
)
def test_same_fake_question_returns_canonical_response_for_every_mode(
    mode: RAGMode,
) -> None:
    service = build_cross_mode_service()

    response = service.answer(
        RAGRequest(query="Use the same fake evidence.", mode=mode, top_k=3)
    )

    assert response.schema_version == "1.0"
    assert response.mode is mode
    assert response.answer == "Shared fake answer."
    assert response.evidence
    if mode is not RAGMode.research_assistant:
        assert response.evidence[0].content == (
            "Canonical evidence for the shared fake answer."
        )
        retriever_score = next(
            score for score in response.evidence[0].scores if score.name == "retriever"
        )
        assert retriever_score.rank == 3
    assert response.run_id == f"run-{mode.value}"
    assert RAGResponse.model_validate_json(response.model_dump_json()) == response
    if mode is RAGMode.research_assistant:
        stream_items = list(
            service.stream(
                RAGRequest(
                    query="Use the same fake evidence.",
                    mode=mode,
                    stream=True,
                )
            )
        )
        assert any(
            isinstance(item, ProgressEvent) and item.event == "retrieval_started"
            for item in stream_items
        )
        assert isinstance(stream_items[-1], RAGResponse)


@pytest.mark.parametrize("mode", [RAGMode.graph, RAGMode.multi_agent])
def test_graph_modes_stream_canonical_progress_then_response(mode: RAGMode) -> None:
    service = build_cross_mode_service()

    items = list(
        service.stream(
            RAGRequest(query="Stream fake graph work.", mode=mode, stream=True)
        )
    )

    assert isinstance(items[0], ProgressEvent)
    assert items[0].event == "run_started"
    assert any(
        isinstance(item, ProgressEvent) and item.event in {"node_update", "agent_step"}
        for item in items[1:-1]
    )
    assert isinstance(items[-1], RAGResponse)
    assert items[-1].mode is mode


def test_single_graph_interrupt_returns_review_response_instead_of_crashing() -> None:
    adapter = GraphModeAdapter(FakeInterruptedGraph())

    response = adapter.answer(
        RAGRequest(query="Needs review.", mode=RAGMode.graph, max_retries=0),
        fixed_context(RAGRequest(query="Needs review.", mode=RAGMode.graph)),
    )

    assert response.answer == "Draft requiring review."
    assert response.next_action is NextAction.human_review
    assert response.verification.verified is False
    assert response.verification.unsupported_claims == ["Draft requiring review."]


def test_single_graph_interrupt_stream_ends_with_review_response() -> None:
    adapter = GraphModeAdapter(FakeInterruptedGraph())
    request = RAGRequest(
        query="Needs review.",
        mode=RAGMode.graph,
        max_retries=0,
        stream=True,
    )

    items = list(adapter.stream(request, fixed_context(request)))

    assert isinstance(items[-1], RAGResponse)
    assert items[-1].next_action is NextAction.human_review

import json

from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver

from app.graphs import agentic_rag_graph, multi_agent_graph


class FakeRetriever:
    def invoke(self, query: str) -> list[Document]:
        return [
            Document(
                page_content="Grounded answer evidence.",
                metadata={
                    "source": "docs/example.pdf",
                    "chunk_id": "chunk-1",
                    "page": 0,
                    "retriever_score": 0.2,
                    "retriever_rank": 1,
                    "selected_rank": 1,
                    "reason_selected": "Fake dependency.",
                },
            )
        ]


class FakeChain:
    def __init__(self, result: str) -> None:
        self.result = result

    def invoke(self, payload: dict) -> str:
        return self.result


def fake_verifier(*, answer: str, evidence: str, threshold: float = 0.35) -> str:
    return json.dumps(
        {
            "faithfulness_score": 1.0,
            "unsupported_claims": [],
            "verdict": "faithful",
            "method": "fake_verifier",
            "threshold": threshold,
        }
    )


def fail_provider_construction(*args, **kwargs):
    raise AssertionError("Injected graph construction must not load providers")


def test_single_graph_runs_with_injected_nodes_only(monkeypatch) -> None:
    monkeypatch.setattr(agentic_rag_graph, "get_llm_client", fail_provider_construction)
    monkeypatch.setattr(
        agentic_rag_graph,
        "build_attributed_retriever",
        fail_provider_construction,
    )
    nodes = agentic_rag_graph.RagGraphNodes(
        retriever=FakeRetriever(),
        answer_chain=FakeChain("Grounded answer."),
        rewrite_chain=FakeChain("rewritten query"),
        verifier=fake_verifier,
    )

    graph = agentic_rag_graph.build_rag_graph(
        nodes=nodes,
        checkpointer=InMemorySaver(),
    )
    result = graph.invoke(
        {"question": "What is grounded?", "retry_count": 0, "max_retries": 1},
        {"configurable": {"thread_id": "single-test"}},
    )

    assert result["final"]["answer"] == "Grounded answer."
    assert result["final"]["verified"] is True
    assert result["final"]["sources"][0]["chunk_id"] == "chunk-1"
    assert result["final"]["sources"][0]["content"] == "Grounded answer evidence."
    assert result["final"]["sources"][0]["retriever_rank"] == 1


def test_multi_agent_graph_runs_with_injected_nodes_only(monkeypatch) -> None:
    monkeypatch.setattr(multi_agent_graph, "get_llm_client", fail_provider_construction)
    monkeypatch.setattr(
        multi_agent_graph,
        "build_attributed_retriever",
        fail_provider_construction,
    )
    nodes = multi_agent_graph.MultiAgentGraphNodes(
        orchestrator=multi_agent_graph.OrchestratorAgent(),
        query_planner=multi_agent_graph.QueryPlannerAgent(FakeChain("grounded query")),
        retriever=multi_agent_graph.RetrieverAgent(FakeRetriever()),
        explainer=multi_agent_graph.ExplainerAgent(FakeChain("Grounded answer.")),
        verifier=multi_agent_graph.VerifierAgent(
            faithfulness_threshold=0.35,
            verifier=fake_verifier,
        ),
        finalizer=multi_agent_graph.finalize,
    )

    graph = multi_agent_graph.build_multi_agent_rag_graph(
        nodes=nodes,
        checkpointer=InMemorySaver(),
    )
    result = graph.invoke(
        {
            "question": "What is grounded?",
            "query": "",
            "retry_count": 0,
            "max_retries": 1,
            "route_history": [],
        },
        {"configurable": {"thread_id": "multi-test"}},
    )

    assert result["final"]["answer"] == "Grounded answer."
    assert result["final"]["verified"] is True
    assert result["final"]["route_history"]
    assert result["final"]["sources"][0]["content"] == "Grounded answer evidence."
    assert result["final"]["sources"][0]["retriever_rank"] == 1

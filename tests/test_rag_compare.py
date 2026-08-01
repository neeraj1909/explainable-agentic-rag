from app.contracts import RAGMode, RAGResponse
from app.rag import compare


class FakeService:
    def __init__(self) -> None:
        self.requests = []

    def answer(self, request):
        self.requests.append(request)
        return RAGResponse(
            mode=request.mode,
            answer=f"{request.mode.value} answer",
        )


def test_run_comparison_runs_both_rag_modes():
    service = FakeService()

    result = compare.run_comparison(
        query="Compare RAG modes",
        k=3,
        service=service,
    )

    assert result["query"] == "Compare RAG modes"
    assert result["two_step_rag"]["result"].mode is RAGMode.two_step
    assert result["agentic_rag"]["result"].mode is RAGMode.agentic
    assert isinstance(result["two_step_rag"]["latency_seconds"], float)
    assert isinstance(result["agentic_rag"]["latency_seconds"], float)
    assert [
        (request.mode, request.query, request.top_k) for request in service.requests
    ] == [
        (RAGMode.two_step, "Compare RAG modes", 3),
        (RAGMode.agentic, "Compare RAG modes", 3),
    ]


def test_run_comparison_uses_injected_canonical_service() -> None:
    service = FakeService()

    result = compare.run_comparison(
        query="Compare canonical modes",
        k=3,
        service=service,
    )

    assert [request.mode for request in service.requests] == [
        RAGMode.two_step,
        RAGMode.agentic,
    ]
    assert isinstance(result["two_step_rag"]["result"], RAGResponse)
    assert result["two_step_rag"]["result"].schema_version == "1.0"
    assert isinstance(result["agentic_rag"]["result"], RAGResponse)

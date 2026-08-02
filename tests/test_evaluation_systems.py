from __future__ import annotations

import pytest

from app.contracts import RAGMode, RAGResponse
from app.evaluation import systems


class FakeService:
    def __init__(self) -> None:
        self.requests = []

    def answer(self, request):
        self.requests.append(request)
        return RAGResponse(
            mode=request.mode,
            answer=f"Answer from {request.mode.value}.",
            run_id=f"run-{request.mode.value}",
        )


def test_build_systems_shares_one_service_and_dispatches_canonical_requests(
    monkeypatch,
) -> None:
    service = FakeService()
    build_calls = []

    def fake_build_default_rag_service(**kwargs):
        build_calls.append(kwargs)
        return service

    monkeypatch.setattr(
        systems,
        "build_default_rag_service",
        fake_build_default_rag_service,
    )

    adapters = systems.build_systems(
        modes=(RAGMode.two_step, RAGMode.graph),
        top_k=7,
        max_retries=2,
    )
    response = adapters[RAGMode.graph].answer("How does the graph work?")

    assert build_calls == [{"modes": (RAGMode.two_step, RAGMode.graph), "top_k": 7}]
    assert tuple(adapters) == (RAGMode.two_step, RAGMode.graph)
    assert response.mode is RAGMode.graph
    assert service.requests[0].query == "How does the graph work?"
    assert service.requests[0].top_k == 7
    assert service.requests[0].max_retries == 2


def test_build_systems_rejects_modes_outside_the_local_pdf_comparison() -> None:
    with pytest.raises(ValueError, match="research-assistant"):
        systems.build_systems(
            modes=(RAGMode.research_assistant,),
            top_k=4,
            service=FakeService(),
        )

import json

import pytest

from app import main as research_main
from app.contracts import RAGMode, RAGResponse
from app.graphs import agentic_rag_graph, multi_agent_graph


class FakeService:
    def __init__(self) -> None:
        self.requests = []

    def answer(self, request):
        self.requests.append(request)
        return RAGResponse(
            mode=request.mode,
            answer=f"Canonical {request.mode.value} answer.",
            run_id=f"run-{request.mode.value}",
        )


@pytest.mark.parametrize(
    ("entrypoint", "argv", "mode"),
    [
        (
            research_main,
            ["--query", "Research this.", "--json"],
            RAGMode.research_assistant,
        ),
        (
            agentic_rag_graph,
            ["--query", "Use one graph.", "--json"],
            RAGMode.graph,
        ),
        (
            multi_agent_graph,
            ["--query", "Use specialists.", "--json"],
            RAGMode.multi_agent,
        ),
    ],
)
def test_entrypoint_json_is_canonical(
    monkeypatch,
    capsys,
    entrypoint,
    argv: list[str],
    mode: RAGMode,
) -> None:
    service = FakeService()
    monkeypatch.setattr(
        entrypoint,
        "setup_phoenix_tracing",
        lambda: print("tracing configured"),
    )

    entrypoint.main(argv, service=service)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert "tracing configured" in captured.err
    assert payload["schema_version"] == "1.0"
    assert payload["mode"] == mode.value
    assert payload["run_id"] == f"run-{mode.value}"
    assert [request.mode for request in service.requests] == [mode]

from app.rag import compare


class FakeAgent:
    def __init__(self):
        self.invocations = []

    def invoke(self, payload):
        self.invocations.append(payload)
        return {"messages": [{"role": "assistant", "content": "agentic answer"}]}


def test_run_comparison_runs_both_rag_modes(monkeypatch):
    calls = []
    fake_agent = FakeAgent()

    def fake_build_two_step_rag(k: int):
        calls.append(("build_two_step", k))

        def answer(query: str):
            calls.append(("two_step_query", query))
            return {"mode": "two_step_rag", "answer": "baseline answer"}

        return answer

    def fake_build_agentic_rag(k: int):
        calls.append(("build_agentic", k))
        return fake_agent

    monkeypatch.setattr(compare, "build_two_step_rag", fake_build_two_step_rag)
    monkeypatch.setattr(compare, "build_agentic_rag", fake_build_agentic_rag)

    result = compare.run_comparison(query="Compare RAG modes", k=3)

    assert result["query"] == "Compare RAG modes"
    assert result["two_step_rag"]["result"] == {
        "mode": "two_step_rag",
        "answer": "baseline answer",
    }
    assert result["agentic_rag"]["result"] == {
        "messages": [{"role": "assistant", "content": "agentic answer"}],
    }
    assert isinstance(result["two_step_rag"]["latency_seconds"], float)
    assert isinstance(result["agentic_rag"]["latency_seconds"], float)
    assert calls == [
        ("build_two_step", 3),
        ("build_agentic", 3),
        ("two_step_query", "Compare RAG modes"),
    ]
    assert fake_agent.invocations == [
        {"messages": [{"role": "user", "content": "Compare RAG modes"}]}
    ]

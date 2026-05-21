import json

from app.rag import cli


def test_parse_args_supports_mode_and_k():
    args = cli.parse_args([
        "--query",
        "When should RAG retrieve?",
        "--mode",
        "agentic",
        "--k",
        "2",
    ])

    assert args.query == "When should RAG retrieve?"
    assert args.mode == "agentic"
    assert args.k == 2


def test_run_query_dispatches_two_step(monkeypatch):
    calls = {}

    def fake_two_step(query: str, k: int):
        calls["two_step"] = (query, k)
        return {"mode": "two_step_rag", "answer": "baseline"}

    monkeypatch.setattr(cli, "run_two_step", fake_two_step)

    result = cli.run_query(mode="two-step", query="What is RAG?", k=3)

    assert result == {"mode": "two_step_rag", "answer": "baseline"}
    assert calls == {"two_step": ("What is RAG?", 3)}


def test_run_query_dispatches_compare(monkeypatch):
    calls = {}

    def fake_run_comparison(query: str, k: int):
        calls["compare"] = (query, k)
        return {
            "query": query,
            "two_step_rag": {"result": {"mode": "two_step_rag"}},
            "agentic_rag": {"result": {"mode": "agentic_rag"}},
        }

    monkeypatch.setattr(cli, "run_comparison", fake_run_comparison)

    result = cli.run_query(mode="compare", query="Compare RAG modes", k=4)

    assert result == {
        "query": "Compare RAG modes",
        "two_step_rag": {"result": {"mode": "two_step_rag"}},
        "agentic_rag": {"result": {"mode": "agentic_rag"}},
    }
    assert calls == {"compare": ("Compare RAG modes", 4)}


def test_main_prints_selected_mode_json_when_requested(monkeypatch, capsys):
    def fake_run_query(mode: str, query: str, k: int):
        return {"mode": mode, "query": query, "k": k}

    monkeypatch.setattr(cli, "run_query", fake_run_query)

    cli.main([
        "--query",
        "Need evidence?",
        "--mode",
        "agentic",
        "--k",
        "5",
        "--json",
    ])

    output = json.loads(capsys.readouterr().out)
    assert output == {"mode": "agentic", "query": "Need evidence?", "k": 5}


def test_format_agentic_output_is_human_readable():
    result = {
        "mode": "agentic_rag",
        "result": {
            "messages": [
                {"type": "human", "content": "What is this project about?"},
                {
                    "type": "ai",
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "retrieve_documents",
                            "args": {"query": "project overview", "k": 4},
                        }
                    ],
                },
                {
                    "type": "tool",
                    "name": "retrieve_documents",
                    "content": json.dumps(
                        {
                            "query": "project overview",
                            "retrieved_count": 1,
                            "results": [
                                {
                                    "source": "docs/example.pdf",
                                    "chunk_id": "chunk-1",
                                    "page": 2,
                                    "content": "Raw retrieved text with JSON-only details.",
                                }
                            ],
                        }
                    ),
                },
                {
                    "type": "ai",
                    "content": "This project is about explainable agentic RAG.",
                    "tool_calls": [],
                },
            ]
        },
    }

    output = cli.format_output(result)

    assert "Agentic RAG" in output
    assert "This project is about explainable agentic RAG." in output
    assert "retrieve_documents(query='project overview', k=4)" in output
    assert "docs/example.pdf | chunk=chunk-1 | page=2" in output
    assert '"results"' not in output
    assert "Raw retrieved text with JSON-only details" not in output


def test_main_prints_human_readable_text_by_default(monkeypatch, capsys):
    def fake_run_query(mode: str, query: str, k: int):
        return {
            "mode": "two_step_rag",
            "answer": "A readable answer.",
            "sources": [],
        }

    monkeypatch.setattr(cli, "run_query", fake_run_query)

    cli.main(["--query", "Need evidence?", "--mode", "two-step", "--k", "5"])

    output = capsys.readouterr().out
    assert "2-Step RAG" in output
    assert "A readable answer." in output

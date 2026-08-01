import argparse
import json
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from typing import Any, Literal

from app.contracts import ProgressEvent, RAGMode, RAGRequest, RAGResponse
from app.observability import setup_phoenix_tracing
from app.rag.config import TOP_K
from app.rag.compare import run_comparison

RagMode = Literal["two-step", "agentic", "graph", "multi-agent", "compare"]
RAG_MODES: tuple[str, ...] = (
    "two-step",
    "agentic",
    "graph",
    "multi-agent",
    "compare",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a canonical RAG mode or compare two-step and agentic RAG."
    )
    parser.add_argument("--query", required=True, help="User question to answer")
    parser.add_argument(
        "--mode",
        choices=RAG_MODES,
        default="two-step",
        help="RAG mode to run, or compare two-step and agentic RAG.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=TOP_K,
        help="Number of chunks to retrieve per retrieval call.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the canonical versioned JSON response.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum graph retrieval/verification retries.",
    )
    parser.add_argument(
        "--thread-id",
        help="Stable thread identifier for checkpointed graph modes.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream canonical progress events before the final response.",
    )

    return parser.parse_args(argv)


def _to_jsonable(value: Any) -> Any:
    """Convert LangChain/Pydantic objects into JSON-serializable values."""
    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}

    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]

    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())

    if hasattr(value, "dict"):
        return _to_jsonable(value.dict())

    return str(value)


def extract_final_ai_answer(message: list[dict]) -> str:
    for message in reversed(message):
        if message.get("type") == "ai" and message.get("content", "").strip():
            return message["content"].strip()
    return "No final answer found."


def extract_tool_calls(messages: list[dict]):
    calls = []

    for message in messages:
        for tool_call in message.get("tool_calls", []) or []:
            calls.append(
                {
                    "name": tool_call.get("name"),
                    "args": tool_call.get("args", {}),
                }
            )

    return calls


def extract_retrieved_sources(messages: list[dict]) -> list[dict]:
    sources = []
    seen = set()

    for message in messages:
        if message.get("type") != "tool":
            continue

        try:
            payload = json.loads(message.get("content", "{}"))
        except json.JSONDecodeError:
            continue

        for item in payload.get("results", []):
            source = {
                "source": item.get("source"),
                "chunk_id": item.get("chunk_id"),
                "page": item.get("page"),
            }

            key = (source["source"], source["chunk_id"], source["page"])
            if key not in seen:
                seen.add(key)
                sources.append(source)

    return sources


def format_agentic_rag(agentic_result: dict) -> str:
    payload = agentic_result.get("result", agentic_result)
    messages = payload.get("messages", [])

    answer = extract_final_ai_answer(messages)
    tool_calls = extract_tool_calls(messages)
    sources = extract_retrieved_sources(messages)

    lines = []
    lines.append("Agentic RAG")
    lines.append("-" * 80)
    lines.append("Answer:")
    lines.append(answer)
    lines.append("")

    lines.append("Tool calls:")
    if tool_calls:
        for i, call in enumerate(tool_calls, start=1):
            args = ", ".join(f"{k}={v!r}" for k, v in call.get("args", {}).items())
            lines.append(f"  {i}. {call.get('name')}({args})")
    else:
        lines.append("  No tools called.")

    lines.append("")
    lines.append("Retrieved sources:")
    if sources:
        for i, source in enumerate(sources, start=1):
            lines.append(
                f"  {i}. {source['source']} "
                f"| chunk={source['chunk_id']} "
                f"| page={source['page']}"
            )
    else:
        lines.append("  No sources retrieved.")

    return "\n".join(lines)


def format_two_step_rag(result: dict) -> str:
    lines = []
    lines.append("2-Step RAG")
    lines.append("-" * 80)
    lines.append("Answer:")
    lines.append(result.get("answer", ""))
    lines.append("")
    lines.append("Sources:")
    sources = result.get("sources", [])
    if sources:
        for source in sources:
            lines.append(
                f"  - {source.get('source')} "
                f"| chunk={source.get('chunk_id')} "
                f"| page={source.get('page')}"
            )
    else:
        lines.append("  No sources retrieved.")

    return "\n".join(lines)


def format_output(result: dict) -> str:
    if isinstance(result, RAGResponse):
        return format_canonical_response(result)

    mode = result.get("mode")

    if (
        mode == "agentic_rag"
        or "result" in result
        and "messages" in result.get("result", {})
    ):
        return format_agentic_rag(result)

    if "two_step_rag" in result and "agentic_rag" in result:
        return format_compare_output(result)

    if mode == "two_step_rag":
        return format_two_step_rag(result)

    return json.dumps(_to_jsonable(result), indent=2, ensure_ascii=False)


def format_canonical_response(response: RAGResponse) -> str:
    """Render a canonical response without exposing transport-specific details."""

    titles = {
        RAGMode.two_step: "2-Step RAG",
        RAGMode.agentic: "Agentic RAG",
        RAGMode.graph: "RAG Graph",
        RAGMode.multi_agent: "Multi-Agent RAG",
        RAGMode.research_assistant: "Research Assistant",
    }
    lines = [titles[response.mode], "-" * 80, "Answer:", response.answer, ""]

    if response.route_history:
        lines.append("Route:")
        for route in response.route_history:
            lines.append(
                f"  {route.step}. {route.agent} -> {route.decision}: {route.reason}"
            )
        lines.append("")

    lines.append("Verification:")
    lines.append(f"  Status: {response.verification.status.value}")
    lines.append(f"  Score: {response.verification.score}")
    lines.append(f"  Retry count: {response.metrics.retry_count}")
    lines.append("")
    lines.append("Sources:")
    if response.evidence:
        for index, chunk in enumerate(response.evidence, start=1):
            lines.append(
                f"  {index}. {chunk.source} "
                f"| chunk={chunk.chunk_id} "
                f"| page={chunk.page}"
            )
            if chunk.reason_selected:
                lines.append(f"     Reason: {chunk.reason_selected}")
    else:
        lines.append("  No sources retrieved.")

    return "\n".join(lines)


def format_compare_output(result: dict) -> str:
    lines = []
    lines.append("RAG Comparison")
    lines.append("=" * 80)
    lines.append(f"Query: {result['query']}")
    lines.append("")

    two_step = result["two_step_rag"]
    agentic = result["agentic_rag"]

    lines.append("2-Step RAG")
    lines.append("-" * 80)
    lines.append(f"Latency: {two_step['latency_seconds']}s")
    lines.append("Answer:")
    lines.append(two_step["result"]["answer"])
    lines.append("")
    lines.append("Sources:")
    for source in two_step["result"].get("evidence", []):
        lines.append(
            f"  - {source['source']} "
            f"| chunk={source['chunk_id']} "
            f"| page={source['page']}"
        )

    lines.append("")
    agentic_response = agentic["result"]
    lines.append("Agentic RAG")
    lines.append("-" * 80)
    lines.append(f"Latency: {agentic['latency_seconds']}s")
    lines.append("Answer:")
    lines.append(agentic_response["answer"])
    lines.append("")
    lines.append("Retrieved sources:")
    for source in agentic_response.get("evidence", []):
        lines.append(
            f"  - {source['source']} "
            f"| chunk={source['chunk_id']} "
            f"| page={source['page']}"
        )

    return "\n".join(lines)


def run_two_step(query: str, k: int = TOP_K) -> RAGResponse:
    """Run deterministic retrieve-then-generate RAG through the service."""

    service = _build_service((RAGMode.two_step,), top_k=k)
    return service.answer(RAGRequest(query=query, mode=RAGMode.two_step, top_k=k))


def run_agentic(query: str, k: int = TOP_K) -> RAGResponse:
    """Run tool-using RAG through the service."""

    service = _build_service((RAGMode.agentic,), top_k=k)
    return service.answer(RAGRequest(query=query, mode=RAGMode.agentic, top_k=k))


def run_graph(
    query: str,
    k: int = TOP_K,
    *,
    max_retries: int = 2,
    thread_id: str | None = None,
) -> RAGResponse:
    service = _build_service((RAGMode.graph,), top_k=k)
    return service.answer(
        RAGRequest(
            query=query,
            mode=RAGMode.graph,
            top_k=k,
            max_retries=max_retries,
            thread_id=thread_id,
        )
    )


def run_multi_agent(
    query: str,
    k: int = TOP_K,
    *,
    max_retries: int = 2,
    thread_id: str | None = None,
) -> RAGResponse:
    service = _build_service((RAGMode.multi_agent,), top_k=k)
    return service.answer(
        RAGRequest(
            query=query,
            mode=RAGMode.multi_agent,
            top_k=k,
            max_retries=max_retries,
            thread_id=thread_id,
        )
    )


def run_query(
    mode: RagMode,
    query: str,
    k: int = TOP_K,
    *,
    max_retries: int = 2,
    thread_id: str | None = None,
    service: Any | None = None,
) -> RAGResponse | dict[str, Any]:
    if mode == "compare":
        if service is None:
            return run_comparison(query=query, k=k)
        return run_comparison(query=query, k=k, service=service)

    if service is not None:
        return service.answer(
            RAGRequest(
                query=query,
                mode=RAGMode(mode),
                top_k=k,
                max_retries=max_retries,
                thread_id=thread_id,
            )
        )

    if mode == "two-step":
        return run_two_step(query=query, k=k)

    if mode == "agentic":
        return run_agentic(query=query, k=k)

    if mode == "graph":
        return run_graph(
            query=query,
            k=k,
            max_retries=max_retries,
            thread_id=thread_id,
        )

    if mode == "multi-agent":
        return run_multi_agent(
            query=query,
            k=k,
            max_retries=max_retries,
            thread_id=thread_id,
        )

    raise ValueError(f"Unsupported RAG mode: {mode}")


def main(argv: Sequence[str] | None = None, *, service: Any | None = None) -> None:
    args = parse_args(argv)

    with redirect_stdout(sys.stderr):
        setup_phoenix_tracing()

    if args.stream:
        if args.mode == "compare":
            raise ValueError("compare mode does not support streaming")
        resolved_service = service or _build_service(
            (RAGMode(args.mode),),
            top_k=args.k,
        )
        request = RAGRequest(
            query=args.query,
            mode=RAGMode(args.mode),
            top_k=args.k,
            max_retries=args.max_retries,
            thread_id=args.thread_id,
            stream=True,
        )
        final_response: RAGResponse | None = None
        for item in resolved_service.stream(request):
            if isinstance(item, ProgressEvent):
                print(f"[{item.event}] {item.message}", file=sys.stderr)
            else:
                final_response = item
        if final_response is None:
            raise RuntimeError("RAG stream ended without a final response")
        result: RAGResponse | dict[str, Any] = final_response
    else:
        if service is None and args.max_retries == 2 and args.thread_id is None:
            result = run_query(mode=args.mode, query=args.query, k=args.k)
        else:
            result = run_query(
                mode=args.mode,
                query=args.query,
                k=args.k,
                max_retries=args.max_retries,
                thread_id=args.thread_id,
                service=service,
            )

    if args.json:
        if isinstance(result, RAGResponse):
            print(result.model_dump_json(indent=2))
        else:
            print(json.dumps(_to_jsonable(result), indent=2, ensure_ascii=False))
        return

    print(
        format_output(
            result if isinstance(result, RAGResponse) else _to_jsonable(result)
        )
    )


def _build_service(modes: Sequence[RAGMode], *, top_k: int):
    from app.bootstrap import build_default_rag_service

    return build_default_rag_service(modes=modes, top_k=top_k)


if __name__ == "__main__":
    main()

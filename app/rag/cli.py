import argparse
import json
from collections.abc import Sequence
from typing import Any, Literal

from app.rag.agentic_rag import build_agentic_rag
from app.rag.config import TOP_K
from app.rag.two_step_rag import build_two_step_rag
from app.rag.compare import run_comparison

RagMode = Literal["two-step", "agentic", "compare"]
RAG_MODES: tuple[str, ...] = ("two-step", "agentic", "compare")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline two-step RAG, agentic RAG, or compare both."
    )
    parser.add_argument(
        "--query", 
        required=True, 
        help="User question to answer"
    )
    parser.add_argument(
        "--mode",
        choices=RAG_MODES,
        default="two-step",
        help="RAG mode to run: fixed retrieve-then-answer, agentic tool use, or both.",
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
        help="Print raw JSON/LangChain trace output.",
        
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
            return message[content].strip()
    return "No final answer found."


def extract_tool_calls(message: list[dict]):
    calls = []
    
    for message in messages:
        for tool_call in message.get("tool_calls", []) or []:
            calls.append({
                "name": tool_call.get("name"),
                "args": tool_call.get("args", {}),
            })
            
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
            if key not in seen():
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
            args = ", ".join(
                f"{k}={v!r}" for k, v in call.get("args", {}).items()
            )
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
    for source in two_step["result"].get("sources", []):
        lines.append(
            f"  - {source['source']} "
            f"| chunk={source['chunk_id']} "
            f"| page={source['page']}"
        )
    
    lines.append("")
    lines.append(format_agentic_rag(agentic))
    
    return "\n".join(lines)


def run_two_step(query: str, k: int = TOP_K) -> dict[str, Any]:
    """Run deterministic retrieve-then-generate RAG."""
    rag_chain = build_two_step_rag(k=k)
    return rag_chain(query)


def run_agentic(query: str, k: int = TOP_K) -> dict[str, Any]:
    """Run agentic RAG where the LLM chooses whether to call retrieval."""
    agent = build_agentic_rag(k=k)
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        }
    )

    return {
        "mode": "agentic_rag",
        "result": _to_jsonable(result),
    }


def run_query(mode: RagMode, query: str, k: int = TOP_K) -> dict[str, Any]:
    if mode == "two-step":
        return run_two_step(query=query, k=k)

    if mode == "agentic":
        return run_agentic(query=query, k=k)

    if mode == "compare":
        # return {
        #     "question": query,  
        #     "two_step_rag": run_two_step(query=query, k=k),
        #     "agentic_rag": run_agentic(query=query, k=k),
        # }
        
        return run_comparison(query=query, k=k)

    raise ValueError(f"Unsupported RAG mode: {mode}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_query(mode=args.mode, query=args.query, k=args.k)
    
    if args.json:
        print(json.dumps(_to_jsonable(result), indent=2, ensure_ascii=False))
        return
    
    if args.mode == "compare":
        print(format_compare_output(_to_jsonable(result)))
    elif args.mode == "agentic":
        print(format_agentic_rag(_to_jsonable(result)))
    else:
        print(json.dumps(_to_jsonable(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

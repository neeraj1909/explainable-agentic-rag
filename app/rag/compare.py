import time
from typing import Any

from app.rag.agentic_rag import build_agentic_rag
from app.rag.config import TOP_K
from app.rag.two_step_rag import build_two_step_rag


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    return str(value)


def run_comparison(query: str, k: int = TOP_K) -> dict[str, Any]:
    two_step = build_two_step_rag(k=k)
    agentic = build_agentic_rag(k=k)

    start = time.perf_counter()
    two_step_result = two_step(query)
    two_step_latency = time.perf_counter() - start

    start = time.perf_counter()
    agentic_result = agentic.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        }
    )
    agentic_latency = time.perf_counter() - start

    return {
        "query": query,
        "two_step_rag": {
            "latency_seconds": round(two_step_latency, 3),
            "result": two_step_result,
        },
        "agentic_rag": {
            "latency_seconds": round(agentic_latency, 3),
            "result": _to_jsonable(agentic_result),
        },
    }

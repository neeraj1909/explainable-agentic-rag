import time
from typing import Any

from app.contracts import RAGMode, RAGRequest
from app.rag.config import TOP_K


def run_comparison(
    query: str,
    k: int = TOP_K,
    *,
    service: Any | None = None,
) -> dict[str, Any]:
    """Run both modes through one canonical service instance."""

    resolved_service = service or _build_comparison_service(k)

    start = time.perf_counter()
    two_step_result = resolved_service.answer(
        RAGRequest(query=query, mode=RAGMode.two_step, top_k=k)
    )
    two_step_latency = time.perf_counter() - start

    start = time.perf_counter()
    agentic_result = resolved_service.answer(
        RAGRequest(query=query, mode=RAGMode.agentic, top_k=k)
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
            "result": agentic_result,
        },
    }


def _build_comparison_service(k: int):
    from app.bootstrap import build_default_rag_service

    return build_default_rag_service(
        modes=(RAGMode.two_step, RAGMode.agentic),
        top_k=k,
    )

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

from app.config import get_llm_client
from app.contracts import ProgressEvent, RAGMode, RAGRequest, RAGResponse
from app.graphs.state import RagGraphState
from app.rag.config import TOP_K
from app.rag.prompts import rag_prompt
from app.rag.retriever import build_attributed_retriever
from app.rag.two_step_rag import format_context
from app.tools.verification_tools import calculate_faithfulness_stub
from app.observability import setup_phoenix_tracing


rewrite_prompt = ChatPromptTemplate.from_template(
    """
    Rewrite the user question into a better retrieval query.
    
    Original question:
    {question} 
    
    Previous query:
    {query}
    
    Reason retrieval was weak: 
    {reason}
    
    Return only the improved search query.
    """
)


@dataclass(slots=True)
class RagGraphNodes:
    """Injectable node dependencies for the single RAG graph."""

    retriever: Any
    answer_chain: Any
    rewrite_chain: Any
    verifier: Any = calculate_faithfulness_stub

    def classify_query(self, state: RagGraphState) -> dict[str, Any]:
        question = state["question"]
        lowered = question.lower()

        small_talk = lowered.strip() in {"hi", "hello", "hey", "thanks"}

        return {
            "query": question,
            "query_type": "small_talk" if small_talk else "document_question",
            "needs_retrieval": not small_talk,
            "retry_count": state.get("retry_count", 0),
            "max_retries": state.get("max_retries", 2),
        }

    def retrieve(self, state: RagGraphState) -> dict[str, Any]:
        query = state.get("query", state["question"])

        docs = self.retriever.invoke(query)
        context = format_context(docs)

        return {
            "docs": docs,
            "context": context,
        }

    def grade_relevance(self, state: RagGraphState) -> dict[str, Any]:
        docs = state.get("docs", [])

        if not docs:
            return {
                "is_relevant": False,
                "relevance_reason": "No documents retrieved.",
            }

        return {
            "is_relevant": True,
            "relevance_reason": f"Retrieved {len(docs)} documents.",
        }

    def rewrite_query(self, state: RagGraphState) -> dict[str, Any]:
        rewritten = self.rewrite_chain.invoke(
            {
                "question": state["question"],
                "query": state.get("query", state["question"]),
                "reason": state.get("relevance_reason", "Weak retrieval."),
            }
        )

        return {
            "query": rewritten.strip(),
            "retry_count": state.get("retry_count", 0) + 1,
        }

    def generate_answer(self, state: RagGraphState) -> dict[str, Any]:
        if not state.get("needs_retrieval", True):
            return {"answer": "Please ask a document-grounded question."}

        answer = self.answer_chain.invoke(
            {
                "question": state["question"],
                "context": state.get("context", ""),
            }
        )

        return {"answer": answer}

    def verify_claims(self, state: RagGraphState) -> dict[str, Any]:
        raw_result = self.verifier(
            answer=state.get("answer", ""),
            evidence=state.get("context", ""),
        )

        result = json.loads(raw_result)

        faithfulness_score = result.get("faithfulness_score", 0.0)
        unsupported_claims = result.get("unsupported_claims", [])

        verified = faithfulness_score >= 0.35 and not unsupported_claims

        return {
            "faithfulness_score": faithfulness_score,
            "unsupported_claims": unsupported_claims,
            "verified": verified,
        }

    def finalize(self, state: RagGraphState) -> dict[str, Any]:
        docs = state.get("docs", [])

        sources = [
            {
                "source": doc.metadata.get("source"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "page": doc.metadata.get("page"),
                "content": doc.page_content,
                "retriever_score": doc.metadata.get("retriever_score"),
                "retriever_rank": doc.metadata.get("retriever_rank"),
                "reranker_score": doc.metadata.get("reranker_score"),
                "selected_rank": doc.metadata.get("selected_rank"),
                "reason_selected": doc.metadata.get("reason_selected"),
            }
            for doc in docs
        ]

        final = {
            "answer": state.get("answer"),
            "sources": sources,
            "faithfulness_score": state.get("faithfulness_score"),
            "unsupported_claims": state.get("unsupported_claims", []),
            "verified": state.get("verified", False),
            "retry_count": state.get("retry_count", 0),
        }

        return {"final": final}

    def human_review(self, state: RagGraphState) -> dict[str, Any]:
        review_payload = {
            "question": state["question"],
            "query": state.get("query"),
            "answer": state.get("answer"),
            "faithfulness_score": state.get("faithfulness_score"),
            "unsupported_claims": state.get("unsupported_claims", []),
            "retry_count": state.get("retry_count", 0),
            "reason": "Faithfulness below threshold or unsupported claims found.",
        }

        human_response = interrupt(review_payload)

        return {
            "needs_human_review": True,
            "human_review_reason": review_payload["reason"],
            "human_feedback": human_response.get("feedback"),
            "human_approved": human_response.get("approved", False),
        }


def build_rag_graph_nodes(
    k: int = TOP_K,
    *,
    retriever: Any | None = None,
    llm: Any | None = None,
    answer_chain: Any | None = None,
    rewrite_chain: Any | None = None,
    verifier: Any = calculate_faithfulness_stub,
) -> RagGraphNodes:
    """Construct node collaborators; callers may inject every dependency."""

    resolved_retriever = (
        retriever if retriever is not None else build_attributed_retriever(k=k)
    )
    if answer_chain is None or rewrite_chain is None:
        resolved_llm = llm if llm is not None else get_llm_client()
        answer_chain = answer_chain or rag_prompt | resolved_llm | StrOutputParser()
        rewrite_chain = (
            rewrite_chain or rewrite_prompt | resolved_llm | StrOutputParser()
        )
    return RagGraphNodes(
        retriever=resolved_retriever,
        answer_chain=answer_chain,
        rewrite_chain=rewrite_chain,
        verifier=verifier,
    )


def build_rag_graph(
    k: int = TOP_K,
    *,
    nodes: RagGraphNodes | None = None,
    checkpointer: Any | None = None,
):
    resolved_nodes = nodes or build_rag_graph_nodes(k=k)

    def route_after_classification(
        state: RagGraphState,
    ) -> Literal["retrieve", "generate_answer"]:
        if state.get("needs_retrieval", True):
            return "retrieve"
        return "generate_answer"

    def route_after_relevance(
        state: RagGraphState,
    ) -> Literal["generate_answer", "rewrite_query"]:
        if state.get("is_relevant"):
            return "generate_answer"

        if state.get("retry_count", 0) < state.get("max_retries", 2):
            return "rewrite_query"

        return "generate_answer"

    def route_after_verification(
        state: RagGraphState,
    ) -> Literal["finalize", "rewrite_query", "human_review"]:
        if state.get("verified"):
            return "finalize"

        if state.get("retry_count", 0) < state.get("max_retries", 2):
            return "rewrite_query"

        return "human_review"

    graph = StateGraph(RagGraphState)

    graph.add_node("classify_query", resolved_nodes.classify_query)
    graph.add_node("retrieve", resolved_nodes.retrieve)
    graph.add_node("grade_relevance", resolved_nodes.grade_relevance)
    graph.add_node("rewrite_query", resolved_nodes.rewrite_query)
    graph.add_node("generate_answer", resolved_nodes.generate_answer)
    graph.add_node("verify_claims", resolved_nodes.verify_claims)
    graph.add_node("finalize", resolved_nodes.finalize)
    graph.add_node("human_review", resolved_nodes.human_review)

    graph.add_edge(START, "classify_query")
    graph.add_conditional_edges(
        "classify_query",
        route_after_classification,
        {
            "retrieve": "retrieve",
            "generate_answer": "generate_answer",
        },
    )

    graph.add_edge("retrieve", "grade_relevance")

    graph.add_conditional_edges(
        "grade_relevance",
        route_after_relevance,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
        },
    )

    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate_answer", "verify_claims")

    graph.add_conditional_edges(
        "verify_claims",
        route_after_verification,
        {
            "finalize": "finalize",
            "rewrite_query": "rewrite_query",
            "human_review": "human_review",
        },
    )

    graph.add_edge("human_review", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver()
    )


def resume_rag_graph(
    graph,
    approved: bool,
    feedback: str,
    thread_id: str = "default",
) -> dict[str, Any]:
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    resume_result = graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback,
            }
        ),
        config=config,
    )

    return resume_result["final"]


def stream_rag_graph_progress(
    question: str,
    k: int = TOP_K,
    thread_id: str = "default",
    max_retries: int = 2,
    *,
    service: Any | None = None,
):
    resolved_service = service or _build_graph_service(k=k)
    yield from resolved_service.stream(
        RAGRequest(
            query=question,
            mode=RAGMode.graph,
            top_k=k,
            max_retries=max_retries,
            thread_id=thread_id,
            stream=True,
        )
    )


def run_rag_graph(
    question: str,
    k: int = TOP_K,
    thread_id: str = "default",
    max_retries: int = 2,
    *,
    service: Any | None = None,
) -> RAGResponse:
    resolved_service = service or _build_graph_service(k=k)
    return resolved_service.answer(
        RAGRequest(
            query=question,
            mode=RAGMode.graph,
            top_k=k,
            max_retries=max_retries,
            thread_id=thread_id,
        )
    )


def format_rag_graph_output(result: RAGResponse | dict[str, Any]) -> str:
    if isinstance(result, RAGResponse):
        from app.rag.cli import format_canonical_response

        return format_canonical_response(result)

    lines: list[str] = []

    lines.append("RAG Graph")
    lines.append("-" * 80)
    lines.append("Answer:")
    lines.append(result.get("answer") or "")
    lines.append("")

    lines.append("Verification:")
    faithfulness_score = result.get("faithfulness_score")
    if isinstance(faithfulness_score, float):
        faithfulness_score = round(faithfulness_score, 3)
    lines.append(f"  Faithfulness score: {faithfulness_score}")
    lines.append(f"  Verified: {result.get('verified')}")
    lines.append(f"  Retry count: {result.get('retry_count')}")
    lines.append("")

    unsupported_claims = result.get("unsupported_claims", [])
    if unsupported_claims:
        lines.append("Unsupported claims:")
        for claim in unsupported_claims:
            lines.append(f"  - {claim}")
        lines.append("")

    lines.append("Sources:")
    sources = result.get("sources", [])
    if sources:
        for index, source in enumerate(sources, start=1):
            retriever_score = source.get("retriever_score")
            if isinstance(retriever_score, float):
                retriever_score = round(retriever_score, 4)

            reranker_score = source.get("reranker_score")
            if isinstance(reranker_score, float):
                reranker_score = round(reranker_score, 4)

            score_parts = [f"retriever_score={retriever_score}"]
            if reranker_score is not None:
                score_parts.append(f"reranker_score={reranker_score}")

            lines.append(
                f"  {index}. {source.get('source')} "
                f"| chunk={source.get('chunk_id')} "
                f"| page={source.get('page')} "
                f"| {', '.join(score_parts)}"
            )

            reason = source.get("reason_selected")
            if reason:
                lines.append(f"     Reason: {reason}")
    else:
        lines.append("  No sources retrieved.")

    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LangGraph RAG workflow.")
    parser.add_argument(
        "--query",
        default="What are the achievements of neeraj in area of AI and ML?",
        help="Question to answer from the indexed documents.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=TOP_K,
        help="Number of chunks to retrieve.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the canonical versioned JSON response.",
    )
    parser.add_argument(
        "--stream", action="store_true", help="Stream graph progress events."
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum retrieval/verification retries.",
    )
    parser.add_argument(
        "--thread-id",
        default="default",
        help="Stable graph checkpoint thread identifier.",
    )

    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    service: Any | None = None,
) -> None:
    args = parse_args(argv)
    with redirect_stdout(sys.stderr):
        setup_phoenix_tracing()

    if args.stream:
        final_result: RAGResponse | None = None
        for progress in stream_rag_graph_progress(
            question=args.query,
            k=args.k,
            thread_id=args.thread_id,
            max_retries=args.max_retries,
            service=service,
        ):
            if isinstance(progress, ProgressEvent):
                print(f"[{progress.event}] {progress.message}", file=sys.stderr)
            else:
                final_result = progress

        if final_result is not None:
            print()
            if args.json:
                print(final_result.model_dump_json(indent=2))
            else:
                print(format_rag_graph_output(final_result))

        return

    result = run_rag_graph(
        args.query,
        k=args.k,
        thread_id=args.thread_id,
        max_retries=args.max_retries,
        service=service,
    )

    if args.json:
        print(result.model_dump_json(indent=2))
        return

    print(format_rag_graph_output(result))


def _build_graph_service(*, k: int):
    from app.bootstrap import build_default_rag_service

    return build_default_rag_service(modes=(RAGMode.graph,), top_k=k)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any, Literal, NotRequired, TypedDict

from langchain_core.documents import Document 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph

from app.config import get_llm_client
from app.rag.config import TOP_K
from app.rag.prompts import rag_prompt
from app.rag.retriever import build_attributed_retriever
from app.rag.two_step_rag import format_context
from app.tools.verification_tools import calculate_faithfulness_stub
from app.observability import setup_phoenix_tracing


class RagGraphState(TypedDict):
    question: str
    query: NotRequired[str]
    
    query_type: NotRequired[str]
    needs_retrieval: NotRequired[bool]
    
    docs: NotRequired[list[Document]]
    context: NotRequired[str]
    
    is_relevant: NotRequired[bool]
    relevance_reason: NotRequired[str]
    
    answer: NotRequired[str]
    
    faithfulness_score: NotRequired[float]
    unsupported_claims: NotRequired[list[str]]
    verified: NotRequired[bool]
    
    retry_count: NotRequired[int]
    max_retries: NotRequired[int]
    
    final: NotRequired[dict[str, Any]]
    
    
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

def build_rag_graph(k: int = TOP_K):
    retriever = build_attributed_retriever(k=k)
    llm = get_llm_client()
    
    answer_chain = rag_prompt | llm | StrOutputParser()
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    
    def classify_query(state: RagGraphState) -> dict[str, Any]:
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
        
    def retrieve(state: RagGraphState) -> dict[str, Any]:
        query = state.get("query", state["question"])
        
        docs = retriever.invoke(query)
        context = format_context(docs)
        
        return {
            "docs": docs,
            "context": context,
        }
        
    def grade_relevance(state: RagGraphState) -> dict[str, Any]:
        docs = state.get("docs", [])
        
        if not docs:
            return {
                "is_relevant": False,
                "relevance_reason": "No documents retrieved.",
            }
            
        return {
            "is_relevant": True,
            "relevance_reason": f"Retrieved {len(docs)} documents."
        }
        
    def rewrite_query(state: RagGraphState) -> dict[str, Any]:
        rewritten = rewrite_chain.invoke(
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
        
    def generate_answer(state: RagGraphState) -> dict[str, Any]:
        if not state.get("needs_retrieval", True):
            return {
                "answer": "Please ask a document-grounded question."
            }
            
        answer = answer_chain.invoke(
            {
                "question": state["question"],
                "context": state.get("context", ""),
            }
        )
        
        return {"answer": answer}
    
    def verify_claims(state: RagGraphState) -> dict[str, Any]:
        raw_result = calculate_faithfulness_stub(
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
        
    def finalize(state: RagGraphState) -> dict[str, Any]:
        docs = state.get("docs", [])
        
        sources = [
            {
                "source": doc.metadata.get("source"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "page": doc.metadata.get("page"),
                "retriever_score": doc.metadata.get("retriever_score"),
                "reranker_score": doc.metadata.get("reranker_score"),
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
    
    def route_after_verfication(
        state: RagGraphState,
    ) -> Literal["finalize", "rewrite_query"]:
        if state.get("verified"):
            return "finalize"
        
        if state.get("retry_count", 0) < state.get("max_retries", 2):
            return "rewrite_query"
        
        return "finalize"
    
    
    graph = StateGraph(RagGraphState)
    
    graph.add_node("classify_query", classify_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_relevance", grade_relevance)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("verify_claims", verify_claims)
    graph.add_node("finalize", finalize) 
    
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
        route_after_verfication,
        {
            "finalize": "finalize",
            "rewrite_query": "rewrite_query",
        },
    )
        
    graph.add_edge("finalize", END)
    
    return graph.compile()


def run_rag_graph(question: str, k: int = TOP_K) -> dict[str, Any]:
    graph = build_rag_graph(k=k)
    
    result = graph.invoke(
        {
            "question": question,
            "retry_count": 0,
            "max_retries": 2,
        }
    )
    
    return result["final"]


def format_rag_graph_output(result: dict[str, Any]) -> str:
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
        help="Print the raw JSON result instead of human-readable output.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    setup_phoenix_tracing()
    result = run_rag_graph(args.query, k=args.k)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print(format_rag_graph_output(result))


if __name__ == "__main__":
    main()

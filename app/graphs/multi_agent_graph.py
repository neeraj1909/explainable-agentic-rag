from __future__ import annotations

import json
from typing import Any, Literal, NotRequired, TypedDict

from langchain_core.documents import Document 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate 
from langgraph.graph import StateGraph, START, END

from app.config import get_llm_client
from app.rag.config import TOP_K
from app.rag.prompts import rag_prompt
from app.rag.retriever import build_attributed_retriever
from app.rag.two_step_rag import format_context
from app.tools.verification_tools import calculate_faithfulness_stub
from app.observability import setup_phoenix_tracing


AgentName = Literal[
    "query_planner",
    "retriever_agent",
    "explainer_agent",
    "verifier_agent",
    "finalize",
]


class RouteStep(TypedDict, total=False):
    step: int
    agent: str
    decision: str
    reason: str 
    
    
class MultiAgentRAGState(TypedDict, total=False):
    question: str
    query: NotRequired[str]
    
    task_plan: NotRequired[list[str]]
    next_agent: NotRequired[AgentName]
    route_history: NotRequired[list[RouteStep]]
    orchestrator_decision_reason: NotRequired[str]
    
    docs: NotRequired[list[Document]]
    context: NotRequired[str]
    relevance_reason: NotRequired[str]
    
    answer: NotRequired[str]
    explanation: NotRequired[str]
    
    faithfulness_score: NotRequired[float]
    unsupported_claims: NotRequired[list[str]]
    needs_verification: NotRequired[bool]
    verified: NotRequired[bool]
    verification_method: NotRequired[str]
    verification_verdict: NotRequired[str]
    
    retry_count: NotRequired[int]
    max_retries: NotRequired[int]
    
    needs_human_review: NotRequired[bool]
    human_review_reason: NotRequired[str]
    
    final: NotRequired[dict[str, Any]]
    

query_planner_prompt = ChatPromptTemplate.from_template(
    """
    You are a retrieval query planning agent.
    
    Your job:
    - Create or improve a search query for document retrieval.
    - Do not answer the user's question.
    - Return only the search query.
    
    User question:
    {question}
    
    Previous query:
    {query}
    
    Retry count:
    {retry_count}
    
    Reason the previous attempt was weak:
    {reason}
    
    Unsupported claims from previous answer:
    {unsupported_claims}
    
    Improved retrieval query:
    """
)


explainer_prompt = ChatPromptTemplate.from_template(
    """
    You are an answer/ explainer agent in a multi-agent RAG system.
    
    Your job:
    - Answer the user's question using only the provided context.
    - Cite source and chunk_id metadata when possible.
    - If the answer is not supported by the context, say:
      "I don't know based on the provided documents."
    - Do not invent facts.
    
    Question:
    {question}
    
    Context:
    {context}
    
    Answer: 
    """
)


def _append_route(
    state: MultiAgentRAGState,
    agent: str,
    decision: str,
    reason: str,
) -> list[RouteStep]:
    route_history = list(state.get("route_history", []))
    
    route_history.append(
        {
            "step": len(route_history) + 1,
            "agent": agent,
            "decision": decision,
            "reason": reason, 
        }
    )
    
    return route_history


class OrchestratorAgent:
    """
    Central multi-agent orchestrator.
    
    It does not retrieve, answer, or verify directly.
    It only decides which specialist agent should run next.
    """
    
    def __call__(self, state: MultiAgentRAGState) -> dict[str, Any]:
        task_plan = state.get(
            "task_plan",
            [
                "Plan retrieval query",
                "Retrieval evidence",
                "Generate grounded answer",
                "Verify answer against evidence",
                "Finalize auditable response"
            ],
        )
        
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 2)
        
        if not state.get("query"):
            next_agent: AgentName = "query_planner"
            reason = "No retrieval query exists yet."
            
        elif not state.get("docs"):
            next_agent: AgentName = "retriever_agent"
            reason = "A query exists, but evidence has not been retrieved yet."
            
        elif not state.get("answer"):
            next_agent: AgentName = "explainer_agent"
            reason = "Evidence is available, but no answer has been generated yet"
            
        elif state.get("needs_verification", False):
            next_agent = "verifier_agent"
            reason = "An answer exists, but it has not been verified yet."
            
        elif state.get("verified"):
            next_agent: AgentName = "finalize"
            reason = "Answer is verified and ready to finalize."
            
        elif retry_count < max_retries:
            next_agent: AgentName = "query_planner"
            reason = (
                "Answer was not verified, but retry budget remains. "
                "Improve the retrieval query and try again."
            )
            
        else:
            next_agent: AgentName = "finalize"
            reason = (
                "Answer was not verified and retry budget is exhausted. "
                "Finalize with verification failure details."
            )
            
        return {
            "task_plan": task_plan,
            "next_agent": next_agent,
            "orchestrator_decision_reason": reason,
            "route_history": _append_route(
                state=state,
                agent="orchestrator",
                decision=next_agent,
                reason=reason, 
            ),
        }
        

class QueryPlannerAgent:
    """
    Agent responsible for creating the initial retrieval query and improving it
    after failed verification.
    """
    
    def __init__(self, query_chain):
        self.query_chain = query_chain
        
    def __call__(self, state: MultiAgentRAGState) -> dict[str, Any]:
        current_query = state.get("query") or state["question"]
        retry_count = state.get("retry_count", 0)
        
        should_rewrite = bool(state.get("answer")) and not state.get("verified", False)
        
        if should_rewrite:
            reason = state.get(
                "orchestration_decision_reason",
                "Previous answer failed verification.",
            )
            
            rewritten_query = self.query_chain.invoke(
                {
                    "question": state["question"],
                    "query": current_query,
                    "retry_count": retry_count,
                    "reason": reason,
                    "unsupported_claims": state.get("unsupported_claims", []),
                }
            ).strip()
            
            next_retry_count = retry_count + 1
            query = rewritten_query or current_query
            
        else:
            query = current_query
            next_retry_count = retry_count
            
        return {
            "query": query,
            "retry_count": next_retry_count,
            "docs": [],
            "context": "",
            "answer": "",
            "explanation": "",
            "needs_verification": False,
            "route_history": _append_route(
                state=state,
                agent="query_planner",
                decision="query_ready",
                reason=(
                    "Improved query after failed verification."
                    if should_rewrite
                    else "Prepared initial retrieval query."
                ), 
            ),
        }


class RetrieverAgent:
    """
    Agent responsible only for retrieving evidence from the existing retriever.
    """
    def __init__(self, retriever):
        self.retriever = retriever
        
    def __call__(self, state: MultiAgentRAGState) -> dict[str, Any]:
        query = state.get("query") or state["question"]
        docs = self.retriever.invoke(query)
        
        relevance_reason = (
            f"Retrieved {len(docs)} document chunks."
            if docs
            else "No document chunks retrieved."
        )
        
        return {
            "query": query,
            "docs": docs,
            "context": format_context(docs),
            "relevance_reason": relevance_reason,
            "route_history": _append_route(
                state=state,
                agent="retriever_agent",
                decision="evidence_retrieved",
                reason=relevance_reason, 
            ),
        }
        

class ExplainerAgent:
    """
    Agent responsible only for producing an answer from verified/ retrieved context.
    """
    def __init__(self, answer_chain):
        self.answer_chain = answer_chain
        
    def __call__(self, state: MultiAgentRAGState) -> dict[str, Any]:
        answer = self.answer_chain.invoke(
            {
                "question": state["question"],
                "context": state.get("context", ""),
            }
        )
        
        return {
            "answer": answer,
            "explanation": (
                "The answer was generated by the explainer agent from retrieved "
                "document chunks using a grounded RAG prompt."
            ),
            "needs_verification": True,
            "route_history": _append_route(
                state=state,
                agent="explainer_agent",
                decision="answer_generated",
                reason="Generated a context-grounded draft answer.",
            ),
        }
        

class VerifierAgent:
    """
    Agent responsible only for checking whether the answer is supported by context.
    """
    
    def __init__(self, faithfulness_threshold: float = 0.50):
        self.faithfulness_threshold = faithfulness_threshold
        
    def __call__(self, state: MultiAgentRAGState) -> dict[str, Any]:
        raw_result = calculate_faithfulness_stub(
            answer=state.get("answer", ""),
            evidence=state.get("context", ""),
            threshold=self.faithfulness_threshold,
        )
        
        payload = json.loads(raw_result)
        
        faithfulness_score = payload.get("faithfulness_score", 0.0)
        unsupported_claims = payload.get("unsupported_claims", [])
        
        verified = (
            faithfulness_score >= self.faithfulness_threshold 
            and not unsupported_claims
        )
        
        reason = (
            "Answer is sufficiently supported by retrieved evidence."
            if verified
            else "Answer has unsupported claims or insufficient faithfulness."
        )
        
        return {
            "faithfulness_score": faithfulness_score,
            "unsupported_claims": unsupported_claims,
            "verified": verified,
            "needs_verification": False,
            "verification_method": payload.get("method"),
            "verification_verdict": payload.get("verdict"),
            "route_history": _append_route(
                state=state,
                agent="verifier_agent",
                decision="verified" if verified else "not_verified",
                reason=reason,
            ),
        }


def route_from_orchestrator(state: MultiAgentRAGState) -> AgentName:
    return state.get("next_agent", "finalize")


def finalize(state: MultiAgentRAGState) -> dict[str, Any]:
    docs = state.get("docs", [])
    
    sources = [
        {
            "source": doc.metadata.get("source"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "page": doc.metadata.get("page"),
            "retriever_score": doc.metadata.get("retriever_score"),
            "reranker_score": doc.metadata.get("reranker_score"),
            "selected_rank": doc.metadata.get("selected_rank"),
            "reason_selected": doc.metadata.get("reason_selected"),
        }
        for doc in docs
    ]
    
    verified = state.get("verified", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    
    needs_human_review = not verified and retry_count >= max_retries
    
    final = {
        "answer": state.get("answer"),
        "explanation": state.get("explanation"),
        "sources": sources,
        "faithfulness_score": state.get("faithfulness_score"),
        "unsupported_claims": state.get("unsupported_claims", []),
        "verified": state.get("verified", False),
        "verification_method": state.get("verification_method"),
        "verification_verdict": state.get("verification_verdict"),
        "retry_count": retry_count,
        "max_retries": max_retries,
        "needs_human_review": needs_human_review,
        "human_review_reason": (
            "verification failed after retry budget was exhausted."
            if needs_human_review
            else None 
        ),
        "task_plan": state.get("task_plan", []),
        "route_history": _append_route(
            state=state,
            agent="finalize",
            decision="final_response_ready",
            reason="Final response payload assembled."
        ),
    }
    
    return {"final": final}
    
    
def build_multi_agent_rag_graph(k: int = TOP_K):
    """
    Build an orchestrator-led multi-agent RAG workflow using existing project 
    components.
    
    Agents:
    - OrchestratorAgent: decides next specialist.
    - QueryPlannerAgent: prepares or rewrites retrieval query.
    - RetrieverAgent: retrieves attributed evidence.
    - ExplainerAgent: generates grounded answer.
    - VerifierAgent: checks answer faithfulness.
    - Finalizer: returns auditable final payload.
    """
    
    llm = get_llm_client()
    retriever = build_attributed_retriever(k=k)
    
    query_chain = query_planner_prompt | llm | StrOutputParser()
    answer_chain = explainer_prompt | llm | StrOutputParser()
    
    orchestrator = OrchestratorAgent()
    query_planner = QueryPlannerAgent(query_chain)    
    retriever_agent = RetrieverAgent(retriever)
    explainer_agent = ExplainerAgent(answer_chain)
    verifier_agent = VerifierAgent()
        
    graph = StateGraph(MultiAgentRAGState)
    
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("query_planner", query_planner) 
    graph.add_node("retriever_agent", retriever_agent)
    graph.add_node("explainer_agent", explainer_agent)
    graph.add_node("verifier_agent", verifier_agent) 
    graph.add_node("finalize", finalize)
    
    graph.add_edge(START, "orchestrator")
    
    graph.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "query_planner": "query_planner",
            "retriever_agent": "retriever_agent",
            "explainer_agent": "explainer_agent",
            "verifier_agent": "verifier_agent",
            "finalize": "finalize", 
        }
    )

    graph.add_edge("query_planner", "orchestrator")
    graph.add_edge("retriever_agent", "orchestrator")
    graph.add_edge("explainer_agent", "orchestrator")
    graph.add_edge("verifier_agent", "orchestrator")
    graph.add_edge("finalize", END)
    
    return graph.compile()


def run_multi_agent_rag_graph(
    question: str,
    k: int = TOP_K,
    max_retries: int = 2,
) -> dict[str, Any]:
    graph = build_multi_agent_rag_graph(k=k)
    
    result = graph.invoke(
        {
            "question": question,
            "query": "",
            "retry_count": 0,
            "max_retries": max_retries,
            "route_history": [],
        }
    )
    
    return result["final"]


if __name__ == "__main__":
    setup_phoenix_tracing()
    result = run_multi_agent_rag_graph(
        question="What are the capabilities of Neeraj in AI and ML?",
        k=5,
    )
    
    print(result["answer"])
    print(json.dumps(result["route_history"], indent=2))
    print(result["sources"])
    
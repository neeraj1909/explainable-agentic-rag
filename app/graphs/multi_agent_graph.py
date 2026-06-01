from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict
from pydantic import BaseModel, Field 

from langchain_core.documents import Document 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver, PersistentDict
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

CHECKPOINT_DIR = Path(".langgraph_checkpoints")
CHECKPOINT_FILE = CHECKPOINT_DIR / "multi_agent_graph.pkl"

def build_persistent_checkpointer(
    checkpoint_file: Path = CHECKPOINT_FILE,
) -> InMemorySaver:
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    
    return InMemorySaver(
        factory=partial(
            PersistentDict,
            filename=str(checkpoint_file),
        )
    )


class StreamEvent(BaseModel):
    event: str
    agent: str | None = None
    decision: str = None 
    reason: str | None = None
    step: int | None = None
    retry_count: int | None = None
    verified: bool | None = None
    faithfulness_score: float | None = None
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    

class RouteStepModel(BaseModel):
    step: int
    agent: str
    decision: str
    reason: str 
    called_by: str | None = None 
    
    
class OrchestratorOutput(BaseModel):
    task_plan: list[str] = Field(default_factory=list)
    next_agent: AgentName
    orchestrator_decision_reason: str
    route_history: list[dict[str, Any]] = Field(default_factory=list)
    
    
class QueryPlannerOutput(BaseModel):
    query: str
    retry_count: int
    docs: list[Any] = Field(default_factory=list)
    context: str = ""
    answer: str = ""
    explanation: str = ""
    needs_verification: bool = False
    route_history: list[dict[str, Any]] = Field(default_factory=list)
    
    
class RetrieverOutput(BaseModel):
    query: str
    context: str
    relevance_reason: str
    route_history: list[dict[str, Any]] = Field(default_factory=list)
    
    
class ExplainerOutput(BaseModel):
    answer: str
    explanation: str 
    needs_verification: bool = True
    route_history: list[dict[str, Any]] = Field(default_factory=list)
    
    
class VerifierOutput(BaseModel):
    faithfulness_score: float 
    unsupported_claims: list[str] = Field(default_factory=list)
    verified: bool
    needs_verification: bool = False
    verification_method: str | None = None
    verification_verdict: str | None = None
    route_history: list[dict[str, Any]] = Field(default_factory=list)
    
    
class FinalOutput(BaseModel):
    answer: str | None = None
    explanation: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    
    faithfulness_score: float 
    unsupported_claims: list[str] = Field(default_factory=list)
    verified: bool = False
    verification_method: str | None = None
    verification_verdict: str | None = None
    
    retry_count: int = 0
    max_retries: int = 2
    needs_human_review: bool = False 
    human_review_reason: str | None = None 
    
    task_plan: list[str] = Field(default_factory=list)
    route_history: list[dict[str, Any]] = Field(default_factory=list)

    
class MultiAgentRAGState(TypedDict, total=False):
    question: str
    query: NotRequired[str]
    
    task_plan: NotRequired[list[str]]
    next_agent: NotRequired[AgentName]
    route_history: NotRequired[list[dict[str, Any]]]
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
    called_by: str | None = "orchestrator",
) -> list[RouteStep]:
    route_history = list(state.get("route_history", []))
    
    route_step = RouteStepModel(
        step=len(route_history) + 1,
        agent=agent,
        decision=decision,
        reason=reason,
        called_by=called_by,
    )
    
    route_history.append(route_step.model_dump(exclude_none=True))
    
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
            
        output = OrchestratorOutput(
            task_plan=task_plan,
            next_agent=next_agent,
            orchestrator_decision_reason=reason,
            route_history=_append_route(
                state=state,
                agent="orchestrator",
                decision=next_agent,
                reason=reason, 
                called_by=None,
            ),
        )
        
        return output.model_dump() 
        

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
            
        output = QueryPlannerOutput(
            query=query,
            retry_count=next_retry_count,
            docs=[],
            context="",
            answer="",
            explanation="",
            needs_verification=False,
            route_history=_append_route(
                state=state,
                agent="query_planner",
                decision="query_ready",
                reason=(
                    "Improved query after failed verification."
                    if should_rewrite
                    else "Prepared initial retrieval query."
                ), 
            ),
        )
        
        return output.model_dump()


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
        
        output = RetrieverOutput(
            query=query,
            docs=[],
            context=format_context(docs),
            relevance_reason=relevance_reason,
            route_history=_append_route(
                state=state,
                agent="retriever_agent",
                decision="evidence_retrieved",
                reason=relevance_reason, 
            ),
        )
        
        payload = output.model_dump(exclude={"docs"})
        payload["docs"] = docs 
        
        return payload
    

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
        
        output = ExplainerOutput(
            answer=answer,
            explanation=(
                "The answer was generated by the explainer agent from retrieved "
                "document chunks using a grounded RAG prompt."
            ),
            needs_verification=True,
            route_history=_append_route(
                state=state,
                agent="explainer_agent",
                decision="answer_generated",
                reason="Generated a context-grounded draft answer.",
            ),
        )
        
        return output.model_dump()
               

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
        
        output = VerifierOutput(
            faithfulness_score=faithfulness_score,
            unsupported_claims=unsupported_claims,
            verified=verified,
            needs_verification=False,
            verification_method=payload.get("method"),
            verification_verdict=payload.get("verdict"),
            route_history=_append_route(
                state=state,
                agent="verifier_agent",
                decision="verified" if verified else "not_verified",
                reason=reason,
            ),
        )
        
        return output.model_dump() 


def route_from_orchestrator(state: MultiAgentRAGState) -> AgentName:
    return state.get("next_agent", "finalize")


def _latest_route_from_update(update: dict[str, Any]) -> dict[str, Any] | None:
    for node_payload in update.values():
        if not isinstance(node_payload, dict):
            continue
        
        route_history = node_payload.get("route_history", [])
        
        if route_history:
            return route_history[-1]
        
    return None 


def _stream_event_from_update(update: dict[str, Any]) -> StreamEvent:
    node_name = next(iter(update.keys()))
    node_payload = update[node_name]
    
    if not isinstance(node_payload, dict):
        return StreamEvent(
            event="node_update",
            agent=node_name,
            message=f"{node_name} emitted an update.",
            payload={"raw": str(node_payload)}, 
        )
        
    latest_route = _latest_route_from_update(update)
    
    if latest_route:
        return StreamEvent(
            event="agent_step",
            agent=latest_route.get("agent", node_name),
            decision=latest_route.get("decision"),
            reason=latest_route.get("reason"),
            step=latest_route.get("step"),
            retry_count=latest_route.get("retry_count"),
            verified=latest_route.get("verified"),
            faithfulness_score=latest_route.get("faithfulness_score"),
            message=(
                f"{latest_route.get('agent', node_name)} -> "
                f"{latest_route.get('decision')}: "
                f"{latest_route.get('reason')}"
            ),
            payload=node_payload,
        )
        
    return StreamEvent(
        event="node_update",
        agent=node_name,
        retry_count=node_payload.get("retry_count"),
        verified=node_payload.get("verified"),
        faithfulness_score=node_payload.get("faithfulness_score"),
        message=f"{node_name} completed.",
        payload=node_payload,
    )


def _doc_metadata(doc: Document | dict[str, Any]) -> dict[str, Any]:
        if isinstance(doc, dict):
            return doc.get("metadata", {})
        return doc.metadata
    

def finalize(state: MultiAgentRAGState) -> dict[str, Any]:
    docs = state.get("docs", [])

    sources = []
    
    for doc in docs:
        metadata = _doc_metadata(doc)
        
        sources.append(
            {
                "source": metadata.get("source"),
                "chunk_id": metadata.get("chunk_id"),
                "page": metadata.get("page"),
                "retriever_score": metadata.get("retriever_score"),
                "reranker_score": metadata.get("reranker_score"),
                "selected_rank": metadata.get("selected_rank"),
                "reason_selected": metadata.get("reason_selected"),
            }
        )
    
    verified = state.get("verified", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    
    needs_human_review = not verified and retry_count >= max_retries
    
    final_output = FinalOutput(
        answer=state.get("answer"),
        explanation=state.get("explanation"),
        sources=sources,
        faithfulness_score=state.get("faithfulness_score"),
        unsupported_claims=state.get("unsupported_claims", []),
        verified=state.get("verified", False),
        verification_method=state.get("verification_method"),
        verification_verdict=state.get("verification_verdict"),
        retry_count=retry_count,
        max_retries=max_retries,
        needs_human_review=needs_human_review,
        human_review_reason=(
            "verification failed after retry budget was exhausted."
            if needs_human_review
            else None 
        ),
        task_plan=state.get("task_plan", []),
        route_history=_append_route(
            state=state,
            agent="finalize",
            decision="final_response_ready",
            reason="Final response payload assembled."
        ),
    )
    
    return {"final": final_output.model_dump()}
    
    
def build_multi_agent_rag_graph(
    k: int = TOP_K,
    checkpoint_file: Path = CHECKPOINT_FILE,
):
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
    
    checkpointer = build_persistent_checkpointer(checkpoint_file)
    
    return graph.compile(checkpointer=checkpointer)


def stream_multi_agent_rag_graph_event(
    question: str,
    k: int = TOP_K,
    max_retries: int = 2,
    thread_id: str = "default",
    checkpoint_file: Path = CHECKPOINT_FILE,
) -> dict[str, Any]: 
    graph = build_multi_agent_rag_graph(
        k=k,
        checkpoint_file=checkpoint_file, 
    )
    
    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id, 
        }
    }
    
    inputs = {
        "question": question,
        "query": "",
        "retry_count": 0,
        "max_retries": max_retries,
        "route_history": [],
    }
    
    yield StreamEvent(
        event="graph_start",
        agent="graph",
        message="Starting multi-agent RAG graph.",
        payload={
            "question": question,
            "thread_id": thread_id,
            "max_retries": max_retries,
        },
    ).model_dump()
    
    final_result: dict[str, Any] | None = None
    
    for update in graph.stream(
        inputs,
        config=config,
        stream_mode="updates",
    ):
        event = _stream_event_from_update(update)
        
        if "finalize" in update:
            final_result = update["finalize"].get("final")
            
        yield event.model_dump()
        
    graph.checkpointer.storage.sync()
    graph.checkpointer.writes.sync()
    graph.checkpointer.blobs.sync()
    
    yield StreamEvent(
        event="graph_end",
        agent="graph",
        message="Multi-agent RAG graph completed.",
        payload={
            "final": final_result,
            "thread_id": thread_id,
        },
    ).model_dump()
    

def print_multi_agent_rag_stream(
    question: str,
    k: int = TOP_K,
    max_retries: int = 2,
    thread_id: str = "default",
) -> dict[str, Any] | None:
    final_result = None
    
    for event in stream_multi_agent_rag_graph_event(
        question=question,
        k=k,
        max_retries=max_retries,
        thread_id=thread_id,
    ):
        print(f"[{event['event']}] {event['message']}")
        
        if event["event"] == "graph_end":
            final_result = event["payload"].get("final")
            
    return final_result


if __name__ == "__main__":
    setup_phoenix_tracing()
    
    final = print_multi_agent_rag_stream(
        question="What are the capabilities of Neeraj in AI and ML?",
        k=5,
        thread_id="multi-agent-stream-demo",
    )
    
    if final:
        print("\nFinal answer:")
        print(final["answer"])
        
        print("\nSources:")
        print(json.dumps(final["sources"], indent=2))
    
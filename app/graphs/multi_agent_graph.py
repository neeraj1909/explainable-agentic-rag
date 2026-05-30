from __future__ import annotations

import json
from typing import Any, Literal, NotRequired, TypedDict

from langchain_core.documents import Document 
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

from app.config import get_llm_client
from app.rag.config import TOP_K
from app.rag.prompts import rag_prompt
from app.rag.retriever import build_attributed_retriever
from app.rag.two_step_rag import format_context
from app.tools.verification_tools import calculate_faithfulness_stub
from app.observability import setup_phoenix_tracing


class MultiAgentRAGState(TypedDict, total=False):
    question: str
    query: NotRequired[str]
    
    docs: NotRequired[list[Document]]
    context: NotRequired[str]
    
    answer: NotRequired[str]
    explanation: NotRequired[str]
    
    faithfulness_score: NotRequired[float]
    unsupported_claims: NotRequired[list[str]]
    verified: NotRequired[bool]
    
    retry_count: NotRequired[int]
    max_retries: NotRequired[int]
    
    final: NotRequired[dict[str, Any]]
    

class RetrieverAgent:
    """
    Agent responsible only for retrieving evidence from the existing retriever.
    """
    def __init__(self, retriever):
        self.retriever = retriever
        
    def __call__(self, state: MultiAgentRAGState) -> dict[str, Any]:
        query = state.get("query") or state["question"]
        docs = self.retriever.invoke(query)
        
        return {
            "query": query,
            "docs": docs,
            "context": format_context(docs),
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
        
        # prompt = f"""
        #     You are an explainer agent.
        #     Answer the user's question using the verified context.
            
        #     Question:
        #     {question}
            
        #     Verified context:
        #     {verified_documents}
            
        #     Return:
        #     1. A clear answer
        #     2. A short explanation of which evidence supports it
        # """
        
        return {
            "answer": answer,
            "explanation": (
                "The answer was generated from retrieved document chunks "
                "using the existing RAG prompt."
            ),
        }
        

class VerifierAgent:
    """
    Agent responsible only for checking whether the answer is supported by context.
    """
    
    # def __init__(self, llm):
    #     self.llm = llm
        
    def __call__(self, state: MultiAgentRAGState) -> dict[str, Any]:
        raw_result = calculate_faithfulness_stub(
            answer=state.get("answer", ""),
            evidence=state.get("context", ""),
        )
        
        payload = json.loads(raw_result)
        
        faithfulness_score = payload.get("faithfulness_score", 0.0)
        unsupported_claims = payload.get("unsupported_claims", [])
        
        verified = faithfulness_score >= 0.50 and not unsupported_claims
        
        return {
            # **state,
            "faithfulness_score": faithfulness_score,
            "unsupported_claims": unsupported_claims,
            "verified": verified,
        }
        
      
class Supervisor:
    """
    Supervisor decides whether to retry retrieval or finish.
    It does not generate answers or verify claims itself.
    """
    
    def __call__(
        self, 
        state: MultiAgentRAGState
    ) -> Literal["retry", "finish"]:
        verified = state.get("verified", False)
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 2)
        
        if verified:
            return "finish"
        
        if retry_count < max_retries:
            return "retry"
        
        return "finish"
        

def prepare_retry(state: MultiAgentRAGState) -> dict[str, Any]:
    """
    Keep retry logic small and explainable.
    
    This does not rewrite the query yet.
    It simply retries the same question with the same existing retriever.
    """
    
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "query": state.get("query") or state["question"],
    }


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
    
    return {
        "final": {
            "answer": state.get("answer"),
            "explanation": state.get("explanation"),
            "sources": sources,
            "faithfulness_score": state.get("faithfulness_score"),
            "unsupported_claims": state.get("unsupported_claims", []),
            "verified": state.get("verified", False),
            "retry_count": state.get("retry_count", 0),
        }
    }
    
    
def build_multi_agent_rag_graph(k: int = TOP_K):
    """
    Build a small nulti-agent RAG workflow using existing project components.
    
    Existing components reused:
    - get_llm_client()
    - build_attributed_retriever()
    - rag_prompt
    - format_context()
    - calculate_faithfulness_stub() 
    """
    llm = get_llm_client()
    retriever = build_attributed_retriever(k=k)
    answer_chain = rag_prompt | llm | StrOutputParser()
    
    retriever_agent = RetrieverAgent(retriever)
    explainer_agent = ExplainerAgent(answer_chain)
    verifier_agent = VerifierAgent()
    supervisor = Supervisor()
    
    graph = StateGraph(MultiAgentRAGState)
    
    graph.add_node("retriever_agent", retriever_agent) 
    graph.add_node("verifier_agent", verifier_agent) 
    graph.add_node("explainer_agent", explainer_agent)
    graph.add_node("prepare_retry", prepare_retry)
    graph.add_node("finalize", finalize)
    
    graph.add_edge(START, "retriever_agent")
    graph.add_edge("retriever_agent", "explainer_agent")
    graph.add_edge("explainer_agent", "verifier_agent")
    graph.add_conditional_edges(
        "verifier_agent",
        supervisor,
        {
            "retry": "prepare_retry",
            "finish": "finalize",
        },
    )
    
    graph.add_edge("prepare_retry", "retriever_agent")
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
            "query": question,
            "retry_count": 0,
            "max_retries": max_retries,
        }
    )
    
    return result["final"]


if __name__ == "__main__":
    result = run_multi_agent_rag_graph(
        question="What are the capabilities of Neeraj in AI and ML?",
        k=5,
    )
    
    print(result["answer"])
    print(result["sources"])
    
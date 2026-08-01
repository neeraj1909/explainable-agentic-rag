"""Typed internal state contracts shared by the two RAG graphs."""

from typing import Any, Literal, NotRequired, TypedDict

from langchain_core.documents import Document


AgentName = Literal[
    "query_planner",
    "retriever_agent",
    "explainer_agent",
    "verifier_agent",
    "finalize",
]


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
    needs_human_review: NotRequired[bool]
    human_review_reason: NotRequired[str]
    human_feedback: NotRequired[str]
    human_approved: NotRequired[bool]


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


# Backward-compatible name for callers of the original unused state module.
GraphState = RagGraphState

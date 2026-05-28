import operator
from typing import Literal
from typing_extensions import TypedDict, Annotated

from langchain_core.messages import AnyMessage


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add] 
    query: str
    retrieved_docs: list[str]
    retrieval_scores: list[float]
    draft_answer: str
    faithfulness_score: float
    unsupported_claims: list[str]
    route_decision: str
    
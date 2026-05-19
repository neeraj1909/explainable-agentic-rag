from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class NextAction(str, Enum):
    no_follow_up_needed = "no_follow_up_needed"
    retrieve_more = "retrieve_more"
    ask_clarifying_question = "ask_clarifying_question"
    human_review = "human_review"
    

class SourceUsed(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    title: str = Field(description="Title of the paper or source used.")
    url: str | None = Field(default=None, description="URL of the source.")
    reason_used: str = Field(description="Why this source supports the answer.")
    
    
class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    answer: str = Field(description="Concise answer grounded in retrieved evidence.")
    confidence: float = Field(ge=0.0, le=1.0)
    sources_used: list[SourceUsed] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    next_action: NextAction 

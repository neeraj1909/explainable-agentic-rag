import pytest
from pydantic import ValidationError

from schemas import AgentResponse, SourceUsed, NextAction


def test_valid_agent_response():
    response = AgentResponse(
        answer="Reranking can improve RA faithfulness when it promotes more relevant evidence.",
        confidence=0.82,
        sources_used=[
            SourceUsed(
                title="Example Paper",
                url="https://arxiv.org/abs/1234.5678",
                reason_used="Discusses reranking and retrieval quality."
            )
        ],
        unsupported_claims=[],
        next_action=NextAction.no_follow_up_needed,
    )
    
    assert response.confidence == 0.82
    

def test_confidence_must_be_between_zero_and_one():
    with pytest.raises(ValidationError):
        AgentResponse(
            answer="Bad confidence",
            confidence=1.5,  # Invalid confidence > 1
            sources_used=[],
            unsupported_claims=[],
            next_action=NextAction.no_follow_up_needed,
        )

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.contracts import (
    CONTRACT_SCHEMA_VERSION,
    ClaimStatus,
    ClaimVerification,
    ConfidenceEstimate,
    EvidenceChunk,
    ProgressEvent,
    RAGMode,
    RAGRequest,
    RAGResponse,
    RouteStep,
    RunMetrics,
    ScoreProvenance,
    VerificationStatus,
    VerificationSummary,
    adapt_agent_response,
)
from app.schemas import AgentResponse, NextAction, SourceUsed


def test_valid_agent_response():
    response = AgentResponse(
        answer="Reranking can improve RA faithfulness when it promotes more relevant evidence.",
        confidence=0.82,
        sources_used=[
            SourceUsed(
                title="Example Paper",
                url="https://arxiv.org/abs/1234.5678",
                reason_used="Discusses reranking and retrieval quality.",
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


def _canonical_response() -> RAGResponse:
    evidence = EvidenceChunk(
        document_id="sha256:document",
        chunk_id="sha256:chunk",
        source="docs/example.pdf",
        title="Example Paper",
        page=2,
        content="Reranking can promote evidence that is more relevant to a query.",
        scores=[
            ScoreProvenance(
                name="dense_retriever",
                value=0.73,
                rank=1,
                higher_is_better=True,
                method="cosine_similarity",
                model="text-embedding-test",
            )
        ],
        selected_rank=1,
        reason_selected="Highest-scoring directly relevant chunk.",
    )
    verification = VerificationSummary(
        status=VerificationStatus.verified,
        verified=True,
        score=0.9,
        method="claim_evidence_entailment",
        claims=[
            ClaimVerification(
                claim_id="claim-1",
                claim="Reranking can improve evidence relevance.",
                status=ClaimStatus.supported,
                evidence_chunk_ids=[evidence.chunk_id],
                score=0.9,
                reason="The evidence directly supports the claim.",
            )
        ],
    )

    return RAGResponse(
        mode=RAGMode.two_step,
        answer="Reranking can improve evidence relevance.",
        evidence=[evidence],
        verification=verification,
        confidence=ConfidenceEstimate(
            score=0.84,
            method="verified_claim_coverage",
            calibrated=False,
        ),
        metrics=RunMetrics(
            latency_ms=125.5,
            model_calls=1,
            retrieval_calls=1,
            retry_count=0,
        ),
        trace_id="trace-123",
        corpus_version="sha256:corpus",
        index_version="sha256:index",
        route_history=[
            RouteStep(
                step=1,
                agent="retriever",
                decision="evidence_retrieved",
                reason="Relevant chunks were found.",
            )
        ],
    )


def test_canonical_response_round_trips_through_json():
    response = _canonical_response()

    payload = response.model_dump_json()
    restored = RAGResponse.model_validate_json(payload)
    decoded = json.loads(payload)

    assert restored == response
    assert decoded["schema_version"] == CONTRACT_SCHEMA_VERSION
    assert decoded["mode"] == "two-step"
    assert decoded["evidence"][0]["scores"][0]["rank"] == 1
    assert decoded["verification"]["claims"][0]["evidence_chunk_ids"] == [
        "sha256:chunk"
    ]


@pytest.mark.parametrize(
    "mode",
    [
        RAGMode.two_step,
        RAGMode.agentic,
        RAGMode.graph,
        RAGMode.multi_agent,
    ],
)
def test_all_four_modes_share_request_and_response_contracts(mode):
    request = RAGRequest(
        query="What evidence supports this answer?",
        mode=mode,
        top_k=4,
        max_retries=2,
    )
    response = RAGResponse(
        mode=mode,
        answer="No verified answer is available yet.",
    )

    assert request.mode is mode
    assert response.mode is mode
    assert response.verification.status is VerificationStatus.not_run
    assert response.model_dump(mode="json")["mode"] == mode.value


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (ConfidenceEstimate, {"score": 1.01}),
        (
            VerificationSummary,
            {
                "status": VerificationStatus.failed,
                "verified": False,
                "score": -0.01,
            },
        ),
    ],
)
def test_normalized_scores_reject_values_outside_zero_and_one(model, payload):
    with pytest.raises(ValidationError):
        model.model_validate(payload)


@pytest.mark.parametrize("field", ["latency_ms", "estimated_cost_usd"])
def test_run_metrics_reject_non_finite_values(field):
    with pytest.raises(ValidationError):
        RunMetrics.model_validate({field: float("inf")})


def test_supported_claim_requires_evidence_ids():
    with pytest.raises(ValidationError, match="supported claim requires evidence"):
        ClaimVerification(
            claim_id="claim-1",
            claim="The claim is supported.",
            status=ClaimStatus.supported,
            evidence_chunk_ids=[],
            reason="No evidence was linked.",
        )


def test_response_rejects_verification_that_references_missing_evidence():
    with pytest.raises(ValidationError, match="unknown evidence chunk"):
        RAGResponse(
            mode=RAGMode.graph,
            answer="A supposedly verified answer.",
            evidence=[
                EvidenceChunk(
                    document_id="doc-1",
                    chunk_id="chunk-1",
                    source="docs/example.pdf",
                )
            ],
            verification=VerificationSummary(
                status=VerificationStatus.verified,
                verified=True,
                claims=[
                    ClaimVerification(
                        claim_id="claim-1",
                        claim="A verified claim.",
                        status=ClaimStatus.supported,
                        evidence_chunk_ids=["missing-chunk"],
                        reason="This reference is invalid.",
                    )
                ],
            ),
        )


def test_verified_response_requires_evidence():
    with pytest.raises(ValidationError, match="verified response requires evidence"):
        RAGResponse(
            mode=RAGMode.graph,
            answer="A supposedly verified answer.",
            verification=VerificationSummary(
                status=VerificationStatus.verified,
                verified=True,
            ),
        )


def test_contract_rejects_unknown_schema_version():
    payload = _canonical_response().model_dump(mode="json")
    payload["schema_version"] = "2.0"

    with pytest.raises(ValidationError):
        RAGResponse.model_validate(payload)


def test_agent_response_adapter_preserves_legacy_json_and_maps_explicitly():
    legacy_payload = {
        "answer": "The retrieved paper supports the answer.",
        "confidence": 0.82,
        "sources_used": [
            {
                "title": "Example Paper",
                "url": "https://arxiv.org/abs/1234.5678",
                "reason_used": "It directly discusses the claim.",
            }
        ],
        "unsupported_claims": ["A separate claim lacks evidence."],
        "next_action": "human_review",
    }
    legacy = AgentResponse.model_validate(legacy_payload)

    canonical = adapt_agent_response(
        legacy,
        corpus_version="arxiv:live",
        trace_id="trace-legacy",
    )
    canonical_again = adapt_agent_response(
        legacy,
        corpus_version="arxiv:live",
        trace_id="trace-legacy",
    )

    assert legacy.model_dump(mode="json") == legacy_payload
    assert canonical.mode is RAGMode.research_assistant
    assert canonical.answer == legacy.answer
    assert canonical.next_action is NextAction.human_review
    assert canonical.confidence.score == legacy.confidence
    assert canonical.confidence.calibrated is False
    assert canonical.verification.status is VerificationStatus.failed
    assert canonical.verification.unsupported_claims == legacy.unsupported_claims
    assert canonical.evidence[0].title == "Example Paper"
    assert canonical.evidence[0].reason_selected == ("It directly discusses the claim.")
    assert canonical.evidence[0].chunk_id == canonical_again.evidence[0].chunk_id


def test_progress_event_is_versioned_timezone_aware_and_json_serializable():
    event = ProgressEvent(
        event="retrieval_started",
        message="Retrieving evidence.",
        sequence=1,
        mode=RAGMode.agentic,
        timestamp=datetime(2026, 7, 29, 8, 30, tzinfo=UTC),
        data={"query": "agentic RAG", "top_k": 4},
    )

    payload = json.loads(event.model_dump_json())

    assert payload["schema_version"] == CONTRACT_SCHEMA_VERSION
    assert payload["mode"] == "agentic"
    assert payload["timestamp"] == "2026-07-29T08:30:00Z"
    assert payload["data"] == {"query": "agentic RAG", "top_k": 4}

    with pytest.raises(ValidationError, match="timezone"):
        ProgressEvent(
            event="retrieval_started",
            message="Retrieving evidence.",
            timestamp=datetime(2026, 7, 29, 8, 30),
        )


def test_contract_models_forbid_undocumented_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RAGResponse(
            mode=RAGMode.two_step,
            answer="An answer.",
            undocumented_mode_field="not allowed",
        )

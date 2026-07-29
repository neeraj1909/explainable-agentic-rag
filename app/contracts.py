"""Canonical, versioned wire contracts for all RAG execution modes."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import Enum
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    JsonValue,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

from app.schemas import AgentResponse, NextAction


CONTRACT_SCHEMA_VERSION = "1.0"


class RAGMode(str, Enum):
    two_step = "two-step"
    agentic = "agentic"
    graph = "graph"
    multi_agent = "multi-agent"
    research_assistant = "research-assistant"


class ClaimStatus(str, Enum):
    supported = "supported"
    unsupported = "unsupported"
    insufficient_evidence = "insufficient_evidence"
    not_checked = "not_checked"


class VerificationStatus(str, Enum):
    not_run = "not_run"
    verified = "verified"
    partially_verified = "partially_verified"
    failed = "failed"


class StrictContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScoreProvenance(StrictContractModel):
    """One score emitted by a named retrieval, fusion, or reranking stage."""

    name: str
    value: FiniteFloat
    rank: PositiveInt | None = None
    higher_is_better: bool | None = None
    method: str | None = None
    model: str | None = None

    @field_validator("name")
    @classmethod
    def require_name(cls, value: str) -> str:
        return _require_text(value, field_name="score name")

    @field_validator("method", "model")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        return _optional_text(value)


class EvidenceChunk(StrictContractModel):
    """A stable evidence reference with optional content and score provenance."""

    document_id: str
    chunk_id: str
    source: str | None = None
    title: str | None = None
    page: NonNegativeInt | None = None
    content: str | None = None
    scores: list[ScoreProvenance] = Field(default_factory=list)
    selected_rank: PositiveInt | None = None
    reason_selected: str | None = None

    @field_validator("document_id", "chunk_id")
    @classmethod
    def require_stable_ids(cls, value: str) -> str:
        return _require_text(value, field_name="evidence identifier")

    @field_validator("source", "title", "content", "reason_selected")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @model_validator(mode="after")
    def require_unique_score_names(self) -> Self:
        names = [score.name for score in self.scores]
        if len(names) != len(set(names)):
            raise ValueError("evidence score names must be unique")
        return self


class ClaimVerification(StrictContractModel):
    claim_id: str
    claim: str
    status: ClaimStatus
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    score: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    reason: str

    @field_validator("claim_id", "claim", "reason")
    @classmethod
    def require_claim_text(cls, value: str) -> str:
        return _require_text(value, field_name="claim verification text")

    @field_validator("evidence_chunk_ids")
    @classmethod
    def normalize_evidence_ids(cls, values: list[str]) -> list[str]:
        normalized = [
            _require_text(value, field_name="evidence chunk identifier")
            for value in values
        ]
        if len(normalized) != len(set(normalized)):
            raise ValueError("evidence chunk identifiers must be unique")
        return normalized

    @model_validator(mode="after")
    def require_evidence_for_supported_claim(self) -> Self:
        if self.status is ClaimStatus.supported and not self.evidence_chunk_ids:
            raise ValueError("supported claim requires evidence chunk identifiers")
        return self


class VerificationSummary(StrictContractModel):
    status: VerificationStatus = VerificationStatus.not_run
    verified: bool = False
    score: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    method: str | None = None
    claims: list[ClaimVerification] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)

    @field_validator("method")
    @classmethod
    def normalize_optional_method(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @field_validator("unsupported_claims")
    @classmethod
    def normalize_unsupported_claims(cls, values: list[str]) -> list[str]:
        normalized = [
            _require_text(value, field_name="unsupported claim") for value in values
        ]
        if len(normalized) != len(set(normalized)):
            raise ValueError("unsupported claims must be unique")
        return normalized

    @model_validator(mode="after")
    def keep_status_consistent(self) -> Self:
        if self.verified != (self.status is VerificationStatus.verified):
            raise ValueError("verified must be true exactly when status is verified")
        if self.verified and self.unsupported_claims:
            raise ValueError("verified summary cannot contain unsupported claims")
        if self.verified and any(
            claim.status is not ClaimStatus.supported for claim in self.claims
        ):
            raise ValueError("verified summary cannot contain unverified claims")
        return self


class ConfidenceEstimate(StrictContractModel):
    score: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    method: str | None = None
    calibrated: bool = False
    calibration_method: str | None = None

    @field_validator("method", "calibration_method")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @model_validator(mode="after")
    def require_calibration_metadata(self) -> Self:
        if self.calibrated and self.score is None:
            raise ValueError("calibrated confidence requires a score")
        if self.calibrated and self.calibration_method is None:
            raise ValueError("calibrated confidence requires a calibration method")
        return self


class RunMetrics(StrictContractModel):
    latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
    )
    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None
    total_tokens: NonNegativeInt | None = None
    estimated_cost_usd: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
    )
    model_calls: NonNegativeInt = 0
    embedding_calls: NonNegativeInt = 0
    retrieval_calls: NonNegativeInt = 0
    tool_calls: NonNegativeInt = 0
    retry_count: NonNegativeInt = 0


class RouteStep(StrictContractModel):
    step: PositiveInt
    agent: str
    decision: str
    reason: str
    called_by: str | None = None
    retry_count: NonNegativeInt | None = None
    verified: bool | None = None
    verification_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )

    @field_validator("agent", "decision", "reason")
    @classmethod
    def require_route_text(cls, value: str) -> str:
        return _require_text(value, field_name="route step text")

    @field_validator("called_by")
    @classmethod
    def normalize_called_by(cls, value: str | None) -> str | None:
        return _optional_text(value)


class RAGRequest(StrictContractModel):
    schema_version: Literal["1.0"] = CONTRACT_SCHEMA_VERSION
    query: str
    mode: RAGMode
    top_k: PositiveInt | None = None
    max_retries: NonNegativeInt | None = None
    thread_id: str | None = None
    stream: bool = False

    @field_validator("query")
    @classmethod
    def require_query(cls, value: str) -> str:
        return _require_text(value, field_name="query")

    @field_validator("thread_id")
    @classmethod
    def normalize_thread_id(cls, value: str | None) -> str | None:
        return _optional_text(value)


class RAGResponse(StrictContractModel):
    schema_version: Literal["1.0"] = CONTRACT_SCHEMA_VERSION
    mode: RAGMode
    answer: str
    explanation: str | None = None
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    verification: VerificationSummary = Field(default_factory=VerificationSummary)
    confidence: ConfidenceEstimate = Field(default_factory=ConfidenceEstimate)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    run_id: str | None = None
    trace_id: str | None = None
    corpus_version: str | None = None
    index_version: str | None = None
    route_history: list[RouteStep] = Field(default_factory=list)
    next_action: NextAction | None = None

    @field_validator("answer")
    @classmethod
    def require_answer(cls, value: str) -> str:
        return _require_text(value, field_name="answer")

    @field_validator(
        "explanation",
        "run_id",
        "trace_id",
        "corpus_version",
        "index_version",
    )
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @model_validator(mode="after")
    def validate_evidence_links(self) -> Self:
        chunk_ids = [chunk.chunk_id for chunk in self.evidence]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("response evidence chunk identifiers must be unique")

        known_chunk_ids = set(chunk_ids)
        referenced_chunk_ids = {
            chunk_id
            for claim in self.verification.claims
            for chunk_id in claim.evidence_chunk_ids
        }
        unknown_chunk_ids = sorted(referenced_chunk_ids - known_chunk_ids)
        if unknown_chunk_ids:
            raise ValueError(
                "verification references unknown evidence chunk identifiers: "
                f"{', '.join(unknown_chunk_ids)}"
            )

        if self.verification.verified and not self.evidence:
            raise ValueError("verified response requires evidence")

        return self


class ProgressEvent(StrictContractModel):
    schema_version: Literal["1.0"] = CONTRACT_SCHEMA_VERSION
    event: str
    message: str
    sequence: NonNegativeInt | None = None
    mode: RAGMode | None = None
    agent: str | None = None
    decision: str | None = None
    reason: str | None = None
    retry_count: NonNegativeInt | None = None
    verified: bool | None = None
    verification_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    run_id: str | None = None
    trace_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    data: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("event", "message")
    @classmethod
    def require_event_text(cls, value: str) -> str:
        return _require_text(value, field_name="progress event text")

    @field_validator(
        "agent",
        "decision",
        "reason",
        "run_id",
        "trace_id",
    )
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("progress event timestamp must include a timezone")
        return value


def adapt_agent_response(
    response: AgentResponse,
    *,
    corpus_version: str | None = None,
    index_version: str | None = None,
    run_id: str | None = None,
    trace_id: str | None = None,
    metrics: RunMetrics | None = None,
) -> RAGResponse:
    """Explicitly map the legacy research response into the canonical contract."""

    evidence = []
    for source in response.sources_used:
        identity = f"{source.title}\0{source.url or ''}"
        evidence_id = _stable_legacy_id("source", identity)
        evidence.append(
            EvidenceChunk(
                document_id=evidence_id,
                chunk_id=f"{evidence_id}:chunk",
                source=source.url,
                title=source.title,
                reason_selected=source.reason_used,
            )
        )

    claim_verifications = [
        ClaimVerification(
            claim_id=_stable_legacy_id("claim", claim),
            claim=claim,
            status=ClaimStatus.unsupported,
            reason="Legacy AgentResponse marked this claim as unsupported.",
        )
        for claim in response.unsupported_claims
    ]
    verification_status = (
        VerificationStatus.failed
        if response.unsupported_claims
        else VerificationStatus.not_run
    )

    return RAGResponse(
        mode=RAGMode.research_assistant,
        answer=response.answer,
        evidence=evidence,
        verification=VerificationSummary(
            status=verification_status,
            verified=False,
            method=(
                "legacy_agent_response_claim_flags"
                if response.unsupported_claims
                else None
            ),
            claims=claim_verifications,
            unsupported_claims=response.unsupported_claims,
        ),
        confidence=ConfidenceEstimate(
            score=response.confidence,
            method="legacy_agent_self_report",
            calibrated=False,
        ),
        metrics=metrics or RunMetrics(),
        run_id=run_id,
        trace_id=trace_id,
        corpus_version=corpus_version,
        index_version=index_version,
        next_action=response.next_action,
    )


def _require_text(value: str, *, field_name: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be blank")
    return stripped


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    return _require_text(value, field_name="optional text")


def _stable_legacy_id(namespace: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"legacy-{namespace}:sha256:{digest}"

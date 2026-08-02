"""Public RAGAS 0.4 metric adapter for comparative evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Literal, Protocol

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, FiniteFloat
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)

from app.config import AppSettings, get_settings


RAGAS_METRIC_NAMES = (
    "faithfulness",
    "context_precision",
    "context_recall",
    "factual_correctness",
    "answer_relevancy",
)

MetricField = Literal[
    "user_input",
    "response",
    "retrieved_contexts",
    "reference",
]


@dataclass(frozen=True, slots=True)
class EvaluationSample:
    """The fields used by the supported public RAGAS metric contracts."""

    user_input: str
    response: str
    retrieved_contexts: list[str]
    reference: str


class MetricScore(BaseModel):
    """Serializable numeric RAGAS output plus optional judge reasoning."""

    model_config = ConfigDict(extra="forbid")

    value: FiniteFloat
    reason: str | None = None


class ScoringMetric(Protocol):
    def score(self, **kwargs): ...


@dataclass(frozen=True, slots=True)
class MetricBinding:
    """Bind one public metric instance to the sample fields it accepts."""

    name: str
    metric: ScoringMetric
    fields: tuple[MetricField, ...]


class MetricSuite(Protocol):
    @property
    def metric_names(self) -> tuple[str, ...]: ...

    def score(self, sample: EvaluationSample) -> dict[str, MetricScore]: ...


@dataclass(frozen=True, slots=True)
class RagasMetricSuite:
    """Score one sample using only RAGAS's public collections API."""

    bindings: tuple[MetricBinding, ...]

    def __post_init__(self) -> None:
        if not self.bindings:
            raise ValueError("at least one metric binding is required")
        names = [binding.name for binding in self.bindings]
        if len(names) != len(set(names)):
            raise ValueError("metric binding names must be unique")

    @property
    def metric_names(self) -> tuple[str, ...]:
        return tuple(binding.name for binding in self.bindings)

    def score(self, sample: EvaluationSample) -> dict[str, MetricScore]:
        scores = {}
        for binding in self.bindings:
            inputs = {field: getattr(sample, field) for field in binding.fields}
            result = binding.metric.score(**inputs)
            raw_value = result.value
            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                raise TypeError(
                    f"RAGAS metric {binding.name!r} returned a non-numeric value"
                )
            scores[binding.name] = MetricScore(
                value=float(raw_value),
                reason=getattr(result, "reason", None),
            )
        return scores


def build_ragas_metric_suite(
    *,
    settings: AppSettings | None = None,
) -> RagasMetricSuite:
    """Build evaluator clients and public RAGAS 0.4.3 collection metrics."""

    resolved = settings or get_settings()
    resolved.require_chat_credentials()
    resolved.require_embedding_credentials()

    chat_client_options = {
        "api_key": resolved.chat_api_key.get_secret_value(),
        "timeout": float(resolved.chat_timeout_seconds),
    }
    if resolved.chat_api_base is not None:
        chat_client_options["base_url"] = resolved.chat_api_base
    chat_client = OpenAI(**chat_client_options)
    evaluator_llm = llm_factory(
        resolved.chat_model,
        provider="openai",
        client=chat_client,
        temperature=0,
    )

    embedding_client = OpenAI(
        api_key=resolved.embedding_api_key.get_secret_value(),
        timeout=float(resolved.embedding_timeout_seconds),
    )
    evaluator_embeddings = OpenAIEmbeddings(
        client=embedding_client,
        model=resolved.embedding_model,
    )

    return RagasMetricSuite(
        bindings=(
            MetricBinding(
                name="faithfulness",
                metric=Faithfulness(llm=evaluator_llm),
                fields=("user_input", "response", "retrieved_contexts"),
            ),
            MetricBinding(
                name="context_precision",
                metric=ContextPrecision(llm=evaluator_llm),
                fields=("user_input", "reference", "retrieved_contexts"),
            ),
            MetricBinding(
                name="context_recall",
                metric=ContextRecall(llm=evaluator_llm),
                fields=("user_input", "retrieved_contexts", "reference"),
            ),
            MetricBinding(
                name="factual_correctness",
                metric=FactualCorrectness(llm=evaluator_llm),
                fields=("response", "reference"),
            ),
            MetricBinding(
                name="answer_relevancy",
                metric=AnswerRelevancy(
                    llm=evaluator_llm,
                    embeddings=evaluator_embeddings,
                    strictness=1,
                ),
                fields=("user_input", "response"),
            ),
        )
    )

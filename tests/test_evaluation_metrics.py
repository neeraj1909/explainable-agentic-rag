from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config import AppSettings
from app.evaluation import metrics
from app.evaluation.metrics import (
    EvaluationSample,
    MetricBinding,
    RagasMetricSuite,
)


class FakeMetric:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = []

    def score(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(value=self.value, reason=f"reason-{self.value}")


def test_ragas_metric_suite_maps_each_public_metric_contract() -> None:
    faithfulness = FakeMetric(0.1)
    context_precision = FakeMetric(0.2)
    context_recall = FakeMetric(0.3)
    factual_correctness = FakeMetric(0.4)
    answer_relevancy = FakeMetric(0.5)
    suite = RagasMetricSuite(
        bindings=(
            MetricBinding(
                name="faithfulness",
                metric=faithfulness,
                fields=("user_input", "response", "retrieved_contexts"),
            ),
            MetricBinding(
                name="context_precision",
                metric=context_precision,
                fields=(
                    "user_input",
                    "reference",
                    "retrieved_contexts",
                ),
            ),
            MetricBinding(
                name="context_recall",
                metric=context_recall,
                fields=(
                    "user_input",
                    "retrieved_contexts",
                    "reference",
                ),
            ),
            MetricBinding(
                name="factual_correctness",
                metric=factual_correctness,
                fields=("response", "reference"),
            ),
            MetricBinding(
                name="answer_relevancy",
                metric=answer_relevancy,
                fields=("user_input", "response"),
            ),
        )
    )
    sample = EvaluationSample(
        user_input="Question?",
        response="Answer.",
        retrieved_contexts=["Evidence."],
        reference="Reference.",
    )

    scores = suite.score(sample)

    assert {name: score.value for name, score in scores.items()} == {
        "faithfulness": 0.1,
        "context_precision": 0.2,
        "context_recall": 0.3,
        "factual_correctness": 0.4,
        "answer_relevancy": 0.5,
    }
    assert faithfulness.calls == [
        {
            "user_input": "Question?",
            "response": "Answer.",
            "retrieved_contexts": ["Evidence."],
        }
    ]
    assert context_precision.calls[0]["reference"] == "Reference."
    assert context_recall.calls[0]["retrieved_contexts"] == ["Evidence."]
    assert factual_correctness.calls == [
        {"response": "Answer.", "reference": "Reference."}
    ]
    assert answer_relevancy.calls == [
        {"user_input": "Question?", "response": "Answer."}
    ]


def test_build_ragas_metric_suite_uses_native_clients_and_public_factories(
    monkeypatch,
) -> None:
    clients = []

    def fake_openai(**kwargs):
        client = SimpleNamespace(kwargs=kwargs)
        clients.append(client)
        return client

    evaluator_llm = object()
    evaluator_embeddings = object()
    llm_calls = []
    embedding_calls = []

    def fake_llm_factory(model, **kwargs):
        llm_calls.append((model, kwargs))
        return evaluator_llm

    def fake_embeddings(*, client, model):
        embedding_calls.append((client, model))
        return evaluator_embeddings

    class FakeMetricFactory:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(metrics, "OpenAI", fake_openai)
    monkeypatch.setattr(metrics, "llm_factory", fake_llm_factory)
    monkeypatch.setattr(metrics, "OpenAIEmbeddings", fake_embeddings)
    monkeypatch.setattr(metrics, "Faithfulness", FakeMetricFactory)
    monkeypatch.setattr(metrics, "ContextPrecision", FakeMetricFactory)
    monkeypatch.setattr(metrics, "ContextRecall", FakeMetricFactory)
    monkeypatch.setattr(metrics, "FactualCorrectness", FakeMetricFactory)
    monkeypatch.setattr(metrics, "AnswerRelevancy", FakeMetricFactory)
    settings = AppSettings(
        _env_file=None,
        chat_model="judge-model",
        chat_api_key="chat-secret",
        chat_api_base="https://chat.example/v1",
        embedding_model="embedding-model",
        embedding_api_key="embedding-secret",
    )

    suite = metrics.build_ragas_metric_suite(settings=settings)

    assert len(clients) == 2
    assert clients[0].kwargs == {
        "api_key": "chat-secret",
        "base_url": "https://chat.example/v1",
        "timeout": 30.0,
    }
    assert clients[1].kwargs == {
        "api_key": "embedding-secret",
        "timeout": 30.0,
    }
    assert llm_calls == [
        (
            "judge-model",
            {
                "provider": "openai",
                "client": clients[0],
                "temperature": 0,
            },
        )
    ]
    assert embedding_calls == [(clients[1], "embedding-model")]
    assert suite.metric_names == (
        "faithfulness",
        "context_precision",
        "context_recall",
        "factual_correctness",
        "answer_relevancy",
    )
    assert suite.bindings[0].metric.kwargs == {"llm": evaluator_llm}
    assert suite.bindings[-1].metric.kwargs == {
        "llm": evaluator_llm,
        "embeddings": evaluator_embeddings,
        "strictness": 1,
    }


def test_metric_suite_rejects_non_finite_results() -> None:
    suite = RagasMetricSuite(
        bindings=(
            MetricBinding(
                name="bad_metric",
                metric=FakeMetric(float("nan")),
                fields=("response",),
            ),
        )
    )

    with pytest.raises(ValueError):
        suite.score(
            EvaluationSample(
                user_input="Question?",
                response="Answer.",
                retrieved_contexts=[],
                reference="Reference.",
            )
        )

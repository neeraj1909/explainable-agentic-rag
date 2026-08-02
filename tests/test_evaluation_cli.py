from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from app.contracts import EvidenceChunk, RAGMode, RAGResponse
from app.evaluation.cli import (
    main,
    parse_args,
    run_evaluation,
    write_evaluation_artifacts,
)
from app.evaluation.metrics import EvaluationSample, MetricScore
from app.evaluation.run_manifest import EvaluationExample


FIXED_TIME = datetime(2026, 8, 2, 10, 30, tzinfo=UTC)


def examples() -> list[EvaluationExample]:
    return [
        EvaluationExample(
            user_input="Question one?",
            reference="Reference one.",
            expected_relevant_sources=["docs/one.pdf"],
        ),
        EvaluationExample(
            user_input="Question two?",
            reference="Reference two.",
            expected_relevant_sources=["docs/two.pdf"],
        ),
    ]


class FakeSystem:
    def __init__(self, mode: RAGMode, *, failure: Exception | None = None) -> None:
        self.mode = mode
        self.failure = failure
        self.questions: list[str] = []

    def answer(self, question: str) -> RAGResponse:
        self.questions.append(question)
        if self.failure is not None:
            raise self.failure
        return RAGResponse(
            mode=self.mode,
            answer=f"{self.mode.value} answer for {question}",
            evidence=[
                EvidenceChunk(
                    document_id=f"document-{self.mode.value}",
                    chunk_id=f"chunk-{self.mode.value}",
                    source=f"docs/{self.mode.value}.pdf",
                    content=f"Evidence for {question}",
                    selected_rank=1,
                )
            ],
            run_id=f"runtime-{self.mode.value}",
        )


class FakeMetricSuite:
    metric_names = ("fake_quality",)

    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.samples: list[EvaluationSample] = []

    def score(self, sample: EvaluationSample) -> dict[str, MetricScore]:
        self.samples.append(sample)
        if self.failure is not None:
            raise self.failure
        return {"fake_quality": MetricScore(value=0.75, reason="offline fake")}


def fake_systems(modes: tuple[RAGMode, ...]) -> dict[RAGMode, FakeSystem]:
    return {mode: FakeSystem(mode) for mode in modes}


def test_runner_constructs_rows_and_dispatches_modes_in_stable_order() -> None:
    modes = (RAGMode.two_step, RAGMode.agentic)
    systems = fake_systems(modes)
    metric_suite = FakeMetricSuite()

    batch = run_evaluation(
        examples=examples(),
        systems=systems,
        metric_suite=metric_suite,
        repeat=2,
        generated_at=FIXED_TIME,
    )

    assert [record.evaluation_run_id for record in batch.records] == [
        "question-001:two-step:repeat-001",
        "question-001:two-step:repeat-002",
        "question-001:agentic:repeat-001",
        "question-001:agentic:repeat-002",
        "question-002:two-step:repeat-001",
        "question-002:two-step:repeat-002",
        "question-002:agentic:repeat-001",
        "question-002:agentic:repeat-002",
    ]
    assert batch.manifest.planned_run_count == 8
    assert batch.manifest.succeeded_run_count == 8
    assert batch.manifest.failed_run_count == 0
    assert batch.manifest.status == "succeeded"
    assert batch.records[0].retrieved_contexts == ["Evidence for Question one?"]
    assert batch.records[0].metric_scores["fake_quality"].value == 0.75
    assert batch.records[0].expected_relevant_sources == ["docs/one.pdf"]
    assert [sample.reference for sample in metric_suite.samples[:2]] == [
        "Reference one.",
        "Reference one.",
    ]
    assert systems[RAGMode.two_step].questions == [
        "Question one?",
        "Question one?",
        "Question two?",
        "Question two?",
    ]


def test_runner_records_system_exceptions_and_continues() -> None:
    systems = {
        RAGMode.two_step: FakeSystem(RAGMode.two_step),
        RAGMode.graph: FakeSystem(
            RAGMode.graph,
            failure=RuntimeError("graph exploded"),
        ),
    }

    batch = run_evaluation(
        examples=examples()[:1],
        systems=systems,
        metric_suite=FakeMetricSuite(),
        repeat=1,
        generated_at=FIXED_TIME,
    )

    assert batch.manifest.status == "partial"
    assert batch.manifest.succeeded_run_count == 1
    assert batch.manifest.failed_run_count == 1
    failed = batch.records[1]
    assert failed.status == "failed"
    assert failed.response is None
    assert failed.error is not None
    assert failed.error.stage == "system"
    assert failed.error.exception_type == "RuntimeError"
    assert failed.error.message == "graph exploded"


def test_runner_preserves_response_when_metric_scoring_fails() -> None:
    batch = run_evaluation(
        examples=examples()[:1],
        systems=fake_systems((RAGMode.multi_agent,)),
        metric_suite=FakeMetricSuite(failure=ValueError("judge unavailable")),
        repeat=1,
        generated_at=FIXED_TIME,
    )

    failed = batch.records[0]
    assert batch.manifest.status == "failed"
    assert failed.status == "failed"
    assert failed.response == "multi-agent answer for Question one?"
    assert failed.retrieved_contexts == ["Evidence for Question one?"]
    assert failed.error is not None
    assert failed.error.stage == "metrics"
    assert failed.error.message == "judge unavailable"


def test_manifest_and_jsonl_are_deterministic_and_create_parent_directories(
    tmp_path: Path,
) -> None:
    kwargs = {
        "examples": examples(),
        "systems": fake_systems((RAGMode.two_step,)),
        "metric_suite": FakeMetricSuite(),
        "repeat": 1,
        "generated_at": FIXED_TIME,
    }
    first = run_evaluation(**kwargs)
    second = run_evaluation(
        **{
            **kwargs,
            "systems": fake_systems((RAGMode.two_step,)),
            "metric_suite": FakeMetricSuite(),
        }
    )

    assert first.model_dump(mode="json") == second.model_dump(mode="json")

    output_dir = tmp_path / "nested" / "evaluation"
    paths = write_evaluation_artifacts(first, output_dir)

    assert paths.manifest == output_dir / "manifest.json"
    assert paths.results == output_dir / "results.jsonl"
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in paths.results.read_text(encoding="utf-8").splitlines()
    ]
    assert manifest["planned_run_count"] == 2
    assert [record["evaluation_run_id"] for record in records] == [
        "question-001:two-step:repeat-001",
        "question-002:two-step:repeat-001",
    ]


def test_cli_parses_all_step_2_1_options() -> None:
    args = parse_args(
        [
            "--modes",
            "agentic",
            "graph",
            "--limit",
            "2",
            "--repeat",
            "3",
            "--output-dir",
            "evaluation/custom",
            "--dry-run",
        ]
    )

    assert args.modes == ["agentic", "graph"]
    assert args.limit == 2
    assert args.repeat == 3
    assert args.output_dir == Path("evaluation/custom")
    assert args.dry_run is True


def test_dry_run_makes_no_providers_or_output_directory(
    tmp_path: Path,
    capsys,
) -> None:
    output_dir = tmp_path / "must-not-exist"

    def fail_builder(**kwargs):
        raise AssertionError(f"provider builder called: {kwargs}")

    exit_code = main(
        [
            "--modes",
            "two-step",
            "multi-agent",
            "--limit",
            "2",
            "--repeat",
            "3",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        dataset_loader=examples,
        system_builder=fail_builder,
        metric_suite_builder=fail_builder,
        clock=lambda: FIXED_TIME,
    )

    plan = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert plan["modes"] == ["two-step", "multi-agent"]
    assert plan["question_count"] == 2
    assert plan["repeat"] == 3
    assert plan["planned_run_count"] == 12
    assert plan["network_calls_performed"] is False
    assert not output_dir.exists()


def test_two_question_four_mode_cli_smoke_is_offline(tmp_path: Path) -> None:
    modes_seen = []

    def build_fake_systems(*, modes, top_k, max_retries):
        del top_k, max_retries
        modes_seen.extend(modes)
        return fake_systems(modes)

    output_dir = tmp_path / "smoke"
    exit_code = main(
        [
            "--modes",
            "two-step",
            "agentic",
            "graph",
            "multi-agent",
            "--limit",
            "2",
            "--output-dir",
            str(output_dir),
        ],
        dataset_loader=examples,
        system_builder=build_fake_systems,
        metric_suite_builder=FakeMetricSuite,
        clock=lambda: FIXED_TIME,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    results = (output_dir / "results.jsonl").read_text(encoding="utf-8")
    assert exit_code == 0
    assert modes_seen == [
        RAGMode.two_step,
        RAGMode.agentic,
        RAGMode.graph,
        RAGMode.multi_agent,
    ]
    assert manifest["succeeded_run_count"] == 8
    assert len(results.splitlines()) == 8


def test_cli_writes_failed_rows_and_returns_nonzero(tmp_path: Path) -> None:
    def build_failing_system(*, modes, top_k, max_retries):
        del top_k, max_retries
        return {
            mode: FakeSystem(mode, failure=RuntimeError("system unavailable"))
            for mode in modes
        }

    output_dir = tmp_path / "failed-run"
    exit_code = main(
        [
            "--modes",
            "graph",
            "--limit",
            "1",
            "--output-dir",
            str(output_dir),
        ],
        dataset_loader=examples,
        system_builder=build_failing_system,
        metric_suite_builder=FakeMetricSuite,
        clock=lambda: FIXED_TIME,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    failed_row = json.loads((output_dir / "results.jsonl").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert manifest["status"] == "failed"
    assert manifest["failed_run_count"] == 1
    assert failed_row["status"] == "failed"
    assert failed_row["error"] == {
        "stage": "system",
        "exception_type": "RuntimeError",
        "message": "system unavailable",
    }

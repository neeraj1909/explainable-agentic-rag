"""Comparative evaluation runner for every local-PDF RAG mode."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

from app.config import get_settings
from app.contracts import EvidenceChunk, RAGMode, RunMetrics
from app.evaluation.metrics import (
    RAGAS_METRIC_NAMES,
    EvaluationSample,
    MetricScore,
    MetricSuite,
    build_ragas_metric_suite,
)
from app.evaluation.run_manifest import (
    EVAL_DATASET_PATH,
    REPO_ROOT,
    EvaluationExample,
    load_evaluation_dataset,
)
from app.evaluation.systems import (
    COMPARISON_MODES,
    EvaluationSystem,
    build_systems,
)


EVALUATION_SCHEMA_VERSION = "1.0"
DEFAULT_OUTPUT_DIR = Path("evaluation")
RESULTS_FILENAME = "results.jsonl"
MANIFEST_FILENAME = "manifest.json"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvaluationError(StrictModel):
    stage: Literal["system", "metrics"]
    exception_type: str = Field(min_length=1)
    message: str


class EvaluationRunRecord(StrictModel):
    schema_version: Literal["1.0"] = EVALUATION_SCHEMA_VERSION
    evaluation_run_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    question_index: PositiveInt
    repetition: PositiveInt
    mode: RAGMode
    user_input: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    expected_relevant_sources: list[str]
    expected_relevant_chunk_ids: list[str]
    status: Literal["succeeded", "failed"]
    response: str | None = None
    retrieved_contexts: list[str] = Field(default_factory=list)
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    metric_scores: dict[str, MetricScore] = Field(default_factory=dict)
    runtime_metrics: RunMetrics | None = None
    runtime_run_id: str | None = None
    trace_id: str | None = None
    error: EvaluationError | None = None

    @model_validator(mode="after")
    def keep_status_consistent(self):
        if self.status == "succeeded" and self.error is not None:
            raise ValueError("a succeeded evaluation run cannot contain an error")
        if self.status == "failed" and self.error is None:
            raise ValueError("a failed evaluation run must contain an error")
        if self.status == "succeeded" and self.response is None:
            raise ValueError("a succeeded evaluation run must contain a response")
        return self


class EvaluationRunManifest(StrictModel):
    schema_version: Literal["1.0"] = EVALUATION_SCHEMA_VERSION
    generated_at: datetime
    modes: list[RAGMode] = Field(min_length=1)
    question_count: PositiveInt
    repeat: PositiveInt
    planned_run_count: PositiveInt
    succeeded_run_count: NonNegativeInt
    failed_run_count: NonNegativeInt
    status: Literal["succeeded", "partial", "failed"]
    metric_names: list[str]
    results_file: Literal["results.jsonl"] = RESULTS_FILENAME

    @field_validator("generated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_counts(self):
        if self.succeeded_run_count + self.failed_run_count != self.planned_run_count:
            raise ValueError("run outcome counts must equal planned_run_count")
        expected_count = self.question_count * len(self.modes) * self.repeat
        if self.planned_run_count != expected_count:
            raise ValueError("planned_run_count does not match the evaluation plan")
        return self


class EvaluationBatch(StrictModel):
    manifest: EvaluationRunManifest
    records: list[EvaluationRunRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_record_count(self):
        if len(self.records) != self.manifest.planned_run_count:
            raise ValueError("record count does not match the run manifest")
        return self


class DryRunPlan(StrictModel):
    schema_version: Literal["1.0"] = EVALUATION_SCHEMA_VERSION
    modes: list[RAGMode]
    question_count: PositiveInt
    repeat: PositiveInt
    planned_run_count: PositiveInt
    output_dir: Path
    metric_names: list[str]
    network_calls_performed: Literal[False] = False
    artifacts_written: Literal[False] = False


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    manifest: Path
    results: Path


def run_evaluation(
    *,
    examples: Sequence[EvaluationExample],
    systems: Mapping[RAGMode, EvaluationSystem],
    metric_suite: MetricSuite,
    repeat: int,
    generated_at: datetime | None = None,
) -> EvaluationBatch:
    """Run every question/mode/repetition and retain failures as result rows."""

    if not examples:
        raise ValueError("at least one evaluation example is required")
    if not systems:
        raise ValueError("at least one evaluation system is required")
    if repeat < 1:
        raise ValueError("repeat must be at least 1")

    modes = tuple(systems)
    records = []
    for question_index, example in enumerate(examples, start=1):
        question_id = f"question-{question_index:03d}"
        for mode in modes:
            system = systems[mode]
            if system.mode is not mode:
                raise ValueError(
                    f"system registry key {mode.value!r} does not match adapter mode "
                    f"{system.mode.value!r}"
                )
            for repetition in range(1, repeat + 1):
                evaluation_run_id = (
                    f"{question_id}:{mode.value}:repeat-{repetition:03d}"
                )
                try:
                    response = system.answer(example.user_input)
                except Exception as exc:
                    records.append(
                        _failed_record(
                            example=example,
                            question_id=question_id,
                            question_index=question_index,
                            repetition=repetition,
                            mode=mode,
                            evaluation_run_id=evaluation_run_id,
                            stage="system",
                            error=exc,
                        )
                    )
                    continue

                retrieved_contexts = [
                    evidence.content
                    for evidence in response.evidence
                    if evidence.content is not None
                ]
                try:
                    metric_scores = metric_suite.score(
                        EvaluationSample(
                            user_input=example.user_input,
                            response=response.answer,
                            retrieved_contexts=retrieved_contexts,
                            reference=example.reference,
                        )
                    )
                except Exception as exc:
                    records.append(
                        _failed_record(
                            example=example,
                            question_id=question_id,
                            question_index=question_index,
                            repetition=repetition,
                            mode=mode,
                            evaluation_run_id=evaluation_run_id,
                            stage="metrics",
                            error=exc,
                            response=response.answer,
                            retrieved_contexts=retrieved_contexts,
                            evidence=response.evidence,
                            runtime_metrics=response.metrics,
                            runtime_run_id=response.run_id,
                            trace_id=response.trace_id,
                        )
                    )
                    continue

                records.append(
                    EvaluationRunRecord(
                        **_record_identity(
                            example=example,
                            question_id=question_id,
                            question_index=question_index,
                            repetition=repetition,
                            mode=mode,
                            evaluation_run_id=evaluation_run_id,
                        ),
                        status="succeeded",
                        response=response.answer,
                        retrieved_contexts=retrieved_contexts,
                        evidence=response.evidence,
                        metric_scores=metric_scores,
                        runtime_metrics=response.metrics,
                        runtime_run_id=response.run_id,
                        trace_id=response.trace_id,
                    )
                )

    failed_count = sum(record.status == "failed" for record in records)
    succeeded_count = len(records) - failed_count
    if not failed_count:
        status = "succeeded"
    elif not succeeded_count:
        status = "failed"
    else:
        status = "partial"

    manifest = EvaluationRunManifest(
        generated_at=generated_at or datetime.now(UTC),
        modes=list(modes),
        question_count=len(examples),
        repeat=repeat,
        planned_run_count=len(records),
        succeeded_run_count=succeeded_count,
        failed_run_count=failed_count,
        status=status,
        metric_names=list(metric_suite.metric_names),
    )
    return EvaluationBatch(manifest=manifest, records=records)


def _record_identity(
    *,
    example: EvaluationExample,
    question_id: str,
    question_index: int,
    repetition: int,
    mode: RAGMode,
    evaluation_run_id: str,
) -> dict:
    return {
        "evaluation_run_id": evaluation_run_id,
        "question_id": question_id,
        "question_index": question_index,
        "repetition": repetition,
        "mode": mode,
        "user_input": example.user_input,
        "reference": example.reference,
        "expected_relevant_sources": example.expected_relevant_sources,
        "expected_relevant_chunk_ids": example.expected_relevant_chunk_ids,
    }


def _failed_record(
    *,
    example: EvaluationExample,
    question_id: str,
    question_index: int,
    repetition: int,
    mode: RAGMode,
    evaluation_run_id: str,
    stage: Literal["system", "metrics"],
    error: Exception,
    response: str | None = None,
    retrieved_contexts: list[str] | None = None,
    evidence: list[EvidenceChunk] | None = None,
    runtime_metrics: RunMetrics | None = None,
    runtime_run_id: str | None = None,
    trace_id: str | None = None,
) -> EvaluationRunRecord:
    return EvaluationRunRecord(
        **_record_identity(
            example=example,
            question_id=question_id,
            question_index=question_index,
            repetition=repetition,
            mode=mode,
            evaluation_run_id=evaluation_run_id,
        ),
        status="failed",
        response=response,
        retrieved_contexts=retrieved_contexts or [],
        evidence=evidence or [],
        runtime_metrics=runtime_metrics,
        runtime_run_id=runtime_run_id,
        trace_id=trace_id,
        error=EvaluationError(
            stage=stage,
            exception_type=type(error).__name__,
            message=str(error),
        ),
    )


def write_evaluation_artifacts(
    batch: EvaluationBatch,
    output_dir: Path,
) -> ArtifactPaths:
    """Create the output directory and write deterministic JSON artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_FILENAME
    results_path = output_dir / RESULTS_FILENAME
    manifest_path.write_text(
        f"{batch.manifest.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )
    results_path.write_text(
        "".join(f"{record.model_dump_json()}\n" for record in batch.records),
        encoding="utf-8",
    )
    return ArtifactPaths(manifest=manifest_path, results=results_path)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare local-PDF RAG modes with public RAGAS metrics."
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[mode.value for mode in COMPARISON_MODES],
        default=[mode.value for mode in COMPARISON_MODES],
        help="One or more local-PDF modes (default: all four).",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Evaluate only the first N curated questions.",
    )
    parser.add_argument(
        "--repeat",
        type=_positive_int,
        default=1,
        help="Run each question/mode pair N times (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for manifest.json and results.jsonl.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run plan without constructing providers or writing files.",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    dataset_loader: Callable[[], list[EvaluationExample]] | None = None,
    system_builder: Callable[..., Mapping[RAGMode, EvaluationSystem]] = build_systems,
    metric_suite_builder: Callable[[], MetricSuite] = build_ragas_metric_suite,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> int:
    args = parse_args(argv)
    modes = tuple(dict.fromkeys(RAGMode(mode) for mode in args.modes))
    load_dataset = dataset_loader or _load_eval_set
    selected_examples = load_dataset()
    if args.limit is not None:
        selected_examples = selected_examples[: args.limit]
    if not selected_examples:
        raise ValueError("the selected evaluation dataset is empty")

    if args.dry_run:
        plan = DryRunPlan(
            modes=list(modes),
            question_count=len(selected_examples),
            repeat=args.repeat,
            planned_run_count=len(selected_examples) * len(modes) * args.repeat,
            output_dir=args.output_dir,
            metric_names=list(RAGAS_METRIC_NAMES),
        )
        print(plan.model_dump_json(indent=2))
        return 0

    settings = get_settings()
    systems = system_builder(
        modes=modes,
        top_k=settings.top_k,
        max_retries=None,
    )
    metric_suite = metric_suite_builder()
    batch = run_evaluation(
        examples=selected_examples,
        systems=systems,
        metric_suite=metric_suite,
        repeat=args.repeat,
        generated_at=clock(),
    )
    paths = write_evaluation_artifacts(batch, args.output_dir)
    print(
        json.dumps(
            {
                **batch.manifest.model_dump(mode="json"),
                "manifest_path": str(paths.manifest),
                "results_path": str(paths.results),
            },
            indent=2,
        )
    )
    return 1 if batch.manifest.failed_run_count else 0


def _load_eval_set() -> list[EvaluationExample]:
    return load_evaluation_dataset(EVAL_DATASET_PATH, repo_root=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

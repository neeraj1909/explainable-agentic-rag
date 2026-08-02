from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from app.evaluation.cli import main as run_evaluation_cli
from app.evaluation.run_manifest import (
    EVAL_DATASET_PATH,
    REPO_ROOT,
    EvaluationExample,
    load_evaluation_dataset,
)


def load_eval_set(path: Path = EVAL_DATASET_PATH) -> list[EvaluationExample]:
    """Load the curated dataset through the offline ground-truth validator."""

    return load_evaluation_dataset(path, repo_root=REPO_ROOT)


def validate_eval_set() -> None:
    """Retained for callers of the original validation helper."""

    load_eval_set()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparative CLI while retaining this historical module path."""

    return run_evaluation_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())

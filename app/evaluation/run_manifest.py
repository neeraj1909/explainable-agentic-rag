from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
)

from app.config import AppSettings, get_settings


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DATASET_PATH = Path("app/evaluation/eval_dataset.jsonl")
MANIFEST_SCHEMA_VERSION = "1.0"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvaluationExample(StrictModel):
    user_input: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    expected_relevant_sources: list[str] = Field(min_length=1)
    expected_relevant_chunk_ids: list[str] = Field(default_factory=list)

    @field_validator("user_input", "reference")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator(
        "expected_relevant_sources",
        "expected_relevant_chunk_ids",
    )
    @classmethod
    def require_unique_labels(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("labels must not contain duplicates")
        return values


class FileDigest(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: NonNegativeInt


class CorpusManifest(StrictModel):
    docs_dir: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: list[FileDigest] = Field(min_length=1)


class DatasetManifest(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_count: PositiveInt
    source_labeled_question_count: NonNegativeInt
    chunk_labeled_question_count: NonNegativeInt


class ModelManifest(StrictModel):
    chat_model: str | None
    embedding_model: str
    temperature: float
    streaming: bool
    chat_timeout_seconds: float
    embedding_timeout_seconds: float


class RetrievalManifest(StrictModel):
    index_dir: str
    chunk_size: PositiveInt
    chunk_overlap: NonNegativeInt
    top_k: PositiveInt
    fetch_k_multiplier: PositiveInt
    reranker_enabled: bool
    reranker_model: str


class DependencyManifest(StrictModel):
    lock_path: str
    lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class GitManifest(StrictModel):
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty: bool


class EvaluationStatus(StrictModel):
    remote_run_performed: bool = False
    reason: str = "Not run: owner approval is required for model endpoint and budget."


class RunManifest(StrictModel):
    schema_version: Literal["1.0"] = MANIFEST_SCHEMA_VERSION
    generated_at: datetime
    git: GitManifest
    corpus: CorpusManifest
    dataset: DatasetManifest
    models: ModelManifest
    retrieval: RetrievalManifest
    dependencies: DependencyManifest
    evaluation: EvaluationStatus = Field(default_factory=EvaluationStatus)

    @field_validator("generated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must include a timezone")
        return value


def _resolve_repo_path(repo_root: Path, path: Path) -> Path:
    root = repo_root.resolve()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path must stay inside repository root: {path}") from exc
    return resolved


def _relative_path(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_source_path(source: str, *, repo_root: Path, row_number: int) -> None:
    pure_path = PurePosixPath(source)
    if (
        pure_path.is_absolute()
        or ".." in pure_path.parts
        or "\\" in source
        or source != pure_path.as_posix()
    ):
        raise ValueError(
            f"Question {row_number} expected source must be a safe "
            f"repository-relative path: {source}"
        )

    source_path = _resolve_repo_path(repo_root, Path(source))
    if not source_path.is_file():
        raise ValueError(
            f"Question {row_number} expected source does not exist: {source}"
        )


def load_evaluation_dataset(
    path: Path = EVAL_DATASET_PATH,
    *,
    repo_root: Path = REPO_ROOT,
) -> list[EvaluationExample]:
    dataset_path = _resolve_repo_path(repo_root, path)
    rows: list[EvaluationExample] = []

    with dataset_path.open(encoding="utf-8") as dataset_file:
        for row_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                row = EvaluationExample.model_validate(payload)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid evaluation row {row_number}: {exc}") from exc

            if row.reference.upper().startswith("TODO"):
                raise ValueError(f"Question {row_number} has a TODO reference")

            for source in row.expected_relevant_sources:
                _validate_source_path(
                    source,
                    repo_root=repo_root,
                    row_number=row_number,
                )

            rows.append(row)

    if not 10 <= len(rows) <= 20:
        raise ValueError("Evaluation dataset must contain 10-20 questions")

    user_inputs = [row.user_input for row in rows]
    if len(user_inputs) != len(set(user_inputs)):
        raise ValueError("Evaluation dataset contains a duplicate user_input")

    return rows


def _file_digest(repo_root: Path, path: Path) -> FileDigest:
    return FileDigest(
        path=_relative_path(repo_root, path),
        sha256=_sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _corpus_digest(files: list[FileDigest]) -> str:
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.path.encode())
        digest.update(b"\0")
        digest.update(item.sha256.encode())
        digest.update(b"\0")
        digest.update(str(item.size_bytes).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _read_git_metadata(repo_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status.strip())


def build_run_manifest(
    *,
    repo_root: Path = REPO_ROOT,
    settings: AppSettings | None = None,
    generated_at: datetime | None = None,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
) -> RunManifest:
    root = repo_root.resolve()
    resolved_settings = settings or get_settings()

    docs_dir = _resolve_repo_path(root, resolved_settings.docs_dir)
    corpus_files = [
        _file_digest(root, pdf_path)
        for pdf_path in sorted(docs_dir.glob("*.pdf"), key=lambda item: item.name)
    ]
    if not corpus_files:
        raise ValueError(f"No PDF corpus files found in {docs_dir}")

    dataset_path = _resolve_repo_path(root, EVAL_DATASET_PATH)
    dataset_rows = load_evaluation_dataset(dataset_path, repo_root=root)
    lock_path = _resolve_repo_path(root, Path("uv.lock"))
    if not lock_path.is_file():
        raise ValueError(f"Dependency lock file does not exist: {lock_path}")

    if git_commit is None or git_dirty is None:
        detected_commit, detected_dirty = _read_git_metadata(root)
        git_commit = detected_commit if git_commit is None else git_commit
        git_dirty = detected_dirty if git_dirty is None else git_dirty

    return RunManifest(
        generated_at=generated_at or datetime.now(UTC),
        git=GitManifest(commit=git_commit, dirty=git_dirty),
        corpus=CorpusManifest(
            docs_dir=_relative_path(root, docs_dir),
            sha256=_corpus_digest(corpus_files),
            files=corpus_files,
        ),
        dataset=DatasetManifest(
            path=_relative_path(root, dataset_path),
            sha256=_sha256_file(dataset_path),
            question_count=len(dataset_rows),
            source_labeled_question_count=sum(
                bool(row.expected_relevant_sources) for row in dataset_rows
            ),
            chunk_labeled_question_count=sum(
                bool(row.expected_relevant_chunk_ids) for row in dataset_rows
            ),
        ),
        models=ModelManifest(
            chat_model=resolved_settings.chat_model,
            embedding_model=resolved_settings.embedding_model,
            temperature=0.0,
            streaming=False,
            chat_timeout_seconds=float(resolved_settings.chat_timeout_seconds),
            embedding_timeout_seconds=float(
                resolved_settings.embedding_timeout_seconds
            ),
        ),
        retrieval=RetrievalManifest(
            index_dir=resolved_settings.index_dir.as_posix(),
            chunk_size=resolved_settings.chunk_size,
            chunk_overlap=resolved_settings.chunk_overlap,
            top_k=resolved_settings.top_k,
            fetch_k_multiplier=resolved_settings.fetch_k_multiplier,
            reranker_enabled=resolved_settings.use_reranker,
            reranker_model=resolved_settings.reranker_embedding_model,
        ),
        dependencies=DependencyManifest(
            lock_path=_relative_path(root, lock_path),
            lock_sha256=_sha256_file(lock_path),
        ),
    )


def write_manifest(manifest: RunManifest, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"{manifest.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )


def validate_manifest_file(path: Path) -> RunManifest:
    return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or validate deterministic evaluation metadata."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate a no-LLM run manifest.",
    )
    generate_parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON to this path; omit to print it.",
    )

    subparsers.add_parser(
        "validate-dataset",
        help="Validate references and expected source labels without model calls.",
    )

    validate_parser = subparsers.add_parser(
        "validate-manifest",
        help="Validate an existing manifest against its Pydantic schema.",
    )
    validate_parser.add_argument("path", type=Path)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.command == "validate-dataset":
        rows = load_evaluation_dataset()
        print(f"evaluation dataset valid: {len(rows)} questions")
        return

    if args.command == "validate-manifest":
        manifest = validate_manifest_file(args.path)
        print(
            "run manifest valid: "
            f"schema={manifest.schema_version} corpus={manifest.corpus.sha256}"
        )
        return

    manifest = build_run_manifest()
    if args.output:
        write_manifest(manifest, args.output)
        print(f"run manifest written: {args.output}")
        return

    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

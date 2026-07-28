from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.config import AppSettings
from app.evaluation.run_manifest import (
    EVAL_DATASET_PATH,
    REPO_ROOT,
    RunManifest,
    build_run_manifest,
    load_evaluation_dataset,
)


def _write_dataset(
    repo_root: Path,
    *,
    source: str = "docs/a.pdf",
    duplicate_questions: bool = False,
    todo_reference: bool = False,
) -> Path:
    dataset_path = repo_root / "app/evaluation/eval_dataset.jsonl"
    dataset_path.parent.mkdir(parents=True)

    rows = []
    for index in range(10):
        question_number = 1 if duplicate_questions else index + 1
        rows.append(
            {
                "user_input": f"Question {question_number}?",
                "reference": (
                    "TODO: add reference"
                    if todo_reference and index == 0
                    else f"Reference answer {index + 1}."
                ),
                "expected_relevant_sources": [source],
            }
        )

    dataset_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )
    return dataset_path


def _build_test_repository(tmp_path: Path) -> tuple[Path, AppSettings]:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "b.pdf").write_bytes(b"second corpus document")
    (docs_dir / "a.pdf").write_bytes(b"first corpus document")
    (tmp_path / "uv.lock").write_text("locked dependencies\n", encoding="utf-8")
    _write_dataset(tmp_path)

    settings = AppSettings(
        _env_file=None,
        chat_model="test-chat-model",
        chat_api_key="must-not-appear",
        chat_api_base="https://private-provider.invalid/v1",
        embedding_model="test-embedding-model",
        embedding_api_key="must-not-appear",
        docs_dir=Path("docs"),
        index_dir=Path(".rag-index"),
        chunk_size=512,
        chunk_overlap=64,
        top_k=6,
        fetch_k_multiplier=3,
        use_reranker=True,
        reranker_embedding_model="test-reranker-model",
    )
    return tmp_path, settings


def test_load_evaluation_dataset_validates_document_labels(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/a.pdf").write_bytes(b"document")
    dataset_path = _write_dataset(tmp_path)

    rows = load_evaluation_dataset(dataset_path, repo_root=tmp_path)

    assert len(rows) == 10
    assert rows[0].expected_relevant_sources == ["docs/a.pdf"]
    assert rows[0].expected_relevant_chunk_ids == []


def test_repository_evaluation_dataset_has_reviewed_document_ground_truth():
    expected_sources = [
        ["docs/s13278-024-01393-9.pdf"],
        ["docs/s13278-024-01393-9.pdf"],
        ["docs/T7-4.pdf"],
        ["docs/2411.12643v2.pdf"],
        ["docs/2411.12643v2.pdf"],
        ["docs/2502.03014v1.pdf"],
        ["docs/2507.08330v2.pdf"],
        ["docs/2311.08314v1.pdf"],
        ["docs/neeraj_cv.pdf"],
        [
            "docs/2411.12643v2.pdf",
            "docs/2502.03014v1.pdf",
            "docs/2507.08330v2.pdf",
            "docs/s13278-024-01393-9.pdf",
            "docs/2311.08314v1.pdf",
        ],
    ]

    rows = load_evaluation_dataset(EVAL_DATASET_PATH, repo_root=REPO_ROOT)

    assert len(rows) == 10
    assert [row.expected_relevant_sources for row in rows] == expected_sources
    assert all(not row.expected_relevant_chunk_ids for row in rows)


@pytest.mark.parametrize(
    ("dataset_options", "message"),
    [
        ({"source": "docs/missing.pdf"}, "does not exist"),
        ({"duplicate_questions": True}, "duplicate user_input"),
        ({"todo_reference": True}, "TODO reference"),
        ({"source": "../outside.pdf"}, "safe repository-relative path"),
    ],
)
def test_load_evaluation_dataset_rejects_invalid_ground_truth(
    tmp_path,
    dataset_options,
    message,
):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/a.pdf").write_bytes(b"document")
    dataset_path = _write_dataset(tmp_path, **dataset_options)

    with pytest.raises(ValueError, match=message):
        load_evaluation_dataset(dataset_path, repo_root=tmp_path)


def test_build_run_manifest_records_reproducible_inputs_without_secrets(tmp_path):
    repo_root, settings = _build_test_repository(tmp_path)
    generated_at = datetime(2026, 7, 28, 8, 30, tzinfo=UTC)

    manifest = build_run_manifest(
        repo_root=repo_root,
        settings=settings,
        generated_at=generated_at,
        git_commit="a" * 40,
        git_dirty=False,
    )

    assert manifest.schema_version == "1.0"
    assert manifest.generated_at == generated_at
    assert manifest.git.commit == "a" * 40
    assert manifest.git.dirty is False
    assert [item.path for item in manifest.corpus.files] == [
        "docs/a.pdf",
        "docs/b.pdf",
    ]
    assert manifest.dataset.question_count == 10
    assert manifest.dataset.source_labeled_question_count == 10
    assert manifest.dataset.chunk_labeled_question_count == 0
    assert manifest.models.chat_model == "test-chat-model"
    assert manifest.models.embedding_model == "test-embedding-model"
    assert manifest.models.temperature == 0.0
    assert manifest.retrieval.chunk_size == 512
    assert manifest.retrieval.chunk_overlap == 64
    assert manifest.retrieval.top_k == 6
    assert manifest.retrieval.fetch_k_multiplier == 3
    assert manifest.retrieval.reranker_enabled is True
    assert manifest.retrieval.reranker_model == "test-reranker-model"
    assert manifest.evaluation.remote_run_performed is False

    expected_lock_hash = hashlib.sha256(b"locked dependencies\n").hexdigest()
    assert manifest.dependencies.lock_sha256 == expected_lock_hash

    payload = manifest.model_dump_json()
    assert "must-not-appear" not in payload
    assert "private-provider.invalid" not in payload
    assert RunManifest.model_validate_json(payload) == manifest
    assert "properties" in RunManifest.model_json_schema()


def test_corpus_digest_changes_when_document_content_changes(tmp_path):
    repo_root, settings = _build_test_repository(tmp_path)
    build_kwargs = {
        "repo_root": repo_root,
        "settings": settings,
        "generated_at": datetime(2026, 7, 28, tzinfo=UTC),
        "git_commit": "b" * 40,
        "git_dirty": False,
    }

    before = build_run_manifest(**build_kwargs)
    (repo_root / "docs/a.pdf").write_bytes(b"changed corpus document")
    after = build_run_manifest(**build_kwargs)

    assert before.corpus.sha256 != after.corpus.sha256
    assert before.corpus.files[0].sha256 != after.corpus.files[0].sha256

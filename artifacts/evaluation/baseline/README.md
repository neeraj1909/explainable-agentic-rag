# Reproducible Pre-change Baseline

This directory freezes the inputs and configuration that existed before the
reliability and productization phases begin. It is a provenance baseline, not a
quality claim: no remote model, embedding, or RAGAS evaluation was run while
creating it.

## Frozen snapshot

The reviewed [`manifest.json`](manifest.json) is the immutable baseline
summary. Its Pydantic schema is defined in
`app/evaluation/run_manifest.py`. It records:

- the SHA-256 digest and size of every PDF plus a deterministic aggregate
  corpus digest;
- the evaluation dataset digest and label counts;
- chat, embedding, and reranker model identifiers without credentials or
  provider endpoints;
- temperature, streaming mode, timeouts, chunk size/overlap, top-k, candidate
  multiplier, reranker state, and index path;
- the `uv.lock` digest, UTC generation timestamp, and Git provenance; and
- whether a paid/networked evaluation was performed.

The recorded Git commit,
`578cc93a2a44450514f6e7a06aeaee7f0a03e78f`, is the last checked-in
pre-baseline-preparation state. `git.dirty` is deliberately `true`: the
manifest was generated while this Step 0.3 dataset/schema/documentation change
was still uncommitted. The corpus, dataset, and lock hashes capture the exact
inputs used by this checked-in artifact; the dirty flag must not be rewritten
to imply a clean worktree.

Corpus files are sorted by repository-relative path. The aggregate digest is
SHA-256 over each UTF-8 encoded
`path + NUL + file_sha256 + NUL + size_bytes + newline` record. Dataset and lock
digests are SHA-256 over their raw bytes.

## Ground truth

All ten reference answers were reviewed against the seven local PDFs. Every
question now has one or more `expected_relevant_sources`, and offline
validation confirms those repository-relative files exist.

Chunk labels remain empty. The current splitter assigns global sequential
`chunk-N` identifiers, so adding, removing, or reordering a document can change
an otherwise identical chunk's ID. Document-level labels are therefore the
honest baseline; chunk labels should be adjudicated only after stable,
content-derived chunk IDs are implemented.

## Offline validation

These commands do not construct provider clients or make model calls:

```bash
uv run python -m app.evaluation.run_manifest validate-dataset
uv run python -m app.evaluation.run_manifest \
  validate-manifest artifacts/evaluation/baseline/manifest.json
uv run pytest -q tests/test_run_manifest.py
```

Generate a candidate manifest for a future run without invoking an LLM:

```bash
uv run python -m app.evaluation.run_manifest generate
```

Compare its corpus, dataset, retrieval, model, dependency, and Git fields with
this baseline. The timestamp is expected to change. Do not overwrite this
baseline to describe a later run; store reviewed summaries separately.

## Artifact policy

Small, reviewed provenance and result summaries may be checked in. This
baseline keeps only Markdown and schema-validated JSON. Generated answers,
retrieved passages, evaluator traces, per-question tables, CSV/Parquet output,
and logs can be large, costly, or corpus-sensitive, so raw runs belong under
ignored `evaluation/`, `artifacts/evaluation/runs/`, or
`artifacts/evaluation/baseline/raw/` directories.

Before publishing any new summary, review it for corpus text, personal data,
provider details, and secrets. The manifest schema intentionally excludes API
keys and provider endpoints.

## Cost gate

The full remote-model baseline remains blocked on owner approval of the model
endpoint and budget (OA-1 in the PRP). The checked manifest therefore records
`remote_run_performed: false`; it contains no RAGAS scores and must not be cited
as performance evidence.

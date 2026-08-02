# Troubleshooting

Start with the no-cost checks below. Commands that answer questions, embed
documents, or run RAGAS can make network calls and incur provider charges.

## Confirm the environment and command surfaces

From the repository root:

```bash
uv --version
uv run python --version
uv sync
uv run python -m app.main --help
uv run python -m app.rag.cli --help
uv run python -m app.graphs.agentic_rag_graph --help
```

The project declares Python 3.11 or newer. Run modules from the repository root
so Python can resolve `app` and relative output paths consistently.

## Missing model settings or authentication failures

Copy `.env.example` to `.env`, then set:

- `LITELLM_MODEL`
- `LITELLM_API_KEY`
- `LITELLM_API_BASE` when using a proxy or compatible endpoint
- `OPENAI_API_KEY`

Chat and embeddings are separate in the current implementation.
`LITELLM_API_BASE` configures the chat client only; PDF indexing uses
`OpenAIEmbeddings` and `OPENAI_API_KEY`. A chat proxy credential therefore does
not automatically make local-PDF RAG work.

Leave `LITELLM_API_BASE` empty for direct OpenAI chat access. Check the model
name against the provider behind that endpoint. Configuration errors list the
missing variable names without printing secret values.

## Phoenix connection or trace export errors

The portable local setting is:

```dotenv
PHOENIX_ENABLED=true
PHOENIX_PROJECT_NAME=explainable-agentic-rag
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces
```

Start Phoenix in another terminal:

```bash
docker run --rm --name phoenix \
    -p 6006:6006 \
    -p 4317:4317 \
    arizephoenix/phoenix:latest
```

Then open <http://localhost:6006>. If Phoenix runs on another machine or port,
set the collector endpoint explicitly. Set `PHOENIX_ENABLED=false` when traces
should be disabled; the portable endpoint above is the runtime default.

Traces may include queries, retrieved text, and responses. Do not send private
documents to a shared collector without an explicit data-handling decision.

## No PDFs found, wrong corpus, or slow startup

List the files the loader can see:

```bash
find docs -maxdepth 1 -type f -name '*.pdf' -print
```

Only PDFs directly under `docs/` are loaded. Subdirectories are not scanned.
Every matching file is included, including `docs/neeraj_cv.pdf` in the current
repository.

Retriever construction loads, chunks, and embeds the entire corpus into a new
in-memory index. Repeated runs can therefore be slow and incur repeated
embedding charges. There is no incremental or persistent index yet.

Before adding a document, verify that you may process it with the configured
providers and that its license permits the intended use and redistribution.

## Reranker behavior is unexpected

The reranker is disabled unless this is set:

```dotenv
RAG_USE_RERANKER=true
RAG_RERANKER_EMBEDDING_MODEL=text-embedding-3-small
```

It performs a second embedding-similarity pass. It is not a cross-encoder, and
enabling it adds embedding requests. Compare measured retrieval and answer
quality before treating it as an improvement.

## Single graph requests human review

`app.graphs.agentic_rag_graph` has an interrupt-and-resume branch, but the
current command-line path automatically approves the review payload. The
non-streaming helper's resumed-result handling is also a prototype boundary.
Do not use it as a real approval control.

If this path prevents a local experiment from completing, use
`app.rag.cli --mode two-step` or `--mode agentic` while the review workflow is
being hardened. A production fix needs to persist the pending state, collect an
explicit approve/reject decision, resume the same thread, and test both paths.

## Multi-agent run uses stale state

`app.graphs.multi_agent_graph` accepts `--query` and `--thread-id`. Reusing a
thread ID can intentionally load earlier local checkpoint state.

The demo writes `.langgraph_checkpoints/multi_agent_graph.pkl`. If a code or
schema change makes old local state incompatible, stop all graph processes,
back up that file, and move it out of `.langgraph_checkpoints/` before retrying.
The directory is ignored by Git.

## Evaluation fails or returns a non-zero status

Confirm the selected run without constructing provider clients or writing an
output directory:

```bash
uv run python -m app.evaluation.run_ragas_eval \
  --modes two-step agentic graph multi-agent \
  --limit 2 \
  --output-dir evaluation/smoke \
  --dry-run
```

Without `--dry-run`, this is a paid, networked operation across the selected
questions, answer systems, and evaluator calls. The runner creates the output
directory itself and writes `manifest.json` and `results.jsonl`. A system or
metric exception is preserved in its JSONL row, increments
`failed_run_count`, and produces a non-zero process status; inspect those
artifacts before retrying.

Validate only the dataset without model calls with:

```bash
uv run python -c \
    "from app.evaluation.run_ragas_eval import validate_eval_set; validate_eval_set(); print('evaluation dataset valid')"
```

The runner targets the public RAGAS 0.4 collections API. Check the locked RAGAS
version and its migration notes before upgrading. No checked-in metric report
currently proves comparative quality.

## Tests or lint checks fail

Run the test suite:

```bash
uv run pytest -q
```

The default development dependency group includes Ruff and pytest-cov. Use
read-only checks first:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest --cov=app
```

Only add `--fix` when you intend to modify files. An undefined type in an
annotation, such as Ruff `F821`, must either be imported from its owning module,
defined locally, or quoted/guarded as a deliberate forward reference; do not
silence it without confirming the runtime type contract.

## Still blocked?

Capture the exact command, complete traceback, Python and `uv` versions, active
RAG mode, and whether Phoenix was running. Never include API keys or the
contents of `.env` in a bug report.

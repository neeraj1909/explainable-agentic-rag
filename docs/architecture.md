# Architecture

This document separates the repository's current implementation from its target
architecture. Unless a section is explicitly labeled **Target**, it describes
code that exists today.

## Current maturity

The project is a local prototype for comparing document-grounded RAG patterns.
It has multiple entry points, shared retrieval utilities, Phoenix/OpenTelemetry
instrumentation, a reproducible input manifest, and strict canonical contract
definitions. It now also has a shared application-service seam, but the existing
runtime entry points have not adopted it. It does not yet have a persistent
index, an HTTP API, or production deployment packaging.

## Current entry points

| Entry point | Data source | Orchestration | Current output |
|---|---|---|---|
| `app.main` | arXiv/web tools | LangChain tool-calling agent | `AgentResponse` |
| `app.rag.cli` (`two-step`) | `docs/*.pdf` | Retrieve once, then generate | Answer plus attributed sources |
| `app.rag.cli` (`agentic`) | `docs/*.pdf` | Model-controlled retrieval tool | Messages, tool calls, and sources |
| `app.rag.cli` (`compare`) | `docs/*.pdf` | Runs both RAG modes | Two mode-specific result objects |
| `app.graphs.agentic_rag_graph` | `docs/*.pdf` | Single typed LangGraph | Answer, sources, verifier state, retries |
| `app.graphs.multi_agent_graph` | `docs/*.pdf` | Orchestrator-led multi-agent graph | Answer, route history, sources, verifier state |
| `app.evaluation.run_ragas_eval` | Ten-question JSONL set and `docs/*.pdf` | Two-step RAG plus RAGAS evaluators | Console table and CSV |

The entry points do not currently emit the canonical response or share one error
schema. `app/contracts.py` defines the versioned target wire shape, while the
RAG CLI still selects a mode directly and no top-level router chooses among all
four local-PDF workflows.

## Current application-service seam

`app/services/rag_service.py` now provides the canonical `answer()` and
`stream()` use cases. It dispatches injected per-mode callables, creates one
run context per invocation, normalizes progress metadata, and rejects responses
whose mode or run identity conflicts with the request. `app/bootstrap.py` is the
composition root and can capture an active OpenTelemetry trace without loading
documents, constructing provider clients, or importing delivery adapters.

The retrieval and verification ports exchange canonical `EvidenceChunk` and
`VerificationSummary` values rather than LangChain document or tool payloads.
The checkpoint port deliberately exposes only generic load/save semantics keyed
by `thread_id`; the LangGraph-specific adapters and production mode wiring are
still pending. This keeps the seam independently testable while Step 1.3 moves
the existing CLI and graph implementations behind it.

## Current local-PDF data flow

```mermaid
flowchart LR
    PDF[PDF files directly under docs/] --> LOAD[PyPDFLoader]
    LOAD --> SPLIT[Chunking and stable metadata]
    SPLIT --> EMBED[OpenAI embeddings]
    EMBED --> MEMORY[InMemoryVectorStore]
    QUERY[Query] --> RETRIEVE[Attributed retriever]
    MEMORY --> RETRIEVE
    RETRIEVE --> OPTIONAL{Reranker enabled?}
    OPTIONAL -->|No| SELECT[Selected chunks]
    OPTIONAL -->|Yes| SECOND[Embedding-similarity rerank]
    SECOND --> SELECT
    SELECT --> MODE[Selected RAG or graph workflow]
    MODE --> RESULT[Mode-specific result]
```

`app/rag/retriever.py` rebuilds the index when
`build_attributed_retriever()` is called. The loader reads every `*.pdf`
directly under `docs/`; it does not recurse into subdirectories. The current
optional reranker embeds the query and candidate chunks again and compares
cosine similarity. It is not a cross-encoder.

## Current component responsibilities

- `app/config.py` owns the typed settings model, context-specific credential
  checks, and construction of the OpenAI-compatible chat client and separate
  OpenAI embeddings client.
- `app/schemas.py` retains the unchanged public `AgentResponse` used by the
  research assistant.
- `app/contracts.py` defines strict versioned request, response, evidence,
  verification, metrics, route, and progress-event models plus the explicit
  legacy response adapter.
- `app/ports/` defines narrow framework-neutral boundaries for canonical
  retrieval, verification, and checkpoint access.
- `app/services/rag_service.py` owns canonical mode dispatch, answer/stream
  entry points, and run-context consistency checks.
- `app/bootstrap.py` composes the service and supplies default run/trace
  identity without constructing network dependencies during import.
- `app/observability.py` registers Phoenix/OpenTelemetry and instruments
  LangChain calls.
- `app/rag/loaders.py` loads PDF pages and attaches source/page metadata.
- `app/rag/splitter.py` creates chunks and stable chunk metadata.
- `app/rag/vectorstore.py` builds the in-memory vector store.
- `app/rag/retriever.py` retrieves, optionally reranks, attaches attribution,
  and records retrieval spans.
- `app/rag/two_step_rag.py`, `app/rag/agentic_rag.py`, and
  `app/rag/compare.py` implement the CLI-selectable RAG modes.
- `app/graphs/agentic_rag_graph.py` implements the single graph, including
  rewrite, bounded retry, heuristic verification, and an interrupt branch.
- `app/graphs/multi_agent_graph.py` implements planner, retriever, explainer,
  verifier, routing, event streaming, and demo checkpoint persistence.
- `app/evaluation/run_ragas_eval.py` evaluates only the two-step baseline.

## Attribution and verification

Retrieved chunks can carry:

- `source`
- `page`
- `chunk_id`
- `retriever_score`
- `selected_rank`
- `reranker_score`
- `reason_selected`

The final shape depends on the selected workflow. Some paths preserve the full
attribution object, while the agentic CLI derives a smaller source list from
tool messages.

Graph verification currently calls `calculate_faithfulness_stub`, a token
overlap heuristic. Its score is useful for exercising conditional graph
routing, but it is not evidence that individual answer claims are entailed by
specific chunks.

## State, checkpoints, and human review

The single graph compiles with `InMemorySaver`, so its checkpoints live only in
the current process. It has an `interrupt()` branch and a `Command(resume=...)`
helper, but the command-line demo automatically supplies approval instead of
collecting a user decision. That is not a durable approval boundary.

The multi-agent graph combines `InMemorySaver` with LangGraph
`PersistentDict` stores and explicitly syncs them to
`.langgraph_checkpoints/multi_agent_graph.pkl`. This is useful for a local demo,
not a concurrent or production-grade state store.

## Observability

The research-assistant, RAG CLI, and graph module entry points call
`setup_phoenix_tracing()` before model or tool work. LangChain instrumentation
records model/tool spans, and the attributed retriever adds a
`rag.retrieve_with_attribution` span with retrieval metadata. The standalone
evaluation runner does not currently initialize Phoenix. The portable
development endpoint is configured through
`PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces`.
Tracing can be disabled without registering an exporter by setting
`PHOENIX_ENABLED=false`.

Traces can contain queries, retrieved document text, and model responses. Treat
the trace store as sensitive data and review it before sharing.

## Current boundaries and known gaps

- Chat and embeddings have separate credentials and endpoint behavior.
- Index construction is synchronous, paid, and repeated per retriever build.
- A checked pre-change corpus/configuration manifest exists, but the runtime
  does not yet propagate its version or enforce document-level access policy.
- There is no lexical/BM25 retrieval or rank fusion.
- There is no measured cross-encoder reranker.
- Runtime output and failure contracts vary by entry point despite the new
  canonical service and contract definitions; production mode adapters are not
  wired yet.
- Human review is not interactive or durable.
- Only the two-step mode has a RAGAS runner.
- There are no HTTP, UI, CI, Docker application, or end-to-end layers.

## Target architecture

The following diagram is a roadmap, not the current implementation.

```mermaid
flowchart TD
    CORPUS[Approved corpus and manifest] --> INGEST[Idempotent ingestion service]
    INGEST --> DENSE[Persistent dense index]
    INGEST --> LEXICAL[BM25 index]
    QUERY[Request plus mode] --> SERVICE[Application service]
    SERVICE --> FUSION[Dense and BM25 retrieval plus rank fusion]
    DENSE --> FUSION
    LEXICAL --> FUSION
    FUSION --> RERANK[Measured cross-encoder reranker]
    RERANK --> MODES{RAG mode adapter}
    MODES --> BASELINE[Two-step]
    MODES --> AGENT[Agentic]
    MODES --> SINGLE[Single graph]
    MODES --> MULTI[Multi-agent graph]
    BASELINE --> VERIFY[Claim-to-evidence verifier]
    AGENT --> VERIFY
    SINGLE --> VERIFY
    MULTI --> VERIFY
    VERIFY --> REVIEW{Approval required?}
    REVIEW -->|Yes| DURABLE[Durable human review]
    REVIEW -->|No| CONTRACT[Canonical response contract]
    DURABLE --> CONTRACT
    CONTRACT --> CLI[CLI]
    CONTRACT --> API[Thin API]
    API --> UI[Explainability UI]
    SERVICE --> OBS[Trace and experiment metadata]
    CONTRACT --> OBS
```

### Intended delivery order

1. Establish reproducible quality tooling, configuration, canonical schemas,
   errors, and an application-service seam.
2. Build a comparative harness and collect baseline evidence for all four
   modes.
3. Introduce claim-to-evidence verification and recalibrate routing with
   measured thresholds.
4. Add approved-corpus ingestion, persistent dense and BM25 indexes, fusion,
   and a benchmarked cross-encoder.
5. Add durable human review, then expose the stable contract through an API and
   UI.
6. Add trace-linked experiment evidence, CI, container packaging, security
   checks, and end-to-end validation.

Each target stage should preserve a runnable baseline and be accepted with
tests plus recorded evaluation evidence before the next layer is treated as
shipped.

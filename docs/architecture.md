# Architecture

This document separates the repository's current implementation from its target
architecture. Unless a section is explicitly labeled **Target**, it describes
code that exists today.

## Current maturity

The project is a local prototype for comparing document-grounded RAG patterns.
It has multiple entry points, shared retrieval utilities, Phoenix/OpenTelemetry
instrumentation, a reproducible input manifest, and strict canonical contract
definitions. Existing command-line entry points now use a shared
application-service seam and canonical mode adapters. A comparative evaluation
CLI drives the same service for all four local-PDF modes. It does not yet have
a persistent index, an HTTP API, or production deployment packaging.

## Current entry points

| Entry point | Data source | Orchestration | Current output |
|---|---|---|---|
| `app.main` | arXiv/web tools | LangChain tool-calling agent | Canonical `RAGResponse` or readable research view |
| `app.rag.cli` (`two-step`) | `docs/*.pdf` | Retrieve once, then generate | Canonical `RAGResponse` or readable view |
| `app.rag.cli` (`agentic`) | `docs/*.pdf` | Model-controlled retrieval tool | Canonical `RAGResponse` or readable view |
| `app.rag.cli` (`graph`) | `docs/*.pdf` | Single typed LangGraph | Canonical response/events or readable view |
| `app.rag.cli` (`multi-agent`) | `docs/*.pdf` | Orchestrator-led multi-agent graph | Canonical response/events or readable view |
| `app.rag.cli` (`compare`) | `docs/*.pdf` | Runs two-step and agentic through one service | Two canonical result objects plus measured latency |
| `app.graphs.agentic_rag_graph` | `docs/*.pdf` | Single typed LangGraph | Canonical response/events or readable view |
| `app.graphs.multi_agent_graph` | `docs/*.pdf` | Orchestrator-led multi-agent graph | Canonical response/events or readable view |
| `app.evaluation.run_ragas_eval` | Ten-question JSONL set and `docs/*.pdf` | Selected local-PDF modes through `RAGService`, then public RAGAS metrics | JSON manifest and per-run JSONL rows |

All runtime CLIs emit the versioned canonical response with `--json`; streaming
paths emit canonical progress events followed by exactly one response. Public
error normalization is still missing. The caller selects a mode explicitly;
no top-level supervisor automatically chooses among the four local-PDF
workflows.

## Current application-service seam

`app/services/rag_service.py` now provides the canonical `answer()` and
`stream()` use cases. It dispatches injected per-mode callables, creates one
run context per invocation, normalizes progress metadata, and rejects responses
whose mode or run identity conflicts with the request. `app/bootstrap.py` is the
composition root: it shares an LLM and attributed retriever across selected
modes, builds graph nodes and checkpointers, and captures an active
OpenTelemetry trace. Importing the module alone does not load documents or
construct provider clients.

The retrieval and verification ports exchange canonical `EvidenceChunk` and
`VerificationSummary` values rather than LangChain document or tool payloads.
The checkpoint port deliberately exposes only generic load/save semantics keyed
by `thread_id`. Current graph composition accepts injected native LangGraph
checkpointers; a durable application adapter remains future work. Concrete mode
adapters translate LangChain/LangGraph values at the service boundary and can
be tested with fakes.

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
    SELECT --> MODE[Selected RAG or graph adapter]
    MODE --> RESULT[Canonical response]
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
  identity, shared runtime dependencies, graph nodes, and checkpointers.
- `app/observability.py` registers Phoenix/OpenTelemetry and instruments
  LangChain calls.
- `app/rag/loaders.py` loads PDF pages and attaches source/page metadata.
- `app/rag/splitter.py` creates chunks and stable chunk metadata.
- `app/rag/vectorstore.py` builds the in-memory vector store.
- `app/rag/retriever.py` retrieves, optionally reranks, attaches attribution,
  and records retrieval spans.
- `app/rag/two_step_rag.py`, `app/rag/agentic_rag.py`, and
  `app/rag/compare.py` implement the CLI-selectable RAG modes.
- `app/rag/mode_adapters.py` translates existing workflow values into
  `RAGResponse` and `ProgressEvent` without leaking framework-specific payloads
  through the service boundary.
- `app/graphs/agentic_rag_graph.py` implements the single graph, including
  rewrite, bounded retry, heuristic verification, and an interrupt branch.
- `app/graphs/multi_agent_graph.py` implements planner, retriever, explainer,
  verifier, routing, event streaming, and demo checkpoint persistence.
- `app/evaluation/systems.py` adapts selected local-PDF modes to evaluation
  questions while sharing one canonical `RAGService`.
- `app/evaluation/metrics.py` binds samples to the public RAGAS 0.4 collections
  API and separate native chat/embedding clients.
- `app/evaluation/cli.py` owns deterministic run ordering, failure rows,
  manifests, JSONL artifacts, dry-run planning, limits, and repetitions.
- `app/evaluation/run_ragas_eval.py` retains the historical module path as a
  thin compatibility entry point.

## Attribution and verification

Retrieved chunks can carry:

- `source`
- `page`
- `chunk_id`
- `retriever_score`
- `retriever_rank`
- `selected_rank`
- `reranker_score`
- `reason_selected`

Every workflow maps available attribution into canonical `EvidenceChunk`
objects. The agentic adapter derives these chunks from retrieval-tool messages;
missing metadata stays explicitly absent rather than being fabricated.

Graph verification currently calls `calculate_faithfulness_stub`, a token
overlap heuristic. Its score is useful for exercising conditional graph
routing, but it is not evidence that individual answer claims are entailed by
specific chunks.

## State, checkpoints, and human review

The single graph compiles with `InMemorySaver`, so its checkpoints live only in
the current process. It has an `interrupt()` branch and a `Command(resume=...)`
helper. The service returns a canonical `human_review` next action when the
graph interrupts, but the CLI does not yet collect a decision or durably resume
the run.

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
- Runtime success responses and progress events are canonical; public failure
  normalization is not yet implemented.
- Evaluation still uses the legacy two-step callable rather than selecting all
  four modes through `RAGService`.
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

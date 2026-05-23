from langchain_core.documents import Document
from openinference.semconv.trace import DocumentAttributes, SpanAttributes

from app.rag import retriever
from app.tools import attribution_tools


def _patch_retriever_dependencies(monkeypatch):
    vectorstore = object()
    monkeypatch.setattr(retriever, "load_pdf_documents", lambda docs_dir: ["doc"])
    monkeypatch.setattr(retriever, "split_documents", lambda documents: ["chunk"])
    monkeypatch.setattr(retriever, "build_vectorstore", lambda chunks: vectorstore)
    return vectorstore


def test_build_attributed_retriever_does_not_construct_reranker_when_disabled(monkeypatch):
    vectorstore = _patch_retriever_dependencies(monkeypatch)
    monkeypatch.setenv("RAG_USE_RERANKER", "false")

    class ExplodingReranker:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Reranker should not be constructed when disabled")

    monkeypatch.setattr(retriever, "Reranker", ExplodingReranker)

    result = retriever.build_attributed_retriever(k=5)

    assert result.vectorstore is vectorstore
    assert result.reranker is None
    assert result.fetch_k == 5


def test_build_attributed_retriever_constructs_reranker_with_env_model(monkeypatch):
    _patch_retriever_dependencies(monkeypatch)
    monkeypatch.setenv("RAG_USE_RERANKER", "true")
    monkeypatch.setenv("RAG_RERANKER_EMBEDDING_MODEL", "custom-reranker-embedding")
    constructed = []

    class RecordingReranker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            constructed.append(self)

    monkeypatch.setattr(retriever, "Reranker", RecordingReranker)

    result = retriever.build_attributed_retriever(k=5)

    assert result.reranker is constructed[0]
    assert constructed[0].kwargs["model_name"] == "custom-reranker-embedding"
    assert result.fetch_k == 20


def test_attach_retrieval_attribution_without_reranker_adds_metadata():
    doc = Document(
        page_content="This project explains agentic RAG with evidence.",
        metadata={"source": "docs/example.pdf", "chunk_id": "chunk-1", "page": 3},
    )

    result = attribution_tools.attach_retrieval_attribution(
        query="What is this project about?",
        docs_and_scores=[(doc, 0.73)],
        k=1,
        reranker=None,
    )

    assert len(result) == 1
    assert result[0].metadata["retriever_score"] == 0.73
    assert result[0].metadata["reranker_score"] is None
    assert result[0].metadata["selected_rank"] == 1
    assert result[0].metadata["retrieval_attribution"]["chunk_id"] == "chunk-1"


def test_set_retrieval_span_attributes_serializes_retrieval_metadata():
    class RecordingSpan:
        def __init__(self):
            self.attributes = {}

        def set_attribute(self, key, value):
            self.attributes[key] = value

    span = RecordingSpan()
    doc = Document(
        page_content="Relevant project overview text",
        metadata={
            "chunk_id": "chunk-1",
            "retrieval_attribution": {
                "chunk_id": "chunk-1",
                "retriever_score": 0.42,
                "reason_selected": "test reason",
            },
        },
    )

    attribution_tools.set_retrieval_span_attributes(span, query="project?", docs=[doc])

    metadata_key = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.0.{DocumentAttributes.DOCUMENT_METADATA}"
    assert '"chunk_id": "chunk-1"' in span.attributes[metadata_key]

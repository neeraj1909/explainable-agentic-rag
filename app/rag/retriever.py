from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from opentelemetry import trace
from pydantic import ConfigDict

from app.config import get_settings
from app.rag.config import TOP_K
from app.rag.loaders import load_pdf_documents
from app.rag.splitter import split_documents
from app.rag.vectorstore import build_vectorstore
from app.tools.attribution_tools import (
    Reranker,
    attach_retrieval_attribution,
    set_retrieval_span_attributes,
)


class AttributedVectorStoreRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: Any
    k: int = TOP_K
    fetch_k: int = TOP_K * 4
    reranker: Any | None = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("rag.retrieve_with_attribution") as span:
            span.set_attribute("retrieval.k", self.k)
            span.set_attribute("retrieval.fetch_k", self.fetch_k)
            span.set_attribute("retrieval.reranker.enabled", self.reranker is not None)

            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query,
                k=self.fetch_k,
            )

            docs = attach_retrieval_attribution(
                query=query,
                docs_and_scores=docs_and_scores,
                k=self.k,
                reranker=self.reranker,
            )

            set_retrieval_span_attributes(span, query=query, docs=docs)

            return docs


def build_attributed_retriever(k: int = TOP_K) -> AttributedVectorStoreRetriever:
    settings = get_settings()
    documents = load_pdf_documents(settings.docs_dir)
    chunks = split_documents(documents)
    vectorstore = build_vectorstore(chunks)

    if settings.use_reranker:
        settings.require_embedding_credentials()
        reranker = Reranker(
            model_name=settings.reranker_embedding_model,
            openai_api_key=settings.embedding_api_key.get_secret_value(),
            timeout=float(settings.embedding_timeout_seconds),
        )
    else:
        reranker = None

    fetch_k = max(k * settings.fetch_k_multiplier, k) if settings.use_reranker else k

    return AttributedVectorStoreRetriever(
        vectorstore=vectorstore,
        k=k,
        fetch_k=fetch_k,
        reranker=reranker,
    )

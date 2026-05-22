"""Attribution helpers for Day-2 retrieval evidence scoring.

Day 2 will add source/chunk attribution, retriever scores, reranker scores, and
reason-selected explanations here.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings
from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import CrossEncoder 


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "with",
}


@dataclass(frozen=True)
class RetrievalAttribution:   
    source: str | None 
    chunk_id: str | None 
    page: int | None 
    retriever_rank: int 
    selected_rank: int
    retriever_score: float | None 
    reranker_score: float | None 
    reason_selected: str
    
    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)
    
    
class Reranker:
    """
    Langchain-only embedding reranker.
    Uses OpenAI embeddings through LangChain.
    
    reranker_score = cosine_similarity(query_embedding, chunk_embedding)
    
    Higher score means more relevant.
    """
    
    def __init__(
        self,
        embeddings: Embeddings | None = None,
        model_name: str = "text-embedding-3-small",
        dimensions: int | None = None,
        max_chars_per_doc: int = 6000,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
    ) -> None:
        self.max_chars_per_doc = max_chars_per_doc
        
        if embeddings is None:
            kwargs: dict[str, Any] = {
                "model": model_name,
            }
            
            if dimensions is not None:
                kwargs['dimensions'] = dimensions
                
            if openai_api_key is not None:
                kwargs['api_key'] = openai_api_key
                
            if openai_base_url is not None:
                kwargs['base_url'] = openai_base_url

            embeddings = OpenAIEmbeddings(**kwargs)
            
        self.embeddings = embeddings
        self.embed_query_chain = RunnableLambda(
            self._embed_query
        ).with_config(
            {
                "run_name": "reranker_embed_query",
                "tags": ["retrieval-attribution", "openai-embeddings"],
            }
        )
        
        self.embed_documents_chain = RunnableLambda(
            self._embed_documents
        ).with_config(
            {
                "run_name": "reranker_embed_documents",
                "tags": ["retrieval-attribution", "openai-embeddings"],
            }
        )
        
    def _embed_query(self, query: str) -> list[float]:
        return self.embeddings.embed_query(query)
    
    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(texts)
         
    def score(self, query: str, docs: list[Document]) -> list[float]:
        """Return one reranker score per document."""
        
        if not docs:
            return []
        
        chunk_texts = [
            _normalize_text(doc.page_content)[: self.max_chars_per_doc]
            for doc in docs
        ]
        
        query_embedding = self.embed_query_chain.invoke(query)
        doc_embeddings = self.embed_documents_chain.invoke(chunk_texts)
        
        scores = [
            _cosine_similarity(query_embedding, doc_embedding)
            for doc_embedding in doc_embeddings
        ]    
        
        if len(scores) != len(docs):
            raise ValueError(
                f"Reranker returned {len(scores)} scores for {len(docs)} docs."
            )
            
        return scores


def attach_retrieval_attribution(
    *,
    query: str,
    docs_and_scores: list[tuple[Document, float]],
    k: int,
    reranker: Reranker | None = None,
) -> list[Document]:
    """
    Attach attribution metadata to retrieved chunks.
    
    Input should usually come from:
        vectorstore.similarity_search_with_score(query, k=fetch_k)
        
    Adds these metadata fields:
        retriever_score
        reranker_score
        retriever_rank
        selected_rank
        reason_selected
        retrieval_attribution
    """
    
    if not docs_and_scores:
        return []
    
    docs = [doc for doc, _ in docs_and_scores]
    
    if reranker is not None and docs:
        reranker_scores = [
            _safe_float(score) 
            for score in reranker.score(query=query, docs=docs)
        ]
    else:
        reranker_scores = [None] * len(docs)
        
    candidates = list[tuple[Document, float | None, float | None, int]] = []
    
    for retriever_rank,((doc, retriever_score), reranker_score) in enumerate(
        zip(docs_and_scores, reranker_scores, strict=False),
        start=1,
    ):
        candidates.append(
            (
                doc,
                _safe_float(retriever_score),
                _safe_float(reranker_score),
                retriever_rank,
            )
        )
        
    candidates = _sort_candidates(candidates, use_reranker=reranker is not None)
    
    attributed_docs: list[Document] = []
    
    for selected_rank, candidate in enumerate(candidates[:k], start=1):
        doc, retriever_score, reranker_score, retriever_rank = candidate
        
        attribution = RetrievalAttribution(
            source=_metadata_str(doc, "source"),
            chunk_id=_metadata_str(doc, "chunk_id") or doc.id,
            page=_metadata_int(doc, "page"),
            retriever_rank=retriever_rank,
            selected_rank=selected_rank,
            retriever_score=retriever_score,
            reranker_score=reranker_score,
            reason_selected=build_reason_selected(
                query=query,
                doc=doc,
                retriever_rank=retriever_rank,
                selected_rank=selected_rank,
                retriever_score=retriever_score,
                reranker_score=reranker_score,
            ),
        )
        
        metadata = dict(doc.metadata)
        metadata.update(
            {
                "retriever_score": attribution.retriever_score,
                "reranker_score": attribution.reranker_score,
                "retriever_rank": attribution.retriever_rank,
                "selected_rank": attribution.selected_rank,
                "reason_selected": attribution.reason_selected,
                "retrieval_attribution": attribution.to_metadata(),
            }
        )
        
        attributed_docs.append(
            Document(
                id=doc.id,
                page_content=doc.page_content,
                metadata=metadata,
            )
        )
        
    return attributed_docs


def build_reason_selected(
    *,
    query: str,
    doc: Document,
    retriever_rank: int,
    selected_rank: int,
    retriever_score: float | None,
    reranker_score: float | None,
) -> str:
    
    if reranker_score is not None:
        reason = (
            f"Initially retrieved at rank #{retriever_rank}, then selected at "
            f"rank #{selected_rank} after LangChain OpenAI embedding reranking"
        )
    else:
        reason = (
            f"Selected at rank #{selected_rank} from initial vector retrieval; "
            f"no reranker configured"
        )
        
    score_parts = []
    
    if retriever_score is not None:
        score_parts.append(f"retriever_score={retriever_score:.4f}")
    
    if reranker_score is not None:
        score_parts.append(f"reranker_score={reranker_score:.4f}") 
        
    if score_parts:
        reason += " (" + ", ".join(score_parts) + ")"
        
    overlaps = _overlap_terms(query, doc.page_content)
    
    if overlaps:
        reason += f". Query terms present in chunk: {', '.join(overlaps)}."
    else:
        reason += ". Selected by embedding similarity; no exact query-term overlap detected."
        
    return reason


def document_attribution_payload(doc: Document) -> dict[str, Any]:
    attribution = doc.metadata.get("retrieval_attribution")
    
    if isinstance(attribution, dict):
        return attribution
    
    return {
        "source": doc.metadata.get("source"),
        "chunk_id": doc.metadata.get("chunk_id") or doc.id,
        "page": doc.metadata.get("page"),
        "retriever_rank": doc.metadata.get("retriever_rank"),
        "selected_rank": doc.metadata.get("selected_rank"),
        "retriever_score": doc.metadata.get("retriever_score"),
        "reranker_score": doc.metadata.get("reranker_score"),
        "reason_selected": doc.metadata.get("reason_selected"),
    }
    
    
def document_attributions(docs: list[Document]) -> list[dict[str, Any]]:
    return [document_attribution_payload(doc) for doc in docs]


def format_attributed_context(docs: list[Document]) -> str:
    blocks = []
    
    for doc in docs:
        attribution = document_attribution_payload(doc)
        
        blocks.append(
            "["
            f"source={attribution.get('source')} "
            f"chunk_id={attribution.get('chunk_id')} "
            f"page={attribution.get('page')} "
            f"retriever_rank={attribution.get('retriever_rank')} "
            f"selected_rank={attribution.get('selected_rank')} "
            f"retriever_score={attribution.get('retriever_score')} "
            f"reranker_score={attribution.get('reranker_score')} "
            f"reason_selected={attribution.get('reason_selected')} "
            "]\n"
            f"{doc.page_content}"
        )
        
    return "\n\n".join(blocks)


def _sort_candidates(
    candidates: list[tuple[Document, float | None, float | None, int]],
    *,
    use_reranker: bool,
) -> list[tuple[Document, float | None, float | None, int]]:
    
    if use_reranker:
        return sorted(
            candidates,
            key=lambda item: (
                item[2] if item[2] is not None else float("-inf"),
                item[1] if item[1] is not None else float("-inf"),
            ),
            reverse=True,
        )
        
    return sorted(
        candidates,
        key=lambda item: item[1] if item[1] is not None else float("-inf"),
        reverse=True,
    )
    
    
def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    if len(vector_a) != len(vector_b):
        raise ValueError(
            f"Embedding dimensions differ: {len(vector_a)} != {len(vector_b)}"
        )
        
    dot_product = sum( a * b for a, b in zip(vector_a, vector_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def _safe_float(value: ANy) -> float | None:
    if value is None:
        return None
    
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    
    if math.isnan(number) or math.isinf(number):
        return None
    
    return number


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _query_terms(query: str) -> list[str]:
    terms = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", query.lower())
    return [term for term in terms if term not in STOPWORDS]


def _overlap_terms(query: str, text: str, limit: int = 8) -> list[str]:
    text_lower = text.lower()
    overlaps: list[str] = []
    
    for term in _query_terms(query):
        if term in text_lower and term not in overlaps:
            overlaps.append(term)
            
        if len(overlaps) >= limit:
            break
        
    return overlaps


def _metadata_str(doc: Document, key: str) -> str | None:
    value = doc.metadata.get(key)
    
    if value is None:
        return None
    
    return str(value)


def _metadata_int(doc: Document, key: str) -> int | None:
    value = doc.metadata.get(key)
    
    if value is None:
        return None
    
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
    

def set_retrieval_span_attributes(span: Any, *, query: str, docs: list[Document]) -> None:
    span.set_attribute(
        SpanAttributes.OPENINFERENCE_SPAN_KIND,
        OpenInferenceSpanKindValues.RETRIEVER.value,
    )
    span.set_attribute(SpanAttributes.INPUT_VALUE, query)
    
    for index, doc in enumerate(docs):
        attribution = doc.metadata.get("retrieval_attribution", {})
        document_score = attribution.get("reranker_score") or attribution.get("retriever_score")
        
        prefix = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{index}"
        
        span.set_attribute(
            f"{prefix}.{DocumentAttributes.DOCUMENT_ID}",
            str(attribution.get("chunk_id") or doc.metadata.get("chunk_id")),
        )
        
        if document_score is not None:
            span.set_attribute(
                f"{prefix}.{DocumentAttributes.DOCUMENT_SCORE}",
                float(document_score),
            )
            
        span.set_attribute(
            f"{prefix}.{DocumentAttributes.DOCUMENT_CONTENT}",
            doc.page_content[:4000],
        )
        
        span.set_attribute(
            f"{prefix}.{DocumentAttributes.DOCUMENT_METADATA}",
            json.dumps(attribution, ensure_ascii=False, default=str),
        )

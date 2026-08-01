from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from app.config import get_llm_client
from app.rag.config import TOP_K
from app.rag.mode_adapters import TwoStepModeAdapter
from app.rag.prompts import rag_prompt
from app.rag.retriever import build_attributed_retriever


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        (
            f"[source={doc.metadata.get('source')} "
            f"chunk={doc.metadata.get('chunk_id')} "
            f"page={doc.metadata.get('page')} "
            f"retriever_score={doc.metadata.get('retriever_score')} "
            f"reranker_score={doc.metadata.get('reranker_score')} "
            f"selected_rank={doc.metadata.get('selected_rank')} "
            f"reason_selected={doc.metadata.get('reason_selected')}]\n"
            f"{doc.page_content}"
        )
        for doc in docs
    )


def build_two_step_mode(
    k: int = TOP_K,
    *,
    retriever=None,
    llm=None,
    answer_chain=None,
) -> TwoStepModeAdapter:
    """Build the canonical mode adapter from injected or default dependencies."""

    resolved_retriever = (
        retriever if retriever is not None else build_attributed_retriever(k=k)
    )
    resolved_chain = answer_chain
    if resolved_chain is None:
        resolved_llm = llm if llm is not None else get_llm_client()
        resolved_chain = rag_prompt | resolved_llm | StrOutputParser()
    return TwoStepModeAdapter(resolved_retriever, resolved_chain)


def build_two_step_rag(k: int = TOP_K, *, retriever=None, llm=None):
    """Retain the legacy callable while routing its work through the adapter."""

    mode = build_two_step_mode(k=k, retriever=retriever, llm=llm)

    def answer(question: str) -> dict:
        from app.contracts import RAGMode, RAGRequest
        from app.services.rag_service import RunContext

        response = mode.answer(
            RAGRequest(query=question, mode=RAGMode.two_step, top_k=k),
            RunContext(run_id="legacy-two-step"),
        )
        payload = response.model_dump(mode="json")
        payload["mode"] = "two_step_rag"
        payload["sources"] = payload.pop("evidence")
        return payload

    return answer

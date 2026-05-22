import json

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config import get_llm_client
from app.rag.config import TOP_K
from app.rag.retriever import build_attributed_retriever


class RetrieveArgs(BaseModel):
    query: str = Field(description="Search query for the local document corpus.")
    k: int = Field(
        default=TOP_K,
        ge=1,
        le=20,
        description="Number of chunks to return from the retriever.",
    )


def make_retrieval_tool(k: int = TOP_K):
    retriever = build_attributed_retriever(k=k)

    @tool("retrieve_documents", args_schema=RetrieveArgs)
    def retrieve_documents(query: str, k: int = TOP_K) -> str:
        """
        Retrieve relevant document chunks from the local knowledge base.

        Use this when the user asks a factual question that requires evidence
        from the indexed documents. Do not use for small talk or general advice
        that does not require the local corpus.
        """
        docs = retriever.invoke(query)
        limit = min(k, len(docs))

        results = []
        for doc in docs[:limit]:
            results.append(
                {
                    "source": doc.metadata.get("source"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "page": doc.metadata.get("page"),
                    "retriever_score": doc.metadata.get("retriever_score"),
                    "reranker_score": doc.metadata.get("reranker_score"),
                    "selected_rank": doc.metadata.get("selected_rank"),
                    "reason_selected": doc.metadata.get("reason_selected"),
                    "content": doc.page_content,
                }
            )

        return json.dumps(
            {
                "query": query,
                "retrieved_count": len(results),
                "results": results,
            },
            indent=2,
        )

    return retrieve_documents


def build_agentic_rag(k: int = TOP_K):
    llm = get_llm_client()
    retrieval_tool = make_retrieval_tool(k=k)

    system_prompt = """
You are an agentic RAG assistant.

You may answer directly only when retrieval is not needed.

Use the retrieve_documents tool when:
- the user asks about facts from the document corpus
- the answer needs citations
- you are unsure and need evidence
- the question is specific, factual, or document-grounded

After using retrieved evidence:
- cite source, chunk_id, and page when available
- do not invent facts
- if evidence is insufficient, say so
- you may call retrieve_documents again with a better query if needed

Prefer concise, grounded answers.
"""

    return create_agent(
        model=llm,
        tools=[retrieval_tool],
        system_prompt=system_prompt,
    )

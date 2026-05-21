from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from app.config import get_llm_client
from app.rag.config import TOP_K
from app.rag.prompts import rag_prompt
from app.rag.retriever import build_retriever


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[source={doc.metadata.get('source')} "
        f"chunk={doc.metadata.get('chunk_id')} "
        f"page={doc.metadata.get('page')}]\n"
        f"{doc.page_content}"
        for doc in docs
    )
    
    
def build_two_step_rag(k: int = TOP_K):
    retriever = build_retriever(k=k)    
    llm = get_llm_client()
    chain = rag_prompt | llm | StrOutputParser()
    
    def answer(question: str) -> dict:
        docs = retriever.invoke(question)
        context = format_context(docs)
        
        response = chain.invoke(
            {
                "question": question,
                "context": context,
            }
        )
        
        return {
            "mode": "two_step_rag",
            "answer": response,
            "sources": [
                {
                    "source": doc.metadata.get("source"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "page": doc.metadata.get("page"),
                }
                for doc in docs
            ],
        }
        
    return answer       

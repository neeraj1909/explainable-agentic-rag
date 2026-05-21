from langchain_core.vectorstores import VectorStoreRetriever

from app.rag.config import DOCS_DIR, TOP_K
from app.rag.loaders import load_pdf_documents
from app.rag.splitter import split_documents
from app.rag.vectorstore import build_vectorstore


def build_retriever(k: int = TOP_K) -> VectorStoreRetriever:
    documents = load_pdf_documents(DOCS_DIR)
    chunks = split_documents(documents)
    vectorstore = build_vectorstore(chunks)
    
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )
    
    return retriever

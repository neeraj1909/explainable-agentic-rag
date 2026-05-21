from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from app.config import get_embedding_client


def build_vectorstore(chunks: list[Document]) -> InMemoryVectorStore:
    """Build a vector store from a given chunks."""
    embeddings = get_embedding_client()
    
    vectorstore = InMemoryVectorStore.from_documents(
        documents=chunks, 
        embedding=embeddings
    )
    
    return vectorstore

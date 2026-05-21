from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    
    chunks = splitter.split_documents(documents)
    
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk-{index + 1}"
        
    return chunks

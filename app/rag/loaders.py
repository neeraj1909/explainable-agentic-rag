from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def load_pdf_documents(doc_dir: Path) -> list[Document]:
    """Load PDF documents from a directory."""
    documents: list[Document] = []
    
    for pdf_path in doc_dir.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        pdf_docs = loader.load()
        
        for doc in pdf_docs:
            doc.metadata["source"] = str(pdf_path)
            doc.metadata["doc_id"] = pdf_path.stem
        
        documents.extend(pdf_docs)
        
    return documents

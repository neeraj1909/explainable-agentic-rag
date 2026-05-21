from langchain_core.prompts import ChatPromptTemplate


rag_prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful RAG assistant.
    
    Answer the question using only the context below. 
    If the answer is not present in the context, say: "I don't know based on the provided documents."
    
    Always cite sources using the source and chunk_id metadata.
    
    Question:
    {question}
    
    Context:
    {context}
    
    Answer:
    """
)
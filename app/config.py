import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


def get_llm_client():
    return ChatOpenAI(
        model=os.environ.get("LITELLM_MODEL"),
        api_key=os.environ.get("LITELLM_API_KEY"),
        base_url=os.environ.get("LITELLM_API_BASE"),
        streaming=os.environ.get("LITELLM_STREAMING", "true").lower() == "true",
    )

def get_embedding_client():
    # return OpenAIEmbeddings(
    #     model=os.environ.get("LITELLM_EMBEDDING_MODEL"),
    #     api_key=os.environ.get("LITELLM_API_KEY"),
    #     base_url=os.environ.get("LITELLM_API_BASE"),
    # )
    
    return HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )

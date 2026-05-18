import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm_client():
    return ChatOpenAI(
        model=os.environ.get("LITELLM_MODEL"),
        api_key=os.environ.get("LITELLM_API_KEY"),
        base_url=os.environ.get("LITELLM_API_BASE"),
    )

# Create a minimal agent with one LLM and 2–3 Python tools. Use tools related to your 
# profile, such as search_papers, summarize_claim, and calculate_faithfulness_stub.
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model=os.environ.get("LITELLM_MODEL"),
    api_key=os.environ.get("LITELLM_API_KEY"),
    base_url=os.environ.get("LITELLM_API_BASE"),
)

def search_papers(query):
    """Placeholder function to simulate searching for papers"""
    return f"Search results for '{query}'"

def summarize_claim(claim):
    """Placeholder function to simulate summarizing a claim"""
    return f"Summary of claim: '{claim}'"

def calculate_faithfulness_stub():
    """Placeholder function to simulate calculating faithfulness"""
    return "Faithfulness score: 0.85"

# Create the agent with the specified tools
agent = create_agent(
    model=llm,
    tools=[search_papers, summarize_claim, calculate_faithfulness_stub],
    system_prompt="You are a helpful assistant",
)

# Example usage of the agent
query = "What are the latest papers on AI ethics?"
result = agent.invoke(
    {"messages": [{"role": "user", "content": query}]}
)

print(result["messages"][-1].content_blocks[0])

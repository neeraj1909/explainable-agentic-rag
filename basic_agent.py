import argparse
import json 

from dotenv import load_dotenv
from langchain.agents import create_agent

from faithfulness import calculate_faithfulness_stub as calculate_faithfulness
from llm_client import get_llm_client
from summarize_results import summarize_claim
from search_papers import search_papers


def build_agent():
    load_dotenv()
    llm = get_llm_client()
    
    def summarize_search_results(search_result_json: str) -> str:
        """Summarize JSON arXiv search results"""
        search_result = json.loads(search_result_json)
        return summarize_claim(search_result, llm)

    system_prompt = (
        "You are a research assistant. Search for papers before answering. "
        "Ground claims in retrieved titles, abstracts, and URLs. "
        "Do not invent papers or findings. If evidence is weak, say so. "
        "Use the faithfulness tool to check whether your final answer is supported."
    )

    # Create the agent with the specified tools
    agent = create_agent(
        model=llm,
        tools=[
            search_papers, 
            summarize_search_results, 
            calculate_faithfulness,
        ], 
        system_prompt=system_prompt,
    )
    
    return agent


def main():
    parser = argparse.ArgumentParser(description="Run the research assistant agent.")
    parser.add_argument("query", nargs="+", help="User search query")
    args = parser.parse_args()
    
    agent = build_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": " ".join(args.query)}]}
    )
    print(result["messages"][-1].content)
    

if __name__ == "__main__":
    main()
 
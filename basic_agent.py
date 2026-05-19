import argparse
import json 

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from schemas import AgentResponse

from faithfulness import calculate_faithfulness_stub as calculate_faithfulness
from llm_client import get_llm_client
from summarize_results import summarize_claim
from search_papers import search_papers


def build_agent(max_results: int = 5):
    load_dotenv()
    llm = get_llm_client()
    
    def search_papers_with_limit(query: str) -> str:
        """Search arXiv papers using the configured max_results limit."""
        return search_papers(query, max_results=max_results)
    
    def summarize_search_results(search_result_json: str) -> str:
        """Summarize JSON arXiv search results"""
        search_result = json.loads(search_result_json)
        return summarize_claim(search_result, llm)

    system_prompt = (
        "You are a research assistant. Search for papers before answering. "
        "Ground claims in retrieved titles, abstracts, and URLs. "
        "Do not invent papers or findings. If evidence is weak, say so. "
        "Use the faithfulness tool to check whether your final answer is supported."
        "Your final response must follow the AgentResponse schema. "
        "Only include sources in sources_used if they were actually retrieved. "
        "If a claim is not supported by the retrieved evidence, include it in unsupported_claims. "
        "Set confidence between 0 and 1 based on evidence strength and faithfulness. "
        "Use next_action='retrieve_more' when evidence is weak, "
        "'ask_clarifying_question' when the query is ambiguous, "
        "'human_review' when the answer may be risky or unsupported, "
        "and 'no_follow_up_needed' when the answer is sufficiently supported." 
    )

    # Create the agent with the specified tools
    agent = create_agent(
        model=llm,
        tools=[
            search_papers_with_limit,
            summarize_search_results,
            calculate_faithfulness,
        ], 
        system_prompt=system_prompt,
        response_format =ToolStrategy(AgentResponse),
    )
    
    return agent


def format_agent_response(response: AgentResponse) -> str:
    """Format the agent's structured response into a readable string."""
    confidence_percent = round(response.confidence * 100, 2)
    
    lines = []
    
    lines.append("\n" + "=" * 80)
    lines.append("Research Assistant Response")
    lines.append("=" * 80)
    
    lines.append("\nAnswer:")
    lines.append(f"{response.answer}\n")
    
    lines.append(f"Confidence: {confidence_percent}%\n")
    
    lines.append("\nSources Used:")
    if response.sources_used:
        for index, source in enumerate(response.sources_used, start=1):
            lines.append(f"{index}. {source.title}")
            if source.url:
                lines.append(f"   URL: {source.url}")
            lines.append(f"   Why used: {source.reason_used}")
    else:
        lines.append("No sources used.")
        
    lines.append("\nUnsupported Claims:")
    if response.unsupported_claims:
        for claim in response.unsupported_claims:
            lines.append(f"- {claim}")
    else:
        lines.append("None.")
        
    lines.append(f"\nNext Action: {response.next_action.value}")
    lines.append("\n" + "=" * 80)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run the research assistant agent.")
    parser.add_argument("--query", required=True, help="User search query")
    parser.add_argument("--max-results", type=int, default=5, help="Maximum papers to retrieve")
    parser.add_argument("--summary", action="store_true", help="Return summarized Markdown output")
    parser.add_argument("--json", action="store_true", help="Print raw JSON output")
    
    args = parser.parse_args()
    
    agent = build_agent(max_results=args.max_results)
    
    user_message = args.query
    
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]}
    )
    
    structured_response = result["structured_response"]
    
    if args.json:
        print(structured_response.model_dump_json(indent=2))
    else:
        formatted_response = format_agent_response(structured_response)
        print(formatted_response)


if __name__ == "__main__":
    main()
 
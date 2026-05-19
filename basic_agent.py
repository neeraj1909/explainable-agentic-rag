import argparse
import json
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from schemas import AgentResponse

from faithfulness import calculate_faithfulness_stub as calculate_faithfulness
from llm_client import get_llm_client
from summarize_results import summarize_claim
from search_papers import search_papers
from progress import emit_progress
from observability import setup_phoenix_tracing


def build_agent(max_results: int = 5):
    load_dotenv()
    llm = get_llm_client()
    
    def search_papers_with_limit(query: str) -> str:
        """Search arXiv papers using the configured max_results limit."""
        emit_progress(
            "retrieval_started",
            "Searching arXiv papers",
            query=query,
            max_results=max_results,
        )
        
        result = search_papers(query, max_results=max_results)
        
        emit_progress(
            "retrieval_finished",
            "Finished arXiv retrieval",
            query=query,
        )
        
        return result
    
    def summarize_search_results(search_result_json: str) -> str:
        """Summarize JSON arXiv search results."""
        search_result = json.loads(search_result_json)
        paper_count = len(search_result.get("results", []))

        emit_progress(
            "summarization_started",
            "Summarizing search results",
            paper_count=paper_count,
        )

        summary = summarize_claim(search_result, llm)

        emit_progress(
            "summarization_finished",
            "Finished summarizing search results",
            paper_count=paper_count,
        )

        return summary

    def calculate_faithfulness_with_progress(
        answer: str,
        evidence: str,
        threshold: float = 0.35,
    ) -> str:
        """Estimate whether an answer is faithful to provided evidence."""
        emit_progress(
            "faithfulness_started",
            "Checking whether the answer is supported by retrieved evidence",
            threshold=threshold,
        )

        result = calculate_faithfulness(
            answer=answer,
            evidence=evidence,
            threshold=threshold,
        )

        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            payload = {}

        emit_progress(
            "faithfulness_finished",
            "Finished faithfulness check",
            faithfulness_score=payload.get("faithfulness_score"),
            verdict=payload.get("verdict"),
            unsupported_claim_count=len(payload.get("unsupported_claims", [])),
        )

        return result

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
            calculate_faithfulness_with_progress,
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


def _tool_call_name(tool_call: Any) -> str | None:
    """Extract a displayable tool name from LangChain tool-call objects."""
    if isinstance(tool_call, dict):
        return tool_call.get("name") or tool_call.get("function", {}).get("name")
    return getattr(tool_call, "name", None)


def _print_agent_update(update: dict[str, Any], seen_tool_call_ids: set[str]) -> AgentResponse | None:
    """Print progress from one LangGraph/LangChain update chunk."""
    messages = update.get("messages", [])
    if not isinstance(messages, list):
        messages = [messages]

    for message in messages:
        for tool_call in getattr(message, "tool_calls", []) or []:
            tool_name = _tool_call_name(tool_call)
            tool_call_id = (
                tool_call.get("id")
                if isinstance(tool_call, dict)
                else getattr(tool_call, "id", None)
            )
            dedupe_key = tool_call_id or tool_name or repr(tool_call)
            if tool_name and dedupe_key not in seen_tool_call_ids:
                seen_tool_call_ids.add(dedupe_key)
                print(f"[tool_selected] {tool_name}")

    structured_response = update.get("structured_response")
    if structured_response is not None:
        print("[answer_generated] Structured answer ready")
        return structured_response

    return None


def stream_agent_response(agent: Any, user_message: str) -> AgentResponse | None:
    """Run the agent with progress streaming and return the final structured response."""
    print("[run_started] Received query")

    structured_response = None
    seen_tool_call_ids: set[str] = set()

    for mode, chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        stream_mode=["updates", "custom"],
    ):
        if mode == "custom":
            event = chunk.get("event", "progress")
            message = chunk.get("message", "")
            data = chunk.get("data") or {}
            detail = f" {json.dumps(data, ensure_ascii=False)}" if data else ""
            print(f"[{event}] {message}{detail}")
            continue

        if mode != "updates" or not isinstance(chunk, dict):
            continue

        for update in chunk.values():
            if not isinstance(update, dict):
                continue
            maybe_response = _print_agent_update(update, seen_tool_call_ids)
            if maybe_response is not None:
                structured_response = maybe_response

    return structured_response


def main():
    parser = argparse.ArgumentParser(description="Run the research assistant agent.")
    parser.add_argument("--query", required=True, help="User search query")
    parser.add_argument("--max-results", type=int, default=5, help="Maximum papers to retrieve")
    parser.add_argument("--summary", action="store_true", help="Return summarized Markdown output")
    parser.add_argument("--json", action="store_true", help="Print raw JSON output")
    parser.add_argument("--stream", action="store_true", help="Stream the agent's response in real-time")
    
    args = parser.parse_args()

    # Load .env and configure Phoenix/OpenInference before constructing or
    # running LangChain objects, so the agent/LLM/tool spans are captured.
    load_dotenv()
    setup_phoenix_tracing()

    agent = build_agent(max_results=args.max_results)

    user_message = args.query

    if args.stream:
        structured_response = stream_agent_response(agent, user_message)
        if structured_response is None:
            raise RuntimeError("Streaming run finished without a structured response.")
    else:
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
 
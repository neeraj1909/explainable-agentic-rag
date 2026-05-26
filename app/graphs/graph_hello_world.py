import operator
from pathlib import Path

from dotenv import load_dotenv
from langchain.tools import tool
# from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage 
from langgraph.graph import StateGraph, START, END
from typing import Literal
from typing_extensions import TypedDict, Annotated

from app.config import get_llm_client
from app.observability import setup_phoenix_tracing


# model = init_chat_model(
#     "claude-sonnet-4-6",
#     temperature=0
# )
model = get_llm_client()

# define tools
@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply `a` and `b`.
    
    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """
    Adds `a` and `b`.
    
    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """
    Divide `a` and `b`.
    
    Args:
        a: First int
        b: Second int
    """
    return a / b


# Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


# define state
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    

# define model node
def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }


# define tool node
def tool_node(state: dict):
    """Performs the tool call"""
    
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    
    return {"messages": result}


# define end logic
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """
    Decide if we should continue the loop or stop based upon whether the LLM made a tool call
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"
    
    # Otherwise, we stop (reply to the user)
    return END


# build and compile the agent
# build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)    
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

def render_graph_for_terminal(output_dir: str = "/tmp") -> tuple[Path, Path]:
    """Save graph artifacts and print a readable terminal representation.

    Raster image previews such as `chafa graph.png` are often too distorted for
    text-heavy diagrams. For terminals, prefer Mermaid source or LangChain's
    ASCII renderer. The PNG is still saved for GUI viewers.
    """
    graph = agent.get_graph(xray=True)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    png_path = output_path / "graph_hello_world.png"
    mermaid_path = output_path / "graph_hello_world.mmd"

    png_path.write_bytes(graph.draw_mermaid_png())
    mermaid_path.write_text(graph.draw_mermaid(), encoding="utf-8")

    print(f"Graph PNG saved to: {png_path}")
    print(f"Graph Mermaid saved to: {mermaid_path}")
    print(f"Open the clean image with: xdg-open {png_path}")

    try:
        terminal_graph = graph.draw_ascii()
        ascii_note = None
    except ImportError:
        terminal_graph = (
            "  START\n"
            "    │\n"
            "    ▼\n"
            "  llm_call ── tool call ──▶ tool_node\n"
            "    │                         │\n"
            "    └──── no tool call ─▶ END │\n"
            "                              │\n"
            "                              └──▶ llm_call\n"
        )
        ascii_note = "For auto-generated ASCII graphs, install: uv add grandalf"

    print("\nTerminal graph:")
    print(terminal_graph)
    if ascii_note:
        print(ascii_note)

    return png_path, mermaid_path


def run_demo() -> None:
    # Show/save the agent graph
    render_graph_for_terminal()
    
    setup_phoenix_tracing()

    # Invoke
    messages = [HumanMessage(content="Add 3 and 4. Then multiply it by 3.")]
    messages = agent.invoke({"messages": messages})

    for m in messages["messages"]:
        m.pretty_print()


if __name__ == "__main__":
    run_demo()

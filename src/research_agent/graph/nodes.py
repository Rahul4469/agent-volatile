"""
A NODE is a function that:
1. Receives the current STATE
2. Does some work (call LLM, execute tool, make decision)
3. Returns STATE UPDATES
NODE CONTRACT
=============
Each node function follows this contract:
- Input: Current state (AgentState TypedDict)
- Output: Partial state dict (LangGraph merges with current state)
- Side Effects: Emit events for streaming

IMPORTANT: Nodes don't return the full state - they return UPDATES.
LangGraph merges these updates into the existing state using reducers.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Literal, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode

from research_agent.core.events import EventEmitter, EventType
from research_agent.core.exceptions import LLMError, ToolError
from research_agent.core.state import AgentState, AgentStatus, ResearchState

logger = logging.getLogger(__name__)
from research_agent.config import get_settings

# SYSTEM PROMPTS -----------------------------------------------------------------------------------

RESEARCH_AGENT_SYSTEM_PROMPT = """You are a sophisticated research agent designed to help users find accurate, comprehensive information.

## Your Capabilities
- **web_search**: Search the internet for current information
- **fetch_page**: Read and extract content from web pages  
- **calculator**: Perform mathematical calculations

## How to Work
1. **Understand the Query**: Carefully analyze what the user is asking
2. **Plan Your Approach**: Think about what information you need
3. **Execute Tools**: Use tools to gather information
4. **Synthesize**: Combine findings into a coherent answer

## Important Guidelines
- Always cite your sources when providing information
- If information is uncertain or conflicting, acknowledge this
- Break complex queries into smaller searchable questions
- Verify important facts from multiple sources when possible
- Be transparent about limitations (e.g., paywalled content, recency)

## Response Format
When you have gathered enough information, provide a clear, well-structured answer.
Include relevant sources and acknowledge any gaps in the information.

Current date: {current_date}
"""

SYNTHESIS_PROMPT = """Based on the research conducted, synthesize a comprehensive answer.

## Research Findings
{findings}

## Sources Used  
{sources}

## Instructions
- Provide a clear, well-organized answer
- Cite sources using [Source N] notation
- Highlight any conflicting information
- Note any limitations or gaps
- Keep the response focused and relevant to the original query
"""

# AGENT NODE - Call LLM with Tools ---------------------------------------------------

async def agent_node(state: AgentState, 
                     model: BaseChatModel,
                     tools: Sequence[BaseTool], 
                     emitter: EventEmitter | None = None) -> dict[str, Any]:
    """
    The AGENT node - calls the LLM to reason about what to do next.
    1. Takes the current conversation (messages)
    2. Sends them to the LLM with tool definitions
    3. Returns the LLM's response, LLM decides to either:
       a. Call one or more tools (returns AIMessage with tool_calls)
       b. Give a final answer (returns AIMessage with content)
    Args:
        state: Current agent state
        model: LangChain chat model (with tools bound)
        emitter: Optional event emitter for streaming
    Returns:
        State update dict with new message
    """
    iteration = state.get("iteration", 0)
    
    # Emit node start event
    if emitter:
        await emitter.emit_node_start("agent", {"iteration": iteration})
        
    # Build messages for the LLM
    # Load from the langgraph "state"
    messages = list(state.get("messages", []))
    
    # Add system prompt if not present
    if not messages or not isinstance(messages[0], SystemMessage):
        system_prompt = RESEARCH_AGENT_SYSTEM_PROMPT.format(
            current_date = datetime.now().strftime("%Y-%m-%d"),
        )
        messages = [SystemMessage(content=system_prompt)] + messages
        
    # Bind tools to model if available
    if tools:
        model_with_tools = model.bind_tools(tools)
    else:
        model_with_tools = model 
    
    try:
        # Emit thinking event
        if emitter:
            await emitter.emit(EventType.THINKING, {"status": "calling_llm"})
        
        # Call the LLM
        response: AIMessage = await model_with_tools.ainvoke(messages)
        
        # Log and emit tool call events
        if response.tool_calls:
            tool_names = [tc["name"] for tc in response.tool_calls]
            logger.info(f"Agent decided to call {len(response.tool_calls)} tool(s): {tool_names}")
            
            if emitter:
                for tc in response.tool_calls:
                    await emitter.emit_tool_call(tc["name"], tc.get("args", {}))
        else:
            logger.info("Agent generated final response (no tool calls)")
            if emitter and response.content:
                await emitter.emit_final_answer(str(response.content))
        
        # Emit node and event
        if emitter:
            await emitter.emit_node_end("agent", {
                "has_tool_calls": bool(response.tool_calls),
                "response_length": len(response.content) if response.content else 0,
            })
        
        # Determine new status
        new_status = AgentStatus.RUNNING if response.tool_calls else AgentStatus.COMPLETED
        
        return {
            "messages": [response],
            "status": new_status,
            "current_node": "agent",
            "iteration": iteration + 1,
            # Set final answer if no tool calls
            "final_answer": str(response.content) if not response.tool_calls and response.content else None,
        }
    except Exception as e:
        logger.error(f"Error in agent node: {e}")
        if emitter:
            await emitter.emit_error(str(e), {"node": "agent"})
        raise LLMError(
            message=f"Failed to get LLM response: {e}",
            model=getattr(model, "model", "unknown"),
            cause=e,
        )

# TOOLS NODE - Execute Tool calls ---------------------------------------------

async def tools_node(state: AgentState,
                     tools: Sequence[BaseTool],
                     emitter: EventEmitter | None = None) -> dict[str, Any]:
    """ Execute tool calls from the agent
    This node:
    1. Gets tool calls from the last AI message
    2. Executes each tool in parallel (when possible)
    3. Returns ToolMessages with the results
    Args:
        state: Current agent state
        tools: List of available tools
        emitter: Optional event emitter for streaming    
    Returns:
        State update dict with tool result messages    
    After execution, LangGraph routes back to agent_node,
    allowing the agent to process the results.
    """
    if emitter:
        await emitter.emit_node_start("tools", {"iteration": state.get("iteration", 0)})
    
    messages = state.get("messages", [])
    if not messages:
        logger.warning("Tool node called with no messages")
        return {"message": []}
    
    # Get the last AI message (should have tool calls)
    last_message = messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        logger.warning("Tools node called but no tool calls in last message")
        return {"messages": []}
    
    # Build tool lookup map
    tool_map = {tool.name: tool for tool in tools}
    
    # Execute each tool call
    tool_messages: list[ToolMessage] = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        tool_id = tool_call["id"]
        
        logger.info(f"Executing tool: {tool_name}")
        logger.debug(f"Tool args: {tool_args}")
        
        if emitter:
            await emitter.emit_tool_call(tool_name, tool_args)
        
        # Find the tool
        tool = tool_map.get(tool_name)
        
        if tool is None:
            error_msg = f"Tool '{tool_name}' not found. Available: {list(tool_map.keys())}"
            logger.error(error_msg)
            tool_messages.append(ToolMessage(
                content=f"Error: {error_msg}",
                tool_call_id=tool_id,
                name=tool_name,
            ))
            if emitter:
                await emitter.emit_error(error_msg, {"tool": tool_name})
            continue
        
        # Execute the tool
        try:
            # Preferasync execution
            if hasattr(tool, "ainvoke"):
                result = await tool.ainvoke(tool_args)
            elif hasattr(tool, "_arun"):
                result = await tool._arun(**tool_args)
            else:
                # Fallback to sync
                result = tool._run(**tool_args)
            
            # Ensure result is string
            if not isinstance(result, str):
                if isinstance(result, (dict, list)):
                    result = json.dumps(result, ident=2, default=str)
                else:
                    result = str(result)
            
            logger.info(f"Tool {tool_name} completed, Result linght: {len(result)}")
            
            tool_messages.append(ToolMessage(
                content = result,
                tool_call_id = tool_id,
                name = tool_name,
            ))
            
            if emitter:
                await emitter.emit_tool_result(tool_name, result)
        
        except ToolError as e:
            error_msg = f"Tool error: {e.message}"
            logger.error(f"Tool {tool_name} failed: {e}")
            tool_messages.append(ToolMessage(
                content=f"Error: {error_msg}",
                tool_call_id=tool_id,
                name=tool_name,
            ))
            if emitter:
                await emitter.emit_error(error_msg, {"tool": tool_name})
                
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.exception(f"Unexpected error executing {tool_name}")
            tool_messages.append(ToolMessage(
                content=f"Error: {error_msg}",
                tool_call_id=tool_id,
                name=tool_name,
            ))
            if emitter:
                await emitter.emit_error(error_msg, {"tool": tool_name})
        
        if emitter:
            await emitter.emit_node_end("tools", {"tools_executed": len(tool_messages)})
            
        return {
            "messages": tool_messages,
            "status": AgentStatus.RUNNING,
            "current_node": tools,
        }

# CONDITIONAL EDGE - Routing logic ------------------------------------------------

def should_continue(state: AgentState, 
                    max_iterations: int = 10) -> Literal["tools", "end", "error"]:
    """
    Conditional edge function - determines which node to execute next.
    
    This function examines the current state and decides:
    - "tools": Agent wants to use tools → go to tools_node
    - "end": Agent is done (no tool calls) → end execution
    - "error": Something went wrong → go to error_handler
    Args:
        state: Current agent state
        max_iterations: Maximum allowed iterations (safety limit)    
    Returns:
        Name of the next node: "tools", "end", or "error"   
    This is the key routing logic that creates the ReAct loop:
    agent → tools → agent → tools → ... → end
    """
    
    # Check for error status
    if state.get("status") == AgentStatus.FAILED:
        logger.info("Routing to error: statis is FAILED")
        return "error"
    
    # Check iteration limit (prevent infinite loops)
    iteration = state.get("iteration", 0)
    if iteration >= max_iterations:
        logger.warning(f"Max iterations ({max_iterations}) reached, routing to error")
        return "error"
    
    # Check the last message for tool calls
    messages = state.get("messages", [])
    if not messages:
        logger.info("No message, routing to end")
        return "end"
    
    last_message = messages[-1]
    
    # if AI message with tool calls -> go to tools
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.debug(f"Agent has {len(last_message.tool_calls)} tool calls, routing to tools")
        return "tools"
    
    # Otherwise , we're done
    logger.info("No tool calls, routing to end")
    return "end"

# ERROR HANDLER NODE ---------------------------------------------------------

async def error_handler_node(state: AgentState,
                             emitter: EventEmitter | None = None) -> dict[str, Any]:
    """
    Handle errors gracefully.
    This node is called when:
    - Max iterations exceeded (possible infinite loop)
    - Error status was set
    - Other exceptional conditions
    Args:
        state: Current agent state
        emitter: Optional event emitter    
    Returns:
        State update with error information and message
    """
    if emitter:
        await emitter.emit_node_start("error_handler")
    
    error = state.get("error")
    iteration = state.get("iteration", 0)
    
    error = state.get("error")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("context", {}).get("max_iterations", 10)
    
    # Determine error message based on condition
    if iteration >= max_iterations:
        error_message = (
            f"I've reached my maximum iteration limit ({max_iterations}) while researching your query. "
            "This usually means the research task is very complex. "
            "Here's what I found so far. You may want to ask a more specific question."
        )
    elif error:
        error_message = f"I encountered an error while processing you request: {error}"
    else:
        error_message = "An unexpected error occured. Please try again."
    
    # Create error response
    error_response = AIMessage(content=error_message)
    
    if emitter:
        await emitter.emit_error(error_message)
        await emitter.emit_node_end("error_handler")
    
    return {
        "messages": [error_response],
        "status": AgentStatus.FAILED,
        "error": error_message,
        "final_answer": error_message,
        "current_node": "error_handler",
    }
    
# SYNTHESIS NODE (for Research Workflow) ----------------------------------------

async def synthesize_node(state: ResearchState,
                          model: BaseChatModel,
                          emitter: EventEmitter | None = None) -> dict[str, Any]:
    """
    Synthesize research findings into a final answer.
    
    This node is used in the extended ResearchState workflow.
    It takes accumulated findings and sources and creates
    a comprehensive, well-cited answer.
    Args:
        state: Current research state (with sources/findings)
        model: LangChain chat model
        emitter: Optional event emitter   
    Returns:
        State update with synthesized final answer
    """
    if emitter:
        await emitter.emit_node_start("synthesize")
        
    findings = state.get("findings", [])
    sources = state.get("sources", [])
    
    # Format findings and sources for the prompt
    if findings:
        findings_text = "\n".join(f"- {f}" for f in findings)
    else:
        findings_text = "No specific findings recorded."
    
    if sources:
        sources_text = "\n".join(
            f"[Source {i+1}] {s.get('title', 'Untitled')}: {s.get('url', 'No URL')}"
            for i, s in enumerate(sources)
        )
    else:
        sources_text = "No sources recorded."
        
    synthesis_prompt = SYNTHESIS_PROMPT.format(
        findings=findings_text,
        sources=sources_text,
    )
    
    try:
        # Get the original query
        original_query = ""
        for msg in state.get("messages", []):
            if isinstance(msg, HumanMessage):
                original_query = str(msg.content)
                break
        
        messages = [
            SystemMessage(content="You are a research synthesis assistant. Create comprehensive, well-cited answers."),
            HumanMessage(content=f"Original Query: {original_query}\n\n{synthesis_prompt}"),
        ]
        
        response = await model.ainvoke(messages)
        
        if emitter:
            await emitter.emit_final_answer(str(response.content))
            await emitter.emit_node_end("synthesize")
        
        return {
            "messages": [response],
            "status": AgentStatus.COMPLETED,
            "final_answer": str(response.content),
            "current_node": "synthesize",
        }
        
    except Exception as e:
        logger.error(f"Error in synthesis: {e}")
        if emitter:
            await emitter.emit_error(str(e), {"node": "synthesize"})
        raise LLMError(
            message=f"Failed to synthesize answer: {e}",
            model=getattr(model, "model", "unknown"),
            cause=e,
        )
    
# UTILITY FUNCTIONS -----------------------------------------------------

def increment_iteration(state: AgentState) -> dict[str, Any]:
    """
    Simple node that increments the iteration counter.
    Useful as a pre-processing step for tracking.
    """
    return {
        "iteration": state.get("iteration", 0) + 1
    }

def extract_final_answer(state: AgentState) -> str | None:
    """
    Extract the final answer from the agent state.
    
    Looks for:
    1. Explicit final_answer in state
    2. Last AI message without tool calls
    
    Returns:
        The final answer string, or None if not found
    """
    # Check explicit final_answer
    if state.get("final_answer"):
        return state["final_answer"]
    
    # Find last AI message without tool calls
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            content = msg.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Handle content blocks
                return " ".join(
                    block.get("text", str(block))
                    for block in content
                    if isinstance(block, dict)
                )
    return None

def format_messages_for_display(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """
    Format LangChain messages for API/display purposes.
    
    Converts internal message types to a simple dict format.
    
    Args:
        messages: Sequence of LangChain messages
        
    Returns:
        List of dicts with role, content, and optional metadata
    """
    formatted = []
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append({
                "role": "user",
                "content": str(msg.content),
            })
        elif isinstance(msg, AIMessage):
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": str(msg.content) if msg.content else "",
            }
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.get("id"),
                        "name": tc.get("name"),
                        "args": tc.get("args", {}),
                    }
                    for tc in msg.tool_calls
                ]
            formatted.append(entry)
        elif isinstance(msg, ToolMessage):
            formatted.append({
                "role": "tool",
                "name": msg.name,
                "content": str(msg.content),
                "tool_call_id": msg.tool_call_id,
            })
        elif isinstance(msg, SystemMessage):
            formatted.append({
                "role": "system",
                "content": str(msg.content),
            })
        
        return formatted
    
def count_messages_by_type(messages: Sequence[BaseMessage]) -> dict[str, int]:
    """
    Count messages by type for analytics/debugging.
    Returns:
        Dict with counts: {"user": N, "assistant": N, "tool": N, "system": N}
    """
    counts = {"user": 0, "assistant": 0, "tool": 0, "system": 0}
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            counts["user"] += 1
        elif isinstance(msg, AIMessage):
            counts["assistant"] += 1
        elif isinstance(msg, ToolMessage):
            counts["tool"] += 1
        elif isinstance(msg, SystemMessage):
            counts["system"] += 1
    
    return counts

# EXPORTS-----------------------------------------------------------------

__all__ = [
    # Node functions
    "agent_node",
    "tools_node",
    "should_continue",
    "error_handler_node",
    "synthesize_node",
    # Utilities
    "increment_iteration",
    "extract_final_answer",
    "format_messages_for_display",
    "count_messages_by_type",
    # Prompts (for customization)
    "RESEARCH_AGENT_SYSTEM_PROMPT",
    "SYNTHESIS_PROMPT",
]
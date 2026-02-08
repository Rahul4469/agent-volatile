# """
# STATE in LANGGRAPH IS LIKE CONTEXT in GO
# In LangGraph, STATE is the single source of truth that:
# 1. Flows between nodes in the graph
# 2. Gets persisted by checkpointers (memory)
# 3. Enables streaming and human-in-the-loop
# 
# Each node:
#   - READS current state
#   - Does work (call LLM, execute tool)
#   - RETURNS updates to state
#   
# REDUCERS:
# =========
# When multiple nodes update the same state key, how do we merge them?
# This is where REDUCERS come in.
# 
# Example: Messages list
#   - Node A returns: {"messages": [msg1]}
#   - Node B returns: {"messages": [msg2]}
#   - With add_messages reducer: state.messages = [msg1, msg2] (appended!)
#   - Without reducer: state.messages = [msg2] (overwritten!)
# """

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ENUMS - Status and Types -------------------------------------------------

class AgentStatus(str, Enum):
    PENDING = "pending"       
    RUNNING = "running"       # Agent is thinking/acting
    COMPLETED = "completed"   
    FAILED = "failed"         
    INTERRUPTED = "interrupted"  # Human-in-the-loop pause

class NodeType(str, Enum):
    """ Types of nodes in the agent graph"""
    LLM = "llm"
    TOOL = "tool"
    DECISION = "decision"
    START = "start"
    END = "end"
    
# STATE DEFINITION - TypedDict with Reducers -------------------------------

from typing import TypedDict

class AgentState(TypedDict):
    
    # Annotated: Wraps a base type + extra metadata
    # add_messages: LangGraph's built-in reducer function, merges old state + new update → new state.
    messages: Annotated[Sequence[BaseMessage], add_messages] 
    
    # This is a flexible dict for storing:
    #   - User information (id, preferences)
    #   - Session data (start time, thread id)
    #   - Custom data (research topic, depth level)
    # No reducer = overwrites on update (which is what we want here)
    context: dict[str, Any]
    
    # Execution state
    status: AgentStatus
    current_node: str
    iteration: int

    # results
    error: str | None
    final_answer: str | None
    
# Helper Functions - State Creation ///---------------------------------

def create_initial_state(query: str, context: dict[str, Any ] | None = None) -> AgentState:
    """ 
    Create the initial state for a new agent run.    
    Call this when a user sends a new message
    USAGE:
    ```python
    state = create_initial_state(
        query="What are the latest developments in AI?",
        context={"user_id": "123", "depth": "deep"}
    )
    result = await graph.ainvoke(state)
    """
    return AgentState(
        message = [HumanMessage(conten=query)],
        context = context or {},
        status = AgentStatus.PENDING,
        current_node = "start",
        iteration = 0,
        error = None,
        final_answer = None,
    )

# Pydantic Models - For API Serialization    

class MessageModel(BaseModel):
    role: str = Field(description="Message role: user, assistant, tool")
    content: str = Field(description="Message content")
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None,
        description="Tool calls made by the assistant",
    )
    name: str | None = Field(
        default=None,
        description="Tool name (for tool messages)",
    )
    
    # TypedDict is great for LangGraph internal use, but for API responses
    # we need Pydantic models with validation and serialization.
    #
    # These models are used in:
    #   - FastAPI endpoints (request/response bodies)
    #   - WebSocket messages
    #   - Event streaming
    @classmethod
    def from_langchain(cls, msg: BaseMessage) -> "MessageModel":
        """ To convert a LangChain message to our Pydantic model."""
        
        # Determine role from message type
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, ToolMessage):
            role = "tool"
        else:
            role = "unknown"    
            
        # Extract content (handle different content formats)    
        if isinstance(msg.content, str):
            content = msg.content
        elif isinstance(msg.content, list):
            # content blocks for multimodel
            content = " ".join(
                block.get("text", str(block))
                for block in msg.content
                if isinstance(block, dict)
            )    
        else:
            content = str(msg.content)
        
        # Extract tool calls (for AIMessage)
        tool_calls = None
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.get("id"),
                    "name": tc.get("name"),
                    "args": tc.get("args", {}),
                }
                for tc in msg.tool_calls
            ]
        
        # Extract tool name (for ToolMessage)
        name = None
        if isinstance(msg, ToolMessage):
            name = msg.name
        
        return cls(
            role=role,
            content=content,
            tool_calls=tool_calls,
            name=name,
        )               

class StateSnapshot(BaseModel):
    """ a snapshot of the agent state for api responses.
    this is what is returned to clients.    
    """
    messages: list[MessageModel] = Field(description="Conversation history")
    status: str = Field(description="Current status")
    iteration: int = Field(description="Current iteration")
    final_answer: str | None = Field(default=None, description="Final response")
    error: str | None = Field(default=None, description="Error message if failed")
    
    @classmethod
    def from_state(cls, state: AgentState) -> "StateSnapshot":
        """
        Create a snapshot from an AgentState.
        
        This converts the internal state to an API-friendly format.
        """
        return cls(
            messages=[
                MessageModel.from_langchain(msg)
                for msg in state.get("messages", [])
            ],
            status=state.get("status", AgentStatus.PENDING).value,
            iteration=state.get("iteration", 0),
            final_answer=state.get("final_answer"),
            error=state.get("error"),
        )    
        
# RESEARCH-SPECIFIC STATE

class ResearchState(AgentState):
    sources: list[dict[str, str]]   # {"url": "...", "title": "...", "summary": "..."}
    findings: list[str]
    depth: str                      # "quick", "normal", "deep" 

def create_research_state(query: str, depth: str = "normal", context: dict[str, Any] | None = None) -> ResearchState:
    base_context = context or {}
    base_context["research_depth"] = depth
    
    return ResearchState(
        messages=[HumanMessage(content=query)],
        context=base_context,
        status=AgentStatus.PENDING,
        current_node="start",
        iteration=0,
        error=None,
        final_answer=None,
        sources=[],
        findings=[],
        depth=depth,
    )
    
# STREAMING EVENT TYPES 

class StreamEvent(BaseModel):
    type: str = Field(description="Event type")   
    timestamp: datetime = Field(
        default_factory = datetime.utcnow,
        description= "Event timestamp",
        )
    node: str | None = Field(default=None, description="Current node name")
    data: dict[str, Any] | None = Field(default = None, description = "Event data")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

# EXPORTS ----------------------------

__all__ = [
    # Enums
    "AgentStatus",
    "NodeType",
    # State types
    "AgentState",
    "ResearchState",
    # Factory functions
    "create_initial_state",
    "create_research_state",
    # API models
    "MessageModel",
    "StateSnapshot",
    "StreamEvent",
]        
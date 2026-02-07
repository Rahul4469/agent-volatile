"""
EVENTS - Event system for Streaming and Observability
"""
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

# EvENT TYPES ------------------------------------------------------

class EventType(str, Enum):
    """
    Types of events emitted during agent execution.
    
    These map to the agent lifecycle:
    1. START -> Agent begins
    2. NODE_START -> Entering a graph node
    3. THINKING -> LLM is generating (optional, for token-level)
    4. TOOL_CALL -> About to call a tool
    5. TOOL_RESULT -> Tool returned a result
    6. NODE_END -> Exiting a graph node
    7. FINAL_ANSWER -> Agent has an answer
    8. ERROR -> Something went wrong
    9. DONE -> Agent finished
    """
    START = "start"
    NODE_START = "node_start"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    NODE_END = "node_end"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"
    DONE = "done"
    
# EVENT DATA CLASSES ------------------------------------------

@dataclass
class Event:
    """
    An event emitted during agent execution.
    This is the core data structure for the event system.
    Events are immutable (frozen=True) for thread safety.
    """
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)  # default_factory: Fresh dict per instance
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_id: str = field(default_factory=lambda: str(uuid()))   
    thread_id: str = ""
    run_id: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert event to dictionary for JSON serialization. 
        Used when sending events over WebSocket or SSE.
        """
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "event_id": self.event_id,
            "thread_id": self.thread_id,
            "run_id": self.run_id,
        }     
        
# PYDANTIC EVENT MODEL (for API responses)

class EventModel(BaseModel):
    """
    Pydantic model for events (API serialization).
    This is used in FastAPI responses and WebSocket messages.
    """      
    type: str = Field(description="Event type")
    data: dict[str, Any] = Field(default_factory=dict, description="Event data") # default_factory: Fresh dict per instance
    timestamp: str = Field(description="ISO format timestamp")
    event_id: str = Field(description="Unique event ID")
    thread_id: str = Field(default="", description="Conversation thread ID")
    run_id: str = Field(default="", description="Agent run ID")
    
    @classmethod
    def from_event(cls, event: Event) -> "EventModel":
        """Convert internal Event to API model."""
        return cls(
            type=event.type.value,
            data=event.data,
            timestamp=event.timestamp.isoformat(),
            event_id=event.event_id,
            thread_id=event.thread_id,
            run_id=event.run_id,
        )  
        
# EVENT EMITTER --------------------------------------------

class EventEmitter:
    """Async event emitter from streaming updates.
    This is the HEART of the streaming system. It:
    1. Accepts events from the agent
    2. Distributes them to all subscribers
    3. Supports multiple subscribers (WebSocket, SSE, logging)
    """
    # Python's asyncio.Queue is equivalent to Go's buffered channel.
           
    def __init__(self, thread_id: str = "", run_id: str = "") -> None:
        """
        Args:
            thread_id: Conversation thread ID (for filtering)
            run_id: Agent run ID (for filtering)
        """
        self._subscribers: list[asyncio.Queue[Event | None]] = []
        self._thread_id = thread_id
        self._run_id = run_id or str(uuid4())
        self._closed = False
        
    @property # getter
    def thread_id(self) -> str:
        return self._thread_id
    
    @property # getter
    def run_id(self) -> str:
        return self._run_id      
    
    def subscribe(self) -> asyncio.Queue[Event | None]:
        """
        Subscribe to events.
        
        Returns an asyncio.Queue that will receive all events.
        The queue yields None when the emitter is closed.
        """
        if self._closed:
            raise RuntimeError("Cannot subscribe to cloed emitter")
        
        queue: asyncio.Queue[Event | None] = asyncio.Queue(maxsize=100)
        self._subscribers.append(queue)
        return queue   
    
    async def emit(self, event_type: EventType, data: dict[str, Any] | None = None) -> Event:
        """ Emit an event to all subs"""
        if self._closed:
            return Event(type=event_type) # No-op if closed
        
        event = Event(
            type=event_type,
            data=data or {},
            thread_id=self._thread_id,
            run_id=self._run_id,
        )                
        
        # Send to all subscribers
        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Queue is full, subscriber is slow
                # In production, log this
                pass
              
        return event    
    
    #CONVENIENCE METHODS------
    # These make it easy to emit common event types
    async def emit_start(self) -> Event:
        """Emit agent start event."""
        return await self.emit(EventType.START, {"run_id": self._run_id})
    
    async def emit_node_start(self, node: str, data: dict[str, Any] | None = None) -> Event:
        """Emit node start event."""
        return await self.emit(EventType.NODE_START, {"node": node, **(data or {})})
    
    async def emit_node_end(self, node: str, data: dict[str, Any] | None = None) -> Event:
        """Emit node end event."""
        return await self.emit(EventType.NODE_END, {"node": node, **(data or {})})
    
    async def emit_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> Event:
        """Emit tool call event."""
        return await self.emit(EventType.TOOL_CALL, {
            "tool": tool_name,
            "args": tool_args,
        })
    
    async def emit_tool_result(
        self,
        tool_name: str,
        result: Any,
    ) -> Event:
        """Emit tool result event."""
        return await self.emit(EventType.TOOL_RESULT, {
            "tool": tool_name,
            "result": str(result)[:1000],  # Truncate large results
        })
    
    async def emit_thinking(self, content: str) -> Event:
        """Emit thinking/token event (for token-level streaming)."""
        return await self.emit(EventType.THINKING, {"content": content})
    
    async def emit_final_answer(self, answer: str) -> Event:
        """Emit final answer event."""
        return await self.emit(EventType.FINAL_ANSWER, {"answer": answer})
    
    async def emit_error(self, error: str, details: dict[str, Any] | None = None) -> Event:
        """Emit error event."""
        return await self.emit(EventType.ERROR, {
            "error": error,
            **(details or {}),
        })
    
    async def emit_done(self) -> Event:
        """Emit done event and close the emitter."""
        event = await self.emit(EventType.DONE)
        await self.close()
        return event
    
    async def close(self) -> None:
        """
        Close the emitter and signal subscribers.
        
        Sends None to all queues to signal completion.
        Subscribers should check for None and exit their loops.
        ```
        """
        if self._closed:
            return
        
        self._closed = True
        
        for queue in self._subscribers:
            try:
                queue.put_nowait(None)  # Signal completion
            except asyncio.QueueFull:
                pass 

# ASYNC ITERATOR FOR STREAMING -----------------------------------------------------

async def event_stream(emitter: EventEmitter) -> AsyncIterator[Event]:
    """
    for consuming events in a for loop.
    """
    queue = emitter.subscribe()
    
    while True:
        event = await queue.get()
        if event is None:
            break
        yield event

# CONTEXT MANAGER FOR EVENT HANDLING------------------------------------------------
        
@asynccontextmanager 
async def event_context(thread_id: str = "", run_id: str = ""):
    """
    Context manager that creates an event emitter and ensures cleanup.
    """
    emitter = EventEmitter(thread_id=thread_id, run_id=run_id)
    try:
        yield emitter
    finally:
        if not emitter._closed:
            await emitter.close()

# EXPORTS ////////////////////////////////////////

__all__ = [
    "EventType",
    "Event",
    "EventModel",
    "EventEmitter",
    "event_stream",
    "event_context",
]                     
               
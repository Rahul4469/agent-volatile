"""
This module provides checkpointer setup for LangGraph, enabling:
1. State persistence across requests (conversation memory)
2. Thread management for multi-user support
3. Interruption and resume capabilities
4. State history and time-travel debugging
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from research_agent.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# THREAD ID MANAGEMENT-------------------------------------------------------------

def generate_thread_id() -> str:
    """
    Thread IDs identify unique conversations/sessions.
    They're used by checkpointers to store and retrieve state.
    
    Returns:
        UUID string suitable for use as thread_id
    """
    return str(uuid4())

def create_config(thread_id: str | None = None,
                  checkpoint_ns: str = "",
                  **extra: Any,) -> dict[str, Any]:
    """
    Create a LangGraph config dict with thread_id.
    This config is passed to graph.invoke() or graph.astream()
    to specify which conversation/thread to use.
    Args:
        thread_id: Unique identifier for the conversation.
                   If None, generates a new one.
        checkpoint_ns: Optional namespace for the checkpoint.
                       Useful for multi-tenant scenarios.
        **extra: Additional configurable options
    Returns:
        Config dict ready for graph invocation
    Example:
        >>> config = create_config(thread_id="user-123-conv-1")
        >>> result = await graph.ainvoke(state, config)
        
        >>> # Auto-generate thread_id
        >>> config = create_config()
        >>> thread_id = config["configurable"]["thread_id"]
    """
    if thread_id is None:
        thread_id = generate_thread_id()
    
    configurable = {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        **extra,
    }
    
    return {"configurable": configurable}

def get_thread_id(config: dict[str, Any]) -> str | None:
    return config.get("configurable", {}.get("thread_id"))

# CHECKPOINT CONFIGURATION ---------------------------------------------------------

@dataclass
class CheckpointerConfig:
    """Configuration for checkpointer setup."""
    
    use_memory: bool = True
    """Use in-memory checkpointer (True) or Postgres (False)."""
    
    database_url: str = ""
    """PostgreSQL connection URL for persistent storage."""
    
    pool_size: int = 5
    """Connection pool size for Postgres."""
    
    max_overflow: int = 10
    """Maximum overflow connections for Postgres pool."""
    
    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "CheckpointerConfig":
        """Create config from application settings."""
        settings = settings or get_settings()
        return cls(
            use_memory = settings.use_memory_checkpointer,
            database_url = settings.database_url,
        )


# CHECKPOINT MANAGER -----------------------------------------------------------

class CheckpointerManager:
    """
    Manages checkpointer lifecycle and provides access to checkpointer instances.
    
    This class handles:
    - Creating the appropriate checkpointer based on configuration
    - Managing connection lifecycle for database checkpointers
    - Providing a clean interface for the rest of the application.
    
    Usage:
    - dev
    >>> manager = CheckpointerManager.create_memory()
    
    - prod - postgres
    >>> manager = CheckpointerManager.create_postgres(database_url)
    """
    
    def __init__(self, config: CheckpointerConfig) -> None:
        """
        Init with config.
        Note: The checkpointer is not created until initialize() is called.
        """
        self._config = config
        self._checkpointer: BaseCheckpointSaver | None = None
        self._pool = None # Connection pool for postgres
        self._initialized = False
    
    @classmethod
    def create_memory(cls) -> "CheckpointerManager":
        """
        Returns:
            Initialized CheckpointerManager with MemorySaver
        """
        config = CheckpointerConfig(use_memory=True)
        manager = cls(config)
        manager._checkpointer = MemorySaver()
        manager._initialized = True
        logger.info("Created in-memory checkpointer")
        return manager
    
    @classmethod
    def create_postgres(cls, database_url: str) -> "CheckpointerManager":
        
        config = CheckpointerConfig(
            use_memory=False,
            database_url=database_url,
        )
        return cls(config)
    
    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "CheckpointerManager":
        """
        Create a cp manager based on application settings.
        Uses use_memory_checkpointer setting to determine type.
        """
        settings = settings or get_settings()
        
        if settings.use_memory_checkpointer:
            return cls.create_memory()
        else:
            return cls.create_postgres(settings.database_url)
    
    async def initialize(self) -> None:
        """
        Initialize the checkpointer (required for Postgres).
        
        For memory checkpointer, this is a no-op.
        For Postgres, this sets up the connection pool and creates tables.
        """
        
        if self._initialized:
            return
        
        if self._config.use_memory:
            self._checkpointer = MemorySaver()
            self._initialized = True
            logger.info("Initialized in-memory checkpointer")
            return

        # Initialize PostgreSQL checkpointer
        try:
            # Import here to avoid dependency issues if not using Postgres
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            import asyncpg
            
            # Create connection pool
            self._pool = await asyncpg.create_pool(
                self._config.database_url,
                min_size = 1,
                max_size = self._config.pool_size,
            )
            
            # Create checkpointer with the pool
            self._checkpointer = AsyncPostgresSaver(self._pool)
            
            # Setup tables (idempotent - creates if not exists)
            await self._checkpointer.setup()
            
            self._initialized = True
            logger.info("Initailized PostgreSQL checkpointer")
        
        except ImportError as e:
            logger.error(
                f"PostgreSQL checkpointer dependencies not installed: {e}. "
                "Install with: pip install langgraph-checkpoint-postgres asyncpg"
            )

            raise
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL checkpointer: {e}")
            raise
    
    async def close(self) -> None:
        """Close the checkpointer and release resources."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("Closed PostgreSQL connection pool")
        
        self._checkpointer = None
        self._initialized = False
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - closes connections."""
        await self.close()
    
    def get_checkpointer(self) -> BaseCheckpointSaver:
        """
        Get the checkpointer instance.
        Returns:
            The configured checkpointer
            
        Raises:
            RuntimeError: If checkpointer not initialized
        """
        if not self._initialized or self._checkpointer is None:
            raise RuntimeError(
                "Checkpointer not initialized."
                "Use 'await manager.initialize()' or async context manager."
            )
            
    @property
    def is_initialized(self) -> bool:
        """Check if checkpointer is initialized."""
        return self._initialized
    
    @property
    def is_persistent(self) -> bool:
        """Check if using persistent (Postgres) storage."""
        return not self._config.use_memory

# STATE HISTORY & REPLAY -----------------------------------------------------------

@dataclass
class StateHistoryEntry:
    """A single entry in the state history."""
    
    checkpoint_id: str
    """Unique identifier for this checkpoint."""
    
    thread_id: str
    """Thread this checkpoint belongs to."""
    
    parent_checkpoint_id: str | None
    """ID of the parent checkpoint (previous state)."""
    
    channel_values: dict[str, Any]
    """The actual state data at this checkpoint."""
    
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata (timestamps, node info, etc)."""
    
    created_at: datetime | None = None
    """When this checkpoint was created."""
    
async def get_state_history(checkpointer: BaseCheckpointSaver,
                            thread_id: str,
                            limit: int = 10) -> list[StateHistoryEntry]:
    """
    Retrieve state history for a thread.
    This enables "time-travel" debugging - you can see what
    the state looked like at each step of execution.
    
    Args:
        checkpointer: The checkpointer instance
        thread_id: Thread to get history for
        limit: Maximum number of entries to return
        
    Returns:
        List of StateHistoryEntry objects, newest first
    """
    config = create_config(thread_id = thread_id)
    entries = []
    
    try:
        async for checkpoint_tuple in checkpointer.alist(config, limit=limit):
            checkpoint = checkpoint_tuple.checkpoint
            
            entry = StateHistoryEntry(
                checkpoint_id=checkpoint.get("id", ""),
                thread_id=thread_id,
                parent_checkpoint_id=checkpoint_tuple.parent_config.get(
                    "configurable", {}
                ).get("checkpoint_id") if checkpoint_tuple.parent_config else None,
                channel_values=checkpoint.get("channel_values", {}),
                metadata=checkpoint_tuple.metadata or {},
            )
            entries.append(entry)
            
    except Exception as e:
        logger.warning(f"Failed to get state history for thread {thread_id}: {e}")
    
    return entries

async def get_latest_state(
    checkpointer: BaseCheckpointSaver,
    thread_id: str,
) -> dict[str, Any] | None:
    """
    Get the latest state for a thread.
    
    Args:
        checkpointer: The checkpointer instance
        thread_id: Thread to get state for
        
    Returns:
        The latest state dict, or None if no state exists
    """
    config = create_config(thread_id=thread_id)
    
    try:
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        if checkpoint_tuple:
            return checkpoint_tuple.checkpoint.get("channel_values", {})
    except Exception as e:
        logger.warning(f"Failed to get latest state for thread {thread_id}: {e}")
    
    return None


async def delete_thread(
    checkpointer: BaseCheckpointSaver,
    thread_id: str,
) -> bool:
    """
    Delete all checkpoints for a thread.
    
    Use this for:
    - GDPR compliance (right to be forgotten)
    - Cleanup of old conversations
    - Testing
    
    Args:
        checkpointer: The checkpointer instance
        thread_id: Thread to delete
        
    Returns:
        True if deletion was successful
        
    Note:
        Only PostgresSaver fully supports deletion.
        MemorySaver will log a warning and return False.
    """
    try:
        # PostgresSaver has adelete method
        if hasattr(checkpointer, "adelete"):
            config = create_config(thread_id=thread_id)
            await checkpointer.adelete(config)
            logger.info(f"Deleted thread {thread_id}")
            return True
        else:
            logger.warning(
                f"Checkpointer {type(checkpointer).__name__} "
                "does not support deletion"
            )
            return False
    except Exception as e:
        logger.error(f"Failed to delete thread {thread_id}: {e}")
        return False

# CONVENIENCE FUNCTIONS -----------------------------------------------------------

@asynccontextmanager
async def checkpointer_context(
    settings: Settings | None = None) -> AsyncGenerator[BaseCheckpointSaver, None]:
    """
    Context manager for easy checkpointer access.
    
    Automatically handles initialization and cleanup.
    Args:
        settings: Optional settings override
        
    Yields:
        Configured checkpointer instance
        
    Example:
        >>> async with checkpointer_context() as checkpointer:
        ...     graph = create_graph(checkpointer=checkpointer)
        ...     result = await graph.ainvoke(state, config)
    """

    manager = CheckpointerManager.from_settings(settings)
    
    try:
        await manager.initialize()
        yield manager.get_checkpointer()
    finally:
        await manager.close()

def create_memory_checkpointer() -> MemorySaver:
    """
    Quick helper to createan in-memory checkpointer.
    
    Useful for testing and simple scripts.
    
    Returns:
        MemorySaver instance
        
    Example:
        >>> checkpointer = create_memory_checkpointer()
        >>> graph = create_graph(checkpointer=checkpointer)
    """
    return MemorySaver()


# THREAD INFO FOR MULTI_USER SYSTEMS ----------------------------------------------

@dataclass
class ThreadInfo:
    """Info about a conversation thread."""
    
    thread_id: str
    """Unique thread identifier."""
    
    user_id: str | None = None
    """Optional user who owns this thread."""
    
    title: str | None = None
    """Optional title/description for the conversation."""
    
    created_at: datetime | None = None
    """When the thread was created."""
    
    updated_at: datetime | None = None
    """When the thread was last updated."""
    
    message_count: int = 0
    """Number of messages in the thread."""
    
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

class ThreadManager:
    """
    Manages threads for multi-user scenarios.
    
    This class provides higher-level thread management:
    - Thread creation with metadata
    - Thread listing and searching
    - User ownership tracking
    
    Note: This is a simple in-memory implementation.
    For production, store thread metadata in your database.
    
    Example:
        >>> manager = ThreadManager(checkpointer)
        >>> thread = await manager.create_thread(user_id="user-123")
        >>> threads = await manager.list_threads(user_id="user-123")
    """
    
    def __init__(self, checkpointer: BaseCheckpointSaver) -> None:
        self._checkpointer = checkpointer
        self._threads: dict[str, ThreadInfo] = {}  # In-memory store
    
    async def create_thread(
        self,
        user_id: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ThreadInfo:
        """
        Create a new conversation thread.
        
        Args:
            user_id: Optional owner of the thread
            title: Optional title/description
            metadata: Additional metadata
            
        Returns:
            ThreadInfo for the new thread
        """
        thread_id = generate_thread_id()
        now = datetime.utcnow()
        
        thread = ThreadInfo(
            thread_id=thread_id,
            user_id=user_id,
            title=title,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
        
        self._threads[thread_id] = thread
        logger.info(f"Created thread {thread_id} for user {user_id}")
        
        return thread
    
    async def get_thread(self, thread_id: str) -> ThreadInfo | None:
        """Get thread info by ID."""
        return self._threads.get(thread_id)
    
    async def list_threads(
        self,
        user_id: str | None = None,
        limit: int = 50,
    ) -> list[ThreadInfo]:
        """
        List threads, optionally filtered by user.
        
        Args:
            user_id: Filter to specific user's threads
            limit: Maximum threads to return
            
        Returns:
            List of ThreadInfo, sorted by updated_at descending
        """
        threads = list(self._threads.values())
        
        if user_id:
            threads = [t for t in threads if t.user_id == user_id]
        
        # Sort by updated_at descending (most recent first)
        threads.sort(
            key=lambda t: t.updated_at or datetime.min,
            reverse=True,
        )
        
        return threads[:limit]
    
    async def update_thread(
        self,
        thread_id: str,
        title: str | None = None,
        message_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ThreadInfo | None:
        """Update thread metadata."""
        thread = self._threads.get(thread_id)
        if not thread:
            return None
        
        if title is not None:
            thread.title = title
        if message_count is not None:
            thread.message_count = message_count
        if metadata is not None:
            thread.metadata.update(metadata)
        
        thread.updated_at = datetime.utcnow()
        return thread
    
    async def delete_thread(self, thread_id: str) -> bool:
        """
        Delete a thread and its checkpoints.
        
        Args:
            thread_id: Thread to delete
            
        Returns:
            True if deletion successful
        """
        if thread_id not in self._threads:
            return False
        
        # Delete from checkpointer
        await delete_thread(self._checkpointer, thread_id)
        
        # Delete from local store
        del self._threads[thread_id]
        
        logger.info(f"Deleted thread {thread_id}")
        return True
    
    def get_config(self, thread_id: str) -> dict[str, Any]:
        """Get LangGraph config dict for a thread."""
        return create_config(thread_id=thread_id)
    
# EXPORTS -----------------------------------------------------------------

__all__ = [
    # Thread ID Management
    "generate_thread_id",
    "create_config",
    "get_thread_id",
    # Checkpointer Setup
    "CheckpointerConfig",
    "CheckpointerManager",
    "checkpointer_context",
    "create_memory_checkpointer",
    # State History
    "StateHistoryEntry",
    "get_state_history",
    "get_latest_state",
    "delete_thread",
    # Thread Management
    "ThreadInfo",
    "ThreadManager",
]
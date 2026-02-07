from typing import Any

# BASE EXCEPTIONS --------------------------------------

class ResearchAgentError(Exceptions):
    """Base exceptions for all research agent errors.
    All custom exceptions inherit from this
    ```python
    try:
        await agent.run(query)
    except ResearchAgentError as e:
        # Handle any agent error
        logger.error(f"Agent error: {e}")
    ```
    """
    def __init__(self, message: str = "An error occured in the research agent",
                 details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }     

# TOOL ERRORS---------------------------------------
class ToolError(ResearchAgentError):
    """
    USAGE:
    ```python
    try:
        result = await search_tool.execute(query)
    except ToolError as e:
        logger.warning(f"Search failed: {e.tool_name} - {e}")
    """
    def __init__(self, message: str, tool_name: str, tool_args: dict[str, Any] | None= None,
                 cause: Exception | None = None,) -> None:
        super().__init__(message=f"Tool '{tool_name}' failed: {message}",
                         details={
                            "tool_name": tool_name,
                            "tool_args": tool_args or {},
                            "cause": str(cause) if cause else None,
                         },)
        self.tool_name = tool_name
        self.tool_args = tool_args or {}
        self.cause = cause
        
class ToolNotFoundError(ToolError):
    def __init__(self, tool_name: str, available_tools: list[str] | None = None) -> None:
        super().__init__(
            message=f"Tool not found. Available: {available_tools}",
            tool_name=tool_name,
        )
        self.available_tools = available_tools or []     

class ToolTimeoutError(ToolError):
    """Tool execution timed out."""
    
    def __init__(self, tool_name: str, timeout: float) -> None:
        super().__init__(
            message=f"Execution timed out after {timeout}s",
            tool_name=tool_name,
        )
        self.timeout = timeout
        
# LLM ERRORS ---------------------------------------------------------
              
class LLMErrors(ResearchAgentError):
    def __init__(self, message: str, model: str | None = None,
                 status_code: int | None = None,
                 cause: Exception | None = None,) -> None:
        super().__init__(
            message=f"LLM error: {message}",
            details={
                "model": model,
                "status_code": status_code,
                "cause": str(cause) if cause else None,
            },
        )
        self.model = model
        self.status_code = status_code
        self.cause = cause

class LLMRateLimitError(LLMError):
    """LLM API rate limit exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message=message, status_code=429)
        self.retry_after = retry_after


class LLMContextLengthError(LLMError):
    """Input exceeded LLM's context window."""
    
    def __init__(self, token_count: int, max_tokens: int) -> None:
        super().__init__(
            message=f"Context length exceeded: {token_count} > {max_tokens}",
        )
        self.token_count = token_count
        self.max_tokens = max_tokens

# GRAPH ERRORS ------------------------------------------------------

class GraphError(ResearchAgentError):
    def __init__(self, message: str, node: str | None = None,
                 state: dict[str, Any] | None = None,
                 cause: Exception | None = None) -> None:
        super().__init__(
            message=f"Graph error: {message}",
            details={
                "node": node,
                "state_keys": list(state.keys()) if state else None,
                "cause": str(cause) if cause else None,
            },
        )
        self.node = node
        self.state = state
        self.cause = cause   

class MaxIterationsError(GraphError):
    """Agent exceeded maximum iterations (possible infinite loop)."""
    
    def __init__(self, max_iterations: int, current_node: str | None = None) -> None:
        super().__init__(
            message=f"Exceeded maximum iterations ({max_iterations})",
            node=current_node,
        )
        self.max_iterations = max_iterations


class InvalidStateError(GraphError):
    """State validation failed."""
    
    def __init__(self, message: str, invalid_keys: list[str] | None = None) -> None:
        super().__init__(message=message)
        self.invalid_keys = invalid_keys or []     

# EXECUTION ERRORS --------------------------------------------------------
            
class ExecutionError(ResearchAgentError):
    """ Error during agent execution."""
    def __init__(
        self,
        message: str,
        step: str | None = None,
        partial_result: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(
            message=message,
            details={
                "step": step,
                "partial_result": partial_result[:500] if partial_result else None,
                "cause": str(cause) if cause else None,
            },
        )
        self.step = step
        self.partial_result = partial_result
        self.cause = cause


class TimeoutError(ExecutionError):
    """Agent execution timed out."""
    
    def __init__(self, timeout: float, step: str | None = None) -> None:
        super().__init__(
            message=f"Execution timed out after {timeout}s",
            step=step,
        )
        self.timeout = timeout


class CancellationError(ExecutionError):
    """Agent execution was cancelled."""
    
    def __init__(self, reason: str = "User requested cancellation") -> None:
        super().__init__(message=f"Execution cancelled: {reason}")
        self.reason = reason 
        

# API ERRORS (for FastAPI integration) ------------------------------------

class APIError(ResearchAgentError):
    """ Includes HTTP status code for response handling."""
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        code: str = "INTERNAL_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message=message, details=details)
        self.status_code = status_code
        self.code = code
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to API error response format."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }
        
# MCP ERRORS --------------------------------------------------------------

class MCPError(ResearchAgentError):
    """Error related to Model Context Protocol."""
    
    def __init__(
        self,
        message: str,
        server_name: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(
            message=f"MCP error: {message}",
            details={
                "server_name": server_name,
                "cause": str(cause) if cause else None,
            },
        )
        self.server_name = server_name
        self.cause = cause

class MCPConnectionError(MCPError):
    """Failed to connect to MCP server."""
    pass

class MCPToolError(MCPError):
    """MCP tool execution failed."""
    
    def __init__(
        self,
        message: str,
        tool_name: str,
        server_name: str | None = None,
    ) -> None:
        super().__init__(message=message, server_name=server_name)
        self.tool_name = tool_name


# EXPORTS -------------------------------------------------------

__all__ = [
    # Base
    "ResearchAgentError",
    # Tool errors
    "ToolError",
    "ToolNotFoundError",
    "ToolTimeoutError",
    # LLM errors
    "LLMError",
    "LLMRateLimitError",
    "LLMContextLengthError",
    # Graph errors
    "GraphError",
    "MaxIterationsError",
    "InvalidStateError",
    # Execution errors
    "ExecutionError",
    "TimeoutError",
    "CancellationError",
    # API errors
    "APIError",
    # MCP errors
    "MCPError",
    "MCPConnectionError",
    "MCPToolError",
]                                                                  
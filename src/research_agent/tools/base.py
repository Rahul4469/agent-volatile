from abc import ABC, abstractmethod
from typing import Any, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from research_agent.core.exceptions import ToolError

# PYDANTIC MODELS FOR TOOL INPUTS----------------------------------------------

class SearchInput(BaseModel):
    """Input schema for search tools.
    query:str, max_results:int.
    """
    query: str = Field(description="The search query to execute",
                       min_lenght=1,
                       max_length=500)
    max_results: int = Field(default=5,
                             description="Maximum number of results to return",
                             ge=1,
                             le=20)

class FetchInput(BaseModel):
    """Input schema for page fetch tools."""
    
    url: str = Field(description="URL of the web pafe to fetch")
    
    extract_text: bool = Field(default=True, 
                               description="Whether to extract text content(vs raw HTML)")

class CalculatorInput(BaseModel):
    """ Innput schema for calculator tool."""
    
    expression: str = Field(description="Mathematical expression to evaluate (e.g., '2 + 2 * 3')")

# ABSTRACT TOOL BASE CLASS -----------------------------------------------------  

class ResearchTool(BaseTool, ABC):
    """
    TO CREATE A TOOL:
    1. Subclass ResearchTool
    2. Set name, description, args_schema
    3. Implement _execute() method
    """
    # configs
    return_direct: bool = False     # Whether to return error directly (vs raising)
    handle_tool_error: bool = True  # Whether this tool handles its own errors 
    
    @abstractmethod
    async def _execute(self, **kwargs: Any) -> str:
        ...   
    
    # LANGCHAIN INTERFACE IMPLEMENTATION
    
    # synchronous, but has async _execute()
    def _run(self, *args: Any, 
    run_manager: CallbackManagerForToolRun | None = None,
    **kwargs: Any) -> str:
        import asyncio
        try:
            loop = asyncio.get_running_loop()                                # 1. Check: "Are we in async context?"
            # If there's a running loop, we need to run in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:            # 2. Use thread pool
                future = pool.submit(asyncio.run, self._execute(**kwargs))   # 3. Run async in thread
                return future.result()                                       # 4. Wait & get result
        except RuntimeError:                                                 # 5. No loop = sync 
            # No running loop, safeto use asyncio.run
            return asyncio.run(self._execute(**kwargs))   
    
    async def _arun(self, *args: Any, run_manager: CallbackManagerForToolRun | None = None,
    **kwargs: Any) -> str:
        try:
            return await self._execute(**kwargs)
        except ToolError: 
            # Re-raise our custom errors
            raise 
        except Exception as e:
            # Wrap unexpected errors
            raise ToolError(
                message=str(e),
                tool_name=self.name,
                tool_args=kwargs,
                cause=e
            ) from e

# \\\ TOOL DECORATOR WRAPPER ///----------------------------------------------

def create_tool(func, *, name:str | None = None,
                description: str | None = None,
                return_direct: bool = False):
    """
    Create a tool from a function with enhanced error handling.
    This wraps LangChain's @tool decorator with our error handling.
    """
    
    decorator = tool( func, return_direct=return_direct)
    
    # Override name/description if provided
    if name:
        decorator.name = name
    if description:
        decorator.description = description
    
    return decorator

# TOOL RESULT FORMATTING -------------------------------------------

def format_tool_result(result: Any, max_length: int = 5000) -> str:
    """
    Tools can return various types - this converts them to a string
    that's useful for the LLM.
    Args:
        result: Raw tool result
        max_length: Maximum result length (truncate if exceeded)
    Returns:
        Formatted string result
    """
    # Convert to string based on type
    if isinstance(result, str):
        text = result
    elif isinstance(result, dict):
        import json
        text = json.dumps(list(result), indent = 2, default=str)  
    elif isinstance(result, (list, tuple)):
        text = json.dumps(list(result), indent=2, default=str)
    else:
        text = str(result)
        
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "\n... (truncate)"
        
    return text     

def format_error_result(error: Exception) -> str:
    """
    Format an error for the LLM. 
    When a tool fails, we want to give the LLM useful information
    so it can try a different approach.
    Args:
        error: The exception that occurred
    Returns:
        Error message string
    """
    if isinstance(error, ToolError):
        return f"Tool Error: {error.message}"
    return f"Error: {str(error)}"      

# TOOL METADATA -----------------------------------------------
class ToolMetadata(BaseModel):
    """
    Metadata about a tool for documentation and discovery.
    Used by the API to list available tools.
    """           
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    parameters: dict[str, Any] = Field(
        description="JSON schema of tool parameters "
    )
    
    @classmethod # Called on class OR instance: ClassName.method() or instance.method()
    def from_tool(cls, tool: BaseTool) -> "ToolMetadata":
        """Create metadata from LangChain tool."""
        #Get JOSN schema from args_schema
        if tool.args_schema:
            params = tool.args_schema.model_json_schema() # model_json_schema() from pydantic
        else:
            params = {}
            
        return cls(
            name=tool.name,
            description=tool.description or "",
            parameters=params,
        )

# EXPORTS--------------------------------------------------------
__all__ = [
    # Input schemas
    "SearchInput",
    "FetchInput",
    "CalculatorInput",
    # Base classes
    "ResearchTool",
    # Decorators
    "create_tool",
    # Utilities
    "format_tool_result",
    "format_error_result",
    "ToolMetadata",
]
                            
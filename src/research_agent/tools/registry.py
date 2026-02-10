"""
Dynamic TOol Management
Instead of hardcoding tools, the registry allows:
1. Dynamic tool registration at runtime
2. Easy enabling/disabling of tools
3. Plugin architecture for future extensions
4. Tool discovery for API endpoints
"""
from typing import Callable

from langchain_core.tools import BaseTool

from research_agent.core.exceptions import ToolNotFoundError
from research_agent.tools.base import ToolMetadata

class ToolRegistry:
    """ """
    def __init__(self) -> None:
        """Init an empty registry"""
        self._tools: dict[str, BaseTool] = {}
        self._disabled: set[str] = set()
        
    def register(self, tool: BaseTool) -> None:
        # """
        # Args:
        #     tool: LangChain tool to register
        # USAGE:
        # ```python
        # from langchain_core.tools import tool
        
        # @tool
        # def my_tool(query: str) -> str:
        #     '''My tool description.'''
        #     return "result"
        
        # registry.register(my_tool)
        # """ 
        self._tools[tool.name] = tool
    
    def registry_many(self, tools: list[BaseTool]) -> None:
        for tool in tools:
            self.register(tool)       
    
    def unregister(self, name: str) -> BaseTool | None:
        return self._tools.pop(name, None)        
    
    # ENABLING/DISABLING ///////////
    
    def disable(self, name: str) -> None:
        """disable a tool withou removing it."""
        if name not in self._tools:
            raise ToolNotFoundError(name, available_tools=list(self._tools.keys()))
        self._disabled.add(name)
    
    def enable(self, name: str) -> None:
        """
        Enable a previously disabled tool.
        """
        self._disabled.discard(name)
    
    def is_enabled(self, name: str) -> bool:
        """Check if a tool is enabled."""
        return name in self._tools and name not in self._disabled                         

    # RETRIEVAL  //////////////////
    
    def get(self, name: str) -> BaseTool:
        """get tool by name"""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(name, available_tools=list(self._tools.keys()))
        return tool
    
    def get_tools(self, include_disabled: bool = False) -> list[BaseTool]:
        """ get all enabled tools."""
        if include_disabled:
           return list[self._tools.values()]
        
        return [
            tool for name, tool in self._tools.items()
            if name not in self._disabled
        ] 
    
    def get_tool_names(self) -> list[str]:
        """Get names of all registered tools."""
        return list(self._tools.keys())
    
    def get_enabled_names(self) -> list[str]:
        """Get names of enabled tools only."""
        return [name for name in self._tools if name not in self._disabled]
    
    # METADATA ////////////////////////////
    
    def list_metadata(self, include_disabled: bool = False) -> list[ToolMetadata]:
        """
        Get metadata for all tools (for API responses).
        Args:
            include_disabled: Include disabled tools 
        Returns:
            List of tool metadata
        USAGE:
        ```python
        # In FastAPI endpoint
        @router.get("/tools")
        def list_tools():
            return registry.list_metadata()
        ```
        """
        tools = self.get_tools(include_disabled=include_disabled)
        return [ToolMetadata.from_tool(tool) for tool in tools]
    
    # DUNDER METHODS ///////////////////
    
    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)
    
    def __iter__(self):
        """Iterate over enabled tools."""
        return iter(self.get_tools())
    
    def __repr__(self) -> str:
        enabled = len(self.get_tools())
        total = len(self._tools)
        return f"ToolRegistry(enabled={enabled}, total={total}, tools={self.get_tool_names()})" 

# DEFAULT REGISTRY (SINGLETON) ----------------------------------------------------
# This is the global registry instance. Import and use directly:
#
# ```python
# from research_agent.tools.registry import default_registry
# default_registry.register(my_tool)
# ```

default_registry = ToolRegistry() 

# FACTORY FUNCTION --------------------------------------------------------

def create_default_registry() -> ToolRegistry:
    """
    Create a new registry with default tools registered.
    """
    from research_agent.tools.calculator import calculator_tool
    from research_agent.tools.web.search import create_search_tool
    from research_agent.tools.web.fetch import create_fetch_tool
    
    registry = ToolRegistry()
    
    # Register default tools
    registry.register(calculator_tool)
    registry.register(create_search_tool())
    registry.register(create_fetch_tool())
    
    return registry   

# EXPORTS -------------------------------------------------------------------

__all__ = [
    "ToolRegistry",
    "default_registry",
    "create_default_registry",
]      
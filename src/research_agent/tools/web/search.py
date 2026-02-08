"""
Models,
SearchProviders [Brave, tavily, serpapi, mock],
WebSearchTool,
create_search_tool [config + tool]
"""

from abc import ABC, abstractmethod
from typing import Any

import httpx

from pydantic import BaseModel, Field

from research_agent.config import get_settings
from research_agent.core.exceptions import ToolError
from research_agent.tools.base import ResearchTool, SearchInput, format_tool_result

# SEARCH RESULT MODEL ------------------------------------------------------------

class SearchResult(BaseModel):
    """A single search result Standard for all providers.
    title, url, snippet, to_string(). 
    """
    title: str = Field(description="Result title")
    url: str = Field(description="Result URL")
    snippet: str = Field(description="Text snippet/description")
    
    def to_string(self) -> str:
        """Format or LLM consumption."""
        return f"**{self.title}**\n{self.url}\n{self.snippet}"
    
class SearchResponse(BaseModel):
    """
    Response from a search query.       
    query, result, total:
    """
    query: str = Field(description="Original query")
    result: list[SearchResult] = Field(description="Search results")
    total: int = Field(description="Total results found")
    
    def to_string(self) -> str:
        """Format all results for LLM consumption."""
        if not self.results:
            return f"No results found for: {self.query}"  
        
        formatted = [f"Search results for: {self.query}\n"]
        for i, result in enumerate(self.results, 1):
            formatted.append(f"\n. {result.to_string()}") 
        
        return "\n".join(formatted)
    
# SEARCH PROVIDERS (Strategy Pattern)--------------------------------------

class SearchProvider(ABC):
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        ...
    
    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        """
        Execute a search query.
        Args:
            query: Search query
            max_results: Maximum results to return
        Returns:
            SearchResponse with results 
        """
        ...

class MockSearchProvider(SearchProvider):
    
    @property
    def name(self) -> str:
        return "mock"
    
    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        """Return mock search results."""
        # Generate mock results based on query
        mock_results = [
            SearchResult(
                title=f"Result {i}: {query[:30]}...",
                url=f"https://example.com/result{i}",
                snippet=f"This is a mock search result for '{query}'. "
                        f"In production, this would contain real content from the web.",
            )
            for i in range(1, min(max_results + 1, 4))
        ]

        return SearchResponse(
            query=query,
            results=mock_results,
            total=len(mock_results),
        )      

class BraveSearchProvider(SearchProvider):
    
    BASE_URL = "https://api.search.brave.com/res/v1/web/search"
    
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None
    
    @property
    def name(self) -> str:
        return "brave"
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={"X-Subscription-Token": self.api_key},
            )
        return self._client   
    
    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        """Execute Brave Search."""
        client = await self._get_client()
        
        try:
            response = await client.get(
                self.BASE_URL,
                params={
                    "q": query,
                    "count": max_results,
                },
            ) 
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("web", {}).get("results", [])[:max_results]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", ""),
                ))
            
            return SearchResponse(
                query=query,
                results=results,
                total=len(results)
            ) 
        except httpx.HTTPError as e:
            raise ToolError(
                message=f"Brave search failed: {e}",
                tool_name="web_search",
                tool_args={"query": query},
                cause=e
            )         

class TavilySearchProvider(SearchProvider):
    """
    Tavily Search API provider.
    
    Tavily is optimized for AI applications - it provides
    summarized content that's easy for LLMs to process.
    Get API key at: https://tavily.com/
    """
    
    BASE_URL = "https://api.tavily.com/search"
    
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None
    
    @property
    def name(self) -> str:
        return "tavily"
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        """Execute Tavily search."""
        client = await self._get_client()
        
        try:
            response = await client.post(
                self.BASE_URL,
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("results", [])[:max_results]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", "")[:500],
                ))
            
            return SearchResponse(
                query=query,
                results=results,
                total=len(results),
            )
        
        except httpx.HTTPError as e:
            raise ToolError(
                message=f"Tavily search failed: {e}",
                tool_name="web_search",
                tool_args={"query": query},
                cause=e,
            )   

class SerpAPISearchProvider(SearchProvider):
    """
    SerpAPI provider (Google Search results).
    API key at: https://serpapi.com/
    """ 
    BASE_URL = "https://serpapi.com/search"
    
    def __init__(self, api_key: str) -> str:
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None
        
    @property
    def name(self) -> str:
        return "serpapi"
    
    # Open a new HTTP connection pool if not alreaddy open
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        """Execute SerpAPI search."""
        client = await self._get_client()
        
        try:
            respsonse = await client.get(
                self.BASE_URL,
                params={
                    "api_key": self.api_key,
                    "q": query,
                    "num": max_results,
                    "engine": "google",
                },
            )
            respsonse.raise_for_status()
            data = respsonse.json()

            results = []
            for item in data.get("organic_results", [])[:max_results]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                ))
            
            return SearchResponse(
                query=query,
                results=results,
                total=len(results),
            )    
        except httpx.HTTPError as e:
            raise ToolError(
                message=f"SerpAPI search failed: {e}",
                tool_name="web_search",
                tool_args={"query": query},
                cause=e,
            )
            
# WEB SEARCH TOOL -------------------------------------------------------------

class WebSearchTool(ResearchTool):
    """
    USAGE:
    ```python
    tool = WebSearchTool(provider="brave", api_key="...")
    result = await tool._execute(query="latest AI news")
    ```
    """
    name: str = "web_search"
    description: str = """Search the web for current information.
        Use this when you need up-to-date information, recent events, or facts you're unsure about.
        Returns a list of relevant web pages with titles, URLs, and snippets."""
    args_schema: type = SearchInput
    
    # Configuration
    provider_name: str = "mock"
    api_key: str = ""
    max_results: int = 5
    
    # Provider instance (set in __init__)
    _provider: SearchProvider | None = None
    
    def __init__(self, provider: str | None = None, api_key: str = "", max_results: int = 5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        settings = get_settings()
        
        self.provider_name = provider or settings.web_search_provider
        self.api_key = api_key or settings.web_search_api_key
        self.max_results = max_results or settings.web_search_max_results
        self._provider = self._create_provider()
    
    def _create_provider(self) -> SearchProvider:
        providers = {
            "mock": lambda: MockSearchProvider(),
            "brave": lambda: BraveSearchProvider(self.api_key),
            "tavily": lambda: TavilySearchProvider(self.api_key),
            "serpapi": lambda: SerpAPISearchProvider(self.api_key),
        }
        
        factory = providers.get(self.provider_name.lower())
        if factory is None:
            raise ValueError(
                f"Unknown provider: {self.provider_name}. "
                f"Available: {list(providers.keys())}"
            )
        
        return factory()
    
    async def _execute(self, query: str, max_results: int | None = None) -> str:
        """
        Args:
            query: Search query
            max_results: Maximum results (uses default if not specified)
        Returns:
            Formatted search results
        """
        if self._provider is None:
            self._provider = self._create_provider()
        
        # Use provided max_results or default
        results_count = max_results or self.max_results
        
        # Execute search
        response = await self._provider.search(query, results_count)
        
        # Format for LLM
        return response.to_string()
    
# FACTORY FUNCTION ----------------------------------------------------------------

def create_search_tool(
    provider: str | None = None,
    api_key: str | None = None) -> WebSearchTool:
    """
    Create a web search tool with configuration from settings.
    
    Args:
        provider: Override provider (uses settings if None)
        api_key: Override API key (uses settings if None)
    
    Returns:
        Configured WebSearchTool
    
    USAGE:
    ```python
    # Use settings
    tool = create_search_tool()
    
    # Override provider
    tool = create_search_tool(provider="brave", api_key="...")
    ```
    """
    settings = get_settings()
    
    return WebSearchTool(
        provider= provider or settings.web_search_provider,
        api_key= api_key or settings.web_search_api_key,
        max_results= settings.web_search_max_results,
    )        
    

# EXPORTS------------------------------------------------------------------

__all__ = [
    "SearchResult",
    "SearchResponse",
    "SearchProvider",
    "MockSearchProvider",
    "BraveSearchProvider",
    "TavilySearchProvider",
    "SerpAPISearchProvider",
    "WebSearchTool",
    "create_search_tool",
]    
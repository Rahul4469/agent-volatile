"""
Web tools for the research agent.
"""

from research_agent.tools.web.search import (
    SearchResult,
    SearchResponse,
    SearchProvider,
    MockSearchProvider,
    BraveSearchProvider,
    TavilySearchProvider,
    SerpAPISearchProvider,
    WebSearchTool,
    create_search_tool,
)

from research_agent.tools.web.fetch import (
    PageContent,
    ContentExtractor,
    FetchPageTool,
    create_fetch_tool,
)

__all__ = [
    # Search
    "SearchResult",
    "SearchResponse",
    "SearchProvider",
    "MockSearchProvider",
    "BraveSearchProvider",
    "TavilySearchProvider",
    "SerpAPISearchProvider",
    "WebSearchTool",
    "create_search_tool",
    # Fetch
    "PageContent",
    "ContentExtractor",
    "FetchPageTool",
    "create_fetch_tool",
]
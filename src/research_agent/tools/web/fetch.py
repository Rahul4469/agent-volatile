"""httpx + BeautifulSoup to retrieve and extract html web page content.
PageContent model,
ContentExtractor extract() _extract_text() _clean_text(),
FetchPageTool,
create_fetch_tool [configs+tool]
"""

import re
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from pydantic import BaseModel, Field

from research_agent.core.exceptions import ToolError
from research_agent.tools.base import ResearchTool, FetchInput

# PAGE CONTENT MODEL -------------------------------------------------------

class PageContent(BaseModel):
    """
    Model.
    Extract content from a web page.
    url, title, text, word_count
    """
    url: str = Field(description="Source URL")
    title: str = Field(description="Page title")
    text: str = Field(description="Extract text content")
    word_count: int = Field(description="Approximate word count")
    
    def to_string(self, max_length: int = 10000) -> str:
        """Format for LLM consumption"""
        text = self.text
        if len(text) > max_length:
            text = text[:max_length] + "\n\n... (content truncated)"
            
        return f"""# {self.title}
    Source: {self.url}
    Words: ~{self.word_count}
    
    {text}"""
    
# CONTENT EXTRACTION -------------------------------------------------------

class ContentExtractor:
    # Elements to remove (navigation, ads, etc.)
    REMOVE_TAGS = [
        "script", "style", "nav", "header", "footer", "aside",
        "form", "button", "input", "select", "textarea",
        "iframe", "noscript", "svg", "canvas",
        "[class*='nav']", "[class*='menu']", "[class*='sidebar']",
        "[class*='footer']", "[class*='header']", "[class*='ad']",
        "[class*='comment']", "[class*='share']", "[class*='social']",
        "[id*='nav']", "[id*='menu']", "[id*='sidebar']",
        "[id*='footer']", "[id*='header']", "[id*='ad']",
    ]
    
    # Main content selectors (priority order)
    MAIN_CONTENT_SELECTORS = [
        "article",
        "main",
        "[role='main']",
        ".post-content",
        ".article-content",
        ".entry-content",
        ".content",
        "#content",
        ".post",
        ".article",
    ]
    
    def extract(self, html: str, url: str = "") -> PageContent:
        """
        Extract readable content from HTML.
        Args:
            html: Raw HTML content
            url: Source URL (for reference)
        Returns:
            PageContent with extracted text
        """
        # PArse HTML
        soup = BeautifulSoup(html, "lxml")
        
        # Get title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else "Untitled"
        
        # Remove unwanted elements
        for selector in self.REMOVE_TAGS:
            for element in soup.select(selector):
                element.decompose()
        
        # Find main content
        main_content = None
        for selector in self.MAIN_CONTENT_SELECTORS:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # Fall back to body
        if main_content is None:
            main_content = soup.find("body")
        
        if main_content is None:
            main_content = soup
        
        # Extract text
        text = self._extract_text(main_content)
        
        # Clean up
        text = self._clean_text(text)
        
        # Count words
        word_count = len(text.split())
        
        return PageContent(
            url=url,
            title=title,
            text=text,
            word_count=word_count,
        )
    
    def _extract_text(self, element) -> str:
        """
        Extract text from an element, preserving some structure.
        We want to keep paragraph breaks for readability.
        """
        texts = []
        
        for child in element.descendants:
            if child.name in ["p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li"]:
                texts.append("\n")
            elif hasattr(child, "string") and child.string:
                text = child.string.strip()
                if text:
                    texts.append(text + " ")
        
        return "".join(texts)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean up extracted text.
        - Remove excessive whitespace
        - Remove empty lines
        - Normalize line breaks
        """
        # Replace multiple spaces with single space
        text = re.sub(r"[ \t]+", " ", text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        
        # Strip leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        
        # Remove lines that are just punctuation or very short
        lines = [
            line for line in text.split("\n")
            if len(line) > 3 or line == ""
        ]
        text = "\n".join(lines)
        
        return text.strip()
    
# FETCH PAGE TOOL ------------------------------------------------------------

class FetchPageTool(ResearchTool):
    """
    Fetch and extract content from web page.
    USAGE:
    ```python
    tool = FetchPageTool()
    content = await tool._execute(url="https://example.com/article")
    ```
    """
    name: str = "fetch_page"
    description: str = """Fetch and read the full content of a web page.
        Use this after searching to get the complete text of a promising result.
        Provide the exact URL you want to fetch.
        Note: Some pages may not work (JavaScript-rendered, paywalled, etc.).
        """
    args_schema: type = FetchInput
    
    # Configuration
    timeout: float = 30.0
    max_content_length: int = 10000
    user_agent: str = (
        "Mozilla/5.0 (compatible; ResearchBot/1.0; +https://example.com/bot)"
    )
    
    # HTTP client (lazy initialized)
    _client: httpx.AsyncClient | None = None
    _extractor: ContentExtractor | None = None
    
    def __init__(self, timeout: float = 30.0, 
                 max_content_length: int = 10000,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.timeout = timeout
        self.max_content_length = max_content_length
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout = self.timeout,
                follow_redirects = True,
                headers = {
                    "User-Agent": self.user_agent,
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
        return self._client
    
    def _get_extractor(self) -> ContentExtractor:
        """Get or create content extractor."""
        if self._extractor is None:
            self._extractor = ContentExtractor()
        return self._extractor
    
    def _validate_url(self, url: str) -> str:
        """Validate and normalize URL."""
        
        # Add schema if missing
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        
        # Parse and validate
        parsed = urlparse(url)
        
        if not parsed.netloc:
            raise ValueError(f"Invlaid URL: {url}")
        
        # Block potentially dangerous URLS
        blocked_schemes = ["file", "javascript", "data"]
        if parsed.scheme.lower() in blocked_schemes:
            raise ValueError(f"Blocked URL scheme: {parsed.scheme}")
        
        return url
    
    # 1. Validate url
    # 2. Fetch Page
    # 3. Check content type
    # 4. Extract content
    async def _execute(self, url: str, extract_text: bool = True) -> str:
        """
        Fetch a web page and extract its content.
        Args:
            url: URL to fetch
            extract_text: If True, extract clean text. If False, return raw HTML.
        Returns:
            Extracted content or error message
        """
        try:
            # Validate URL
            url = self._validate_url(url)
            
            # Fetch page
            client = await self._get_client()
            response = await client.get(url)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get("content-type", "")
            if "html" not in content_type.lower() and "text" not in content_type.lower():
                return f"Cannot process content type: {content_type}"
            
            html = response.text
            
            if not extract_text:
                # Return raw HTML (truncated)
                if len(html) > self.max_content_length:
                    html = html[:self.max_content_length] + "\n... (truncated)"
                return html
            
            # Extract content
            extractor = self._get_extractor()
            content = extractor.extract(html, url)
            
            return content.to_string(max_length=self.max_content_length)
            
            
        except httpx.TimeoutException:
            return f"Timeout: The page took too long to load ({url})"
        
        except httpx.HTTPStatusError as e:
            return f"HTTP error {e.response.status_code}: Could not fetch {url}"
        
        except httpx.RequestError as e:
            return f"Request failed: {e}"
        
        except ValueError as e:
            return f"Invalid URL: {e}"
        
        except Exception as e:
            raise ToolError(
                message=str(e),
                tool_name=self.name,
                tool_args={"url": url},
                cause=e,
            )
            
# FACTORY FUNCTION ------------------------------------------------------------

def create_fetch_tool(timeout: float = 30.0, max_content_length: int = 10000) -> FetchPageTool:
    """
    Create a page fetch tool.
    
    Args:
        timeout: HTTP request timeout
        max_content_length: Maximum content length to return
    
    Returns:
        Configured FetchPageTool
    """
    return FetchPageTool(
        timeout=timeout,
        max_content_length=max_content_length,
    )
    

# EXPORTS

__all__ = [
    "PageContent",
    "ContentExtractor",
    "FetchPageTool",
    "create_fetch_tool",
]
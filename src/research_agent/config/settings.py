""" LRU cache for .env variables loading"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application configuration loaded from enc variables"""
    
    # PYDANTIC SETTINGS CONFIGURATION----------------------
    # This is the thing that does the work
    model_config = SettingsConfigDict(
        # Load from .env file if it exists
        env_file=".env",
        env_file_encoding="utf-8",
        
        case_sensitive=False,
        extra="ignore",
    )
    
    # APPLICATION SETTINGS --------------------------------
    
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment",
    )
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    
    # SERVER SETTINGS ------------------------------------
    
    host: str = Field(
        default="0.0.0.0",
        description="Server bind address",
    )
    
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Server port",
    )
    
    # LLM SETTINGS --------------------------------------
    
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key (sk-ant-...)",
    )
    
    # Default model for the agent
    # claude-sonnet-4-20250514 is the latest Sonnet (fast, capable)
    # claude-opus-4-20250514 for complex reasoning tasks
    default_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Default Claude model to use",
    )
    # Maximum tokens for LLM responses
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=200000,
        description="Maximum tokens in LLM response",
    )
    
    # Temperature for LLM responses (0 = deterministic, 1 = creative)
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="LLM temperature (0-1)",
    )
    
    # AGENT SETTINGS -------------------------------------------
    
    # Maximum iterations for the ReAct loop
    # Prevents infinite loops if the agent gets stuck
    max_iterations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum ReAct loop iterations",
    )
    
    # Timeout for each agent step (seconds)
    step_timeout: int = Field(
        default=60,
        ge=1,
        description="Timeout per agent step in seconds",
    )
    
    # WEB SEARCH SETTINGS -----------------------------------------
    
    # Web search provider: brave, serpapi, tavily, or mock
    web_search_provider: Literal["brave", "serpapi", "tavily", "mock"] = Field(
        default="mock",
        description="Web search API provider",
    )
    
    # API key for the web search provider
    web_search_api_key: str = Field(
        default="",
        description="API key for web search provider",
    )
    
    # Maximum results per search
    web_search_max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum search results to return",
    )
    
    # DATABASE SETTINGS (for checkpointing) --------------------------
    
    # PostgreSQL connection URL for state persistence
    # Format: postgresql://user:password@host:port/database
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/research_agent",
        description="PostgreSQL connection URL for checkpointing",
    )
    
    # Use in-memory checkpointer (for development)
    # Set to False in production to use PostgreSQL
    use_memory_checkpointer: bool = Field(
        default=True,
        description="Use in-memory checkpointer (dev only)",
    )
    
    
    # MCP SETTINGS ------------------------------------------------
    
    
    # Enable MCP server
    mcp_enabled: bool = Field(
        default=True,
        description="Enable Model Context Protocol server",
    )
    
    # MCP server port (if running HTTP transport)
    mcp_port: int = Field(
        default=8001,
        description="MCP server port",
    )
    
    
    # VALIDATORS -----------------------------------------------------
    
    
    @field_validator("anthropic_api_key")
    @classmethod
    def validate_api_key_format(cls, v: str) -> str:
        """
        Validate Anthropic API key format.
        
        Keys should start with 'sk-ant-' (but we're lenient for testing).
        """
        if v and not v.startswith(("sk-ant-", "sk-", "test-")):
            # Warning only, don't fail (might be testing)
            pass
        return v
    
    
    # COMPUTED PROPERTIES -------------------------------------------
    
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"
    
    @property
    def debug(self) -> bool:
        """Debug mode enabled (development + DEBUG log level)."""
        return self.is_development and self.log_level == "DEBUG"



# SINGLETON PATTERN --------------------------------------------------

# We use @lru_cache to ensure only one Settings instance exists.
# This is important because:
#   1. Loading .env files is slow
#   2. We want consistent configuration across the app
#   3. Validation runs only once
#
# GO COMPARISON:
# ```go
# var (
#     config     *Config
#     configOnce sync.Once
# )
#
# func GetConfig() *Config {
#     configOnce.Do(func() {
#         config = loadConfig()
#     })
#     return config
# }
# ```
#
# Python's @lru_cache achieves the same thing more elegantly.


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get the application settings singleton.
    
    First call loads from environment, subsequent calls return cached instance.
    
    USAGE:
    ```python
    from research_agent.config import get_settings
    
    settings = get_settings()
    print(settings.anthropic_api_key)
    ```
    """
    return Settings()



# DEVELOPMENT HELPER ---------------------------------------------------


if __name__ == "__main__":
    """Print current settings (useful for debugging)."""
    settings = get_settings()
    
    print("=" * 60)
    print("CURRENT SETTINGS")
    print("=" * 60)
    
    for key, value in settings.model_dump().items():
        # Hide sensitive values
        if "key" in key.lower() or "password" in key.lower():
            display = "***" if value else "(not set)"
        else:
            display = value
        print(f"  {key}: {display}")
    
    print("=" * 60)
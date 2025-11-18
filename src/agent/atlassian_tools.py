"""Atlassian MCP Tools for LangChain Agent.

This module uses langchain-mcp-adapters to connect to Atlassian MCP server.
The MCP server handles OAuth authentication automatically via callback.

Documentation:
- LangChain MCP: https://docs.langchain.com/oss/python/langchain/mcp
- Atlassian MCP: https://support.atlassian.com/atlassian-rovo-mcp-server/
"""

import logging
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)


async def get_atlassian_mcp_tools() -> list[Any]:
    """Get Atlassian MCP tools using langchain-mcp-adapters.

    This connects to Atlassian Remote MCP Server via mcp-remote proxy.
    The proxy is automatically spawned and handles OAuth authentication.

    On first run, a browser window will open for OAuth authorization.
    The token is cached, so subsequent runs won't require re-authentication.

    Requirements:
        - Node.js v18+ installed
        - npx available in PATH

    Returns:
        List of LangChain tools from Atlassian MCP server

    """
    logger.info("🔌 Starting mcp-remote proxy and connecting to Atlassian...")

    # Configure Atlassian MCP server with mcp-remote proxy
    # Using stdio transport to communicate with the local proxy process
    # The proxy handles OAuth and forwards requests to Atlassian Remote MCP
    client = MultiServerMCPClient(
        {
            "atlassian": {
                "transport": "stdio",  # Communicate with local proxy via stdin/stdout
                "command": "npx",  # Use npx to run mcp-remote
                "args": [
                    "-y",  # Auto-install if needed
                    "mcp-remote",
                    "https://mcp.atlassian.com/v1/sse",
                ],
            },
            
            # "atlassian": {
            # "transport": "sse",  # Server-Sent Events
            # "url": "https://mcp.atlassian.com/v1/sse",
            # }
        },
    )

    # Get tools from MCP server
    # First run: Browser will open for OAuth authorization
    # Subsequent runs: Cached token will be used
    logger.info("⏳ Fetching tools (may open browser for OAuth on first run)...")
    tools = await client.get_tools()

    logger.info("✅ Connected to Atlassian MCP. Found %d tools", len(tools))
    
    # Log first tool schema to check if cloudId is auto-injected
    if tools:
        logger.info("📋 Sample tool schema: %s", tools[0].name if hasattr(tools[0], 'name') else str(tools[0])[:200])

    return tools


# For backward compatibility - will be populated when agent is created
ATLASSIAN_TOOLS = []

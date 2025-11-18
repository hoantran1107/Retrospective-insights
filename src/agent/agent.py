"""AI Agent for Retrospective Insights Analysis with Atlassian MCP.

This module creates a LangChain agent that can analyze metrics,
generate hypotheses, and access Jira/Confluence via Atlassian MCP.

Uses langchain-mcp-adapters to connect to Atlassian Remote MCP Server.
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from openai import azure_endpoint

from .atlassian_tools import get_atlassian_mcp_tools

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

async def create_retrospective_agent(
    model: str = None,
    enable_atlassian: bool = True,
) -> Any:
    """Create LangChain agent for retrospective analysis.

    Args:
        model: Model name (defaults to env AZURE_OPENAI_DEPLOYMENT)
        enable_atlassian: Enable Atlassian MCP tools (default: True)

    Returns:
        Configured LangChain agent

    """
    model_name = model or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")

    # Load tools from Atlassian MCP server
    if enable_atlassian:
        tools = await get_atlassian_mcp_tools()
    else:
        tools = []

    logger.info(
        "Creating agent with %d tools (Atlassian: %s)",
        len(tools),
        enable_atlassian,
    )
    from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
    from langchain_openai import AzureChatOpenAI

    endpoint = os.environ["AZURE_OPENAI_ENDPOINT_AGENT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    llm = AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=model_name,
    )
    # Create agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""You are an expert AI assistant for analyzing team retrospective metrics and Jira data.

Your capabilities:
1. Search Jira issues and Confluence pages (via Atlassian MCP)
2. Analyze sprint data, velocity, and team performance
3. Extract insights from retrospective notes and comments
4. Generate data-driven hypotheses about team performance
5. Provide actionable recommendations

When analyzing:
- Use actual Jira data when available (via MCP tools)
- Look for patterns and correlations in metrics
- Consider both quantitative data and qualitative feedback
- Be specific and actionable in recommendations
- Focus on team improvement and growth

Always be objective, constructive, and empathetic.""",
    )

    return agent


async def enrich_insights_with_jira(
    insights: dict[str, Any],
    team_name: str,
) -> dict[str, Any]:
    """Enrich insights using Jira data via agent.

    Args:
        insights: Existing insights from analysis
        team_name: Team name to search for

    Returns:
        Enriched insights with Jira context

    """
    agent = await create_retrospective_agent()

    # Build enrichment query
    query = f"""Analyze Jira data for project "{team_name}" over last 5 months.
Here is my cloud_id: 3ff050c1-7363-42e6-b738-597caf0aa005

Current insights summary:
- Trends detected: {len(insights.get("analysis", {}).get("trends", []))}
- Patterns found: {len(insights.get("analysis", {}).get("patterns", []))}
- Anomalies: {len(insights.get("analysis", {}).get("anomalies", []))}

Please:
1. Search for recent sprint issues and retrospective notes
2. Identify any recurring themes or blockers mentioned in Jira
3. Compare our metrics trends with actual Jira activity
4. Provide additional context that enriches our analysis
"""

    logger.info("Enriching insights with Jira data for %s", team_name)

    result = await agent.ainvoke(
        {
            "messages": [{"role": "user", "content": query}],
        }
    )

    # Add agent findings to insights
    insights["jira_enrichment"] = {
        "agent_analysis": result.get("messages")[-1].text,
        "data_source": "Atlassian MCP",
    }

    return insights


if __name__ == "__main__":
    import asyncio

    async def main():
        """Example usage of Atlassian MCP agent."""
        logging.basicConfig(level=logging.INFO)

        agent = await create_retrospective_agent()
        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "search GRIDSZ Data ticket in current sprint",
                    },
                ],
            }
        )

        logger.info("=" * 60)
        logger.info("AGENT RESPONSE:")
        logger.info("=" * 60)
        logger.info(response)

    asyncio.run(main())

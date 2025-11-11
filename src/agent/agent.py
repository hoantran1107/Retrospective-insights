"""AI Agent for Retrospective Insights Analysis.

This module creates a LangChain agent that can analyze metrics,
generate hypotheses, and provide insights for retrospective analysis.
"""

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Define tools for the agent
@tool
def analyze_metrics(team_name: str, lookback_months: int = 5) -> str:
    """Analyze team metrics and identify trends.

    Args:
        team_name: Name of the team to analyze
        lookback_months: Number of months to look back (default: 5)

    Returns:
        Analysis summary with trends and patterns
    """
    # TODO: Implement actual metrics analysis
    return f"Analyzing metrics for {team_name} over {lookback_months} months"


@tool
def generate_hypotheses(analysis_summary: str) -> str:
    """Generate hypotheses based on analysis results.

    Args:
        analysis_summary: Summary of the metrics analysis

    Returns:
        List of hypotheses about team performance
    """
    # TODO: Implement hypothesis generation
    return "Generated hypotheses based on analysis"


@tool
def search_documentation(query: str) -> str:
    """Search team documentation and past retrospectives.

    Args:
        query: Search query

    Returns:
        Relevant documentation snippets
    """
    # TODO: Implement documentation search
    return f"Search results for: {query}"


# Create the agent
agent = create_agent(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"),
    tools=[analyze_metrics, generate_hypotheses, search_documentation],
    system_prompt="""You are an AI assistant specialized in analyzing team retrospective metrics.
    
Your responsibilities:
1. Analyze team metrics data to identify trends and patterns
2. Generate meaningful hypotheses about team performance
3. Provide actionable insights for retrospectives
4. Help teams understand their strengths and areas for improvement

Always be data-driven, objective, and constructive in your analysis.""",
)


def run_agent(user_query: str) -> dict:
    """Run the agent with a user query.

    Args:
        user_query: The user's question or request

    Returns:
        Agent response with analysis and insights
    """
    result = agent.invoke({"messages": [{"role": "user", "content": user_query}]})
    return result


if __name__ == "__main__":
    # Example usage
    response = run_agent("Analyze metrics for the Data Team over the last 5 months")
    print(response)

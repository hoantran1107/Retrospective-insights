"""Retrospective Insights Bot.

An AI-powered tool that analyzes Jira metrics to generate actionable retrospective insights.
"""

__version__ = "1.0.0"
__author__ = "AI Development Team"

from .ai_insights import AIInsightsGenerator
from .analysis_engine import MetricsAnalysisEngine
from .data_fetcher import MetricsDataFetcher
from .report_generator import RetrospectiveReportGenerator

__all__ = [
    "AIInsightsGenerator",
    "MetricsAnalysisEngine",
    "MetricsDataFetcher",
    "RetrospectiveReportGenerator",
]

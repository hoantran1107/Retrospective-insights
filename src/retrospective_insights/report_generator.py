"""Report generator for creating formatted retrospective insight reports.

This module generates comprehensive reports with visualizations, insights,
and facilitation notes for Scrum teams.
"""

import base64
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from jinja2 import Environment, FileSystemLoader
from plotly.subplots import make_subplots

from .ai_insights import Experiment, FacilitationNote, Hypothesis

logger = logging.getLogger(__name__)

# Set matplotlib backend for server environments
plt.switch_backend("Agg")

# Configure seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")


class RetrospectiveReportGenerator:
    """Generates comprehensive retrospective reports with visualizations and insights.

    Creates formatted reports in multiple formats (Markdown, HTML, PDF) with
    charts, analysis results, and facilitation notes.
    """

    def __init__(self, template_dir: str = "templates", output_dir: str = "reports"):
        """Initialize the report generator.

        Args:
            template_dir: Directory containing report templates
            output_dir: Directory for output reports

        """
        self.template_dir = Path(template_dir)
        self.output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Jinja2 environment
        try:
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                autoescape=True,
            )
        except Exception:
            # Use string templates if template directory doesn't exist
            self.jinja_env = None
            logger.warning(
                f"Template directory {template_dir} not found. Using built-in templates.",
            )

        logger.info(
            f"Initialized RetrospectiveReportGenerator with output dir: {output_dir}",
        )

    def generate_charts(self, analysis_results: dict[str, Any]) -> dict[str, str]:
        """Generate charts from analysis results.

        Args:
            analysis_results: Results from MetricsAnalysisEngine

        Returns:
            Dictionary mapping chart names to base64-encoded images or file paths

        """
        logger.info("Generating charts for report")
        charts = {}

        try:
            # Generate trend charts
            trends_chart = self._create_trends_chart(analysis_results.get("trends", {}))
            if trends_chart:
                charts["trends"] = trends_chart

            # Generate correlation heatmap
            patterns_chart = self._create_patterns_chart(
                analysis_results.get("patterns", {}),
            )
            if patterns_chart:
                charts["patterns"] = patterns_chart

            # Generate anomalies timeline
            anomalies_chart = self._create_anomalies_chart(
                analysis_results.get("anomalies", {}),
            )
            if anomalies_chart:
                charts["anomalies"] = anomalies_chart

            # Generate summary dashboard
            summary_chart = self._create_summary_dashboard(analysis_results)
            if summary_chart:
                charts["summary"] = summary_chart

            logger.info(f"Generated {len(charts)} charts")

        except Exception as e:
            logger.error(f"Failed to generate charts: {e}")

        return charts

    def create_report(
        self,
        analysis_results: dict[str, Any],
        hypotheses: list[Hypothesis],
        experiments: list[Experiment],
        facilitation_notes: FacilitationNote,
        team_name: str = "Team",
        format_type: str = "markdown",
        jira_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a comprehensive retrospective report.

        Args:
            analysis_results: Analysis results from MetricsAnalysisEngine
            hypotheses: Generated hypotheses
            experiments: Suggested experiments
            facilitation_notes: Facilitation notes for Scrum Masters
            team_name: Name of the team
            format_type: Output format ('markdown', 'html', 'pdf')
            jira_context: Optional Jira enrichment data from MCP agent

        Returns:
            Dictionary with report content and metadata

        """
        logger.info(f"Creating {format_type} report for {team_name}")

        # Generate charts
        charts = self.generate_charts(analysis_results)

        # Prepare report data
        report_data = {
            "team_name": team_name,
            "generation_timestamp": datetime.now().isoformat(),
            "analysis_period": f"{analysis_results.get('lookback_months', 5)} months",
            "executive_summary": self._generate_executive_summary(
                analysis_results,
                hypotheses,
            ),
            "metrics_overview": self._format_metrics_overview(analysis_results),
            "trends": analysis_results.get("trends", {}),
            "patterns": analysis_results.get("patterns", {}),
            "anomalies": analysis_results.get("anomalies", {}),
            "hypotheses": hypotheses,
            "experiments": experiments,
            "facilitation_notes": facilitation_notes,
            "charts": charts,
            "key_insights": analysis_results.get("key_insights", []),
            "jira_context": jira_context or {},
        }

        # Generate report based on format
        if format_type.lower() == "markdown":
            content = self._create_markdown_report(report_data)
            file_extension = ".md"
        elif format_type.lower() == "html":
            content = self._create_html_report(report_data)
            file_extension = ".html"
        else:
            logger.warning(f"Unsupported format: {format_type}. Using markdown.")
            content = self._create_markdown_report(report_data)
            file_extension = ".md"

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"retrospective_report_{team_name.lower().replace(' ', '_')}_{timestamp}{file_extension}"

        return {
            "content": content,
            "filename": filename,
            "format": format_type,
            "metadata": {
                "team_name": team_name,
                "generated_at": datetime.now().isoformat(),
                "metrics_count": len(
                    analysis_results.get("trends", {}).get("details", {}),
                ),
                "hypotheses_count": len(hypotheses),
                "experiments_count": len(experiments),
                "charts_count": len(charts),
            },
        }

    def export_to_file(
        self,
        report: dict[str, Any],
        output_path: str | None = None,
    ) -> str:
        """Export report to file.

        Args:
            report: Report dictionary from create_report
            output_path: Optional custom output path

        Returns:
            Path to the exported file

        """
        if output_path:
            file_path = Path(output_path)
        else:
            file_path = self.output_dir / report["filename"]

        # Ensure output directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write report content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(report["content"])

        logger.info(f"Exported report to: {file_path}")
        return str(file_path)

    def _create_trends_chart(self, trends_data: dict[str, Any]) -> str | None:
        """Create trends visualization chart."""
        try:
            trend_details = trends_data.get("details", {})
            if not trend_details:
                return None

            # Prepare data for plotting
            metrics = []
            changes = []
            colors = []

            for metric_name, trend in trend_details.items():
                if trend.significance in ["high", "medium"]:
                    metrics.append(metric_name.replace("-", " ").title())
                    changes.append(trend.percent_change * 100)

                    # Color based on trend direction and metric type
                    if self._is_positive_change(metric_name, trend.trend_direction):
                        colors.append("green")
                    elif trend.trend_direction == "stable":
                        colors.append("blue")
                    else:
                        colors.append("red")

            if not metrics:
                return None

            # Create plotly chart
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=metrics,
                        y=changes,
                        marker_color=colors,
                        text=[f"{change:.1f}%" for change in changes],
                        textposition="auto",
                    ),
                ],
            )

            fig.update_layout(
                title="Month-over-Month Trends",
                xaxis_title="Metrics",
                yaxis_title="Percent Change (%)",
                height=400,
                showlegend=False,
            )

            # Convert to base64 string for embedding
            return self._plotly_to_base64(fig)

        except Exception as e:
            logger.error(f"Failed to create trends chart: {e}")
            return None

    def _create_patterns_chart(self, patterns_data: dict[str, Any]) -> str | None:
        """Create correlation patterns visualization."""
        try:
            patterns = patterns_data.get("details", [])
            if not patterns:
                return None

            # Create correlation matrix data
            metrics = set()
            for pattern in patterns:
                metrics.add(pattern.primary_metric)
                metrics.add(pattern.secondary_metric)

            metrics = sorted(list(metrics))
            correlation_matrix = pd.DataFrame(index=metrics, columns=metrics, data=0.0)

            # Fill correlation matrix
            for pattern in patterns:
                correlation_matrix.loc[
                    pattern.primary_metric,
                    pattern.secondary_metric,
                ] = pattern.correlation_coefficient
                correlation_matrix.loc[
                    pattern.secondary_metric,
                    pattern.primary_metric,
                ] = pattern.correlation_coefficient

            # Set diagonal to 1
            for metric in metrics:
                correlation_matrix.loc[metric, metric] = 1.0

            # Create heatmap
            fig = px.imshow(
                correlation_matrix.values,
                x=[m.replace("-", " ").title() for m in metrics],
                y=[m.replace("-", " ").title() for m in metrics],
                color_continuous_scale="RdBu_r",
                aspect="auto",
                title="Metrics Correlation Patterns",
            )

            fig.update_layout(height=400)

            return self._plotly_to_base64(fig)

        except Exception as e:
            logger.error(f"Failed to create patterns chart: {e}")
            return None

    def _create_anomalies_chart(self, anomalies_data: dict[str, Any]) -> str | None:
        """Create anomalies timeline visualization."""
        try:
            anomalies = anomalies_data.get("details", [])
            if not anomalies:
                return None

            # Group anomalies by metric
            anomaly_groups = {}
            for anomaly in anomalies:
                if anomaly.metric_name not in anomaly_groups:
                    anomaly_groups[anomaly.metric_name] = []
                anomaly_groups[anomaly.metric_name].append(anomaly)

            # Create subplot for each metric with anomalies
            fig = make_subplots(
                rows=len(anomaly_groups),
                cols=1,
                subplot_titles=[
                    f"{metric.replace('-', ' ').title()} Anomalies"
                    for metric in anomaly_groups
                ],
                vertical_spacing=0.1,
            )

            for i, (metric, metric_anomalies) in enumerate(anomaly_groups.items(), 1):
                timestamps = [pd.to_datetime(a.timestamp) for a in metric_anomalies]
                values = [a.value for a in metric_anomalies]
                severities = [a.severity for a in metric_anomalies]

                colors = ["red" if s == "high" else "orange" for s in severities]

                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=values,
                        mode="markers",
                        marker=dict(color=colors, size=10),
                        name=f"{metric} Anomalies",
                        showlegend=False,
                    ),
                    row=i,
                    col=1,
                )

            fig.update_layout(
                title="Detected Anomalies Timeline",
                height=200 * len(anomaly_groups),
            )

            return self._plotly_to_base64(fig)

        except Exception as e:
            logger.error(f"Failed to create anomalies chart: {e}")
            return None

    def _create_summary_dashboard(
        self,
        analysis_results: dict[str, Any],
    ) -> str | None:
        """Create summary dashboard with key metrics."""
        try:
            # Create summary metrics
            trends = analysis_results.get("trends", {})
            patterns = analysis_results.get("patterns", {})
            anomalies = analysis_results.get("anomalies", {})

            # Summary data
            summary_data = {
                "Metrics Analyzed": len(trends.get("details", {})),
                "Significant Trends": trends.get("significant", 0),
                "Strong Patterns": patterns.get("strong", 0),
                "High Anomalies": anomalies.get("high_severity", 0),
            }

            # Create gauge charts
            fig = make_subplots(
                rows=2,
                cols=2,
                specs=[
                    [{"type": "indicator"}, {"type": "indicator"}],
                    [{"type": "indicator"}, {"type": "indicator"}],
                ],
                subplot_titles=list(summary_data.keys()),
            )

            positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

            for i, (label, value) in enumerate(summary_data.items()):
                row, col = positions[i]

                fig.add_trace(
                    go.Indicator(
                        mode="number",
                        value=value,
                        title={"text": label},
                        number={"font": {"size": 40}},
                    ),
                    row=row,
                    col=col,
                )

            fig.update_layout(title="Analysis Summary Dashboard", height=400)

            return self._plotly_to_base64(fig)

        except Exception as e:
            logger.error(f"Failed to create summary dashboard: {e}")
            return None

    def _plotly_to_base64(self, fig: go.Figure) -> str:
        """Convert plotly figure to base64 string for embedding."""
        try:
            # Export as PNG
            img_bytes = fig.to_image(format="png", width=800, height=600, scale=1)

            # Convert to base64
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")

            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"Failed to convert chart to base64: {e}")
            return ""

    def _generate_executive_summary(
        self,
        analysis_results: dict[str, Any],
        hypotheses: list[Hypothesis],
    ) -> str:
        """Generate executive summary for the report."""
        trends = analysis_results.get("trends", {})
        patterns = analysis_results.get("patterns", {})

        # Find most significant trend
        trend_details = trends.get("details", {})
        significant_trends = [
            (name, trend)
            for name, trend in trend_details.items()
            if trend.significance == "high"
        ]

        if significant_trends:
            metric_name, trend = significant_trends[0]
            trend_summary = (
                f"{metric_name.replace('-', ' ')} {trend.trend_direction} "
                f"{abs(trend.percent_change):.0%} month-over-month"
            )
        else:
            trend_summary = "No highly significant trends detected"

        # Find strongest pattern
        pattern_details = patterns.get("details", [])
        if pattern_details:
            strongest_pattern = pattern_details[0]
            pattern_summary = f" Strong correlation between {strongest_pattern.primary_metric} and {strongest_pattern.secondary_metric}"
        else:
            pattern_summary = ""

        # Top hypothesis
        if hypotheses:
            hypothesis_summary = f" Key hypothesis: {hypotheses[0].title}"
        else:
            hypothesis_summary = ""

        return f"{trend_summary}.{pattern_summary}.{hypothesis_summary}."

    def _format_metrics_overview(
        self,
        analysis_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Format metrics overview for display."""
        trends = analysis_results.get("trends", {})

        return {
            "total_metrics": len(trends.get("details", {})),
            "analysis_period": f"{analysis_results.get('lookback_months', 5)} months",
            "significant_changes": trends.get("significant", 0),
            "improving_trends": trends.get("improving", 0),
            "declining_trends": trends.get("declining", 0),
        }

    def _is_positive_change(self, metric_name: str, trend_direction: str) -> bool:
        """Determine if a trend change is positive based on metric type."""
        # Metrics where increase is good
        positive_metrics = ["happiness", "sp-distribution"]
        # Metrics where decrease is good
        negative_metrics = [
            "testing-time",
            "review-time",
            "coding-time",
            "defect-rate-prod",
            "defect-rate-all",
            "open-bugs-over-time",
            "items-out-of-sprint",
        ]

        if metric_name in positive_metrics:
            return trend_direction == "increasing"
        if metric_name in negative_metrics:
            return trend_direction == "decreasing"
        return trend_direction == "stable"

    def _create_markdown_report(self, report_data: dict[str, Any]) -> str:
        """Create markdown report content."""
        if self.jinja_env:
            try:
                template = self.jinja_env.get_template("retrospective_report.md.j2")
                return template.render(**report_data)
            except Exception as e:
                logger.warning(
                    f"Failed to load template: {e}. Using built-in template.",
                )

        return self._create_builtin_markdown_report(report_data)

    def _create_html_report(self, report_data: dict[str, Any]) -> str:
        """Create HTML report content."""
        if self.jinja_env:
            try:
                template = self.jinja_env.get_template("retrospective_report.html.j2")
                return template.render(**report_data)
            except Exception as e:
                logger.warning(
                    f"Failed to load template: {e}. Using built-in template.",
                )

        return self._create_builtin_html_report(report_data)

    def _create_builtin_markdown_report(self, data: dict[str, Any]) -> str:
        """Create markdown report using built-in template."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        md = f"""# Retrospective Insights Report

**Team:** {data["team_name"]}  
**Generated:** {timestamp}  
**Analysis Period:** {data["analysis_period"]}  

## Executive Summary

{data["executive_summary"]}

## Key Insights

"""

        for insight in data["key_insights"]:
            md += f"- {insight}\n"

        md += f"""

## Metrics Overview

- **Total Metrics Analyzed:** {data["metrics_overview"]["total_metrics"]}
- **Significant Changes:** {data["metrics_overview"]["significant_changes"]}
- **Analysis Period:** {data["metrics_overview"]["analysis_period"]}

## Jira Context & Enrichment

"""

        if data.get("jira_context"):
            jira_ctx = data["jira_context"]

            if jira_ctx.get("agent_analysis"):
                md += f"**Agent Analysis:**\n\n{jira_ctx['agent_analysis']}\n\n"

            if jira_ctx.get("related_issues"):
                md += "**Related Jira Issues:**\n\n"
                for issue in jira_ctx["related_issues"][:5]:
                    md += f"- [{issue.get('key', 'N/A')}] {issue.get('summary', 'N/A')} - Status: {issue.get('status', 'N/A')}\n"
                md += "\n"

            if jira_ctx.get("sprint_context"):
                sprint = jira_ctx["sprint_context"]
                md += f"**Sprint Context:** {sprint.get('name', 'N/A')} ({sprint.get('state', 'N/A')})\n\n"
        else:
            md += "*No Jira enrichment data available*\n\n"

        md += """## Top 3 Hypotheses

"""

        for i, hypothesis in enumerate(data["hypotheses"][:3], 1):
            md += f"""
### {i}. {hypothesis.title}

**Description:** {hypothesis.description}

**Evidence:**
"""
            for evidence in hypothesis.evidence:
                md += f"- {evidence}\n"

            md += f"""
**Confidence:** {hypothesis.confidence_level}  
**Impact:** {hypothesis.impact_assessment}

"""

        md += """
## Suggested Experiments

"""

        for i, experiment in enumerate(data["experiments"][:3], 1):
            md += f"""
### {i}. {experiment.title}

**Description:** {experiment.description}

**Success Criteria:**
"""
            for criteria in experiment.success_criteria:
                md += f"- {criteria}\n"

            md += """
**Implementation Steps:**
"""
            for step in experiment.implementation_steps:
                md += f"1. {step}\n"

            md += f"""
**Effort:** {experiment.estimated_effort}  
**Duration:** {experiment.duration}

"""

        md += """
## Facilitation Notes

### Suggested Questions
"""

        for question in data["facilitation_notes"].suggested_questions:
            md += f"- {question}\n"

        md += """
### Agenda Items
"""

        for item in data["facilitation_notes"].agenda_items:
            md += f"- {item}\n"

        md += """
---
*This report was generated by the AI Retrospective Insights Bot*
"""

        return md

    def _create_builtin_html_report(self, data: dict[str, Any]) -> str:
        """Create HTML report using built-in template."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build charts HTML
        charts_html = ""
        for chart_name, chart_data in data["charts"].items():
            if chart_data:
                charts_html += f'<div class="chart"><img src="{chart_data}" alt="{chart_name}" style="max-width: 100%; height: auto;"></div>\n'

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Retrospective Insights Report - {data["team_name"]}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; }}
        h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; }}
        .summary {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; }}
        .hypothesis {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff; }}
        .experiment {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #28a745; }}
        .chart {{ text-align: center; margin: 20px 0; }}
        ul {{ margin: 10px 0; }}
        .confidence-high {{ color: #27ae60; font-weight: bold; }}
        .confidence-medium {{ color: #f39c12; font-weight: bold; }}
        .confidence-low {{ color: #e74c3c; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Retrospective Insights Report</h1>
    
    <div class="summary">
        <p><strong>Team:</strong> {data["team_name"]}</p>
        <p><strong>Generated:</strong> {timestamp}</p>
        <p><strong>Analysis Period:</strong> {data["analysis_period"]}</p>
    </div>
    
    <h2>Executive Summary</h2>
    <p>{data["executive_summary"]}</p>
    
    <h2>Key Insights</h2>
    <ul>
"""

        for insight in data["key_insights"]:
            html += f"        <li>{insight}</li>\n"

        html += """    </ul>
    
    <h2>Jira Context & Enrichment</h2>
"""

        if data.get("jira_context"):
            jira_ctx = data["jira_context"]
            html += '    <div class="jira-context" style="background-color: #e8f4f8; padding: 15px; margin: 10px 0; border-left: 4px solid #17a2b8;">\n'

            if jira_ctx.get("agent_analysis"):
                html += f"        <h3>Agent Analysis</h3>\n"
                html += f"        <p>{jira_ctx['agent_analysis']}</p>\n"

            if jira_ctx.get("related_issues"):
                html += "        <h3>Related Jira Issues</h3>\n"
                html += "        <ul>\n"
                for issue in jira_ctx["related_issues"][:5]:
                    key = issue.get("key", "N/A")
                    summary = issue.get("summary", "N/A")
                    status = issue.get("status", "N/A")
                    html += f"            <li><strong>[{key}]</strong> {summary} - <em>Status: {status}</em></li>\n"
                html += "        </ul>\n"

            if jira_ctx.get("sprint_context"):
                sprint = jira_ctx["sprint_context"]
                html += f"        <h3>Sprint Context</h3>\n"
                html += f"        <p><strong>{sprint.get('name', 'N/A')}</strong> ({sprint.get('state', 'N/A')})</p>\n"

            html += "    </div>\n"
        else:
            html += "    <p><em>No Jira enrichment data available</em></p>\n"

        html += f"""    
    <h2>Visualizations</h2>
    {charts_html}"
    
    <h2>Top 3 Hypotheses</h2>
"""

        for i, hypothesis in enumerate(data["hypotheses"][:3], 1):
            confidence_class = f"confidence-{hypothesis.confidence_level}"
            html += f"""
    <div class="hypothesis">
        <h3>{i}. {hypothesis.title}</h3>
        <p><strong>Description:</strong> {hypothesis.description}</p>
        <p><strong>Evidence:</strong></p>
        <ul>
"""
            for evidence in hypothesis.evidence:
                html += f"            <li>{evidence}</li>\n"

            html += f"""        </ul>
        <p><strong>Confidence:</strong> <span class="{confidence_class}">{hypothesis.confidence_level}</span></p>
        <p><strong>Impact:</strong> {hypothesis.impact_assessment}</p>
    </div>
"""

        html += """
    <h2>Suggested Experiments</h2>
"""

        for i, experiment in enumerate(data["experiments"][:3], 1):
            html += f"""
    <div class="experiment">
        <h3>{i}. {experiment.title}</h3>
        <p><strong>Description:</strong> {experiment.description}</p>
        <p><strong>Success Criteria:</strong></p>
        <ul>
"""
            for criteria in experiment.success_criteria:
                html += f"            <li>{criteria}</li>\n"

            html += """        </ul>
        <p><strong>Implementation Steps:</strong></p>
        <ol>
"""
            for step in experiment.implementation_steps:
                html += f"            <li>{step}</li>\n"

            html += f"""        </ol>
        <p><strong>Effort:</strong> {experiment.estimated_effort} | <strong>Duration:</strong> {experiment.duration}</p>
    </div>
"""

        html += """
    <h2>Facilitation Notes</h2>
    
    <h3>Suggested Questions</h3>
    <ul>
"""

        for question in data["facilitation_notes"].suggested_questions:
            html += f"        <li>{question}</li>\n"

        html += """    </ul>
    
    <h3>Agenda Items</h3>
    <ul>
"""

        for item in data["facilitation_notes"].agenda_items:
            html += f"        <li>{item}</li>\n"

        html += """    </ul>
    
    <hr>
    <p><em>This report was generated by the AI Retrospective Insights Bot</em></p>
</body>
</html>"""

        return html


# Utility functions


def create_report_generator_from_env() -> RetrospectiveReportGenerator:
    """Create a RetrospectiveReportGenerator from environment variables.

    Optional environment variables:
        TEMPLATE_DIR: Template directory path (default: templates)
        OUTPUT_DIR: Output directory path (default: reports)

    Returns:
        Configured RetrospectiveReportGenerator instance

    """
    from dotenv import load_dotenv

    load_dotenv()

    template_dir = os.getenv("TEMPLATE_DIR", "templates")
    output_dir = os.getenv("OUTPUT_DIR", "reports")

    return RetrospectiveReportGenerator(
        template_dir=template_dir,
        output_dir=output_dir,
    )

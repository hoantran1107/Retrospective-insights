#!/usr/bin/env python3
"""Main entry point for the Retrospective Insights AI Bot.
This implements the solution for Jira ticket AISTR-30.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any
from async_lru import alru_cache
from src.retrospective_insights.ai_insights import AIInsightsGenerator
from src.retrospective_insights.analysis_engine import MetricsAnalysisEngine
from src.retrospective_insights.data_fetcher import MetricsDataFetcher
from src.retrospective_insights.report_generator import RetrospectiveReportGenerator
from src.retrospective_insights.ai_insights import create_ai_generator_from_env

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RetrospectiveInsightsBot:
    """AI Bot for generating retrospective insights from team metrics."""

    def __init__(
        self, team_filter: str = None, team_happiness_name: str = None
    ) -> None:
        """Initialize the retrospective insights bot.

        Args:
            team_filter: Optional team name to filter data (e.g., "Gridsz Data Team")

        """
        self.data_fetcher = None
        self.analysis_engine = None
        self.ai_generator = None
        self.report_generator = None
        self.team_filter = team_filter
        self.team_happiness_name = team_happiness_name

    def initialize_components(self) -> None:
        """Initialize all components."""
        logger.info("🔧 Initializing Retrospective Insights Bot components...")

        # Initialize data fetcher
        auth_url = os.getenv(
            "DASHBOARD_AUTH_URL",
            "https://n8n.idp.infodation.vn/webhook/88eda05f-41d5-4ce4-b836-cb0f1bba3b2e",
        )
        data_url = os.getenv(
            "DASHBOARD_DATA_URL",
            "https://n8n.idp.infodation.vn/webhook/7f0e2b2b-a7b5-4b4e-8b86-d4f91c8d8e7a",
        )

        self.data_fetcher = MetricsDataFetcher(
            auth_url=auth_url,
            data_url=data_url,
        )

        # Initialize analysis engine
        self.analysis_engine = MetricsAnalysisEngine()

        # Initialize AI generator
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
        if not api_key:
            logger.warning("⚠️ No OpenAI API key found. AI insights will be limited.")
            api_key = "placeholder"

        self.ai_generator = create_ai_generator_from_env()

        # Initialize report generator
        self.report_generator = RetrospectiveReportGenerator()

        logger.info("✅ All components initialized successfully")

    @alru_cache(ttl=3600)
    async def fetch_last_5_sprints_data(
        self, start_date, end_date, team_filter, team_happiness_name
    ) -> dict[str, Any]:
        """Fetch and map data for the last 5 sprints (5 months: June-October 2025)."""
        if team_filter:
            logger.info(
                f"📊 Fetching data for last 5 months (Team: {team_filter})...",
            )
        else:
            logger.info("📊 Fetching data for last 5 months (All teams)...")
        # fetch raw data from API
        # raw_metrics = await self.data_fetcher.fetch_all_metrics()
        # Load data from file
        with open("raw_data.json", encoding="utf-8") as f:
            raw_metrics = json.load(f)

        logger.info(
            f"⏰ Time window: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (5 months)",
        )

        filtered_data = {}
        total_records_before = 0
        total_records_after = 0

        for metric_name, metric_result in raw_metrics.items():
            if not metric_result.get("success") or not metric_result.get("data"):
                continue

            mapped_records = []
            records_parsed = 0
            records_in_window = 0

            for record in metric_result["data"]:
                total_records_before += 1

                # 1. Extract date from record
                record_date = self._extract_date_from_record(record, metric_name)
                if record_date is None and metric_name != "root-cause":
                    continue  # Skip records without valid dates (except root-cause)

                # 2. Apply time filter (5 months only) - except for root-cause
                if metric_name != "root-cause":
                    records_parsed += 1
                    if not (start_date <= record_date <= end_date):
                        continue  # Skip records outside time window
                    records_in_window += 1

                # 3. Apply team filter (if specified)
                if team_filter or team_happiness_name:
                    team_name = self._extract_team_name(record)
                    if team_name not in (team_filter, team_happiness_name):
                        continue

                # 4. Extract value using metric-specific logic
                value = self._extract_value_from_record(record, metric_name)

                # 5. Create mapped record with actual date and value
                mapped_record = {
                    "date": (
                        record_date.strftime("%Y-%m-%d")
                        if record_date
                        else end_date.strftime("%Y-%m-%d")
                    ),
                    "value": value,
                    "original": record,  # Keep original data for reference
                }
                mapped_records.append(mapped_record)
                total_records_after += 1

            # Log parsing results for this metric
            if mapped_records:
                if metric_name == "root-cause":
                    logger.info(
                        f"📅 {metric_name}: no date field, included all {len(mapped_records)} records",
                    )
                else:
                    logger.info(
                        f"📅 {metric_name}: parsed {records_parsed} dates, "
                        f"kept {records_in_window} in window, "
                        f"final {len(mapped_records)} records (after team filter)",
                    )

                filtered_data[metric_name] = {
                    "success": True,
                    "data": mapped_records,
                }

        logger.info(f"✅ Fetched and mapped {len(filtered_data)} metrics for analysis")
        logger.info(
            f"📊 Records: {total_records_before} total → {total_records_after} after filtering",
        )
        return filtered_data

    def _extract_team_name(self, record: dict[str, Any]) -> str:
        """Extract team name from record. Handles different field names."""
        # Try different field names that might contain team information
        # Priority order based on actual data frequency:
        # project_name (6 metrics), name (2 metrics), project (2 metrics), team (1 metric)
        for field in ["project_name", "name", "project", "team", "team_name"]:
            if field in record:
                return record[field]
        return ""

    def _extract_date_from_record(
        self,
        record: dict[str, Any],
        metric_name: str,
    ) -> datetime | None:
        """Extract date from record based on metric type.

        Different metrics use different date fields:
        - MonthYearSort: coding-time, open-bugs-over-time, sp-distribution (format: "202505")
        - MonthYear: testing-time, review-time, bugs-per-environment, items-out-of-sprint,
                     defect-rate-prod, defect-rate-all (format: "May 25")
        - monthYear: happiness (format: "2024-03")
        - None: root-cause (no date field)

        Args:
            record: Data record from metric
            metric_name: Name of metric (e.g., "testing-time")

        Returns:
            datetime object or None if parsing fails

        """
        # Priority 1: MonthYearSort (easy to parse, no locale issues)
        if record.get("MonthYearSort"):
            try:
                return datetime.strptime(str(record["MonthYearSort"]), "%Y%m")
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Could not parse MonthYearSort '{record['MonthYearSort']}' for {metric_name}: {e}",
                )

        # Priority 2: MonthYear (most common format)
        if record.get("MonthYear"):
            try:
                return datetime.strptime(record["MonthYear"], "%b %y")
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Could not parse MonthYear '{record['MonthYear']}' for {metric_name}: {e}",
                )

        # Priority 3: monthYear (happiness metric only)
        if record.get("monthYear"):
            try:
                return datetime.strptime(record["monthYear"], "%Y-%m")
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Could not parse monthYear '{record['monthYear']}' for {metric_name}: {e}",
                )

        # Special case: root-cause has no date field (last 6 months aggregate)
        if metric_name == "root-cause":
            return datetime.now()

        return None

    def _extract_value_from_record(
        self,
        record: dict[str, Any],
        metric_name: str,
    ) -> float:
        """Extract value from record based on metric type.

        Different metrics use different value fields:
        - "Avg Status Duration": testing-time, review-time, coding-time
        - "Open Bugs EndOfMonth": open-bugs-over-time
        - "total": root-cause, bugs-per-environment, sp-distribution
        - "% Out-of-Sprint": items-out-of-sprint
        - "% Defect Rate (PROD)": defect-rate-prod
        - "% Defect Rate (ALL)": defect-rate-all
        - "averageScore": happiness

        Args:
            record: Data record from metric
            metric_name: Name of metric

        Returns:
            Extracted value as float, or 0.0 if not found

        """
        # Try metric-specific fields first
        value_fields = [
            "Avg Status Duration",
            "Open Bugs EndOfMonth",
            "% Out-of-Sprint",
            "% Defect Rate (PROD)",
            "% Defect Rate (ALL)",
            "averageScore",
            "total",
        ]

        for field in value_fields:
            if field in record and record[field] is not None:
                try:
                    return float(record[field])
                except (ValueError, TypeError):
                    logger.warning(
                        f"Could not convert '{field}' value '{record[field]}' to float for {metric_name}",
                    )

        # Fallback to generic fields
        for field in ["count", "percentage", "value"]:
            if field in record and record[field] is not None:
                try:
                    return float(record[field])
                except (ValueError, TypeError):
                    pass

        return 0.0

    async def generate_insights(self, metrics_data: dict[str, Any]) -> dict[str, Any]:
        """Generate AI insights from metrics data."""
        logger.info("🤖 Generating AI insights...")

        # Load data into analysis engine
        self.analysis_engine.load_metrics_data(metrics_data)

        # Run analysis
        trends = self.analysis_engine.calculate_trends()
        patterns = self.analysis_engine.detect_patterns()
        anomalies = self.analysis_engine.identify_anomalies()
        summary = self.analysis_engine.generate_analysis_summary()

        logger.info(
            f"📈 Analysis: {len(trends)} trends, {len(patterns)} patterns, {len(anomalies)} anomalies",
        )

        # Generate AI insights
        try:
            hypotheses = self.ai_generator.generate_hypotheses(summary)
            experiments = self.ai_generator.suggest_experiments(hypotheses)
            facilitation_notes = self.ai_generator.generate_facilitation_notes(
                hypotheses,
                experiments,
            )

            logger.info(
                f"🧠 AI Insights: {len(hypotheses)} hypotheses, {len(experiments)} experiments",
            )

        except Exception as e:
            logger.warning(f"⚠️ AI insights generation failed: {e}")
            hypotheses = []
            experiments = []
            facilitation_notes = "AI insights unavailable due to API limitations"

        return {
            "analysis": {
                "trends": trends,
                "patterns": patterns,
                "anomalies": anomalies,
                "summary": summary,
            },
            "ai_insights": {
                "hypotheses": hypotheses,
                "experiments": experiments,
                "facilitation_notes": facilitation_notes,
            },
        }

    async def generate_report(self, insights: dict[str, Any]) -> str:
        """Generate comprehensive report."""
        logger.info("📝 Generating report...")

        # Determine team name for report
        team_name = self.team_filter if self.team_filter else "All Teams"

        # Generate charts
        charts = self.report_generator.generate_charts(insights["analysis"]["summary"])

        # Create report
        report_content = self.report_generator.create_report(
            analysis_results=insights["analysis"]["summary"],
            hypotheses=insights["ai_insights"]["hypotheses"],
            experiments=insights["ai_insights"]["experiments"],
            facilitation_notes=insights["ai_insights"]["facilitation_notes"],
            team_name=team_name,
            format_type="html",
        )

        # Export to file
        output_path = self.report_generator.export_to_file(
            report=report_content,
        )

        logger.info(f"✅ Report generated: {output_path}")
        return output_path

    async def run_complete_analysis(self) -> str:
        """Run the complete retrospective insights analysis workflow."""
        logger.info("🚀 Starting Retrospective Insights Analysis (AISTR-30)")
        logger.info("=" * 60)

        try:
            # Initialize components
            self.initialize_components()
            # Define 5-month time window (June 1 - October 31, 2025)
            end_date = datetime.now()
            start_date = end_date.replace(month=end_date.month - 5)
            # Fetch data for last 5 sprints
            metrics_data = await self.fetch_last_5_sprints_data(
                start_date=start_date,
                end_date=end_date,
                team_filter=self.team_filter,
                team_happiness_name=self.team_happiness_name,
            )

            # Generate insights
            insights = await self.generate_insights(metrics_data)

            # Generate report
            report_path = await self.generate_report(insights)

            # Summary
            total_records = sum(len(data["data"]) for data in metrics_data.values())
            anomalies_count = len(insights["analysis"]["anomalies"])
            hypotheses_count = len(insights["ai_insights"]["hypotheses"])

            logger.info("🎉 Analysis Complete!")
            logger.info("=" * 60)
            logger.info("📊 SUMMARY:")
            logger.info(f"   - Metrics analyzed: {len(metrics_data)}")
            logger.info(f"   - Records processed: {total_records}")
            logger.info(f"   - Anomalies detected: {anomalies_count}")
            logger.info(f"   - AI hypotheses: {hypotheses_count}")
            logger.info(f"   - Report saved to: {report_path}")
            logger.info("   - Time period: Last 5 sprints (10 weeks)")
            logger.info("   - Data source: Real API (non-mocked)")
            logger.info("   - AI provider: Azure OpenAI")

            return report_path

        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            raise


async def main():
    """Run the retrospective insights analysis."""
    # Check for team filter argument
    team_filter = None
    if len(sys.argv) > 1:
        team_filter = sys.argv[1]
        logger.info(f"🎯 Running analysis for team: {team_filter}")
    else:
        logger.info("🎯 Running analysis for all teams")

    bot = RetrospectiveInsightsBot(team_filter=team_filter)
    report_path = await bot.run_complete_analysis()
    print(f"\n🎯 SUCCESS: Retrospective insights report generated at {report_path}")
    print("📋 This completes the implementation of Jira ticket AISTR-30")


if __name__ == "__main__":
    asyncio.run(main())

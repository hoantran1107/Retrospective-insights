"""Analysis engine for processing metrics data and identifying trends and patterns.

This module calculates month-over-month changes, detects significant patterns,
and identifies correlations between different metrics.
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Suppress pandas warnings for cleaner output
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


@dataclass
class TrendAnalysis:
    """Represents trend analysis results for a metric."""

    metric_name: str
    current_value: float
    previous_value: float
    percent_change: float
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    significance: str  # 'high', 'medium', 'low'
    confidence: float
    data_points: int


@dataclass
class PatternMatch:
    """Represents a detected pattern between metrics."""

    primary_metric: str
    secondary_metric: str
    correlation_coefficient: float
    correlation_type: str  # 'positive', 'negative', 'none'
    strength: str  # 'strong', 'moderate', 'weak'
    significance_p_value: float
    description: str


@dataclass
class Anomaly:
    """Represents an anomaly detected in metrics data."""

    metric_name: str
    timestamp: str
    value: float
    expected_range: tuple[float, float]
    severity: str  # 'high', 'medium', 'low'
    description: str


class MetricsAnalysisEngine:
    """Core analysis engine for processing metrics data.

    Identifies trends, patterns, correlations, and anomalies in team metrics
    to generate insights for retrospective analysis.
    """

    # Thresholds for trend significance
    SIGNIFICANT_CHANGE_THRESHOLD = 0.20  # 20%
    HIGH_CHANGE_THRESHOLD = 0.40  # 40%

    # Correlation strength thresholds
    STRONG_CORRELATION_THRESHOLD = 0.7
    MODERATE_CORRELATION_THRESHOLD = 0.4

    def __init__(self, lookback_months: int = 5):
        """Initialize the analysis engine.

        Args:
            lookback_months: Number of months to analyze for trends

        """
        self.lookback_months = lookback_months
        self.metrics_data: dict[str, Any] = {}
        self.processed_data: dict[str, pd.DataFrame] = {}

        logger.info(
            f"Initialized MetricsAnalysisEngine with {lookback_months} months lookback",
        )

    def load_metrics_data(self, metrics_data: dict[str, dict[str, Any]]) -> None:
        """Load and preprocess metrics data.

        Args:
            metrics_data: Raw metrics data from data fetcher

        """
        self.metrics_data = metrics_data
        self.processed_data = {}

        successful_metrics = [
            name
            for name, result in metrics_data.items()
            if result["success"] and result["data"]
        ]

        logger.info(
            f"Loading {len(successful_metrics)} successful metrics for analysis",
        )

        for metric_name in successful_metrics:
            try:
                raw_data = metrics_data[metric_name]["data"]
                processed_df = self._preprocess_metric_data(metric_name, raw_data)

                if processed_df is not None and not processed_df.empty:
                    self.processed_data[metric_name] = processed_df
                    logger.debug(
                        f"Processed {metric_name}: {len(processed_df)} data points",
                    )
                else:
                    logger.warning(f"No valid data for metric: {metric_name}")

            except Exception as e:
                logger.error(f"Failed to process {metric_name}: {e}")

        logger.info(f"Successfully loaded {len(self.processed_data)} metrics")

    def _preprocess_metric_data(
        self,
        metric_name: str,
        raw_data: Any,
    ) -> pd.DataFrame | None:
        """Preprocess raw metric data into a standardized DataFrame.

        Args:
            metric_name: Name of the metric
            raw_data: Raw data from API

        Returns:
            Processed DataFrame with timestamp and value columns

        """
        try:
            # Handle different data formats
            if isinstance(raw_data, dict):
                if "time_series" in raw_data:
                    # Time series data format
                    df = pd.DataFrame(raw_data["time_series"])
                elif "data" in raw_data:
                    # Nested data format
                    df = pd.DataFrame(raw_data["data"])
                else:
                    # Assume dict with date keys
                    df = pd.DataFrame(
                        list(raw_data.items()),
                        columns=["timestamp", "value"],
                    )
            elif isinstance(raw_data, list):
                # List of records format
                df = pd.DataFrame(raw_data)
            else:
                logger.warning(
                    f"Unsupported data format for {metric_name}: {type(raw_data)}",
                )
                return None

            # Standardize column names
            if "date" in df.columns:
                df = df.rename(columns={"date": "timestamp"})
            if "amount" in df.columns:
                df = df.rename(columns={"amount": "value"})
            if "count" in df.columns:
                df = df.rename(columns={"count": "value"})

            # Ensure required columns exist
            if "timestamp" not in df.columns or "value" not in df.columns:
                logger.warning(f"Missing required columns for {metric_name}")
                return None

            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])

            # Convert value to numeric
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"])

            # Sort by timestamp
            df = df.sort_values("timestamp")

            # Filter to lookback period
            cutoff_date = datetime.now() - timedelta(days=self.lookback_months * 30)
            df = df[df["timestamp"] >= cutoff_date]

            # Add month grouping for trend analysis
            df["month"] = df["timestamp"].dt.to_period("M")

            return df

        except Exception as e:
            logger.error(f"Error preprocessing {metric_name}: {e}")
            return None

    def calculate_trends(self) -> dict[str, TrendAnalysis]:
        """Calculate month-over-month trends for all metrics.

        Returns:
            Dictionary mapping metric names to trend analysis results

        """
        logger.info("Calculating trends for all metrics")
        trends = {}

        for metric_name, df in self.processed_data.items():
            try:
                trend = self._calculate_metric_trend(metric_name, df)
                if trend:
                    trends[metric_name] = trend
                    logger.debug(
                        f"{metric_name}: {trend.percent_change:.1%} change "
                        f"({trend.trend_direction})",
                    )
            except Exception as e:
                logger.error(f"Failed to calculate trend for {metric_name}: {e}")

        logger.info(f"Calculated trends for {len(trends)} metrics")
        return trends

    def _calculate_metric_trend(
        self,
        metric_name: str,
        df: pd.DataFrame,
    ) -> TrendAnalysis | None:
        """Calculate trend analysis for a single metric.

        Args:
            metric_name: Name of the metric
            df: Processed DataFrame with monthly data

        Returns:
            TrendAnalysis object or None if insufficient data

        """
        # Group by month and calculate averages
        if metric_name == "root-cause":
            return None
        monthly_data = df.groupby("month")["value"].mean().sort_index()

        if len(monthly_data) < 2:
            logger.warning(f"Insufficient data for trend analysis: {metric_name}")
            return None

        # Get first and last month values (overall trend across all months)
        current_value = monthly_data.iloc[-1]  # Last month
        previous_value = monthly_data.iloc[0]  # First month

        # Calculate percent change from first to last month
        if previous_value != 0:
            percent_change = (current_value - previous_value) / previous_value
        else:
            percent_change = 0 if current_value == 0 else float("inf")

        # Determine trend direction using linear regression if enough data
        if len(monthly_data) >= 3:
            # Use linear regression to determine long-term trend
            x = np.arange(len(monthly_data))
            y = monthly_data.values
            slope, _, _, _, _ = stats.linregress(x, y)

            # Determine direction based on slope
            if abs(slope) < (monthly_data.mean() * 0.01):  # Less than 1% per month
                trend_direction = "stable"
            elif slope > 0:
                trend_direction = "increasing"
            else:
                trend_direction = "decreasing"
        # Fallback to simple comparison for limited data
        elif abs(percent_change) < 0.05:  # Less than 5% change
            trend_direction = "stable"
        elif percent_change > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"

        # Determine significance based on overall percent change
        abs_change = abs(percent_change)
        if abs_change >= self.HIGH_CHANGE_THRESHOLD:
            significance = "high"
        elif abs_change >= self.SIGNIFICANT_CHANGE_THRESHOLD:
            significance = "medium"
        else:
            significance = "low"

        # Calculate confidence based on data stability
        confidence = self._calculate_trend_confidence(monthly_data)

        return TrendAnalysis(
            metric_name=metric_name,
            current_value=current_value,
            previous_value=previous_value,
            percent_change=percent_change,
            trend_direction=trend_direction,
            significance=significance,
            confidence=confidence,
            data_points=len(monthly_data),
        )

    def _calculate_trend_confidence(self, monthly_data: pd.Series) -> float:
        """Calculate confidence level for trend based on data stability.

        Args:
            monthly_data: Series of monthly values

        Returns:
            Confidence score between 0 and 1

        """
        if len(monthly_data) < 3:
            return 0.5  # Low confidence with limited data

        # Calculate coefficient of variation
        cv = monthly_data.std() / monthly_data.mean() if monthly_data.mean() != 0 else 1

        # Calculate trend consistency (using linear regression)
        x = np.arange(len(monthly_data))
        y = monthly_data.values

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_squared = r_value**2

            # Combine stability and trend consistency
            stability_score = max(0, 1 - cv)  # Lower CV = higher stability
            trend_score = r_squared  # Higher R² = more consistent trend

            confidence = (stability_score + trend_score) / 2
            return min(1.0, max(0.0, confidence))

        except Exception:
            return 0.5

    def detect_patterns(self) -> list[PatternMatch]:
        """Detect patterns and correlations between metrics.

        Returns:
            List of detected pattern matches

        """
        logger.info("Detecting patterns between metrics")
        patterns = []

        metric_names = list(self.processed_data.keys())

        # Calculate correlations between all metric pairs
        for i, metric1 in enumerate(metric_names):
            for metric2 in metric_names[i + 1 :]:
                try:
                    pattern = self._analyze_metric_correlation(metric1, metric2)
                    if pattern and pattern.strength != "none":
                        patterns.append(pattern)
                        logger.debug(
                            f"Pattern detected: {metric1} <-> {metric2} "
                            f"({pattern.correlation_coefficient:.2f})",
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to analyze correlation {metric1}-{metric2}: {e}",
                    )

        # Sort by correlation strength
        patterns.sort(key=lambda p: abs(p.correlation_coefficient), reverse=True)

        logger.info(f"Detected {len(patterns)} significant patterns")
        return patterns

    def _analyze_metric_correlation(
        self,
        metric1: str,
        metric2: str,
    ) -> PatternMatch | None:
        """Analyze correlation between two metrics.

        Args:
            metric1: Name of first metric
            metric2: Name of second metric

        Returns:
            PatternMatch object or None if insufficient data

        """
        df1 = self.processed_data[metric1]
        df2 = self.processed_data[metric2]

        # Group by month and get monthly averages
        monthly1 = df1.groupby("month")["value"].mean()
        monthly2 = df2.groupby("month")["value"].mean()

        # Align data by month
        common_months = monthly1.index.intersection(monthly2.index)

        if len(common_months) < 3:
            return None  # Insufficient data

        values1 = monthly1[common_months].values
        values2 = monthly2[common_months].values

        # Calculate correlation
        try:
            correlation_coef, p_value = stats.pearsonr(values1, values2)
        except Exception:
            return None

        # Determine correlation type and strength
        abs_corr = abs(correlation_coef)

        if abs_corr >= self.STRONG_CORRELATION_THRESHOLD:
            strength = "strong"
        elif abs_corr >= self.MODERATE_CORRELATION_THRESHOLD:
            strength = "moderate"
        else:
            strength = "weak"

        # Only return significant correlations
        if strength == "weak" or p_value > 0.05:
            return None

        correlation_type = "positive" if correlation_coef > 0 else "negative"

        # Generate description
        description = self._generate_correlation_description(
            metric1,
            metric2,
            correlation_type,
            strength,
            correlation_coef,
        )

        return PatternMatch(
            primary_metric=metric1,
            secondary_metric=metric2,
            correlation_coefficient=correlation_coef,
            correlation_type=correlation_type,
            strength=strength,
            significance_p_value=p_value,
            description=description,
        )

    def _generate_correlation_description(
        self,
        metric1: str,
        metric2: str,
        correlation_type: str,
        strength: str,
        coefficient: float,
    ) -> str:
        """Generate human-readable description of correlation."""
        direction = "increases" if correlation_type == "positive" else "decreases"

        return (
            f"When {metric1.replace('-', ' ')} increases, "
            f"{metric2.replace('-', ' ')} {direction} "
            f"({strength} {correlation_type} correlation: {coefficient:.2f})"
        )

    def identify_anomalies(self) -> list[Anomaly]:
        """Identify anomalies in metrics data using statistical methods.

        Returns:
            List of detected anomalies

        """
        logger.info("Identifying anomalies in metrics data")
        anomalies = []

        for metric_name, df in self.processed_data.items():
            try:
                metric_anomalies = self._detect_metric_anomalies(metric_name, df)
                anomalies.extend(metric_anomalies)
            except Exception as e:
                logger.error(f"Failed to detect anomalies for {metric_name}: {e}")

        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        anomalies.sort(key=lambda a: severity_order[a.severity])

        logger.info(f"Identified {len(anomalies)} anomalies")
        return anomalies

    def _detect_metric_anomalies(
        self,
        metric_name: str,
        df: pd.DataFrame,
    ) -> list[Anomaly]:
        """Detect anomalies for a single metric using IQR method.

        Args:
            metric_name: Name of the metric
            df: Processed DataFrame

        Returns:
            List of anomalies for this metric

        """
        if len(df) < 10:  # Need sufficient data
            return []

        values = df["value"].values

        # Calculate IQR bounds
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        # Define outlier bounds (1.5 * IQR is standard, 3 * IQR for extreme)
        moderate_lower = q1 - 1.5 * iqr
        moderate_upper = q3 + 1.5 * iqr
        extreme_lower = q1 - 3 * iqr
        extreme_upper = q3 + 3 * iqr

        anomalies = []

        for _, row in df.iterrows():
            value = row["value"]
            timestamp = row["timestamp"].isoformat()

            if value < extreme_lower or value > extreme_upper:
                severity = "high"
                expected_range = (moderate_lower, moderate_upper)
                description = (
                    f"Extreme outlier: value {value:.2f} far outside normal range"
                )
            elif value < moderate_lower or value > moderate_upper:
                severity = "medium"
                expected_range = (moderate_lower, moderate_upper)
                description = (
                    f"Moderate outlier: value {value:.2f} outside normal range"
                )
            else:
                continue  # Not an anomaly

            anomalies.append(
                Anomaly(
                    metric_name=metric_name,
                    timestamp=timestamp,
                    value=value,
                    expected_range=expected_range,
                    severity=severity,
                    description=description,
                ),
            )

        return anomalies

    def generate_analysis_summary(self) -> dict[str, Any]:
        """Generate a comprehensive summary of all analysis results.

        Returns:
            Dictionary containing analysis summary

        """
        trends = self.calculate_trends()
        patterns = self.detect_patterns()
        anomalies = self.identify_anomalies()

        # Categorize trends by significance
        significant_trends = [
            t for t in trends.values() if t.significance in ["high", "medium"]
        ]
        improving_metrics = [
            t for t in significant_trends if self._is_improving_trend(t)
        ]
        declining_metrics = [
            t for t in significant_trends if self._is_declining_trend(t)
        ]

        # Categorize patterns by strength
        strong_patterns = [p for p in patterns if p.strength == "strong"]
        moderate_patterns = [p for p in patterns if p.strength == "moderate"]

        # Categorize anomalies by severity
        high_severity_anomalies = [a for a in anomalies if a.severity == "high"]

        summary = {
            "analysis_timestamp": datetime.now().isoformat(),
            "metrics_analyzed": len(self.processed_data),
            "lookback_months": self.lookback_months,
            "trends": {
                "total": len(trends),
                "significant": len(significant_trends),
                "improving": len(improving_metrics),
                "declining": len(declining_metrics),
                "details": trends,
            },
            "patterns": {
                "total": len(patterns),
                "strong": len(strong_patterns),
                "moderate": len(moderate_patterns),
                "details": patterns,
            },
            "anomalies": {
                "total": len(anomalies),
                "high_severity": len(high_severity_anomalies),
                "details": anomalies,
            },
            "key_insights": self._generate_key_insights(
                significant_trends,
                strong_patterns,
                high_severity_anomalies,
            ),
        }

        logger.info(
            f"Generated analysis summary: {len(significant_trends)} significant trends, "
            f"{len(strong_patterns)} strong patterns, {len(high_severity_anomalies)} high-severity anomalies",
        )

        return summary

    def _is_improving_trend(self, trend: TrendAnalysis) -> bool:
        """Determine if a trend represents improvement based on metric type."""
        # Metrics where increase is good
        improving_metrics = ["happiness", "sp-distribution"]
        # Metrics where decrease is good
        declining_metrics = [
            "testing-time",
            "review-time",
            "coding-time",
            "defect-rate-prod",
            "defect-rate-all",
            "open-bugs-over-time",
            "items-out-of-sprint",
        ]

        if trend.metric_name in improving_metrics:
            return trend.trend_direction == "increasing"
        if trend.metric_name in declining_metrics:
            return trend.trend_direction == "decreasing"
        return False  # Neutral or unknown metric type

    def _is_declining_trend(self, trend: TrendAnalysis) -> bool:
        """Determine if a trend represents decline based on metric type."""
        return not self._is_improving_trend(trend) and trend.trend_direction != "stable"

    def _generate_key_insights(
        self,
        significant_trends: list[TrendAnalysis],
        strong_patterns: list[PatternMatch],
        high_anomalies: list[Anomaly],
    ) -> list[str]:
        """Generate key insights from analysis results."""
        insights = []

        # Trend insights
        if significant_trends:
            improving = [t for t in significant_trends if self._is_improving_trend(t)]
            declining = [t for t in significant_trends if self._is_declining_trend(t)]

            if improving:
                insight = f"Positive trends in {len(improving)} metrics: " + ", ".join(
                    [t.metric_name.replace("-", " ") for t in improving[:3]],
                )
                insights.append(insight)

            if declining:
                insight = (
                    f"Concerning trends in {len(declining)} metrics: "
                    + ", ".join(
                        [t.metric_name.replace("-", " ") for t in declining[:3]],
                    )
                )
                insights.append(insight)

        # Pattern insights
        for pattern in strong_patterns[:2]:  # Top 2 patterns
            insights.append(f"Strong correlation detected: {pattern.description}")

        # Anomaly insights
        if high_anomalies:
            insights.append(
                f"Critical anomalies detected in {len(high_anomalies)} metrics",
            )

        return insights

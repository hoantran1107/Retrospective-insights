"""Data fetcher module for retrieving metrics from the dashboard API.

This module handles authentication and data retrieval from the dashboard endpoints
specified in IFDRDD-345.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, ClassVar

import requests

logger = logging.getLogger(__name__)


@dataclass
class AuthToken:
    """Represents an authentication token with expiration."""

    token: str
    expires_in: int
    created_at: datetime

    @property
    def is_expired(self) -> bool:
        """Check if the token has expired (with 30 second buffer)."""
        expiry_time = self.created_at + timedelta(seconds=self.expires_in - 30)
        return datetime.now() >= expiry_time


class MetricsDataFetcher:
    """Handles authentication and data retrieval from dashboard APIs.

    Manages token lifecycle and provides methods to fetch all available metrics.
    """

    # Available metrics as specified in IFDRDD-345
    AVAILABLE_METRICS: ClassVar[list[str]] = [
        "testing-time",
        "review-time",
        "coding-time",
        "root-cause",
        "open-bugs-over-time",
        "bugs-per-environment",
        "sp-distribution",
        "items-out-of-sprint",
        "defect-rate-prod",
        "defect-rate-all",
        "happiness",
    ]

    def __init__(
        self,
        auth_url: str,
        data_url: str,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """Initialize the data fetcher.

        Args:
            auth_url: URL for token authentication
            data_url: URL for data retrieval
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts

        """
        self.auth_url = auth_url
        self.data_url = data_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._current_token: AuthToken | None = None

        # Create session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Retrospective-Insights-Bot/0.1.0",
                "Accept": "application/json",
            },
        )

        logger.info(f"Initialized MetricsDataFetcher with auth_url={auth_url}")

    def get_token(self) -> str:
        """Get a valid authentication token.

        Returns:
            Valid authentication token string

        Raises:
            requests.RequestException: If token retrieval fails

        """
        # Return cached token if still valid
        if self._current_token and not self._current_token.is_expired:
            logger.debug("Using cached authentication token")
            return self._current_token.token

        logger.info("Fetching new authentication token")

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(self.auth_url, timeout=self.timeout)
                response.raise_for_status()

                token_data = response.json()

                # Validate response structure
                if "token" not in token_data:
                    raise ValueError("Token not found in response")

                expires_in = token_data.get("expires_in", 300)  # Default 5 minutes

                self._current_token = AuthToken(
                    token=token_data["token"],
                    expires_in=expires_in,
                    created_at=datetime.now(),
                )

                logger.info(f"Successfully obtained token, expires in {expires_in}s")
                return self._current_token.token

            except requests.RequestException as e:
                logger.warning(f"Token request attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2**attempt)  # Exponential backoff

        raise requests.RequestException("Failed to obtain authentication token")

    def fetch_metric(
        self,
        metric_name: str,
        token: str | None = None,
    ) -> dict[str, Any]:
        """Fetch data for a specific metric.

        Args:
            metric_name: Name of the metric to fetch
            token: Authentication token (will get new one if not provided)

        Returns:
            Dictionary containing the metric data

        Raises:
            ValueError: If metric name is not supported
            requests.RequestException: If data retrieval fails

        """
        if metric_name not in self.AVAILABLE_METRICS:
            raise ValueError(
                f"Unsupported metric: {metric_name}. "
                f"Available metrics: {', '.join(self.AVAILABLE_METRICS)}",
            )

        if not token:
            token = self.get_token()

        headers = {"Authorization": f"Bearer {token}"}

        params = {"name": metric_name}

        logger.debug(f"Fetching metric: {metric_name}")

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    self.data_url,
                    headers=headers,
                    params=params,
                    timeout=self.timeout,
                )

                # Handle token expiration
                if response.status_code == 401:
                    logger.info("Token expired, refreshing...")
                    self._current_token = None
                    token = self.get_token()
                    headers["Authorization"] = f"Bearer {token}"
                    continue

                response.raise_for_status()
                data = response.json()

                logger.debug(f"Successfully fetched {metric_name} data")
                return {
                    "metric": metric_name,
                    "data": data,
                    "timestamp": datetime.now().isoformat(),
                    "success": True,
                }

            except requests.RequestException as e:
                logger.warning(
                    f"Metric fetch attempt {attempt + 1} failed for {metric_name}: {e}",
                )
                if attempt == self.max_retries - 1:
                    return {
                        "metric": metric_name,
                        "data": None,
                        "timestamp": datetime.now().isoformat(),
                        "success": False,
                        "error": str(e),
                    }
                time.sleep(2**attempt)

        return {
            "metric": metric_name,
            "data": None,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "error": "Max retries exceeded",
        }

    async def fetch_all_metrics(
        self,
        max_workers: int | None = 5,
    ) -> dict[str, dict[str, Any]]:
        """Fetch data for all available metrics.

        Returns:
            Dictionary mapping metric names to their data

        """
        logger.info(f"Fetching all {len(self.AVAILABLE_METRICS)} metrics")

        # Get token once for all requests
        token = self.get_token()
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_metric = {
                executor.submit(self.fetch_metric, metric_name, token): metric_name
                for metric_name in self.AVAILABLE_METRICS
            }
            for future in as_completed(future_to_metric):
                metric_name = future_to_metric[future]
                try:
                    result = future.result()
                    results[metric_name] = result
                    logger.debug(f"Completed data for {metric_name}")

                except Exception as e:
                    logger.error(f"Failed to fetch {metric_name}: {e}")
                    results[metric_name] = {
                        "metric": metric_name,
                        "data": None,
                        "timestamp": datetime.now().isoformat(),
                        "success": False,
                        "error": str(e),
                    }

        successful_metrics = [m for m, r in results.items() if r["success"]]
        failed_metrics = [m for m, r in results.items() if not r["success"]]

        logger.info(
            "Fetch complete: %d successful, %d failed",
            len(successful_metrics),
            len(failed_metrics),
        )

        if failed_metrics:
            logger.warning(f"Failed metrics: {', '.join(failed_metrics)}")

        return results

    def validate_metrics_data(
        self,
        metrics_data: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Validate the structure and completeness of fetched metrics data.

        Args:
            metrics_data: Dictionary of metric results

        Returns:
            Validation summary with statistics and recommendations

        """
        total_metrics = len(self.AVAILABLE_METRICS)
        successful_metrics = sum(
            1 for result in metrics_data.values() if result["success"]
        )
        failed_metrics = total_metrics - successful_metrics

        completeness_ratio = successful_metrics / total_metrics

        validation_result = {
            "total_metrics": total_metrics,
            "successful_metrics": successful_metrics,
            "failed_metrics": failed_metrics,
            "completeness_ratio": completeness_ratio,
            "is_sufficient": completeness_ratio >= 0.7,  # 70% threshold
            "failed_metric_names": [
                name for name, result in metrics_data.items() if not result["success"]
            ],
            "recommendation": self._get_recommendation(
                completeness_ratio,
                failed_metrics,
            ),
        }

        logger.info(
            f"Metrics validation: {successful_metrics}/{total_metrics} successful "
            f"({completeness_ratio:.1%} completeness)",
        )

        return validation_result

    def _get_recommendation(self, completeness_ratio: float, failed_count: int) -> str:
        """Generate recommendation based on data completeness."""
        if completeness_ratio >= 0.9:
            return "Excellent data completeness. Proceed with full analysis."
        if completeness_ratio >= 0.7:
            return (
                "Good data completeness. Analysis can proceed with minor limitations."
            )
        if completeness_ratio >= 0.5:
            return "Moderate data completeness. Consider retrying failed metrics or proceeding with limited analysis."
        return "Poor data completeness. Recommend investigating API issues before analysis."

    def close(self) -> None:
        """Close the session and clean up resources."""
        if hasattr(self, "session"):
            self.session.close()
            logger.debug("Closed HTTP session")


# Utility functions for easy module usage


def create_data_fetcher_from_env() -> MetricsDataFetcher:
    """Create a MetricsDataFetcher instance from environment variables.

    Required environment variables:
        DASHBOARD_AUTH_URL: Authentication endpoint URL
        DASHBOARD_DATA_URL: Data retrieval endpoint URL

    Optional environment variables:
        REQUEST_TIMEOUT: Request timeout in seconds (default: 30)
        MAX_RETRIES: Maximum retry attempts (default: 3)

    Returns:
        Configured MetricsDataFetcher instance

    Raises:
        ValueError: If required environment variables are missing

    """
    import os

    from dotenv import load_dotenv

    load_dotenv()

    auth_url = os.getenv("DASHBOARD_AUTH_URL")
    data_url = os.getenv("DASHBOARD_DATA_URL")

    if not auth_url or not data_url:
        raise ValueError(
            "Missing required environment variables: DASHBOARD_AUTH_URL and/or DASHBOARD_DATA_URL",
        )

    timeout = int(os.getenv("REQUEST_TIMEOUT", "30"))
    max_retries = int(os.getenv("MAX_RETRIES", "3"))

    return MetricsDataFetcher(
        auth_url=auth_url,
        data_url=data_url,
        timeout=timeout,
        max_retries=max_retries,
    )

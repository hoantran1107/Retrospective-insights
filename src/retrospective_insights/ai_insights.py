"""AI insights generator for creating hypotheses and actionable suggestions.

This module uses AI/LLM to analyze metrics trends and patterns to generate
evidence-backed hypotheses and actionable experiments for retrospectives.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from dataclasses import dataclass
from typing import Any, List

from openai import AzureOpenAI, OpenAI, AsyncAzureOpenAI
from typing import TypeVar, Generic, Type
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """Represents a hypothesis about team performance based on metrics."""

    id: str
    title: str
    description: str
    evidence: list[str]
    confidence_level: str  # 'high', 'medium', 'low'
    confidence_score: float  # 0.0 to 1.0
    impact_assessment: str
    supporting_metrics: list[str]


class HypothesisReponseModel(BaseModel):
    """Pydantic model for parsing hypothesis response."""

    hypotheses: list[Hypothesis]


@dataclass
class Experiment:
    """Represents a suggested experiment for the next sprint."""

    id: str
    title: str
    description: str
    hypothesis_id: str
    success_criteria: list[str]
    implementation_steps: list[str]
    metrics_to_track: list[str]
    estimated_effort: str  # 'low', 'medium', 'high'
    duration: str  # e.g., "1 sprint", "2 weeks"


class ExperimentResponseModel(BaseModel):
    """Pydantic model for parsing experiment response."""

    experiment: Experiment  # Will be parsed into Experiment dataclass


@dataclass
class FacilitationNote:
    """Represents facilitation notes for retrospective meetings."""

    suggested_questions: list[str]
    agenda_items: list[str]
    discussion_prompts: list[str]
    action_item_templates: list[str]


class FacilitationResponseModel(BaseModel):
    """Pydantic model for parsing facilitation notes response."""

    facilitation_notes: FacilitationNote


class AIInsightsGenerator:
    """Generates insights using AI/LLM based on metrics analysis results.

    Creates hypotheses, suggests experiments, and provides facilitation
    notes for retrospective meetings. Supports both Azure OpenAI and OpenAI.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo",
        max_tokens: int = 2000,
        temperature: float = 0.7,
        azure_endpoint: str | None = None,
        api_version: str = "2024-02-15-preview",
        azure_deployment: str | None = None,
    ):
        """Initialize the AI insights generator.

        Args:
            api_key: OpenAI or Azure OpenAI API key
            model: Model to use for generation
            max_tokens: Maximum tokens per response
            temperature: Temperature for generation (0.0 to 1.0)
            azure_endpoint: Azure OpenAI endpoint URL (if using Azure)
            api_version: Azure OpenAI API version

        """
        if azure_endpoint:
            # Use Azure OpenAI

            self.client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                azure_deployment=azure_deployment,
            )
            self.is_azure = True
            logger.info(
                f"Initialized AIInsightsGenerator with Azure OpenAI with deployment {azure_deployment}",
            )
        else:
            # Use regular OpenAI
            self.client = OpenAI(api_key=api_key)
            self.is_azure = False
            logger.info(f"Initialized AIInsightsGenerator with OpenAI model {model}")

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def generate_hypotheses_async(
        self, analysis_results: dict[str, Any]
    ) -> list[Hypothesis]:
        """Generate evidence-backed hypotheses from analysis results.

        Args:
            analysis_results: Results from MetricsAnalysisEngine

        Returns:
            List of generated hypotheses

        """
        logger.info("Generating hypotheses from analysis results")

        try:
            # Prepare analysis summary for AI
            analysis_summary = self._prepare_analysis_summary(analysis_results)

            # Generate hypotheses using AI
            prompt = self._create_hypothesis_prompt(analysis_summary)
            response = await self._call_openai_async(
                prompt, response_format=HypothesisReponseModel
            )

            # Parse and validate response
            hypotheses = self._parse_hypotheses_response(response)

            logger.info(f"Generated {len(hypotheses)} hypotheses")
            return hypotheses

        except Exception as e:
            logger.error(f"Failed to generate hypotheses: {e}")
            raise

    async def suggest_experiments_async(
        self, hypotheses: list[Hypothesis]
    ) -> list[Experiment]:
        """Generate experiments in parallel for all hypotheses."""

        async def generate_single_experiment(
            hypothesis: Hypothesis,
        ) -> Experiment | None:
            try:
                prompt = self._create_experiment_prompt(hypothesis)
                response = await self._call_openai_async(
                    prompt, ExperimentResponseModel
                )
                return self._parse_experiment_response(response, hypothesis.id)
            except Exception as e:
                logger.error(f"Failed for {hypothesis.id}: {e}")
                return self._create_fallback_experiment(hypothesis)

        # ✅ Run all async functions in parallel
        experiments = await asyncio.gather(
            *[generate_single_experiment(h) for h in hypotheses]
        )

        # Filter, sort, return
        experiments = [e for e in experiments if e is not None]
        experiments.sort(
            key=lambda e: self._calculate_experiment_priority(e, hypotheses),
            reverse=True,
        )
        return experiments[:3]

    async def generate_facilitation_notes_async(
        self,
        hypotheses: list[Hypothesis],
        experiments: list[Experiment],
    ) -> FacilitationNote:
        """Generate facilitation notes for retrospective meetings.

        Args:
            hypotheses: Generated hypotheses
            experiments: Suggested experiments

        Returns:
            Facilitation notes for Scrum Masters

        """
        logger.info("Generating facilitation notes")

        try:
            prompt = self._create_facilitation_prompt(hypotheses, experiments)
            response = await self._call_openai_async(prompt, FacilitationResponseModel)

            notes = self._parse_facilitation_response(response)
            logger.info("Successfully generated facilitation notes")

            return notes

        except Exception as e:
            logger.error(f"Failed to generate facilitation notes: {e}")
            return self._create_fallback_facilitation_notes(hypotheses, experiments)

    def calculate_confidence(
        self,
        hypothesis: Hypothesis,
        analysis_results: dict[str, Any],
    ) -> float:
        """Calculate confidence score for a hypothesis based on evidence strength.

        Args:
            hypothesis: The hypothesis to evaluate
            analysis_results: Analysis results for evidence validation

        Returns:
            Confidence score between 0.0 and 1.0

        """
        confidence_factors = []

        # Factor 1: Number of supporting metrics
        metrics_count = len(hypothesis.supporting_metrics)
        metrics_factor = min(1.0, metrics_count / 3.0)  # Normalize to max 3 metrics
        confidence_factors.append(metrics_factor)

        # Factor 2: Strength of trends in supporting metrics
        trends = analysis_results.get("trends", {}).get("details", {})
        trend_strengths = []

        for metric in hypothesis.supporting_metrics:
            if metric in trends:
                trend = trends[metric]
                if trend.significance == "high":
                    trend_strengths.append(0.9)
                elif trend.significance == "medium":
                    trend_strengths.append(0.6)
                else:
                    trend_strengths.append(0.3)

        trend_factor = (
            sum(trend_strengths) / len(trend_strengths) if trend_strengths else 0.5
        )
        confidence_factors.append(trend_factor)

        # Factor 3: Pattern correlation strength
        patterns = analysis_results.get("patterns", {}).get("details", [])
        pattern_factor = 0.5  # Default

        for pattern in patterns:
            if (
                pattern.primary_metric in hypothesis.supporting_metrics
                or pattern.secondary_metric in hypothesis.supporting_metrics
            ):
                if pattern.strength == "strong":
                    pattern_factor = max(pattern_factor, 0.9)
                elif pattern.strength == "moderate":
                    pattern_factor = max(pattern_factor, 0.7)

        confidence_factors.append(pattern_factor)

        # Calculate weighted average
        final_confidence = sum(confidence_factors) / len(confidence_factors)

        return min(1.0, max(0.0, final_confidence))

    def _prepare_analysis_summary(self, analysis_results: dict[str, Any]) -> str:
        """Prepare a concise summary of analysis results for AI prompts."""
        summary_parts = []

        # Trends summary
        trends = analysis_results.get("trends", {})
        if trends.get("significant"):
            summary_parts.append(
                f"Significant trends detected in {trends['significant']} metrics:",
            )

            trend_details = trends.get("details", {})
            for metric, trend in trend_details.items():
                if trend.significance in ["high", "medium"]:
                    direction = trend.trend_direction
                    change = trend.percent_change * 100
                    summary_parts.append(f"- {metric}: {direction} by {change:.1f}%")

        # Patterns summary
        patterns = analysis_results.get("patterns", {})
        if patterns:
            summary_parts.append("\nCorrelation patterns detected:")
            for pattern_detail in patterns["details"][:3]:  # Top 3 patterns
                summary_parts.append(f"- {pattern_detail.description}")

        # Anomalies summary
        anomalies = analysis_results.get("anomalies", {})
        if anomalies.get("high_severity"):
            summary_parts.append(
                f"\nHigh-severity anomalies: {anomalies['high_severity']}",
            )

        return "\n".join(summary_parts)

    def _create_hypothesis_prompt(self, analysis_summary: str) -> str:
        """Create prompt for hypothesis generation."""
        return f"""
You are an expert Agile coach analyzing team metrics for a retrospective. Based on the following analysis results, generate 2-3 evidence-backed hypotheses about team performance patterns.

Analysis Results:
{analysis_summary}

For each hypothesis, provide:
1. A clear, actionable title
2. Detailed description of the hypothesis
3. Supporting evidence from the metrics
4. Confidence level (high/medium/low)
5. Impact assessment on team performance

Format your response as JSON with the following structure:
{{
  "hypotheses": [
    {{
      "title": "Brief hypothesis title",
      "description": "Detailed explanation of the hypothesis",
      "evidence": ["Evidence point 1", "Evidence point 2"],
      "confidence_level": "high|medium|low", 
      "impact_assessment": "How this affects team performance",
      "supporting_metrics": ["metric1", "metric2"]
    }}
  ]
}}

Focus on:
- Actionable insights that teams can address
- Clear cause-and-effect relationships
- Evidence-based conclusions
- Practical implications for team improvement
"""

    def _create_experiment_prompt(self, hypothesis: Hypothesis) -> str:
        """Create prompt for experiment suggestion."""
        return f"""
You are an expert Agile coach designing experiments to validate team performance hypotheses. Based on the following hypothesis, suggest a concrete, timeboxed experiment for the next sprint.

Hypothesis:
Title: {hypothesis.title}
Description: {hypothesis.description}
Evidence: {", ".join(hypothesis.evidence)}
Supporting Metrics: {", ".join(hypothesis.supporting_metrics)}

Design an experiment that:
1. Can be completed in 1-2 sprints
2. Has clear success criteria
3. Is specific and actionable
4. Measures the right metrics
5. Has low risk and effort

Make it practical and immediately actionable for a Scrum team.
"""

    def _create_facilitation_prompt(
        self,
        hypotheses: list[Hypothesis],
        experiments: list[Experiment],
    ) -> str:
        """Create prompt for facilitation notes."""
        hypotheses_summary = "\n".join(
            [f"- {h.title}: {h.description}" for h in hypotheses],
        )
        experiments_summary = "\n".join(
            [f"- {e.title}: {e.description}" for e in experiments],
        )

        return f"""
You are an experienced Scrum Master preparing facilitation notes for a retrospective meeting. Based on the following AI-generated insights, create facilitation notes to help validate hypotheses and plan experiments.

Hypotheses:
{hypotheses_summary}

Suggested Experiments:
{experiments_summary}

Create facilitation notes including:
1. 3-4 discussion questions to validate hypotheses with the team
2. Agenda items for a 15-minute insights review
3. Discussion prompts to engage team members
4. Action item templates for experiment planning

Focus on:
- Questions that encourage team reflection
- Collaborative validation of insights
- Practical next steps
- Team ownership of improvements
"""

    async def _call_openai_async(
        self, prompt: str, response_format: type[T] | None = None
    ) -> str:
        """Make API call to OpenAI and return response."""
        try:
            # Build request parameters
            request_params = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert Agile coach and data analyst.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }

            # Add structured output if Pydantic model provided
            if response_format is not None:
                request_params["response_format"] = response_format
                response = await self.client.chat.completions.parse(**request_params)
                return response.choices[0].message.content  # Type: T

            else:
                response = await self.client.chat.completions.create(**request_params)
                return response.choices[0].message.content  # Type: str

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

    def _parse_hypotheses_response(self, response: str) -> list[Hypothesis]:
        """Parse AI response into Hypothesis objects."""
        try:
            data = json.loads(response)
            hypotheses = []

            for i, hyp_data in enumerate(data.get("hypotheses", [])):
                hypothesis = Hypothesis(
                    id=f"hyp_{i + 1}",
                    title=hyp_data.get("title", ""),
                    description=hyp_data.get("description", ""),
                    evidence=hyp_data.get("evidence", []),
                    confidence_level=hyp_data.get("confidence_level", "medium"),
                    confidence_score=self._confidence_level_to_score(
                        hyp_data.get("confidence_level", "medium"),
                    ),
                    impact_assessment=hyp_data.get("impact_assessment", ""),
                    supporting_metrics=hyp_data.get("supporting_metrics", []),
                )
                hypotheses.append(hypothesis)

            return hypotheses

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse hypotheses response: {e}")
            return []

    def _parse_experiment_response(
        self,
        response: str,
        hypothesis_id: str,
    ) -> Experiment | None:
        """Parse AI response into Experiment object."""
        try:
            data = json.loads(response)
            exp_data = data.get("experiment", {})

            experiment = Experiment(
                id=f"exp_{hypothesis_id}",
                title=exp_data.get("title", ""),
                description=exp_data.get("description", ""),
                hypothesis_id=hypothesis_id,
                success_criteria=exp_data.get("success_criteria", []),
                implementation_steps=exp_data.get("implementation_steps", []),
                metrics_to_track=exp_data.get("metrics_to_track", []),
                estimated_effort=exp_data.get("estimated_effort", "medium"),
                duration=exp_data.get("duration", "1 sprint"),
            )

            return experiment

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse experiment response: {e}")
            return None

    def _parse_facilitation_response(self, response: str) -> FacilitationNote:
        """Parse AI response into FacilitationNote object."""
        try:
            data = json.loads(response)
            notes_data = data.get("facilitation_notes", {})

            return FacilitationNote(
                suggested_questions=notes_data.get("suggested_questions", []),
                agenda_items=notes_data.get("agenda_items", []),
                discussion_prompts=notes_data.get("discussion_prompts", []),
                action_item_templates=notes_data.get("action_item_templates", []),
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse facilitation response: {e}")
            return self._create_default_facilitation_notes()

    def _confidence_level_to_score(self, level: str) -> float:
        """Convert confidence level string to numeric score."""
        mapping = {"high": 0.85, "medium": 0.6, "low": 0.35}
        return mapping.get(level.lower(), 0.5)

    def _calculate_experiment_priority(
        self,
        experiment: Experiment,
        hypotheses: list[Hypothesis],
    ) -> float:
        """Calculate priority score for experiment ranking."""
        # Find related hypothesis
        hypothesis = next(
            (h for h in hypotheses if h.id == experiment.hypothesis_id),
            None,
        )
        if not hypothesis:
            return 0.0

        # Priority factors
        confidence_factor = hypothesis.confidence_score

        effort_factor = {"low": 1.0, "medium": 0.7, "high": 0.4}.get(
            experiment.estimated_effort,
            0.5,
        )

        criteria_factor = min(1.0, len(experiment.success_criteria) / 3.0)

        return confidence_factor * 0.5 + effort_factor * 0.3 + criteria_factor * 0.2

    def _create_fallback_hypotheses(
        self,
        analysis_results: dict[str, Any],
    ) -> list[Hypothesis]:
        """Create fallback hypotheses when AI generation fails."""
        logger.info("Creating fallback hypotheses")

        hypotheses = []
        trends = analysis_results.get("trends", {}).get("details", {})

        # Create basic hypothesis for significant trends
        significant_trends = [
            (name, trend)
            for name, trend in trends.items()
            if trend.significance in ["high", "medium"]
        ]

        if significant_trends:
            metric_name, trend = significant_trends[0]

            hypothesis = Hypothesis(
                id="hyp_fallback_1",
                title=f"{metric_name.replace('-', ' ').title()} Trend Analysis",
                description=f"The {metric_name} metric shows a {trend.trend_direction} trend of {trend.percent_change:.1%}, "
                f"which may indicate process bottlenecks or improvements in team workflow.",
                evidence=[
                    f"{metric_name} changed by {trend.percent_change:.1%} month-over-month",
                ],
                confidence_level="medium",
                confidence_score=0.6,
                impact_assessment="This trend could affect team velocity and quality delivery.",
                supporting_metrics=[metric_name],
            )
            hypotheses.append(hypothesis)

        return hypotheses

    def _create_fallback_experiment(
        self,
        hypothesis: Hypothesis,
    ) -> Experiment | None:
        """Create fallback experiment when AI generation fails."""
        if not hypothesis.supporting_metrics:
            return None

        metric = hypothesis.supporting_metrics[0]

        return Experiment(
            id=f"exp_fallback_{hypothesis.id}",
            title=f"Monitor {metric.replace('-', ' ').title()} Changes",
            description=f"Track {metric} metric closely for the next sprint to validate the hypothesis and identify improvement opportunities.",
            hypothesis_id=hypothesis.id,
            success_criteria=[
                f"Measure {metric} daily",
                "Identify specific improvement actions",
            ],
            implementation_steps=[
                "Set up daily tracking",
                "Review weekly in standup",
                "Document observations",
            ],
            metrics_to_track=[metric],
            estimated_effort="low",
            duration="1 sprint",
        )

    def _create_fallback_facilitation_notes(
        self,
        hypotheses: list[Hypothesis],
        experiments: list[Experiment],
    ) -> FacilitationNote:
        """Create fallback facilitation notes when AI generation fails."""
        return FacilitationNote(
            suggested_questions=[
                "Do these insights match your experience from the past sprint?",
                "What factors might have contributed to these trends?",
                "Which experiment would have the most impact on our team?",
                "How can we measure success for this experiment?",
            ],
            agenda_items=[
                "Review AI-generated insights (5 min)",
                "Validate hypotheses with team experience (5 min)",
                "Select experiment for next sprint (5 min)",
            ],
            discussion_prompts=[
                "Let's discuss if these patterns match what we've observed",
                "What additional context should we consider?",
                "How confident are we in these conclusions?",
            ],
            action_item_templates=[
                "Implement [experiment] for next sprint",
                "Track [metrics] daily/weekly",
                "Review experiment results in next retro",
            ],
        )

    def _create_default_facilitation_notes(self) -> FacilitationNote:
        """Create default facilitation notes structure."""
        return FacilitationNote(
            suggested_questions=[],
            agenda_items=[],
            discussion_prompts=[],
            action_item_templates=[],
        )


# Utility functions


def create_ai_generator_from_env() -> AIInsightsGenerator:
    """Create an AIInsightsGenerator from environment variables.

    For Azure OpenAI, required environment variables:
        AZURE_OPENAI_API_KEY: Azure OpenAI API key
        AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL

    For regular OpenAI, required environment variables:
        OPENAI_API_KEY: OpenAI API key

    Optional environment variables:
        OPENAI_MODEL or AZURE_OPENAI_MODEL: Model to use (default: gpt-4-turbo)
        OPENAI_MAX_TOKENS: Max tokens (default: 2000)
        AZURE_OPENAI_API_VERSION: API version for Azure (default: 2024-02-15-preview)

    Returns:
        Configured AIInsightsGenerator instance

    Raises:
        ValueError: If required environment variables are missing

    """
    import os

    from dotenv import load_dotenv

    load_dotenv()

    # Check for Azure OpenAI first
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    if azure_api_key and azure_endpoint:
        # Use Azure OpenAI
        model = os.getenv("AZURE_OPENAI_MODEL") or os.getenv(
            "OPENAI_MODEL",
            "gpt-4-turbo",
        )
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        return AIInsightsGenerator(
            api_key=azure_api_key,
            model=model,
            max_tokens=max_tokens,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            azure_deployment=azure_deployment,
        )

    # Fall back to regular OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "Missing required environment variables. For Azure OpenAI: "
            "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT. "
            "For OpenAI: OPENAI_API_KEY",
        )

    model = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))

    return AIInsightsGenerator(
        api_key=openai_api_key,
        model=model,
        max_tokens=max_tokens,
    )

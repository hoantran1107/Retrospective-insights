"""Command-line interface for the Retrospective Insights Bot.

Provides easy-to-use commands for generating retrospective reports.
"""

import logging
import sys
from pathlib import Path

import click

from .ai_insights import create_ai_generator_from_env
from .analysis_engine import MetricsAnalysisEngine
from .data_fetcher import create_data_fetcher_from_env
from .report_generator import create_report_generator_from_env


def setup_logging(level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
)
def cli(log_level: str):
    """AI-powered retrospective insights bot for Scrum teams."""
    setup_logging(log_level)


@cli.command()
@click.option("--team-name", "-t", default="Team", help="Name of the team")
@click.option("--months", "-m", default=5, help="Number of months to analyze")
@click.option(
    "--format",
    "-f",
    "output_format",
    default="markdown",
    type=click.Choice(["markdown", "html"]),
    help="Output format",
)
@click.option("--output", "-o", help="Output file path (optional)")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without actually doing it",
)
def generate(
    team_name: str,
    months: int,
    output_format: str,
    output: str | None,
    dry_run: bool,
):
    """Generate a retrospective insights report."""
    click.echo(f"🚀 Generating retrospective insights for {team_name}")

    if dry_run:
        click.echo("🔍 DRY RUN MODE - No actual report will be generated")
        click.echo(f"   Team: {team_name}")
        click.echo(f"   Analysis Period: {months} months")
        click.echo(f"   Output Format: {output_format}")
        click.echo(f"   Output File: {output or 'auto-generated'}")
        return

    try:
        # Step 1: Fetch data
        click.echo("📊 Fetching metrics data...")
        data_fetcher = create_data_fetcher_from_env()

        with click.progressbar(length=100, label="Fetching metrics") as bar:
            metrics_data = data_fetcher.fetch_all_metrics()
            bar.update(100)

        validation = data_fetcher.validate_metrics_data(metrics_data)

        if not validation["is_sufficient"]:
            click.echo(f"⚠️  Warning: {validation['recommendation']}")
            if not click.confirm("Continue with limited data?"):
                return

        click.echo(
            f"✅ Successfully fetched {validation['successful_metrics']}/{validation['total_metrics']} metrics",
        )

        # Step 2: Analyze data
        click.echo("🔍 Analyzing trends and patterns...")
        analysis_engine = MetricsAnalysisEngine(lookback_months=months)

        with click.progressbar(length=100, label="Analyzing data") as bar:
            analysis_engine.load_metrics_data(metrics_data)
            bar.update(30)

            analysis_results = analysis_engine.generate_analysis_summary()
            bar.update(100)

        click.echo(
            f"✅ Analysis complete: {analysis_results['trends']['significant']} significant trends detected",
        )

        # Step 3: Generate insights
        click.echo("🤖 Generating AI insights...")
        ai_generator = create_ai_generator_from_env()

        with click.progressbar(length=100, label="AI analysis") as bar:
            hypotheses = ai_generator.generate_hypotheses(analysis_results)
            bar.update(50)

            experiments = ai_generator.suggest_experiments(hypotheses)
            bar.update(80)

            facilitation_notes = ai_generator.generate_facilitation_notes(
                hypotheses,
                experiments,
            )
            bar.update(100)

        click.echo(
            f"✅ Generated {len(hypotheses)} hypotheses and {len(experiments)} experiments",
        )

        # Step 4: Generate report
        click.echo("📝 Creating report...")
        report_generator = create_report_generator_from_env()

        with click.progressbar(length=100, label="Generating report") as bar:
            report = report_generator.create_report(
                analysis_results=analysis_results,
                hypotheses=hypotheses,
                experiments=experiments,
                facilitation_notes=facilitation_notes,
                team_name=team_name,
                format_type=output_format,
            )
            bar.update(80)

            output_path = report_generator.export_to_file(report, output)
            bar.update(100)

        click.echo("✅ Report generated successfully!")
        click.echo(f"📄 File: {output_path}")
        click.echo(
            f"📊 Contains: {report['metadata']['charts_count']} charts, "
            f"{report['metadata']['hypotheses_count']} hypotheses, "
            f"{report['metadata']['experiments_count']} experiments",
        )

    except Exception as e:
        click.echo(f"❌ Error generating report: {e}")
        if click.get_current_context().find_root().params.get("log_level") == "DEBUG":
            import traceback

            traceback.print_exc()
        sys.exit(1)
    finally:
        if "data_fetcher" in locals():
            data_fetcher.close()


@cli.command()
def test_connection():
    """Test connection to dashboard APIs."""
    click.echo("🔧 Testing API connections...")

    try:
        data_fetcher = create_data_fetcher_from_env()

        # Test authentication
        click.echo("🔑 Testing authentication...")
        token = data_fetcher.get_token()
        click.echo("✅ Authentication successful")

        # Test data fetching with one metric
        click.echo("📊 Testing data retrieval...")
        test_result = data_fetcher.fetch_metric("happiness", token)

        if test_result["success"]:
            click.echo("✅ Data retrieval successful")
        else:
            click.echo(
                f"❌ Data retrieval failed: {test_result.get('error', 'Unknown error')}",
            )

        click.echo("🎉 All connections working!")

    except Exception as e:
        click.echo(f"❌ Connection test failed: {e}")
        sys.exit(1)
    finally:
        if "data_fetcher" in locals():
            data_fetcher.close()


@cli.command()
@click.option("--output-dir", "-o", default="reports", help="Output directory")
def init_templates(output_dir: str):
    """Initialize report templates in the specified directory."""
    template_dir = Path(output_dir) / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)

    # Create markdown template
    md_template = """# Retrospective Insights Report

**Team:** {{ team_name }}  
**Generated:** {{ generation_timestamp }}  
**Analysis Period:** {{ analysis_period }}  

## Executive Summary

{{ executive_summary }}

## Key Insights

{% for insight in key_insights %}
- {{ insight }}
{% endfor %}

## Metrics Overview

- **Total Metrics Analyzed:** {{ metrics_overview.total_metrics }}
- **Significant Changes:** {{ metrics_overview.significant_changes }}
- **Analysis Period:** {{ metrics_overview.analysis_period }}

## Top 3 Hypotheses

{% for hypothesis in hypotheses[:3] %}
### {{ loop.index }}. {{ hypothesis.title }}

**Description:** {{ hypothesis.description }}

**Evidence:**
{% for evidence in hypothesis.evidence %}
- {{ evidence }}
{% endfor %}

**Confidence:** {{ hypothesis.confidence_level }}  
**Impact:** {{ hypothesis.impact_assessment }}

{% endfor %}

## Suggested Experiments

{% for experiment in experiments[:3] %}
### {{ loop.index }}. {{ experiment.title }}

**Description:** {{ experiment.description }}

**Success Criteria:**
{% for criteria in experiment.success_criteria %}
- {{ criteria }}
{% endfor %}

**Implementation Steps:**
{% for step in experiment.implementation_steps %}
1. {{ step }}
{% endfor %}

**Effort:** {{ experiment.estimated_effort }}  
**Duration:** {{ experiment.duration }}

{% endfor %}

## Facilitation Notes

### Suggested Questions
{% for question in facilitation_notes.suggested_questions %}
- {{ question }}
{% endfor %}

### Agenda Items
{% for item in facilitation_notes.agenda_items %}
- {{ item }}
{% endfor %}

---
*This report was generated by the AI Retrospective Insights Bot*
"""

    # Write templates
    with open(template_dir / "retrospective_report.md.j2", "w") as f:
        f.write(md_template)

    click.echo(f"✅ Templates initialized in {template_dir}")
    click.echo("📝 You can now customize the templates as needed")


@cli.command()
def version():
    """Show version information."""
    from . import __version__

    click.echo(f"Retrospective Insights Bot v{__version__}")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()

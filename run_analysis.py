#!/usr/bin/env python3
"""
Helper script to run retrospective analysis with team filtering.

Usage:
    python run_analysis.py                      # All teams
    python run_analysis.py "Gridsz Data Team"  # Specific team
    python run_analysis.py --list-teams        # List available teams
"""

import asyncio
import json
import sys
from pathlib import Path
from main import RetrospectiveInsightsBot, logger


def list_available_teams():
    """List all available teams from mapped data."""
    data_file = Path("mapped_sprint_data.json")
    if not data_file.exists():
        print("❌ mapped_sprint_data.json not found. Please run analysis first.")
        return

    print("📋 Scanning for available teams...")
    teams = set()

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

        # Scan all metrics for team names
        for metric_name, metric_data in data.items():
            if isinstance(metric_data, dict) and "data" in metric_data:
                for record in metric_data["data"]:
                    if isinstance(record, dict):
                        # Check various field names
                        for field in [
                            "project_name",
                            "name",
                            "project",
                            "team",
                            "team_name",
                        ]:
                            if field in record and record[field]:
                                teams.add(record[field])

    if teams:
        print(f"\n✅ Found {len(teams)} unique teams:")
        print("=" * 60)
        for team in sorted(teams):
            print(f"  • {team}")
        print("=" * 60)
        print("\nTo run analysis for a specific team:")
        print('  python run_analysis.py "Team Name"')
    else:
        print("❌ No teams found in data")


async def run_team_analysis(team_name: str = None):
    """Run analysis for a specific team or all teams."""
    if team_name:
        print(f"🎯 Running analysis for: {team_name}")
    else:
        print("🎯 Running analysis for: All Teams")

    bot = RetrospectiveInsightsBot(team_filter=team_name)
    report_path = await bot.run_complete_analysis()

    print(f"\n✅ SUCCESS!")
    print(f"📄 Report: {report_path}")
    if team_name:
        print(f"👥 Team: {team_name}")
    else:
        print("👥 Teams: All Teams")


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--list-teams":
        list_available_teams()
        return

    team_filter = None
    if len(sys.argv) > 1:
        team_filter = sys.argv[1]

    asyncio.run(run_team_analysis(team_filter))


if __name__ == "__main__":
    main()

# Retrospective Insights Bot

An AI-powered bot that analyzes Jira metrics from the last 5 sprints to generate actionable retrospective insights for Scrum teams. Implements Jira ticket AISTR-30.

## Features

- **Automated Data Collection**: Fetches metrics from real dashboard APIs (no mocks)
- **Team Filtering**: Analyze specific teams or all teams combined
- **Trend Analysis**: Identifies sprint-over-sprint changes and patterns
- **AI-Powered Insights**: Generates evidence-backed hypotheses using Azure OpenAI
- **Actionable Reports**: Creates HTML reports with charts and facilitation notes
- **Real-Time Processing**: ~10 weeks of data analyzed in under 2 minutes
- **REST API**: FastAPI-based REST API for integration with other tools

## Quick Start

### Installation

```bash
# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

### Configuration

Set environment variables or create a `.env` file:

```env
# Dashboard API endpoints
DASHBOARD_AUTH_URL=<DASHBOARD_AUTH_URL>
DASHBOARD_DATA_URL=<DASHBOARD_DATA_URL>

# Azure OpenAI (auto-detected)
AZURE_OPENAI_API_KEY=your-azure-api-key
AZURE_OPENAI_ENDPOINT=https://your-instance.openai.azure.com/

# Or regular OpenAI (fallback)
OPENAI_API_KEY=sk-your-openai-api-key
```

### Usage

#### Command Line Interface

##### Analyze All Teams

```bash
# Using main.py
python main.py

# Using helper script (recommended)
python run_analysis.py
```

##### Analyze Specific Team

```bash
# Using main.py
python main.py "Gridsz Data Team"

# Using helper script (recommended)
python run_analysis.py "Gridsz Data Team"
```

##### List Available Teams

```bash
python run_analysis.py --list-teams
```

#### REST API

Start the FastAPI server:

```powershell
# Development mode (with auto-reload)
python api.py

# Production mode
uvicorn api:app --host 127.0.0.1 --port 8000 --workers 4
```

The API will be available at:
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

**Example API Usage:**

```powershell
# Start analysis for specific team
$body = @{ team_filter = "Gridsz Data Team" } | ConvertTo-Json
$response = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/analysis" `
    -Method Post -ContentType "application/json" -Body $body
$taskId = $response.task_id

# Check status
$status = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/analysis/$taskId"

# Download report when completed
Invoke-WebRequest -Uri "http://localhost:8000/api/v1/analysis/$taskId/report" `
    -OutFile "report.html"
```

**Interactive Test Script:**

```powershell
# Run the interactive test script
.\test_api.ps1
```

For detailed API documentation, see [API_README.md](API_README.md).

## Team Filtering

The bot supports filtering data by team name. When a team filter is applied:

- Only records matching the specified team are analyzed
- Report is generated with the team name in the title
- Summary shows filtered record count vs. total records

For detailed documentation on team filtering, see [TEAM_FILTERING.md](TEAM_FILTERING.md).

**Example Output:**
```
📊 Fetching data for last 5 sprints (Team: Gridsz Data Team)...
✅ Fetched and mapped 10 metrics for analysis
📈 Analysis: 0 trends, 0 patterns, 0 anomalies
📄 Report: reports\retrospective_report_gridsz_data_team_20251030_223701.html
👥 Team: Gridsz Data Team
📊 Records processed: 98 (filtered from 3566 total)
```

## Available Metrics

The bot analyzes the following metrics:

- **Time Metrics**: coding-time, review-time, testing-time
- **Quality Metrics**: defect-rate-prod, defect-rate-all, root-cause
- **Flow Metrics**: items-out-of-sprint, sp-distribution
- **Bug Metrics**: open-bugs-over-time, bugs-per-environment
- **Team Metrics**: happiness

## Report Structure

Each generated report includes:

1. **Executive Summary**: 1-2 line verdict with key findings
2. **Trend Charts**: Month-over-month visualizations with callouts
3. **Top 3 Hypotheses**: Evidence-backed insights with confidence levels
4. **Suggested Experiments**: 1-3 actionable items for next sprint
5. **Facilitation Notes**: Questions and agenda for retrospective meetings

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=retrospective_insights

# Run specific test category
uv run pytest tests/unit/
uv run pytest tests/integration/
```

### Code Quality

```bash
# Format code
uv run black src/ tests/
uv run isort src/ tests/

# Lint code
uv run flake8 src/ tests/
uv run mypy src/
```

## Architecture

The bot consists of four main components:

- **Data Fetcher**: Handles API authentication and data retrieval
- **Analysis Engine**: Processes metrics and identifies patterns
- **AI Insights Generator**: Creates hypotheses and suggestions using LLMs
- **Report Generator**: Formats results into actionable reports

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions or issues, please create an issue in the GitHub repository or contact the development team.
Retrospective insights for Jira

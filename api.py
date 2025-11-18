"""FastAPI application for Retrospective Insights Bot.

Provides REST API endpoints for generating retrospective insights.
"""

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from main import RetrospectiveInsightsBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Retrospective Insights API",
    description="AI-powered bot that analyzes team metrics to generate retrospective insights",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Store analysis tasks
analysis_tasks = {}


# Request/Response Models
class AnalysisRequest(BaseModel):
    """Request model for analysis."""

    team_filter: str | None = Field(
        None,
        description="Team name to filter data (e.g., 'Gridsz Data Team'). If not provided, analyzes all teams.",
        example="Gridsz Data Team",
    )

    team_happiness_name: str | None = Field(
        None,
        description="Team name for happiness metric (e.g., 'Gridsz Data'). If not provided, uses default team.",
        example="Gridsz Data",
    )

    project_name: str | None = Field(
        None,
        description="Project name to filter data (e.g., 'Project X'). If not provided, analyzes all projects.",
        example="GRIDSZDT",
    )


class AnalysisResponse(BaseModel):
    """Response model for analysis initiation."""

    task_id: str = Field(..., description="Unique task ID for tracking the analysis")
    status: str = Field(..., description="Current status of the analysis")
    message: str = Field(..., description="Status message")
    team_filter: str | None = (
        Field(
            None,
            description="Team filter applied to the analysis",
        ),
    )
    team_happiness_name: str | None = Field(
        None,
        description="Team name for happiness metric",
    )


class TaskStatus(BaseModel):
    """Response model for task status."""

    task_id: str = Field(..., description="Task ID")
    status: str = Field(
        ...,
        description="Status: 'pending', 'running', 'completed', 'failed'",
    )
    message: str = Field(..., description="Status message")
    team_filter: str | None = Field(None, description="Team filter used")
    report_path: str | None = Field(None, description="Path to generated report")
    error: str | None = Field(None, description="Error message if failed")
    created_at: str = Field(..., description="Task creation timestamp")
    completed_at: str | None = Field(None, description="Task completion timestamp")


# Background task function
async def run_analysis_task(
    task_id: str,
    team_filter: str | None = None,
    team_happiness_name: str | None = None,
    project_name: str | None = None,
) -> None:
    """Run the analysis in the background."""
    try:
        logger.info(
            "Starting analysis task %s for team: %s",
            task_id,
            team_filter or "All",
        )
        analysis_tasks[task_id]["status"] = "running"
        analysis_tasks[task_id]["message"] = "Analysis in progress..."

        # Create bot instance and run analysis
        bot = RetrospectiveInsightsBot(
            team_filter=team_filter,
            team_happiness_name=team_happiness_name,
            project_name=project_name,
        )
        report_path = await bot.run_complete_analysis()

        # Update task status
        analysis_tasks[task_id]["status"] = "completed"
        analysis_tasks[task_id]["message"] = "Analysis completed successfully"
        analysis_tasks[task_id]["report_path"] = report_path
        analysis_tasks[task_id]["completed_at"] = datetime.now(tz=UTC).isoformat()

        logger.info("Task %s completed successfully", task_id)

    except Exception:
        logger.exception("Task %s failed", task_id)
        analysis_tasks[task_id]["status"] = "failed"
        analysis_tasks[task_id]["message"] = "Analysis failed"
        analysis_tasks[task_id]["error"] = "Analysis error occurred"
        analysis_tasks[task_id]["completed_at"] = datetime.now(tz=UTC).isoformat()


# API Endpoints
@app.get("/", tags=["Root"])
async def root() -> dict[str, str | dict[str, str]]:
    """Root endpoint with API information."""
    return {
        "name": "Retrospective Insights API",
        "version": "0.1.0",
        "description": "AI-powered bot for generating retrospective insights from team metrics",
        "endpoints": {
            "POST /api/v1/analysis": "Start a new retrospective analysis",
            "GET /api/v1/analysis/{task_id}": "Get analysis task status",
            "GET /api/v1/analysis/{task_id}/report": "Download generated report",
            "GET /api/v1/tasks": "List all analysis tasks",
            "GET /health": "Health check endpoint",
        },
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "service": "retrospective-insights-api",
    }


@app.post(
    "/api/v1/analysis",
    response_model=AnalysisResponse,
    tags=["Analysis"],
    status_code=202,
)
async def create_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
) -> AnalysisResponse:
    """Start a new retrospective analysis.

    This endpoint initiates an analysis that runs in the background.
    Use the returned task_id to check status and retrieve results.

    Args:
        request: Analysis request with optional team filter
        background_tasks: FastAPI background tasks manager

    Returns:
        AnalysisResponse with task_id and status

    """
    # Generate unique task ID
    task_id = (
        f"task_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}_{len(analysis_tasks)}"
    )

    # Initialize task status
    analysis_tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "message": "Analysis queued",
        "team_filter": request.team_filter,
        "report_path": None,
        "error": None,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "completed_at": None,
    }

    # Add background task
    background_tasks.add_task(
        run_analysis_task,
        task_id,
        request.team_filter,
        request.team_happiness_name,
        request.project_name,
    )

    logger.info(
        "Created analysis task %s for team: %s",
        task_id,
        request.team_filter or "All",
    )

    return AnalysisResponse(
        task_id=task_id,
        status="pending",
        message="Analysis queued successfully. Use the task_id to check status.",
        team_filter=request.team_filter,
        team_happiness_name=request.team_happiness_name,
    )


@app.get(
    "/api/v1/analysis/{task_id}",
    response_model=TaskStatus,
    tags=["Analysis"],
)
async def get_analysis_status(task_id: str) -> TaskStatus:
    """Get the status of an analysis task.

    Args:
        task_id: The task ID returned when creating the analysis

    Returns:
        TaskStatus with current status and details

    Raises:
        HTTPException: If task_id not found

    """
    if task_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return TaskStatus(**analysis_tasks[task_id])


@app.get(
    "/api/v1/analysis/{task_id}/report",
    tags=["Analysis"],
    response_class=FileResponse,
)
async def download_report(task_id: str) -> FileResponse:
    """Download the generated report for a completed analysis.

    Args:
        task_id: The task ID of the completed analysis

    Returns:
        FileResponse with the HTML report

    Raises:
        HTTPException: If task not found, not completed, or report missing

    """
    if task_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = analysis_tasks[task_id]

    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Analysis not completed. Current status: {task['status']}",
        )

    report_path = task.get("report_path")
    if not report_path or not Path(report_path).exists():
        raise HTTPException(status_code=404, detail="Report file not found")

    # Return the HTML report as a file download
    return FileResponse(
        path=report_path,
        media_type="text/html",
        filename=Path(report_path).name,
    )


@app.get("/api/v1/tasks", tags=["Analysis"])
async def list_tasks(
    status: str | None = None,
    team_filter: str | None = None,
) -> dict[str, int | list[dict[str, str | None]]]:
    """List all analysis tasks with optional filters.

    Args:
        status: Optional status filter
        team_filter: Optional team name filter

    Returns:
        List of tasks matching the filters

    """
    tasks = list(analysis_tasks.values())

    # Apply filters
    if status:
        tasks = [t for t in tasks if t["status"] == status]
    if team_filter:
        tasks = [t for t in tasks if t.get("team_filter") == team_filter]

    return {
        "total": len(tasks),
        "tasks": tasks,
    }


@app.delete("/api/v1/analysis/{task_id}", tags=["Analysis"])
async def delete_task(task_id: str) -> dict[str, str]:
    """Delete a task from the task list.

    Args:
        task_id: The task ID to delete

    Returns:
        Success message

    Raises:
        HTTPException: If task not found or still running

    """
    if task_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = analysis_tasks[task_id]
    if task["status"] == "running":
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a running task",
        )

    del analysis_tasks[task_id]
    return {"message": f"Task {task_id} deleted successfully"}


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(_request: object, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now(tz=UTC).isoformat(),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

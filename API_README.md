# Retrospective Insights API

FastAPI-based REST API for generating AI-powered retrospective insights from team metrics.

## Features

- 🚀 Asynchronous analysis execution with background tasks
- 📊 Team-specific or organization-wide analysis
- 📈 Real-time task status tracking
- 📄 HTML report generation and download
- 🔍 Task filtering and management
- 📚 Interactive API documentation (Swagger UI)

## Installation

1. Install dependencies:
```powershell
pip install -e .
```

Or install with development dependencies:
```powershell
pip install -e ".[dev]"
```

2. Set up environment variables in `.env`:
```env
DASHBOARD_AUTH_URL=https://n8n.idp.infodation.vn/webhook/...
DASHBOARD_DATA_URL=https://n8n.idp.infodation.vn/webhook/...
OPENAI_API_KEY=your_openai_api_key
# Or for Azure OpenAI:
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
```

## Running the API

### Development Mode (with auto-reload)
```powershell
python api.py
```

### Production Mode
```powershell
uvicorn api:app --host 127.0.0.1 --port 8000 --workers 4
```

The API will be available at:
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Root
- `GET /` - API information and available endpoints
- `GET /health` - Health check

### Analysis
- `POST /api/v1/analysis` - Start a new analysis
- `GET /api/v1/analysis/{task_id}` - Get task status
- `GET /api/v1/analysis/{task_id}/report` - Download HTML report
- `GET /api/v1/tasks` - List all tasks (with optional filters)
- `DELETE /api/v1/analysis/{task_id}` - Delete a task

## Usage Examples

### 1. Start Analysis for All Teams

```powershell
# Using Invoke-RestMethod (PowerShell)
$response = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/analysis" `
    -Method Post `
    -ContentType "application/json" `
    -Body '{}'

$taskId = $response.task_id
Write-Host "Task ID: $taskId"
```

### 2. Start Analysis for Specific Team

```powershell
$body = @{
    team_filter = "Gridsz Data Team"
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/analysis" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body

$taskId = $response.task_id
```

### 3. Check Task Status

```powershell
$status = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/analysis/$taskId"
Write-Host "Status: $($status.status)"
Write-Host "Message: $($status.message)"
```

### 4. Download Report (when completed)

```powershell
# Check if completed
$status = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/analysis/$taskId"

if ($status.status -eq "completed") {
    # Download report
    Invoke-WebRequest -Uri "http://localhost:8000/api/v1/analysis/$taskId/report" `
        -OutFile "report.html"
    Write-Host "Report downloaded: report.html"
}
```

### 5. List All Tasks

```powershell
# All tasks
$tasks = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tasks"

# Filter by status
$completedTasks = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tasks?status=completed"

# Filter by team
$teamTasks = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tasks?team_filter=Gridsz Data Team"
```

### 6. Delete a Task

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/analysis/$taskId" -Method Delete
```

## Request/Response Models

### AnalysisRequest
```json
{
  "team_filter": "Gridsz Data Team"  // Optional
}
```

### AnalysisResponse
```json
{
  "task_id": "task_20251102_143025_0",
  "status": "pending",
  "message": "Analysis queued successfully",
  "team_filter": "Gridsz Data Team"
}
```

### TaskStatus
```json
{
  "task_id": "task_20251102_143025_0",
  "status": "completed",  // pending, running, completed, failed
  "message": "Analysis completed successfully",
  "team_filter": "Gridsz Data Team",
  "report_path": "reports/retrospective_report_gridsz_data_team_20251102_143045.html",
  "error": null,
  "created_at": "2025-11-02T14:30:25.123456+00:00",
  "completed_at": "2025-11-02T14:30:45.654321+00:00"
}
```

## Complete Workflow Example

```powershell
# 1. Start analysis
$body = @{ team_filter = "Gridsz Data Team" } | ConvertTo-Json
$response = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/analysis" `
    -Method Post -ContentType "application/json" -Body $body
$taskId = $response.task_id

# 2. Poll for completion
do {
    Start-Sleep -Seconds 5
    $status = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/analysis/$taskId"
    Write-Host "Status: $($status.status) - $($status.message)"
} while ($status.status -in @("pending", "running"))

# 3. Download report if successful
if ($status.status -eq "completed") {
    Invoke-WebRequest -Uri "http://localhost:8000/api/v1/analysis/$taskId/report" `
        -OutFile "report.html"
    Write-Host "✅ Report downloaded successfully!"
    Start-Process "report.html"  # Open in browser
} else {
    Write-Host "❌ Analysis failed: $($status.error)"
}
```

## Error Handling

The API returns standard HTTP status codes:

- `200` - Success
- `202` - Accepted (analysis queued)
- `400` - Bad Request (invalid parameters)
- `404` - Not Found (task or report not found)
- `500` - Internal Server Error

Error response format:
```json
{
  "error": "Error type",
  "detail": "Detailed error message",
  "timestamp": "2025-11-02T14:30:25.123456+00:00"
}
```

## Development

### Testing the API

```powershell
# Install test dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### API Documentation

Visit http://localhost:8000/docs for interactive Swagger UI documentation where you can:
- View all endpoints and their parameters
- Test API calls directly from the browser
- See request/response schemas
- Download OpenAPI specification

## Production Deployment

For production deployment, consider:

1. **Use a production ASGI server:**
   ```powershell
   uvicorn api:app --host 127.0.0.1 --port 8000 --workers 4
   ```

2. **Add authentication/authorization** (e.g., API keys, OAuth2)

3. **Set up HTTPS** with reverse proxy (nginx, Caddy)

4. **Configure logging** for production

5. **Add rate limiting** to prevent abuse

6. **Use persistent task storage** (Redis, PostgreSQL) instead of in-memory dict

7. **Set up monitoring** (Prometheus, Grafana)

## License

See LICENSE file for details.

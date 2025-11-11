# Quick Start Guide - FastAPI Implementation

## ✅ Implementation Complete!

I've successfully implemented a FastAPI REST API for your RetrospectiveInsightsBot with the following features:

### 📦 Files Created

1. **`api.py`** - Complete FastAPI application (342 lines)
   - Background task processing
   - Team filter parameter support
   - Task status tracking
   - Report download endpoint
   - Interactive API documentation

2. **`API_README.md`** - Comprehensive API documentation
   - Installation instructions
   - All endpoints documented
   - PowerShell usage examples
   - Complete workflow examples

3. **`test_api.ps1`** - Interactive test script
   - Menu-driven interface
   - Complete workflow demo
   - Health check functionality
   - Task management

4. **Updated `pyproject.toml`** - Added FastAPI dependencies
   - fastapi>=0.104.0
   - uvicorn[standard]>=0.24.0
   - pydantic>=2.0.0

5. **Updated `README.md`** - Added API usage section

### 🚀 How to Start the API

#### Option 1: Development Mode (Recommended for testing)
```powershell
python api.py
```

#### Option 2: Production Mode
```powershell
uvicorn api:app --host 127.0.0.1 --port 8000
```

#### Option 3: With Multiple Workers
```powershell
uvicorn api:app --host 127.0.0.1 --port 8000 --workers 4
```

### 🔍 API Endpoints

The API provides 11 endpoints:

1. **Root Endpoints:**
   - `GET /` - API information
   - `GET /health` - Health check

2. **Analysis Endpoints:**
   - `POST /api/v1/analysis` - Start new analysis
   - `GET /api/v1/analysis/{task_id}` - Get task status
   - `GET /api/v1/analysis/{task_id}/report` - Download report
   - `DELETE /api/v1/analysis/{task_id}` - Delete task

3. **Task Management:**
   - `GET /api/v1/tasks` - List all tasks (with filters)

4. **Documentation:**
   - `GET /docs` - Swagger UI (interactive)
   - `GET /redoc` - ReDoc documentation

### 📝 Quick Test

1. **Start the API:**
```powershell
python api.py
```

2. **Open Swagger UI in browser:**
```
http://localhost:8000/docs
```

3. **Or use the interactive test script:**
```powershell
.\test_api.ps1
```

4. **Or test with PowerShell commands:**
```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Start analysis
$body = @{ team_filter = "Gridsz Data Team" } | ConvertTo-Json
$response = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/analysis" `
    -Method Post -ContentType "application/json" -Body $body
$taskId = $response.task_id

# Check status
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/analysis/$taskId"
```

### 🎯 Key Features

1. **Asynchronous Processing:**
   - Analysis runs in background
   - Returns task ID immediately
   - Non-blocking API calls

2. **Team Filtering:**
   - Optional `team_filter` parameter
   - Analyze specific teams or all teams

3. **Task Management:**
   - Track multiple analyses
   - Filter by status or team
   - Delete completed tasks

4. **Interactive Documentation:**
   - Swagger UI at `/docs`
   - Test endpoints directly from browser
   - See request/response schemas

5. **Error Handling:**
   - Proper HTTP status codes
   - Detailed error messages
   - Global exception handler

### 📊 API Response Examples

**Start Analysis:**
```json
{
  "task_id": "task_20251102_143025_0",
  "status": "pending",
  "message": "Analysis queued successfully",
  "team_filter": "Gridsz Data Team"
}
```

**Task Status:**
```json
{
  "task_id": "task_20251102_143025_0",
  "status": "completed",
  "message": "Analysis completed successfully",
  "team_filter": "Gridsz Data Team",
  "report_path": "reports/retrospective_report_gridsz_data_team_20251102_143045.html",
  "error": null,
  "created_at": "2025-11-02T14:30:25.123456+00:00",
  "completed_at": "2025-11-02T14:30:45.654321+00:00"
}
```

### 🛠️ Next Steps

1. **Test the API:**
   ```powershell
   python api.py
   ```
   Then visit http://localhost:8000/docs

2. **Run the interactive test:**
   ```powershell
   .\test_api.ps1
   ```

3. **Try the complete workflow:**
   - Start analysis
   - Poll for completion
   - Download report

### 📚 Documentation

- Full API docs: [API_README.md](API_README.md)
- Main README: [README.md](README.md)
- Code documentation: http://localhost:8000/docs (when running)

### ✨ Benefits

1. **Easy Integration:** REST API can be called from any language/tool
2. **Background Processing:** Long-running analyses don't block
3. **Task Tracking:** Monitor multiple analyses simultaneously
4. **Team Flexibility:** Filter by team or analyze all teams
5. **Developer Friendly:** Interactive documentation and test scripts

### 🎉 Ready to Use!

Your FastAPI implementation is complete and ready to use. Start the server and test it out:

```powershell
python api.py
```

Then open your browser to http://localhost:8000/docs to explore the API!

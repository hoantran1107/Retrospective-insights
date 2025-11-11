# AI Agent Improvement Plan: Better Hypotheses & Experiments

## 📋 Current State Analysis

### Current Implementation Issues

**Observed in latest run** (Gridsz Data Team - 2025-10-31):

```
❌ OpenAI API call failed: Error code: 401 - Incorrect API key
⚠️  Fallback hypotheses generated (generic, not data-driven)
⚠️  AI insights limited due to API authentication failure
```

**Current Flow**:
```
Analysis Results → AI Insights Generator → Generic Fallback
                           ↓ (if API fails)
                    Fallback Hypotheses
```

**Problems**:
1. **API Key Issue**: Using Azure OpenAI key for regular OpenAI endpoint
2. **Generic Fallbacks**: Not utilizing actual analysis data
3. **Limited Context**: AI doesn't see full metric relationships
4. **No Iteration**: Single-shot generation without refinement
5. **Poor Prioritization**: Top 3 not truly ranked by impact

---

## 🎯 Improvement Goals

### Primary Objectives

1. **Fix API Authentication**: Use correct Azure OpenAI endpoint and credentials
2. **Data-Driven Hypotheses**: Generate insights based on actual metrics analysis
3. **Better Context**: Provide richer context to AI agent
4. **Iterative Refinement**: Multi-step generation with validation
5. **Impact-Based Ranking**: Prioritize by potential impact on team performance

### Success Criteria

✅ **API calls succeed** with Azure OpenAI  
✅ **Hypotheses reference specific metrics** from analysis  
✅ **Experiments are actionable** with clear success metrics  
✅ **Top 3 ranked by impact score** (data-driven)  
✅ **Confidence levels calculated** from analysis trends

---

## 🏗️ Architecture Design

### New AI Agent Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Analysis Results                          │
│  • 10 metrics analyzed                                       │
│  • 9 trends (with direction + confidence)                   │
│  • 1 pattern (correlations)                                  │
│  • 1 anomaly                                                 │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│         STEP 1: Context Enrichment                          │
│  • Extract key metrics (testing-time, review-time, etc.)    │
│  • Identify bottlenecks (highest values)                    │
│  • Detect trends (improving/degrading)                      │
│  • Find correlations (metric relationships)                 │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│         STEP 2: Hypothesis Generation (AI Agent)            │
│  Prompt Engineering:                                         │
│  • "You are a Scrum Master analyzing team metrics..."       │
│  • Provide structured analysis data                         │
│  • Request 5-7 hypothesis candidates                        │
│  • Require evidence from metrics                            │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│         STEP 3: Hypothesis Validation & Scoring             │
│  For each hypothesis:                                        │
│  • Calculate confidence score (trend strength)              │
│  • Calculate impact score (affected metrics)                │
│  • Calculate evidence score (data support)                  │
│  • Overall score = weighted combination                     │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│         STEP 4: Top 3 Selection                             │
│  • Sort by overall score (descending)                       │
│  • Ensure diversity (different metric categories)           │
│  • Select top 3                                             │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│         STEP 5: Experiment Generation (AI Agent)            │
│  For each selected hypothesis:                              │
│  • Generate 2-3 experiment options                          │
│  • Provide: description, metrics, duration, success criteria│
│  • Rank by feasibility & impact                             │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│         STEP 6: Experiment Validation                       │
│  • Ensure experiments are measurable                        │
│  • Check if success criteria reference actual metrics       │
│  • Validate duration is realistic (1-2 sprints)             │
│  • Select best experiment per hypothesis                    │
└────────────────────┬────────────────────────────────────────┘
                     ↓
                  OUTPUT:
         Top 3 Hypotheses + Experiments
```

---

## 🔧 Implementation Steps

### Phase 1: Fix API Authentication (HIGH PRIORITY)

**Problem**: Using Azure OpenAI key with OpenAI endpoint

**Current Code** (`src/retrospective_insights/ai_insights.py`):
```python
# Currently tries OpenAI endpoint with Azure key
self.client = OpenAI(api_key=api_key)  # ❌ Wrong!
```

**Fix**:
```python
# Use Azure OpenAI endpoint
if azure_endpoint:
    self.client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version
    )
else:
    self.client = OpenAI(api_key=api_key)
```

**Files to Modify**:
- `src/retrospective_insights/ai_insights.py` (line 76-108)
- `.env` file (ensure correct Azure settings)

**Estimated Time**: 30 minutes

---

### Phase 2: Enhanced Prompt Engineering

**Goal**: Provide richer context to AI for better hypotheses

**Current Prompt** (simplified):
```python
prompt = f"Analyze these metrics and generate hypotheses..."
```

**Improved Prompt Structure**:
```python
prompt = f"""
You are an expert Scrum Master analyzing sprint metrics for {team_name}.

## Context
- Time Period: {start_date} to {end_date} (5 months)
- Metrics Analyzed: {len(metrics)} metrics
- Team: {team_name}

## Key Findings

### Trends (Month-over-Month)
{format_trends(trends)}

### Correlations Detected
{format_patterns(patterns)}

### Anomalies
{format_anomalies(anomalies)}

### Bottlenecks
{identify_bottlenecks(analysis_results)}

## Task
Generate 5-7 evidence-based hypotheses that explain WHY these patterns exist.

For each hypothesis:
1. **Title**: Clear, specific hypothesis statement
2. **Evidence**: List specific metrics and values that support this
3. **Impact**: Which metrics would improve if this is addressed?
4. **Root Cause**: What underlying issue might this indicate?
5. **Confidence**: Your confidence level (0.0-1.0)

Format as JSON array.
"""
```

**New Helper Methods**:
```python
def _format_trends_for_prompt(self, trends: Dict[str, TrendAnalysis]) -> str:
    """Format trends in readable way for AI."""
    
def _format_patterns_for_prompt(self, patterns: List[PatternMatch]) -> str:
    """Format correlation patterns for AI."""
    
def _identify_bottlenecks(self, analysis_results: Dict[str, Any]) -> str:
    """Identify top 3 bottleneck metrics."""
```

**Files to Modify**:
- `src/retrospective_insights/ai_insights.py`
  - Update `generate_hypotheses()` method
  - Add helper methods for prompt formatting

**Estimated Time**: 2 hours

---

### Phase 3: Hypothesis Scoring System

**Goal**: Rank hypotheses by data-driven impact score

**Scoring Algorithm**:
```python
def calculate_hypothesis_score(
    self, 
    hypothesis: Hypothesis,
    analysis_results: Dict[str, Any]
) -> float:
    """
    Calculate overall score for hypothesis prioritization.
    
    Score = (0.4 × Evidence) + (0.3 × Impact) + (0.3 × Confidence)
    
    Returns: float (0.0 - 1.0)
    """
    
    # Evidence Score (0.0 - 1.0)
    # Based on: number of supporting metrics, trend strength, statistical significance
    evidence_score = self._calculate_evidence_score(hypothesis, analysis_results)
    
    # Impact Score (0.0 - 1.0)
    # Based on: number of affected metrics, severity of issues, potential improvement
    impact_score = self._calculate_impact_score(hypothesis, analysis_results)
    
    # Confidence Score (0.0 - 1.0)
    # From AI + statistical confidence from trends
    confidence_score = hypothesis.confidence
    
    overall_score = (
        0.4 * evidence_score +
        0.3 * impact_score +
        0.3 * confidence_score
    )
    
    return overall_score
```

**Components**:

#### Evidence Score
```python
def _calculate_evidence_score(
    self, 
    hypothesis: Hypothesis,
    analysis_results: Dict[str, Any]
) -> float:
    """
    Evidence quality based on:
    - Number of supporting metrics (more = better)
    - Trend confidence levels (higher = better)
    - Statistical significance (p-value < 0.05)
    """
    
    supporting_metrics = self._extract_referenced_metrics(hypothesis.evidence)
    
    if not supporting_metrics:
        return 0.3  # Low score if no metrics referenced
    
    # Average trend confidence for referenced metrics
    trend_confidences = [
        analysis_results['trends'][metric].confidence
        for metric in supporting_metrics
        if metric in analysis_results['trends']
    ]
    
    if trend_confidences:
        avg_confidence = sum(trend_confidences) / len(trend_confidences)
    else:
        avg_confidence = 0.5
    
    # Number of supporting metrics (normalized)
    metric_score = min(len(supporting_metrics) / 5.0, 1.0)
    
    return (0.6 * avg_confidence) + (0.4 * metric_score)
```

#### Impact Score
```python
def _calculate_impact_score(
    self, 
    hypothesis: Hypothesis,
    analysis_results: Dict[str, Any]
) -> float:
    """
    Potential impact based on:
    - Number of affected metrics
    - Severity of current issues
    - Scope of potential improvement
    """
    
    affected_metrics = self._extract_referenced_metrics(hypothesis.impact)
    
    # High-impact metrics (testing-time, review-time, defect rates)
    high_impact_keywords = [
        'testing', 'review', 'defect', 'bug', 'quality', 'velocity'
    ]
    
    high_impact_count = sum(
        1 for metric in affected_metrics
        if any(keyword in metric.lower() for keyword in high_impact_keywords)
    )
    
    # Severity from anomalies
    severity_score = 0.0
    for anomaly in analysis_results.get('anomalies', []):
        if anomaly.metric in affected_metrics:
            severity_score += anomaly.severity / 3.0  # Normalize (max severity = 3)
    
    # Scope (number of affected areas)
    scope_score = min(len(affected_metrics) / 5.0, 1.0)
    
    return (
        0.4 * min(high_impact_count / 3.0, 1.0) +
        0.3 * min(severity_score, 1.0) +
        0.3 * scope_score
    )
```

**New Data Structures**:
```python
@dataclass
class ScoredHypothesis(Hypothesis):
    """Hypothesis with scoring information."""
    evidence_score: float
    impact_score: float
    overall_score: float
    referenced_metrics: List[str]
```

**Files to Modify**:
- `src/retrospective_insights/ai_insights.py`
  - Add scoring methods
  - Update `Hypothesis` dataclass
  - Modify `generate_hypotheses()` to score and rank

**Estimated Time**: 3 hours

---

### Phase 4: Improved Experiment Generation

**Goal**: Generate actionable, measurable experiments

**Enhanced Prompt**:
```python
experiment_prompt = f"""
You are helping a Scrum team design an experiment to test this hypothesis:

## Hypothesis
{hypothesis.title}

**Evidence**: {hypothesis.evidence}
**Impact**: {hypothesis.impact}
**Root Cause**: {hypothesis.root_cause}

## Available Metrics
{list_available_metrics()}

## Task
Design 2-3 concrete experiments to test this hypothesis.

For each experiment:
1. **Title**: Short, action-oriented title (5-8 words)
2. **Description**: What the team will do (be specific)
3. **Duration**: How long to run (1-2 sprints recommended)
4. **Success Metrics**: Which metrics to track (use actual metric names above)
5. **Success Criteria**: Numeric targets (e.g., "reduce testing-time by 20%")
6. **Feasibility**: Low/Medium/High
7. **Expected Impact**: Low/Medium/High

Prioritize experiments by feasibility and impact.
Format as JSON array.
"""
```

**Experiment Validation**:
```python
def _validate_experiment(
    self, 
    experiment: Experiment,
    available_metrics: List[str]
) -> bool:
    """
    Validate experiment has:
    - Realistic duration (1-3 sprints)
    - Measurable success metrics (actual metric names)
    - Quantifiable success criteria (numbers)
    - Reasonable feasibility
    """
    
    # Check duration
    if not (1 <= experiment.duration_weeks <= 6):
        logger.warning(f"Experiment duration {experiment.duration_weeks} weeks is unrealistic")
        return False
    
    # Check metrics are valid
    referenced_metrics = self._extract_referenced_metrics(experiment.success_metrics)
    valid_metrics = [m for m in referenced_metrics if m in available_metrics]
    
    if len(valid_metrics) == 0:
        logger.warning(f"Experiment references no valid metrics")
        return False
    
    # Check success criteria has numbers
    if not any(char.isdigit() for char in experiment.success_criteria):
        logger.warning(f"Success criteria has no numeric targets")
        return False
    
    return True
```

**Files to Modify**:
- `src/retrospective_insights/ai_insights.py`
  - Update `suggest_experiments()` method
  - Add `_validate_experiment()` method
  - Add better prompt formatting

**Estimated Time**: 2 hours

---

### Phase 5: Diversity in Top 3 Selection

**Goal**: Ensure Top 3 cover different aspects of team performance

**Selection Algorithm**:
```python
def select_top_3_diverse(
    self, 
    scored_hypotheses: List[ScoredHypothesis]
) -> List[ScoredHypothesis]:
    """
    Select top 3 hypotheses ensuring diversity.
    
    Diversity factors:
    - Different metric categories (testing, review, quality, velocity)
    - Different trend directions (improving vs degrading)
    - Different impact types (process, quality, collaboration)
    """
    
    # Sort by overall score
    sorted_hypotheses = sorted(
        scored_hypotheses, 
        key=lambda h: h.overall_score, 
        reverse=True
    )
    
    selected = []
    metric_categories_used = set()
    
    for hypothesis in sorted_hypotheses:
        if len(selected) >= 3:
            break
        
        # Determine category of this hypothesis
        categories = self._categorize_hypothesis(hypothesis)
        
        # Check if this adds diversity
        if not any(cat in metric_categories_used for cat in categories):
            selected.append(hypothesis)
            metric_categories_used.update(categories)
        elif len(selected) < 3 and hypothesis.overall_score > 0.7:
            # Include high-scoring even if not diverse
            selected.append(hypothesis)
    
    # Fill remaining slots with highest scores
    while len(selected) < 3 and len(sorted_hypotheses) > len(selected):
        next_candidate = sorted_hypotheses[len(selected)]
        if next_candidate not in selected:
            selected.append(next_candidate)
    
    return selected


def _categorize_hypothesis(self, hypothesis: ScoredHypothesis) -> Set[str]:
    """
    Categorize hypothesis by affected areas.
    
    Categories:
    - 'testing': Testing-related metrics
    - 'review': Code review metrics
    - 'quality': Defect rates, bug counts
    - 'velocity': Story points, sprint completion
    - 'collaboration': Happiness, communication
    """
    categories = set()
    
    all_metrics = hypothesis.referenced_metrics
    
    category_keywords = {
        'testing': ['testing', 'test'],
        'review': ['review', 'pr', 'pull request'],
        'quality': ['defect', 'bug', 'quality'],
        'velocity': ['sp', 'story point', 'velocity', 'sprint'],
        'collaboration': ['happiness', 'team', 'communication']
    }
    
    for category, keywords in category_keywords.items():
        if any(
            any(keyword in metric.lower() for keyword in keywords)
            for metric in all_metrics
        ):
            categories.add(category)
    
    return categories if categories else {'general'}
```

**Files to Modify**:
- `src/retrospective_insights/ai_insights.py`
  - Add `select_top_3_diverse()` method
  - Add `_categorize_hypothesis()` method
  - Update `generate_hypotheses()` to use diversity selection

**Estimated Time**: 1.5 hours

---

## 📊 Expected Improvements

### Before (Current State)

```
Top 3 Hypotheses:
1. Generic fallback hypothesis (confidence: 0.5)
   Evidence: "Based on historical patterns"
   Impact: Vague, non-specific
   
Experiments:
- Failed to generate (API error)
```

### After (With Improvements)

```
Top 3 Hypotheses:

1. Testing bottleneck slowing delivery (Score: 0.87, Confidence: 0.85)
   Evidence: 
   - testing-time: 22 hours avg (↑ 23% from last month)
   - Strong correlation with review-time (r=0.73)
   - Affects 5 metrics: testing-time, review-time, items-out-of-sprint
   
   Impact: Addressing this could reduce cycle time by ~30%
   
   Experiment: "Parallel Testing Strategy"
   - Split test suites into parallel jobs
   - Duration: 2 sprints
   - Success Metric: Reduce testing-time from 22h to <16h
   - Expected Impact: HIGH

2. Code review delays blocking velocity (Score: 0.82, Confidence: 0.78)
   Evidence:
   - review-time: 4.5 hours avg (↑ 15%)
   - defect-rate-prod: ↓ 10% (good quality, but slow)
   - Happiness score: 7.2 → 6.8 (↓ declining)
   
   Impact: Faster reviews = more throughput, better morale
   
   Experiment: "Review SLA Policy"
   - 4-hour review SLA, daily review rotation
   - Duration: 1 sprint
   - Success Metric: review-time < 3 hours, happiness > 7.5
   - Expected Impact: MEDIUM

3. Root cause analysis reveals testing gaps (Score: 0.76, Confidence: 0.80)
   Evidence:
   - root-cause: 4 issues, 75% "Missed during testing"
   - defect-rate-all: 8% (↑ slight increase)
   
   Impact: Better testing coverage = fewer production defects
   
   Experiment: "Test Coverage Analysis"
   - Mandatory test coverage reports in PRs
   - Duration: 2 sprints  
   - Success Metric: defect-rate-prod < 5%, zero "missed testing" defects
   - Expected Impact: HIGH
```

---

## 🧪 Testing Strategy

### Unit Tests

```python
# test_ai_insights_scoring.py

def test_evidence_score_calculation():
    """Test evidence score based on metric references."""
    
def test_impact_score_calculation():
    """Test impact score based on affected metrics."""
    
def test_overall_score_calculation():
    """Test weighted combination of scores."""
    
def test_diversity_selection():
    """Test top 3 selection ensures diversity."""
    
def test_hypothesis_validation():
    """Test hypothesis has required evidence."""
    
def test_experiment_validation():
    """Test experiment has measurable criteria."""
```

### Integration Tests

```python
# test_e2e_hypothesis_generation.py

def test_full_hypothesis_generation_flow():
    """Test complete flow with real analysis data."""
    
def test_azure_openai_integration():
    """Test API calls work with Azure endpoint."""
    
def test_fallback_quality():
    """Test fallback hypotheses are still data-driven."""
```

---

## 📅 Implementation Timeline

| Phase | Task | Duration | Priority |
|-------|------|----------|----------|
| 1 | Fix Azure OpenAI authentication | 0.5 hour | 🔴 CRITICAL |
| 2 | Enhanced prompt engineering | 2 hours | 🔴 HIGH |
| 3 | Hypothesis scoring system | 3 hours | 🟡 MEDIUM |
| 4 | Improved experiment generation | 2 hours | 🟡 MEDIUM |
| 5 | Diversity selection algorithm | 1.5 hours | 🟢 LOW |
| 6 | Testing & validation | 2 hours | 🔴 HIGH |
| 7 | Documentation | 1 hour | 🟢 LOW |

**Total Estimated Time**: 12 hours (~1.5 days)

---

## 🚀 Implementation Checklist

### Phase 1: Authentication Fix
- [ ] Update `AIInsightsGenerator.__init__()` to use `AzureOpenAI` client
- [ ] Verify `.env` has correct Azure credentials
- [ ] Test API connection with simple prompt
- [ ] Update error handling for Azure-specific errors

### Phase 2: Prompt Engineering
- [ ] Create `_format_trends_for_prompt()` method
- [ ] Create `_format_patterns_for_prompt()` method
- [ ] Create `_identify_bottlenecks()` method
- [ ] Update `generate_hypotheses()` with enhanced prompt
- [ ] Test prompt with real analysis data

### Phase 3: Scoring System
- [ ] Implement `_calculate_evidence_score()` method
- [ ] Implement `_calculate_impact_score()` method
- [ ] Implement `calculate_hypothesis_score()` method
- [ ] Update `Hypothesis` dataclass with score fields
- [ ] Add unit tests for scoring logic

### Phase 4: Experiment Generation
- [ ] Update experiment generation prompt
- [ ] Implement `_validate_experiment()` method
- [ ] Add experiment ranking by feasibility & impact
- [ ] Test with various hypothesis types

### Phase 5: Diversity Selection
- [ ] Implement `_categorize_hypothesis()` method
- [ ] Implement `select_top_3_diverse()` method
- [ ] Test diversity across different metric categories
- [ ] Verify top 3 selection logic

### Phase 6: Testing
- [ ] Write unit tests for all new methods
- [ ] Write integration tests for full flow
- [ ] Test with real Gridsz Data Team data
- [ ] Verify report quality improvement

### Phase 7: Documentation
- [ ] Update `ai_insights.py` docstrings
- [ ] Create examples in README
- [ ] Document scoring algorithm
- [ ] Add troubleshooting guide

---

## 🎯 Success Metrics

### Quantitative Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| API Success Rate | 0% | 95%+ | % of successful AI calls |
| Hypotheses with Evidence | ~30% | 90%+ | % referencing specific metrics |
| Experiments with Metrics | ~20% | 100% | % with measurable criteria |
| Avg Confidence Score | 0.5 | 0.75+ | Average across top 3 |
| Diversity Score | 1.0 | 2.5+ | # of unique categories in top 3 |

### Qualitative Metrics

✅ **Hypotheses are actionable**: Teams can understand and act on them  
✅ **Experiments are specific**: Clear steps, not vague suggestions  
✅ **Evidence is compelling**: Metrics cited with actual values  
✅ **Impact is measurable**: Success criteria are quantifiable  
✅ **Top 3 are diverse**: Cover different aspects of performance

---

## 🔍 Example Output Comparison

### Current Output (Fallback)

```json
{
  "hypothesis_id": "hyp_fallback_1",
  "title": "Team performance varies by sprint",
  "evidence": "Based on historical patterns and team observations",
  "impact": "Could affect overall delivery",
  "root_cause": "Various factors including workload and complexity",
  "confidence": 0.5
}
```

### Target Output (Improved)

```json
{
  "hypothesis_id": "hyp_001",
  "title": "Testing bottleneck delaying sprint completion",
  "evidence": "testing-time: 22.3h avg (↑23% from 18.2h), review-time correlation r=0.73, items-out-of-sprint: 15% avg",
  "impact": "Reducing testing-time by 30% could decrease items-out-of-sprint to <10% and improve velocity by ~12%",
  "root_cause": "Sequential test execution and insufficient test environment resources",
  "confidence": 0.87,
  "evidence_score": 0.85,
  "impact_score": 0.90,
  "overall_score": 0.87,
  "referenced_metrics": ["testing-time", "review-time", "items-out-of-sprint"],
  "category": ["testing", "velocity"],
  "experiment": {
    "title": "Parallel Testing Implementation",
    "description": "Split test suite into 4 parallel jobs, provision 2 additional test environments",
    "duration_weeks": 2,
    "success_metrics": ["testing-time", "items-out-of-sprint"],
    "success_criteria": "testing-time < 16h (30% reduction), items-out-of-sprint < 10%",
    "feasibility": "MEDIUM",
    "expected_impact": "HIGH"
  }
}
```

---

## 📚 References & Resources

### Azure OpenAI Documentation
- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Python SDK Guide](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/switching-endpoints)

### Prompt Engineering Best Practices
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Few-Shot Learning](https://www.promptingguide.ai/techniques/fewshot)

### Agile Metrics Analysis
- [Scrum Metrics Guide](https://www.scrum.org/resources/blog/11-scrum-metrics-and-their-value)
- [DevOps Metrics](https://cloud.google.com/architecture/devops/devops-measurement-devops-capabilities-and-metrics)

---

**Document Version**: 1.0  
**Created**: October 31, 2025  
**Status**: Ready for Implementation  
**Priority**: HIGH (API fix is critical)

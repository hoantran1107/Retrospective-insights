# Design Document Update Summary

## 🔄 Key Changes from Initial Analysis

### Previous Understanding (WRONG ❌)
- **Only 3 metrics had data**: testing-time, review-time, happiness
- **Field name**: `project_name` for team identifier
- **Date formats**: Only 2 formats ("May 25" and "2024-03")

### Actual Reality (CORRECT ✅)
- **ALL 11 metrics have data!** (except root-cause has no date field)
- **Field name**: `name` is the primary team identifier (not `project_name`)
- **Date formats**: Consistent "Month YY" format (e.g., "May 25", "Jun 25", "Sep 25")
- **Bonus field**: All metrics have `MonthYearSort` (e.g., "202505") - **easier to parse!**

---

## 📊 Complete Metrics Breakdown

| # | Metric | Records | Date Field | Date Format | Team Field | Value Field |
|---|--------|---------|------------|-------------|------------|-------------|
| 1 | testing-time | ~110 | `MonthYear` | "May 25" | `project_name` | `Avg Status Duration` |
| 2 | review-time | ~22 | `MonthYear` | "May 25" | `project_name` | `Avg Status Duration` |
| 3 | coding-time | ~85 | `MonthYearSort` | `202506` | `name` | `Avg Status Duration` |
| 4 | root-cause | ~200 | ❌ None | N/A | `project` | `total` |
| 5 | open-bugs-over-time | ~120 | `MonthYearSort` | `202505` | `project_name` | `Open Bugs EndOfMonth` |
| 6 | bugs-per-environment | ~180 | `MonthYear` | "May 25" | `project` | `total` |
| 7 | sp-distribution | ~3000 | `MonthYearSort` | `202509` | `name` | `total` (+ `issue_type`) |
| 8 | items-out-of-sprint | ~150 | `MonthYear` | "Sep 25" | `project_name` | `% Out-of-Sprint` |
| 9 | defect-rate-prod | ~100 | `MonthYear` | "May 25" | `project_name` | `% Defect Rate (PROD)` |
| 10 | defect-rate-all | ~100 | `MonthYear` | "May 25" | `project_name` | `% Defect Rate (ALL)` |
| 11 | happiness | ~500 | `monthYear` | "2024-03" | `team` | `averageScore` |

**Total Records**: ~4,567 records across all metrics!

**⚠️ Important**: Date field varies by metric!
- Some use `MonthYear` (format: "May 25", "Sep 25")
- Some use `MonthYearSort` (format: `202505`, `202506`, `202509`)
- Some use `monthYear` (format: "2024-03")
- root-cause has NO date field

---

## 🎯 Special Cases Discovered

### 1. **MonthYearSort Field** (New Discovery!)
All metrics have `MonthYearSort` field for easy parsing:

```json
{
  "MonthYear": "May 25",      // Human-readable
  "MonthYearSort": "202505"   // Machine-readable (YYYYMM)
}
```

**Recommendation**: Use `MonthYearSort` instead of `MonthYear` for parsing!

```python
# Easy parsing with MonthYearSort
date = datetime.strptime(record["MonthYearSort"], "%Y%m")

# vs harder parsing with MonthYear
date = datetime.strptime(record["MonthYear"], "%b %y")  # Requires locale
```

### 2. **sp-distribution** (Multi-dimensional)

Each team/month has **multiple records** (one per issue type):

```json
{
  "name": "Automation Testing",
  "MonthYear": "Jun 25",
  "MonthYearSort": "202506",
  "issue_type": "Task",                // 🔥 Additional dimension!
  "total": 0                            // Value field
},
{
  "name": "Automation Testing",
  "MonthYear": "Jun 25",
  "MonthYearSort": "202506",
  "issue_type": "Question",            // 🔥 Different issue type!
  "total": 0
}
```

**Handling**: 
- Each record has both `total` (value) AND `issue_type` (dimension)
- Need to either:
  - **Option A**: Keep all records separately (preserve issue_type breakdown)
  - **Option B**: Aggregate `total` by team/month (sum across all issue types)
  - **Recommended**: Keep separate for detailed analysis

### 3. **root-cause** (No Date Field)
This metric has **no date field** - represents last 6 months aggregate:

```json
{
  "project": "DFN-BS",
  "root_cause": "Missed during testing",
  "total": "3"
}
```

**Handling**: Include all root-cause records (no time filtering).

### 4. **happiness** (Different Format)
Only metric with different date format and field names:

```json
{
  "monthYear": "2024-03",      // YYYY-MM format (different!)
  "team": "Gridsz Data Team",  // "team" field (different!)
  "averageScore": 7.89
}
```

---

## 🔧 Implementation Impact

### Date Parsing Strategy

**Smart Parsing (RECOMMENDED ✅)**

Parse based on which field is available:

```python
def _extract_date_from_record(self, record: Dict, metric_name: str) -> datetime | None:
    # Priority 1: MonthYearSort (easy to parse, some metrics use this)
    # coding-time, open-bugs-over-time, sp-distribution
    if "MonthYearSort" in record and record["MonthYearSort"]:
        try:
            return datetime.strptime(str(record["MonthYearSort"]), "%Y%m")
        except ValueError:
            pass
    
    # Priority 2: MonthYear (most metrics use this)
    # testing-time, review-time, bugs-per-environment, items-out-of-sprint, 
    # defect-rate-prod, defect-rate-all
    if "MonthYear" in record and record["MonthYear"]:
        try:
            return datetime.strptime(record["MonthYear"], "%b %y")
        except ValueError:
            pass
    
    # Priority 3: monthYear (happiness only)
    if "monthYear" in record and record["monthYear"]:
        try:
            return datetime.strptime(record["monthYear"], "%Y-%m")
        except ValueError:
            pass
    
    # Special case: root-cause has no date field
    if metric_name == "root-cause":
        return datetime.now()  # Include all records (last 6 months aggregate)
    
    return None
```

**Key Points**:
- Try `MonthYearSort` first (easier to parse, no locale issues)
- Fallback to `MonthYear` (most common)
- Handle `monthYear` for happiness metric
- Handle root-cause specially (no date)

### Team Field Priority

Update priority order to match actual data:

```python
# OLD (wrong)
for field in ["project_name", "name", "project", "team"]:
```

To:

```python
# NEW (correct) - prioritize by frequency
for field in ["project_name", "name", "project", "team", "team_name"]:
```

**Field Usage by Metric**:
- `project_name`: testing-time, review-time, open-bugs-over-time, items-out-of-sprint, defect-rate-prod, defect-rate-all (6 metrics)
- `name`: coding-time, sp-distribution (2 metrics)
- `project`: root-cause, bugs-per-environment (2 metrics)
- `team`: happiness (1 metric)

**Status**: ✅ Keep current implementation - already handles all cases!

### Expected Record Counts

**Before** (current):
- Only 3 metrics processed
- ~632 total records

**After** (with all metrics + 5-month filter):
- All 11 metrics processed
- Estimated ~1,500-2,000 records (after 5-month filtering)

---

## ✅ Updated Implementation Checklist

- [ ] **Smart date parsing**: Try `MonthYearSort` first, fallback to `MonthYear`, handle `monthYear`
- [ ] **Team field priority**: `project_name` → `name` → `project` → `team` (already correct!)
- [ ] **Handle root-cause specially**: No date filtering, include all records (last 6 months aggregate)
- [ ] **Handle sp-distribution multi-dimensional data**: Keep all issue_type records (preserve dimension)
- [ ] **Parse three date formats**: 
  - `MonthYearSort`: `"202505"` → `datetime.strptime(value, "%Y%m")`
  - `MonthYear`: `"May 25"` → `datetime.strptime(value, "%b %y")`
  - `monthYear`: `"2024-03"` → `datetime.strptime(value, "%Y-%m")`
- [ ] **Test with all 11 metrics** (not just 3)
- [ ] **Verify ~1,500+ records** after filtering (vs current ~98)

---

## 📝 Next Steps

1. ✅ **Design Document Updated**: `METRIC_PARSING_DESIGN.md`
2. ⏳ **Implement Changes**: Update `main.py` with new logic
3. ⏳ **Test with Real Data**: Run with "Gridsz Data Team" filter
4. ⏳ **Verify Record Counts**: Should see 11 metrics processed
5. ⏳ **Update Documentation**: Reflect new understanding

---

**Key Insight**: Using `MonthYearSort` instead of `MonthYear` will make parsing **much simpler** and avoid locale issues!

---

**Updated**: October 31, 2025  
**Status**: Ready for implementation with corrected data structure understanding

# Unified LLM Agent Evaluation Pipeline

## Overview

A production-ready evaluation framework that unifies trace-based analysis with best-in-class open-source evaluation tools (RAGAS, DeepEval) and custom heuristics. Designed to measure every meaningful dimension of LLM agent behavior and provide actionable insights for iterative improvement.

**Target Users:** LLM engineers, AI ops teams, and researchers building and optimizing multi-tool agents.



## Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  LLM AGENT (AutoGen / LangChain / Custom)                   │
│  • RAG retrieval  • Python/code execution                   │
│  • Tool orchestration  • Multi-step reasoning               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  TRACE LOGGING SYSTEM (Real-time Capture)                   │
│  • Execution steps  • Tool invocations                       │
│  • Token usage      • Latency per step                       │
│  • Structured JSONL/SQLite storage                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  EVALUATION PIPELINE (Multi-Framework)                       │
│  ┌──────────────────┬──────────────┬─────────────────────┐  │
│  │ RAGAS Grounding  │ DeepEval QA  │ Custom Heuristics   │  │
│  │ • Faithfulness   │ • Correctness│ • Tool efficiency   │  │
│  │ • Context rel.   │ • Hallucin.  │ • Redundancy detect │  │
│  └──────────────────┴──────────────┴─────────────────────┘  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Cost & Latency Analysis  (Per-run breakdown)           │  │
│  └────────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  UNIFIED EVALUATION LAYER                                    │
│  • Aggregates all metrics into one trace evaluation         │
│  • Runs full pipeline: evaluate_trace(trace, ground_truth)  │
│  • Returns EvalResult with 12+ metrics                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  RELIABILITY SCORING & GRADING                               │
│  • Composite score combining all dimensions                 │
│  • Weighted metric synthesis (configurable)                 │
│  • Letter grades (A–F) for easy interpretation              │
│  • Per-dimension breakdowns for diagnosis                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  CROSS-RUN ANALYSIS & REPORTING                              │
│  • Multi-run leaderboards  • Statistical comparison         │
│  • Trend analysis          • JSON/formatted export          │
└─────────────────────────────────────────────────────────────┘
```


## Key Features

### 1. **Comprehensive Trace Logging**
- Capture every step: reasoning, tool calls, observations, final response
- Per-step token usage and latency (wall-clock milliseconds)
- Structured metadata support for custom fields
- Built-in `Timer` context manager for easy instrumentation

### 2. **Multi-Framework Evaluation**
- **RAGAS-style:** Faithfulness, context relevance, and grounding scores
- **DeepEval-style:** Correctness (vs. ground truth), answer relevance, hallucination risk
- **Custom Heuristics:** Tool efficiency, redundancy detection, cost tracking
- All in-process with no external API calls required

### 3. **Unified Evaluation API**
```python
result = evaluate_trace(
    trace=my_trace,
    ground_truth="Reference answer",
    retrieved_contexts=["Context chunk 1", "Context chunk 2"]
)
```
Single function call returns 12+ metrics across all dimensions.

### 4. **Reliability Scoring System**
- Composite score (0–1) from weighted combination:
  - Correctness (30%)
  - Grounding (25%)
  - Tool Efficiency (20%)
  - Hallucination Risk (15%)
  - Cost Efficiency (5%)
  - Latency Efficiency (5%)
- Configurable weights per use case
- Letter grades for stakeholder communication

### 5. **Cross-Run Comparison**
- Leaderboard ranking across multiple agent runs
- Per-metric aggregation: mean, std dev, best, worst
- Filter by task, top-k selection, multi-run analysis
- Export to JSON for downstream dashboards

### 6. **Production Storage**
- **JSONL:** Line-delimited JSON for streaming and version control
- **SQLite:** Indexed relational storage for querying
- **JSON Export:** Standard format for integration with other tools


## Installation & Setup

### Prerequisites
```bash
pip install -r requirements.txt
```

### Quick Start

```python
from utility import (
    create_trace, log_step, log_tool_call, finalise_trace,
    evaluate_trace, format_eval_summary
)

# 1. Create and populate a trace
trace = create_trace(task_id="my_task", task_input="What is 2+2?")

# Log reasoning steps
log_step(trace, "reasoning", "I need to compute 2+2", 
         input_tokens=50, output_tokens=20, latency_ms=100)

# Log tool calls
log_tool_call(trace, 
              tool_name="calculator",
              tool_input={"expr": "2+2"},
              tool_output=4,
              latency_ms=10,
              success=True)

# Finalize with agent's response
trace = finalise_trace(trace, final_response="2+2 equals 4")

# 2. Evaluate the trace
result = evaluate_trace(
    trace=trace,
    ground_truth="The sum of 2 and 2 is 4",
    retrieved_contexts=[]
)

# 3. View results
print(format_eval_summary(result))
```


## Metric Reference

### Grounding Metrics (RAGAS)
| Metric | Range | Definition | Notes |
|--------|-------|-----------|-------|
| **Faithfulness** | [0, 1] | Fraction of answer claims attributable to retrieved contexts | Higher = more grounded |
| **Context Relevance** | [0, 1] | Jaccard similarity between query and contexts | Higher = better match |
| **Grounding Score** | [0, 1] | Equal-weighted average of faithfulness + context relevance | Combined grounding measure |

### Correctness Metrics (DeepEval)
| Metric | Range | Definition | Notes |
|--------|-------|-----------|-------|
| **Correctness** | [0, 1] | Token F1 or exact match vs. ground truth | Requires reference answer |
| **Answer Relevance** | [0, 1] | Query keyword coverage in answer | Lightweight relevance check |
| **Hallucination Risk** | [0, 1] | Inverse faithfulness, scaled by correctness | Higher = more risky |

### Efficiency Metrics (Custom)
| Metric | Range | Definition | Notes |
|--------|-------|-----------|-------|
| **Tool Efficiency** | [0, 1] | Unique successful tools / total calls | 1.0 = no redundancy |
| **Redundancy Ratio** | [0, 1] | Duplicate calls / total calls | 0.0 = no duplicates |
| **Cost (USD)** | [0, ∞] | Actual monetary cost | Based on token rates |
| **Cost Score** | [0, 1] | 1 − (cost / budget) | 1.0 = free, 0.0 = over budget |
| **Latency (ms)** | [0, ∞] | Total wall-clock execution time | Sum of all steps |
| **Latency Score** | [0, 1] | 1 − (latency / threshold) | 1.0 = instant, 0.0 = at threshold |

### Composite Metric
| Metric | Range | Formula |
|--------|-------|---------|
| **Reliability Score** | [0, 1] | Weighted sum of all dimensions | See weights above |



## API Reference

### Trace Logging

#### `create_trace(task_id, task_input, run_id=None) → TraceRecord`
Initialize a new trace for an agent execution.

#### `log_step(trace, step_type, content, input_tokens=0, output_tokens=0, latency_ms=0, metadata=None) → dict`
Append a reasoning or intermediate step to the trace.

#### `log_tool_call(trace, tool_name, tool_input, tool_output, latency_ms=0, success=True) → dict`
Record a tool invocation with inputs, outputs, and timing.

#### `finalise_trace(trace, final_response) → TraceRecord`
Close the trace with the agent's final response.

### Evaluation

#### `evaluate_trace(trace, ground_truth, retrieved_contexts=None) → EvalResult`
Run the full evaluation pipeline on a completed trace. **Primary entry point.**

#### `compute_reliability_score(result, weights=None) → float`
Compute composite score from all dimensions (automatically called by `evaluate_trace`).

#### `reliability_grade(score) → str`
Convert numeric score to letter grade: "A" (≥0.90), "B" (≥0.75), "C" (≥0.60), "D" (≥0.45), "F" (<0.45).

### Cross-Run Comparison

#### `compare_runs(results) → dict`
Aggregate metrics across multiple `EvalResult` objects. Returns: mean, std dev, best, worst per metric + leaderboard.

#### `filter_runs_by_task(results, task_id) → list[EvalResult]`
Select all results for a specific task.

#### `top_k_runs(results, k=3) → list[EvalResult]`
Return top-k runs by reliability score.

### Persistence

#### `save_trace_jsonl(trace, path="traces.jsonl")`
Append trace to JSONL file.

#### `load_traces_jsonl(path="traces.jsonl") → list[dict]`
Load all traces from JSONL file.

#### `init_sqlite(db_path="traces.db") → sqlite3.Connection`
Initialize SQLite database with `traces` and `eval_results` tables.

#### `save_trace_sqlite(trace, conn)`
Upsert trace into SQLite.

#### `save_eval_result_sqlite(result, conn)`
Upsert evaluation result into SQLite.

#### `export_results_json(results, path="eval_results.json")`
Export list of `EvalResult` objects to JSON.

### Reporting

#### `format_eval_summary(result) → str`
Return formatted multi-line summary of one evaluation result.


## Configuration

### Weights (Customize Reliability Scoring)
Modify `RELIABILITY_WEIGHTS` in `utility.py`:

```python
RELIABILITY_WEIGHTS = {
    "correctness":     0.30,   # Prioritize answer quality
    "grounding":       0.25,   # Ensure factual grounding
    "tool_efficiency": 0.20,   # Minimize wasted tool calls
    "hallucination":   0.15,   # Penalize unfounded claims
    "cost_score":      0.05,   # Secondary cost constraint
    "latency_score":   0.05,   # Secondary latency constraint
}
```

### Cost Rates
Update `COST_PER_INPUT_TOKEN` and `COST_PER_OUTPUT_TOKEN` to match your model pricing:

```python
# Example: Claude Sonnet pricing (as of 2024-10)
COST_PER_INPUT_TOKEN  = 3e-6   # $3 per 1M input tokens
COST_PER_OUTPUT_TOKEN = 15e-6  # $15 per 1M output tokens
```

### Cost & Latency Thresholds
In evaluation functions, adjust budget and latency thresholds:

```python
cost_score    = cost_to_score(cost_usd, budget_usd=0.05)      # $0.05 budget
latency_score = latency_to_score(latency_ms, threshold_ms=30_000)  # 30s threshold
```



## Workflow Examples

### Example 1: Single-Task Single-Run Evaluation

```python
# 1. Create and execute agent trace
trace = create_trace(task_id="revenue_query", 
                     task_input="Total Q3 revenue?")
log_step(trace, "reasoning", "Retrieving Q3 data...", 
         input_tokens=100, output_tokens=50, latency_ms=500)
log_tool_call(trace, "rag_retrieval", {"query": "Q3 revenue"},
              ["Q3 revenue was $12.7M"], latency_ms=300)
log_tool_call(trace, "python_calc", {"expr": "sum([...])"},
              12.7, latency_ms=100)
trace = finalise_trace(trace, "Q3 revenue totaled $12.7M")

# 2. Evaluate
ground_truth = "Total Q3 revenue: $12.7 million"
contexts = ["Q3 revenue breakdown: Product A $4.2M, Product B $3.1M, Product C $5.4M"]
result = evaluate_trace(trace, ground_truth, contexts)

# 3. Report
print(format_eval_summary(result))
# Outputs:
# ═════════════════════════════════════════════════════
#   Run ID  : abc12345…
#   Task    : revenue_query
# ═════════════════════════════════════════════════════
#   Correctness       : 0.950
#   Grounding         : 0.880
#     Faithfulness    : 0.850
#     Context Rel.    : 0.910
#   Tool Efficiency   : 1.000
#   Redundancy Ratio  : 0.000
#   Hallucination Risk: 0.150
#   Cost (USD)        : $0.000025
#   Latency (ms)      : 900.0
# ─────────────────────────────────────────────────────
#   RELIABILITY SCORE : 0.8964  [A]
# ═════════════════════════════════════════════════════
```

### Example 2: Multi-Run Comparison & Leaderboard

```python
# Collect results from multiple agent runs
results = [
    evaluate_trace(trace1, ground_truth, contexts),
    evaluate_trace(trace2, ground_truth, contexts),
    evaluate_trace(trace3, ground_truth, contexts),
]

# Compare
comparison = compare_runs(results)

# Metrics summary
for metric, stats in comparison["metrics"].items():
    print(f"{metric:20s}: mean={stats['mean']:.4f}, "
          f"std={stats['std']:.4f}, "
          f"best={stats['best']:.4f}, worst={stats['worst']:.4f}")

# Leaderboard
for entry in comparison["leaderboard"]:
    print(f"  {entry['rank']}. {entry['run_id'][:8]}  "
          f"→  {entry['reliability_score']:.4f}  [{entry['grade']}]")
```

### Example 3: Task-Specific Analysis

```python
# Filter results by task
task_results = filter_runs_by_task(all_results, task_id="revenue_query")

# Get top-3 runs for this task
top3 = top_k_runs(task_results, k=3)

# Export for reporting
export_results_json(top3, path="revenue_query_top3.json")
```

### Example 4: Persistence to Database

```python
# SQLite workflow
db_path = "/data/evals.db"
conn = init_sqlite(db_path)

# Save all traces and results
for trace in [trace1, trace2, trace3]:
    save_trace_sqlite(trace, conn)
    result = evaluate_trace(trace, ground_truth, contexts)
    save_eval_result_sqlite(result, conn)

# Later: reload and analyze
loaded_traces = load_traces_sqlite(conn, task_id="revenue_query")
```


## Advanced Topics

### Custom Metric Weights

Adjust weights to reflect your priorities:

```python
custom_weights = {
    "correctness":     0.40,   # Maximize accuracy (was 0.30)
    "grounding":       0.20,   # De-emphasize grounding (was 0.25)
    "tool_efficiency": 0.15,   # Reduce tool optimization (was 0.20)
    "hallucination":   0.20,   # Increase hallucination penalty (was 0.15)
    "cost_score":      0.03,
    "latency_score":   0.02,
}

custom_score = compute_reliability_score(result, weights=custom_weights)
```

### Handling Missing Ground Truth

If ground truth is unavailable, pass an empty string to `evaluate_trace`:

```python
result = evaluate_trace(
    trace=trace,
    ground_truth="",  # Will score 0 on correctness
    retrieved_contexts=contexts
)
# Still get grounding, tool efficiency, and cost/latency metrics
```

### Custom Context Extraction

Automatically extract contexts from trace steps:

```python
# Extract RAG outputs from trace
rag_outputs = [
    step["content"] 
    for step in trace.steps 
    if "rag" in step.get("metadata", {}).get("source", "").lower()
]

result = evaluate_trace(trace, ground_truth, retrieved_contexts=rag_outputs)
```
Executor



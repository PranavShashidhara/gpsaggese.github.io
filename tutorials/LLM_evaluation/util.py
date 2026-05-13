"""
utility.py
==========
Unified Evaluation Pipeline — Utility Functions
Covers: trace logging, RAGAS grounding, DeepEval correctness,
custom heuristics, cost/latency tracking, and composite reliability scoring.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
import hashlib
import statistics
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# 0.  TYPE ALIASES & CONSTANTS
# ---------------------------------------------------------------------------

TraceStep   = dict[str, Any]
ToolCall    = dict[str, Any]
MetricDict  = dict[str, float]

COST_PER_INPUT_TOKEN  = 3e-6   # $ per token (placeholder — override per model)
COST_PER_OUTPUT_TOKEN = 15e-6
RELIABILITY_WEIGHTS   = {
    "correctness":    0.30,
    "grounding":      0.25,
    "tool_efficiency":0.20,
    "hallucination":  0.15,
    "cost_score":     0.05,
    "latency_score":  0.05,
}


# ---------------------------------------------------------------------------
# 1.  DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class TraceRecord:
    """One fully-populated execution trace for a single agent task."""
    run_id:           str
    task_id:          str
    task_input:       str
    final_response:   str
    steps:            list[TraceStep]   = field(default_factory=list)
    tool_calls:       list[ToolCall]    = field(default_factory=list)
    total_input_tokens:  int            = 0
    total_output_tokens: int            = 0
    total_latency_ms:    float          = 0.0
    metadata:         dict[str, Any]    = field(default_factory=dict)
    created_at:       str               = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class EvalResult:
    """Aggregated evaluation result for one trace."""
    run_id:           str
    task_id:          str
    correctness:      float = 0.0
    grounding:        float = 0.0
    faithfulness:     float = 0.0
    context_relevance:float = 0.0
    tool_efficiency:  float = 0.0
    redundancy_ratio: float = 0.0
    hallucination_risk:float = 0.0
    cost_usd:         float = 0.0
    cost_score:       float = 0.0
    latency_ms:       float = 0.0
    latency_score:    float = 0.0
    reliability_score:float = 0.0
    raw_metrics:      dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 2.  TRACE LOGGING
# ---------------------------------------------------------------------------

def generate_run_id() -> str:
    """Return a unique run identifier (UUID4-based)."""
    return str(uuid.uuid4())


def create_trace(task_id: str, task_input: str, run_id: Optional[str] = None) -> TraceRecord:
    """
    Initialise an empty TraceRecord for a new agent run.

    Parameters
    ----------
    task_id    : logical task name / identifier
    task_input : raw prompt or question posed to the agent
    run_id     : optional; auto-generated if omitted

    Returns
    -------
    TraceRecord
    """
    return TraceRecord(
        run_id=run_id or generate_run_id(),
        task_id=task_id,
        task_input=task_input,
        final_response="",
        steps=[],
        tool_calls=[],
    )


def log_step(
    trace: TraceRecord,
    step_type: str,
    content: Any,
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: float = 0.0,
    metadata: Optional[dict] = None,
) -> TraceStep:
    """
    Append one reasoning / tool step to an in-progress trace.

    Parameters
    ----------
    trace        : the TraceRecord being built
    step_type    : e.g. "reasoning", "tool_call", "observation", "final"
    content      : arbitrary step payload (string, dict, …)
    input_tokens : tokens consumed in this step
    output_tokens: tokens generated in this step
    latency_ms   : wall-clock time for this step
    metadata     : optional extra fields

    Returns
    -------
    The step dict that was appended.
    """
    step: TraceStep = {
        "step_index":    len(trace.steps),
        "step_type":     step_type,
        "content":       content,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "latency_ms":    latency_ms,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "metadata":      metadata or {},
    }
    trace.steps.append(step)
    trace.total_input_tokens  += input_tokens
    trace.total_output_tokens += output_tokens
    trace.total_latency_ms    += latency_ms
    return step


def log_tool_call(
    trace: TraceRecord,
    tool_name: str,
    tool_input: Any,
    tool_output: Any,
    latency_ms: float = 0.0,
    success: bool = True,
) -> ToolCall:
    """
    Record a single tool invocation within a trace.

    Parameters
    ----------
    trace       : the active TraceRecord
    tool_name   : name of the tool (e.g. "rag_retrieval", "python_exec")
    tool_input  : inputs passed to the tool
    tool_output : output returned by the tool
    latency_ms  : execution time of the tool call
    success     : whether the tool call completed without error

    Returns
    -------
    The tool-call dict that was appended.
    """
    call: ToolCall = {
        "call_index": len(trace.tool_calls),
        "tool_name":  tool_name,
        "input":      tool_input,
        "output":     tool_output,
        "latency_ms": latency_ms,
        "success":    success,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }
    trace.tool_calls.append(call)
    log_step(
        trace,
        step_type="tool_call",
        content={"tool": tool_name, "success": success},
        latency_ms=latency_ms,
    )
    return call


def finalise_trace(trace: TraceRecord, final_response: str) -> TraceRecord:
    """
    Mark an agent run as complete by attaching the final response.

    Parameters
    ----------
    trace          : the TraceRecord to close
    final_response : agent's final answer string

    Returns
    -------
    The mutated TraceRecord.
    """
    trace.final_response = final_response
    log_step(trace, step_type="final", content=final_response)
    return trace


# ---------------------------------------------------------------------------
# 3.  PERSISTENCE  (JSONL + SQLite)
# ---------------------------------------------------------------------------

def save_trace_jsonl(trace: TraceRecord, path: str = "traces.jsonl") -> None:
    """
    Append a TraceRecord to a JSONL file (one JSON object per line).

    Parameters
    ----------
    trace : completed TraceRecord
    path  : file path for the JSONL store
    """
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(trace), ensure_ascii=False) + "\n")


def load_traces_jsonl(path: str = "traces.jsonl") -> list[dict]:
    """
    Load all traces from a JSONL file.

    Parameters
    ----------
    path : file path for the JSONL store

    Returns
    -------
    List of raw trace dicts.
    """
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def init_sqlite(db_path: str = "traces.db") -> sqlite3.Connection:
    """
    Initialise (or open) a SQLite database for trace storage.

    Parameters
    ----------
    db_path : filesystem path for the SQLite file

    Returns
    -------
    sqlite3.Connection
    """
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            run_id      TEXT PRIMARY KEY,
            task_id     TEXT,
            task_input  TEXT,
            final_response TEXT,
            payload     TEXT,
            created_at  TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eval_results (
            run_id            TEXT PRIMARY KEY,
            task_id           TEXT,
            correctness       REAL,
            grounding         REAL,
            faithfulness      REAL,
            context_relevance REAL,
            tool_efficiency   REAL,
            redundancy_ratio  REAL,
            hallucination_risk REAL,
            cost_usd          REAL,
            cost_score        REAL,
            latency_ms        REAL,
            latency_score     REAL,
            reliability_score REAL,
            raw_metrics       TEXT,
            created_at        TEXT
        )
    """)
    conn.commit()
    return conn


def save_trace_sqlite(trace: TraceRecord, conn: sqlite3.Connection) -> None:
    """
    Upsert a TraceRecord into SQLite.

    Parameters
    ----------
    trace : completed TraceRecord
    conn  : open SQLite connection (from init_sqlite)
    """
    payload = json.dumps(asdict(trace), ensure_ascii=False)
    conn.execute("""
        INSERT OR REPLACE INTO traces
            (run_id, task_id, task_input, final_response, payload, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        trace.run_id, trace.task_id, trace.task_input,
        trace.final_response, payload, trace.created_at,
    ))
    conn.commit()


def load_traces_sqlite(
    conn: sqlite3.Connection,
    task_id: Optional[str] = None,
) -> list[dict]:
    """
    Query traces from SQLite, optionally filtered by task_id.

    Parameters
    ----------
    conn    : open SQLite connection
    task_id : optional filter

    Returns
    -------
    List of raw trace dicts.
    """
    if task_id:
        rows = conn.execute(
            "SELECT payload FROM traces WHERE task_id = ?", (task_id,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT payload FROM traces").fetchall()
    return [json.loads(r[0]) for r in rows]


def save_eval_result_sqlite(result: EvalResult, conn: sqlite3.Connection) -> None:
    """
    Upsert an EvalResult into SQLite.

    Parameters
    ----------
    result : populated EvalResult
    conn   : open SQLite connection
    """
    conn.execute("""
        INSERT OR REPLACE INTO eval_results
            (run_id, task_id, correctness, grounding, faithfulness,
             context_relevance, tool_efficiency, redundancy_ratio,
             hallucination_risk, cost_usd, cost_score, latency_ms,
             latency_score, reliability_score, raw_metrics, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        result.run_id, result.task_id, result.correctness, result.grounding,
        result.faithfulness, result.context_relevance, result.tool_efficiency,
        result.redundancy_ratio, result.hallucination_risk, result.cost_usd,
        result.cost_score, result.latency_ms, result.latency_score,
        result.reliability_score, json.dumps(result.raw_metrics),
        datetime.now(timezone.utc).isoformat(),
    ))
    conn.commit()


# ---------------------------------------------------------------------------
# 4.  RAGAS-STYLE GROUNDING METRICS
# ---------------------------------------------------------------------------

def compute_faithfulness(
    answer: str,
    retrieved_contexts: list[str],
) -> float:
    """
    Heuristic faithfulness score: fraction of answer sentences that can be
    attributed to at least one retrieved context chunk.

    Parameters
    ----------
    answer             : agent final response
    retrieved_contexts : list of context strings from RAG retrieval

    Returns
    -------
    float in [0, 1]
    """
    if not answer or not retrieved_contexts:
        return 0.0
    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not sentences:
        return 0.0
    context_blob = " ".join(retrieved_contexts).lower()
    attributed = sum(
        1 for s in sentences
        if any(word in context_blob for word in s.lower().split() if len(word) > 4)
    )
    return round(attributed / len(sentences), 4)


def compute_context_relevance(
    query: str,
    retrieved_contexts: list[str],
) -> float:
    """
    Heuristic context relevance: average Jaccard similarity between the query
    token set and each retrieved context token set.

    Parameters
    ----------
    query              : original user question
    retrieved_contexts : list of context strings from RAG retrieval

    Returns
    -------
    float in [0, 1]
    """
    if not query or not retrieved_contexts:
        return 0.0
    query_tokens = set(query.lower().split())

    def jaccard(a: set, b: set) -> float:
        return len(a & b) / len(a | b) if (a | b) else 0.0

    scores = [
        jaccard(query_tokens, set(ctx.lower().split()))
        for ctx in retrieved_contexts
    ]
    return round(statistics.mean(scores), 4)


def compute_grounding_score(faithfulness: float, context_relevance: float) -> float:
    """
    Combined grounding score (equal-weighted average of faithfulness
    and context relevance).

    Parameters
    ----------
    faithfulness      : output of compute_faithfulness
    context_relevance : output of compute_context_relevance

    Returns
    -------
    float in [0, 1]
    """
    return round((faithfulness + context_relevance) / 2, 4)


def run_ragas_evaluation(
    query: str,
    answer: str,
    retrieved_contexts: list[str],
) -> MetricDict:
    """
    Run all RAGAS-style grounding metrics for one query/answer pair.

    Parameters
    ----------
    query              : user question
    answer             : agent answer
    retrieved_contexts : list of RAG context strings

    Returns
    -------
    Dict with keys: faithfulness, context_relevance, grounding_score
    """
    faith = compute_faithfulness(answer, retrieved_contexts)
    ctx_rel = compute_context_relevance(query, retrieved_contexts)
    return {
        "faithfulness":      faith,
        "context_relevance": ctx_rel,
        "grounding_score":   compute_grounding_score(faith, ctx_rel),
    }


# ---------------------------------------------------------------------------
# 5.  DEEPEVAL-STYLE CORRECTNESS & QUALITY METRICS
# ---------------------------------------------------------------------------

def compute_answer_relevance(answer: str, query: str) -> float:
    """
    Lightweight answer relevance score based on query-keyword coverage.

    Parameters
    ----------
    answer : agent response
    query  : original user question

    Returns
    -------
    float in [0, 1]
    """
    if not answer or not query:
        return 0.0
    query_keywords = {w.lower() for w in query.split() if len(w) > 3}
    if not query_keywords:
        return 1.0
    hits = sum(1 for kw in query_keywords if kw in answer.lower())
    return round(hits / len(query_keywords), 4)


def compute_correctness(
    answer: str,
    ground_truth: str,
    method: str = "token_f1",
) -> float:
    """
    Estimate answer correctness vs. a ground-truth string.

    Parameters
    ----------
    answer       : agent final response
    ground_truth : reference answer
    method       : "token_f1" (default) or "exact_match"

    Returns
    -------
    float in [0, 1]
    """
    if not answer or not ground_truth:
        return 0.0
    if method == "exact_match":
        return float(answer.strip().lower() == ground_truth.strip().lower())

    # Token F1
    pred_tokens  = set(answer.lower().split())
    truth_tokens = set(ground_truth.lower().split())
    if not pred_tokens or not truth_tokens:
        return 0.0
    common   = pred_tokens & truth_tokens
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(truth_tokens)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def compute_hallucination_risk(
    answer: str,
    retrieved_contexts: list[str],
    ground_truth: Optional[str] = None,
) -> float:
    """
    Hallucination risk = 1 − faithfulness (lower faithfulness → higher risk).
    Optionally weighted by correctness if ground_truth is provided.

    Parameters
    ----------
    answer             : agent final response
    retrieved_contexts : context chunks available to the agent
    ground_truth       : optional reference answer

    Returns
    -------
    float in [0, 1]  (higher = more risk)
    """
    faith = compute_faithfulness(answer, retrieved_contexts)
    base_risk = 1.0 - faith
    if ground_truth:
        correctness = compute_correctness(answer, ground_truth)
        # Scale down risk if answer is actually correct
        base_risk = base_risk * (1.0 - 0.5 * correctness)
    return round(base_risk, 4)


def run_deepeval_evaluation(
    query: str,
    answer: str,
    ground_truth: str,
    retrieved_contexts: Optional[list[str]] = None,
) -> MetricDict:
    """
    Run all DeepEval-style quality metrics for one sample.

    Parameters
    ----------
    query              : user question
    answer             : agent answer
    ground_truth       : reference answer
    retrieved_contexts : optional context for hallucination check

    Returns
    -------
    Dict with keys: correctness, answer_relevance, hallucination_risk
    """
    contexts = retrieved_contexts or []
    return {
        "correctness":       compute_correctness(answer, ground_truth),
        "answer_relevance":  compute_answer_relevance(answer, query),
        "hallucination_risk":compute_hallucination_risk(answer, contexts, ground_truth),
    }


# ---------------------------------------------------------------------------
# 6.  CUSTOM HEURISTICS — TOOL EFFICIENCY & REDUNDANCY
# ---------------------------------------------------------------------------

def compute_tool_efficiency(tool_calls: list[ToolCall]) -> float:
    """
    Tool efficiency = successful unique tool types / total tool calls.
    A perfect score means every call was to a distinct, successful tool.

    Parameters
    ----------
    tool_calls : list of tool-call dicts (from TraceRecord.tool_calls)

    Returns
    -------
    float in [0, 1]
    """
    if not tool_calls:
        return 1.0   # no tools needed → trivially efficient
    successful = [tc for tc in tool_calls if tc.get("success", True)]
    if not successful:
        return 0.0
    unique_tools = {tc["tool_name"] for tc in successful}
    return round(len(unique_tools) / len(tool_calls), 4)


def compute_redundancy_ratio(tool_calls: list[ToolCall]) -> float:
    """
    Redundancy ratio = fraction of tool calls that are exact duplicates
    (same tool name + identical input fingerprint).

    Parameters
    ----------
    tool_calls : list of tool-call dicts

    Returns
    -------
    float in [0, 1]  (0 = no redundancy, 1 = all calls are redundant)
    """
    if len(tool_calls) < 2:
        return 0.0

    def fingerprint(tc: ToolCall) -> str:
        key = f"{tc['tool_name']}:{json.dumps(tc.get('input', ''), sort_keys=True)}"
        return hashlib.md5(key.encode()).hexdigest()

    seen: set[str] = set()
    redundant = 0
    for tc in tool_calls:
        fp = fingerprint(tc)
        if fp in seen:
            redundant += 1
        seen.add(fp)
    return round(redundant / len(tool_calls), 4)


def detect_redundant_calls(tool_calls: list[ToolCall]) -> list[int]:
    """
    Return the indices of redundant tool calls (duplicate fingerprints).

    Parameters
    ----------
    tool_calls : list of tool-call dicts

    Returns
    -------
    List of call_index values for duplicate calls.
    """
    def fingerprint(tc: ToolCall) -> str:
        key = f"{tc['tool_name']}:{json.dumps(tc.get('input', ''), sort_keys=True)}"
        return hashlib.md5(key.encode()).hexdigest()

    seen: set[str] = set()
    redundant_indices: list[int] = []
    for tc in tool_calls:
        fp = fingerprint(tc)
        if fp in seen:
            redundant_indices.append(tc.get("call_index", -1))
        seen.add(fp)
    return redundant_indices


def run_custom_heuristics(tool_calls: list[ToolCall]) -> MetricDict:
    """
    Run all custom heuristic metrics for one trace.

    Parameters
    ----------
    tool_calls : list of tool-call dicts from a TraceRecord

    Returns
    -------
    Dict with keys: tool_efficiency, redundancy_ratio, redundant_call_indices
    """
    return {
        "tool_efficiency":        compute_tool_efficiency(tool_calls),
        "redundancy_ratio":       compute_redundancy_ratio(tool_calls),
        "redundant_call_indices": detect_redundant_calls(tool_calls),
    }


# ---------------------------------------------------------------------------
# 7.  COST & LATENCY TRACKING
# ---------------------------------------------------------------------------

def compute_cost(
    input_tokens: int,
    output_tokens: int,
    cost_per_input:  float = COST_PER_INPUT_TOKEN,
    cost_per_output: float = COST_PER_OUTPUT_TOKEN,
) -> float:
    """
    Compute monetary cost for one agent run in USD.

    Parameters
    ----------
    input_tokens    : total prompt tokens consumed
    output_tokens   : total completion tokens generated
    cost_per_input  : $/token for input  (default: Claude Sonnet rate)
    cost_per_output : $/token for output (default: Claude Sonnet rate)

    Returns
    -------
    float  (USD)
    """
    return round(
        input_tokens * cost_per_input + output_tokens * cost_per_output, 6
    )


def cost_to_score(cost_usd: float, budget_usd: float = 0.05) -> float:
    """
    Normalise cost to a [0, 1] score (1 = free, 0 = at or beyond budget).

    Parameters
    ----------
    cost_usd   : actual cost in USD
    budget_usd : upper bound considered "unacceptable"

    Returns
    -------
    float in [0, 1]
    """
    if budget_usd <= 0:
        return 0.0
    return round(max(0.0, 1.0 - cost_usd / budget_usd), 4)


def compute_per_step_latency(steps: list[TraceStep]) -> list[float]:
    """
    Extract per-step latency values from a trace's step list.

    Parameters
    ----------
    steps : list of step dicts from TraceRecord.steps

    Returns
    -------
    List of latency_ms floats in step order.
    """
    return [s.get("latency_ms", 0.0) for s in steps]


def latency_to_score(latency_ms: float, threshold_ms: float = 30_000.0) -> float:
    """
    Normalise total latency to a [0, 1] score (1 = instant, 0 = at threshold).

    Parameters
    ----------
    latency_ms   : total wall-clock time in milliseconds
    threshold_ms : upper bound considered "unacceptable" (default 30 s)

    Returns
    -------
    float in [0, 1]
    """
    if threshold_ms <= 0:
        return 0.0
    return round(max(0.0, 1.0 - latency_ms / threshold_ms), 4)


def run_cost_latency_analysis(trace: TraceRecord) -> MetricDict:
    """
    Compute cost and latency metrics for one trace.

    Parameters
    ----------
    trace : completed TraceRecord

    Returns
    -------
    Dict with keys: cost_usd, cost_score, total_latency_ms,
                    latency_score, per_step_latency_ms
    """
    cost = compute_cost(trace.total_input_tokens, trace.total_output_tokens)
    return {
        "cost_usd":             cost,
        "cost_score":           cost_to_score(cost),
        "total_latency_ms":     trace.total_latency_ms,
        "latency_score":        latency_to_score(trace.total_latency_ms),
        "per_step_latency_ms":  compute_per_step_latency(trace.steps),
    }


# ---------------------------------------------------------------------------
# 8.  UNIFIED EVALUATION LAYER
# ---------------------------------------------------------------------------

def evaluate_trace(
    trace: TraceRecord,
    ground_truth: str,
    retrieved_contexts: Optional[list[str]] = None,
) -> EvalResult:
    """
    Run the full evaluation pipeline on a single completed trace.

    Internally calls:
      - run_ragas_evaluation
      - run_deepeval_evaluation
      - run_custom_heuristics
      - run_cost_latency_analysis
      - compute_reliability_score

    Parameters
    ----------
    trace              : completed TraceRecord
    ground_truth       : reference answer for this task
    retrieved_contexts : RAG context chunks (if any)

    Returns
    -------
    EvalResult
    """
    contexts = retrieved_contexts or []

    ragas   = run_ragas_evaluation(trace.task_input, trace.final_response, contexts)
    deepev  = run_deepeval_evaluation(
        trace.task_input, trace.final_response, ground_truth, contexts
    )
    heur    = run_custom_heuristics(trace.tool_calls)
    cl      = run_cost_latency_analysis(trace)

    raw = {**ragas, **deepev, **heur, **cl}

    result = EvalResult(
        run_id=trace.run_id,
        task_id=trace.task_id,
        correctness=deepev["correctness"],
        grounding=ragas["grounding_score"],
        faithfulness=ragas["faithfulness"],
        context_relevance=ragas["context_relevance"],
        tool_efficiency=heur["tool_efficiency"],
        redundancy_ratio=heur["redundancy_ratio"],
        hallucination_risk=deepev["hallucination_risk"],
        cost_usd=cl["cost_usd"],
        cost_score=cl["cost_score"],
        latency_ms=cl["total_latency_ms"],
        latency_score=cl["latency_score"],
        raw_metrics=raw,
    )
    result.reliability_score = compute_reliability_score(result)
    return result


# ---------------------------------------------------------------------------
# 9.  RELIABILITY SCORING SYSTEM
# ---------------------------------------------------------------------------

def compute_reliability_score(
    result: EvalResult,
    weights: Optional[dict[str, float]] = None,
) -> float:
    """
    Compute a composite reliability score in [0, 1].

    Default weights (overridable via `weights` argument):
        correctness     : 0.30
        grounding       : 0.25
        tool_efficiency : 0.20
        hallucination   : 0.15   (inverted: 1 − risk)
        cost_score      : 0.05
        latency_score   : 0.05

    Parameters
    ----------
    result  : populated EvalResult (reliability_score field is ignored)
    weights : optional weight dict (must sum to 1.0)

    Returns
    -------
    float in [0, 1]
    """
    w = weights or RELIABILITY_WEIGHTS
    score = (
        w["correctness"]     * result.correctness
        + w["grounding"]     * result.grounding
        + w["tool_efficiency"]* result.tool_efficiency
        + w["hallucination"] * (1.0 - result.hallucination_risk)
        + w["cost_score"]    * result.cost_score
        + w["latency_score"] * result.latency_score
    )
    return round(min(max(score, 0.0), 1.0), 4)


def reliability_grade(score: float) -> str:
    """
    Map a reliability score to a human-readable letter grade.

    Parameters
    ----------
    score : float in [0, 1] from compute_reliability_score

    Returns
    -------
    str  one of "A", "B", "C", "D", "F"
    """
    if score >= 0.90: return "A"
    if score >= 0.75: return "B"
    if score >= 0.60: return "C"
    if score >= 0.45: return "D"
    return "F"


# ---------------------------------------------------------------------------
# 10.  CROSS-RUN COMPARISON
# ---------------------------------------------------------------------------

def compare_runs(results: list[EvalResult]) -> dict[str, Any]:
    """
    Aggregate and compare evaluation results across multiple agent runs.

    Parameters
    ----------
    results : list of EvalResult objects (one per run)

    Returns
    -------
    Dict containing per-metric mean/std/best/worst and a ranked leaderboard.
    """
    if not results:
        return {}

    metrics = [
        "correctness", "grounding", "faithfulness", "context_relevance",
        "tool_efficiency", "redundancy_ratio", "hallucination_risk",
        "cost_usd", "cost_score", "latency_ms", "latency_score",
        "reliability_score",
    ]

    summary: dict[str, Any] = {"n_runs": len(results), "metrics": {}}
    for m in metrics:
        vals = [getattr(r, m) for r in results]
        summary["metrics"][m] = {
            "mean":  round(statistics.mean(vals), 4),
            "std":   round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
            "best":  round(max(vals), 4),
            "worst": round(min(vals), 4),
        }

    # Leaderboard sorted by reliability_score descending
    leaderboard = sorted(
        [
            {
                "rank":              rank + 1,
                "run_id":            r.run_id,
                "task_id":           r.task_id,
                "reliability_score": r.reliability_score,
                "grade":             reliability_grade(r.reliability_score),
            }
            for rank, r in enumerate(
                sorted(results, key=lambda x: x.reliability_score, reverse=True)
            )
        ],
        key=lambda x: x["rank"],
    )
    summary["leaderboard"] = leaderboard
    return summary


def filter_runs_by_task(
    results: list[EvalResult], task_id: str
) -> list[EvalResult]:
    """
    Return only the EvalResults whose task_id matches the given value.

    Parameters
    ----------
    results : list of EvalResult objects
    task_id : filter value

    Returns
    -------
    Filtered list of EvalResult objects.
    """
    return [r for r in results if r.task_id == task_id]


def top_k_runs(results: list[EvalResult], k: int = 3) -> list[EvalResult]:
    """
    Return the top-k runs sorted by reliability_score descending.

    Parameters
    ----------
    results : list of EvalResult objects
    k       : number of top runs to return

    Returns
    -------
    List of up to k EvalResult objects.
    """
    return sorted(results, key=lambda r: r.reliability_score, reverse=True)[:k]


# ---------------------------------------------------------------------------
# 11.  REPORTING HELPERS
# ---------------------------------------------------------------------------

def format_eval_summary(result: EvalResult) -> str:
    """
    Return a human-readable summary string for one EvalResult.

    Parameters
    ----------
    result : populated EvalResult

    Returns
    -------
    Formatted multi-line string.
    """
    grade = reliability_grade(result.reliability_score)
    lines = [
        f"{'='*55}",
        f"  Run ID  : {result.run_id[:8]}…",
        f"  Task    : {result.task_id}",
        f"{'='*55}",
        f"  Correctness       : {result.correctness:.3f}",
        f"  Grounding         : {result.grounding:.3f}",
        f"    Faithfulness    : {result.faithfulness:.3f}",
        f"    Context Rel.    : {result.context_relevance:.3f}",
        f"  Tool Efficiency   : {result.tool_efficiency:.3f}",
        f"  Redundancy Ratio  : {result.redundancy_ratio:.3f}",
        f"  Hallucination Risk: {result.hallucination_risk:.3f}",
        f"  Cost (USD)        : ${result.cost_usd:.6f}",
        f"  Latency (ms)      : {result.latency_ms:.1f}",
        f"{'─'*55}",
        f"  RELIABILITY SCORE : {result.reliability_score:.4f}  [{grade}]",
        f"{'='*55}",
    ]
    return "\n".join(lines)


def export_results_json(
    results: list[EvalResult], path: str = "eval_results.json"
) -> None:
    """
    Export a list of EvalResults to a JSON file.

    Parameters
    ----------
    results : list of EvalResult objects
    path    : output file path
    """
    data = [asdict(r) for r in results]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def load_results_json(path: str = "eval_results.json") -> list[dict]:
    """
    Load EvalResults from a JSON file.

    Parameters
    ----------
    path : file path written by export_results_json

    Returns
    -------
    List of raw dicts (one per EvalResult).
    """
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 12.  TIMER CONTEXT MANAGER  (convenience)
# ---------------------------------------------------------------------------

class Timer:
    """
    Simple wall-clock timer for instrumenting tool calls and steps.

    Usage
    -----
    with Timer() as t:
        do_something()
    elapsed_ms = t.elapsed_ms
    """
    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_ms = round((time.perf_counter() - self._start) * 1000, 3)

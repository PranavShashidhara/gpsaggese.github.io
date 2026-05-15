"""
util.py
========
Unified Evaluation Pipeline Utilities

Features
--------
- Trace logging
- Tool call instrumentation
- JSONL + SQLite persistence
- RAGAS evaluation
- DeepEval evaluation
- Tool efficiency heuristics
- Cost + latency tracking
- Reliability scoring
- Cross-run comparison
- Benchmark execution helpers
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
import hashlib
import statistics

from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field

# =========================
# RAGAS
# =========================

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# =========================================================
# CONSTANTS
# =========================================================

COST_PER_INPUT_TOKEN = 3e-6
COST_PER_OUTPUT_TOKEN = 15e-6

RELIABILITY_WEIGHTS = {
    "correctness": 0.30,
    "grounding": 0.25,
    "tool_efficiency": 0.20,
    "hallucination": 0.15,
    "cost_score": 0.05,
    "latency_score": 0.05,
}

# =========================================================
# TYPE ALIASES
# =========================================================

TraceStep = dict[str, Any]
MetricDict = dict[str, float]

# =========================================================
# DATA STRUCTURES
# =========================================================


@dataclass
class ToolCall:
    call_index: int
    tool_name: str
    input: Any
    output: Any
    latency_ms: float
    success: bool
    timestamp: str
    error: Optional[str] = None
    token_usage: Optional[dict] = None


@dataclass
class TraceRecord:
    run_id: str
    task_id: str
    task_input: str

    final_response: str

    steps: list[TraceStep] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)

    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class EvalResult:
    run_id: str
    task_id: str

    correctness: float = 0.0

    grounding: float = 0.0
    faithfulness: float = 0.0
    context_relevance: float = 0.0

    tool_efficiency: float = 0.0
    redundancy_ratio: float = 0.0

    hallucination_risk: float = 0.0

    cost_usd: float = 0.0
    cost_score: float = 0.0

    latency_ms: float = 0.0
    latency_score: float = 0.0

    reliability_score: float = 0.0

    raw_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentRun:
    experiment_id: str
    model_name: str
    framework: str
    temperature: float
    timestamp: str
    run_ids: list[str]


# =========================================================
# TRACE LOGGING
# =========================================================


def generate_run_id() -> str:
    return str(uuid.uuid4())


def create_trace(
    task_id: str,
    task_input: str,
    run_id: Optional[str] = None,
) -> TraceRecord:

    return TraceRecord(
        run_id=run_id or generate_run_id(),
        task_id=task_id,
        task_input=task_input,
        final_response="",
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

    step = {
        "step_index": len(trace.steps),
        "step_type": step_type,
        "content": content,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }

    trace.steps.append(step)

    trace.total_input_tokens += input_tokens
    trace.total_output_tokens += output_tokens
    trace.total_latency_ms += latency_ms

    return step


def log_tool_call(
    trace: TraceRecord,
    tool_name: str,
    tool_input: Any,
    tool_output: Any,
    latency_ms: float = 0.0,
    success: bool = True,
    error: Optional[str] = None,
    token_usage: Optional[dict] = None,
) -> ToolCall:

    call = ToolCall(
        call_index=len(trace.tool_calls),
        tool_name=tool_name,
        input=tool_input,
        output=tool_output,
        latency_ms=latency_ms,
        success=success,
        timestamp=datetime.now(timezone.utc).isoformat(),
        error=error,
        token_usage=token_usage,
    )

    trace.tool_calls.append(call)

    log_step(
        trace,
        step_type="tool_call",
        content={
            "tool": tool_name,
            "success": success,
            "error": error,
        },
        latency_ms=latency_ms,
    )

    return call


def finalise_trace(
    trace: TraceRecord,
    final_response: str,
) -> TraceRecord:

    trace.final_response = final_response

    log_step(
        trace,
        step_type="final",
        content=final_response,
    )

    return trace


# =========================================================
# PERSISTENCE
# =========================================================


def save_trace_jsonl(
    trace: TraceRecord,
    path: str = "traces.jsonl",
) -> None:

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(trace), ensure_ascii=False) + "\n")


def load_traces_jsonl(
    path: str = "traces.jsonl",
) -> list[dict]:

    if not Path(path).exists():
        return []

    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def init_sqlite(
    db_path: str = "traces.db",
) -> sqlite3.Connection:

    conn = sqlite3.connect(db_path)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS traces (
            run_id TEXT PRIMARY KEY,
            task_id TEXT,
            task_input TEXT,
            final_response TEXT,
            payload TEXT,
            created_at TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_results (
            run_id TEXT PRIMARY KEY,
            task_id TEXT,
            correctness REAL,
            grounding REAL,
            faithfulness REAL,
            context_relevance REAL,
            tool_efficiency REAL,
            redundancy_ratio REAL,
            hallucination_risk REAL,
            cost_usd REAL,
            cost_score REAL,
            latency_ms REAL,
            latency_score REAL,
            reliability_score REAL,
            raw_metrics TEXT,
            created_at TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT PRIMARY KEY,
            payload TEXT,
            created_at TEXT
        )
        """
    )

    conn.commit()

    return conn


def save_trace_sqlite(
    trace: TraceRecord,
    conn: sqlite3.Connection,
) -> None:

    payload = json.dumps(asdict(trace), ensure_ascii=False)

    conn.execute(
        """
        INSERT OR REPLACE INTO traces
        (run_id, task_id, task_input, final_response, payload, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            trace.run_id,
            trace.task_id,
            trace.task_input,
            trace.final_response,
            payload,
            trace.created_at,
        ),
    )

    conn.commit()


def save_eval_result_sqlite(
    result: EvalResult,
    conn: sqlite3.Connection,
) -> None:

    conn.execute(
        """
        INSERT OR REPLACE INTO eval_results
        (
            run_id,
            task_id,
            correctness,
            grounding,
            faithfulness,
            context_relevance,
            tool_efficiency,
            redundancy_ratio,
            hallucination_risk,
            cost_usd,
            cost_score,
            latency_ms,
            latency_score,
            reliability_score,
            raw_metrics,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.run_id,
            result.task_id,
            result.correctness,
            result.grounding,
            result.faithfulness,
            result.context_relevance,
            result.tool_efficiency,
            result.redundancy_ratio,
            result.hallucination_risk,
            result.cost_usd,
            result.cost_score,
            result.latency_ms,
            result.latency_score,
            result.reliability_score,
            json.dumps(result.raw_metrics),
            datetime.now(timezone.utc).isoformat(),
        ),
    )

    conn.commit()


# =========================================================
# RAGAS EVALUATION
# =========================================================


def run_ragas_evaluation(
    query: str,
    answer: str,
    retrieved_contexts: list[str],
    ground_truth: str,
) -> MetricDict:
    """
    Run RAGAS evaluation with robust score extraction.
    """
    if not retrieved_contexts:
        return {
            "faithfulness": 0.0,
            "context_relevance": 0.0,
            "grounding_score": 0.0,
        }

    # Use wrappers to ensure compatibility with Ragas internal expectations
    r_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    r_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    dataset = Dataset.from_dict({
        "question": [query],
        "answer": [answer],
        "contexts": [retrieved_contexts],
        "reference": [ground_truth],
    })

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=r_llm,
        embeddings=r_embeddings,
    )

    # Convert Result object to a dictionary via Pandas for reliable indexing
    try:
        # result.to_pandas() returns a DataFrame; we take the first row (iloc[0])
        scores_dict = result.to_pandas().iloc[0].to_dict()
        
        faith = float(scores_dict.get("faithfulness", 0.0))
        ctx_p = float(scores_dict.get("context_precision", 0.0))
        ctx_r = float(scores_dict.get("context_recall", 0.0))
    except Exception as e:
        # Fallback for environments where Pandas behavior might differ
        print(f"⚠️ Ragas score extraction fallback: {e}")
        # Attempt to access the 'scores' attribute directly as a fallback
        try:
            s = result.scores
            faith = s["faithfulness"][0] if isinstance(s["faithfulness"], list) else s["faithfulness"]
            ctx_p = s["context_precision"][0] if isinstance(s["context_precision"], list) else s["context_precision"]
            ctx_r = s["context_recall"][0] if isinstance(s["context_recall"], list) else s["context_recall"]
        except:
            faith, ctx_p, ctx_r = 0.0, 0.0, 0.0

    ctx_rel = (ctx_p + ctx_r) / 2

    return {
        "faithfulness": round(float(faith), 4),
        "context_relevance": round(float(ctx_rel), 4),
        "grounding_score": round(float((faith + ctx_rel) / 2), 4),
    }

# =========================================================
# DEEPEVAL
# =========================================================


def run_deepeval_evaluation(
    query: str,
    answer: str,
    ground_truth: str,
    retrieved_contexts=None,
):

    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        expected_output=ground_truth,
        context=retrieved_contexts or [],
    )

    correctness_metric = GEval(
        name="Correctness",
        criteria="Evaluate factual correctness vs expected output.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
    )

    relevance_metric = GEval(
        name="Answer Relevance",
        criteria="Does the answer address the question?",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
    )

    correctness_metric.measure(test_case)
    relevance_metric.measure(test_case)

    return {
        "correctness": float(correctness_metric.score or 0.0),
        "answer_relevance": float(relevance_metric.score or 0.0),
        "hallucination_risk": 1.0 - float(correctness_metric.score or 0.0),
    }


# =========================================================
# HEURISTICS
# =========================================================


def compute_tool_efficiency(
    tool_calls: list[ToolCall],
) -> float:

    if not tool_calls:
        return 1.0

    successful = [tc for tc in tool_calls if tc.success]

    if not successful:
        return 0.0

    unique_tools = {tc.tool_name for tc in successful}

    return round(len(unique_tools) / len(tool_calls), 4)


def compute_redundancy_ratio(
    tool_calls: list[ToolCall],
) -> float:

    if len(tool_calls) < 2:
        return 0.0

    def fingerprint(tc: ToolCall) -> str:
        key = (
            f"{tc.tool_name}:"
            f"{json.dumps(tc.input, sort_keys=True)}"
        )

        return hashlib.md5(key.encode()).hexdigest()

    seen = set()

    redundant = 0

    for tc in tool_calls:
        fp = fingerprint(tc)

        if fp in seen:
            redundant += 1

        seen.add(fp)

    return round(redundant / len(tool_calls), 4)


def run_custom_heuristics(
    tool_calls: list[ToolCall],
) -> MetricDict:

    return {
        "tool_efficiency": compute_tool_efficiency(tool_calls),
        "redundancy_ratio": compute_redundancy_ratio(tool_calls),
    }


# =========================================================
# COST + LATENCY
# =========================================================


def compute_cost(
    input_tokens: int,
    output_tokens: int,
    cost_per_input: float = COST_PER_INPUT_TOKEN,
    cost_per_output: float = COST_PER_OUTPUT_TOKEN,
) -> float:

    return round(
        input_tokens * cost_per_input
        + output_tokens * cost_per_output,
        6,
    )


def cost_to_score(
    cost_usd: float,
    budget_usd: float = 0.05,
) -> float:

    if budget_usd <= 0:
        return 0.0

    return round(max(0.0, 1.0 - cost_usd / budget_usd), 4)


def latency_to_score(
    latency_ms: float,
    threshold_ms: float = 30000,
) -> float:

    return round(
        max(0.0, 1.0 - latency_ms / threshold_ms),
        4,
    )


def run_cost_latency_analysis(
    trace: TraceRecord,
) -> MetricDict:

    cost = compute_cost(
        trace.total_input_tokens,
        trace.total_output_tokens,
    )

    return {
        "cost_usd": cost,
        "cost_score": cost_to_score(cost),
        "total_latency_ms": trace.total_latency_ms,
        "latency_score": latency_to_score(trace.total_latency_ms),
    }


# =========================================================
# UNIFIED EVALUATION
# =========================================================


def evaluate_trace(
    trace: TraceRecord,
    ground_truth: str,
    retrieved_contexts: Optional[list[str]] = None,
) -> EvalResult:

    contexts = retrieved_contexts or []

    ragas = run_ragas_evaluation(
        trace.task_input,
        trace.final_response,
        contexts,
        ground_truth
    )

    deepev = run_deepeval_evaluation(
        trace.task_input,
        trace.final_response,
        ground_truth,
        contexts,
    )

    heur = run_custom_heuristics(trace.tool_calls)

    cl = run_cost_latency_analysis(trace)

    raw = {
        **ragas,
        **deepev,
        **heur,
        **cl,
    }

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


# =========================================================
# RELIABILITY
# =========================================================


def compute_reliability_score(
    result: EvalResult,
    weights: Optional[dict[str, float]] = None,
) -> float:

    w = weights or RELIABILITY_WEIGHTS

    score = (
        w["correctness"] * result.correctness
        + w["grounding"] * result.grounding
        + w["tool_efficiency"] * result.tool_efficiency
        + w["hallucination"] * (1.0 - result.hallucination_risk)
        + w["cost_score"] * result.cost_score
        + w["latency_score"] * result.latency_score
    )

    return round(min(max(score, 0.0), 1.0), 4)


def reliability_grade(score: float) -> str:

    if score >= 0.90:
        return "A"

    if score >= 0.75:
        return "B"

    if score >= 0.60:
        return "C"

    if score >= 0.45:
        return "D"

    return "F"


# =========================================================
# COMPARISON
# =========================================================


def compare_runs(
    results: list[EvalResult],
) -> dict[str, Any]:

    if not results:
        return {}

    metrics = [
        "correctness",
        "grounding",
        "tool_efficiency",
        "hallucination_risk",
        "cost_score",
        "latency_score",
        "reliability_score",
    ]

    summary = {
        "n_runs": len(results),
        "metrics": {},
    }

    for m in metrics:

        vals = [getattr(r, m) for r in results]

        summary["metrics"][m] = {
            "mean": round(statistics.mean(vals), 4),
            "std": round(
                statistics.stdev(vals)
                if len(vals) > 1
                else 0.0,
                4,
            ),
            "best": round(max(vals), 4),
            "worst": round(min(vals), 4),
        }

    leaderboard = sorted(
        results,
        key=lambda r: r.reliability_score,
        reverse=True,
    )

    summary["leaderboard"] = [
        {
            "rank": i + 1,
            "run_id": r.run_id,
            "task_id": r.task_id,
            "score": r.reliability_score,
            "grade": reliability_grade(r.reliability_score),
        }
        for i, r in enumerate(leaderboard)
    ]

    return summary


# =========================================================
# BENCHMARK RUNNER
# =========================================================


def benchmark_agent(
    agent,
    dataset,
    conn,
):

    all_results = []

    for sample in dataset:

        trace = create_trace(
            task_id=sample["id"],
            task_input=sample["question"],
        )

        answer = agent.run(sample["question"])

        finalise_trace(trace, answer)

        result = evaluate_trace(
            trace,
            ground_truth=sample["ground_truth"],
            retrieved_contexts=sample.get("contexts", []),
        )

        save_trace_sqlite(trace, conn)

        save_eval_result_sqlite(result, conn)

        all_results.append(result)

    return compare_runs(all_results)


# =========================================================
# REPORTING
# =========================================================


def format_eval_summary(
    result: EvalResult,
) -> str:

    grade = reliability_grade(result.reliability_score)

    lines = [
        "=" * 55,
        f"Run ID: {result.run_id[:8]}...",
        f"Task: {result.task_id}",
        "=" * 55,
        f"Correctness: {result.correctness:.3f}",
        f"Grounding: {result.grounding:.3f}",
        f"Faithfulness: {result.faithfulness:.3f}",
        f"Context Relevance: {result.context_relevance:.3f}",
        f"Tool Efficiency: {result.tool_efficiency:.3f}",
        f"Redundancy Ratio: {result.redundancy_ratio:.3f}",
        f"Hallucination Risk: {result.hallucination_risk:.3f}",
        f"Cost USD: ${result.cost_usd:.6f}",
        f"Latency ms: {result.latency_ms:.1f}",
        "-" * 55,
        f"Reliability Score: {result.reliability_score:.4f} [{grade}]",
        "=" * 55,
    ]

    return "\n".join(lines)

# =========================================================
# ADDITIONAL HELPERS / CONTINUATION SECTION
# =========================================================


def filter_runs_by_task(
    results: list[EvalResult],
    task_id: str,
) -> list[EvalResult]:
    """
    Filter EvalResults by task ID.
    """

    return [r for r in results if r.task_id == task_id]



def top_k_runs(
    results: list[EvalResult],
    k: int = 3,
) -> list[EvalResult]:
    """
    Return top-k runs by reliability score.
    """

    return sorted(
        results,
        key=lambda r: r.reliability_score,
        reverse=True,
    )[:k]


# =========================================================
# EXPORT HELPERS
# =========================================================


def export_results_json(
    results: list[EvalResult],
    path: str = "eval_results.json",
) -> None:
    """
    Export evaluation results to JSON.
    """

    data = [asdict(r) for r in results]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )



def load_results_json(
    path: str = "eval_results.json",
) -> list[dict]:
    """
    Load evaluation results from JSON.
    """

    if not Path(path).exists():
        return []

    with open(path, encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# SQLITE LOADERS
# =========================================================


def load_traces_sqlite(
    conn: sqlite3.Connection,
    task_id: Optional[str] = None,
) -> list[dict]:
    """
    Load traces from SQLite.
    """

    if task_id:
        rows = conn.execute(
            "SELECT payload FROM traces WHERE task_id = ?",
            (task_id,),
        ).fetchall()

    else:
        rows = conn.execute(
            "SELECT payload FROM traces"
        ).fetchall()

    return [json.loads(r[0]) for r in rows]



def load_eval_results_sqlite(
    conn: sqlite3.Connection,
) -> list[dict]:
    """
    Load evaluation results from SQLite.
    """

    rows = conn.execute(
        "SELECT raw_metrics FROM eval_results"
    ).fetchall()

    return [json.loads(r[0]) for r in rows]


# =========================================================
# EXPERIMENT TRACKING
# =========================================================


def save_experiment_sqlite(
    experiment: ExperimentRun,
    conn: sqlite3.Connection,
) -> None:
    """
    Save experiment metadata.
    """

    payload = json.dumps(
        asdict(experiment),
        ensure_ascii=False,
    )

    conn.execute(
        """
        INSERT OR REPLACE INTO experiments
        (experiment_id, payload, created_at)
        VALUES (?, ?, ?)
        """,
        (
            experiment.experiment_id,
            payload,
            datetime.now(timezone.utc).isoformat(),
        ),
    )

    conn.commit()



def load_experiments_sqlite(
    conn: sqlite3.Connection,
) -> list[dict]:
    """
    Load experiments from SQLite.
    """

    rows = conn.execute(
        "SELECT payload FROM experiments"
    ).fetchall()

    return [json.loads(r[0]) for r in rows]


# =========================================================
# RETRIEVAL TRACE HELPERS
# =========================================================


def log_retrieval_step(
    trace: TraceRecord,
    query: str,
    retrieved_docs: list[str],
    similarity_scores: Optional[list[float]] = None,
):
    """
    Log retrieval information.
    """

    log_step(
        trace,
        step_type="retrieval",
        content={
            "query": query,
            "retrieved_docs": retrieved_docs,
            "similarity_scores": similarity_scores or [],
        },
    )


# =========================================================
# FAILURE ANALYSIS
# =========================================================


def count_failed_tool_calls(
    tool_calls: list[ToolCall],
) -> int:
    """
    Count failed tool calls.
    """

    return sum(1 for tc in tool_calls if not tc.success)



def failed_tool_call_rate(
    tool_calls: list[ToolCall],
) -> float:
    """
    Compute failure rate.
    """

    if not tool_calls:
        return 0.0

    failures = count_failed_tool_calls(tool_calls)

    return round(failures / len(tool_calls), 4)


# =========================================================
# TOKEN ACCOUNTING HELPERS
# =========================================================


def update_trace_token_usage(
    trace: TraceRecord,
    prompt_tokens: int,
    completion_tokens: int,
):
    """
    Increment token counters.
    """

    trace.total_input_tokens += prompt_tokens
    trace.total_output_tokens += completion_tokens


# =========================================================
# TRACE SERIALIZATION
# =========================================================


def trace_to_dict(
    trace: TraceRecord,
) -> dict:
    """
    Convert trace to serializable dict.
    """

    return asdict(trace)



def eval_result_to_dict(
    result: EvalResult,
) -> dict:
    """
    Convert EvalResult to dict.
    """

    return asdict(result)


# =========================================================
# METRIC NORMALIZATION
# =========================================================


def normalize_metric(
    value: float,
    min_value: float,
    max_value: float,
) -> float:
    """
    Normalize arbitrary metric to [0,1].
    """

    if max_value == min_value:
        return 0.0

    normalized = (value - min_value) / (max_value - min_value)

    return round(
        min(max(normalized, 0.0), 1.0),
        4,
    )


# =========================================================
# RUN SUMMARY GENERATOR
# =========================================================


def generate_run_summary(
    trace: TraceRecord,
    result: EvalResult,
) -> dict[str, Any]:
    """
    Generate compact run summary.
    """

    return {
        "run_id": trace.run_id,
        "task_id": trace.task_id,
        "n_steps": len(trace.steps),
        "n_tool_calls": len(trace.tool_calls),
        "latency_ms": trace.total_latency_ms,
        "input_tokens": trace.total_input_tokens,
        "output_tokens": trace.total_output_tokens,
        "reliability_score": result.reliability_score,
        "grade": reliability_grade(result.reliability_score),
    }


# =========================================================
# DATASET BENCHMARK UTILITIES
# =========================================================


def benchmark_dataset_statistics(
    dataset,
) -> dict[str, Any]:
    """
    Compute dataset statistics.
    """

    n_samples = len(dataset)

    avg_question_length = statistics.mean(
        len(sample["question"].split())
        for sample in dataset
    )

    return {
        "n_samples": n_samples,
        "avg_question_length": round(avg_question_length, 2),
    }


# =========================================================
# DEBUGGING HELPERS
# =========================================================


def print_trace(trace: TraceRecord):
    """
    Pretty-print trace.
    """

    print("=" * 80)
    print(f"RUN ID: {trace.run_id}")
    print(f"TASK ID: {trace.task_id}")
    print("=" * 80)

    for step in trace.steps:

        print(
            f"[{step['step_index']}] "
            f"{step['step_type']}"
        )

        print(step["content"])
        print("-" * 80)



def print_tool_calls(trace: TraceRecord):
    """
    Pretty-print tool calls.
    """

    for tc in trace.tool_calls:

        print("=" * 60)
        print(f"TOOL: {tc.tool_name}")
        print(f"SUCCESS: {tc.success}")
        print(f"LATENCY: {tc.latency_ms} ms")

        if tc.error:
            print(f"ERROR: {tc.error}")

        print("INPUT:")
        print(tc.input)

        print("OUTPUT:")
        print(tc.output)


# =========================================================
# OPTIONAL OPENAI TOKEN HELPER
# =========================================================


def extract_openai_token_usage(response) -> dict:
    """
    Extract OpenAI token usage safely.
    """

    usage = getattr(response, "usage", None)

    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


# =========================================================
# END OF FILE
# =========================================================


# =========================================================
# TIMER
# =========================================================

class Timer:
    """
    Simple context-manager timer utility.

    Usage:
    -------
    with Timer() as t:
        run_expensive_operation()

    print(t.elapsed_ms)
    """

    def __enter__(self):

        self._start = time.perf_counter()

        self.elapsed_ms = 0.0

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.elapsed_ms = round(
            (time.perf_counter() - self._start) * 1000,
            3,
        )

        # Do not suppress exceptions
        return False
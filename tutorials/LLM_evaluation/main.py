# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# !pip install "langchain>=0.1.0" "langchain-core>=0.1.0" "langchain-openai>=0.1.0" yfinance

# %%
# %load_ext autoreload
# %autoreload 2
     
import json
import os
import sys
import time
import asyncio
from pathlib import Path
import statistics
import sqlite3
import nest_asyncio
nest_asyncio.apply()
try:
    import yfinance as yf
    from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from typing import Any
except ImportError as exc:
    print(
        f"\n  Missing dependency: {exc}\n"
        "Run the following and try again:\n\n"
        '    pip install "langchain>=0.1.0" "langchain-core>=0.1.0" '
        '"langchain-openai>=0.1.0" yfinance\n'
    )
    sys.exit(1)


# %%
# %load_ext autoreload
# %autoreload 2
import os
from util import (
    # Constants
    COST_PER_INPUT_TOKEN,
    COST_PER_OUTPUT_TOKEN,
    RELIABILITY_WEIGHTS,
    # Data structures
    EvalResult,
    TraceRecord,
    # Trace logging
    create_trace,
    finalise_trace,
    generate_run_id,
    log_step,
    log_tool_call,
    # Persistence
    init_sqlite,
    load_traces_jsonl,
    save_eval_result_sqlite,
    save_trace_jsonl,
    save_trace_sqlite,
    # Evaluation
    evaluate_trace,
    # Scoring
    compute_reliability_score,
    reliability_grade,
    # Cross-run analysis
    filter_runs_by_task,
    top_k_runs,
    # Reporting
    export_results_json,
    format_eval_summary,
    load_results_json,
)
 

# --- API key ---
# Set OPENAI_API_KEY in your shell; never hard-code it here.
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"  # <-- REPLACE WITH YOUR KEY OR SET AS ENV VAR


# %%
# --- Model ---
MODEL_NAME  = "gpt-4o"
TEMPERATURE = 0
SEED        = 42
 
# --- Persistence paths ---
JSONL_PATH = "agent_traces.jsonl"
DB_PATH    = "agent_traces.db"
EXPORT_PATH = "eval_results.json"
 
# --- Leaderboard top-K ---
TOP_K = 3

# --- Task definitions ---
TASKS = [
    {
        "task_id": "aapl_price",
        "question": "What is the current stock price of Apple (AAPL)?",
        "ground_truth": "The current stock price of Apple is approximately $150-$300 (varies with market conditions)",
    },
    {
        "task_id": "msft_pe_ratio",
        "question": "What is the trailing P/E ratio of Microsoft (MSFT)?",
        "ground_truth": "The trailing P/E ratio of Microsoft is typically between 20-40",
    },
    {
        "task_id": "googl_market_cap",
        "question": "What is the market capitalisation of Alphabet (GOOGL)?",
        "ground_truth": "The market capitalisation of Alphabet is in the range of $1-2 trillion",
    },
    {
        "task_id": "nvda_52w_range",
        "question": "What are the 52-week high and low prices for NVIDIA (NVDA)?",
        "ground_truth": "The 52-week high and low prices vary with market conditions",
    },
    {
        "task_id": "tsla_revenue",
        "question": "What is Tesla's (TSLA) trailing 12-month revenue?",
        "ground_truth": "Tesla's trailing 12-month revenue is typically between $80-120 billion",
    },
]


# %%
@tool
def get_financial_data(query: str) -> str:
    """
    Retrieve financial data for a stock ticker using yfinance.
 
    The query must be a JSON string with:
        ticker  : str  — stock symbol, e.g. "AAPL"
        metric  : str  — one of:
                         "price"           current price
                         "market_cap"      market capitalisation
                         "pe_ratio"        trailing P/E ratio
                         "revenue"         trailing 12-month revenue
                         "net_income"      trailing 12-month net income
                         "52w_high"        52-week high
                         "52w_low"         52-week low
                         "dividend_yield"  annual dividend yield
                         "info"            partial info dict (verbose)
 
    Returns a plain-text result string.
    """
    try:
        params  = json.loads(query)
        ticker  = params.get("ticker", "").upper()
        metric  = params.get("metric", "price").lower()
    except Exception:
        return (
            f"ERROR: query must be valid JSON with "
            f"'ticker' and 'metric' keys. Got: {query}"
        )
 
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info
 
        mapping = {
            "price":          info.get("currentPrice") or info.get("regularMarketPrice"),
            "market_cap":     info.get("marketCap"),
            "pe_ratio":       info.get("trailingPE"),
            "revenue":        info.get("totalRevenue"),
            "net_income":     info.get("netIncomeToCommon"),
            "52w_high":       info.get("fiftyTwoWeekHigh"),
            "52w_low":        info.get("fiftyTwoWeekLow"),
            "dividend_yield": info.get("dividendYield"),
            "info": json.dumps(
                {k: info[k] for k in list(info)[:30]},
                default=str,
                indent=2,
            ),
        }
 
        if metric not in mapping:
            return (
                f"Unknown metric '{metric}'. "
                f"Choose from: {list(mapping.keys())}"
            )
 
        value = mapping[metric]
        if value is None:
            return f"No data available for {metric} on {ticker}."
 
        return f"{ticker} {metric}: {value}"
 
    except Exception as e:
        return f"ERROR fetching {ticker}: {e}"


# %%
TOOLS = [get_financial_data]
 
# =========================================================
# 5  Instrumented LangChain callback
# =========================================================
 
class TraceCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback that writes every reasoning step and tool
    invocation into a TraceRecord via util.py logging functions.
    Attach one instance per agent run for isolated traces.
    """
 
    def __init__(self, trace: TraceRecord) -> None:
        super().__init__()
        self.trace             = trace
        self._step_start: float = 0.0
        self._tool_start: float = 0.0
        self._tool_name:  str   = ""
        self._tool_input: str   = ""
 
    # ── LLM events ──────────────────────────────────────────────
 
    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        self._step_start = time.perf_counter()
 
    def on_llm_end(self, response, **kwargs) -> None:
        elapsed_ms = round((time.perf_counter() - self._step_start) * 1000, 2)
 
        # langchain 0.1+: generations is list[list[ChatGeneration]]
        in_tok, out_tok = 0, 0
        try:
            gen0 = response.generations[0][0] if response.generations else None
            info = getattr(gen0, "generation_info", None) or {}
            usage = info.get("token_usage") or info.get("usage") or {}
            if isinstance(usage, dict):
                in_tok  = usage.get("prompt_tokens", 0)
                out_tok = usage.get("completion_tokens", 0)
            else:
                in_tok  = getattr(usage, "prompt_tokens", 0)
                out_tok = getattr(usage, "completion_tokens", 0)
        except Exception:
            pass

        # ChatGeneration has .message (AIMessage); fall back to .text
        try:
            gen0 = response.generations[0][0] if response.generations else None
            msg = getattr(gen0, "message", None)
            if msg is not None:
                content = getattr(msg, "content", "") or "(tool-call decision)"
            else:
                content = getattr(gen0, "text", "") or "(LLM step)"
        except Exception:
            content = "(LLM step)"
 
        log_step(
            self.trace,
            step_type="reasoning",
            content=content,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=elapsed_ms,
        )
 
    # ── Tool events ──────────────────────────────────────────────
 
    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        self._tool_start = time.perf_counter()
        self._tool_name  = serialized.get("name", "unknown_tool")
        self._tool_input = input_str
 
    def on_tool_end(self, output, **kwargs) -> None:
        elapsed_ms = round((time.perf_counter() - self._tool_start) * 1000, 2)
        log_tool_call(
            self.trace,
            tool_name=self._tool_name,
            tool_input=self._tool_input,
            tool_output=str(output),
            latency_ms=elapsed_ms,
            success=True,
        )
 
    def on_tool_error(self, error, **kwargs) -> None:
        elapsed_ms = round((time.perf_counter() - self._tool_start) * 1000, 2)
        log_tool_call(
            self.trace,
            tool_name=self._tool_name,
            tool_input=self._tool_input,
            tool_output=str(error),
            latency_ms=elapsed_ms,
            success=False,
            error=str(error),
        )


# =========================================================
# 6  Agent factory
# =========================================================

def build_agent_executor(trace: TraceRecord) -> AgentExecutor:
    """
    Build a fresh agent executor with the callback attached.
    The callback is attached ONLY to the executor (not the LLM) to
    avoid double-firing. Tool use is forced via bind_tools so the
    model cannot answer from memory.
    """
    cb = TraceCallbackHandler(trace)

    # FIX 1: seed is a first-class param in langchain-openai >= 0.1,
    #         not model_kwargs — eliminates the UserWarning.
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        seed=SEED,
    )

    # FIX 2: bind_tools with tool_choice="get_financial_data" forces
    #         the model to call the tool on its first turn, so
    #         on_tool_start/end always fire and tool_calls is never 0.
    llm_with_tools = llm.bind_tools(TOOLS)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You MUST use the get_financial_data tool for ALL financial questions. "
            "Do not answer from memory. "
            "If you do not use the tool, your answer is incorrect."
        )),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm_with_tools, TOOLS, prompt)

    # FIX 3: callback only on the executor — it propagates to tool
    #         events automatically; attaching to LLM too causes
    #         double-logging of every reasoning step.
    executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=False,
        callbacks=[cb],
        return_intermediate_steps=True,
        max_iterations=5,
        handle_parsing_errors=True,
    )

    return executor


# =========================================================
# 7  Setup persistence
# =========================================================

def setup_persistence() -> sqlite3.Connection:
    """Initialize SQLite database and return connection."""
    import sqlite3
    
    # Clear old database if exists (for fresh runs)
    # Uncomment if you want fresh runs each time:
    # if Path(DB_PATH).exists():
    #     Path(DB_PATH).unlink()
    
    return init_sqlite(DB_PATH)


# =========================================================
# 8  Task runner and reporter
# =========================================================

def compare_runs(results: list[EvalResult]) -> dict[str, Any]:
    """Compare runs across metrics and build leaderboard."""
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
            "std": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
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
            "score": r.reliability_score,          # ✅ standardized
            "reliability_score": r.reliability_score,  # optional alias (safe)
            "grade": reliability_grade(r.reliability_score),
        }
        for i, r in enumerate(leaderboard)
    ]

    return summary
    

def run_single_task(task: dict, conn) -> tuple[TraceRecord, EvalResult]:
    """
    Run the agent on one task, capture its trace, evaluate it,
    and persist everything.  Returns (trace, result).
    
    FIX: Removed asyncio context issues by avoiding nested event loops.
    """
    print(f"\n{'='*60}")
    print(f"  Task : {task['task_id']}")
    print(f"  Query: {task['question']}")
    print(f"{'='*60}")
 
    # 1. Create trace
    trace = create_trace(task["task_id"], task["question"])
 
    # 2. Build a fresh agent executor (isolated callbacks per run)
    executor = build_agent_executor(trace)
 
    # 3. Run the agent (synchronously - no asyncio issues)
    wall_start = time.perf_counter()
    try:
        # ---> THE FIX: Pass the callback inside the config dict <---
        # This forces LangChain to trigger the tracker for every inner LLM & Tool step
        cb = TraceCallbackHandler(trace)
        response = executor.invoke(
            {"input": task["question"]},
            config={"callbacks": [cb]}
        )
        agent_answer = response.get("output", "")
    except Exception as exc:
        agent_answer = f"AGENT ERROR: {exc}"
        print(f"    Error: {exc}")
 
    wall_ms = round((time.perf_counter() - wall_start) * 1000, 2)
 
    # Print a preview of the answer
    preview = agent_answer[:100] + ("..." if len(agent_answer) > 100 else "")
    print(f"  Answer : {preview}")
 
    # 4. Finalise trace
    finalise_trace(trace, agent_answer)
 
    # 5. Build context list from successful tool outputs
    tool_contexts = [
        tc["output"] if isinstance(tc, dict) else tc.output
        for tc in trace.tool_calls
        if (tc.get("success", True) if isinstance(tc, dict) else tc.success)
    ]
    task["contexts"] = tool_contexts
 
    # 6. Persist trace
    save_trace_jsonl(trace, JSONL_PATH)
    save_trace_sqlite(trace, conn)
 
    # 7. Evaluate
    result = evaluate_trace(
        trace,
        ground_truth=task.get("ground_truth", ""),
        retrieved_contexts=tool_contexts,
    )
    save_eval_result_sqlite(result, conn)
 
    print(f"  Steps      : {len(trace.steps)}")
    print(f"  Tool calls : {len(trace.tool_calls)}")
    print(f"  Tokens     : {trace.total_input_tokens}in / {trace.total_output_tokens}out")
    print(f"  Latency    : {trace.total_latency_ms:.0f} ms  (wall: {wall_ms:.0f} ms)")
    print(
        f"  Reliability: {result.reliability_score:.4f}  "
        f"[{reliability_grade(result.reliability_score)}]"
    )
 
    return trace, result
 
 
def run_pipeline(tasks: list[dict]) -> list[EvalResult]:
    """Run the full evaluation pipeline over every task."""
    conn        = setup_persistence()
    all_results: list[EvalResult] = []
 
    for task in tasks:
        _, result = run_single_task(task, conn)
        all_results.append(result)
 
    print(f"\n✅  All {len(tasks)} tasks completed.")
    conn.close()
    return all_results
 
 
# =========================================================
# 9  Reporting helpers
# =========================================================
 
def print_leaderboard(all_results: list[EvalResult]) -> None:
    """Print the cross-run comparison table and leaderboard."""
    comparison = compare_runs(all_results)

    print(f"\nRuns compared : {comparison['n_runs']}\n")
    print("Per-metric aggregates:")
    print(f"{'Metric':<24} {'Mean':>7} {'Std':>7} {'Best':>7} {'Worst':>7}")
    print("─" * 56)

    for metric, stats in comparison["metrics"].items():
        print(
            f"{metric:<24} "
            f"{stats['mean']:>7.3f} "
            f"{stats['std']:>7.3f} "
            f"{stats['best']:>7.3f} "
            f"{stats['worst']:>7.3f}"
        )

    print("\nLeaderboard:")
    print(f"{'Rank':<6} {'Run ID':<10} {'Task':<22} {'Score':<8} Grade")
    print("─" * 54)

    for entry in comparison["leaderboard"]:
        print(
            f"{entry['rank']:<6}"
            f"{entry['run_id'][:8]:<10}"
            f"{entry['task_id']:<22}"
            f"{entry['score']:<8.4f}"   # ✅ FIXED HERE
            f"{entry['grade']}"
        )
 
 
def print_top_k(all_results: list[EvalResult], k: int = TOP_K) -> None:
    """Print the top-K runs by reliability score."""
    top = top_k_runs(all_results, k=k)
    print(f"\nTop-{k} runs:")
    print(f"{'RUN ID':<10} {'TASK':<20} {'SCORE':<8} GRADE")
    print("-" * 50)
    for r in top:
        print(
            f"{r.run_id[:8]:<10} "
            f"{r.task_id:<20} "
            f"{r.reliability_score:<8.4f} "
            f"{reliability_grade(r.reliability_score)}"
        )
 
 
def print_per_run_reports(all_results: list[EvalResult]) -> None:
    """Print the detailed evaluation report for every run."""
    print("\n" + "=" * 60)
    print("  DETAILED PER-RUN EVALUATION REPORTS")
    print("=" * 60)
    for result in all_results:
        print(format_eval_summary(result))
        print()
 
 
def inspect_traces(n: int = 1) -> None:
    """Pretty-print the first n raw traces from the JSONL file."""
    raw_traces = load_traces_jsonl(JSONL_PATH)
    for t in raw_traces[:n]:
        print(f"\nrun_id    : {t['run_id']}")
        print(f"task_id   : {t['task_id']}")
        print(f"input     : {t['task_input']}")
        print(f"response  : {t['final_response'][:120]}")
        print(f"steps     : {len(t['steps'])}")
        print(f"tool calls: {len(t['tool_calls'])}")
 
        if t["tool_calls"]:
            print("\nTool call detail:")
            for tc in t["tool_calls"]:
                tc_dict = tc if isinstance(tc, dict) else tc.__dict__
                print(f"  [{tc_dict['call_index']}] {tc_dict['tool_name']}")
                print(f"       input  : {str(tc_dict['input'])[:80]}")
                print(f"       output : {str(tc_dict['output'])[:80]}")
                print(f"       latency: {tc_dict['latency_ms']} ms  success: {tc_dict['success']}")
 
        print("\nStep-by-step trace:")
        for s in t["steps"]:
            content_str = str(s["content"])
            print(
                f"  [{s['step_index']}] {s['step_type']:<12} "
                f"lat={s['latency_ms']:>7.1f}ms  "
                f"in={s['input_tokens']}tok  out={s['output_tokens']}tok"
            )
            print(f"       {content_str[:100]}")


# %%
# =========================================================
# 10  Entry point
# =========================================================
 
def main() -> None:
    print("=" * 60)
    print("  Unified LLM Agent Evaluation Pipeline")
    print("=" * 60)
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Tasks      : {len(TASKS)}")
    print(f"  JSONL      : {JSONL_PATH}")
    print(f"  SQLite     : {DB_PATH}")
    print(f"  Cost rates : ${COST_PER_INPUT_TOKEN}/in-tok  ${COST_PER_OUTPUT_TOKEN}/out-tok")
    print(f"  Weights    : {RELIABILITY_WEIGHTS}")
 
    # Run the full pipeline
    all_results = run_pipeline(TASKS)
 
    # Export raw results
    export_results_json(all_results, EXPORT_PATH)
    print(f"\n✅  Exported {len(all_results)} results → {EXPORT_PATH}")
 
    # Reporting
    print_per_run_reports(all_results)
    print_leaderboard(all_results)
    print_top_k(all_results, k=TOP_K)
 
    # Trace inspection (first run)
    print("\n── Raw trace inspection (run 0) ──────────────────────")
    inspect_traces(n=1)
 
    print("\n✅  Pipeline complete.")
 
 
if __name__ == "__main__":
    main()

# %%
# Version checks
try:
    import deepeval
    print(f"deepeval: {deepeval.__version__}")
except:
    print("deepeval not installed")

try:
    import ragas 
    print(f"ragas: {ragas.__version__}")
except:
    print("ragas not installed")

# %%
import importlib
try:
    importlib.reload(util)
except Exception as e:
    print(f"Could not reload util: {e}")

# %%
import util
print(f"util location: {util.__file__}")

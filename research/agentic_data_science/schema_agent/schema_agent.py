"""
Data Profiler Agent — modular implementation.

Main pipeline and CLI orchestration for end-to-end data profiling.

Usage:
    python schema_agent.py data.csv
    python schema_agent.py data.csv --model gpt-4o-mini --llm-scope nulls
    python schema_agent.py data.csv --metrics mean std min max --output-json out.json
    python schema_agent.py data.csv data2.csv --tags sales inventory

Import as:

import research.agentic_data_science.schema_agent.schema_agent as radsasag
"""

import argparse
import logging
import os
import sys
import typing
import pytz

import dotenv
import pandas as pd
import schema_agent_hllmcli as radsasah
import schema_agent_loader as radsasal
import schema_agent_report as radsasar
import schema_agent_stats as radsasas

import helpers.hdbg as hdbg
import helpers.hlogging as hloggin

# =============================================================================
# Configuration & Logging
# =============================================================================

dotenv.load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

# Use dassert to ensure the API key exists
hdbg.dassert(api_key, "OPENAI_API_KEY not found in environment.")

_LOG = hloggin.getLogger(__name__)
_LOG.setLevel(logging.DEBUG)

# Ensure sys is imported for the handler
console_handler = logging.StreamHandler(sys.stdout)
hloggin.set_v2_formatter(
    ch=console_handler,
    root_logger=_LOG,
    force_no_warning=False,
    force_print_format=False,
    force_verbose_format=True,
    report_memory_usage=True,
    report_cpu_usage=True,
)


# =============================================================================
# Pipeline
# =============================================================================


def run_pipeline(
    csv_paths: typing.List[str],
    tags: typing.Optional[typing.List[str]] = None,
    model: str = "gpt-4o",
    metrics: typing.Optional[typing.List[str]] = None,
    llm_scope: str = "all",
    output_json: str = "data_profile_report.json",
    output_md: str = "data_profile_summary.md",
    use_langchain: bool = False,
) -> typing.Tuple[typing.Dict[str, pd.DataFrame], typing.Dict[str, typing.Any]]:
    """
    Execute the full data profiling pipeline over one or more CSV files.
    """
    if tags is None:
        tags = [os.path.splitext(os.path.basename(p))[0] for p in csv_paths]

    # Use dassert_eq to check that the number of tags matches files
    hdbg.dassert_eq(
        len(tags), 
        len(csv_paths), 
        msg="Number of tags must match number of CSV paths"
    )

    # --- Load & type-coerce ---
    tag_to_df, cat_cols_map = radsasal.prepare_dataframes(csv_paths, tags)

    # Merge datetime metadata
    combined_for_dt = pd.concat(list(tag_to_df.values()), axis=0, ignore_index=True)
    _, datetime_meta = radsasal.infer_and_convert_datetime_columns(combined_for_dt)

    # --- Compute stats ---
    stats = radsasas.compute_llm_agent_stats(
        tag_to_df,
        categorical_cols_map=cat_cols_map,
        metrics=metrics,
    )
    stats["datetime_columns"] = datetime_meta

    # --- LLM scope ---
    combined_df = pd.concat(list(tag_to_df.values()), axis=0, ignore_index=True)
    columns_for_llm = radsasah._select_columns_for_llm(combined_df, scope=llm_scope)
    
    _LOG.info(
        "LLM will profile %d / %d columns (scope=%s).",
        len(columns_for_llm),
        len(combined_df.columns),
        llm_scope,
    )

    # --- LLM call ---
    if use_langchain:
        prompt_text = radsasah.build_llm_prompt(
            stats, columns_to_include=columns_for_llm
        )
        semantic_insights = radsasah.get_llm_semantic_insights_langchain(
            prompt_text, model=model
        )
    else:
        semantic_insights = radsasah.generate_hypotheses_via_cli(
            stats,
            model=model,
            columns_to_include=columns_for_llm,
        )

    # --- Build column profiles ---
    # Ensure tag_to_df is not empty before accessing
    hdbg.dassert(tag_to_df, "No dataframes were loaded.")
    primary_df = list(tag_to_df.values())[0]
    
    column_profiles = radsasar.build_column_profiles(
        df=primary_df,
        stats=stats,
        insights=semantic_insights,
    )

    # --- Export ---
    radsasar.merge_and_export_results(
        stats=stats,
        insights=semantic_insights,
        column_profiles=column_profiles,
        output_path=output_json,
    )
    radsasar.export_markdown_from_profiles(
        column_profiles,
        numeric_stats=stats.get("numeric_summary", {}),
        output_path=output_md,
    )

    return tag_to_df, stats

# =============================================================================
# CLI
# =============================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="schema_agent",
        description="Data Profiler Agent — statistical + LLM column profiling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Inputs ---
    parser.add_argument(
        "csv_paths",
        nargs="+",
        metavar="CSV",
        help="One or more CSV file paths to profile.",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        metavar="TAG",
        help="Human-readable tag for each CSV (must match number of csv_paths).",
    )
    # --- LLM options ---
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="LLM model for semantic analysis.",
    )
    parser.add_argument(
        "--llm-scope",
        choices=["all", "semantic", "nulls"],
        default="all",
        dest="llm_scope",
        help=(
            "Which columns to send to the LLM. "
            "'all'=every column, 'semantic'=non-numeric only, "
            "'nulls'=high-null columns only (saves cost)."
        ),
    )
    parser.add_argument(
        "--use-langchain",
        action="store_true",
        dest="use_langchain",
        help="Use LangChain pipeline instead of hllmcli for LLM calls.",
    )
    # --- Stat options ---
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=radsasas.VALID_METRICS,
        default=None,
        metavar="METRIC",
        help=(
            f"Numeric metrics to include in the summary. "
            f"Valid: {', '.join(radsasas.VALID_METRICS)}. "
            f"Default: {', '.join(radsasas.DEFAULT_METRICS)}."
        ),
    )
    # --- Output options ---
    parser.add_argument(
        "--output-json",
        default="data_profile_report.json",
        dest="output_json",
        metavar="PATH",
        help="Output path for the merged JSON report.",
    )
    parser.add_argument(
        "--output-md",
        default="data_profile_summary.md",
        dest="output_md",
        metavar="PATH",
        help="Output path for the Markdown summary.",
    )
    return parser


def main() -> None:
    """
    CLI entry point.

    Parses arguments and delegates to run_pipeline().
    """
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_pipeline(
        csv_paths=args.csv_paths,
        tags=args.tags,
        model=args.model,
        metrics=args.metrics,
        llm_scope=args.llm_scope,
        output_json=args.output_json,
        output_md=args.output_md,
        use_langchain=args.use_langchain,
    )


if __name__ == "__main__":
    main()

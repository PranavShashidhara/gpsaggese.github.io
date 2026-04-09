# Data Profiler Agent

Automated statistical profiling and LLM-powered semantic analysis for CSV datasets. Generates column-level insights including semantic classification, data quality assessment, and testable business hypotheses.

## Features

- **Temporal detection** — Auto-detects and converts date/datetime columns across multiple formats
- **Statistical profiling** — Computes numeric summaries, data quality metrics, and categorical distributions
- **LLM semantic analysis** — Infers column roles (ID, Feature, Target, Timestamp), semantic meaning, and hypotheses
- **Cost optimization** — Filter columns prior to LLM analysis to control token usage and API costs
- **Multi-format output** — JSON reports and Markdown summaries

## Setup

Navigate to the project directory:
```bash
> cd research/agentic_data_science/schema_agent
```

Install dependencies:
```bash
> pip install -r requirements.txt
```

Set your API key:
```bash
> export OPENAI_API_KEY=sk-...
```

Make the entry point executable:
```bash
> chmod +x schema_agent.py
```

## Module Structure

The agent is organized into six focused modules:

| Module | Responsibility |
|--------|----------------|
| `schema_agent_models.py` | Pydantic schemas for type-safe column and dataset insights |
| `schema_agent_loader.py` | CSV loading, type inference, and datetime detection |
| `schema_agent_stats.py` | Numeric summaries, quality reports, and categorical distributions |
| `schema_agent_llm.py` | Prompt construction, OpenAI/LangChain calls, and structured output parsing |
| `schema_agent_report.py` | Column profiles, JSON export, and Markdown export |
| `schema_agent.py` | Pipeline orchestration and CLI entry point |

## Usage

### Basic

```bash
> ./schema_agent.py data.csv
```

Produces two output files:

- `data_profile_report.json` — Machine-readable column profiles and statistics
- `data_profile_summary.md` — Human-readable summary table

### Advanced

```bash
# Profile multiple files with custom tags
> ./schema_agent.py dataset1.csv dataset2.csv --tags sales_2024 inv_q1

# Cost-optimized: analyze only high-null columns
> ./schema_agent.py data.csv --llm-scope nulls --model gpt-4o-mini

# Custom metrics and output path
> ./schema_agent.py data.csv --metrics mean std max --output-json my_report.json

# Use LangChain as the inference backend
> ./schema_agent.py data.csv --use-langchain
```

## Command-Line Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `csv_paths` | Required | One or more CSV file paths |
| `--tags` | File stems | Labels for each CSV (count must match `csv_paths`) |
| `--model` | `gpt-4o` | OpenAI model (`gpt-4o`, `gpt-4o-mini`, etc.) |
| `--llm-scope` | `all` | Column selection strategy: `all`, `semantic`, or `nulls` |
| `--metrics` | Subset | Numeric summary stats: `mean`, `std`, `min`, `25%`, `50%`, `75%`, `max` |
| `--use-langchain` | `false` | Use LangChain instead of the default inference client |
| `--output-json` | `data_profile_report.json` | Output path for the JSON report |
| `--output-md` | `data_profile_summary.md` | Output path for the Markdown summary |

## LLM Scoping

Control which columns are sent to the LLM to manage cost and latency:

| Scope | Behavior |
|-------|----------|
| `all` | Profiles every column — most comprehensive, highest cost |
| `semantic` | Profiles non-numeric columns only |
| `nulls` | Profiles only columns with >5% null values — most cost-efficient |

## Python API

### Full pipeline

```python
import research.agentic_data_science.schema_agent.schema_agent as agent

tag_to_df, stats = agent.run_pipeline(
    csv_paths=["data.csv"],
    model="gpt-4o-mini",
    llm_scope="semantic"
)
```

### Individual modules

Each module can be imported independently for exploratory use or testing:

```python
import research.agentic_data_science.schema_agent.schema_agent_loader as loader
import research.agentic_data_science.schema_agent.schema_agent_stats as stats
import research.agentic_data_science.schema_agent.schema_agent_llm as llm
import research.agentic_data_science.schema_agent.schema_agent_report as report
```

## Output Reference

### `data_profile_report.json`

Structured report containing per-column profiles, statistical summaries, and LLM-generated insights.

### `data_profile_summary.md`

Formatted Markdown table with columns: **Column · Meaning · Role · Quality · Hypotheses**

## Troubleshooting

**API key not set**

```bash
> export OPENAI_API_KEY=sk-...
```

**Validation or parsing errors**

Reduce the number of columns sent to the LLM:

```bash
> ./schema_agent.py data.csv --llm-scope nulls
> ./schema_agent.py data.csv --llm-scope semantic --model gpt-4o-mini
```

**No datetime columns detected**

Expected behavior — datetime detection is skipped automatically when no temporal columns are present in the dataset.
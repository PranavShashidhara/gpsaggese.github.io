# Data Profiler Agent

Automated statistical profiling and LLM-powered semantic analysis for CSV datasets. Generates column-level insights including semantic classification, data quality assessment, and testable business hypotheses.

## Key Features

- **Automatic temporal detection** — Identifies and converts date/datetime columns across multiple formats
- **Statistical profiling** — Computes numeric summaries, data quality metrics, and categorical distributions
- **LLM-powered semantic analysis** — Infers column roles (ID, Feature, Target, Timestamp), semantic meaning, and generates testable business hypotheses
- **Smart cost control** — Selectively analyze columns to optimize API usage and reduce costs
- **Flexible output formats** — Generate machine-readable JSON reports and human-friendly Markdown summaries

## Quick Start

### Installation

Navigate to the project directory and install dependencies:

```bash
cd research/agentic_data_science/schema_agent
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
chmod +x schema_agent.py
```

### Basic Usage

Profile a single CSV file:

```bash
./schema_agent.py data.csv
```

This generates two output files:
- **`data_profile_report.json`** — Complete statistical and semantic analysis
- **`data_profile_summary.md`** — Readable summary table with insights

### Advanced Usage

```bash
# Profile multiple files with custom labels
./schema_agent.py dataset1.csv dataset2.csv --tags sales_2024 inventory_q1

# Cost-optimized analysis (only high-null columns)
./schema_agent.py data.csv --llm-scope nulls --model gpt-4o-mini

# Custom metrics and output paths
./schema_agent.py data.csv --metrics mean std max --output-json my_report.json

# Use LangChain as the inference backend
./schema_agent.py data.csv --use-langchain
```

## Architecture

The agent consists of six focused modules working together:

| Module | Purpose |
|--------|---------|
| `schema_agent_models.py` | Type-safe Pydantic schemas for column profiles and dataset insights |
| `schema_agent_loader.py` | CSV loading, type inference, and datetime detection |
| `schema_agent_stats.py` | Numeric summaries, data quality metrics, and categorical distributions |
| `schema_agent_llm.py` | LLM integration for semantic analysis and hypothesis generation |
| `schema_agent_report.py` | Report generation in JSON and Markdown formats |
| `schema_agent.py` | Pipeline orchestration and command-line interface |

For detailed examples of individual module usage, see `schema_agent.example`. For end-to-end pipeline examples, see `schema_agent.API`.

## Command-Line Options

| Argument | Default | Description |
|----------|---------|-------------|
| `csv_paths` | Required | One or more CSV file paths to analyze |
| `--tags` | File stems | Custom labels for each CSV (must match number of files) |
| `--model` | `gpt-4o` | OpenAI model to use (`gpt-4o`, `gpt-4o-mini`, etc.) |
| `--llm-scope` | `all` | Strategy for column selection: `all`, `semantic`, or `nulls` |
| `--metrics` | Subset | Statistics to compute: `mean`, `std`, `min`, `25%`, `50%`, `75%`, `max` |
| `--use-langchain` | `false` | Use LangChain instead of default inference client |
| `--output-json` | `data_profile_report.json` | Path for JSON report output |
| `--output-md` | `data_profile_summary.md` | Path for Markdown summary output |

## Cost Optimization with LLM Scoping

The `--llm-scope` parameter controls which columns are sent to the LLM, helping you balance analysis depth with costs:

| Scope | What Gets Analyzed | Cost Level | Best For |
|-------|-------------------|-----------|----------|
| `all` | Every column | High | Complete dataset understanding |
| `semantic` | Non-numeric columns only | Medium | Text and categorical analysis |
| `nulls` | Columns with >5% null values | Low | Data quality issues only |

## Python API

### Run the full pipeline programmatically

```python
import research.agentic_data_science.schema_agent.schema_agent as agent

tag_to_df, stats = agent.run_pipeline(
    csv_paths=["data.csv"],
    model="gpt-4o-mini",
    llm_scope="semantic"
)
```

### Use individual modules independently

Each module can be imported and used separately for custom workflows:

```python
import research.agentic_data_science.schema_agent.schema_agent_loader as loader
import research.agentic_data_science.schema_agent.schema_agent_stats as stats
import research.agentic_data_science.schema_agent.schema_agent_llm as llm
import research.agentic_data_science.schema_agent.schema_agent_report as report
```

## Output Details

### `data_profile_report.json`

A structured JSON report containing:
- Per-column statistical profiles
- Data quality metrics
- LLM-generated semantic insights
- Column role classifications

### `data_profile_summary.md`

A formatted Markdown table with columns:
- **Column** — Column name
- **Meaning** — Inferred semantic description
- **Role** — Classified role (ID, Feature, Target, Timestamp)
- **Quality** — Data quality assessment
- **Hypotheses** — Generated business insights

## Troubleshooting

### API key not configured

Set your OpenAI API key:
```bash
export OPENAI_API_KEY=sk-...
```

### Validation or parsing errors on large datasets

Reduce the number of columns analyzed by the LLM:
```bash
./schema_agent.py data.csv --llm-scope nulls
./schema_agent.py data.csv --llm-scope semantic --model gpt-4o-mini
```

### No datetime columns detected

This is normal behavior — the agent automatically skips temporal detection when no date-like columns are present in the dataset.

## Next Steps

- Check out example notebooks for detailed workflows
- Integrate into your data science pipelines
- Extend with custom metrics or export formats
- Review individual module documentation for advanced use cases
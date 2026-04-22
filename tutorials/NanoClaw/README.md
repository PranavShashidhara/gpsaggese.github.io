# Nanoclaw Tutorial - Agentic EDA System Using MCP Tools

An end-to-end exploratory data analysis system powered by a Claude agent (NanoClaw) and a custom MCP server (`hpandas`). Instead of writing pandas code manually, you describe your analysis in plain English — the agent translates it into tool calls against your dataset in real time.

## How It Works

A Claude agent runs inside a NanoClaw container. When you describe an analysis task, the agent issues structured tool calls over JSON-RPC to a Python MCP server on your host machine. That server executes pandas operations and streams results back to the agent — no manual scripting required.

```
You (natural language prompt)
        │
        ▼
Claude Agent (NanoClaw container)
        │  tool call
        ▼
MCP Client (NanoClaw)
        │  JSON-RPC
        ▼
hpandas MCP Server (host Python process)
        │
        ▼
pandas → results returned to agent
```

## Prerequisites

- GitHub CLI (`gh`) or `git`
- Python 3.x
- Docker or Apple Container runtime
- Claude Code (`claude` CLI)

## 1. Clone the Repository

**With GitHub CLI (recommended):**
```bash
gh repo fork qwibitai/nanoclaw --clone
cd nanoclaw
```

**With Git:**
```bash
git clone https://github.com/qwibitai/nanoclaw.git
cd nanoclaw
```

## 2. Install Dependencies & Bootstrap

```bash
claude
```

Inside Claude Code, run:
```
/setup
```

This will:
- Install Node dependencies
- Configure the container runtime (Docker / Apple Container)
- Initialize the MCP system
- Set up default channels

## 3. Register the `hpandas` MCP Server

Create or edit `.mcp.json` in the NanoClaw root:

```bash
nano .mcp.json
```

Add the following:

```json
{
  "mcpServers": {
    "hpandas": {
      "command": "python",
      "args": [
        "absolute_path/umd_msml610/tutorials/NanoClaw/hpandas_mcp_server.py"
      ]
    }
  }
}
```

> Update `absolute_path` to match your local filesystem.

## 4. Verify the MCP Server

Before starting NanoClaw, confirm the server runs without errors:

```bash
python umd_msml610/tutorials/NanoClaw/hpandas_mcp_server.py
```

A healthy server will stay running without crashes or JSON-RPC errors.

## 5. Generate the Dataset

```bash
cd umd_msml610/tutorials/NanoClaw
python dataset_generator.py
```

This produces `dummy_users.csv` in the same directory. Keep the file here — the MCP server expects to find it at this path.

## 6. Start NanoClaw

```bash
cd nanoclaw
claude
```

Then inside Claude Code:
```
/setup
```

NanoClaw will:
- Read `.mcp.json`
- Spawn the MCP server process
- Register `hpandas` tools under the `mcp__hpandas__*` namespace

## 7. Verify MCP Is Connected

Inside Claude Code, prompt:
```
Load dummy_users.csv and show the first 10 rows.
```

You should see tool calls like:
```
mcp__hpandas__read_csv
mcp__hpandas__df_to_str
```

## 8. Run EDA — Natural Language Prompts

Once connected, interact with your dataset entirely in natural language:

| Task | Prompt |
|||
| Overview | `"Describe the dataset"` |
| Data quality | `"Find missing values"` |
| Filtering | `"Show users with income > 50k"` |
| Aggregation | `"Average spend score by country"` |

## MCP Tools Reference

All tools are exposed under the `mcp__hpandas__` namespace. DataFrames are passed between tools as JSON strings using the internal `_df_to_json` / `_df_from_json` format (`records` + `index` + `shape` + `columns`).

### I/O

| Tool | Description |
|||
| `read_csv` | Read a CSV (or `.gz` / `.zip`) file from disk into a DataFrame |
| `read_parquet` | Read a Parquet file from disk into a DataFrame |
| `write_csv` | Write a DataFrame to a CSV file |
| `write_parquet` | Write a DataFrame to a Parquet file |
| `str_to_df` | Parse a raw CSV string into a DataFrame |

### Display

| Tool | Description |
|||
| `get_df_signature` | Compact shape + head/tail summary of a DataFrame |
| `convert_df_to_json_string` | Pretty-printed JSON showing head and tail rows |
| `df_to_str` | Human-readable string representation (head + tail) |

### Cleaning

| Tool | Description |
|||
| `drop_duplicates` | Remove duplicate rows, optionally considering the index |
| `dropna` | Drop rows or columns containing NaN values |
| `drop_axis_with_all_nans` | Remove rows and/or columns that are entirely NaN |
| `impute_nans` | Replace string literal `"nan"` entries in a column with a given value |
| `remove_outliers` | Clip values outside a quantile range to NaN |
| `remove_columns` | Drop specified columns from a DataFrame |

### Analysis

| Tool | Description |
|||
| `describe_df` | Descriptive statistics (wraps `df.describe()`) |
| `print_column_variability` | Unique value counts and coefficient of variation per column |
| `rolling_corr_over_time` | Exponentially-weighted rolling correlation matrix over a time index |

### Filtering & Transformation

| Tool | Description |
|||
| `filter_df` | Keep or drop rows matching specific values in a column |
| `head` | Return the first N rows |
| `subset_df` | Return a random sample of N rows |
| `add_pct` | Add a percentage column (`col / total_col * 100`) |
| `resample_df` | Resample a time-indexed DataFrame to a new frequency (mean) |
| `trim_df_by_time_period` | Slice a DataFrame to a timestamp range |
| `find_gaps_in_time_series` | Identify missing timestamps in a regularly-spaced time series |
| `merge_dfs` | Merge two DataFrames (wraps `pd.merge`) |

### Type Conversion

| Tool | Description |
|||
| `infer_column_types` | Detect whether each column is bool / numeric / string |
| `convert_df_types` | Auto-convert each column to its detected type |
| `convert_col_to_int` | Cast a single column to `int64` |
| `to_series` | Convert a single-column DataFrame to a Series |

### Comparison

| Tool | Description |
|||
| `compare_dfs` | Element-wise diff (absolute or % change) between two DataFrames |
| `compare_nans_in_dataframes` | Highlight positions where NaN status differs between two DataFrames |
| `find_common_columns` | Report columns shared across multiple DataFrames |

### Validation

| Tool | Description |
|||
| `check_index_is_datetime` | Assert the DataFrame index is a `DatetimeIndex` |
| `resolve_column_names` | Validate and resolve a column specification to a concrete list |

### Check Summary (Audit Log)

A lightweight session-based reporting system for logging pass/fail checks during analysis.

| Tool | Description |
|||
| `check_summary_create` | Start a new named check session |
| `check_summary_add` | Append a pass/fail check result to a session |
| `check_summary_report` | Print a formatted summary table for a session |

### MultiIndex

| Tool | Description |
|||
| `multiindex_df_info` | Return shape, level values, and time range metadata for a 2-level MultiIndex DataFrame |

## Deployment Layout

NanoClaw and the MCP server are **separate processes** — they communicate over JSON-RPC, not shared memory or a shared container.

| Component | Location |
|||
| NanoClaw agent | Docker / Apple Container |
| `hpandas` MCP server | Host machine (Python process) |
| Dataset (`dummy_users.csv`) | Local filesystem |

## Troubleshooting

**Error: `Invalid JSON: EOF while parsing` / `Internal Server Error`**

This means the MCP server either crashed mid-response or wrote non-JSON output to stdout, corrupting the JSON-RPC stream.

**Fix:**
- Remove all `print()` debug statements from the MCP server
- Ensure the server outputs **only** valid JSON-RPC messages to stdout
- Run the server standalone first to confirm it stays healthy

## Quick Demo Flow

```bash
gh repo fork qwibitai/nanoclaw --clone
cd nanoclaw
claude
# → /setup

# Inside Claude Code:
# "Load dummy_users.csv and analyze it using hpandas MCP server"
```
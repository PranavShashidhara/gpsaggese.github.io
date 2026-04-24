---
title: "NanoClaw: Agentic EDA with MCP Tools in 60 Minutes"
authors:
  - PranavShashidhara
  - gpsaggese
date: 2026-04-24
description:
categories:
  - AI Research
  - Software Engineering
---

TL;DR: Learn how to build an agentic exploratory data analysis (EDA) system
using Claude and a custom MCP server — no manual pandas scripting required.

<!-- more -->

## Tutorial in 30 Seconds

NanoClaw is an end-to-end EDA system powered by a Claude agent and a custom
Python MCP server called `hpandas`. Instead of writing pandas code manually,
you describe your analysis in plain English — the agent translates it into
structured tool calls against your dataset in real time.

Key capabilities:

- **Natural language EDA**: Describe tasks like filtering, aggregation, or
  missing value detection — Claude handles the rest
- **MCP tool architecture**: All pandas operations are exposed as structured
  JSON-RPC tools under the `mcp__hpandas__*` namespace
- **Modular design**: The Claude agent and MCP server run as separate processes,
  communicating over JSON-RPC — not shared memory
- **Extensible toolset**: 30+ tools covering I/O, cleaning, filtering,
  transformation, type conversion, and validation

This tutorial's goal is to show you in 60 minutes:

- How to wire a Claude agent (running inside a NanoClaw container) to a custom
  Python MCP server on your host machine
- Concrete examples of natural language EDA prompts against a real dataset

## Official References

- [NanoClaw GitHub repo](https://github.com/qwibitai/nanoclaw)

## Tutorial Content

This tutorial includes all the code, MCP server, and dataset generator in
`tutorials/NanoClaw`:

- [`README.md`](../../../../tutorials/NanoClaw/README.md):
  Full setup instructions for the NanoClaw environment
- [`hpandas_mcp_server.py`](../../../../tutorials/NanoClaw/hpandas_mcp_server.py): The custom
  MCP server that exposes pandas operations as JSON-RPC tools
- [`dataset_generator.py`](../../../../tutorials/NanoClaw/dataset_generator.py): Generates
  `dummy_users.csv` — the sample dataset used throughout the tutorial
- `.mcp.json` configuration: Registers `hpandas` with the NanoClaw agent so
  tools are available under the `mcp__hpandas__*` namespace

### System Architecture

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

### Example EDA Prompts

Once the system is running, interact with your dataset entirely in natural
language:

| Task | Prompt |
|---|---|
| Overview | `"Describe the dataset"` |
| Data quality | `"Find missing values"` |
| Filtering | `"Show users with income > 50k"` |
| Aggregation | `"Average spend score by country"` |

### MCP Tool Categories

The `hpandas` server exposes 30+ tools across seven categories:

- **I/O**: `read_csv`, `read_parquet`, `write_csv`, `write_parquet`
- **Display**: `df_to_str`, `get_df_signature`, `convert_df_to_json_string`
- **Cleaning**: `drop_duplicates`, `dropna`, `impute_nans`, `remove_outliers`
- **Analysis**: `describe_df`, `print_column_variability`, `rolling_corr_over_time`
- **Filtering & Transformation**: `filter_df`, `merge_dfs`, `resample_df`, `add_pct`
- **Type Conversion**: `infer_column_types`, `convert_df_types`, `convert_col_to_int`
- **Validation**: `check_index_is_datetime`, `resolve_column_names`

## Quick Demo Flow

```bash
gh repo fork qwibitai/nanoclaw --clone
cd nanoclaw
claude
# → /setup

# Generate the dataset
cd umd_msml610/tutorials/NanoClaw
python dataset_generator.py

# Inside Claude Code:
# "Load dummy_users.csv and analyze it using hpandas MCP server"
```
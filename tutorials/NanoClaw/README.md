# hpandas MCP Server

## Overview

This project exposes a collection of pandas-based data processing utilities as **MCP (Model Context Protocol) tools**. The goal is to make structured data operations accessible to LLM agents in a controlled, observable, and reusable way.

This repository is part of an ongoing effort to integrate **NanoClaw agents with external MCP tool servers**, enabling agents to reason about data while delegating execution to a well-defined tool layer.

## Motivation

Large language models are effective at reasoning about tasks, but they should not directly execute arbitrary code or access raw datasets. This project separates concerns:

- The **agent (NanoClaw / Claude)** decides what operations to perform  
- The **MCP server (this project)** executes those operations  
- The **container environment (Docker / NanoClaw runtime)** enforces isolation  

This architecture improves safety, reproducibility, and transparency while enabling complex workflows over structured data.

## What This Server Provides

The server exposes a wide range of DataFrame operations as MCP tools, including:

### Data Loading and I/O
- `read_csv`, `read_parquet`
- `write_csv`, `write_parquet`

### Cleaning and Transformation
- `dropna`, `drop_duplicates`, `remove_outliers`
- `filter_df`, `merge_dfs`, `trim_df`, `resample_df`

### Analysis
- `describe_df`, `rolling_corr_over_time`
- `print_column_variability`

### Validation and Checks
- Index and schema validation tools
- DataFrame comparison utilities

### Utilities
- DataFrame ↔ JSON conversion
- Sampling, formatting, and column resolution

All DataFrames are passed as JSON strings to ensure compatibility with LLM tool interfaces.

## Architecture
NanoClaw Agent (LLM planner)  
↓  
MCP Client (tool invocation layer)  
↓  
hpandas_mcp_server (this project)  
↓  
pandas / numpy execution  

The agent does not directly manipulate data. Instead, it issues structured tool calls, which are executed by the MCP server and returned as structured outputs.

## Running the Server

```bash
python hpandas_mcp_server.py
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # API usage Notebook 
# - This notebook shows the implementation of each function from the respective libraries.

# %%
# %load_ext autoreload
# %autoreload 2

import dotenv
import os
import pandas as pd
import numpy as np

# Load environment variables (ensure OPENAI_API_KEY is set in your .env)
dotenv.load_dotenv()

# Import the schema agent modules
import research.agentic_data_science.schema_agent.schema_agent_loader as radsasal
import research.agentic_data_science.schema_agent.schema_agent_stats as radsasas
import research.agentic_data_science.schema_agent.schema_agent_hllmcli as radsasah
import research.agentic_data_science.schema_agent.schema_agent_report as radsasar

# %% [markdown]
# ## 1. Create dummy Dataset

# %%
# 1. Create a dummy dataset
np.random.seed(42)
num_rows = 100

dummy_data = pd.DataFrame({
    "employee_id": range(1000, 1000 + num_rows),
    "department": np.random.choice(["Engineering", "Sales", "HR", "Marketing"], num_rows),
    "salary": np.random.normal(85000, 20000, num_rows),
    "satisfaction_score": np.random.uniform(1.0, 5.0, num_rows),
    "hire_date": pd.date_range(start="2018-01-01", periods=num_rows, freq="W").astype(str),
    "notes": ["Good performance"] * 50 + [None] * 50  # 50% nulls
})

# Inject some missing values into salary
dummy_data.loc[10:20, "salary"] = np.nan

# Save to CSV
csv_path = "dummy_employees.csv"
dummy_data.to_csv(csv_path, index=False)
print(f"Created dummy dataset at: {csv_path}")
dummy_data.head()

csv_paths = [csv_path]
tags = ["dummy_employees"]

# %% [markdown]
# ## 2. Load and Infer datatypes from the columns

# %%
# 1. Load and prepare DataFrames - now receiving 3 variables
tag_to_df, cat_cols_map, datetime_meta = radsasal.prepare_dataframes(csv_paths, tags)

print("--- Loaded DataFrames ---")
# The index will now show as a DatetimeIndex instead of a RangeIndex
print(tag_to_df["dummy_employees"].info())

# 2. Combine DataFrames while preserving the index
# We do NOT use ignore_index=True here because we want to keep the DatetimeIndex 
# we just created in the loader.
updated_df = pd.concat(list(tag_to_df.values()), axis=0)

print("\n--- Datetime Inference Metadata ---")
# This will now correctly show your temporal column info
print(datetime_meta)

# %% [markdown]
# ## 3. Statistical Profiling

# %%
# We pass the metadata we just generated into the stats function
stats = radsasas.compute_llm_agent_stats(
    tag_to_df=tag_to_df,
    categorical_cols_map=cat_cols_map,
    metrics=["mean", "std", "min", "max"]
)

# Manually ensure the datetime_columns key is populated for the LLM
stats["datetime_columns"] = datetime_meta

print("\n--- Stats Computation Complete ---")
print(f"Calculated stats for tags: {list(stats['numeric_summary'].keys())}")

# %% [markdown]
# ## 4. Call LLM for column type inferencing

# %%
# 1. Select columns (e.g., let's just send everything)
columns_for_llm = radsasah._select_columns_for_llm(updated_df, scope="all")
print(f"Selected columns for LLM: {columns_for_llm}\n")

# 2. Build the exact prompt string that goes to the LLM
prompt_text = radsasah.build_llm_prompt(stats, columns_to_include=columns_for_llm)
print("--- LLM Prompt Snippet ---")
print(prompt_text[:500] + "\n...\n")

# 3. Call the LLM to generate hypotheses (using gpt-4o as default)
# If you don't have an API key configured, you can mock this response by creating a static dict.
try:
    semantic_insights = radsasah.generate_hypotheses_via_cli(
        stats=stats,
        model="gpt-4o",
        columns_to_include=columns_for_llm
    )
    print("--- LLM Insights Retrieved Successfully ---")
except Exception as e:
    print(f"LLM call failed (Check API key): {e}")
    semantic_insights = {"columns": {}} # Fallback empty dict

# %% [markdown]
# ## 5. Export to JSON and Markdown

# %%
# 1. Build structured column profiles
primary_df = list(tag_to_df.values())[0]
column_profiles = radsasar.build_column_profiles(
    df=primary_df,
    stats=stats,
    insights=semantic_insights
)

# 2. Export to JSON
json_out = "dummy_profile_report.json"
radsasar.merge_and_export_results(
    stats=stats,
    insights=semantic_insights,
    column_profiles=column_profiles,
    output_path=json_out
)

# 3. Export to Markdown
md_out = "dummy_profile_summary.md"
radsasar.export_markdown_from_profiles(
    column_profiles=column_profiles,
    numeric_stats=stats.get("numeric_summary", {}),
    output_path=md_out
)

print(f"\nPipeline complete! Check your directory for:")
print(f"1. {json_out}")
print(f"2. {md_out}")

# Clean up dummy CSV if desired
# os.remove(csv_path)

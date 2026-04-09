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
# # Schema Parser example 
# - This implementation in the notebook utilizes a suite of pre-existing functions to parse a single Excel (or CSV) file, automatically inferring data types and capturing temporal metadata for downstream analysis.

# %%
# %load_ext autoreload
# %autoreload 2
import research.agentic_data_science.schema_agent.schema_agent as radsasag

# Now run the pipeline
csv_files = ["global_ecommerce_forecasting.csv"]
tags = ["ecommerce_data"]

tag_to_df, stats = radsasag.run_pipeline(
    csv_paths=csv_files,
    tags=tags,
    model="gpt-4o",
    llm_scope="semantic"
)

display(tag_to_df["ecommerce_data"].head())

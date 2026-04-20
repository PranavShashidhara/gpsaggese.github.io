#!/usr/bin/env python3
"""
hpandas MCP Server
==================
Exposes the hpandas helper library as MCP tools so any MCP client
(Claude Desktop, Claude Code, nano-claw, etc.) can call them.

Run as a standalone stdio server:
    python hpandas_mcp_server.py

Or register it in Claude Desktop's config:
    {
      "mcpServers": {
        "hpandas": {
          "command": "python",
          "args": ["/absolute/path/to/hpandas_mcp_server.py"]
        }
      }
    }

All DataFrames are exchanged as JSON strings (orient="records") with an
optional "index" key for the row labels.  Timestamps should be ISO-8601
strings.

Import as:
    import hpandas_mcp_server
"""

import io
import json
import traceback
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Helpers – JSON ↔ DataFrame
# ---------------------------------------------------------------------------

def _df_from_json(payload: str) -> pd.DataFrame:
    """
    Deserialise a JSON string into a DataFrame.

    Accepts two shapes:
    * ``{"records": [...], "index": [...]}``  – records + explicit row index
    * A bare JSON array ``[{...}, ...]``       – records only (default RangeIndex)
    """
    data = json.loads(payload)
    if isinstance(data, dict) and "records" in data:
        df = pd.DataFrame(data["records"])
        if "index" in data:
            df.index = data["index"]
    else:
        df = pd.DataFrame(data)
    # Try to parse datetime columns / index.
    for col in df.columns:
        if "time" in str(col).lower() or "date" in str(col).lower():
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
    if isinstance(df.index, pd.Index) and df.index.dtype == object:
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df


def _df_to_json(df: pd.DataFrame) -> str:
    """Serialise a DataFrame to a JSON string (records + index)."""
    return json.dumps(
        {
            "records": json.loads(
                df.to_json(orient="records", date_format="iso", default_handler=str)
            ),
            "index": [str(i) for i in df.index],
            "shape": list(df.shape),
            "columns": list(df.columns),
        },
        indent=2,
    )


def _srs_to_json(srs: pd.Series) -> str:
    return json.dumps(
        {
            "values": json.loads(srs.to_json(date_format="iso", default_handler=str)),
            "name": srs.name,
            "dtype": str(srs.dtype),
        },
        indent=2,
    )


def _safe(fn, *args, **kwargs):
    """Call *fn* and return (result, error_str) tuple."""
    try:
        return fn(*args, **kwargs), None
    except Exception:
        return None, traceback.format_exc()


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "hpandas",
    instructions=(
        "Tools that wrap the hpandas helper library for pandas DataFrames. "
        "DataFrames are passed / returned as JSON strings produced by "
        "_df_to_json / _df_from_json helpers inside this server."
    ),
)


# ===========================================================================
# ── DISPLAY ─────────────────────────────────────────────────────────────────
# ===========================================================================

@mcp.tool()
def get_df_signature(df_json: str, num_rows: int = 6) -> str:
    """
    Return a compact signature string for a DataFrame: shape + head + tail rows.

    :param df_json: DataFrame serialised with _df_to_json.
    :param num_rows: total number of sample rows to show.
    :return: human-readable signature string.
    """
    df = _df_from_json(df_json)
    txt: List[str] = [f"df.shape={df.shape}"]
    with pd.option_context("display.max_colwidth", int(1e6), "display.max_columns", None):
        if len(df) > num_rows:
            txt.append(f"df.head=\n{df.head(num_rows // 2)}")
            txt.append(f"df.tail=\n{df.tail(num_rows // 2)}")
        else:
            txt.append(f"df.full=\n{df}")
    return "\n".join(txt)


@mcp.tool()
def convert_df_to_json_string(
    df_json: str,
    n_head: Optional[int] = 10,
    n_tail: Optional[int] = 10,
) -> str:
    """
    Convert a DataFrame to a pretty-printed JSON string showing head and tail.

    :param df_json: DataFrame serialised with _df_to_json.
    :param n_head: number of top rows (None = all rows).
    :param n_tail: number of bottom rows (None = skip tail).
    :return: formatted JSON string.
    """
    df = _df_from_json(df_json)
    shape = f"original shape={df.shape}"
    head_df = df.head(n_head) if n_head is not None else df
    head_json = head_df.to_json(orient="index", force_ascii=False, indent=4,
                                default_handler=str, date_format="iso", date_unit="s")
    if n_tail is not None:
        tail_json = df.tail(n_tail).to_json(
            orient="index", force_ascii=False, indent=4,
            default_handler=str, date_format="iso", date_unit="s")
    else:
        tail_json = ""
    return "\n".join([shape, "Head:", head_json, "Tail:", tail_json])


# ===========================================================================
# ── CLEAN ───────────────────────────────────────────────────────────────────
# ===========================================================================

@mcp.tool()
def drop_duplicates(
    df_json: str,
    use_index: bool = False,
    column_subset: Optional[List[str]] = None,
    keep: str = "first",
) -> str:
    """
    Drop duplicate rows from a DataFrame.

    :param df_json: DataFrame serialised with _df_to_json.
    :param use_index: if True, the index is included when detecting duplicates.
    :param column_subset: columns to consider; None = all columns.
    :param keep: which duplicate to keep – "first", "last", or False.
    :return: deduplicated DataFrame as JSON.
    """
    df = _df_from_json(df_json)
    cols = column_subset or df.columns.tolist()
    if use_index:
        tmp = "__idx_tmp__"
        df[tmp] = df.index
        cols = [tmp] + cols
        df = df.drop_duplicates(subset=cols, keep=keep)
        df = df.drop(columns=[tmp])
    else:
        df = df.drop_duplicates(subset=cols, keep=keep)
    return _df_to_json(df)


@mcp.tool()
def dropna(
    df_json: str,
    drop_infs: bool = False,
    axis: int = 0,
    how: str = "any",
    subset: Optional[List[str]] = None,
) -> str:
    """
    Drop rows (or columns) that contain NaN values.

    :param df_json: DataFrame serialised with _df_to_json.
    :param drop_infs: if True, treat ±inf as NaN before dropping.
    :param axis: 0 = drop rows, 1 = drop columns.
    :param how: "any" or "all".
    :param subset: columns to check (only used when axis=0).
    :return: cleaned DataFrame as JSON.
    """
    df = _df_from_json(df_json)
    if drop_infs:
        df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=axis, how=how, subset=subset)
    return _df_to_json(df)


@mcp.tool()
def drop_axis_with_all_nans(
    df_json: str,
    drop_rows: bool = True,
    drop_columns: bool = False,
    drop_infs: bool = False,
) -> str:
    """
    Remove rows and/or columns that are entirely NaN.

    :param df_json: DataFrame serialised with _df_to_json.
    :param drop_rows: remove all-NaN rows.
    :param drop_columns: remove all-NaN columns.
    :param drop_infs: treat ±inf as NaN before checking.
    :return: cleaned DataFrame as JSON.
    """
    df = _df_from_json(df_json)
    if drop_infs:
        df = df.replace([np.inf, -np.inf], np.nan)
    if drop_columns:
        df = df.dropna(axis=1, how="all")
    if drop_rows:
        df = df.dropna(axis=0, how="all")
    return _df_to_json(df)


@mcp.tool()
def impute_nans(df_json: str, column: str, value: Any) -> str:
    """
    Replace string literal "nan" values in a column with a specified value.

    :param df_json: DataFrame serialised with _df_to_json.
    :param column: name of the column to fix.
    :param value: replacement value for "nan" entries.
    :return: updated DataFrame as JSON.
    """
    df = _df_from_json(df_json)
    df[column] = df[column].astype(str)
    mask = df[column] == "nan"
    df[column] = np.where(mask, value, df[column])
    return _df_to_json(df)


@mcp.tool()
def remove_outliers(
    df_json: str,
    lower_quantile: float,
    column_set: Optional[List[str]] = None,
    upper_quantile: Optional[float] = None,
) -> str:
    """
    Clip values outside the given quantile range to NaN.

    :param df_json: DataFrame serialised with _df_to_json.
    :param lower_quantile: lower quantile threshold in [0, 1].
    :param column_set: columns to apply the filter to; None = all numeric.
    :param upper_quantile: upper quantile; defaults to 1 - lower_quantile.
    :return: DataFrame with outliers replaced by NaN, as JSON.
    """
    df = _df_from_json(df_json)
    if upper_quantile is None:
        upper_quantile = 1.0 - lower_quantile
    cols = column_set or df.select_dtypes(include=[np.number]).columns.tolist()
    for col in cols:
        lo = df[col].quantile(lower_quantile)
        hi = df[col].quantile(upper_quantile)
        df[col] = df[col].where((df[col] >= lo) & (df[col] <= hi), np.nan)
    return _df_to_json(df)


# ===========================================================================
# ── COMPARE ─────────────────────────────────────────────────────────────────
# ===========================================================================

@mcp.tool()
def compare_dfs(
    df1_json: str,
    df2_json: str,
    row_mode: str = "inner",
    column_mode: str = "inner",
    diff_mode: str = "diff",
) -> str:
    """
    Compare two DataFrames element-wise and return a diff DataFrame.

    :param df1_json: first DataFrame as JSON.
    :param df2_json: second DataFrame as JSON.
    :param row_mode: "equal" (must share index) or "inner" (intersect index).
    :param column_mode: "equal" (must share columns) or "inner" (intersect columns).
    :param diff_mode: "diff" (absolute) or "pct_change" (percentage).
    :return: diff DataFrame as JSON.
    """
    df1 = _df_from_json(df1_json)
    df2 = _df_from_json(df2_json)
    # Align rows.
    if row_mode == "inner":
        common = df1.index.intersection(df2.index)
        df1, df2 = df1.loc[common], df2.loc[common]
    # Align columns.
    if column_mode == "inner":
        common_cols = sorted(set(df1.columns) & set(df2.columns))
        df1, df2 = df1[common_cols], df2[common_cols]
    # Select only numeric columns.
    num_cols = df1.select_dtypes(include=[np.number]).columns.tolist()
    df1, df2 = df1[num_cols], df2[num_cols]
    if diff_mode == "pct_change":
        diff = 100 * (df1 - df2) / df2.abs()
        diff = diff.replace([np.inf, -np.inf], np.nan)
    else:
        diff = df1 - df2
        diff = diff.replace([np.inf, -np.inf], np.nan)
    diff = diff.add_suffix(f".{diff_mode}")
    return _df_to_json(diff)


@mcp.tool()
def compare_nans_in_dataframes(df1_json: str, df2_json: str) -> str:
    """
    Return a DataFrame highlighting positions where NaN status differs.

    :param df1_json: first DataFrame as JSON.
    :param df2_json: second DataFrame as JSON.
    :return: DataFrame showing NaN mismatches, as JSON.
    """
    df1 = _df_from_json(df1_json)
    df2 = _df_from_json(df2_json)
    common = df1.index.intersection(df2.index)
    common_cols = sorted(set(df1.columns) & set(df2.columns))
    df1, df2 = df1.loc[common, common_cols], df2.loc[common, common_cols]
    mask = (df1.isna() & ~df2.isna()) | (~df1.isna() & df2.isna())
    result = df1[mask].compare(df2[mask], result_names=("df1", "df2"))
    return _df_to_json(result)


@mcp.tool()
def find_common_columns(names_json: str, dfs_json: List[str]) -> str:
    """
    Report columns shared between every pair of DataFrames.

    :param names_json: JSON array of string labels, one per DataFrame.
    :param dfs_json: list of DataFrames serialised with _df_to_json.
    :return: summary DataFrame as JSON.
    """
    names = json.loads(names_json)
    dfs = [_df_from_json(j) for j in dfs_json]
    rows = []
    for i in range(len(dfs)):
        for j in range(i + 1, len(dfs)):
            common = [c for c in dfs[i].columns if c in dfs[j].columns]
            rows.append({
                "table1": names[i], "num_cols1": len(dfs[i].columns),
                "table2": names[j], "num_cols2": len(dfs[j].columns),
                "num_common": len(common), "common_cols": ", ".join(common),
            })
    return _df_to_json(pd.DataFrame(rows))


# ===========================================================================
# ── CONVERSION ──────────────────────────────────────────────────────────────
# ===========================================================================

@mcp.tool()
def to_series(df_json: str, series_dtype: str = "float64") -> str:
    """
    Convert a single-column DataFrame into a Series.

    :param df_json: single-column DataFrame as JSON.
    :param series_dtype: dtype to use if the DataFrame is empty.
    :return: Series as JSON.
    """
    df = _df_from_json(df_json)
    if df.shape[1] != 1:
        raise ValueError(f"Expected a single-column DataFrame, got {df.shape[1]} columns.")
    if df.empty:
        return _srs_to_json(pd.Series(dtype=series_dtype))
    if df.shape[0] > 1:
        srs = df.squeeze()
    else:
        srs = pd.Series(df.iloc[0, 0], index=[df.index.values[0]])
        srs.name = df.index.name
    return _srs_to_json(srs)


@mcp.tool()
def infer_column_types(df_json: str) -> str:
    """
    Infer the predominant type (bool / numeric / string) for every column.

    :param df_json: DataFrame as JSON.
    :return: JSON object mapping column name → type string.
    """
    df = _df_from_json(df_json)
    result: Dict[str, str] = {}
    for col in df.columns:
        is_bool = float(df[col].map(lambda x: isinstance(x, bool)).mean())
        is_num = float(pd.to_numeric(df[col], errors="coerce").notna().mean())
        is_str = float(df[col].map(lambda x: isinstance(x, str)).mean())
        if is_bool >= is_num and is_bool != 0:
            result[col] = "is_bool"
        elif is_num >= is_str and is_num != 0:
            result[col] = "is_numeric"
        else:
            result[col] = "is_string"
    return json.dumps(result, indent=2)


@mcp.tool()
def convert_df_types(df_json: str) -> str:
    """
    Convert every column to its detected predominant type (bool / numeric / string).

    :param df_json: DataFrame as JSON.
    :return: type-converted DataFrame as JSON.
    """
    df = _df_from_json(df_json)
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        s = df[col]
        is_bool = float(s.map(lambda x: isinstance(x, bool)).mean())
        is_num = float(pd.to_numeric(s, errors="coerce").notna().mean())
        is_str = float(s.map(lambda x: isinstance(x, str)).mean())
        if is_bool >= is_num and is_bool != 0:
            out[col] = s.map(lambda x: True if x in ["True", 1, "1", "true", True]
                             else (False if x in [0, "0", "False", False, "false"] else None))
        elif is_num >= is_str and is_num != 0:
            out[col] = pd.to_numeric(s, errors="coerce")
        else:
            out[col] = s.astype(str)
    return _df_to_json(out)


@mcp.tool()
def convert_col_to_int(df_json: str, col: str) -> str:
    """
    Cast a single column to int64.

    :param df_json: DataFrame as JSON.
    :param col: name of the column to convert.
    :return: updated DataFrame as JSON.
    """
    df = _df_from_json(df_json)
    df[col] = df[col].astype("int64")
    return _df_to_json(df)


# ===========================================================================
# ── DASSERT (validation) ────────────────────────────────────────────────────
# ===========================================================================

@mcp.tool()
def check_index_is_datetime(df_json: str) -> Dict[str, Any]:
    """
    Check whether the DataFrame index is a DatetimeIndex.

    :param df_json: DataFrame as JSON.
    :return: {"is_datetime": bool, "index_type": str}.
    """
    df = _df_from_json(df_json)
    return {
        "is_datetime": isinstance(df.index, pd.DatetimeIndex),
        "index_type": type(df.index).__name__,
    }


@mcp.tool()
def check_unique_index(df_json: str) -> Dict[str, Any]:
    """
    Check whether the DataFrame index contains duplicates.

    :param df_json: DataFrame as JSON.
    :return: {"is_unique": bool, "num_duplicates": int, "duplicate_values": list}.
    """
    df = _df_from_json(df_json)
    dups = df.index[df.index.duplicated(keep=False)].tolist()
    return {
        "is_unique": df.index.is_unique,
        "num_duplicates": len(dups),
        "duplicate_values": [str(d) for d in dups[:20]],
    }


@mcp.tool()
def check_increasing_index(df_json: str) -> Dict[str, Any]:
    """
    Check whether the DataFrame index is monotonically increasing.

    :param df_json: DataFrame as JSON.
    :return: {"is_monotonic_increasing": bool, "is_strictly_increasing": bool}.
    """
    df = _df_from_json(df_json)
    return {
        "is_monotonic_increasing": bool(df.index.is_monotonic_increasing),
        "is_strictly_increasing": bool(df.index.is_monotonic_increasing and df.index.is_unique),
    }


@mcp.tool()
def check_axes_equal(df1_json: str, df2_json: str) -> Dict[str, Any]:
    """
    Check whether two DataFrames share identical indices and columns.

    :param df1_json: first DataFrame as JSON.
    :param df2_json: second DataFrame as JSON.
    :return: dict with "index_equal", "columns_equal" booleans and difference lists.
    """
    df1 = _df_from_json(df1_json)
    df2 = _df_from_json(df2_json)
    idx_eq = df1.index.equals(df2.index)
    col_eq = df1.columns.equals(df2.columns)
    return {
        "index_equal": idx_eq,
        "columns_equal": col_eq,
        "index_only_in_df1": [str(x) for x in df1.index.difference(df2.index)[:10]],
        "index_only_in_df2": [str(x) for x in df2.index.difference(df1.index)[:10]],
        "columns_only_in_df1": list(df1.columns.difference(df2.columns)),
        "columns_only_in_df2": list(df2.columns.difference(df1.columns)),
    }


@mcp.tool()
def check_series_dtype(series_json: str, expected_dtype: str) -> Dict[str, Any]:
    """
    Check whether a Series has the expected dtype.

    :param series_json: Series serialised by _srs_to_json.
    :param expected_dtype: dtype string to check against, e.g. "float64".
    :return: {"matches": bool, "actual_dtype": str}.
    """
    data = json.loads(series_json)
    return {
        "matches": data.get("dtype") == expected_dtype,
        "actual_dtype": data.get("dtype"),
        "expected_dtype": expected_dtype,
    }


# ===========================================================================
# ── TRANSFORM ───────────────────────────────────────────────────────────────
# ===========================================================================

@mcp.tool()
def trim_df(
    df_json: str,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    ts_col_name: Optional[str] = None,
    left_close: bool = True,
    right_close: bool = True,
) -> str:
    """
    Trim a DataFrame to a timestamp range.

    :param df_json: DataFrame as JSON.
    :param start_ts: ISO-8601 start timestamp; None = no lower bound.
    :param end_ts: ISO-8601 end timestamp; None = no upper bound.
    :param ts_col_name: column to filter on; None = use the index.
    :param left_close: include start_ts in the result.
    :param right_close: include end_ts in the result.
    :return: trimmed DataFrame as JSON.
    """
    df = _df_from_json(df_json)
    if ts_col_name is not None:
        series = pd.to_datetime(df[ts_col_name])
    else:
        series = pd.to_datetime(df.index.to_series())
    mask = pd.Series(True, index=df.index)
    if start_ts:
        ts = pd.Timestamp(start_ts)
        mask &= (series >= ts) if left_close else (series > ts)
    if end_ts:
        ts = pd.Timestamp(end_ts)
        mask &= (series <= ts) if right_close else (series < ts)
    return _df_to_json(df[mask])


@mcp.tool()
def resample_df(df_json: str, frequency: str) -> str:
    """
    Resample a time-indexed DataFrame to a new frequency (mean aggregation).

    :param df_json: DataFrame with a DatetimeIndex, as JSON.
    :param frequency: pandas frequency string, e.g. "1H", "D", "15T".
    :return: resampled DataFrame as JSON.
    """
    df = _df_from_json(df_json)
    df.index = pd.to_datetime(df.index)
    return _df_to_json(df.resample(frequency).mean())


@mcp.tool()
def merge_dfs(
    df1_json: str,
    df2_json: str,
    how: str = "outer",
    on: Optional[List[str]] = None,
    left_on: Optional[List[str]] = None,
    right_on: Optional[List[str]] = None,
    suffixes: Optional[List[str]] = None,
) -> str:
    """
    Merge two DataFrames (wrapper around pd.merge).

    :param df1_json: left DataFrame as JSON.
    :param df2_json: right DataFrame as JSON.
    :param how: join type – "inner", "outer", "left", "right".
    :param on: column(s) to join on (must exist in both).
    :param left_on: column(s) in the left DataFrame to join on.
    :param right_on: column(s) in the right DataFrame to join on.
    :param suffixes: list of two suffix strings for overlapping columns.
    :return: merged DataFrame as JSON.
    """
    df1 = _df_from_json(df1_json)
    df2 = _df_from_json(df2_json)
    sfx = tuple(suffixes) if suffixes else ("_x", "_y")
    merged = pd.merge(df1, df2, how=how, on=on, left_on=left_on,
                      right_on=right_on, suffixes=sfx)
    return _df_to_json(merged)


@mcp.tool()
def filter_df(
    df_json: str,
    filter_col: str,
    filter_values: List[Any],
    mode: str = "keep",
) -> str:
    """
    Filter a DataFrame by keeping or dropping rows matching specific values.

    :param df_json: DataFrame as JSON.
    :param filter_col: column whose values are tested.
    :param filter_values: list of values to match.
    :param mode: "keep" (rows matching filter_values) or "drop" (rows not matching).
    :return: filtered DataFrame as JSON.
    """
    df = _df_from_json(df_json)
    mask = df[filter_col].isin(filter_values)
    if mode == "drop":
        mask = ~mask
    return _df_to_json(df[mask])


@mcp.tool()
def remove_columns(df_json: str, columns: List[str]) -> str:
    """
    Drop specified columns from a DataFrame.

    :param df_json: DataFrame as JSON.
    :param columns: list of column names to remove.
    :return: DataFrame without the specified columns, as JSON.
    """
    df = _df_from_json(df_json)
    existing = [c for c in columns if c in df.columns]
    return _df_to_json(df.drop(columns=existing))


@mcp.tool()
def str_to_df(csv_string: str, sep: str = ",") -> str:
    """
    Parse a CSV string into a DataFrame.

    :param csv_string: raw CSV text.
    :param sep: column delimiter.
    :return: parsed DataFrame as JSON.
    """
    df = pd.read_csv(io.StringIO(csv_string), sep=sep)
    return _df_to_json(df)


@mcp.tool()
def head(df_json: str, nrows: int = 5) -> str:
    """
    Return the first *nrows* rows of a DataFrame.

    :param df_json: DataFrame as JSON.
    :param nrows: number of rows to return.
    :return: subset DataFrame as JSON.
    """
    df = _df_from_json(df_json)
    return _df_to_json(df.head(nrows))


@mcp.tool()
def subset_df(df_json: str, nrows: int, seed: int = 42) -> str:
    """
    Return a random sample of *nrows* rows.

    :param df_json: DataFrame as JSON.
    :param nrows: how many rows to sample.
    :param seed: random seed for reproducibility.
    :return: sampled DataFrame as JSON.
    """
    df = _df_from_json(df_json)
    n = min(nrows, len(df))
    return _df_to_json(df.sample(n, random_state=seed))


# ===========================================================================
# ── UTILS ───────────────────────────────────────────────────────────────────
# ===========================================================================

@mcp.tool()
def df_to_str(df_json: str, num_rows: int = 6) -> str:
    """
    Format a DataFrame as a readable string (head + tail).

    :param df_json: DataFrame as JSON.
    :param num_rows: total rows to show (split evenly head / tail).
    :return: string representation.
    """
    df = _df_from_json(df_json)
    with pd.option_context(
        "display.max_columns", None,
        "display.max_colwidth", 200,
        "display.width", 10000,
    ):
        if len(df) <= num_rows:
            return df.to_string()
        half = num_rows // 2
        top = df.head(half).to_string()
        bot = df.tail(half).to_string()
        return top + "\n...\n" + bot


@mcp.tool()
def add_pct(
    df_json: str,
    col: str,
    total_col: str,
    pct_col: str = "pct",
) -> str:
    """
    Add a percentage column (col / total_col * 100).

    :param df_json: DataFrame as JSON.
    :param col: numerator column name.
    :param total_col: denominator column name.
    :param pct_col: name for the new percentage column.
    :return: DataFrame with the new percentage column, as JSON.
    """
    df = _df_from_json(df_json)
    df[pct_col] = (df[col] / df[total_col] * 100).round(2)
    return _df_to_json(df)


@mcp.tool()
def find_gaps_in_time_series(
    df_json: str,
    frequency: str,
) -> str:
    """
    Identify missing timestamps in a regularly-spaced time series.

    :param df_json: time-indexed DataFrame as JSON.
    :param frequency: expected frequency string, e.g. "1T", "H", "D".
    :return: JSON list of missing timestamps (ISO-8601 strings).
    """
    df = _df_from_json(df_json)
    df.index = pd.to_datetime(df.index)
    expected = pd.date_range(df.index.min(), df.index.max(), freq=frequency)
    missing = expected.difference(df.index)
    return json.dumps([str(ts) for ts in missing], indent=2)


@mcp.tool()
def resolve_column_names(
    df_json: str,
    column_set: Optional[Union[str, List[str]]] = None,
) -> List[str]:
    """
    Resolve a column specification to a concrete list of column names.

    :param df_json: DataFrame as JSON (used to validate column existence).
    :param column_set: None = all columns, str = single column, list = subset.
    :return: resolved list of column names.
    """
    df = _df_from_json(df_json)
    all_cols = df.columns.tolist()
    if column_set is None:
        return all_cols
    if isinstance(column_set, str):
        column_set = [column_set]
    missing = [c for c in column_set if c not in all_cols]
    if missing:
        raise ValueError(f"Columns not found: {missing}")
    return column_set


# ===========================================================================
# ── MULTI-INDEX ─────────────────────────────────────────────────────────────
# ===========================================================================

@mcp.tool()
def multiindex_df_info(df_json: str) -> str:
    """
    Return metadata about a 2-level MultiIndex DataFrame.

    :param df_json: MultiIndex DataFrame as JSON.
    :return: human-readable info string.
    """
    df = _df_from_json(df_json)
    if not isinstance(df.columns, pd.MultiIndex):
        return f"Not a MultiIndex DataFrame. columns type: {type(df.columns).__name__}"
    l0 = df.columns.get_level_values(0).unique().tolist()
    l1 = df.columns.get_level_values(1).unique().tolist()
    rows = df.index.tolist()
    lines = [
        f"shape={len(l0)} x {len(l1)} x {len(rows)}",
        f"columns_level0={l0}",
        f"columns_level1={l1}",
        f"num_rows={len(rows)}",
    ]
    if isinstance(df.index, pd.DatetimeIndex):
        lines += [
            f"start_timestamp={df.index.min()}",
            f"end_timestamp={df.index.max()}",
            f"frequency={df.index.freq or pd.infer_freq(df.index)}",
        ]
    return "\n".join(lines)


# ===========================================================================
# ── IO ──────────────────────────────────────────────────────────────────────
# ===========================================================================

@mcp.tool()
def read_csv(
    file_path: str,
    index_col: Optional[Union[int, str]] = None,
    parse_dates: bool = True,
) -> str:
    """
    Read a CSV file from disk into a DataFrame.

    :param file_path: path to the CSV (or .gz / .zip) file.
    :param index_col: column to use as the row index.
    :param parse_dates: attempt to parse the index as dates.
    :return: DataFrame as JSON.
    """
    kwargs: Dict[str, Any] = {}
    if index_col is not None:
        kwargs["index_col"] = index_col
    if parse_dates:
        kwargs["parse_dates"] = True
    if any(file_path.endswith(ext) for ext in (".gz", ".gzip", ".tgz")):
        kwargs["compression"] = "gzip"
    elif file_path.endswith(".zip"):
        kwargs["compression"] = "zip"
    df = pd.read_csv(file_path, **kwargs)
    return _df_to_json(df)


@mcp.tool()
def read_parquet(file_path: str) -> str:
    """
    Read a Parquet file from disk into a DataFrame.

    :param file_path: path to the Parquet file.
    :return: DataFrame as JSON.
    """
    df = pd.read_parquet(file_path)
    return _df_to_json(df)


@mcp.tool()
def write_csv(df_json: str, file_path: str, index: bool = True) -> str:
    """
    Write a DataFrame to a CSV file.

    :param df_json: DataFrame as JSON.
    :param file_path: destination path.
    :param index: whether to write the row index.
    :return: confirmation message.
    """
    df = _df_from_json(df_json)
    df.to_csv(file_path, index=index)
    return f"Saved {df.shape[0]} rows × {df.shape[1]} columns to '{file_path}'"


@mcp.tool()
def write_parquet(df_json: str, file_path: str) -> str:
    """
    Write a DataFrame to a Parquet file.

    :param df_json: DataFrame as JSON.
    :param file_path: destination path.
    :return: confirmation message.
    """
    df = _df_from_json(df_json)
    df.to_parquet(file_path)
    return f"Saved {df.shape[0]} rows × {df.shape[1]} columns to '{file_path}'"


# ===========================================================================
# ── ANALYSIS ────────────────────────────────────────────────────────────────
# ===========================================================================

@mcp.tool()
def rolling_corr_over_time(
    df_json: str,
    com: float,
    nan_mode: str = "drop",
) -> str:
    """
    Compute an exponentially-weighted rolling correlation matrix over time.

    :param df_json: time-indexed DataFrame as JSON.
    :param com: center-of-mass for the EWM calculation.
    :param nan_mode: "drop", "fill_with_zero", or "abort".
    :return: multi-index correlation DataFrame as JSON.
    """
    df = _df_from_json(df_json)
    df.index = pd.to_datetime(df.index)
    if nan_mode == "drop":
        df = df.dropna(how="any")
    elif nan_mode == "fill_with_zero":
        df = df.fillna(0.0)
    elif nan_mode == "abort":
        n = int(df.isna().sum().sum())
        if n > 0:
            raise ValueError(f"DataFrame has {n} NaN values.")
    corr_df = df.ewm(com=com, min_periods=int(3 * com)).corr()
    return _df_to_json(corr_df.reset_index())


@mcp.tool()
def describe_df(df_json: str, percentiles: Optional[List[float]] = None) -> str:
    """
    Return descriptive statistics for a DataFrame (wrapper of df.describe()).

    :param df_json: DataFrame as JSON.
    :param percentiles: list of percentiles to include, e.g. [0.1, 0.5, 0.9].
    :return: describe DataFrame as JSON.
    """
    df = _df_from_json(df_json)
    stats = df.describe(percentiles=percentiles)
    return _df_to_json(stats)


@mcp.tool()
def print_column_variability(df_json: str) -> str:
    """
    Report the number of unique values and coefficient of variation per column.

    :param df_json: DataFrame as JSON.
    :return: JSON object mapping column → {nunique, cv, dtype}.
    """
    df = _df_from_json(df_json)
    result: Dict[str, Any] = {}
    for col in df.columns:
        s = df[col]
        info: Dict[str, Any] = {
            "nunique": int(s.nunique()),
            "dtype": str(s.dtype),
        }
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().any() and num.mean() != 0:
            info["cv"] = round(float(num.std() / abs(num.mean())), 4)
        result[col] = info
    return json.dumps(result, indent=2)


# ===========================================================================
# ── CHECK SUMMARY ───────────────────────────────────────────────────────────
# ===========================================================================

# In-process store for CheckSummary objects keyed by a session_id string.
_SUMMARIES: Dict[str, Any] = {}


@mcp.tool()
def check_summary_create(session_id: str, title: str = "") -> str:
    """
    Create a new CheckSummary session.

    :param session_id: unique string identifier for this session.
    :param title: optional title shown in reports.
    :return: confirmation message.
    """
    _SUMMARIES[session_id] = {"title": title, "rows": []}
    return f"CheckSummary '{session_id}' created."


@mcp.tool()
def check_summary_add(
    session_id: str,
    description: str,
    comment: str,
    is_ok: bool,
) -> str:
    """
    Add a check result to an existing CheckSummary session.

    :param session_id: session created with check_summary_create.
    :param description: short label for this check.
    :param comment: details / evidence.
    :param is_ok: True if the check passed.
    :return: confirmation message.
    """
    if session_id not in _SUMMARIES:
        raise KeyError(f"Session '{session_id}' not found. Call check_summary_create first.")
    _SUMMARIES[session_id]["rows"].append(
        {"description": description, "comment": comment, "is_ok": is_ok}
    )
    return f"Added check '{description}' (is_ok={is_ok})."


@mcp.tool()
def check_summary_report(session_id: str) -> str:
    """
    Return a formatted text report for a CheckSummary session.

    :param session_id: session to report on.
    :return: plain-text summary table.
    """
    if session_id not in _SUMMARIES:
        raise KeyError(f"Session '{session_id}' not found.")
    sess = _SUMMARIES[session_id]
    rows = sess["rows"]
    title = sess["title"]
    df = pd.DataFrame(rows)
    all_ok = all(r["is_ok"] for r in rows)
    report_lines = []
    if title:
        report_lines.append(f"# {title}")
    report_lines.append(df.to_string(index=False))
    report_lines.append(f"\nis_ok={all_ok}")
    return "\n".join(report_lines)


# ===========================================================================
# ── Entry point ─────────────────────────────────────────────────────────────
# ===========================================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")
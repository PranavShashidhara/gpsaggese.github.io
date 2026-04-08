"""
Data loading and preprocessing for the profiler agent.

Handles CSV loading, datetime inference, and type coercion.

Import as:

import research.agentic_data_science.schema_agent.schema_agent_loader as radsasal
"""

import datetime
import typing

import pandas as pd

import helpers.hdbg as hdbg
import helpers.hlogging as hloggin
import helpers.hpandas_conversion as hpanconv
import helpers.hpandas_io as hpanio

_LOG = hloggin.getLogger(__name__)


def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a CSV into a DataFrame with clear error handling.

    :param csv_path: Path to the CSV file.
    :type csv_path: str
    :return: Loaded dataframe.
    :rtype: pd.DataFrame
    """
    hdbg.dassert_isinstance(csv_path, str)
    try:
        df = hpanio.read_csv_to_df(csv_path)
    except FileNotFoundError:
        _LOG.error("CSV not found at '%s'.", csv_path)
        raise
        
    hdbg.dassert_lt(0, len(df), "CSV at '%s' loaded as an empty DataFrame.", csv_path)
    
    _LOG.info(
        "Loaded '%s': %d rows × %d columns.", csv_path, len(df), len(df.columns)
    )
    return df


def infer_and_convert_datetime_columns(
    df: pd.DataFrame,
    sample_size: int = 100,
    threshold: float = 0.8,
) -> typing.Tuple[pd.DataFrame, typing.Dict[str, typing.Any]]:
    metadata: typing.Dict[str, typing.Any] = {}
    df_out = df.copy()

    for col in df.columns:
        # 1. If it's already datetime, just ensure UTC awareness
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df_out[col] = pd.to_datetime(df[col], utc=True)
            metadata[col] = {
                "semantic_type": "temporal",
                "granularity": "datetime",
                "format": "pre-converted",
                "confidence": 1.0,
            }
            continue

        # 2. Only attempt conversion on strings/objects
        if not (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])):
            continue

        # Try to parse
        try:
            # We use errors="coerce" so non-dates become NaT
            parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
            
            valid_count = parsed.notna().sum()
            if valid_count == 0:
                continue
                
            confidence = float(valid_count / len(df[col]))
            
            # Only convert if it meets our confidence threshold
            if confidence >= threshold:
                df_out[col] = parsed
                has_time = (parsed.dt.time != pd.Timestamp("00:00:00").time()).any()
                metadata[col] = {
                    "semantic_type": "temporal",
                    "granularity": "datetime" if has_time else "date",
                    "format": "inferred",
                    "confidence": confidence,
                }
                _LOG.info("Converted column '%s' to datetime", col)
        except Exception:
            continue

    return df_out, metadata


def _try_strptime(val: str, fmt: str) -> bool:
    """
    Return True if val parses under fmt, False otherwise.
    """
    try:
        datetime.datetime.strptime(val, fmt)
        return True
    except Exception:  # pylint: disable=broad-exception-caught
        return False



def prepare_dataframes(
    csv_paths: typing.List[str],
    tags: typing.Optional[typing.List[str]] = None,
) -> typing.Tuple[
    typing.Dict[str, pd.DataFrame], 
    typing.Dict[str, typing.List[str]],
    typing.Dict[str, typing.Any]  # Added return type for metadata
]:
    """
    Load and prepare all CSV files in one pass.
    """
    hdbg.dassert_isinstance(csv_paths, list)
    if tags is None:
        import os
        tags = [os.path.splitext(os.path.basename(p))[0] for p in csv_paths]
    
    tag_to_df: typing.Dict[str, pd.DataFrame] = {}
    cat_cols_map: typing.Dict[str, typing.List[str]] = {}
    combined_dt_meta: typing.Dict[str, typing.Any] = {} # Store metadata here

    for path, tag in zip(csv_paths, tags):
        # 1. Load and perform initial type conversion
        df = load_csv(path)
        df = hpanconv.convert_df(df)
        
        # 2. Perform datetime inference and CAPTURE metadata
        df, dt_meta = infer_and_convert_datetime_columns(df)
        combined_dt_meta.update(dt_meta) # Merge metadata
        
        # 3. FIX: Automatically promote the first detected temporal column to 
        # the Index for Quality and Duration reports.
        temporal_cols = [c for c, m in dt_meta.items() if m.get("semantic_type") == "temporal"]
        if temporal_cols:
            df = df.set_index(temporal_cols[0], drop=False)
            
        tag_to_df[tag] = df

        # 4. Identify categorical/string columns
        cat_cols_map[tag] = df.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

    return tag_to_df, cat_cols_map, combined_dt_meta
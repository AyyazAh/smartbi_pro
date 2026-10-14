"""Auto-cleaning: dedup, missing impute, outlier clip, type coercion."""
import pandas as pd
import numpy as np
from utils.logger import get_logger

log = get_logger("cleaning")


def _drop_duplicates(df: pd.DataFrame, log_dict: dict) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        log_dict["duplicates_removed"] = removed
        log.info(f"Dropped {removed:,} duplicate rows")
    return df


def _coerce_types(df: pd.DataFrame, schema: dict, log_dict: dict) -> pd.DataFrame:
    """Force columns into their semantic types."""
    for col, stype in schema.items():
        if col not in df.columns:
            continue
        if stype == "date":
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    log_dict["coerced"].append(f"{col} -> datetime")
                except Exception as e:
                    log.warning(f"Could not coerce {col} to date: {e}")
        elif stype in ("revenue", "numeric"):
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")
                log_dict["coerced"].append(f"{col} -> numeric")
    return df


def _impute_missing(df: pd.DataFrame, schema: dict, log_dict: dict) -> pd.DataFrame:
    """Impute missing values based on semantic type."""
    for col, stype in schema.items():
        if col not in df.columns:
            continue
        miss = df[col].isna().sum()
        if miss == 0:
            continue

        if stype in ("revenue", "numeric"):
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            log_dict["imputed"].append(f"{col}: {miss} values -> median ({fill_val:.2f})")

        elif stype in ("category", "geo", "text"):
            fill_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
            log_dict["imputed"].append(f"{col}: {miss} values -> mode ('{fill_val}')")

        elif stype == "date":
            # forward-fill then backward-fill
            df[col] = df[col].ffill().bfill()
            log_dict["imputed"].append(f"{col}: {miss} dates -> ffill/bfill")

        elif stype == "id":
            # don't impute IDs — drop them instead
            df = df.dropna(subset=[col])
            log_dict["imputed"].append(f"{col}: dropped {miss} rows with missing ID")

    return df


def _clip_outliers(df: pd.DataFrame, schema: dict,
                   low_q: float, high_q: float, log_dict: dict) -> pd.DataFrame:
    """Clip outliers via quantile bounds."""
    for col, stype in schema.items():
        if stype not in ("revenue", "numeric") or col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        q_low = df[col].quantile(low_q)
        q_high = df[col].quantile(high_q)

        if q_low == q_high:
            continue

        n_clipped = ((df[col] < q_low) | (df[col] > q_high)).sum()
        if n_clipped:
            df[col] = df[col].clip(q_low, q_high)
            log_dict["clipped"].append(
                f"{col}: {n_clipped} values clipped to [{q_low:.2f}, {q_high:.2f}]"
            )
    return df


def auto_clean(df: pd.DataFrame, schema: dict, config: dict = None) -> tuple:
    """
    Run full auto-cleaning pipeline.
    Returns (cleaned_df, cleaning_log_dict).
    """
    config = config or {}
    cfg = config.get("cleaning", {})
    low_q = cfg.get("outlier_iqr_low", 0.01)
    high_q = cfg.get("outlier_iqr_high", 0.99)
    do_dedup = cfg.get("drop_duplicates", True)

    log.info(f"Starting auto-clean on {len(df):,} rows")

    cleaning_log = {
        "original_rows": len(df),
        "duplicates_removed": 0,
        "coerced": [],
        "imputed": [],
        "clipped": [],
    }

    df = df.copy()

    # 1. Types first
    df = _coerce_types(df, schema, cleaning_log)

    # 2. Duplicates
    if do_dedup:
        df = _drop_duplicates(df, cleaning_log)

    # 3. Missing values
    df = _impute_missing(df, schema, cleaning_log)

    # 4. Outliers
    df = _clip_outliers(df, schema, low_q, high_q, cleaning_log)

    cleaning_log["final_rows"] = len(df)
    cleaning_log["rows_delta"] = cleaning_log["original_rows"] - cleaning_log["final_rows"]

    log.info(f"Cleaning done: {cleaning_log['original_rows']:,} -> {cleaning_log['final_rows']:,} rows")
    return df, cleaning_log


def print_cleaning_log(log_dict: dict) -> None:
    """Pretty-print cleaning summary."""
    print(f"\n{'='*70}")
    print(f"🧹 CLEANING REPORT")
    print(f"{'='*70}")
    print(f"Original rows: {log_dict['original_rows']:,}")
    print(f"Final rows:    {log_dict['final_rows']:,}")

    if log_dict["duplicates_removed"]:
        print(f"🗑️  Duplicates removed: {log_dict['duplicates_removed']:,}")

    if log_dict["coerced"]:
        print(f"\n🔄 Type Coercions ({len(log_dict['coerced'])}):")
        for c in log_dict["coerced"]:
            print(f"   - {c}")

    if log_dict["imputed"]:
        print(f"\n🩹 Imputations ({len(log_dict['imputed'])}):")
        for i in log_dict["imputed"]:
            print(f"   - {i}")

    if log_dict["clipped"]:
        print(f"\n✂️  Outlier Clips ({len(log_dict['clipped'])}):")
        for c in log_dict["clipped"]:
            print(f"   - {c}")

    print(f"{'='*70}\n")


__all__ = ["auto_clean", "print_cleaning_log"]
"""Data profiling: quality score, missing values, uniqueness, distributions."""
import pandas as pd
import numpy as np
from utils.logger import get_logger

log = get_logger("profiling")


def _column_score(series: pd.Series, semantic_type: str, n_rows: int) -> float:
    """
    Score a single column 0-100 based on:
      - Completeness (missing values)  -> 50%
      - Uniqueness (not constant)      -> 30%
      - Type validity (semantic match) -> 20%
    """
    if n_rows == 0:
        return 0.0

    # 1. Completeness
    miss_pct = series.isna().mean()
    completeness = (1 - miss_pct) * 50

    # 2. Uniqueness — penalize constant columns (low info)
    nunique = series.nunique(dropna=True)
    if nunique <= 1:
        uniqueness = 0.0
    elif nunique / max(n_rows, 1) >= 0.9:
        uniqueness = 30.0        # near-unique (likely ID) -> healthy
    else:
        # scale 2..N unique values -> 10..30
        uniqueness = min(30.0, 10 + (nunique / n_rows) * 100)

    # 3. Type validity
    validity = 20.0
    if semantic_type == "numeric" and not pd.api.types.is_numeric_dtype(series):
        validity = 5.0
    elif semantic_type == "date" and not pd.api.types.is_datetime64_any_dtype(series):
        validity = 8.0
    elif semantic_type == "revenue" and not pd.api.types.is_numeric_dtype(series):
        validity = 5.0

    return round(completeness + uniqueness + validity, 2)


def _column_stats(series: pd.Series, semantic_type: str) -> dict:
    """Compute per-column stats."""
    n = len(series)
    missing = int(series.isna().sum())
    miss_pct = round((missing / n) * 100, 2) if n else 0
    nunique = int(series.nunique(dropna=True))

    stats = {
        "dtype": str(series.dtype),
        "semantic_type": semantic_type,
        "missing": missing,
        "missing_pct": miss_pct,
        "unique": nunique,
        "unique_pct": round((nunique / n) * 100, 2) if n else 0,
    }

    # numeric extras
    if pd.api.types.is_numeric_dtype(series):
        clean = series.dropna()
        if len(clean):
            stats.update({
                "min": float(clean.min()),
                "max": float(clean.max()),
                "mean": round(float(clean.mean()), 2),
                "median": float(clean.median()),
                "std": round(float(clean.std()), 2) if len(clean) > 1 else 0.0,
            })

    # categorical extras
    if semantic_type in ("category", "geo", "text") and nunique > 0 and nunique <= 50:
        stats["top_values"] = series.value_counts().head(5).to_dict()

    return stats


def profile(df: pd.DataFrame, schema: dict) -> dict:
    """
    Full profile of a DataFrame.
    Returns:
      {
        rows, cols,
        columns: {col: {..stats..}},
        quality_score,
        issues: [ ... ]
      }
    """
    log.info(f"Profiling {len(df):,} rows x {len(df.columns)} cols")
    n_rows = len(df)

    report = {
        "rows": n_rows,
        "cols": len(df.columns),
        "columns": {},
        "issues": [],
    }

    total_score = 0.0
    for col in df.columns:
        stype = schema.get(col, "unknown")
        col_stats = _column_stats(df[col], stype)
        col_score = _column_score(df[col], stype, n_rows)
        col_stats["score"] = col_score

        report["columns"][col] = col_stats
        total_score += col_score

        # Issue detection
        if col_stats["missing_pct"] > 20:
            report["issues"].append({
                "severity": "HIGH" if col_stats["missing_pct"] > 50 else "MEDIUM",
                "column": col,
                "type": "high_missing",
                "message": f"{col_stats['missing_pct']}% values missing",
            })
        if col_stats["unique"] <= 1:
            report["issues"].append({
                "severity": "LOW",
                "column": col,
                "type": "constant_column",
                "message": f"Column '{col}' has only 1 unique value",
            })

    report["quality_score"] = round(total_score / max(len(df.columns), 1), 2)
    log.info(f"Quality score: {report['quality_score']} | Issues: {len(report['issues'])}")
    return report


def print_report(report: dict) -> None:
    """Pretty-print profile to console."""
    print(f"\n{'='*70}")
    print(f"📊 DATA PROFILING REPORT")
    print(f"{'='*70}")
    print(f"Rows: {report['rows']:,}  |  Columns: {report['cols']}")
    print(f"🎯 Overall Quality Score: {report['quality_score']}/100")
    print(f"{'-'*70}")

    print(f"\n{'Column':<18} {'Type':<12} {'Missing%':<10} {'Unique':<8} {'Score':<6}")
    print(f"{'-'*70}")
    for col, s in report["columns"].items():
        print(f"{col:<18} {s['semantic_type']:<12} {s['missing_pct']:<10} "
              f"{s['unique']:<8} {s['score']:<6}")

    if report["issues"]:
        print(f"\n⚠️  Issues Found ({len(report['issues'])}):")
        for iss in report["issues"]:
            print(f"   [{iss['severity']}] {iss['column']}: {iss['message']}")
    else:
        print("\n✅ No major issues detected")

    print(f"{'='*70}\n")


__all__ = ["profile", "print_report"]
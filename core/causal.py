"""Causal analysis: correlation, lead-lag, Granger-lite, impact ranking."""
import pandas as pd
import numpy as np
from scipy import stats
from utils.logger import get_logger

log = get_logger("causal")


# Priority order for picking the "main" revenue/target column
_TARGET_PRIORITY = ["revenue", "sales", "total_revenue", "total_sales", "amount"]


def _pick_target_column(rev_cols: list, num_cols: list, df: pd.DataFrame) -> str:
    """
    Pick the most meaningful target column:
      1. Exact name match from priority list
      2. Highest-variance revenue column (likely aggregated)
      3. First numeric column
    """
    for name in _TARGET_PRIORITY:
        if name in rev_cols:
            return name

    if rev_cols:
        variances = {}
        for c in rev_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            variances[c] = s.var()
        # Filter NaN
        variances = {k: v for k, v in variances.items() if pd.notna(v)}
        if variances:
            return max(variances, key=variances.get)
        return rev_cols[0]

    if num_cols:
        return num_cols[0]

    return None


def _numeric_columns(df: pd.DataFrame, schema: dict) -> list:
    """Pick numeric columns suitable for causal analysis."""
    cols = [c for c, t in schema.items()
            if t in ("revenue", "numeric") and c in df.columns]
    keep = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) > 10 and s.std() > 0:
            keep.append(c)
    return keep


def correlation_matrix(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Pearson correlation matrix."""
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 10:
        return pd.DataFrame()
    return sub.corr(method="pearson").round(3)


def lead_lag_analysis(df: pd.DataFrame, x_col: str, y_col: str,
                      max_lag: int = 6) -> dict:
    """
    Test whether x leads y (i.e., past x predicts current y).
    Aggregates both to monthly series first.
    """
    work = df[[x_col, y_col]].copy()
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna()

    if len(work) < 30:
        return {}

    # Simple approach: shift X by lag and compute correlation with Y
    best_lag, best_corr, best_p = 0, 0.0, 1.0
    for lag in range(0, max_lag + 1):
        if lag == 0:
            xs, ys = work[x_col], work[y_col]
        else:
            xs = work[x_col].iloc[:-lag]
            ys = work[y_col].iloc[lag:]
        if len(xs) < 20:
            break
        try:
            corr, p = stats.pearsonr(xs, ys)
            if abs(corr) > abs(best_corr):
                best_corr, best_lag, best_p = corr, lag, p
        except Exception:
            continue

    return {
        "x": x_col,
        "y": y_col,
        "best_lag": best_lag,
        "correlation": round(best_corr, 3),
        "p_value": round(best_p, 4),
        "significant": best_p < 0.05,
    }


def granger_lite(df: pd.DataFrame, x_col: str, y_col: str,
                 lag: int = 1) -> dict:
    """
    Simplified Granger test: does X's past value improve prediction of Y?
    Uses linear regression: Y_t ~ Y_{t-1} + X_{t-lag}
    Compares R² with baseline Y_t ~ Y_{t-1}.
    """
    work = df[[x_col, y_col]].copy()
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna()

    if len(work) < lag + 20:
        return {}

    y = work[y_col].values
    x = work[x_col].values

    # Build lagged arrays
    y_target = y[lag:]
    y_prev = y[:-lag]
    x_prev = x[:-lag]

    # Baseline model: Y_t ~ Y_{t-1}
    A_base = np.column_stack([y_prev, np.ones(len(y_prev))])
    coef_base, *_ = np.linalg.lstsq(A_base, y_target, rcond=None)
    pred_base = A_base @ coef_base
    ss_res_base = np.sum((y_target - pred_base) ** 2)
    ss_tot = np.sum((y_target - np.mean(y_target)) ** 2)
    r2_base = 1 - ss_res_base / ss_tot if ss_tot > 0 else 0

    # Full model: Y_t ~ Y_{t-1} + X_{t-lag}
    A_full = np.column_stack([y_prev, x_prev, np.ones(len(y_prev))])
    coef_full, *_ = np.linalg.lstsq(A_full, y_target, rcond=None)
    pred_full = A_full @ coef_full
    ss_res_full = np.sum((y_target - pred_full) ** 2)
    r2_full = 1 - ss_res_full / ss_tot if ss_tot > 0 else 0

    # F-test for improvement
    n = len(y_target)
    df_num = 1
    df_den = n - 3
    if ss_res_full <= 0 or df_den <= 0:
        f_stat, p_val = 0.0, 1.0
    else:
        f_stat = ((ss_res_base - ss_res_full) / df_num) / (ss_res_full / df_den)
        p_val = 1 - stats.f.cdf(f_stat, df_num, df_den)

    return {
        "x": x_col,
        "y": y_col,
        "lag": lag,
        "r2_base": round(float(r2_base), 4),
        "r2_full": round(float(r2_full), 4),
        "r2_improvement": round(float(r2_full - r2_base), 4),
        "f_stat": round(float(f_stat), 3),
        "p_value": round(float(p_val), 4),
        "significant": p_val < 0.05,
    }


def rank_drivers(df: pd.DataFrame, schema: dict,
                 target_col: str = None,
                 max_lag: int = 6) -> pd.DataFrame:
    """
    Rank which numeric columns most strongly drive the target column.
    Uses lead-lag + Granger-lite.
    """
    num_cols = _numeric_columns(df, schema)
    if target_col is None:
        rev = [c for c, t in schema.items() if t == "revenue" and c in df.columns]
        target_col = _pick_target_column(rev, num_cols, df)

    if target_col is None:
        return pd.DataFrame()

    candidates = [c for c in num_cols if c != target_col]
    rows = []
    for x in candidates:
        ll = lead_lag_analysis(df, x, target_col, max_lag)
        if not ll:
            continue
        gr = granger_lite(df, x, target_col, lag=max(1, ll["best_lag"]))
        rows.append({
            "driver": x,
            "target": target_col,
            "best_lag": ll["best_lag"],
            "correlation": ll["correlation"],
            "corr_p_value": ll["p_value"],
            "granger_r2_gain": gr.get("r2_improvement", 0.0),
            "granger_p_value": gr.get("p_value", 1.0),
            "significant": ll["significant"] or gr.get("significant", False),
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["impact_score"] = (
        out["correlation"].abs() * 0.6 +
        out["granger_r2_gain"].clip(lower=0) * 0.4
    ).round(4)
    out = out.sort_values("impact_score", ascending=False).reset_index(drop=True)
    return out


def analyze(df: pd.DataFrame, schema: dict, config: dict = None) -> dict:
    """
    Full causal analysis pipeline.
    Returns:
      {
        "correlation_matrix": DataFrame,
        "driver_ranking": DataFrame,
        "target_col": str,
        "top_driver": dict,
        "insights": [list of str]
      }
    """
    log.info(f"Running causal analysis on {len(df):,} rows")
    num_cols = _numeric_columns(df, schema)

    if len(num_cols) < 2:
        log.warning("Not enough numeric columns for causal analysis")
        return {
            "correlation_matrix": pd.DataFrame(),
            "driver_ranking": pd.DataFrame(),
            "target_col": None,
            "top_driver": None,
            "insights": ["Insufficient numeric columns for causal analysis"],
        }

    # --- Smart target column selection ---
    rev_cols = [c for c, t in schema.items() if t == "revenue" and c in df.columns]
    target = _pick_target_column(rev_cols, num_cols, df)
    log.info(f"Target column selected: {target}")

    corr_mat = correlation_matrix(df, num_cols)
    drivers = rank_drivers(df, schema, target_col=target)

    # Insights
    insights = []
    if not drivers.empty:
        top = drivers.iloc[0]
        if top["significant"]:
            insights.append(
                f"🎯 '{top['driver']}' is the strongest driver of '{top['target']}' "
                f"(corr={top['correlation']}, lag={int(top['best_lag'])} periods)."
            )
        else:
            insights.append(
                f"📊 Strongest correlate: '{top['driver']}' "
                f"(corr={top['correlation']}, not statistically significant)."
            )

    if not corr_mat.empty:
        # Find strongest off-diagonal correlation
        # Convert to numpy explicitly (avoids read-only view issue)
        arr = np.array(corr_mat.abs().to_numpy(), dtype=float, copy=True)
        np.fill_diagonal(arr, 0)
        corr_abs = pd.DataFrame(arr, index=corr_mat.index, columns=corr_mat.columns)
        max_pair = corr_abs.stack().idxmax()
        c1, c2 = max_pair
        val = corr_mat.loc[c1, c2]
        insights.append(f"🔗 Strongest correlation: {c1} ↔ {c2} (r={val:.2f})")

    return {
        "correlation_matrix": corr_mat,
        "driver_ranking": drivers,
        "target_col": target,
        "top_driver": drivers.iloc[0].to_dict() if not drivers.empty else None,
        "insights": insights,
    }


def print_causal_report(result: dict) -> None:
    """Pretty-print causal analysis."""
    print(f"\n{'='*70}")
    print(f"🔍 CAUSAL ANALYSIS REPORT")
    print(f"{'='*70}")

    cm = result["correlation_matrix"]
    if not cm.empty:
        print(f"\n📊 CORRELATION MATRIX")
        print(cm.to_string())
    else:
        print("\n⚠️  Correlation matrix empty")

    dr = result["driver_ranking"]
    if not dr.empty:
        print(f"\n🎯 DRIVER RANKING (target: {result['target_col']})")
        print(f"{'-'*70}")
        print(f"{'Driver':<16} {'Lag':>4} {'Corr':>8} {'R²gain':>8} {'p-val':>8} {'Sig':>5}")
        print(f"{'-'*70}")
        for _, row in dr.iterrows():
            print(f"{row['driver']:<16} {int(row['best_lag']):>4} "
                  f"{row['correlation']:>8.3f} {row['granger_r2_gain']:>8.4f} "
                  f"{row['granger_p_value']:>8.4f} "
                  f"{'✓' if row['significant'] else ' ':>5}")

    if result["insights"]:
        print(f"\n💡 INSIGHTS")
        for ins in result["insights"]:
            print(f"   {ins}")

    print(f"{'='*70}\n")


__all__ = ["analyze", "print_causal_report", "correlation_matrix",
           "lead_lag_analysis", "granger_lite", "rank_drivers"]
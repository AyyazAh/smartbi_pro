"""Auto KPI discovery from cleaned DataFrame."""
import pandas as pd
import numpy as np
from utils.logger import get_logger

log = get_logger("kpi_engine")


# --- Pandas version-safe month-end frequency ---
def _month_freq() -> str:
    """Return 'ME' for pandas >= 2.2, else 'M' (backward compatible)."""
    major, minor = map(int, pd.__version__.split(".")[:2])
    return "ME" if (major, minor) >= (2, 2) else "M"


def _monthly_series(df: pd.DataFrame, date_col: str, value_col: str,
                    agg: str = "sum") -> pd.Series:
    """Aggregate value_col by month."""
    ts = df.set_index(date_col)[value_col].resample(_month_freq()).agg(agg).dropna()
    return ts


def _mom_growth(ts: pd.Series) -> float:
    """Month-over-month growth % (last vs prev)."""
    if len(ts) < 2:
        return 0.0
    prev, last = ts.iloc[-2], ts.iloc[-1]
    if prev == 0:
        return 0.0
    return round(((last - prev) / prev) * 100, 2)


def _pareto_concentration(df: pd.DataFrame, dim_col: str,
                          value_col: str, top_pct: float = 0.20) -> dict:
    """What % of revenue comes from top 20% of dimension values?"""
    grouped = df.groupby(dim_col)[value_col].sum().sort_values(ascending=False)
    if grouped.empty:
        return {}
    n_top = max(1, int(len(grouped) * top_pct))
    top_sum = grouped.head(n_top).sum()
    total = grouped.sum()
    return {
        "dimension": dim_col,
        "top_n": n_top,
        "top_n_pct_of_values": round((n_top / len(grouped)) * 100, 1),
        "revenue_share_pct": round((top_sum / total) * 100, 2) if total else 0.0,
    }


def _period_comparison(df: pd.DataFrame, date_col: str, value_col: str) -> dict:
    """First half vs second half of the timeline."""
    ts = df.set_index(date_col)[value_col].resample(_month_freq()).sum().dropna()
    if len(ts) < 4:
        return {}
    mid = len(ts) // 2
    first = ts.iloc[:mid].sum()
    second = ts.iloc[mid:].sum()
    change = ((second - first) / first) * 100 if first else 0.0
    return {
        "first_half_sum": float(first),
        "second_half_sum": float(second),
        "change_pct": round(float(change), 2),
    }


def discover_kpis(df: pd.DataFrame, schema: dict,
                  config: dict = None) -> dict:
    """
    Auto-discover business KPIs from cleaned dataframe + schema.
    Returns dict with all KPIs.
    """
    config = config or {}
    top_n = config.get("insights", {}).get("top_n_contributors", 5)

    log.info(f"Discovering KPIs on {len(df):,} rows")

    kpis = {
        "row_count": len(df),
        "column_count": len(df.columns),
    }

    # --- Find key columns ---
    rev_cols  = [c for c, t in schema.items() if t == "revenue"]
    date_cols = [c for c, t in schema.items() if t == "date"]
    cat_cols  = [c for c, t in schema.items() if t in ("category", "geo")]
    num_cols  = [c for c, t in schema.items() if t == "numeric"]

    # --- Revenue KPIs ---
    if rev_cols:
        main_rev = rev_cols[0]
        kpis["main_revenue_col"] = main_rev
        kpis["total_revenue"] = float(df[main_rev].sum())
        kpis["avg_revenue"] = float(df[main_rev].mean())
        kpis["median_revenue"] = float(df[main_rev].median())
        kpis["revenue_std"] = float(df[main_rev].std()) if len(df) > 1 else 0.0

        # --- Time series + growth ---
        if date_cols:
            date_col = date_cols[0]
            kpis["main_date_col"] = date_col

            try:
                ts = _monthly_series(df, date_col, main_rev, agg="sum")
                if len(ts) >= 2:
                    kpis["trend_series"] = ts
                    kpis["mom_growth_pct"] = _mom_growth(ts)
                    kpis["best_month"] = {
                        "month": str(ts.idxmax().date()),
                        "value": float(ts.max()),
                    }
                    kpis["worst_month"] = {
                        "month": str(ts.idxmin().date()),
                        "value": float(ts.min()),
                    }
                    kpis["period_comparison"] = _period_comparison(df, date_col, main_rev)
            except Exception as e:
                log.warning(f"Time-series KPIs failed: {e}")

        # --- Top contributors per dimension ---
        for col in cat_cols:
            grouped = df.groupby(col)[main_rev].sum().sort_values(ascending=False)
            kpis[f"top_{col}"] = {
                str(k): round(float(v), 2) for k, v in grouped.head(top_n).items()
            }
            kpis[f"n_unique_{col}"] = int(grouped.shape[0])

        # --- Pareto concentration ---
        if cat_cols:
            kpis["pareto"] = _pareto_concentration(df, cat_cols[0], main_rev)

    # --- Numeric column stats ---
    for col in num_cols[:5]:  # top 5 numeric cols
        kpis[f"stat_{col}"] = {
            "mean": round(float(df[col].mean()), 2),
            "median": round(float(df[col].median()), 2),
            "std": round(float(df[col].std()), 2) if len(df) > 1 else 0.0,
            "min": round(float(df[col].min()), 2),
            "max": round(float(df[col].max()), 2),
        }

    log.info(f"Discovered {len(kpis)} KPIs")
    return kpis


def print_kpi_report(kpis: dict) -> None:
    """Pretty-print KPI report."""
    print(f"\n{'='*70}")
    print(f"📈 KPI DISCOVERY REPORT")
    print(f"{'='*70}")

    print(f"Rows: {kpis.get('row_count', 0):,} | Columns: {kpis.get('column_count', 0)}")
    print(f"{'-'*70}")

    if "total_revenue" in kpis:
        print(f"\n💰 REVENUE METRICS")
        print(f"   Total:      {kpis['total_revenue']:>15,.2f}")
        print(f"   Average:    {kpis['avg_revenue']:>15,.2f}")
        print(f"   Median:     {kpis['median_revenue']:>15,.2f}")
        print(f"   Std Dev:    {kpis['revenue_std']:>15,.2f}")

    if "mom_growth_pct" in kpis:
        g = kpis["mom_growth_pct"]
        arrow = "📈" if g > 0 else "📉"
        print(f"\n{arrow} GROWTH")
        print(f"   MoM Growth: {g:>15.2f}%")
        if "best_month" in kpis:
            print(f"   Best Month: {kpis['best_month']['month']}  ({kpis['best_month']['value']:,.2f})")
            print(f"   Worst Month:{kpis['worst_month']['month']}  ({kpis['worst_month']['value']:,.2f})")

    if "period_comparison" in kpis and kpis["period_comparison"]:
        pc = kpis["period_comparison"]
        print(f"\n📊 PERIOD COMPARISON")
        print(f"   First half:  {pc['first_half_sum']:>15,.2f}")
        print(f"   Second half: {pc['second_half_sum']:>15,.2f}")
        print(f"   Change:      {pc['change_pct']:>15.2f}%")

    # Top contributors
    top_dims = [k for k in kpis if k.startswith("top_")]
    if top_dims:
        print(f"\n🏆 TOP CONTRIBUTORS")
        for key in top_dims:
            dim = key[4:]  # strip "top_"
            print(f"\n   By {dim}:")
            for name, val in kpis[key].items():
                print(f"      {name:<20} {val:>15,.2f}")

    if "pareto" in kpis and kpis["pareto"]:
        p = kpis["pareto"]
        print(f"\n📐 PARETO CONCENTRATION")
        print(f"   Top {p['top_n_pct_of_values']}% of '{p['dimension']}' accounts for "
              f"{p['revenue_share_pct']}% of revenue")

    print(f"{'='*70}\n")


__all__ = ["discover_kpis", "print_kpi_report"]
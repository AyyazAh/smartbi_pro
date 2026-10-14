"""Anomaly detection: Isolation Forest + Z-score methods."""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from utils.logger import get_logger

log = get_logger("anomaly")


def _row_level(df: pd.DataFrame, schema: dict,
               contamination: float, seed: int) -> pd.DataFrame:
    """Isolation Forest on numeric columns."""
    num_cols = [c for c, t in schema.items()
                if t in ("revenue", "numeric") and c in df.columns]
    if not num_cols:
        return df.iloc[0:0].copy()

    X = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    model = IsolationForest(
        contamination=contamination,
        random_state=seed,
        n_estimators=100,
    )
    labels = model.fit_predict(X)
    scores = model.decision_function(X)

    df = df.copy()
    df["_anomaly_flag"] = labels
    df["_anomaly_score"] = scores
    df["_anomaly_severity"] = -scores

    out = df[df["_anomaly_flag"] == -1].copy()
    out = out.sort_values("_anomaly_severity", ascending=False)
    log.info(f"Row-level anomalies: {len(out)} (contamination={contamination})")
    return out


def _time_series_anomalies(df: pd.DataFrame, schema: dict,
                           threshold_z: float = 2.5) -> pd.DataFrame:
    """Detect anomalous months in time series (Z-score)."""
    rev_cols = [c for c, t in schema.items() if t == "revenue" and c in df.columns]
    date_cols = [c for c, t in schema.items() if t == "date" and c in df.columns]
    if not rev_cols or not date_cols:
        return pd.DataFrame()

    rev, dt = rev_cols[0], date_cols[0]
    freq = "ME" if tuple(map(int, pd.__version__.split(".")[:2])) >= (2, 2) else "M"

    work = df[[dt, rev]].copy()
    if not pd.api.types.is_datetime64_any_dtype(work[dt]):
        work[dt] = pd.to_datetime(work[dt], errors="coerce")
    if not pd.api.types.is_numeric_dtype(work[rev]):
        work[rev] = pd.to_numeric(work[rev], errors="coerce")
    work = work.dropna(subset=[dt, rev])

    if work.empty:
        return pd.DataFrame()

    monthly = work.set_index(dt)[rev].resample(freq).sum().dropna()
    if len(monthly) < 4:
        return pd.DataFrame()

    mean, std = monthly.mean(), monthly.std()
    if std == 0:
        return pd.DataFrame()

    z = (monthly - mean) / std
    anomalies = monthly[np.abs(z) > threshold_z]

    if anomalies.empty:
        return pd.DataFrame()

    out = pd.DataFrame({
        "month": anomalies.index,
        "value": anomalies.values,
        "z_score": z.loc[anomalies.index].values.round(2),
        "type": ["spike" if v > 0 else "dip" for v in z.loc[anomalies.index].values],
    })
    log.info(f"Time-series anomalies: {len(out)} months")
    return out


def _category_spikes(df: pd.DataFrame, schema: dict,
                     threshold_z: float = 2.5) -> pd.DataFrame:
    """Detect anomalous category/region values by revenue contribution."""
    rev_cols = [c for c, t in schema.items() if t == "revenue" and c in df.columns]
    cat_cols = [c for c, t in schema.items()
                if t in ("category", "geo") and c in df.columns]
    if not rev_cols or not cat_cols:
        return pd.DataFrame()

    rev = rev_cols[0]

    work = df.copy()
    if not pd.api.types.is_numeric_dtype(work[rev]):
        work[rev] = pd.to_numeric(work[rev], errors="coerce")
    work = work.dropna(subset=[rev])

    results = []
    for col in cat_cols:
        grouped = work.groupby(col)[rev].sum()
        if len(grouped) < 4:
            continue
        mean, std = grouped.mean(), grouped.std()
        if std == 0:
            continue
        z = (grouped - mean) / std
        for name, zval in z.items():
            if abs(zval) > threshold_z:
                results.append({
                    "dimension": col,
                    "value": name,
                    "revenue": float(grouped[name]),
                    "z_score": round(float(zval), 2),
                })
    if results:
        log.info(f"Category spikes: {len(results)}")
    return pd.DataFrame(results)


def detect_anomalies(df: pd.DataFrame, schema: dict,
                     config: dict = None) -> dict:
    """Full anomaly detection pipeline."""
    config = config or {}
    acfg = config.get("anomaly", {})
    contamination = acfg.get("contamination", 0.05)
    seed = acfg.get("random_state", 42)
    z_threshold = acfg.get("z_threshold", 2.5)

    log.info(f"Running anomaly detection on {len(df):,} rows")

    row_anom = _row_level(df, schema, contamination, seed)
    ts_anom  = _time_series_anomalies(df, schema, z_threshold)
    cat_spik = _category_spikes(df, schema, z_threshold)

    summary = {
        "row_anomaly_count":  len(row_anom),
        "ts_anomaly_count":   len(ts_anom),
        "cat_spike_count":    len(cat_spik),
        "contamination":      contamination,
        "z_threshold":        z_threshold,
        "method_row":         "IsolationForest",
        "method_ts":          "Z-score",
    }

    log.info(f"Anomaly summary: rows={summary['row_anomaly_count']}, "
             f"ts={summary['ts_anomaly_count']}, cat={summary['cat_spike_count']}")

    return {
        "row_anomalies": row_anom,
        "ts_anomalies": ts_anom,
        "cat_spikes": cat_spik,
        "summary": summary,
    }


def print_anomaly_report(result: dict) -> None:
    """Pretty-print anomaly detection summary."""
    s = result["summary"]
    print(f"\n{'='*70}")
    print(f"⚠️  ANOMALY DETECTION REPORT")
    print(f"{'='*70}")
    print(f"Method (rows):  {s['method_row']} (contamination={s['contamination']})")
    print(f"Method (time):  {s['method_ts']} (threshold=±{s['z_threshold']})")
    print(f"{'-'*70}")

    print(f"\n📊 COUNTS")
    print(f"   Row anomalies:      {s['row_anomaly_count']:>6,}")
    print(f"   Time-series spikes: {s['ts_anomaly_count']:>6,}")
    print(f"   Category spikes:    {s['cat_spike_count']:>6,}")

    ra = result["row_anomalies"]
    if not ra.empty:
        print(f"\n🔴 TOP 5 ROW ANOMALIES (by severity)")
        cols = [c for c in ra.columns if not c.startswith("_")][:5]
        print(f"   {'   '.join(cols)}")
        for _, row in ra.head(5).iterrows():
            vals = "   ".join(str(row[c])[:15] for c in cols)
            print(f"   {vals}")

    ts = result["ts_anomalies"]
    if not ts.empty:
        print(f"\n📅 TIME-SERIES ANOMALIES")
        for _, row in ts.iterrows():
            dt = pd.to_datetime(row["month"]).strftime("%Y-%m")
            print(f"   {dt}  {row['type']:<8}  value={row['value']:>12,.2f}  z={row['z_score']:>6.2f}")

    cs = result["cat_spikes"]
    if not cs.empty:
        print(f"\n🔥 CATEGORY SPIKES")
        for _, row in cs.head(5).iterrows():
            print(f"   {row['dimension']}={row['value']:<15} "
                  f"revenue={row['revenue']:>12,.2f}  z={row['z_score']:>6.2f}")

    print(f"{'='*70}\n")


__all__ = ["detect_anomalies", "print_anomaly_report"]
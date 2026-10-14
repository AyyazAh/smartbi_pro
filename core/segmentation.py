"""Segmentation: KMeans with automatic k-selection via silhouette score."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils.logger import get_logger

log = get_logger("segmentation")


def _select_features(df: pd.DataFrame, schema: dict) -> list:
    """Pick numeric + revenue columns for clustering."""
    cols = [c for c, t in schema.items()
            if t in ("revenue", "numeric") and c in df.columns]
    # Deduplicate & drop very-low-variance cols
    keep = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) > 0 and s.std() > 0:
            keep.append(c)
    return keep


def _prep_matrix(df: pd.DataFrame, features: list):
    """Coerce to numeric, impute, scale."""
    X_raw = df[features].apply(pd.to_numeric, errors="coerce")
    X_raw = X_raw.fillna(X_raw.median(numeric_only=True))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw.values)
    return X_scaled, scaler


def _find_best_k(X_scaled: np.ndarray, max_k: int, seed: int) -> dict:
    """Try k=2..max_k, return best by silhouette."""
    n = len(X_scaled)
    # Prevent k >= n_samples
    upper = min(max_k, max(2, n // 20))
    if upper < 2:
        return {"k": 1, "score": 0.0, "labels": np.zeros(n, dtype=int)}

    best = {"k": 2, "score": -1.0, "labels": None}
    for k in range(2, upper + 1):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=seed)
            labels = km.fit_predict(X_scaled)
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(X_scaled, labels)
            if score > best["score"]:
                best = {"k": k, "score": round(float(score), 4), "labels": labels}
        except Exception as e:
            log.warning(f"KMeans k={k} failed: {e}")
    return best


def _cluster_profiles(df: pd.DataFrame, labels: np.ndarray,
                      features: list, schema: dict) -> pd.DataFrame:
    """Per-cluster stats: size, means, medians."""
    work = df[features].apply(pd.to_numeric, errors="coerce").copy()
    work["_cluster"] = labels

    rows = []
    for cid, group in work.groupby("_cluster"):
        row = {
            "cluster": int(cid),
            "size": int(len(group)),
            "share_pct": round(len(group) / len(work) * 100, 2),
        }
        for f in features:
            row[f"{f}_mean"] = round(float(group[f].mean()), 2)
            row[f"{f}_median"] = round(float(group[f].median()), 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def _label_clusters(profiles: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """
    Assign human labels based on dominant revenue metric.
    Ranks clusters by revenue_mean (or first revenue column).
    """
    rev_feats = [c for c in profiles.columns if c.endswith("_mean")
                 and schema.get(c.replace("_mean", "")) == "revenue"]
    if not rev_feats:
        # fallback: first mean column
        mean_cols = [c for c in profiles.columns if c.endswith("_mean")]
        if not mean_cols:
            profiles["label"] = "Cluster"
            return profiles
        key = mean_cols[0]
    else:
        key = rev_feats[0]

    profiles = profiles.sort_values(key, ascending=False).reset_index(drop=True)
    n = len(profiles)

    # Assign tier labels
    if n == 1:
        labels = ["Overall"]
    elif n == 2:
        labels = ["High-Value", "Low-Value"]
    elif n == 3:
        labels = ["VIP", "Mid-Tier", "Low-Tier"]
    elif n == 4:
        labels = ["VIP", "High", "Mid", "Low"]
    else:
        labels = ["VIP", "High", "Mid", "Low", "Very Low"][:n]
        while len(labels) < n:
            labels.append(f"Tier-{len(labels)+1}")

    profiles["label"] = labels
    return profiles


def segment(df: pd.DataFrame, schema: dict, config: dict = None) -> dict:
    """
    Auto-segment rows via KMeans.
    Returns:
      {
        "clusters": df with '_cluster' column (or None),
        "profiles": DataFrame with per-cluster stats,
        "k": int,
        "silhouette": float,
        "features_used": [list],
        "label_map": {cluster_id: label}
      }
    """
    config = config or {}
    scfg = config.get("segmentation", {})
    max_k = scfg.get("max_k", 6)
    min_rows = scfg.get("min_rows", 20)
    seed = scfg.get("random_state", 42)

    log.info(f"Segmenting {len(df):,} rows (max_k={max_k})")

    features = _select_features(df, schema)
    if not features:
        log.warning("No numeric features for segmentation")
        return {"clusters": None, "profiles": None, "k": 0,
                "silhouette": 0.0, "features_used": [], "label_map": {}}

    if len(df) < min_rows:
        log.warning(f"Too few rows ({len(df)}) for segmentation (min={min_rows})")
        return {"clusters": None, "profiles": None, "k": 0,
                "silhouette": 0.0, "features_used": features, "label_map": {}}

    # Prep + cluster
    X_scaled, _ = _prep_matrix(df, features)
    best = _find_best_k(X_scaled, max_k, seed)

    if best["k"] < 2 or best["labels"] is None:
        log.warning("Could not find meaningful clusters")
        return {"clusters": None, "profiles": None, "k": 1,
                "silhouette": 0.0, "features_used": features, "label_map": {}}

    # Build output
    df_out = df.copy()
    df_out["_cluster"] = best["labels"]

    profiles = _cluster_profiles(df_out, best["labels"], features, schema)
    profiles = _label_clusters(profiles, schema)

    label_map = dict(zip(profiles["cluster"], profiles["label"]))

    log.info(f"Best k={best['k']}, silhouette={best['score']}, "
             f"features={features}")

    return {
        "clusters": df_out,
        "profiles": profiles,
        "k": best["k"],
        "silhouette": best["score"],
        "features_used": features,
        "label_map": label_map,
    }


def print_segment_report(result: dict) -> None:
    """Pretty-print segmentation summary."""
    print(f"\n{'='*70}")
    print(f"🧩 SEGMENTATION REPORT")
    print(f"{'='*70}")

    if result["k"] < 2 or result["profiles"] is None:
        print("⚠️  Segmentation not possible (insufficient data or no variance)")
        print(f"{'='*70}\n")
        return

    print(f"Clusters found:   {result['k']}")
    print(f"Silhouette score: {result['silhouette']}  "
          f"({'excellent' if result['silhouette'] > 0.5 else 'good' if result['silhouette'] > 0.3 else 'fair'})")
    print(f"Features used:    {', '.join(result['features_used'])}")
    print(f"{'-'*70}")

    profiles = result["profiles"]
    print(f"\n{'Cluster':<10} {'Label':<12} {'Size':>8} {'Share%':>8}")
    print(f"{'-'*70}")
    for _, row in profiles.iterrows():
        print(f"{int(row['cluster']):<10} {row['label']:<12} "
              f"{int(row['size']):>8,} {row['share_pct']:>8.2f}")

    # Feature breakdown
    print(f"\n📊 CLUSTER PROFILES (means):")
    mean_cols = [c for c in profiles.columns if c.endswith("_mean")]
    for _, row in profiles.iterrows():
        print(f"\n   Cluster {int(row['cluster'])} [{row['label']}]:")
        for mc in mean_cols:
            feat = mc.replace("_mean", "")
            print(f"      {feat:<20} {row[mc]:>14,.2f}")

    print(f"{'='*70}\n")


__all__ = ["segment", "print_segment_report"]
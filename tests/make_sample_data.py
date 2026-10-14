"""Generate realistic business sample data for testing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

np.random.seed(42)

dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
regions = ["North", "South", "East", "West"]
categories = ["Electronics", "Clothing", "Food", "Home"]
products = [f"P{i:03d}" for i in range(1, 21)]

rows = []
for d in dates:
    n_orders = np.random.randint(5, 25)
    for _ in range(n_orders):
        rows.append({
            "order_id": f"ORD{len(rows)+1:06d}",
            "order_date": d,
            "region": np.random.choice(regions),
            "category": np.random.choice(categories),
            "product": np.random.choice(products),
            "quantity": np.random.randint(1, 10),
            "unit_price": round(np.random.uniform(10, 500), 2),
            "revenue": 0.0,
            "customer_age": np.random.randint(18, 70),
        })

df = pd.DataFrame(rows)
df["revenue"] = (df["quantity"] * df["unit_price"]).round(2)

# Inject anomalies for later testing
anomaly_idx = np.random.choice(df.index, 15, replace=False)
df.loc[anomaly_idx, "revenue"] *= 15

# Inject missing values
miss_idx = np.random.choice(df.index, 50, replace=False)
df.loc[miss_idx, "customer_age"] = np.nan

Path("data").mkdir(exist_ok=True)
out = "data/sample_business.csv"
df.to_csv(out, index=False)
print(f"✅ Sample data created: {out}")
print(f"   Rows: {len(df):,} | Columns: {len(df.columns)}")
print(f"\n   Preview:\n{df.head(3).to_string()}")
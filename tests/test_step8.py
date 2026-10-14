"""Verify forecasting works with Prophet + fallbacks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml
from core.ingestion import read_any_file, build_schema
from core.cleaning import auto_clean
from core.kpi_engine import discover_kpis
from core.forecasting import forecast_series, print_forecast_report
from utils.logger import get_logger

log = get_logger("test_step8")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Prepare data
df = read_any_file("data/sample_business.csv")["data"]
schema = build_schema(df)
df_clean, _ = auto_clean(df, schema, config)

# Get trend series from KPIs
kpis = discover_kpis(df_clean, schema, config)
assert "trend_series" in kpis, "trend_series missing — check Step 7"

ts = kpis["trend_series"]
print(f"\n📥 Input series: {len(ts)} months "
      f"({ts.index[0].date()} → {ts.index[-1].date()})")

# Forecast
result = forecast_series(ts, periods=6, config=config)
print_forecast_report(result)

# Assertions
fc = result["forecast"]
assert len(fc) == 6, f"Expected 6 forecast rows, got {len(fc)}"
assert result["method"] in ("prophet", "linear", "rolling_mean"), "Unknown method"
assert (fc["yhat"] >= 0).all(), "Forecast should be non-negative"
assert (fc["yhat_upper"] >= fc["yhat"]).all(), "Upper bound must be >= yhat"
assert (fc["yhat_lower"] <= fc["yhat"]).all(), "Lower bound must be <= yhat"
assert fc["ds"].is_monotonic_increasing, "Dates must be ascending"

print(f"🎉 STEP 8 PASSED — Forecasting working (method: {result['method']})!")
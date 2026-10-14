"""Verify anomaly detection works."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from core.ingestion import read_any_file, build_schema
from core.cleaning import auto_clean
from core.anomaly import detect_anomalies, print_anomaly_report
from utils.logger import get_logger

log = get_logger("test_step9")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

df = read_any_file("data/sample_business.csv")["data"]
schema = build_schema(df)
# NOTE: raw data use karo, cleaned nahi — kyunki cleaning ne outliers clip kar diye
# Hum raw pe anomalies detect karenge, taake wo visible hon

result = detect_anomalies(df, schema, config)
print_anomaly_report(result)

# Assertions
s = result["summary"]
assert s["row_anomaly_count"] > 0, "Should detect some anomalies"
assert s["row_anomaly_count"] >= 100, \
    f"Expected >=100 anomalies (5% of 10K rows), got {s['row_anomaly_count']}"

# Check structure
ra = result["row_anomalies"]
assert "_anomaly_score" in ra.columns, "Missing score column"
assert "_anomaly_severity" in ra.columns, "Missing severity column"
assert ra["_anomaly_severity"].is_monotonic_decreasing, "Not sorted by severity"

# TS anomalies
ts = result["ts_anomalies"]
if not ts.empty:
    assert "z_score" in ts.columns, "TS anomalies missing z_score"
    assert set(ts["type"].unique()).issubset({"spike", "dip"}), "Invalid type"

print("🎉 STEP 9 PASSED — Anomaly Detection working!")
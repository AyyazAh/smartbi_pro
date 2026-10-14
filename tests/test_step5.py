"""Verify profiling works on sample CSV."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ingestion import read_any_file, build_schema
from core.profiling import profile, print_report
from utils.logger import get_logger

log = get_logger("test_step5")

# Load + schema
df = read_any_file("data/sample_business.csv")["data"]
schema = build_schema(df)

# Profile
report = profile(df, schema)
print_report(report)

# Assertions
assert report["rows"] == len(df), "Row count mismatch"
assert report["cols"] == len(df.columns), "Col count mismatch"
assert 0 <= report["quality_score"] <= 100, "Score must be 0-100"
assert "customer_age" in report["columns"], "customer_age missing"
assert report["columns"]["customer_age"]["missing"] > 0, "Should have missing values (injected)"

print("🎉 STEP 5 PASSED — Profiling + Quality Score working!")
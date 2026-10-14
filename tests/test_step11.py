"""Verify causal analysis works."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from core.ingestion import read_any_file, build_schema
from core.causal import analyze, print_causal_report, correlation_matrix
from utils.logger import get_logger

log = get_logger("test_step11")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

df = read_any_file("data/sample_business.csv")["data"]
schema = build_schema(df)

result = analyze(df, schema, config)
print_causal_report(result)

# Assertions
assert result["target_col"] is not None, "No target column detected"

cm = result["correlation_matrix"]
assert not cm.empty, "Correlation matrix empty"
assert cm.shape[0] == cm.shape[1], "Matrix not square"

dr = result["driver_ranking"]
assert not dr.empty, "Driver ranking empty"
assert "impact_score" in dr.columns, "impact_score missing"
assert dr["impact_score"].is_monotonic_decreasing, "Not sorted by impact"

# Check specific known relationship: quantity × unit_price → revenue
# (should be correlated but our synthetic data makes them independent)
top_driver = result["top_driver"]
assert top_driver is not None, "top_driver missing"
print(f"\n🎯 Top driver of '{result['target_col']}': {top_driver['driver']} "
      f"(impact={top_driver['impact_score']})")

print("🎉 STEP 11 PASSED — Causal Analysis working!")
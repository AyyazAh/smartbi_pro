"""Verify cleaning works + improves quality score."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from core.ingestion import read_any_file, build_schema
from core.profiling import profile
from core.cleaning import auto_clean, print_cleaning_log
from utils.logger import get_logger

log = get_logger("test_step6")

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Load + schema
df_raw = read_any_file("data/sample_business.csv")["data"]
schema = build_schema(df_raw)

# Profile BEFORE
print("\n### BEFORE CLEANING ###")
before = profile(df_raw, schema)
print(f"Quality score: {before['quality_score']}")

# Clean
df_clean, clog = auto_clean(df_raw, schema, config)
print_cleaning_log(clog)

# Profile AFTER
print("### AFTER CLEANING ###")
after = profile(df_clean, schema)
print(f"Quality score: {after['quality_score']}")

# Assertions
assert after["quality_score"] >= before["quality_score"], \
    f"Quality dropped! {before['quality_score']} -> {after['quality_score']}"
assert df_clean["customer_age"].isna().sum() == 0, "Missing values not imputed"
assert df_clean["order_date"].dtype.kind == "M", "order_date not datetime"
assert len(df_clean) <= len(df_raw), "Rows increased after cleaning (unexpected)"

print(f"\n📈 Score change: {before['quality_score']} → {after['quality_score']} "
      f"(+{after['quality_score'] - before['quality_score']:.2f})")
print("🎉 STEP 6 PASSED — Auto-cleaning working!")
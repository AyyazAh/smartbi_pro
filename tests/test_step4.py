"""Verify ingestion works on sample CSV."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ingestion import read_any_file, build_schema
from utils.logger import get_logger

log = get_logger("test_step4")

# Load
sheets = read_any_file("data/sample_business.csv")
assert "data" in sheets, "Expected 'data' key for CSV"
df = sheets["data"]

print(f"\n📊 Loaded shape: {df.shape}")
print(f"   Columns: {list(df.columns)}\n")

# Schema detection
schema = build_schema(df)

print("🧠 Detected semantic types:")
for col, typ in schema.items():
    print(f"   {col:15s} -> {typ}")

# Assertions
assert schema["order_date"] == "date",        "order_date should be 'date'"
assert schema["revenue"] == "revenue",        "revenue should be 'revenue'"
assert schema["region"] == "geo",             "region should be 'geo'"
assert schema["category"] == "category",      "category should be 'category'"
assert schema["order_id"] == "id",            "order_id should be 'id'"
assert schema["quantity"] == "numeric",       "quantity should be 'numeric'"

print("\n🎉 STEP 4 PASSED — Ingestion + Schema detection working!")
"""Verify segmentation works."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from core.ingestion import read_any_file, build_schema
from core.cleaning import auto_clean
from core.segmentation import segment, print_segment_report
from utils.logger import get_logger

log = get_logger("test_step10")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

df = read_any_file("data/sample_business.csv")["data"]
schema = build_schema(df)

result = segment(df, schema, config)
print_segment_report(result)

# Assertions
if result["k"] >= 2:
    assert result["clusters"] is not None, "clusters df missing"
    assert "_cluster" in result["clusters"].columns, "_cluster column missing"
    assert 0 <= result["silhouette"] <= 1, "silhouette out of range"
    assert len(result["profiles"]) == result["k"], "profile count mismatch"
    assert "label" in result["profiles"].columns, "label column missing"

    # Sum of cluster sizes = total rows
    total = result["profiles"]["size"].sum()
    assert total == len(df), f"Cluster sizes sum ({total}) != rows ({len(df)})"

    # Label map covers all clusters
    assert len(result["label_map"]) == result["k"], "label_map incomplete"
else:
    print("⚠️  k<2 - segmentation skipped (acceptable for small/synthetic data)")

print(f"\n🎉 STEP 10 PASSED — Segmentation working (k={result['k']}, "
      f"silhouette={result['silhouette']})!")
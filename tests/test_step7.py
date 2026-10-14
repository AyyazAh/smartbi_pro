"""Verify KPI engine discovers business KPIs automatically."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from core.ingestion import read_any_file, build_schema
from core.cleaning import auto_clean
from core.kpi_engine import discover_kpis, print_kpi_report
from utils.logger import get_logger

log = get_logger("test_step7")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

df = read_any_file("data/sample_business.csv")["data"]
schema = build_schema(df)
df_clean, _ = auto_clean(df, schema, config)

kpis = discover_kpis(df_clean, schema, config)
print_kpi_report(kpis)

# Assertions
assert "total_revenue" in kpis, "total_revenue KPI missing"
assert "mom_growth_pct" in kpis, "mom_growth_pct KPI missing"
assert "trend_series" in kpis, "trend_series missing"
assert "top_region" in kpis, "top_region KPI missing"
assert "top_category" in kpis, "top_category KPI missing"
assert "pareto" in kpis, "pareto missing"
assert kpis["total_revenue"] > 0, "Total revenue should be positive"
assert len(kpis["top_region"]) <= 5, "Top regions should be <= 5"

print("🎉 STEP 7 PASSED — KPI Engine working!")
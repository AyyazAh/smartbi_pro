"""End-to-end test: full pipeline → insights."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from core.ingestion import read_any_file, build_schema
from core.profiling import profile
from core.cleaning import auto_clean
from core.kpi_engine import discover_kpis
from core.forecasting import forecast_series
from core.anomaly import detect_anomalies
from core.segmentation import segment
from core.causal import analyze as causal_analyze
from core.insight_generator import (
    generate_insights, print_insights, insights_to_markdown
)
from utils.logger import get_logger

log = get_logger("test_step12")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# ---- Full pipeline ----
df_raw = read_any_file("data/sample_business.csv")["data"]
schema = build_schema(df_raw)

prof = profile(df_raw, schema)
df_clean, _ = auto_clean(df_raw, schema, config)
kpis = discover_kpis(df_clean, schema, config)

forecast_result = None
if "trend_series" in kpis:
    forecast_result = forecast_series(kpis["trend_series"], periods=6, config=config)

anomaly_result = detect_anomalies(df_raw, schema, config)
seg_result = segment(df_clean, schema, config)
causal_result = causal_analyze(df_clean, schema, config)

# ---- Insights ----
bundle = generate_insights(
    profile=prof,
    kpis=kpis,
    forecast_result=forecast_result,
    anomaly_result=anomaly_result,
    seg_result=seg_result,
    causal_result=causal_result,
    decisions=None,   # Step 13 ke baad
)

print_insights(bundle)

# ---- Assertions ----
assert "executive_summary" in bundle, "Missing executive_summary"
assert len(bundle["executive_summary"]) >= 10, "Summary too short"
assert bundle["headline"], "Headline empty"

# Section presence
sections = bundle["sections"]
assert len(sections["data_quality"]) > 0, "data_quality empty"
assert len(sections["revenue"]) > 0, "revenue empty"
assert len(sections["contributors"]) > 0, "contributors empty"
assert len(sections["forecast"]) > 0, "forecast empty"
assert len(sections["anomalies"]) > 0, "anomalies empty"
assert len(sections["segments"]) > 0, "segments empty"
assert len(sections["causal"]) > 0, "causal empty"

# Markdown export
md = insights_to_markdown(bundle)
assert "Executive Summary" in md, "Markdown export broken"
assert len(md) > 200, "Markdown too short"

# Save for preview
Path("outputs").mkdir(exist_ok=True)
with open("outputs/insights_preview.md", "w", encoding="utf-8") as f:
    f.write(md)

print(f"📄 Insights saved to outputs/insights_preview.md "
      f"({len(md)} chars)")
print("🎉 STEP 12 PASSED — Insight Generator working!")
"""Full pipeline including Decision Engine."""
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
from core.decision_engine import make_decisions, print_decision_report
from utils.logger import get_logger

log = get_logger("test_step13")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# ---- Full pipeline ----
df_raw = read_any_file("data/sample_business.csv")["data"]
schema = build_schema(df_raw)

prof = profile(df_raw, schema)
df_clean, _ = auto_clean(df_raw, schema, config)
kpis = discover_kpis(df_clean, schema, config)

forecast_result = forecast_series(kpis["trend_series"], 6, config) \
    if "trend_series" in kpis else None
anomaly_result = detect_anomalies(df_raw, schema, config)
seg_result = segment(df_clean, schema, config)
causal_result = causal_analyze(df_clean, schema, config)

# ---- Decisions ----
decisions = make_decisions(
    kpis=kpis,
    profile=prof,
    forecast_result=forecast_result,
    anomaly_result=anomaly_result,
    seg_result=seg_result,
    causal_result=causal_result,
    config=config,
)

print_decision_report(decisions)

# ---- Assertions ----
assert isinstance(decisions, list), "decisions must be a list"
assert len(decisions) >= 2, f"Expected at least 2 decisions, got {len(decisions)}"

for d in decisions:
    assert "priority" in d and d["priority"] in ("HIGH", "MEDIUM", "LOW"), \
        f"Invalid priority: {d.get('priority')}"
    assert "action" in d and len(d["action"]) > 10, "Action too short"
    assert 0 <= d["confidence"] <= 1, f"Confidence out of range: {d['confidence']}"
    assert "expected_impact" in d, "Missing expected_impact"
    assert "category" in d, "Missing category"
    assert "evidence" in d, "Missing evidence"

# Priority ordering check
order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
ranks = [order[d["priority"]] for d in decisions]
assert ranks == sorted(ranks), "Decisions not sorted by priority"

print(f"✅ Generated {len(decisions)} decisions")
print(f"🎉 STEP 13 PASSED — Decision Engine working!")
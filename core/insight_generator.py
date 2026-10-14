"""Insight Generator: convert all engine outputs into NL business narratives."""
from datetime import datetime
from utils.logger import get_logger

log = get_logger("insight_generator")


# ---------- Helpers ----------

def _fmt_currency(v: float) -> str:
    """Format number as currency-ish string (no symbol)."""
    if v is None:
        return "N/A"
    v = float(v)
    if abs(v) >= 1e7:
        return f"{v/1e7:.2f} Cr"
    if abs(v) >= 1e5:
        return f"{v/1e5:.2f} L"
    if abs(v) >= 1e3:
        return f"{v/1e3:.2f} K"
    return f"{v:.2f}"


def _pct(v: float) -> str:
    if v is None:
        return "N/A"
    return f"{v:+.2f}%"


def _severity_from_score(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Fair"
    if score >= 40:
        return "Poor"
    return "Critical"


def _severity_emoji(sev: str) -> str:
    return {
        "Excellent": "🟢",
        "Good": "🟢",
        "Fair": "🟡",
        "Poor": "🟠",
        "Critical": "🔴",
    }.get(sev, "⚪")


# ---------- Section generators ----------

def _data_quality_section(profile: dict) -> list:
    out = []
    if not profile:
        return ["📊 Data quality: insufficient info."]

    score = profile.get("quality_score", 0)
    sev = _severity_from_score(score)
    emoji = _severity_emoji(sev)
    rows = profile.get("rows", 0)
    cols = profile.get("cols", 0)

    out.append(
        f"{emoji} **Data Quality: {score}/100 ({sev})** — "
        f"{rows:,} rows × {cols} columns analyzed."
    )

    issues = profile.get("issues", [])
    if issues:
        high = [i for i in issues if i["severity"] == "HIGH"]
        med  = [i for i in issues if i["severity"] == "MEDIUM"]
        if high:
            names = ", ".join(i["column"] for i in high[:3])
            out.append(f"   ⚠️ {len(high)} high-severity issues: {names}")
        if med:
            out.append(f"   ⚡ {len(med)} medium-severity issues detected.")
    else:
        out.append("   ✅ No major data quality issues.")

    return out


def _revenue_section(kpis: dict) -> list:
    out = []
    if not kpis or "total_revenue" not in kpis:
        return ["💰 Revenue data unavailable."]

    total = kpis["total_revenue"]
    avg = kpis.get("avg_revenue", 0)
    median = kpis.get("median_revenue", 0)

    out.append(
        f"💰 **Total Revenue: {_fmt_currency(total)}** "
        f"(avg per row: {_fmt_currency(avg)}, median: {_fmt_currency(median)})"
    )

    if "mom_growth_pct" in kpis:
        g = kpis["mom_growth_pct"]
        arrow = "📈" if g > 0 else ("📉" if g < 0 else "➡️")
        direction = "grew" if g > 0 else ("declined" if g < 0 else "stayed flat")
        out.append(f"   {arrow} MoM revenue {direction} **{abs(g):.2f}%**.")

    if "best_month" in kpis and "worst_month" in kpis:
        bm = kpis["best_month"]
        wm = kpis["worst_month"]
        out.append(
            f"   🏆 Best month: **{bm['month']}** ({_fmt_currency(bm['value'])}) | "
            f"🔻 Worst: **{wm['month']}** ({_fmt_currency(wm['value'])})"
        )

    pc = kpis.get("period_comparison")
    if pc:
        chg = pc.get("change_pct", 0)
        trend_word = "improved" if chg > 0 else "slowed"
        out.append(
            f"   📊 Second half {trend_word} by **{abs(chg):.2f}%** "
            f"vs first half."
        )

    return out


def _top_contributors_section(kpis: dict) -> list:
    out = []
    if not kpis:
        return out

    dims = [k for k in kpis if k.startswith("top_")]
    if not dims:
        return out

    out.append("🏆 **Top Contributors**")
    for key in dims[:3]:
        dim = key.replace("top_", "")
        top = kpis[key]
        if not top:
            continue
        first_name = next(iter(top))
        first_val = top[first_name]
        out.append(
            f"   • By {dim}: **{first_name}** leads with "
            f"{_fmt_currency(first_val)}"
        )

    pareto = kpis.get("pareto")
    if pareto:
        pct = pareto.get("top_n_pct_of_values", 0)
        share = pareto.get("revenue_share_pct", 0)
        if share > 60:
            out.append(
                f"   📐 Concentration: top {pct}% of "
                f"'{pareto['dimension']}' drives **{share}%** of revenue."
            )
    return out


def _forecast_section(forecast_result: dict) -> list:
    out = []
    if not forecast_result or not forecast_result.get("forecast") is not None:
        return out

    fc = forecast_result["forecast"]
    method = forecast_result.get("method", "unknown")
    periods = forecast_result.get("periods", len(fc))

    if fc.empty:
        return out

    total = fc["yhat"].sum()
    avg = fc["yhat"].mean()
    first = fc["yhat"].iloc[0]
    last = fc["yhat"].iloc[-1]
    trend_pct = ((last - first) / first * 100) if first else 0

    trend_word = "growth" if trend_pct > 0 else ("decline" if trend_pct < 0 else "stability")
    arrow = "📈" if trend_pct > 0 else ("📉" if trend_pct < 0 else "➡️")

    out.append(
        f"🔮 **Forecast (next {periods} months, via {method})**: "
        f"projected total **{_fmt_currency(total)}** "
        f"(avg {_fmt_currency(avg)}/month)."
    )
    out.append(
        f"   {arrow} Trend across horizon: **{trend_pct:+.2f}%** ({trend_word})."
    )
    return out


def _anomaly_section(anomaly_result: dict) -> list:
    out = []
    if not anomaly_result:
        return out

    s = anomaly_result.get("summary", {})
    row_cnt = s.get("row_anomaly_count", 0)
    ts_cnt = s.get("ts_anomaly_count", 0)
    cat_cnt = s.get("cat_spike_count", 0)

    if row_cnt == 0 and ts_cnt == 0 and cat_cnt == 0:
        out.append("⚠️ **Anomalies**: none detected — data looks healthy.")
        return out

    out.append(
        f"⚠️ **Anomalies Detected**: "
        f"{row_cnt:,} row-level, {ts_cnt} time-series, {cat_cnt} category spikes."
    )

    # Top anomaly context
    ra = anomaly_result.get("row_anomalies")
    if ra is not None and not ra.empty:
        top = ra.iloc[0]
        region = top.get("region", "?")
        cat = top.get("category", "?")
        out.append(
            f"   🔴 Highest severity record from region **{region}**, "
            f"category **{cat}** — investigate operations."
        )

    return out


def _segmentation_section(seg_result: dict) -> list:
    out = []
    if not seg_result or seg_result.get("k", 0) < 2:
        return out

    k = seg_result["k"]
    sil = seg_result.get("silhouette", 0)
    profiles = seg_result.get("profiles")

    if profiles is None or profiles.empty:
        return out

    quality = "excellent" if sil > 0.5 else "good" if sil > 0.3 else "fair"
    out.append(
        f"🧩 **Segments Discovered**: **{k} natural groups** "
        f"(silhouette {sil:.3f} — {quality})."
    )

    # Top segment insights
    for _, row in profiles.head(3).iterrows():
        out.append(
            f"   • **{row['label']}**: {int(row['size']):,} records "
            f"({row['share_pct']:.1f}%)"
        )
    return out


def _causal_section(causal_result: dict) -> list:
    out = []
    if not causal_result:
        return out

    target = causal_result.get("target_col")
    top = causal_result.get("top_driver")

    if not target or not top:
        return out

    corr = top.get("correlation", 0)
    driver = top.get("driver", "?")
    sig = top.get("significant", False)
    lag = int(top.get("best_lag", 0))

    # Strength descriptor
    if abs(corr) > 0.7:
        strength = "very strong"
    elif abs(corr) > 0.5:
        strength = "strong"
    elif abs(corr) > 0.3:
        strength = "moderate"
    else:
        strength = "weak"

    sig_mark = "statistically significant" if sig else "not statistically significant"

    out.append(
        f"🔗 **Causal Discovery**: '{driver}' is the {strength} driver "
        f"of '{target}' (r={corr:.2f}, lag={lag}) — {sig_mark}."
    )
    return out


def _action_section(decisions: list) -> list:
    out = []
    if not decisions:
        out.append("🎯 **Actions**: no critical actions needed.")
        return out

    out.append(f"🎯 **Recommended Actions ({len(decisions)})**:")
    for d in decisions[:3]:
        pr = d.get("priority", "MEDIUM")
        action = d.get("action", "")
        conf = d.get("confidence", 0)
        out.append(f"   • [{pr}] {action} (confidence: {conf:.2f})")
    return out


# ---------- Master ----------

def generate_insights(profile: dict, kpis: dict, forecast_result: dict = None,
                      anomaly_result: dict = None, seg_result: dict = None,
                      causal_result: dict = None, decisions: list = None,
                      ) -> dict:
    """
    Compose all sections into a single insight bundle.

    Returns:
      {
        "executive_summary": [list of markdown lines],
        "sections": {name: [lines]},
        "generated_at": ISO timestamp,
        "headline": str,
      }
    """
    log.info("Generating insights narrative")

    sections = {
        "data_quality": _data_quality_section(profile),
        "revenue":      _revenue_section(kpis),
        "contributors": _top_contributors_section(kpis),
        "forecast":     _forecast_section(forecast_result),
        "anomalies":    _anomaly_section(anomaly_result),
        "segments":     _segmentation_section(seg_result),
        "causal":       _causal_section(causal_result),
        "actions":      _action_section(decisions or []),
    }

    # Headline — most important thing
    headline = _compose_headline(kpis, profile)

    # Executive summary = concatenation
    summary_lines = [f"## 🎯 Executive Summary — {headline}"]
    for name, lines in sections.items():
        if lines:
            summary_lines.append("")
            summary_lines.extend(lines)

    bundle = {
        "executive_summary": summary_lines,
        "sections": sections,
        "generated_at": datetime.utcnow().isoformat(),
        "headline": headline,
    }

    log.info(f"Insights generated: {len(summary_lines)} lines")
    return bundle


def _compose_headline(kpis: dict, profile: dict) -> str:
    """Single-line headline for the whole report."""
    if not kpis:
        return "Business Snapshot"

    total = kpis.get("total_revenue", 0)
    growth = kpis.get("mom_growth_pct")

    if total and growth is not None:
        trend = "growing" if growth > 0 else ("declining" if growth < 0 else "stable")
        return (
            f"{_fmt_currency(total)} revenue, {trend} "
            f"({_pct(growth)} MoM)"
        )
    if total:
        return f"{_fmt_currency(total)} total revenue"

    return "Business Snapshot"


def print_insights(bundle: dict) -> None:
    """Pretty print the executive summary."""
    print(f"\n{'='*70}")
    print(f"📝 EXECUTIVE INSIGHTS")
    print(f"{'='*70}")
    for line in bundle["executive_summary"]:
        print(line)
    print(f"{'='*70}\n")


def insights_to_markdown(bundle: dict) -> str:
    """Convert bundle to a plain markdown string (for downloads)."""
    return "\n".join(bundle["executive_summary"])


__all__ = ["generate_insights", "print_insights", "insights_to_markdown"]
"""Decision Engine: convert insights into ranked business decisions."""
from utils.logger import get_logger

log = get_logger("decision_engine")


# ---------- Helpers ----------

def _decision(priority: str, category: str, action: str,
              confidence: float, impact: str, evidence: str) -> dict:
    return {
        "priority": priority,
        "category": category,
        "action": action,
        "confidence": round(float(confidence), 2),
        "expected_impact": impact,
        "evidence": evidence,
    }


def _priority_order(p: str) -> int:
    return {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(p, 3)


# ---------- Individual rule engines ----------

def _growth_decisions(kpis: dict, config: dict) -> list:
    out = []
    if not kpis or "mom_growth_pct" not in kpis:
        return out

    cfg = config.get("decisions", {})
    high_thr = cfg.get("high_priority_threshold", -5.0)
    pos_thr  = cfg.get("growth_positive_threshold", 10.0)

    g = kpis["mom_growth_pct"]

    if g <= high_thr:
        out.append(_decision(
            priority="HIGH",
            category="Revenue Recovery",
            action=("Revenue is declining significantly. Launch retention campaign, "
                    "audit pricing, and investigate churn drivers."),
            confidence=min(0.95, 0.6 + abs(g) / 50),
            impact=f"Recover 3-8% of lost MoM revenue (~{abs(g):.1f}% gap)",
            evidence=f"MoM growth {g:+.2f}% (threshold {high_thr}%)",
        ))
    elif g >= pos_thr:
        out.append(_decision(
            priority="MEDIUM",
            category="Growth Capitalization",
            action=("Strong revenue growth detected. Increase inventory/marketing "
                    "spend to capitalize before momentum slows."),
            confidence=min(0.85, 0.55 + g / 100),
            impact="Compound growth 5-12% next quarter",
            evidence=f"MoM growth {g:+.2f}% (threshold {pos_thr}%)",
        ))

    pc = kpis.get("period_comparison")
    if pc and pc.get("change_pct", 0) < -3:
        out.append(_decision(
            priority="MEDIUM",
            category="Trend Reversal",
            action=("Second-half performance is weaker than first half. "
                    "Reassess strategy and double down on best-performing channels."),
            confidence=0.72,
            impact="Stabilize trajectory, avoid further decline",
            evidence=f"H2 vs H1 change: {pc['change_pct']:+.2f}%",
        ))
    return out


def _anomaly_decisions(anomaly_result: dict) -> list:
    out = []
    if not anomaly_result:
        return out

    s = anomaly_result.get("summary", {})
    row_cnt = s.get("row_anomaly_count", 0)

    total_rows = 0
    ra = anomaly_result.get("row_anomalies")
    if ra is not None and not ra.empty:
        total_rows = len(ra)

    if row_cnt >= 100:
        # Find top regions from anomaly rows
        top_region = None
        if ra is not None and not ra.empty and "region" in ra.columns:
            top_region = ra["region"].value_counts().idxmax()

        region_note = f" Top cluster in region '{top_region}'." if top_region else ""
        out.append(_decision(
            priority="HIGH",
            category="Risk & Integrity",
            action=(f"Investigate {row_cnt:,} anomalous records for potential "
                    f"fraud, data errors, or operational issues.{region_note}"),
            confidence=0.88,
            impact="Prevent revenue leakage; improve data integrity",
            evidence=f"{row_cnt:,} row-level anomalies detected",
        ))
    elif row_cnt > 0:
        out.append(_decision(
            priority="LOW",
            category="Data Monitoring",
            action=f"Review {row_cnt} anomalous records — likely minor outliers.",
            confidence=0.65,
            impact="Prevent small-scale issues from escalating",
            evidence=f"{row_cnt} anomalies",
        ))
    return out


def _forecast_decisions(forecast_result: dict) -> list:
    out = []
    if not forecast_result:
        return out

    fc = forecast_result.get("forecast")
    if fc is None or fc.empty:
        return out

    first = fc["yhat"].iloc[0]
    last = fc["yhat"].iloc[-1]
    if first <= 0:
        return out
    trend = (last - first) / first * 100

    if trend < -10:
        out.append(_decision(
            priority="HIGH",
            category="Strategic Planning",
            action=("Forecast shows significant declining trend. "
                    "Cut discretionary spend, optimize costs, and refocus on "
                    "high-margin segments."),
            confidence=0.78,
            impact="Protect margins; avoid cash flow shock",
            evidence=f"Projected trend {trend:+.2f}% over horizon",
        ))
    elif trend > 10:
        out.append(_decision(
            priority="MEDIUM",
            category="Capacity Planning",
            action=("Forecast shows strong upward trend. Lock in supplier "
                    "contracts early and expand capacity."),
            confidence=0.75,
            impact="Cost savings 2-5%; avoid stockouts",
            evidence=f"Projected trend {trend:+.2f}% over horizon",
        ))
    return out


def _segmentation_decisions(seg_result: dict, kpis: dict) -> list:
    out = []
    if not seg_result or seg_result.get("k", 0) < 2:
        return out

    profiles = seg_result.get("profiles")
    if profiles is None or profiles.empty:
        return out

    # Check top segment share
    top = profiles.iloc[0]
    share_pct = top.get("share_pct", 0)
    label = top.get("label", "Top segment")

    # Concentration check
    pareto = (kpis or {}).get("pareto")
    if pareto and pareto.get("revenue_share_pct", 0) > 60:
        out.append(_decision(
            priority="MEDIUM",
            category="Customer Strategy",
            action=(f"Revenue is highly concentrated (top {pareto['top_n_pct_of_values']}% "
                    f"of '{pareto['dimension']}' = {pareto['revenue_share_pct']}%). "
                    f"Build a VIP retention program and diversify revenue base."),
            confidence=0.80,
            impact="Reduce concentration risk by 10-20%",
            evidence=f"Pareto: {pareto['revenue_share_pct']}% concentration",
        ))

    if share_pct > 50:
        out.append(_decision(
            priority="LOW",
            category="Segment Focus",
            action=(f"'{label}' segment dominates ({share_pct:.1f}%). "
                    f"Design tailored offers and measure LTV for this group."),
            confidence=0.68,
            impact="+2-5% engagement from personalization",
            evidence=f"Top segment share: {share_pct:.1f}%",
        ))
    return out


def _causal_decisions(causal_result: dict) -> list:
    out = []
    if not causal_result:
        return out

    top = causal_result.get("top_driver")
    target = causal_result.get("target_col")
    if not top or not target:
        return out

    driver = top.get("driver")
    corr = top.get("correlation", 0)
    lag = int(top.get("best_lag", 0))
    sig = top.get("significant", False)

    if lag > 0 and sig:
        out.append(_decision(
            priority="MEDIUM",
            category="Leading Indicator",
            action=(f"'{driver}' leads '{target}' by {lag} periods "
                    f"(r={corr:.2f}). Build early-warning dashboard on '{driver}' "
                    f"and use it for pre-emptive planning."),
            confidence=0.72,
            impact=f"1-{lag}-period planning advantage",
            evidence=f"Lead-lag correlation {corr:.2f}, lag={lag}",
        ))
    elif abs(corr) > 0.5 and sig:
        out.append(_decision(
            priority="LOW",
            category="Driver Monitoring",
            action=(f"'{driver}' strongly correlates with '{target}' (r={corr:.2f}). "
                    f"Track '{driver}' weekly as a proactive signal."),
            confidence=0.70,
            impact="Earlier detection of revenue shifts",
            evidence=f"Correlation {corr:.2f}",
        ))
    return out


def _quality_decisions(profile: dict) -> list:
    out = []
    if not profile:
        return out

    score = profile.get("quality_score", 100)
    issues = profile.get("issues", [])
    high_issues = [i for i in issues if i.get("severity") == "HIGH"]

    if score < 60:
        out.append(_decision(
            priority="HIGH",
            category="Data Hygiene",
            action=("Data quality is poor. Clean missing values, drop duplicates, "
                    "and re-upload before making business decisions."),
            confidence=0.92,
            impact="Improve decision reliability by 30-50%",
            evidence=f"Quality score {score}/100",
        ))
    elif high_issues:
        cols = ", ".join(i["column"] for i in high_issues[:3])
        out.append(_decision(
            priority="MEDIUM",
            category="Data Hygiene",
            action=f"Fix high-severity issues in columns: {cols}.",
            confidence=0.80,
            impact="Reduce bias in downstream analytics",
            evidence=f"{len(high_issues)} high-severity issues",
        ))
    return out


# ---------- Master ----------

def make_decisions(kpis: dict, profile: dict = None,
                   forecast_result: dict = None,
                   anomaly_result: dict = None,
                   seg_result: dict = None,
                   causal_result: dict = None,
                   config: dict = None) -> list:
    """
    Generate ranked decision list from all engine outputs.
    Returns sorted list of decision dicts.
    """
    config = config or {}
    log.info("Generating decisions")

    decisions = []
    decisions += _quality_decisions(profile)
    decisions += _growth_decisions(kpis, config)
    decisions += _anomaly_decisions(anomaly_result)
    decisions += _forecast_decisions(forecast_result)
    decisions += _segmentation_decisions(seg_result, kpis)
    decisions += _causal_decisions(causal_result)

    # Deduplicate by (category, action) signature
    seen = set()
    unique = []
    for d in decisions:
        key = (d["category"], d["action"][:60])
        if key not in seen:
            seen.add(key)
            unique.append(d)

    # Sort by priority then confidence
    unique.sort(key=lambda d: (_priority_order(d["priority"]),
                               -d["confidence"]))

    log.info(f"Generated {len(unique)} decisions "
             f"(HIGH={sum(1 for d in unique if d['priority']=='HIGH')}, "
             f"MEDIUM={sum(1 for d in unique if d['priority']=='MEDIUM')}, "
             f"LOW={sum(1 for d in unique if d['priority']=='LOW')})")

    return unique


def print_decision_report(decisions: list) -> None:
    """Pretty-print decisions."""
    print(f"\n{'='*70}")
    print(f"🎯 DECISION ENGINE REPORT")
    print(f"{'='*70}")
    print(f"Total decisions: {len(decisions)}")

    if not decisions:
        print("✅ No critical actions required.")
        print(f"{'='*70}\n")
        return

    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for d in decisions:
        counts[d["priority"]] = counts.get(d["priority"], 0) + 1
    print(f"Breakdown: HIGH={counts['HIGH']}, MEDIUM={counts['MEDIUM']}, LOW={counts['LOW']}")
    print(f"{'-'*70}")

    for i, d in enumerate(decisions, 1):
        pr_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(d["priority"], "⚪")
        print(f"\n{pr_emoji} [{d['priority']}] {d['category']}  "
              f"(confidence: {d['confidence']:.2f})")
        print(f"   Action:   {d['action']}")
        print(f"   Impact:   {d['expected_impact']}")
        print(f"   Evidence: {d['evidence']}")

    print(f"\n{'='*70}\n")


__all__ = ["make_decisions", "print_decision_report"]
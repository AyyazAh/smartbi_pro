"""SmartBI Pro — Red & Black Executive One-Page Business Intelligence Dashboard."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datetime import datetime
import json
import yaml
import pandas as pd
import streamlit as st

# Core modules
from core.ingestion import read_any_file, build_schema
from core.profiling import profile
from core.cleaning import auto_clean
from core.kpi_engine import discover_kpis
from core.forecasting import forecast_series
from core.anomaly import detect_anomalies
from core.segmentation import segment
from core.causal import analyze as causal_analyze
from core.insight_generator import generate_insights, insights_to_markdown
from core.decision_engine import make_decisions

# UI components
from ui.components import (
    trend_chart, forecast_chart, bar_top_values, pie_segments,
    correlation_heatmap, driver_ranking_chart, anomaly_severity_chart,
)


# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="SmartBI Pro · Executive BI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# Red & Black Theme CSS (Obsidian + Crimson Bento UI)
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #09090b !important;
        color: #f4f4f5 !important;
    }

    /* Hide default Streamlit chrome */
    [data-testid="stSidebar"], header, footer, #MainMenu {
        display: none !important;
        visibility: hidden !important;
    }

    /* Maximize viewport usage for one-pager feel */
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
        max-width: 1600px !important;
    }
    .block-container > div > div {
        gap: 0.55rem;
    }

    /* Top Executive Banner (Obsidian to Deep Crimson Gradient) */
    .topbar {
        background: linear-gradient(135deg, #09090b 0%, #18080b 55%, #3b070e 100%);
        border: 1px solid #3f1118;
        border-left: 4px solid #ef4444;
        padding: 16px 24px;
        border-radius: 12px;
        color: #f8fafc;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        box-shadow: 0 8px 24px rgba(220, 38, 38, 0.08);
        margin-bottom: 4px;
    }
    .topbar-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .brand-badge {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.45);
        color: #f87171;
        font-weight: 700;
        font-size: 0.8rem;
        padding: 6px 12px;
        border-radius: 8px;
        letter-spacing: 0.5px;
        white-space: nowrap;
        text-transform: uppercase;
    }
    .topbar-headline {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.2px;
    }
    .topbar-meta {
        font-size: 0.78rem;
        color: #a1a1aa;
        margin-top: 2px;
    }

    /* KPI Strip Styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #141418 0%, #0f0f12 100%);
        border: 1px solid #27272a;
        border-top: 3px solid #ef4444;
        border-radius: 10px;
        padding: 12px 16px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #ef4444;
        box-shadow: 0 6px 18px rgba(239, 68, 68, 0.14);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.74rem !important;
        color: #a1a1aa !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        line-height: 1.2 !important;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
    }

    /* Bento Card Containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #121215 !important;
        border-radius: 12px !important;
        border: 1px solid #27272a !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45);
    }

    /* Bento Card Header */
    .card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
        padding-bottom: 8px;
        border-bottom: 1px solid #27272a;
    }
    .card-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #f4f4f5;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-sub {
        font-size: 0.73rem;
        color: #fca5a5;
        font-weight: 600;
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.28);
        padding: 2px 9px;
        border-radius: 999px;
    }

    /* Priority Summary Pills */
    .priority-bar {
        display: flex;
        gap: 8px;
        margin: 6px 0 10px 0;
    }
    .p-pill {
        flex: 1;
        text-align: center;
        padding: 6px 8px;
        border-radius: 8px;
        font-size: 0.74rem;
        font-weight: 700;
        border: 1px solid transparent;
    }
    .p-pill.high {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border-color: rgba(239, 68, 68, 0.35);
    }
    .p-pill.med {
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border-color: rgba(245, 158, 11, 0.3);
    }
    .p-pill.low {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border-color: rgba(16, 185, 129, 0.3);
    }

    /* Decision Cards (Dark Carbon + Neon Left Accent) */
    .decision-card {
        border-left: 4px solid #52525b;
        padding: 11px 14px;
        background: #18181c;
        border-radius: 8px;
        margin-bottom: 8px;
        border: 1px solid #27272a;
        border-left-width: 4px;
    }
    .decision-card.HIGH {
        border-left-color: #ef4444;
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.08) 0%, #18181c 100%);
    }
    .decision-card.MEDIUM {
        border-left-color: #f59e0b;
    }
    .decision-card.LOW {
        border-left-color: #10b981;
    }

    .decision-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    .decision-card.HIGH .decision-top   { color: #f87171; }
    .decision-card.MEDIUM .decision-top { color: #fbbf24; }
    .decision-card.LOW .decision-top    { color: #34d399; }

    .conf-badge {
        background: #27272a;
        color: #d4d4d8;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.68rem;
    }
    .decision-action {
        margin-top: 5px;
        color: #f4f4f5;
        font-size: 0.84rem;
        font-weight: 500;
        line-height: 1.35;
    }
    .decision-meta {
        margin-top: 5px;
        color: #a1a1aa;
        font-size: 0.73rem;
        line-height: 1.3;
    }
    .decision-meta b {
        color: #e4e4e7;
    }

    /* Tabs Styling (Crimson Active Underline) */
    button[data-baseweb="tab"] {
        padding-top: 6px !important;
        padding-bottom: 6px !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #a1a1aa !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #f87171 !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #ef4444 !important;
    }

    /* Popover, Uploader & Action Buttons */
    div[data-testid="stPopover"] > button,
    .stDownloadButton button {
        background: #18181c !important;
        border: 1px solid #3f3f46 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        width: 100% !important;
        color: #f4f4f5 !important;
        transition: all 0.15s ease;
    }
    div[data-testid="stPopover"] > button:hover,
    .stDownloadButton button:hover {
        border-color: #ef4444 !important;
        color: #ffffff !important;
        background: rgba(239, 68, 68, 0.15) !important;
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.25);
    }
    div[data-baseweb="popover"] {
        background-color: #141418 !important;
        border: 1px solid #27272a !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #18181c !important;
        border: 1px dashed #ef4444 !important;
        color: #e4e4e7 !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Config & Pipeline Caching
# ============================================================
@st.cache_resource
def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


CONFIG = load_config()


@st.cache_data(show_spinner=False, max_entries=3)
def run_pipeline(file_bytes: bytes, file_name: str, _cfg_hash: str):
    tmp = Path("outputs/tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / file_name
    p.write_bytes(file_bytes)

    sheets = read_any_file(str(p))
    sheet_name = list(sheets.keys())[0]
    df_raw = sheets[sheet_name]

    schema = build_schema(df_raw)
    prof = profile(df_raw, schema)
    df_clean, clean_log = auto_clean(df_raw, schema, CONFIG)
    kpis = discover_kpis(df_clean, schema, CONFIG)

    forecast_result = None
    if "trend_series" in kpis:
        try:
            forecast_result = forecast_series(
                kpis["trend_series"], periods=6, config=CONFIG
            )
        except Exception:
            pass

    anomaly_result = detect_anomalies(df_raw, schema, CONFIG)
    seg_result = segment(df_clean, schema, CONFIG)
    causal_result = causal_analyze(df_clean, schema, CONFIG)

    decisions = make_decisions(
        kpis=kpis, profile=prof, forecast_result=forecast_result,
        anomaly_result=anomaly_result, seg_result=seg_result,
        causal_result=causal_result, config=CONFIG,
    )

    insights = generate_insights(
        profile=prof, kpis=kpis, forecast_result=forecast_result,
        anomaly_result=anomaly_result, seg_result=seg_result,
        causal_result=causal_result, decisions=decisions,
    )

    return {
        "df_raw": df_raw, "df_clean": df_clean, "schema": schema,
        "profile": prof, "clean_log": clean_log, "kpis": kpis,
        "forecast": forecast_result, "anomaly": anomaly_result,
        "segmentation": seg_result, "causal": causal_result,
        "decisions": decisions, "insights": insights,
        "file_name": file_name, "sheet": sheet_name,
        "run_at": datetime.now().strftime("%b %d, %Y · %H:%M"),
    }


# ============================================================
# UI & Dark Red Chart Restyling Helpers
# ============================================================
def fmt_money(v):
    if v is None:
        return "—"
    v = float(v)
    if abs(v) >= 1e7:
        return f"{v/1e7:.2f} Cr"
    if abs(v) >= 1e5:
        return f"{v/1e5:.2f} L"
    if abs(v) >= 1e3:
        return f"{v/1e3:.1f} K"
    return f"{v:,.0f}"


def card_header(icon: str, title: str, badge: str = ""):
    badge_html = f'<span class="card-sub">{badge}</span>' if badge else ""
    st.markdown(
        f'<div class="card-header">'
        f'<div class="card-title"><span>{icon}</span><span>{title}</span></div>'
        f'{badge_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def polish_chart(fig, height: int = 260, hide_title: bool = True):
    """Converts any Plotly chart into a Red & Black dark-mode visual."""
    if fig is None:
        return None

    red_palette = ["#ef4444", "#f97316", "#f43f5e", "#eab308", "#a1a1aa", "#71717a"]

    update_kwargs = dict(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=12 if hide_title else 34, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=red_palette,
        font=dict(family="Inter, sans-serif", size=11, color="#d4d4d8"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d4d4d8", size=10),
        ),
    )
    if hide_title:
        update_kwargs["title"] = None

    fig.update_layout(**update_kwargs)
    fig.update_xaxes(
        gridcolor="#27272a",
        zerolinecolor="#27272a",
        tickfont=dict(color="#a1a1aa"),
        title_font=dict(color="#a1a1aa"),
    )
    fig.update_yaxes(
        gridcolor="#27272a",
        zerolinecolor="#27272a",
        tickfont=dict(color="#a1a1aa"),
        title_font=dict(color="#a1a1aa"),
    )

    # Recolor traces to match Crimson / Scarlet / Obsidian palette
    for i, trace in enumerate(fig.data):
        t_type = getattr(trace, "type", "")
        if t_type == "bar":
            trace.marker.color = "#ef4444"
            trace.marker.line = dict(color="#7f1d1d", width=0.5)
        elif t_type == "scatter":
            name_lower = (getattr(trace, "name", "") or "").lower()
            if "forecast" in name_lower:
                trace.line.color = "#f97316"  # Bright orange-red for forecast
            elif "confidence" in name_lower or getattr(trace, "fill", None):
                trace.fillcolor = "rgba(239, 68, 68, 0.16)"
            else:
                trace.line.color = "#ef4444" if i == 0 else red_palette[i % len(red_palette)]
        elif t_type == "pie":
            trace.marker.colors = ["#ef4444", "#27272a", "#f97316", "#7f1d1d", "#52525b"]
            trace.marker.line = dict(color="#121215", width=2)

    return fig


# ============================================================
# Session State Initialization
# ============================================================
if "results" not in st.session_state:
    st.session_state.results = None


# ============================================================
# Empty State (Landing Page Before Upload)
# ============================================================
if st.session_state.results is None:
    st.markdown(
        '<div class="topbar" style="padding:28px 32px; margin-bottom:24px;">'
        '<div>'
        '<div class="brand-badge" style="display:inline-block; margin-bottom:8px;">🔥 SMARTBI PRO</div>'
        '<h1 style="margin:0; font-size:1.75rem; font-weight:700; color:white;">'
        'Autonomous Business Intelligence'
        '</h1>'
        '<p style="margin:6px 0 0 0; color:#a1a1aa; font-size:0.92rem;">'
        'Zero-config · Upload CSV/Excel → Instant KPIs, Forecasts, Anomalies & Executive Decisions'
        '</p>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        with st.container(border=True):
            uploaded_init = st.file_uploader(
                "Drop your CSV or Excel file to launch your one-pager",
                type=["csv", "xlsx", "xls"],
                key="init_uploader",
            )
            if uploaded_init is not None:
                with st.spinner("🧠 Running 8 intelligence engines (~40s on first run)..."):
                    try:
                        cfg_hash = str(hash(str(CONFIG)))
                        st.session_state.results = run_pipeline(
                            uploaded_init.getbuffer().tobytes(),
                            uploaded_init.name,
                            cfg_hash,
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Pipeline failed: {e}")
                        st.stop()

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    chip_cols = st.columns(6)
    chips = [
        ("📊 Auto KPIs", "Revenue & MoM"),
        ("🔮 Prophet Forecast", "6-Month Horizon"),
        ("⚠️ Anomaly Engine", "Row-Level Severity"),
        ("🧩 Auto Segments", "K-Means Clustering"),
        ("🔗 Causal Drivers", "Correlation & Impact"),
        ("🎯 Action Engine", "Prioritized Decisions"),
    ]
    for col, (title, sub) in zip(chip_cols, chips):
        with col:
            st.markdown(
                f"<div style='text-align:center; padding:14px 10px; background:#121215; "
                f"border:1px solid #27272a; border-top:2px solid #ef4444; border-radius:10px;'>"
                f"<div style='font-weight:700; color:#f4f4f5; font-size:0.85rem;'>{title}</div>"
                f"<div style='color:#a1a1aa; font-size:0.74rem; margin-top:3px;'>{sub}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    st.stop()


# ============================================================
# Active Dashboard State (One-Pager View)
# ============================================================
R = st.session_state.results
kpis = R["kpis"]
prof = R["profile"]
ra = R["anomaly"]["row_anomalies"]
cm = R["causal"]["correlation_matrix"]
dr = R["causal"]["driver_ranking"]


# ------------------------------------------------------------
# 1. COMPACT TOP EXECUTIVE BAR + POP-OVER CONTROLS
# ------------------------------------------------------------
top_left, top_right = st.columns([3.4, 1.2])

with top_left:
    st.markdown(
        f'<div class="topbar">'
        f'<div class="topbar-left">'
        f'<span class="brand-badge">🔥 SmartBI Pro</span>'
        f'<div>'
        f'<div class="topbar-headline">🎯 {R["insights"]["headline"]}</div>'
        f'<div class="topbar-meta">'
        f'📁 <b style="color:#e4e4e7;">{R["file_name"]}</b> (Sheet: {R["sheet"]}) · '
        f'{len(R["df_raw"]):,} rows · Analyzed {R["run_at"]}'
        f'</div>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with top_right:
    btn_c1, btn_c2 = st.columns(2)
    with btn_c1:
        with st.popover("🔄 Switch File", use_container_width=True):
            st.markdown("**Upload a new dataset**")
            new_file = st.file_uploader(
                "CSV or Excel",
                type=["csv", "xlsx", "xls"],
                key="switch_uploader",
                label_visibility="collapsed",
            )
            if new_file is not None and new_file.name != R["file_name"]:
                with st.spinner("🧠 Analyzing new dataset..."):
                    cfg_hash = str(hash(str(CONFIG)))
                    st.session_state.results = run_pipeline(
                        new_file.getbuffer().tobytes(),
                        new_file.name,
                        cfg_hash,
                    )
                    st.rerun()

    with btn_c2:
        with st.popover("📥 Export Report", use_container_width=True):
            st.markdown("**Download BI Artifacts**")
            st.download_button(
                "📝 Insights Report (.md)",
                data=insights_to_markdown(R["insights"]),
                file_name=f"smartbi_insights_{datetime.now():%Y%m%d_%H%M}.md",
                mime="text/markdown",
            )
            kpi_json = {k: v for k, v in kpis.items() if isinstance(v, (int, float, str))}
            st.download_button(
                "📊 KPI Summary (.json)",
                data=json.dumps(kpi_json, indent=2, default=str),
                file_name=f"smartbi_kpis_{datetime.now():%Y%m%d_%H%M}.json",
                mime="application/json",
            )
            st.download_button(
                "🧹 Cleaned Dataset (.csv)",
                data=R["df_clean"].to_csv(index=False).encode("utf-8"),
                file_name=f"smartbi_cleaned_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv",
            )
            if R["decisions"]:
                st.download_button(
                    "🎯 Action Plan (.csv)",
                    data=pd.DataFrame(R["decisions"]).to_csv(index=False).encode("utf-8"),
                    file_name=f"smartbi_decisions_{datetime.now():%Y%m%d_%H%M}.csv",
                    mime="text/csv",
                )


# ------------------------------------------------------------
# 2. HIGH-IMPACT KPI STRIP (5 Cards)
# ------------------------------------------------------------
mom = kpis.get("mom_growth_pct", 0)
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("💰 Total Revenue", fmt_money(kpis.get("total_revenue", 0)))
k2.metric(
    "📈 MoM Growth",
    f"{mom:+.2f}%",
    delta=f"{mom:+.2f}% MoM" if mom else None,
    delta_color="normal" if mom >= 0 else "inverse",
)
k3.metric(
    "🛡️ Quality Score",
    f"{prof['quality_score']}/100",
    delta="Strong" if prof["quality_score"] >= 80 else "Needs Review",
    delta_color="normal" if prof["quality_score"] >= 80 else "off",
)
k4.metric(
    "⚠️ Anomalies",
    f"{R['anomaly']['summary']['row_anomaly_count']:,}",
    delta="Flagged rows",
    delta_color="inverse" if R["anomaly"]["summary"]["row_anomaly_count"] > 0 else "normal",
)
k5.metric(
    "🎯 AI Decisions",
    f"{len(R['decisions'])}",
    delta=f"{sum(1 for d in R['decisions'] if d['priority']=='HIGH')} High Priority",
    delta_color="off",
)


# ------------------------------------------------------------
# 3. MAIN ONE-PAGER BENTO GRID (Left: Visuals | Right: Actions)
# ------------------------------------------------------------
left_col, right_col = st.columns([2.15, 1], gap="medium")

# ==================== LEFT COLUMN (CHARTS & DEEP DIVES) ====================
with left_col:

    # --- Top Card: Unified Revenue Trajectory & 6-Month Forecast ---
    with st.container(border=True):
        has_forecast = R["forecast"] is not None
        has_trend = "trend_series" in kpis

        badge_txt = (
            f"{R['forecast']['method'].title()} · {R['forecast']['periods']}M Horizon"
            if has_forecast else "Monthly Trajectory"
        )
        card_header("📈", "Revenue Trajectory & 6-Month Forecast", badge_txt)

        if has_forecast and has_trend:
            t_fcast, t_hist, t_data = st.tabs([
                "🔮 Combined History + Forecast",
                "📊 Monthly Revenue Trend",
                "📋 Forecast Table",
            ])
            with t_fcast:
                fig_f = forecast_chart(R["forecast"]["history"], R["forecast"]["forecast"])
                st.plotly_chart(polish_chart(fig_f, height=255), use_container_width=True)
            with t_hist:
                fig_t = trend_chart(kpis["trend_series"], "Revenue Trend (Monthly)")
                st.plotly_chart(polish_chart(fig_t, height=255), use_container_width=True)
            with t_data:
                st.dataframe(
                    R["forecast"]["forecast"].style.format({
                        "yhat": "{:,.2f}",
                        "yhat_lower": "{:,.2f}",
                        "yhat_upper": "{:,.2f}",
                    }),
                    use_container_width=True,
                    height=240,
                )
        elif has_trend:
            fig_t = trend_chart(kpis["trend_series"], "Revenue Trend (Monthly)")
            st.plotly_chart(polish_chart(fig_t, height=265), use_container_width=True)
        else:
            st.info("No time-series trend column detected in dataset.")

    # --- Bottom Bento Row: Contributors (Left) + Deep-Dive Intelligence (Right) ---
    bento_left, bento_right = st.columns(2, gap="medium")

    # BENTO 1: Top Contributors (Tabbed by Dimension)
    with bento_left:
        with st.container(border=True):
            top_dims = [k for k in kpis if k.startswith("top_")]
            card_header("🏆", "Top Revenue Contributors", f"{len(top_dims)} dimensions")

            if top_dims:
                tab_labels = [f"By {k.replace('top_', '').title()}" for k in top_dims[:4]]
                dim_tabs = st.tabs(tab_labels)
                for tab, key in zip(dim_tabs, top_dims[:4]):
                    with tab:
                        dim_name = key.replace("top_", "").title()
                        fig_bar = bar_top_values(kpis[key], f"Top by {dim_name}")
                        st.plotly_chart(
                            polish_chart(fig_bar, height=250),
                            use_container_width=True,
                        )
            else:
                st.info("No categorical breakdown dimensions available.")

    # BENTO 2: Segments, Causal Drivers & Anomaly Severity (Tabbed)
    with bento_right:
        with st.container(border=True):
            card_header("🧠", "Deep-Dive Intelligence", "Segments · Drivers · Anomalies")

            tab_seg, tab_causal, tab_anom = st.tabs([
                "🧩 Segments",
                "🔗 Key Drivers",
                "⚠️ Anomalies",
            ])

            # Tab 1: Customer / Record Segments
            with tab_seg:
                if R["segmentation"].get("k", 0) >= 2:
                    fig_pie = pie_segments(R["segmentation"]["profiles"])
                    st.plotly_chart(
                        polish_chart(fig_pie, height=215),
                        use_container_width=True,
                    )
                    with st.popover("📋 View Cluster Profile Table", use_container_width=True):
                        st.caption(
                            f"{R['segmentation']['k']} clusters · "
                            f"Silhouette score: {R['segmentation']['silhouette']:.3f}"
                        )
                        st.dataframe(R["segmentation"]["profiles"], use_container_width=True)
                else:
                    st.info("Insufficient numeric features for segmentation.")

            # Tab 2: Causal Drivers & Correlation
            with tab_causal:
                if not dr.empty:
                    fig_dr = driver_ranking_chart(dr)
                    st.plotly_chart(
                        polish_chart(fig_dr, height=215),
                        use_container_width=True,
                    )
                    if not cm.empty:
                        with st.popover("🔍 View Full Correlation Matrix", use_container_width=True):
                            st.plotly_chart(
                                polish_chart(correlation_heatmap(cm), height=320, hide_title=False),
                                use_container_width=True,
                            )
                elif not cm.empty:
                    st.plotly_chart(
                        polish_chart(correlation_heatmap(cm), height=250),
                        use_container_width=True,
                    )
                else:
                    st.info("No causal driver relationships identified.")

            # Tab 3: Anomaly Severity Chart
            with tab_anom:
                if ra.empty:
                    st.success("✅ Zero anomalies detected.")
                else:
                    fig_anom = anomaly_severity_chart(ra, top_n=8)
                    st.plotly_chart(
                        polish_chart(fig_anom, height=215),
                        use_container_width=True,
                    )
                    with st.popover(
                        f"🔴 Inspect Top Flagged Records ({len(ra):,} total)",
                        use_container_width=True,
                    ):
                        cols = [c for c in ra.columns if not c.startswith("_")][:6]
                        st.dataframe(ra[cols].head(25), use_container_width=True, height=300)


# ==================== RIGHT COLUMN (PINNED ACTION CENTER) ====================
with right_col:
    with st.container(border=True):
        card_header("🎯", "Recommended Actions", f"{len(R['decisions'])} decisions")

        if not R["decisions"]:
            st.success("✅ No critical interventions needed right now.")
        else:
            counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
            for d in R["decisions"]:
                counts[d["priority"]] = counts.get(d["priority"], 0) + 1

            # Compact Priority Summary Bar
            st.markdown(
                f'<div class="priority-bar">'
                f'<div class="p-pill high">🔴 {counts["HIGH"]} HIGH</div>'
                f'<div class="p-pill med">🟡 {counts["MEDIUM"]} MEDIUM</div>'
                f'<div class="p-pill low">🟢 {counts["LOW"]} LOW</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Decision Cards List
            for d in R["decisions"]:
                pr = d["priority"]
                emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(pr, "⚪")
                st.markdown(
                    f'<div class="decision-card {pr}">'
                    f'<div class="decision-top">'
                    f'<span>{emoji} {pr} · {d.get("category", "")}</span>'
                    f'<span class="conf-badge">Conf: {d.get("confidence", 0):.2f}</span>'
                    f'</div>'
                    f'<div class="decision-action">{d.get("action", "")}</div>'
                    f'<div class="decision-meta">'
                    f'<b>Impact:</b> {d.get("expected_impact", "")}<br>'
                    f'<b>Evidence:</b> {d.get("evidence", "")}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Compact Top Anomalies Watchlist Card right underneath Decisions
    if not ra.empty:
        with st.container(border=True):
            card_header(
                "🚨",
                "Top Anomaly Watchlist",
                f"{R['anomaly']['summary']['row_anomaly_count']:,} total",
            )
            display_cols = [c for c in ra.columns if not c.startswith("_")][:4]
            st.dataframe(
                ra[display_cols].head(5),
                use_container_width=True,
                height=195,
                hide_index=True,
            )
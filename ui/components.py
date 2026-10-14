"""Reusable Streamlit UI components + Plotly chart builders."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------- Chart Builders ----------

def trend_chart(ts: pd.Series, title: str = "Revenue Trend") -> go.Figure:
    """Line chart of a monthly time series."""
    df = ts.reset_index()
    df.columns = ["date", "value"]
    fig = px.line(df, x="date", y="value", title=title, markers=True)
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title=None, yaxis_title=None,
        hovermode="x unified",
    )
    return fig


def forecast_chart(history: pd.Series, forecast_df: pd.DataFrame) -> go.Figure:
    """Historical line + forecast line with confidence band."""
    fig = go.Figure()

    hist = history.reset_index()
    hist.columns = ["date", "value"]
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["value"],
        mode="lines+markers", name="History",
        line=dict(color="#1f77b4", width=2),
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["ds"], y=forecast_df["yhat"],
        mode="lines+markers", name="Forecast",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=list(forecast_df["ds"]) + list(forecast_df["ds"])[::-1],
        y=list(forecast_df["yhat_upper"]) + list(forecast_df["yhat_lower"])[::-1],
        fill="toself", fillcolor="rgba(255,127,14,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Confidence",
        showlegend=True,
    ))

    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        title="Revenue Forecast",
        hovermode="x unified",
        xaxis_title=None, yaxis_title=None,
    )
    return fig


def bar_top_values(data: dict, title: str, color: str = "#2E86AB") -> go.Figure:
    """Horizontal bar chart of top contributors."""
    df = pd.DataFrame(list(data.items()), columns=["name", "value"])
    df = df.sort_values("value", ascending=True)
    fig = px.bar(df, x="value", y="name", orientation="h",
                 title=title, text="value")
    fig.update_traces(marker_color=color, texttemplate="%{text:,.0f}",
                      textposition="outside")
    fig.update_layout(
        height=max(250, 30 * len(df) + 80),
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
        xaxis_title=None, yaxis_title=None,
    )
    return fig


def pie_segments(profiles: pd.DataFrame) -> go.Figure:
    """Pie chart of segment shares."""
    fig = px.pie(
        profiles, values="size", names="label",
        title="Segment Distribution",
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=True,
    )
    return fig


def correlation_heatmap(corr_mat: pd.DataFrame) -> go.Figure:
    """Heatmap of correlation matrix."""
    fig = px.imshow(
        corr_mat,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
        title="Correlation Matrix",
    )
    fig.update_layout(
        height=max(300, 60 * len(corr_mat)),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def driver_ranking_chart(dr: pd.DataFrame) -> go.Figure:
    """Bar chart of driver impact scores."""
    if dr.empty:
        return go.Figure()
    df = dr.sort_values("impact_score", ascending=True)
    fig = px.bar(
        df, x="impact_score", y="driver",
        orientation="h", title="Driver Impact Ranking",
        text="impact_score",
    )
    fig.update_traces(
        marker_color="#E63946",
        texttemplate="%{text:.3f}",
        textposition="outside",
    )
    fig.update_layout(
        height=max(200, 40 * len(df) + 80),
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
        xaxis_title="Impact Score", yaxis_title=None,
    )
    return fig


def anomaly_severity_chart(anomalies_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Bar chart of top anomalies by severity."""
    if anomalies_df.empty:
        return go.Figure()
    df = anomalies_df.head(top_n).copy()
    # Create label from first few non-underscore cols
    label_cols = [c for c in df.columns if not c.startswith("_")][:2]
    df["_label"] = df[label_cols].astype(str).agg(" | ".join, axis=1)
    df = df.sort_values("_anomaly_severity", ascending=True)

    fig = px.bar(
        df, x="_anomaly_severity", y="_label",
        orientation="h", title=f"Top {top_n} Anomalies (by severity)",
        text="_anomaly_severity",
    )
    fig.update_traces(
        marker_color="#D62828",
        texttemplate="%{text:.3f}",
        textposition="outside",
    )
    fig.update_layout(
        height=max(300, 35 * len(df) + 80),
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
        xaxis_title="Severity", yaxis_title=None,
    )
    return fig


# ---------- Widget Helpers ----------

def metric_row(items: list):
    """Render multiple st.metric in a row. items = [(label, value, delta), ...]"""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value = item[0], item[1]
        delta = item[2] if len(item) > 2 else None
        col.metric(label, value, delta)


def decision_card(d: dict):
    """Render a single decision as a styled card."""
    pr = d.get("priority", "MEDIUM")
    color = {"HIGH": "#D62828", "MEDIUM": "#F77F00", "LOW": "#06A77D"}.get(pr, "#666")
    emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(pr, "⚪")

    st.markdown(
        f"""
        <div style="border-left: 5px solid {color}; padding: 12px 16px;
                    background: #f8f9fa; border-radius: 6px; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between;">
                <span style="font-weight: 600; color: {color};">
                    {emoji} [{pr}] {d.get('category', '')}
                </span>
                <span style="color: #555; font-size: 0.9em;">
                    Confidence: <b>{d.get('confidence', 0):.2f}</b>
                </span>
            </div>
            <div style="margin-top: 8px;">
                <b>Action:</b> {d.get('action', '')}
            </div>
            <div style="margin-top: 6px; color: #333;">
                <b>Impact:</b> {d.get('expected_impact', '')}
            </div>
            <div style="margin-top: 4px; color: #666; font-size: 0.85em;">
                <b>Evidence:</b> {d.get('evidence', '')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def quality_gauge(score: float):
    """Progress bar for quality score."""
    color = "#06A77D" if score >= 75 else ("#F77F00" if score >= 60 else "#D62828")
    st.markdown(
        f"""
        <div style="margin-top: 10px;">
            <div style="display: flex; justify-content: space-between;
                        font-size: 0.9em; color: #333;">
                <span>Data Quality</span>
                <b>{score}/100</b>
            </div>
            <div style="background: #e0e0e0; height: 12px; border-radius: 6px;
                        margin-top: 4px; overflow: hidden;">
                <div style="width: {score}%; height: 100%; background: {color};
                            transition: width 0.3s;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
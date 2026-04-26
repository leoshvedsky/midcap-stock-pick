"""
dashboard/app.py
----------------
Streamlit dashboard for the Stock Alpha Ranker.

Run: streamlit run dashboard/app.py
"""

import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

DATA_PATH = "data/processed/ranked_stocks.csv"

st.set_page_config(
    page_title="Stock Alpha Ranker",
    page_icon="📈",
    layout="wide",
)

# --- Styling ---
st.markdown("""
<style>
    .metric-card {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 16px;
        border-left: 3px solid #00d4aa;
    }
    .risk-flag {
        color: #ff6b6b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Stock Alpha Ranker")
st.caption("Mid-cap stocks ranked by short-term upside propensity score")

# --- Load data ---
@st.cache_data(ttl=3600)
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

if df is None:
    st.error("No ranked data found. Run `python run_pipeline.py` first.")
    st.stop()

# --- Sidebar filters ---
with st.sidebar:
    st.header("Filters")
    sectors = ["All"] + sorted(df["sector"].dropna().unique().tolist()) if "sector" in df.columns else ["All"]
    sector_filter = st.selectbox("Sector", sectors)
    show_risk = st.checkbox("Hide high-risk stocks", value=False)
    top_n = st.slider("Show top N stocks", 5, 50, 20)

# Apply filters
filtered = df.copy()
if sector_filter != "All" and "sector" in filtered.columns:
    filtered = filtered[filtered["sector"] == sector_filter]
if show_risk and "risk_flag" in filtered.columns:
    filtered = filtered[filtered["risk_flag"] == 0]
filtered = filtered.head(top_n)

# --- KPIs ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Stocks in Universe", len(df))
with col2:
    st.metric("Showing", len(filtered))
with col3:
    avg_score = filtered["propensity_score"].mean() if "propensity_score" in filtered.columns else 0
    st.metric("Avg Propensity Score", f"{avg_score:.2f}")
with col4:
    risk_count = filtered["risk_flag"].sum() if "risk_flag" in filtered.columns else 0
    st.metric("⚠️ Risk Flags", int(risk_count))

st.divider()

# --- Main ranked table ---
st.subheader("Ranked Stocks")

display_cols = [c for c in [
    "rank", "ticker", "name", "sector", "price",
    "propensity_score", "return_30d", "rsi_14",
    "analyst_target_upside", "pe_ratio", "earnings_within_30d", "risk_flag"
] if c in filtered.columns]

def color_score(val):
    if isinstance(val, float):
        if val >= 0.7:
            return "background-color: #1a4a1a; color: #7fff7f"
        elif val >= 0.5:
            return "background-color: #3a3a1a; color: #ffff7f"
        else:
            return "background-color: #3a1a1a; color: #ff7f7f"
    return ""

styled = (
    filtered[display_cols]
    .style
    .format({
        "propensity_score": "{:.3f}",
        "return_30d": "{:.1%}",
        "analyst_target_upside": "{:.1%}",
        "pe_ratio": "{:.1f}",
        "price": "${:.2f}",
    }, na_rep="—")
    .applymap(color_score, subset=["propensity_score"] if "propensity_score" in display_cols else [])
)
st.dataframe(styled, use_container_width=True, height=500)

st.divider()

# --- Charts ---
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Propensity Score Distribution")
    if "propensity_score" in df.columns:
        fig = px.histogram(
            df, x="propensity_score", nbins=20,
            color_discrete_sequence=["#00d4aa"],
            template="plotly_dark",
        )
        fig.update_layout(
            xaxis_title="Propensity Score",
            yaxis_title="Count",
            plot_bgcolor="#0e0e1a",
            paper_bgcolor="#0e0e1a",
        )
        st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Score vs 30-Day Return")
    if "propensity_score" in filtered.columns and "return_30d" in filtered.columns:
        fig2 = px.scatter(
            filtered,
            x="return_30d",
            y="propensity_score",
            text="ticker",
            color="risk_flag" if "risk_flag" in filtered.columns else None,
            color_discrete_map={0: "#00d4aa", 1: "#ff6b6b"},
            template="plotly_dark",
            hover_data=["name", "sector", "price"] if "name" in filtered.columns else [],
        )
        fig2.update_traces(textposition="top center", textfont_size=10)
        fig2.update_layout(
            xaxis_title="30-Day Return",
            yaxis_title="Propensity Score",
            xaxis_tickformat=".0%",
            plot_bgcolor="#0e0e1a",
            paper_bgcolor="#0e0e1a",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

# --- Sector breakdown ---
if "sector" in df.columns and "propensity_score" in df.columns:
    st.subheader("Average Propensity by Sector")
    sector_avg = (
        df.groupby("sector")["propensity_score"]
        .mean()
        .sort_values(ascending=True)
        .reset_index()
    )
    fig3 = px.bar(
        sector_avg,
        x="propensity_score",
        y="sector",
        orientation="h",
        color="propensity_score",
        color_continuous_scale=["#ff6b6b", "#ffff7f", "#7fff7f"],
        template="plotly_dark",
    )
    fig3.update_layout(
        xaxis_title="Avg Propensity Score",
        yaxis_title="",
        plot_bgcolor="#0e0e1a",
        paper_bgcolor="#0e0e1a",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig3, use_container_width=True)

st.caption("Data: yfinance · SEC EDGAR · Russell Mid-Cap (IWR) · Scores are not investment advice.")

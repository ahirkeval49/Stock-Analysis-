import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from data_fetcher import fetch_price_history, fetch_fundamentals, fetch_news
from technicals import calculate_technicals
from fundamentals import calculate_dcf
from gemini_brain import generate_agent_analysis

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Project Atlas", page_icon="📈")

# --- Custom CSS (Bloomberg-Lite Theme) [cite: 129] ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    div[data-testid="stMetric"] {
        background-color: #262730;
        border: 1px solid #464B5C;
        padding: 15px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("Project Atlas 🌐")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
run_analysis = st.sidebar.button("Run Deep Analysis")

if run_analysis:
    with st.spinner(f"Agents analyzing {ticker}..."):
        # 1. Fetch Data
        df, msg = fetch_price_history(ticker)
        if df is None:
            st.error(msg)
            st.stop()
            
        overview = fetch_fundamentals(ticker, "OVERVIEW")
        cash_flow = fetch_fundamentals(ticker, "CASH_FLOW")
        news = fetch_news(ticker)
        
        # 2. Process Data
        df = calculate_technicals(df)
        dcf_value = calculate_dcf(cash_flow, overview)
        current_price = df['close'].iloc[-1]
        
        # --- Dashboard ---
        
        # Header Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${current_price:.2f}")
        col2.metric("Intrinsic Value (DCF)", f"${dcf_value}", delta=round(dcf_value - current_price, 2))
        col3.metric("Sector", overview.get("Sector", "N/A"))
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["Technical Chart", "Fundamental Analysis", "Agent Reports"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'))
            fig.update_layout(template='plotly_dark', height=600, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.subheader("Company Overview")
            st.json(overview, expanded=False)
            
        with tab3:
            # The Council of Agents [cite: 79]
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("### 👮 The Value Investor")
                analysis = generate_agent_analysis(
                    "Value Investor (Warren Buffett Persona)", 
                    f"Overview: {str(overview)[:1000]}... \nDCF Value: {dcf_value}", 
                    "Is this company a buy based on fundamentals? Focus on MOAT and Safety."
                )
                st.markdown(analysis)
                
            with col_b:
                st.markdown("### 🤖 The Technician")
                tech_context = df.tail(10).to_string()
                analysis = generate_agent_analysis(
                    "Technical Analyst (Jim Simons Persona)",
                    f"Recent Price Action: {tech_context}",
                    "Analyze the trend. Bullish or Bearish? Look for momentum."
                )
                st.markdown(analysis)

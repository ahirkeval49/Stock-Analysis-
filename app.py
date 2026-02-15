import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from data_fetcher import fetch_price_history, fetch_fundamentals, fetch_news
from technicals import calculate_technicals
from fundamentals import calculate_dcf
from gemini_brain import generate_agent_analysis

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Hedge", page_icon="📈")

# --- Custom CSS (Bloomberg-Lite Theme) ---
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
run_analysis = st.sidebar.button("Run Analysis")

# --- Main App Logic ---
if run_analysis:
    # 1. ALWAYS Fetch Price (Cheap & Essential - 1 API Call)
    with st.spinner(f"Fetching Price Action for {ticker}..."):
        df, msg = fetch_price_history(ticker)
        
        if df is None or df.empty:
            st.error(msg)
            st.stop()
            
        # Calculate Technicals locally (0 API Calls)
        df = calculate_technicals(df)
        current_price = df['close'].iloc[-1]
        
        # Display Basic Metrics immediately
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${current_price:.2f}")
        
        # Display the Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'))
        fig.update_layout(template='plotly_dark', height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    # 2. OPTIONAL: Fetch Fundamentals & News (Expensive - 3 API Calls)
    st.markdown("---")
    st.info("💡 Tip: Check the box below to unlock AI Analysis & Fair Value (Uses 3 API Calls).")
    
    if st.checkbox("Unlock Deep Analysis (Fundamentals + News + AI)"):
        with st.spinner("Fetching financial statements and news..."):
            
            # These calls only happen if the box is checked!
            overview = fetch_fundamentals(ticker, "OVERVIEW")   
            cash_flow = fetch_fundamentals(ticker, "CASH_FLOW") 
            news = fetch_news(ticker)                           
            
            # Process Data
            dcf_value = calculate_dcf(cash_flow, overview)
            
            # --- Deep Dive Dashboard ---
            
            # Expanded Metrics
            col2.metric("Intrinsic Value (DCF)", f"${dcf_value}", delta=round(dcf_value - current_price, 2))
            col3.metric("Sector", overview.get("Sector", "N/A"))
            
            # Tabs for Deep Data
            tab1, tab2 = st.tabs(["Fundamental Analysis", "Agent Reports"])
            
            with tab1:
                st.subheader("Company Overview")
                st.json(overview, expanded=False)
                
            with tab2:
                # The Council of Agents
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

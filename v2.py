import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

# --- Page Config ---
st.set_page_config(page_title="Titan: AI Investment Terminal", layout="wide", initial_sidebar_state="collapsed")

# --- CSS for Bloomberg Terminal Aesthetic ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .stMetric { background-color: #1c1f26; padding: 15px; border-radius: 8px; border-left: 5px solid #0072F0; }
    [data-testid="stMetricLabel"] { font-size: 14px; color: #aab; }
    [data-testid="stMetricValue"] { font-size: 24px; font-weight: 700; color: #fff; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; letter-spacing: -0.5px; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #1c1f26; border-radius: 4px; color: #fff; }
    .stTabs [aria-selected="true"] { background-color: #0072F0; color: white; }
    div[data-testid="stExpander"] { background-color: #1c1f26; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# INITIALIZATION
# -----------------------------------------------------------------------------
if 'analysis_cache' not in st.session_state: st.session_state.analysis_cache = {}

# --------------------------------
# UTILITIES
# --------------------------------
def format_large_number(num):
    if not isinstance(num, (int, float)): return "N/A"
    if num >= 1e12: return f"${num/1e12:.2f}T"
    if num >= 1e9: return f"${num/1e9:.2f}B"
    if num >= 1e6: return f"${num/1e6:.2f}M"
    return f"${num:,.0f}"

def get_signal_color(signal):
    s = str(signal).upper()
    return "#00FF00" if s in ["BUY", "STRONG BUY"] else "#FF0000" if s in ["SELL", "STRONG SELL"] else "#FFA500"

# --------------------------------
# ADVANCED DATA FETCHER
# --------------------------------
@st.cache_data(ttl=600)
def fetch_comprehensive_data(ticker: str):
    """Fetches deep financial data, holders, and history."""
    try:
        t = yf.Ticker(ticker)
        
        # 1. Price History (Max for charts, 1y for analysis)
        hist = t.history(period="max")
        if hist.empty: return {"error": f"No history found for {ticker}"}
        
        # 2. Fundamentals (Income, Balance, Cash Flow)
        # Transpose to get dates as columns, metrics as rows
        inc = t.income_stmt
        bal = t.balance_sheet
        cash = t.cashflow
        
        # 3. Institutional & Analyst Data
        inst = t.institutional_holders
        rec = t.recommendations
        if rec is not None and not rec.empty:
            # Clean up recommendations (get latest)
            rec = rec.tail(10)
        
        # 4. News
        news = t.news

        # 5. Core Info
        info = t.info

        return {
            "history": hist,
            "info": info,
            "financials": {"income": inc, "balance": bal, "cash_flow": cash},
            "holders": inst,
            "recommendations": rec,
            "news": news
        }
    except Exception as e:
        return {"error": str(e)}

# --------------------------------
# LLM CLIENT (GEMINI)
# --------------------------------
class ModelClient:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        try:
            # Using Gemini 2.0 Flash for speed and large context window
            response = self.client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"LLM Error: {e}"

# --------------------------------
# SPECIALIZED AGENTS
# --------------------------------

class FundamentalExpert:
    def __init__(self, client): self.client = client
    
    def run(self, ticker, data):
        fin = data["financials"]
        info = data["info"]
        
        # Prepare context (Taking latest 2 years for comparison)
        try:
            inc_txt = fin["income"].iloc[:, :2].to_string() if not fin["income"].empty else "N/A"
            bal_txt = fin["balance"].iloc[:, :2].to_string() if not fin["balance"].empty else "N/A"
            cash_txt = fin["cash_flow"].iloc[:, :2].to_string() if not fin["cash_flow"].empty else "N/A"
        except:
            inc_txt = bal_txt = cash_txt = "Insufficient Data"

        prompt = f"""
        You are a Warren Buffett-style Value Investor. Analyze {ticker}.
        
        Financial Data (Last 2 periods):
        INCOME STATEMENT:
        {inc_txt}
        BALANCE SHEET:
        {bal_txt}
        CASH FLOW:
        {cash_txt}
        
        Metrics:
        Trailing P/E: {info.get('trailingPE', 'N/A')}
        Forward P/E: {info.get('forwardPE', 'N/A')}
        Price/Book: {info.get('priceToBook', 'N/A')}
        Return on Equity: {info.get('returnOnEquity', 'N/A')}
        
        Task:
        1. Analyze the "Moat" (Competitive Advantage).
        2. Assess Financial Health (Debt levels, Solvency).
        3. Check Capital Allocation (ROIC, Cash Flow vs Net Income).
        4. Detect Red Flags (e.g., Earnings growing but Cash Flow shrinking).
        
        Output:
        SIGNAL: [BUY/SELL/HOLD]
        THESIS: [Bullet points of analysis]
        """
        return self.client.generate(prompt)

class TechnicalExpert:
    def run(self, ticker, data):
        hist = data["history"]
        if hist.empty: return {"signal": "HOLD", "summary": "No Data"}
        
        # Calculate Indicators Manually
        df = hist.copy()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        latest = df.iloc[-1]
        
        # Trend Logic
        trend = "BULLISH" if latest['Close'] > latest['SMA_200'] else "BEARISH"
        rsi_status = "OVERBOUGHT" if latest['RSI'] > 70 else "OVERSOLD" if latest['RSI'] < 30 else "NEUTRAL"
        
        return {
            "signal": "BUY" if trend == "BULLISH" and rsi_status != "OVERBOUGHT" else "SELL" if trend == "BEARISH" else "HOLD",
            "trend": trend,
            "rsi": latest['RSI'],
            "sma_50": latest['SMA_50'],
            "sma_200": latest['SMA_200'],
            "summary": f"Trend is {trend}. RSI is {rsi_status} ({latest['RSI']:.1f}). Price is {'Above' if latest['Close'] > latest['SMA_50'] else 'Below'} 50-SMA."
        }

class InstitutionalAgent:
    def __init__(self, client): self.client = client
    
    def run(self, ticker, data):
        holders = data.get("holders")
        recs = data.get("recommendations")
        
        holders_txt = holders.to_string() if holders is not None else "No Data"
        recs_txt = recs.tail(5).to_string() if recs is not None else "No Data"
        
        prompt = f"""
        You are an Institutional Flow Analyst. Analyze ownership for {ticker}.
        
        Top Institutional Holders:
        {holders_txt}
        
        Analyst Recommendations (Recent):
        {recs_txt}
        
        Task:
        1. Is "Smart Money" (Institutions) heavily invested?
        2. What is the consensus among analysts?
        3. Are there divergent views?
        
        Output:
        SIGNAL: [BUY/SELL/HOLD]
        SUMMARY: [Concise analysis of flow and sentiment]
        """
        return self.client.generate(prompt)

class MasterPortfolioManager:
    def __init__(self, client): self.client = client
    
    def run(self, ticker, fund_analysis, tech_analysis, inst_analysis, news_data):
        # Format News Safely
        news_summary = ""
        # --- FIXED: Added .get() to prevent KeyError ---
        for n in news_data[:5]:
            title = n.get('title', 'Unknown Title')
            publisher = n.get('publisher', 'Unknown Source')
            news_summary += f"- {title} (Source: {publisher})\n"
            
        prompt = f"""
        You are the Chief Investment Officer. Synthesize these reports for {ticker} into a final investment decision.
        
        1. FUNDAMENTAL REPORT:
        {fund_analysis}
        
        2. TECHNICAL DATA:
        Signal: {tech_analysis.get('signal', 'HOLD')}
        Data: {tech_analysis.get('summary', 'No Data')}
        
        3. INSTITUTIONAL/ANALYST REPORT:
        {inst_analysis}
        
        4. RECENT NEWS HEADLINES:
        {news_summary}
        
        TASK:
        Provide a final investment memo.
        - Weight Fundamentals 50%, Technicals 30%, Sentiment 20%.
        - Be decisive.
        
        OUTPUT FORMAT:
        DECISION: [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]
        CONFIDENCE: [0-100]%
        ONE_LINER: [The single most important reason]
        KEY_RISKS: [Top 2 risks]
        EXECUTIVE_SUMMARY: [A paragraph explaining the thesis]
        """
        return self.client.generate(prompt)

# --------------------------------
# CHARTING ENGINE
# --------------------------------
def create_chart(ticker, hist, chart_type="Candlestick"):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'{ticker} Price', 'Volume'), 
                        row_width=[0.2, 0.7])

    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(x=hist.index,
                                     open=hist['Open'], high=hist['High'],
                                     low=hist['Low'], close=hist['Close'], name='OHLC'), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close', line=dict(color='#0072F0')), row=1, col=1)

    # Add Moving Averages
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(50).mean(), line=dict(color='orange', width=1), name='50 MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(200).mean(), line=dict(color='purple', width=1), name='200 MA'), row=1, col=1)

    # Volume
    colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for index, row in hist.iterrows()]
    fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color=colors, showlegend=False), row=2, col=1)

    fig.update_layout(
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig

# --------------------------------
# MAIN APP
# --------------------------------
def main():
    api_key = st.secrets.get("API_KEY")
    if not api_key:
        st.error("API_KEY not found in secrets.toml")
        st.stop()
        
    client = ModelClient(api_key)

    # Sidebar inputs
    with st.sidebar:
        st.title("🛡️ TITAN Terminal")
        ticker = st.text_input("Ticker Symbol", "NVDA").upper()
        chart_type = st.radio("Chart Type", ["Candlestick", "Line"])
        if st.button("Initialize Analysis", type="primary"):
            st.session_state.trigger_analysis = True

    if not st.session_state.get('trigger_analysis', False):
        st.info("Enter a ticker in the sidebar and click 'Initialize Analysis' to begin.")
        st.stop()

    # --- EXECUTION ---
    with st.spinner(f"Accessing Global Markets Data for {ticker}..."):
        data = fetch_comprehensive_data(ticker)
    
    if "error" in data:
        st.error(data["error"])
        st.stop()

    # Run Agents (Parallel simulation)
    with st.spinner("Council of Experts deliberating..."):
        info = data["info"]
        
        # 1. Technical Analysis (Fast)
        tech_report = TechnicalExpert().run(ticker, data)
        
        # 2. Fundamental Analysis (LLM)
        fund_report = FundamentalExpert(client).run(ticker, data)
        
        # 3. Institutional Analysis (LLM)
        inst_report = InstitutionalAgent(client).run(ticker, data)
        
        # 4. Master Thesis
        master_thesis = MasterPortfolioManager(client).run(ticker, fund_report, tech_report, inst_report, data["news"])

    # --- DASHBOARD LAYOUT ---
    
    # 1. Header Metrics
    st.title(f"{ticker} • {info.get('longName', ticker)}")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    
    current_price = info.get('currentPrice', data["history"]['Close'].iloc[-1])
    # Safe previous close check
    if len(data["history"]) > 1:
        prev_close = info.get('previousClose', data["history"]['Close'].iloc[-2])
    else:
        prev_close = current_price
        
    delta = current_price - prev_close
    delta_pct = (delta / prev_close) * 100 if prev_close != 0 else 0
    
    m1.metric("Price", f"${current_price:.2f}", f"{delta:.2f} ({delta_pct:.2f}%)")
    m2.metric("Market Cap", format_large_number(info.get('marketCap', 0)))
    m3.metric("Beta", f"{info.get('beta', 'N/A')}")
    m4.metric("PE Ratio", f"{info.get('trailingPE', 'N/A')}")
    m5.metric("Target Mean", f"${info.get('targetMeanPrice', 'N/A')}")

    # 2. Executive Summary (The Master Thesis)
    with st.container(border=True):
        st.subheader("🏛️ Investment Committee Verdict")
        
        # Extract Decision from LLM Text
        decision = "HOLD"
        for d in ["STRONG BUY", "BUY", "SELL", "STRONG SELL", "HOLD"]:
            if d in master_thesis:
                decision = d
                break
                
        color = get_signal_color(decision)
        st.markdown(f"<h2 style='color: {color}; text-align: center; border-bottom: 2px solid {color}; padding-bottom: 10px;'>{decision}</h2>", unsafe_allow_html=True)
        st.markdown(master_thesis)

    # 3. Detailed Tabs
    tab_chart, tab_fund, tab_own, tab_news = st.tabs(["📈 Price Action", "📊 Deep Financials", "🏢 Ownership & Analysts", "📰 News Wire"])

    with tab_chart:
        st.plotly_chart(create_chart(ticker, data["history"], chart_type), use_container_width=True)
        st.info(f"Technical Indicator Summary: {tech_report.get('summary', 'No summary')}")

    with tab_fund:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Financial Statements")
            stmt_type = st.radio("Select Statement", ["Income Statement", "Balance Sheet", "Cash Flow"], horizontal=True)
            
            if stmt_type == "Income Statement":
                st.dataframe(data["financials"]["income"], height=400, use_container_width=True)
            elif stmt_type == "Balance Sheet":
                st.dataframe(data["financials"]["balance"], height=400, use_container_width=True)
            else:
                st.dataframe(data["financials"]["cash_flow"], height=400, use_container_width=True)
        
        with col2:
            st.subheader("Fundamental Analysis")
            st.markdown(fund_report)

    with tab_own:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Institutional Holders")
            st.dataframe(data["holders"], use_container_width=True)
        with col2:
            st.subheader("Analyst Recommendations")
            st.dataframe(data["recommendations"], use_container_width=True)
            st.markdown("---")
            st.markdown(inst_report)

    with tab_news:
        st.subheader("Latest Market News")
        # --- FIXED: Added safety checks for keys ---
        for news_item in data["news"]:
            title = news_item.get('title', 'No Title')
            link = news_item.get('link', '#')
            publisher = news_item.get('publisher', 'Unknown')
            
            # Safe timestamp conversion
            ts = news_item.get('providerPublishTime', 0)
            date_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M') if ts else "N/A"
            
            with st.container(border=True):
                col_img, col_txt = st.columns([1, 4])
                with col_txt:
                    st.markdown(f"### [{title}]({link})")
                    st.caption(f"Publisher: {publisher} • {date_str}")
                    if 'relatedTickers' in news_item:
                        st.code(f"Related: {', '.join(news_item['relatedTickers'])}")

if __name__ == "__main__":
    main()

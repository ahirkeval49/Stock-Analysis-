import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google import genai
from newsapi import NewsApiClient
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page Config ---
st.set_page_config(page_title="Titan: Ultra Dashboard", layout="wide", initial_sidebar_state="collapsed")

# --- CSS for "Light Mode" & Professional UI ---
st.markdown("""
<style>
    /* Global Settings */
    .stApp { background-color: #F8F9FA; color: #212529; font-family: 'Inter', sans-serif; }
    
    /* Instruction Text */
    .instruction-text { font-size: 1.1rem; font-weight: 500; color: #495057; margin-bottom: 10px; text-align: center; }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        border: 1px solid #E9ECEF;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        border-radius: 8px;
        color: #495057;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        padding: 10px 20px;
        font-weight: 500;
        border: 1px solid #E9ECEF;
    }
    .stTabs [aria-selected="true"] { background-color: #2563EB; color: white; }
    
    /* AI Report Box */
    .ai-report {
        background-color: #F0F9FF;
        border-left: 5px solid #0EA5E9;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------
# UTILITIES
# --------------------------------
if 'analysis_cache' not in st.session_state: st.session_state.analysis_cache = {}
if 'timeframe' not in st.session_state: st.session_state.timeframe = "1y"

def format_large_number(num):
    if not isinstance(num, (int, float)): return "N/A"
    if num >= 1e12: return f"${num/1e12:.2f}T"
    if num >= 1e9: return f"${num/1e9:.2f}B"
    if num >= 1e6: return f"${num/1e6:.2f}M"
    return f"${num:,.0f}"

def get_signal_color(text):
    if "BUY" in text.upper(): return "#10B981" # Green
    if "SELL" in text.upper(): return "#EF4444" # Red
    return "#F59E0B" # Orange

# --------------------------------
# DATA FETCHING (MAXIMUM DEPTH)
# --------------------------------
@st.cache_data(ttl=600)
def fetch_data(ticker, period="1y"):
    try:
        t = yf.Ticker(ticker)
        
        # 1. Core Data
        hist = t.history(period=period)
        info = t.info
        
        # 2. Financial Statements (Annual & Quarterly)
        inc = t.income_stmt
        bal = t.balance_sheet
        cash = t.cashflow
        q_inc = t.quarterly_income_stmt
        
        # 3. Ownership & Insiders
        inst_holders = t.institutional_holders
        major_holders = t.major_holders
        insider_tx = t.insider_transactions
        
        # 4. Analyst Data
        recs = t.recommendations
        upgrades = t.upgrades_downgrades
        
        # 5. NewsAPI
        news = []
        newsapi_key = st.secrets.get("NEWSAPI_KEY")
        if newsapi_key:
            try:
                napi = NewsApiClient(api_key=newsapi_key)
                query = info.get('longName', ticker)
                # News from last 14 days
                start = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
                resp = napi.get_everything(q=query, from_param=start, language='en', sort_by='relevancy', page_size=7)
                if resp['status'] == 'ok': news = resp['articles']
            except: pass
        
        if not news: # Fallback
            news = [{"title": n['title'], "source": {"name": n['publisher']}, "url": n['link']} for n in t.news]

        return {
            "history": hist, "info": info, "news": news,
            "financials": {"inc": inc, "bal": bal, "cash": cash, "q_inc": q_inc},
            "ownership": {"inst": inst_holders, "major": major_holders, "insider": insider_tx},
            "analysts": {"recs": recs, "upgrades": upgrades}
        }
    except Exception as e:
        return {"error": str(e)}

# --------------------------------
# AI AGENTS
# --------------------------------
class ModelClient:
    def __init__(self, api_key): self.client = genai.Client(api_key=api_key)
    def generate(self, prompt):
        try: return self.client.models.generate_content(model="gemini-2.0-flash", contents=prompt).text
        except Exception as e: return f"AI Error: {e}"

# --- Agent 1: The Technical Analyst ---
def run_technical_agent(client, ticker, hist):
    latest = hist.iloc[-1]
    
    # Simple Technicals
    sma50 = hist['Close'].rolling(50).mean().iloc[-1]
    sma200 = hist['Close'].rolling(200).mean().iloc[-1]
    
    # RSI Calculation
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    
    prompt = f"""
    Act as a Technical Analyst. Analyze {ticker}.
    Price: {latest['Close']:.2f}
    SMA50: {sma50:.2f} | SMA200: {sma200:.2f}
    RSI: {rsi:.2f}
    Volume: {latest['Volume']}
    
    Output strictly:
    SIGNAL: [BUY/SELL/HOLD]
    REASON: [Max 15 words summary]
    """
    return client.generate(prompt)

# --- Agent 2: The Financial Statement Expert ---
def run_financial_agent(client, ticker, fin):
    inc = fin['inc']
    bal = fin['bal']
    cash = fin['cash']
    
    # Prepare Data Context (Last 2 Years)
    try:
        inc_txt = inc.iloc[:, :2].to_string() if not inc.empty else "N/A"
        bal_txt = bal.iloc[:, :2].to_string() if not bal.empty else "N/A"
        cash_txt = cash.iloc[:, :2].to_string() if not cash.empty else "N/A"
    except: inc_txt = bal_txt = cash_txt = "N/A"
    
    prompt = f"""
    Act as a Forensic Accountant. Analyze {ticker}.
    
    INCOME STATEMENT: {inc_txt}
    BALANCE SHEET: {bal_txt}
    CASH FLOW: {cash_txt}
    
    Task:
    1. Check Margins (Gross, Operating).
    2. Check Debt levels.
    3. Check Cash Flow quality.
    
    Output strictly:
    SIGNAL: [BUY/SELL/HOLD]
    REASON: [Max 15 words summary]
    """
    return client.generate(prompt)

# --- Agent 3: The Insider & Institutional Spy ---
def run_ownership_agent(client, ticker, own):
    insider = own['insider']
    inst = own['inst']
    
    insider_txt = insider.head(5).to_string() if insider is not None and not insider.empty else "No Insider Data"
    inst_txt = inst.head(5).to_string() if inst is not None and not inst.empty else "No Institutional Data"
    
    prompt = f"""
    Act as an Insider Trading Analyst. Analyze {ticker}.
    
    RECENT INSIDER TRANSACTIONS:
    {insider_txt}
    
    INSTITUTIONAL HOLDERS:
    {inst_txt}
    
    Task: Are insiders buying or selling? Is smart money entering?
    
    Output strictly:
    SIGNAL: [BUY/SELL/HOLD]
    REASON: [Max 15 words summary]
    """
    return client.generate(prompt)

# --- Agent 4: The Master Strategist ---
def run_master_agent(client, ticker, info, tech, fund, own, news):
    news_sum = "\n".join([f"- {n['title']}" for n in news[:3]])
    
    prompt = f"""
    You are the Chief Investment Officer. Consolidate these reports for {ticker}.
    
    1. TECHNICAL REPORT: {tech}
    2. FINANCIAL REPORT: {fund}
    3. OWNERSHIP REPORT: {own}
    4. NEWS: {news_sum}
    5. VALUATION: PE {info.get('trailingPE')}, PEG {info.get('pegRatio')}
    
    Task: Write a final investment memo.
    
    Output Format:
    **VERDICT:** [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]
    **CONFIDENCE:** [0-100]%
    **THESIS:** [3-4 sentence explanation]
    **RISK:** [1 key risk]
    """
    return client.generate(prompt)

# --------------------------------
# MAIN APP
# --------------------------------
def main():
    api_key = st.secrets.get("API_KEY")
    if not api_key: st.error("Missing API_KEY."); st.stop()
    client = ModelClient(api_key)

    # --- Header ---
    st.markdown('<div class="instruction-text">Type the name of the company or ticker symbol</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        col_in, col_btn = st.columns([3, 1])
        ticker_input = col_in.text_input("Ticker", "PLTR", label_visibility="collapsed").upper()
        if col_btn.button("Analyze", type="primary", use_container_width=True):
            st.session_state.run_analysis = True

    # --- Analysis Engine ---
    if st.session_state.get('run_analysis'):
        if 'last_ticker' in st.session_state and st.session_state.last_ticker != ticker_input:
            st.session_state.ai_results = None # Reset cache
            
        with st.spinner(f"Titan Agents Deployed on {ticker_input}..."):
            data = fetch_data(ticker_input, period=st.session_state.timeframe)
            if "error" in data: st.error(data["error"]); st.stop()
            st.session_state.data = data
            
            # Run Agents if not cached
            if not st.session_state.get('ai_results'):
                with st.status("Council of Experts Working...", expanded=True) as status:
                    st.write("📈 Technical Agent analyzing charts...")
                    tech_res = run_technical_agent(client, ticker_input, data['history'])
                    
                    st.write("💰 Financial Agent auditing books...")
                    fund_res = run_financial_agent(client, ticker_input, data['financials'])
                    
                    st.write("🕵️ Insider Agent tracking trades...")
                    own_res = run_ownership_agent(client, ticker_input, data['ownership'])
                    
                    st.write("🧠 Master Strategist synthesizing...")
                    master_res = run_master_agent(client, ticker_input, data['info'], tech_res, fund_res, own_res, data['news'])
                    
                    st.session_state.ai_results = {
                        "tech": tech_res, "fund": fund_res, "own": own_res, "master": master_res
                    }
                    status.update(label="Analysis Complete", state="complete", expanded=False)
                st.session_state.last_ticker = ticker_input

    # --- Dashboard ---
    if 'data' in st.session_state:
        data = st.session_state.data
        info = data['info']
        hist = data['history']
        res = st.session_state.ai_results

        # 1. Company Header
        st.markdown(f"### {info.get('longName', ticker_input)} ({ticker_input})")
        
        # 2. Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        current = hist['Close'].iloc[-1]
        
        m1.metric("Price", f"${current:.2f}", f"{(current - hist['Close'].iloc[-2]):.2f}", help="Live Price")
        m2.metric("Market Cap", format_large_number(info.get('marketCap')), help="Total Value")
        m3.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}", help="Trailing Price-to-Earnings")
        m4.metric("PEG Ratio", f"{info.get('pegRatio', 'N/A')}", help="Price/Earnings-to-Growth (Lower is better)")

        # 3. Master AI Report
        st.markdown('<div class="ai-report">', unsafe_allow_html=True)
        st.subheader("🤖 Chief Investment Officer Verdict")
        st.markdown(res['master'])
        st.markdown('</div>', unsafe_allow_html=True)

        # 4. Charting
        st.markdown("---")
        with st.container(border=True):
            # Plotly Chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.7])
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color='rgba(37, 99, 235, 0.5)', name='Volume'), row=2, col=1)
            fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_white", margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # Timeframe Toggle
            cols = st.columns(6)
            periods = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
            for i, p in enumerate(periods):
                if cols[i].button(p, type="primary" if st.session_state.timeframe == p else "secondary", use_container_width=True):
                    st.session_state.timeframe = p
                    st.session_state.run_analysis = True
                    st.rerun()

        # 5. Deep Dive Tabs
        tab_fund, tab_tech, tab_own, tab_news = st.tabs(["📊 Financials", "📈 Technicals", "🕵️ Insiders & Inst.", "📰 News"])

        with tab_fund:
            st.subheader("Financial Statement Analysis")
            st.info(f"**AI Agent says:** {res['fund']}")
            
            f_type = st.radio("View Statement", ["Income", "Balance Sheet", "Cash Flow"], horizontal=True)
            if f_type == "Income": st.dataframe(data['financials']['inc'], use_container_width=True)
            elif f_type == "Balance Sheet": st.dataframe(data['financials']['bal'], use_container_width=True)
            else: st.dataframe(data['financials']['cash'], use_container_width=True)

        with tab_tech:
            st.subheader("Technical Analysis")
            st.info(f"**AI Agent says:** {res['tech']}")
            
            # Show raw technical data
            t_col1, t_col2 = st.columns(2)
            t_col1.metric("50-Day SMA", f"${hist['Close'].rolling(50).mean().iloc[-1]:.2f}")
            t_col2.metric("200-Day SMA", f"${hist['Close'].rolling(200).mean().iloc[-1]:.2f}")

        with tab_own:
            st.subheader("Ownership Intelligence")
            st.info(f"**AI Agent says:** {res['own']}")
            
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Recent Insider Transactions**")
                if data['ownership']['insider'] is not None:
                    st.dataframe(data['ownership']['insider'], use_container_width=True)
                else: st.warning("No Insider Data Found")
            with c2:
                st.write("**Top Institutional Holders**")
                if data['ownership']['inst'] is not None:
                    st.dataframe(data['ownership']['inst'], use_container_width=True)
                else: st.warning("No Institutional Data Found")

        with tab_news:
            st.subheader("Global News Wire")
            for n in data['news']:
                with st.container(border=True):
                    src = n.get('source', {}).get('name', n.get('publisher', 'Unknown'))
                    st.markdown(f"**{src}** • [{n['title']}]({n.get('url', n.get('link'))})")

if __name__ == "__main__":
    main()
    

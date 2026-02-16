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
    .news-summary-box {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
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
        q_bal = t.quarterly_balance_sheet
        q_cash = t.quarterly_cashflow
        
        # 3. Ownership & Insiders
        inst_holders = t.institutional_holders
        mutual_holders = t.mutualfund_holders
        insider_tx = t.insider_transactions
        major_holders = t.major_holders
        
        # 4. Analyst & Calendar
        recs = t.recommendations
        calendar = t.calendar
        
        # 5. NewsAPI (Fetch 15 to ensure we have 10 good ones)
        news = []
        newsapi_key = st.secrets.get("NEWSAPI_KEY")
        if newsapi_key:
            try:
                napi = NewsApiClient(api_key=newsapi_key)
                query = info.get('longName', ticker)
                # News from last 14 days
                start = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
                resp = napi.get_everything(q=query, from_param=start, language='en', sort_by='relevancy', page_size=15)
                if resp['status'] == 'ok': news = resp['articles']
            except: pass
        
        if not news: # Fallback to yfinance news if NewsAPI fails
            news = [{"title": n.get('title'), "source": {"name": n.get('publisher')}, "url": n.get('link'), "description": "No description"} for n in t.news]

        return {
            "history": hist, "info": info, "news": news,
            "financials": {"inc": inc, "bal": bal, "cash": cash, "q_inc": q_inc, "q_bal": q_bal, "q_cash": q_cash},
            "ownership": {"inst": inst_holders, "mutual": mutual_holders, "insider": insider_tx, "major": major_holders},
            "analysts": {"recs": recs, "calendar": calendar}
        }
    except Exception as e:
        return {"error": str(e)}

# --------------------------------
# TECHNICAL CALCULATOR (Helper)
# --------------------------------
def calculate_technicals(df):
    if df.empty: return df
    
    # Moving Averages
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
    
    return df

# --------------------------------
# AI AGENTS
# --------------------------------
class ModelClient:
    def __init__(self, api_key): self.client = genai.Client(api_key=api_key)
    def generate(self, prompt):
        try: return self.client.models.generate_content(model="gemini-3-pro-preview"", contents=prompt).text
        except Exception as e: return f"AI Error: {e}"

# --- Agent 1: The Technical Analyst ---
def run_technical_agent(client, ticker, hist):
    df = calculate_technicals(hist.copy())
    latest = df.iloc[-1]
    
    prompt = f"""
    Act as a Technical Analyst. Analyze {ticker} using these indicators:
    Price: {latest['Close']:.2f}
    SMA50: {latest['SMA_50']:.2f} | SMA200: {latest['SMA_200']:.2f}
    RSI (14): {latest['RSI']:.2f}
    MACD: {latest['MACD']:.2f} (Signal: {latest['Signal_Line']:.2f})
    Bollinger Bands: Upper {latest['BB_Upper']:.2f} | Lower {latest['BB_Lower']:.2f}
    
    Determine trend strength and support/resistance levels.
    
    Output strictly:
    SIGNAL: [BUY/SELL/HOLD]
    ANALYSIS: [Max 30 words summary]
    """
    return client.generate(prompt)

# --- Agent 2: The Financial Statement Expert ---
def run_financial_agent(client, ticker, fin):
    # Prepare Data Context (Last 2 Years + Recent Quarter)
    try:
        inc_txt = fin['inc'].iloc[:, :2].to_string() if not fin['inc'].empty else "N/A"
        q_inc_txt = fin['q_inc'].iloc[:, :2].to_string() if not fin['q_inc'].empty else "N/A"
        bal_txt = fin['bal'].iloc[:, :2].to_string() if not fin['bal'].empty else "N/A"
        cash_txt = fin['cash'].iloc[:, :2].to_string() if not fin['cash'].empty else "N/A"
    except: inc_txt = q_inc_txt = bal_txt = cash_txt = "N/A"
    
    prompt = f"""
    Act as a CFA Charterholder. Analyze {ticker} financials.
    
    ANNUAL INCOME STATEMENT: {inc_txt}
    QUARTERLY INCOME (Trend): {q_inc_txt}
    BALANCE SHEET: {bal_txt}
    CASH FLOW: {cash_txt}
    
    Task:
    1. Assess Revenue Growth & Margin Trends.
    2. Evaluate Solvency (Debt/Equity) and Liquidity.
    3. Analyze Quality of Earnings (Cash Flow vs Net Income).
    
    Output strictly:
    SIGNAL: [BUY/SELL/HOLD]
    ANALYSIS: [Max 40 words summary]
    """
    return client.generate(prompt)

# --- Agent 3: The Insider & Institutional Spy ---
def run_ownership_agent(client, ticker, own):
    insider = own['insider']
    inst = own['inst']
    mutual = own['mutual']
    
    insider_txt = insider.head(5).to_string() if insider is not None and not insider.empty else "No Insider Data"
    inst_txt = inst.head(5).to_string() if inst is not None and not inst.empty else "No Institutional Data"
    mutual_txt = mutual.head(5).to_string() if mutual is not None and not mutual.empty else "No Mutual Fund Data"
    
    prompt = f"""
    Act as an Institutional Flow Analyst. Analyze {ticker}.
    
    RECENT INSIDER TRADES:
    {insider_txt}
    
    TOP INSTITUTIONS:
    {inst_txt}
    
    TOP MUTUAL FUNDS:
    {mutual_txt}
    
    Task: Detect accumulation or distribution. Are insiders dumping stock? Are big funds buying?
    
    Output strictly:
    SIGNAL: [BUY/SELL/HOLD]
    ANALYSIS: [Max 30 words summary]
    """
    return client.generate(prompt)

# --- Agent 4: The News & Sentiment Analyst ---
def run_news_agent(client, ticker, news_data):
    # Process top 10 articles
    articles_text = ""
    for i, n in enumerate(news_data[:10]):
        title = n.get('title', 'No Title')
        desc = n.get('description', 'No Description')
        source = n.get('source', {}).get('name', 'Unknown')
        articles_text += f"{i+1}. {title} ({source}): {desc}\n"
    
    prompt = f"""
    Act as a Sentiment Analyst. Read these 10 recent articles about {ticker}:
    
    {articles_text}
    
    Task:
    1. Summarize the dominant narrative (Bullish/Bearish/Neutral).
    2. Identify specific catalysts (Earnings, Lawsuits, Product Launches).
    
    Output strictly:
    SENTIMENT_SCORE: [0-10, where 10 is Euphoric]
    SUMMARY: [3-4 bullet points summarizing the news]
    """
    return client.generate(prompt)

# --- Agent 5: The Master Strategist ---
def run_master_agent(client, ticker, info, tech, fund, own, news_analysis):
    prompt = f"""
    You are the Chief Investment Officer (CIO) of Titan Capital. 
    Synthesize these expert reports for {ticker} into a final investment memo.
    
    1. TECHNICAL REPORT: {tech}
    2. FINANCIAL REPORT: {fund}
    3. OWNERSHIP REPORT: {own}
    4. NEWS SENTIMENT: {news_analysis}
    5. VALUATION: PE {info.get('trailingPE')}, Market Cap {info.get('marketCap')}
    
    Task: Write a final verdict.
    
    Output Format:
    **VERDICT:** [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]
    **CONFIDENCE:** [0-100]%
    **THESIS:** [3-4 sentence detailed explanation]
    **KEY RISKS:** [1-2 sentences]
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
                    st.write("📈 Technical Agent calculating indicators (RSI, MACD)...")
                    tech_res = run_technical_agent(client, ticker_input, data['history'])
                    
                    st.write("💰 Financial Agent auditing annual & quarterly books...")
                    fund_res = run_financial_agent(client, ticker_input, data['financials'])
                    
                    st.write("🕵️ Insider Agent tracking institutional flows...")
                    own_res = run_ownership_agent(client, ticker_input, data['ownership'])
                    
                    st.write("📰 News Agent reading top 10 articles...")
                    news_res = run_news_agent(client, ticker_input, data['news'])
                    
                    st.write("🧠 Master Strategist synthesizing final verdict...")
                    master_res = run_master_agent(client, ticker_input, data['info'], tech_res, fund_res, own_res, news_res)
                    
                    st.session_state.ai_results = {
                        "tech": tech_res, "fund": fund_res, "own": own_res, "news": news_res, "master": master_res
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
        tab_fund, tab_tech, tab_own, tab_news = st.tabs(["📊 Financials", "📈 Technicals", "🕵️ Insiders & Inst.", "📰 News Analyst"])

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
            st.subheader("News & Sentiment Analysis")
            
            # Display AI Summary
            st.markdown('<div class="news-summary-box">', unsafe_allow_html=True)
            st.markdown(f"**AI News Analyst Summary:**\n\n{res['news']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("### Source Articles")
            for n in data['news'][:10]: # Limit to top 10 for display
                with st.container(border=True):
                    src = n.get('source', {}).get('name', n.get('publisher', 'Unknown'))
                    title = n.get('title', 'No Title')
                    url = n.get('url', n.get('link', '#'))
                    st.markdown(f"**{src}** • [{title}]({url})")

if __name__ == "__main__":
    main()

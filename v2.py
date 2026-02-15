import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from google import genai
from dotenv import load_dotenv
import re
import json

# --- Page Config ---
st.set_page_config(page_title="AI Hedge Fund (Gemini 2.0)", layout="wide", initial_sidebar_state="expanded")

# --- CSS for Aesthetics ---
st.markdown("""
<style>
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; border: 1px solid #e0e0e0; }
    [data-testid="stMetricLabel"] { font-size: 14px; color: #555; }
    [data-testid="stMetricValue"] { font-size: 20px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if 'live_output' not in st.session_state: st.session_state.live_output = {}

# --------------------------------
# Utility Functions
# --------------------------------
def get_signal_color(signal):
    s = str(signal).upper()
    return "green" if s in ["BUY", "STRONG_BUY"] else "red" if s in ["SELL", "STRONG_SELL"] else "orange"

# --------------------------------
# Data Fetchers
# --------------------------------
@st.cache_data(ttl=300)
def fetch_market_data(ticker: str):
    """Fetches Price, Info, and Financials in one go."""
    try:
        t = yf.Ticker(ticker)
        # Fetch minimal history for speed
        hist = t.history(period="1y")
        info = t.info
        
        # Safe extraction of financials
        try:
            inc = t.income_stmt.iloc[:, 0].to_dict() if not t.income_stmt.empty else {}
            bal = t.balance_sheet.iloc[:, 0].to_dict() if not t.balance_sheet.empty else {}
        except:
            inc, bal = {}, {}

        # Safe extraction of news
        news = []
        try:
            for n in t.news[:5]:
                news.append({
                    "title": n.get('title'),
                    "link": n.get('link'),
                    "publisher": n.get('publisher'),
                    "time": datetime.fromtimestamp(n.get('providerPublishTime', 0)).strftime('%Y-%m-%d')
                })
        except: pass

        return {
            "history": hist,
            "info": info,
            "financials": {"income": inc, "balance": bal},
            "news": news
        }
    except Exception as e:
        return {"error": str(e)}

# --------------------------------
# Gemini 2.0 Client
# --------------------------------
class ModelClient:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Error: {e}"

# --------------------------------
# Agents
# --------------------------------
class FinancialsAgent:
    def __init__(self, client): self.client = client
    def run(self, ticker, data):
        fin = data.get("financials", {})
        bal = fin.get("balance", {})
        inc = fin.get("income", {})
        
        # Calculate Ratios
        assets = bal.get("Total Assets", 1)
        liab = bal.get("Total Liabilities Net Minority Interest", 0)
        equity = bal.get("Stockholders Equity", 1)
        net_inc = inc.get("Net Income", 0)
        rev = inc.get("Total Revenue", 1)
        
        de_ratio = liab / equity if equity else 0
        margin = net_inc / rev if rev else 0
        
        prompt = f"""
        Analyze {ticker}:
        Debt/Equity: {de_ratio:.2f}
        Net Margin: {margin:.2%}
        Revenue: {rev}
        
        Is this healthy? Output strictly in this format:
        SIGNAL: [BUY/SELL/HOLD]
        SUMMARY: [Max 20 words explanation]
        """
        resp = self.client.generate(prompt)
        
        sig = "HOLD"
        if "BUY" in resp: sig = "BUY"
        elif "SELL" in resp: sig = "SELL"
        
        summary = resp.split("SUMMARY:")[-1].strip() if "SUMMARY:" in resp else resp
        
        return {
            "financial_signal": sig,
            "financial_summary": summary,
            "de_ratio": de_ratio,
            "net_margin": margin
        }

class TechnicalAgent:
    def run(self, ticker, data):
        hist = data.get("history", pd.DataFrame())
        if hist.empty: return {"price_signal": "HOLD", "sma50": 0, "sma200": 0}
        
        # Calculate Indicators
        hist["SMA50"] = hist["Close"].rolling(50).mean()
        hist["SMA200"] = hist["Close"].rolling(200).mean()
        
        latest = hist.iloc[-1]
        prev_year = hist.iloc[0]["Close"] if len(hist) > 0 else latest["Close"]
        
        # Logic
        sig = "HOLD"
        if latest["SMA50"] > latest["SMA200"]: sig = "BUY"
        elif latest["SMA50"] < latest["SMA200"]: sig = "SELL"
        
        momentum = (latest["Close"] / prev_year) - 1
        
        return {
            "price_signal": sig,
            "sma50": latest["SMA50"],
            "sma200": latest["SMA200"],
            "momentum_12m": momentum,
            "current_price": latest["Close"]
        }

class SentimentAgent:
    def __init__(self, client): self.client = client
    def run(self, ticker, data):
        news = data.get("news", [])
        if not news: return {"sentiment_score": 0, "sentiment_signal": "HOLD"}
        
        headlines = "\n".join([f"- {n['title']}" for n in news[:5]])
        prompt = f"""
        Analyze sentiment for {ticker} based on:
        {headlines}
        
        Output ONLY a score from -1.0 (Worst) to 1.0 (Best).
        Example: 0.45
        """
        resp = self.client.generate(prompt)
        
        try:
            match = re.search(r"[-+]?\d*\.\d+|\d+", resp)
            score = float(match.group()) if match else 0.0
            score = max(-1.0, min(1.0, score)) # Safety clamp
        except:
            score = 0.0
            
        sig = "BUY" if score > 0.2 else "SELL" if score < -0.2 else "HOLD"
        return {"sentiment_score": score, "sentiment_signal": sig}

# --------------------------------
# Main App Logic
# --------------------------------
def run_analysis(tickers, client):
    results = {}
    bar = st.progress(0, text="Initializing Agents...")
    
    for i, t in enumerate(tickers):
        bar.progress((i)/len(tickers), text=f"Scanning {t}...")
        
        raw_data = fetch_market_data(t)
        if "error" in raw_data: 
            st.error(f"Failed to fetch {t}: {raw_data['error']}")
            continue
            
        tech = TechnicalAgent().run(t, raw_data)
        fund = FinancialsAgent(client).run(t, raw_data)
        sent = SentimentAgent(client).run(t, raw_data)
        
        # Composite Score
        score = 0
        score += 1 if tech["price_signal"] == "BUY" else -1 if tech["price_signal"] == "SELL" else 0
        score += 1 if fund["financial_signal"] == "BUY" else -1 if fund["financial_signal"] == "SELL" else 0
        score += sent["sentiment_score"] * 2 
        
        decision = "BUY" if score > 0.5 else "SELL" if score < -0.5 else "HOLD"
        
        results[t] = {
            **tech, **fund, **sent, 
            "composite_score": score,
            "final_decision": decision,
            "info": raw_data["info"],
            "news": raw_data["news"]
        }
    
    bar.empty()
    return results

# --------------------------------
# UI Layout
# --------------------------------
api_key = st.secrets.get("API_KEY")
if not api_key: st.error("Missing API_KEY in secrets.toml"); st.stop()

client = ModelClient(api_key)

st.title("⚡ AI Hedge Fund (Gemini 2.0)")

# Input Section
with st.container(border=True):
    col1, col2 = st.columns([3, 1])
    tickers = col1.text_input("Tickers", "AAPL, NVDA, TSLA, AMD")
    
    # Button to Clear Cache and Reset (Fixes Stale Data Issues)
    if st.sidebar.button("⚠️ Clear Cache / Reset"):
        st.session_state.live_output = {}
        st.rerun()

    if col2.button("Run Analysis", type="primary", use_container_width=True):
        t_list = [x.strip().upper() for x in tickers.split(",")]
        st.session_state.live_output = run_analysis(t_list, client)

# Results Section
if st.session_state.live_output:
    st.subheader("Market Reconnaissance")
    
    # Overview Table
    data = []
    for t, res in st.session_state.live_output.items():
        data.append({
            "Ticker": t,
            "Price": f"${res.get('current_price', 0):.2f}",
            "Decision": res.get('final_decision', 'HOLD'),
            "Score": f"{res.get('composite_score', 0):.2f}",
            "Sentiment": f"{res.get('sentiment_score', 0):.2f}"
        })
    
    df = pd.DataFrame(data)
    st.dataframe(
        df.style.map(lambda x: f'color: {get_signal_color(x)}', subset=['Decision']), 
        use_container_width=True
    )
    
    # Detailed Tabs
    selected = st.selectbox("Deep Dive", list(st.session_state.live_output.keys()))
    res = st.session_state.live_output[selected]
    
    # --- ROBUST DATA ACCESS (Fixes KeyError) ---
    # Tries 'info' first (new code), falls back to 'ticker_info' (old code), or empty dict
    info = res.get("info", res.get("ticker_info", {}))
    
    st.markdown("---")
    st.header(f"{selected} • {info.get('longName', selected)}")
    
    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Action", res.get("final_decision", "HOLD"), delta=f"Score: {res.get('composite_score', 0):.2f}")
    k2.metric("Net Margin", f"{res.get('net_margin', 0):.2%}")
    k3.metric("Debt/Equity", f"{res.get('de_ratio', 0):.2f}", delta_color="inverse")
    k4.metric("Momentum (1Y)", f"{res.get('momentum_12m', 0):.2%}")
    
    # Charts & AI
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Analysis")
        
        # Safe Progress Bar
        raw_sent = res.get('sentiment_score', 0)
        norm_sent = (raw_sent + 1) / 2
        safe_sent = max(0.0, min(1.0, norm_sent))
        
        st.write(f"**Sentiment Analysis ({raw_sent:.2f})**")
        st.progress(safe_sent)
        
        st.info(f"**Financial AI:** {res.get('financial_summary', 'No summary available.')}")
        
        st.write("**Recent Headlines**")
        for n in res.get('news', [])[:3]:
            st.markdown(f"- [{n.get('title')}]({n.get('link')}) *({n.get('time')})*")
            
    with c2:
        st.subheader("Raw Data")
        st.json({
            "SMA50": res.get("sma50"),
            "SMA200": res.get("sma200"),
            "Market Cap": info.get("marketCap"),
            "Beta": info.get("beta")
        })

import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import google.generativeai as genai
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import re
import time
import random
from newsapi import NewsApiClient
import altair as alt
import json

# --- Page Config (Must be the first Streamlit command) ---
st.set_page_config(page_title="AI Hedge Fund Simulator (Gemini Powered)", layout="wide", initial_sidebar_state="expanded")

# Load environment variables
load_dotenv()

# SEC EDGAR User-Agent (important for compliance)
SEC_USER_AGENT = "KevalAhirApp/1.0 keval.ahir2019@gmail.com"

# File paths for portfolio persistence
PORTFOLIOS_FILE = "portfolios.json"
VIRTUAL_PORTFOLIO_FILE = "virtual_portfolio.json"

# --- CSS for Aesthetics ---
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    [data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #555;
    }
    [data-testid="stMetricValue"] {
        font-size: 20px;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #0072F0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------

def _get_initial_portfolios():
    if os.path.exists(PORTFOLIOS_FILE):
        try:
            with open(PORTFOLIOS_FILE, 'r') as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}

def _get_initial_virtual_portfolio():
    if os.path.exists(VIRTUAL_PORTFOLIO_FILE):
        try:
            with open(VIRTUAL_PORTFOLIO_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return { "cash": 3500.0, "holdings": [], "transaction_history": [], "last_scan_date": None }

if 'portfolios_data' not in st.session_state:
    st.session_state.portfolios_data = _get_initial_portfolios()
if 'selected_portfolio_name' not in st.session_state:
    st.session_state.selected_portfolio_name = list(st.session_state.portfolios_data.keys())[0] if st.session_state.portfolios_data else None
if 'portfolio_stock_analysis' not in st.session_state:
    st.session_state.portfolio_stock_analysis = {}
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = {}
if 'live_output' not in st.session_state:
    st.session_state.live_output = {}
if 'virtual_portfolio' not in st.session_state:
    st.session_state.virtual_portfolio = _get_initial_virtual_portfolio()
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "Live Analysis"
if 'live_analysis_triggered' not in st.session_state:
    st.session_state.live_analysis_triggered = False
if 'backtest_triggered' not in st.session_state:
    st.session_state.backtest_triggered = False

# --------------------------------
# Global Utility Functions
# --------------------------------
def get_signal_color(signal):
    signal = str(signal).upper()
    if signal in ["BUY", "STRONG_BUY"]: return "green"
    if signal in ["SELL", "STRONG_SELL"]: return "red"
    return "orange"

def save_portfolios(portfolios_data):
    with open(PORTFOLIOS_FILE, 'w') as f:
        json.dump(portfolios_data, f, indent=4)

def save_virtual_portfolio(data):
    with open(VIRTUAL_PORTFOLIO_FILE, 'w') as f:
        json.dump(data, f, indent=4, default=str)

def get_default_virtual_portfolio():
    return { "cash": 3500.0, "holdings": [], "transaction_history": [], "last_scan_date": None }

# --------------------------------
# Data Fetchers (yfinance + Gemini Optimized)
# --------------------------------

@st.cache_data(ttl=300)
def fetch_price_history(ticker: str, period: str = "max", interval: str = "1d") -> pd.DataFrame:
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, interval=interval)
        if df.empty: return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_ticker_info(ticker: str) -> dict:
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        if not info or 'regularMarketPrice' not in info:
             pass
        return info
    except Exception: return {}

@st.cache_data(ttl=600)
def fetch_financial_statements(ticker: str):
    """Fetches Income Statement and Balance Sheet."""
    try:
        t = yf.Ticker(ticker)
        # Fetching annual statements
        inc = t.income_stmt
        bal = t.balance_sheet
        
        # Convert to a simpler dict format for LLM consumption
        inc_recent = inc.iloc[:, 0].to_dict() if not inc.empty else {}
        bal_recent = bal.iloc[:, 0].to_dict() if not bal.empty else {}
        
        return {"income_statement": inc_recent, "balance_sheet": bal_recent, "currency": t.info.get('currency', 'USD')}
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=300)
def fetch_enriched_news(ticker: str, ticker_info_data: dict) -> list[dict]:
    try:
        ticker_obj = yf.Ticker(ticker)
        raw_news = ticker_obj.news
        enriched = []
        for n in raw_news:
            enriched.append({
                "title": n.get('title'),
                "link": n.get('link'),
                "publisher": n.get('publisher'),
                "publish_time_readable": datetime.fromtimestamp(n.get('providerPublishTime', 0)).strftime('%Y-%m-%d'),
                "content_snippet": n.get('relatedTickers') 
            })
        return enriched
    except Exception: return []

# --------------------------------
# Gemini LLM Client
# --------------------------------
class ModelClient:
    """Manages interaction with Google Gemini."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        if not self.api_key: raise ValueError("API key required for LLM.")
        
        genai.configure(api_key=self.api_key)
        # Fallback logic for model selection
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except:
            self.model = genai.GenerativeModel('gemini-pro')

    def generate(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Gemini Error: {e}"

# --------------------------------
# Agents
# --------------------------------

class FinancialsAgent:
    """Analyzes Income Statement and Balance Sheet."""
    def __init__(self, client):
        self.client = client
        
    def run(self, ticker: str, data: dict) -> dict:
        fin_data = data.get("financials", {})
        if "error" in fin_data:
            return {"ticker": ticker, "financial_signal": "hold", "financial_summary": "Data unavailable."}
        
        inc = fin_data.get("income_statement", {})
        bal = fin_data.get("balance_sheet", {})
        
        try:
            total_assets = bal.get("Total Assets", 1)
            total_liab = bal.get("Total Liabilities Net Minority Interest", 0)
            equity = bal.get("Stockholders Equity", 1)
            net_income = inc.get("Net Income", 0)
            revenue = inc.get("Total Revenue", 1)
            
            debt_to_equity = total_liab / equity if equity else 0
            net_margin = net_income / revenue if revenue else 0
            
            context = f"""
            Company: {ticker}
            Net Income: {net_income}
            Total Revenue: {revenue}
            Total Assets: {total_assets}
            Total Debt: {total_liab}
            Calculated Net Margin: {net_margin:.2%}
            Calculated Debt/Equity: {debt_to_equity:.2f}
            """
            
            prompt = f"""
            Analyze these financials for {ticker}. 
            1. Is the Debt-to-Equity ratio healthy (< 2.0)?
            2. Is the Net Margin positive?
            
            Context: {context}
            
            Output a short summary (max 30 words) and a signal (BUY/SELL/HOLD).
            Format: Signal: [SIGNAL] | Summary: [Summary]
            """
            
            response = self.client.generate(prompt)
            
            sig = "HOLD"
            if "BUY" in response.upper(): sig = "BUY"
            elif "SELL" in response.upper(): sig = "SELL"
            
            summary = response.split("Summary:")[-1].strip() if "Summary:" in response else response
            
            return {
                "ticker": ticker,
                "financial_signal": sig.lower(),
                "financial_summary": summary,
                "debt_to_equity": debt_to_equity,
                "net_margin": net_margin
            }
            
        except Exception as e:
            return {"ticker": ticker, "financial_signal": "hold", "financial_summary": f"Error calculating metrics: {e}"}

class PriceAgent:
    def run(self, ticker: str, price_data_slice: pd.DataFrame) -> dict:
        if price_data_slice.empty or len(price_data_slice) < 200:
            return {"ticker": ticker, "price_signal": "hold", "sma50": np.nan, "price_confidence_score": 0}
        
        df = price_data_slice.copy()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["SMA200"] = df["Close"].rolling(200).mean()
        latest = df.iloc[-1]
        
        sig = "hold"
        score = 0.0
        if latest.SMA50 > latest.SMA200:
            sig = "buy"; score = 0.5
        elif latest.SMA50 < latest.SMA200:
            sig = "sell"; score = -0.5
            
        return {"ticker": ticker, "price_signal": sig, "sma50": latest.SMA50, "sma200": latest.SMA200, "price_confidence_score": score}

class MomentumAgent:
    def run(self, ticker: str, price_data_slice: pd.DataFrame) -> dict:
        if len(price_data_slice) < 252: return {"ticker": ticker, "momentum_signal": "hold", "momentum_12m": 0}
        current = price_data_slice["Close"].iloc[-1]
        prev_12m = price_data_slice["Close"].iloc[-252]
        mom = (current / prev_12m) - 1
        sig = "buy" if mom > 0.1 else ("sell" if mom < -0.1 else "hold")
        return {"ticker": ticker, "momentum_signal": sig, "momentum_12m": mom, "momentum_confidence_score": mom}

class SentimentAgent:
    def __init__(self, client): self.client = client
    def run(self, ticker: str, data: dict) -> dict:
        news = data.get("news", [])[:5] 
        if not news: return {"ticker": ticker, "sentiment_signal": "hold", "sentiment_score": 0}
        
        txt = "\n".join([f"- {n.get('title')}" for n in news])
        prompt = f"""
        Analyze sentiment for {ticker} based on these headlines:
        {txt}
        
        Output only a score from -1.0 (Very Negative) to 1.0 (Very Positive).
        """
        try:
            resp = self.client.generate(prompt)
            match = re.search(r"([-+]?\d*\.\d+)|([-+]?\d+)", resp)
            score = float(match.group(0)) if match else 0
            # Normalize and clamp score
            score = max(-1.0, min(1.0, score))
            
            sig = "buy" if score > 0.3 else ("sell" if score < -0.3 else "hold")
            return {"ticker": ticker, "sentiment_signal": sig, "sentiment_score": score, "sentiment_confidence_score": abs(score)}
        except: return {"ticker": ticker, "sentiment_signal": "hold", "sentiment_score": 0}

class PortfolioAgent:
    def run(self, ticker: str, signals: list) -> dict:
        score = 0
        weights = {"price": 1.0, "momentum": 0.8, "sentiment": 0.7, "financial": 1.2}
        
        for s in signals:
            if "price_signal" in s: score += (1 if s["price_signal"]=="buy" else -1 if s["price_signal"]=="sell" else 0) * weights["price"]
            if "momentum_signal" in s: score += (1 if s["momentum_signal"]=="buy" else -1 if s["momentum_signal"]=="sell" else 0) * weights["momentum"]
            if "sentiment_signal" in s: score += (1 if s["sentiment_signal"]=="buy" else -1 if s["sentiment_signal"]=="sell" else 0) * weights["sentiment"]
            if "financial_signal" in s: score += (1 if s["financial_signal"]=="buy" else -1 if s["financial_signal"]=="sell" else 0) * weights["financial"]
            
        final_decision = "buy" if score > 0.5 else ("sell" if score < -0.5 else "hold")
        return {"ticker": ticker, "composite_score": score, "final_decision": final_decision}

# --------------------------------
# Analysis Orchestration
# --------------------------------

def run_live_analysis(tickers, llm_client):
    results = {}
    progress_bar = st.progress(0, text="Starting analysis...")
    
    for i, t in enumerate(tickers):
        progress_bar.progress((i + 1) / len(tickers), text=f"Analyzing {t}...")
        
        ph = fetch_price_history(t)
        info = fetch_ticker_info(t)
        news = fetch_enriched_news(t, info)
        financials = fetch_financial_statements(t)
        
        data_bundle = {
            "price_history": ph,
            "ticker_info": info,
            "news": news,
            "financials": financials
        }
        
        p_res = PriceAgent().run(t, ph)
        m_res = MomentumAgent().run(t, ph)
        s_res = SentimentAgent(llm_client).run(t, data_bundle)
        f_res = FinancialsAgent(llm_client).run(t, data_bundle)
        
        signals = [p_res, m_res, s_res, f_res]
        final = PortfolioAgent().run(t, signals)
        
        combined = {**p_res, **m_res, **s_res, **f_res, **final}
        combined["ticker_info"] = info
        combined["current_price"] = info.get("currentPrice", ph["Close"].iloc[-1] if not ph.empty else 0)
        combined["news"] = news
        results[t] = combined
        
    progress_bar.empty()
    return results

# --------------------------------
# Display Functions (Aesthetic & Tooltips)
# --------------------------------

def display_detailed_analysis(res):
    t = res.get("ticker")
    info = res.get("ticker_info", {})
    
    st.header(f"{t} - {info.get('longName', t)}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Final Decision", res.get("final_decision", "HOLD").upper(), 
                  delta=f"Score: {res.get('composite_score', 0):.2f}",
                  help="AI aggregated decision based on Price, Momentum, Sentiment, and Financials.")
    with col2:
        st.metric("Current Price", f"${res.get('current_price', 0):.2f}",
                  help="Latest market price fetched from Yahoo Finance.")
    with col3:
        st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.2f}B",
                  help="Total value of company shares (Stock Price * Shares Outstanding).")
    with col4:
        rec = info.get('recommendationKey', 'none').upper().replace('_', ' ')
        st.metric("Analyst Rating", rec, help="Consensus rating from Wall Street analysts.")

    tabs = st.tabs(["📊 Financials & Ratios", "📈 Technicals", "📰 News & Sentiment", "🤖 AI Logic"])

    with tabs[0]:
        st.subheader("Financial Health Analysis")
        f_col1, f_col2, f_col3 = st.columns(3)
        
        f_col1.metric("Financial Signal", res.get("financial_signal", "hold").upper(),
                      help="AI derived signal based on Income Statement and Balance Sheet health.")
        
        f_col2.metric("Debt-to-Equity", f"{res.get('debt_to_equity', 0):.2f}",
                      delta="-High Risk" if res.get('debt_to_equity', 0) > 2 else "Healthy",
                      delta_color="inverse",
                      help="Formula: Total Liabilities / Shareholder Equity. Indicates financial leverage. > 2.0 is often considered risky.")
        
        f_col3.metric("Net Profit Margin", f"{res.get('net_margin', 0):.2%}",
                      help="Formula: Net Income / Total Revenue. Percentage of revenue left after all expenses.")
        
        with st.container(border=True):
            st.markdown("**AI Financial Summary:**")
            st.info(res.get("financial_summary", "No summary available."))

    with tabs[1]:
        st.subheader("Technical Indicators")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.metric("Momentum (12M)", f"{res.get('momentum_12m', 0):.2%}",
                      help="Formula: (Current Price / Price 12 Months Ago) - 1. Shows year-over-year trend.")
        with t_col2:
            st.metric("50-Day SMA", f"${res.get('sma50', 0):.2f}",
                      help="Simple Moving Average of the last 50 trading days. Used for short-term trends.")
            
    with tabs[2]:
        st.subheader("News Sentiment")
        s_score = res.get("sentiment_score", 0)
        
        # --- FIX: CLAMP PROGRESS BAR VALUE ---
        # Ensure the value passed to st.progress is strictly between 0.0 and 1.0
        normalized_score = (s_score + 1) / 2
        safe_progress = max(0.0, min(1.0, normalized_score))
        
        st.progress(safe_progress, text=f"Sentiment Score: {s_score:.2f} (-1.0 to 1.0)")
        
        for n in res.get("news", [])[:3]:
            with st.container(border=True):
                st.markdown(f"**[{n['title']}]({n['link']})**")
                st.caption(f"{n['publisher']} • {n['publish_time_readable']}")

    with tabs[3]:
        st.json(res)

# --------------------------------
# Main Execution
# --------------------------------

# Setup Gemini Client
api_key = st.secrets.get("API_KEY")
if not api_key:
    st.error("Please set API_KEY in .streamlit/secrets.toml")
    st.stop()

llm_client = ModelClient(api_key=api_key)

# Sidebar
st.sidebar.title("🤖 AI Hedge Fund")
st.sidebar.success("Using Google Gemini (Flash) for low latency.")

mode = st.sidebar.radio("Mode", ["Live Analysis", "Virtual Portfolio"])

if mode == "Live Analysis":
    st.title("🚀 Live Market Analysis")
    tickers_input = st.text_input("Enter Tickers (comma separated)", "AAPL, GOOG, TSLA, NVDA")
    
    if st.button("Run Analysis", type="primary"):
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        st.session_state.live_output = run_live_analysis(tickers, llm_client)
        st.session_state.live_analysis_triggered = True

    if st.session_state.live_analysis_triggered and st.session_state.live_output:
        # Summary Table
        df_summary = pd.DataFrame(st.session_state.live_output.values())
        if not df_summary.empty:
            cols = ["ticker", "final_decision", "composite_score", "current_price", "financial_signal", "sentiment_signal"]
            st.dataframe(df_summary[cols].style.applymap(lambda x: f'color: {get_signal_color(x)}', subset=['final_decision']), use_container_width=True)
            
            selected = st.selectbox("Select Ticker for Deep Dive", df_summary["ticker"].tolist())
            if selected:
                display_detailed_analysis(st.session_state.live_output[selected])

elif mode == "Virtual Portfolio":
    st.title("💼 Virtual Portfolio")
    st.metric("Cash Balance", f"${st.session_state.virtual_portfolio['cash']:,.2f}")
    st.info("Virtual trading logic can be connected to the signals generated in Live Analysis.")

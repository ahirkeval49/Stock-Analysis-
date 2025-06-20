import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from openai import OpenAI
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import re
import json

# --- Page Config (Must be the first Streamlit command) ---
st.set_page_config(page_title="AI Financial Analyst", layout="wide")

# --- Environment and API Keys ---
load_dotenv()
SEC_USER_AGENT = "KevalAhirApp/1.0 keval.ahir2019@gmail.com"

# --- Portfolio Helper Functions (Your existing code) ---
PORTFOLIOS_FILE = "portfolios.json"
VIRTUAL_PORTFOLIO_FILE = "virtual_portfolio.json"

def load_portfolios():
    if os.path.exists(PORTFOLIOS_FILE):
        try:
            with open(PORTFOLIOS_FILE, 'r') as f: return json.load(f)
        except json.JSONDecodeError: return {}
    return {}

def save_portfolios(data):
    with open(PORTFOLIOS_FILE, 'w') as f: json.dump(data, f, indent=4)

def load_virtual_portfolio():
    if os.path.exists(VIRTUAL_PORTFOLIO_FILE):
        try:
            with open(VIRTUAL_PORTFOLIO_FILE, 'r') as f: return json.load(f)
        except json.JSONDecodeError: return get_default_virtual_portfolio()
    return get_default_virtual_portfolio()

def save_virtual_portfolio(data):
    with open(VIRTUAL_PORTFOLIO_FILE, 'w') as f: json.dump(data, f, indent=4, default=str)

def get_default_virtual_portfolio():
    return {"cash": 10000.0, "holdings": [], "transaction_history": [], "last_scan_date": None}

# --- Session State Initialization ---
if 'portfolios_data' not in st.session_state:
    st.session_state.portfolios_data = load_portfolios()
if 'selected_portfolio_name' not in st.session_state:
    st.session_state.selected_portfolio_name = list(st.session_state.portfolios_data.keys())[0] if st.session_state.portfolios_data else None
if 'live_output' not in st.session_state:
    st.session_state.live_output = {}
if 'virtual_portfolio' not in st.session_state:
    st.session_state.virtual_portfolio = load_virtual_portfolio()
if 'live_analysis_triggered' not in st.session_state:
    st.session_state.live_analysis_triggered = False

# --- Data Fetcher Functions ---

@st.cache_data(ttl=1800)
def get_all_data(ticker: str) -> dict:
    """Fetches all necessary data from yfinance and SEC in a single bundle."""
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        if not info or not info.get('regularMarketPrice'):
            return {"error": f"Could not retrieve valid data for {ticker}. It may be delisted."}

        data_bundle = {
            "ticker": ticker,
            "info": info,
            "price_history": ticker_obj.history(period="max", interval="1d"),
            "institutional_holders": ticker_obj.institutional_holders.to_dict("records") if ticker_obj.institutional_holders is not None else [],
            "news": ticker_obj.news if ticker_obj.news else [],
            "sec_all_filings_raw": []
        }
        
        headers = {'User-Agent': SEC_USER_AGENT}
        api_url = "https://efts.sec.gov/LATEST/search-index"
        payload = {"q": ticker.lower(), "from": 0, "size": 100, "sort": [{"filed_date": "desc"}]}
        response = requests.post(api_url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        filings_data = response.json()
        
        filings_list = []
        if filings_data.get('hits', {}).get('hits'):
            for hit in filings_data['hits']['hits']:
                source = hit.get('_source', {})
                filings_list.append({
                    "Filing Date": source.get('file_date', 'N/A')[:10],
                    "Form Type": source.get('form', 'N/A'),
                    "Link": f"https://www.sec.gov/Archives/edgar/data/{source.get('ciks')[0]}/{source.get('adsh').replace('-', '')}/{source.get('adsh')}-index.html"
                })
        data_bundle["sec_all_filings_raw"] = filings_list
        return data_bundle
    except Exception as e:
        return {"error": f"A data fetching error occurred for {ticker}: {e}"}

# --- Agent Classes (Unified & Final) ---

class ModelClient:
    def __init__(self, api_key: str, provider: str):
        self.api_key, self.provider = api_key, provider
        models = {"openai": "gpt-4o", "deepseek": "deepseek-reasoner"}
        if not api_key: raise ValueError(f"{provider} API key required.")
        self.model_name = models.get(provider)
        if provider == "deepseek": self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        elif provider == "openai": self.client = OpenAI(api_key=api_key)
        else: raise ValueError(f"Unsupported provider: {provider}")
    def generate(self, prompt: str) -> str:
        try:
            stream = self.client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt}], stream=True)
            return "".join(c.choices[0].delta.content for c in stream if c.choices and c.choices[0].delta and c.choices[0].delta.content)
        except Exception as e: raise Exception(f"LLM Error ({self.provider}): {e}")

class PriceAgent:
    def run(self, data: dict) -> dict:
        price_data = data.get("price_history")
        if price_data is None or len(price_data) < 200: return {"price_signal": "hold"}
        df = price_data.copy()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["SMA200"] = df["Close"].rolling(200).mean()
        latest = df.iloc[-1]
        if pd.isna(latest.SMA50) or pd.isna(latest.SMA200): return {"price_signal": "hold"}
        signal = "buy" if latest.SMA50 > latest.SMA200 and latest.Close > latest.SMA50 else "sell"
        return {"price_signal": signal}

class MomentumAgent:
    def run(self, data: dict) -> dict:
        price_data = data.get("price_history")
        if price_data is None or len(price_data) < 253: return {"momentum_signal": "hold", "momentum_12m": np.nan}
        momentum_12m = price_data["Close"].pct_change(252).iloc[-1]
        signal = "buy" if momentum_12m > 0.15 else "sell" if momentum_12m < -0.15 else "hold"
        return {"momentum_signal": signal, "momentum_12m": momentum_12m}

class SECFilingAgent: # The original, simple agent for comparison
    def run(self, data: dict) -> dict:
        filings = data.get("sec_all_filings_raw", [])
        if not filings: return {"sec_filings_signal_original": "hold"}
        has_recent_form4 = any(f.get('Form Type') == '4' for f in filings[:10])
        return {"sec_filings_signal_original": "buy" if has_recent_form4 else "hold"}

class InstitutionalHoldingsAgent: # The original, simple agent for comparison
    def run(self, data: dict) -> dict:
        holdings = data.get("institutional_holdings", [])
        if not holdings: return {"inst_holdings_signal_original": "hold", "inst_top_holders": []}
        total_pct = sum(h.get('% Out', 0.0) for h in holdings)
        sig = "buy" if total_pct > 0.70 else "sell" if total_pct < 0.20 else "hold"
        return {"inst_holdings_signal_original": sig, "inst_top_holders": sorted(holdings, key=lambda x: x.get('Shares', 0), reverse=True)[:10]}

class SECReportAnalysisAgent: # The new, advanced agent
    def __init__(self, client: ModelClient): self.client = client
    def _fetch_filing_text(self, url: str) -> str:
        if not url: return ""
        try:
            headers = {'User-Agent': SEC_USER_AGENT}
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup.get_text(separator='\n', strip=True)
        except Exception: return "Could not fetch filing text."
    def run(self, data: dict) -> dict:
        ticker, filings = data.get("ticker"), data.get("sec_all_filings_raw", [])
        if not self.client: return {"sec_analysis": {"error": "LLM client not available."}}
        if not filings or (isinstance(filings[0], dict) and "error" in filings[0]): return {"sec_analysis": {"error": "No filings data to analyze."}}
        latest_report = next((f for f in filings if f.get('Form Type') in ['10-K', '10-Q']), None)
        if not latest_report: return {"sec_analysis": {"error": "No recent 10-K/Q found to analyze."}}
        filing_text = self._fetch_filing_text(latest_report.get('Link'))
        if len(filing_text) < 500: return {"sec_analysis": {"error": "Could not extract sufficient text."}}
        prompt = f"Analyze MD&A/Risk Factors text from {ticker}'s {latest_report['Form Type']}. Provide JSON with keys: 'summary' (3 sentences), 'key_risks' (list of 2-3 risks), 'key_opportunities' (list of 2-3 opportunities), 'management_tone' (one adjective). TEXT: {filing_text[:18000]}"
        try:
            response = self.client.generate(prompt).strip()
            match = re.search(r'\{.*\}', response, re.DOTALL)
            analysis = json.loads(match.group())
            analysis['source_filing'] = f"{latest_report['Form Type']} ({latest_report['Filing Date']})"
            return {"sec_analysis": analysis}
        except Exception: return {"sec_analysis": {"error": "LLM analysis failed."}}

class EnhancedInstitutionalHoldingsAgent: # The new, advanced agent
    def run(self, data: dict) -> dict:
        holdings = data.get("institutional_holdings",[])
        inst_data = {"enhanced_inst_signal": "hold", "inst_recently_reported_holders": []}
        if not holdings: return inst_data
        recent_date_limit = datetime.now() - timedelta(days=45)
        for h in holdings:
            try:
                if isinstance(h.get('Date Reported'), str) and datetime.strptime(h['Date Reported'], '%Y-%m-%d') > recent_date_limit:
                    inst_data["inst_recently_reported_holders"].append(h)
            except: continue
        if len(inst_data["inst_recently_reported_holders"]) > 3: inst_data["enhanced_inst_signal"] = "buy"
        return inst_data

class PortfolioAgent:
    WEIGHTS = {"price": 1.2, "momentum": 1.0, "analyst": 0.8, "sec_filings_original": 1.5, "inst_holdings_original": 0.7, "enhanced_inst": 1.0}
    def run(self, ticker: str, signals: list[dict]) -> dict:
        total_score, sum_w, agg_s = 0.0, 0.0, {}
        for s_dict in signals:
            if isinstance(s_dict, dict): agg_s.update(s_dict)
        s_map = {k + "_signal": k for k in self.WEIGHTS.keys()}
        for s_key, w_key in s_map.items():
            s_val, w = agg_s.get(s_key), self.WEIGHTS.get(w_key, 0)
            if s_val and w > 0:
                raw_score = {"buy": 1.0, "hold": 0.0, "sell": -1.0}.get(str(s_val).lower(), 0)
                total_score += raw_score * w; sum_w += w
        comp_score = (total_score / sum_w) if sum_w else 0.0
        decision = "buy" if comp_score > 0.25 else ("sell" if comp_score < -0.25 else "hold")
        return {"composite_score": comp_score, "final_decision": decision}

# --- Main Orchestrator & UI ---

def run_live_analysis(tickers, llm_client):
    results = {}
    progress_bar = st.progress(0, text="Starting analysis...")
    for i, t in enumerate(tickers):
        progress_text = f"Analyzing {t}... ({i+1}/{len(tickers)})"
        progress_bar.progress((i + 1) / len(tickers), text=progress_text)
        
        data_bundle = get_all_data(t)
        if "error" in data_bundle:
            results[t] = data_bundle; continue
        
        agents = [PriceAgent(), MomentumAgent(), AnalystRatingAgent(), SECFilingAgent(), InstitutionalHoldingsAgent(), EnhancedInstitutionalHoldingsAgent()]
        if llm_client: agents.append(SECReportAnalysisAgent(llm_client))
        
        agent_res_list = [agent.run(data_bundle) for agent in agents]
        final_dec = PortfolioAgent().run(t, agent_res_list)
        
        curr_res_dict = {}
        for r_dict in [data_bundle, *agent_res_list, final_dec]:
            if isinstance(r_dict, dict): curr_res_dict.update(r_dict)
        results[t] = curr_res_dict
        
    progress_bar.empty()
    return results

def display_detailed_analysis(res_detail):
    ticker, ticker_info = res_detail.get("ticker", "N/A"), res_detail.get("info", {})
    tab_titles = ["📈 Chart & Core", "📊 Fundamentals & Value", "📰 News & Filings", "⚙️ All Signals"]
    tabs = st.tabs(tab_titles)

    def get_signal_color(signal):
        signal = str(signal).upper()
        if "BUY" in signal: return "green"
        if "SELL" in signal: return "red"
        return "orange"

    st.subheader(f"Detailed Analysis for {ticker_info.get('longName', ticker)}")
    sig_col1, sig_col2, sig_col3 = st.columns(3)
    sig_col1.metric("Final AI Decision", str(res_detail.get('final_decision', 'N/A')).upper())
    sig_col2.metric("Composite Score", f"{res_detail.get('composite_score', 0):.2f}")
    filing_analysis = res_detail.get("filing_analysis", {})
    sig_col3.metric("Management Tone", str(filing_analysis.get('management_tone', 'N/A')).upper())
    st.markdown("---")

    with tabs[0]:
        st.subheader("Price Performance & Technical Signals")
        price_hist = res_detail.get("price_history", pd.DataFrame())
        if not price_hist.empty: st.line_chart(price_hist["Close"], use_container_width=True)
        col1, col2 = st.columns(2)
        with col1: st.metric("Price Signal", str(res_detail.get('price_signal', 'N/A')).upper())
        with col2: st.metric("Momentum Signal", str(res_detail.get('momentum_signal', 'N/A')).upper())

    with tabs[1]:
        st.subheader(f"Fundamental & Value Overview")
        if ticker_info.get('longBusinessSummary'):
            with st.expander("Show Business Summary"): st.markdown(ticker_info.get('longBusinessSummary'))
        f_col1, f_col2, f_col3 = st.columns(3)
        f_col1.metric("Market Cap", f"${ticker_info.get('marketCap', 0) / 1e9:.2f}B" if isinstance(ticker_info.get('marketCap'), (int, float)) else "N/A")
        f_col2.metric("Forward P/E", f"{ticker_info.get('forwardPE'):.2f}" if isinstance(ticker_info.get('forwardPE'), float) else "N/A")
        f_col3.metric("Return on Equity", f"{ticker_info.get('returnOnEquity', 0) * 100:.2f}%" if isinstance(ticker_info.get('returnOnEquity'), float) else "N/A")

    with tabs[2]:
        st.subheader("Filings, Ownership & News")
        with st.expander("**[NEW] AI-Powered Filing Analysis**", expanded=True):
            if filing_analysis and not filing_analysis.get("error"):
                st.success(f"**Source:** {filing_analysis.get('source_filing', 'N/A')}")
                st.write(filing_analysis.get('summary', "No summary available."))
            else: st.warning(f"AI analysis failed: {filing_analysis.get('error', 'Unknown')}")
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### SEC Filings")
            with st.popover("View Filings Data"):
                st.metric("Original Agent Signal", str(res_detail.get('sec_filings_signal_original', 'N/A')).upper())
                st.dataframe(pd.DataFrame(res_detail.get('sec_all_filings_raw', [])), hide_index=True)
        with c2:
            st.markdown("##### Institutional Ownership")
            with st.popover("View Holder Details"):
                st.metric("Original Agent Signal", str(res_detail.get('inst_holdings_signal_original', 'N/A')).upper())
                st.metric("[NEW] Recent Activity Signal", str(res_detail.get('enhanced_inst_signal', 'N/A')).upper())
                st.dataframe(pd.DataFrame(res_detail.get('inst_top_holders', [])), hide_index=True)

    with tabs[3]:
        st.subheader("All Agent Signals")
        signals_data = {k: v for k, v in res_detail.items() if k.endswith('_signal')}
        st.dataframe(pd.DataFrame(signals_data.items(), columns=["Agent", "Signal"]))
        with st.expander("View Full Raw Data Dictionary"):
            st.json({k:v for k, v in res_detail.items() if not isinstance(v, pd.DataFrame)})

# --- Main App UI ---
llm_client = None
if 'DEEPSEEK_API_KEY' in st.secrets:
    try:
        llm_client = ModelClient(api_key=st.secrets["DEEPSEEK_API_KEY"], provider="deepseek")
        st.sidebar.caption("✅ LLM: DeepSeek Reasoner")
    except Exception as e:
        st.sidebar.error(f"LLM Init Error: {e}")
else:
    st.sidebar.warning("DeepSeek API key missing.")

st.title("🚀 AI Financial Analyst")
st.header("⚙️ Configuration")
with st.container(border=True):
    tickers_in_live = st.text_input("Enter Ticker Symbols (comma-separated):", "AAPL,LULU,MSFT")
    
    if st.button("🚀 Run Live Analysis", use_container_width=True, type="primary"):
        live_tickers = [t.strip().upper() for t in tickers_in_live.split(",") if t.strip()]
        if not live_tickers:
            st.error("Please enter at least one ticker.")
        else:
            with st.spinner("Gathering data and running AI analysis..."):
                st.session_state.live_output = run_live_analysis(live_tickers, llm_client)
                st.session_state.live_analysis_triggered = True
                st.rerun()

st.header("📊 Live Analysis Results")
if st.session_state.get('live_analysis_triggered'):
    live_output = st.session_state.live_output
    cols = st.columns(len(live_output))
    for idx, (sym, res) in enumerate(live_output.items()):
        with cols[idx]:
            if res.get("error"):
                st.error(f"**{sym}**: {res.get('error')}")
                continue
            with st.container(border=True):
                dec = res.get("final_decision", "N/A").upper()
                price = res.get("info", {}).get("currentPrice", "N/A")
                price_str = f"${price:.2f}" if isinstance(price, (int, float)) else "N/A"
                st.metric(label=f"{sym} ({price_str})", value=dec, delta=f"Score: {res.get('composite_score', 0):.2f}")
                with st.expander(f"View Detailed Thesis for {sym}"):
                    display_detailed_analysis(res)

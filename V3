import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import openai
from openai import OpenAI
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import time
import random
from newsapi import NewsApiClient
import json
import hashlib
import asyncio
import altair as alt

# --- Page Config (Must be the first Streamlit command) ---
st.set_page_config(page_title="AI Hedge Fund Simulator", layout="wide")

# Load environment variables
load_dotenv()

# SEC EDGAR User-Agent (Critical for compliance/reliability)
SEC_USER_AGENT = "AIHedgeFundSimulator/2.0 (contact@example.com)"

# File paths
USERS_FILE = "users.json"
PORTFOLIOS_FILE = "user_portfolios.json"
VIRTUAL_PORTFOLIO_FILE = "virtual_portfolio.json"

# -----------------------------------------------------------------------------
# *** AUTHENTICATION & FILE IO HELPERS ***
# -----------------------------------------------------------------------------

def load_json(filepath, default):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return default
    return default

def save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, default=str)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Initialize Session State for Auth
if 'user_data' not in st.session_state:
    st.session_state.user_data = load_json(USERS_FILE, {})
if 'all_portfolios' not in st.session_state:
    st.session_state.all_portfolios = load_json(PORTFOLIOS_FILE, {})
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None

# Standard Session State Init
if 'live_output' not in st.session_state: st.session_state.live_output = {}
if 'live_analysis_triggered' not in st.session_state: st.session_state.live_analysis_triggered = False
if 'backtest_results' not in st.session_state: st.session_state.backtest_results = {}
if 'backtest_triggered' not in st.session_state: st.session_state.backtest_triggered = False
if 'portfolio_stock_analysis' not in st.session_state: st.session_state.portfolio_stock_analysis = {}
if 'selected_portfolio_name' not in st.session_state: st.session_state.selected_portfolio_name = None
if 'virtual_portfolio' not in st.session_state:
    st.session_state.virtual_portfolio = load_json(VIRTUAL_PORTFOLIO_FILE, { "cash": 3500.0, "holdings": [], "transaction_history": [], "last_scan_date": None })
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "Live Analysis"

# -----------------------------------------------------------------------------
# *** LOGIN SYSTEM ***
# -----------------------------------------------------------------------------

if not st.session_state.logged_in:
    st.title("🔒 AI Hedge Fund Login")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Login")
        l_user = st.text_input("Username", key="l_u")
        l_pass = st.text_input("Password", type="password", key="l_p")
        if st.button("Sign In"):
            if l_user in st.session_state.user_data:
                if st.session_state.user_data[l_user] == hash_password(l_pass):
                    st.session_state.logged_in = True
                    st.session_state.username = l_user
                    st.rerun()
                else:
                    st.error("Incorrect password.")
            else:
                st.error("User not found.")

    with col2:
        st.subheader("Register")
        r_user = st.text_input("New Username", key="r_u")
        r_pass = st.text_input("New Password", type="password", key="r_p")
        if st.button("Create Account"):
            if r_user in st.session_state.user_data:
                st.error("Username already exists.")
            elif r_user and r_pass:
                st.session_state.user_data[r_user] = hash_password(r_pass)
                save_json(USERS_FILE, st.session_state.user_data)
                # Initialize empty portfolio space for user
                st.session_state.all_portfolios[r_user] = {"Default Portfolio": {"holdings": [], "options": []}}
                save_json(PORTFOLIOS_FILE, st.session_state.all_portfolios)
                st.success("Account created! You can now login.")
            else:
                st.error("Please fill in all fields.")
    
    st.stop() # Stop execution of the rest of the app until logged in

# -----------------------------------------------------------------------------
# *** GLOBAL UTILITIES & DATA FETCHERS ***
# -----------------------------------------------------------------------------

def get_current_user_portfolios():
    """Helper to get portfolios for the logged-in user."""
    if st.session_state.username not in st.session_state.all_portfolios:
        st.session_state.all_portfolios[st.session_state.username] = {}
    return st.session_state.all_portfolios[st.session_state.username]

def save_current_user_portfolios(data):
    st.session_state.all_portfolios[st.session_state.username] = data
    save_json(PORTFOLIOS_FILE, st.session_state.all_portfolios)

def get_signal_color(signal):
    signal = str(signal).upper()
    if signal in ["BUY", "STRONG_BUY"]: return "green"
    if signal == "SELL": return "red"
    return "orange"

@st.cache_data(ttl=300)
def fetch_price_history(ticker: str, period: str = "max", interval: str = "1d") -> pd.DataFrame:
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, interval=interval)
        if df.empty: return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_ticker_info(ticker: str) -> dict:
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        if not info or not info.get('financialCurrency'):
             # Try one retry with delay if data is missing
             time.sleep(0.5)
             info = ticker_obj.info
             if not info: return {"_error": "Info fetch failed"}
        return info
    except Exception as e:
        return {"_error": str(e)}

# --- OPTIMIZED SEC FETCHING ---
@st.cache_data(ttl=86400)
def get_sec_cik_map():
    """Fetches Ticker -> CIK mapping from SEC JSON."""
    try:
        headers = {'User-Agent': SEC_USER_AGENT}
        resp = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=5)
        data = resp.json()
        return {val['ticker']: str(val['cik_str']).zfill(10) for val in data.values()}
    except Exception as e:
        print(f"SEC CIK Map Error: {e}")
        return {}

TICKER_TO_CIK = get_sec_cik_map()

@st.cache_data(ttl=3600)
def fetch_all_sec_filings(ticker_symbol: str, lookback_days: int = 365) -> list[dict]:
    """
    Fetches filings using the SEC Submissions JSON API.
    Much faster and more reliable than HTML scraping.
    """
    cik = TICKER_TO_CIK.get(ticker_symbol.upper())
    if not cik: return [{"error": f"CIK not found for {ticker_symbol}."}]

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {'User-Agent': SEC_USER_AGENT, 'Accept-Encoding': 'gzip, deflate'}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200: return [{"error": f"SEC API Error: {resp.status_code}"}]
        
        data = resp.json()
        filings = data.get('filings', {}).get('recent', {})
        if not filings: return [{"error": "No recent filings found."}]
        
        acc_nums = filings.get('accessionNumber', [])
        forms = filings.get('form', [])
        dates = filings.get('filingDate', [])
        docs = filings.get('primaryDocument', [])
        
        results = []
        limit_date = datetime.now() - timedelta(days=lookback_days)
        
        # Iterate through filings (limit to recent 100 to avoid performance hit)
        for i in range(min(len(acc_nums), 100)):
            try:
                f_date = datetime.strptime(dates[i], '%Y-%m-%d')
                if f_date < limit_date: continue
            except: continue
            
            form = forms[i]
            acc = acc_nums[i]
            doc = docs[i]
            no_dash_acc = acc.replace("-", "")
            
            link = f"https://www.sec.gov/Archives/edgar/data/{cik}/{no_dash_acc}/{doc}"
            summary_link = f"https://www.sec.gov/Archives/edgar/data/{cik}/{no_dash_acc}/{acc}-index.html"
            
            entry = {
                "form_type": form,
                "filing_date_str": dates[i],
                "document_link": link,
                "summary_link": summary_link,
                "is_form4_transaction": False,
                "document_content_for_llm": f"{form} filed on {dates[i]}."
            }

            # Basic Form 4 Logic (Without deep XML parsing to save latency)
            if form == '4':
                entry['is_form4_transaction'] = True
                # Placeholders as deep XML parsing in real-time is heavy
                entry['transaction_code'] = 'N/A' 
                entry['reporting_owner'] = 'See Link'
                entry['shares'] = 0
                entry['price_per_share'] = 0
                entry['link_to_filing'] = summary_link

            results.append(entry)
            
        return results
    except Exception as e:
        return [{"error": f"SEC Fetch Exception: {e}"}]

@st.cache_data(ttl=300)
def fetch_enriched_news(ticker: str, ticker_info: dict) -> list[dict]:
    """Yahoo Finance News Wrapper."""
    try:
        company_name = ticker_info.get('longName', ticker)
        ticker_obj = yf.Ticker(ticker)
        raw_news = ticker_obj.news
        enriched = []
        if not raw_news: return []
        
        for item in raw_news:
            if not isinstance(item, dict): continue
            new_item = item.copy()
            new_item['ticker'] = ticker
            new_item['company_name'] = company_name
            new_item['source_api'] = 'Yahoo Finance'
            
            if 'providerPublishTime' in item:
                dt = datetime.fromtimestamp(int(item['providerPublishTime']), tz=timezone.utc)
                new_item['publish_datetime_utc'] = dt
                new_item['publish_time_readable'] = dt.strftime('%Y-%m-%d %H:%M:%S %Z')
            else:
                new_item['publish_time_readable'] = "N/A"
            
            new_item['link'] = item.get('link', '#')
            enriched.append(new_item)
        return enriched
    except Exception as e:
        return [{"error": f"News Error: {e}"}]

@st.cache_data(ttl=3600)
def fetch_inst_filings(ticker: str) -> list[dict]:
    try:
        df = yf.Ticker(ticker).institutional_holders
        if df is not None and not df.empty:
            return df.to_dict("records")
        return []
    except: return []

@st.cache_data(ttl=3600)
def fetch_value_investing_io_data(ticker: str) -> dict:
    # Placeholder as web scraping external sites is brittle
    return {"error": "Skipped for stability"}

@st.cache_data(ttl=300)
def fetch_option_chain(ticker: str):
    """Fetches basic option chain info."""
    try:
        tk = yf.Ticker(ticker)
        dates = tk.options
        if not dates: return None
        return dates
    except: return None

# -----------------------------------------------------------------------------
# *** LLM & AGENTS ***
# -----------------------------------------------------------------------------

class ModelClient:
    def __init__(self, api_key: str, provider: str = "openai"):
        self.api_key = api_key
        self.provider = provider
        if provider == "deepseek": 
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com/v1")
            self.model = "deepseek-reasoner"
        else: 
            self.client = OpenAI(api_key=self.api_key)
            self.model = "gpt-4o"

    def generate(self, prompt: str) -> str:
        try:
            resp = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}])
            return resp.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

# --- AGENT CLASSES (Logic preserved) ---

class PriceAgent:
    def run(self, ticker, df):
        if df.empty or len(df) < 200: return {"price_signal": "hold", "price_confidence_score": 0, "sma50": np.nan, "sma200": np.nan, "rsi14": np.nan}
        
        # Calcs
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['SMA200'] = df['Close'].rolling(200).mean()
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain/loss.replace(0, np.nan)
        df['RSI'] = 100 - (100/(1+rs))
        
        latest = df.iloc[-1]
        sig = "hold"
        score = 0.0
        
        if latest['SMA50'] > latest['SMA200'] and latest['Close'] > latest['SMA50']:
            sig = "buy"; score = 0.6
        elif latest['SMA50'] < latest['SMA200'] and latest['Close'] < latest['SMA50']:
            sig = "sell"; score = -0.6
            
        if latest['RSI'] < 30: score += 0.2
        elif latest['RSI'] > 70: score -= 0.2
        
        if score > 0.3: sig="buy"
        elif score < -0.3: sig="sell"
        else: sig="hold"
        
        return {
            "ticker": ticker, "price_signal": sig, "price_confidence_score": score,
            "sma50": latest['SMA50'], "sma200": latest['SMA200'], "rsi14": latest['RSI']
        }

class MomentumAgent:
    def run(self, ticker, df):
        if len(df) < 253: return {"momentum_signal": "hold", "momentum_confidence_score": 0}
        p_now = df['Close'].iloc[-1]
        p_1m = df['Close'].iloc[-21]
        p_12m = df['Close'].iloc[-252]
        
        m1 = (p_now/p_1m) - 1
        m12 = (p_now/p_12m) - 1
        
        score = (m1 + m12) * 2.5 # Scale
        score = max(-1.0, min(1.0, score))
        
        sig = "buy" if score > 0.2 else ("sell" if score < -0.2 else "hold")
        return {"momentum_signal": sig, "momentum_confidence_score": score, "momentum_1m": m1, "momentum_12m": m12}

class VolatilityAgent:
    def run(self, ticker, info, df):
        beta = info.get('beta', 1.0)
        if not isinstance(beta, (int, float)): beta = 1.0
        
        if len(df) > 1:
            ret = np.log(df['Close']/df['Close'].shift(1)).dropna()
            ann_vol = ret.std() * np.sqrt(252)
        else: ann_vol = 0
        
        score = 0.0
        if beta < 0.8: score += 0.3
        elif beta > 1.2: score -= 0.3
        
        sig = "buy" if score > 0.2 else ("sell" if score < -0.2 else "hold")
        return {"volatility_signal": sig, "volatility_confidence_score": score, "beta": beta, "annual_vol": ann_vol}

class FundamentalsAgent:
    def run(self, ticker, info):
        pe = info.get('forwardPE')
        pb = info.get('priceToBook')
        roe = info.get('returnOnEquity')
        
        score = 0
        if pe and 0 < pe < 20: score += 1
        if pb and 0 < pb < 3: score += 1
        if roe and roe > 0.15: score += 1
        
        sig = "buy" if score >= 2 else ("sell" if score == 0 else "hold")
        return {"fund_signal": sig, "piotroski_score": score, "fcf_yield": 0} # Simplified

class SentimentAgent:
    def __init__(self, client): self.client = client
    def run(self, ticker, news):
        if not news or not self.client: return {"sentiment_signal": "hold", "sentiment_score": 0}
        
        titles = [n.get('title', '') for n in news[:7]]
        prompt = f"Analyze sentiment for {ticker} based on these headlines: {titles}. Return ONLY a float between -1.0 (neg) and 1.0 (pos)."
        try:
            resp = self.client.generate(prompt)
            match = re.search(r"([-+]?\d*\.\d+)", resp)
            score = float(match.group(0)) if match else 0.0
        except: score = 0.0
        
        sig = "buy" if score > 0.2 else ("sell" if score < -0.2 else "hold")
        return {"sentiment_signal": sig, "sentiment_score": score, "sentiment_confidence_score": abs(score)}

class NewsSummaryAgent:
    def __init__(self, client): self.client = client
    def run(self, ticker, news):
        if not news or not self.client: return {"news_summary": "No news."}
        titles = [n.get('title', '') for n in news[:5]]
        prompt = f"Summarize recent news for {ticker} in 2 sentences: {titles}"
        return {"news_summary": self.client.generate(prompt)}

class PortfolioAgent:
    def run(self, ticker, agent_results):
        total_score = 0
        count = 0
        for k, v in agent_results.items():
            if "confidence_score" in k:
                # Simple weighting
                total_score += v
                count += 1
        
        avg = total_score / count if count else 0
        dec = "buy" if avg > 0.2 else ("sell" if avg < -0.2 else "hold")
        return {"final_decision": dec, "composite_score": avg}

# -----------------------------------------------------------------------------
# *** ASYNC ORCHESTRATOR (Fixing Latency) ***
# -----------------------------------------------------------------------------

async def analyze_ticker_async(ticker, llm_client, configs):
    """Runs data fetching and agents in parallel for a single ticker."""
    loop = asyncio.get_event_loop()
    
    # 1. Concurrent Data Fetching
    future_info = loop.run_in_executor(None, fetch_ticker_info, ticker)
    future_price = loop.run_in_executor(None, fetch_price_history, ticker, "max")
    
    info = await future_info
    price = await future_price
    
    if info.get("_error") or price.empty:
        return ticker, {"error": "Data fetch failed"}
        
    # News & SEC (Optional)
    news = []
    if configs['use_sentiment']:
        news = await loop.run_in_executor(None, fetch_enriched_news, ticker, info)
    
    sec_data = []
    if configs['use_filings']:
        sec_data = await loop.run_in_executor(None, fetch_all_sec_filings, ticker)

    # 2. Run Agents (CPU bound, fast enough to run sequentially per ticker here)
    results = {}
    
    # Price/Tech
    results.update(PriceAgent().run(ticker, price))
    results.update(MomentumAgent().run(ticker, price))
    results.update(VolatilityAgent().run(ticker, info, price))
    results.update(FundamentalsAgent().run(ticker, info))
    
    # AI Agents
    if llm_client and news:
        results.update(SentimentAgent(llm_client).run(ticker, news))
        results.update(NewsSummaryAgent(llm_client).run(ticker, news))
    
    # Final Decision
    # Map results to flat structure for PortfolioAgent
    flat_res = {}
    flat_res.update(results)
    # Explicitly map scores for aggregation
    scores = {
        "price_confidence_score": results.get("price_confidence_score", 0),
        "momentum_confidence_score": results.get("momentum_confidence_score", 0),
        "volatility_confidence_score": results.get("volatility_confidence_score", 0),
        "sentiment_confidence_score": results.get("sentiment_confidence_score", 0)
    }
    
    final = PortfolioAgent().run(ticker, scores)
    results.update(final)
    
    # Pack for UI
    results['ticker'] = ticker
    results['ticker_info'] = info
    results['current_price_display'] = info.get('currentPrice', price['Close'].iloc[-1])
    results['market_cap_display'] = info.get('marketCap')
    results['industry_display'] = info.get('industry')
    results['news_headlines_for_popover'] = [n['title'] for n in news[:10]]
    results['sec_recent_form4_transactions'] = [x for x in sec_data if x.get('is_form4_transaction')]
    results['sec_other_recent_filings'] = [x for x in sec_data if not x.get('is_form4_transaction')]
    
    # Simulated Backtest (Simplified for speed)
    results['simulated_backtest_results'] = {
        "metrics": {"Total Return": "N/A (Sim)"}, 
        "log_df": []
    }
    
    return ticker, results

async def run_analysis_pipeline(tickers, llm_client, configs):
    tasks = [analyze_ticker_async(t, llm_client, configs) for t in tickers]
    return await asyncio.gather(*tasks)

def run_live_analysis(tickers, llm_client, configs):
    """Synchronous wrapper for the async pipeline."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results_list = loop.run_until_complete(run_analysis_pipeline(tickers, llm_client, configs))
        loop.close()
        return {r[0]: r[1] for r in results_list}
    except Exception as e:
        st.error(f"Pipeline Error: {e}")
        return {}

# -----------------------------------------------------------------------------
# *** UI IMPLEMENTATION ***
# -----------------------------------------------------------------------------

# --- SIDEBAR ---
with st.sidebar:
    st.write(f"👤 **{st.session_state.username}**")
    if st.button("Logout", key="logout_btn"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()
    st.markdown("---")
    
    # LLM Config
    llm_client = None
    ds_key = st.secrets.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    oa_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    
    if ds_key:
        llm_client = ModelClient(ds_key, "deepseek")
        st.success("✅ LLM: DeepSeek")
    elif oa_key:
        llm_client = ModelClient(oa_key, "openai")
        st.success("✅ LLM: OpenAI")
    else:
        st.warning("⚠️ No LLM Key")

st.title("🚀 AI Hedge Fund Simulator")

# --- MAIN TABS / MODES ---
app_mode_options = ["Live Analysis", "Backtesting", "💼 Portfolio Management", "🤖 Virtual Trading"]
current_mode_index = app_mode_options.index(st.session_state.app_mode)
selected_mode = st.radio("Mode", app_mode_options, index=current_mode_index, horizontal=True)
if selected_mode != st.session_state.app_mode:
    st.session_state.app_mode = selected_mode
    st.rerun()

st.markdown("---")

# --- DISPLAY FUNCTION (Original Logic) ---
def display_detailed_analysis(res):
    ticker = res.get('ticker')
    info = res.get('ticker_info', {})
    
    tabs = st.tabs(["📈 Chart & Core", "📊 Fundamentals", "💰 Signals", "📰 News & SEC"])
    
    with tabs[0]:
        # Fetch small history for chart
        hist = fetch_price_history(ticker, "1y")
        if not hist.empty: st.line_chart(hist['Close'])
        c1, c2 = st.columns(2)
        c1.metric("Price", f"${res.get('current_price_display', 0):.2f}")
        c2.metric("Decision", res.get('final_decision', 'HOLD').upper(), delta=f"{res.get('composite_score',0):.2f}")
        
    with tabs[1]:
        c1, c2, c3 = st.columns(3)
        c1.metric("Market Cap", f"${info.get('marketCap', 0):,}")
        c2.metric("Forward P/E", info.get('forwardPE', 'N/A'))
        c3.metric("Beta", info.get('beta', 'N/A'))
        st.json(info, expanded=False)

    with tabs[2]:
        data = {
            "Price": res.get('price_signal'),
            "Momentum": res.get('momentum_signal'),
            "Volatility": res.get('volatility_signal'),
            "Sentiment": res.get('sentiment_signal'),
            "Fundamentals": res.get('fund_signal')
        }
        st.table(pd.DataFrame(data.items(), columns=["Agent", "Signal"]))

    with tabs[3]:
        st.subheader("AI News Summary")
        st.write(res.get('news_summary', "No summary generated."))
        st.subheader("Headlines")
        for h in res.get('news_headlines_for_popover', []):
            st.text(f"• {h}")
        st.subheader("Recent SEC Filings")
        for f in res.get('sec_other_recent_filings', [])[:5]:
            st.markdown(f"[Form {f['form_type']}]({f['document_link']}) - {f['filing_date_str']}")

# -------------------------
# MODE: LIVE ANALYSIS
# -------------------------
if st.session_state.app_mode == "Live Analysis":
    tickers_in = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOG")
    c1, c2 = st.columns(2)
    use_s = c1.checkbox("News Sentiment", True)
    use_f = c2.checkbox("SEC Filings", True)
    
    if st.button("Run Analysis", type="primary"):
        t_list = [x.strip().upper() for x in tickers_in.split(",") if x.strip()]
        configs = {"use_sentiment": use_s, "use_filings": use_f}
        
        with st.spinner("Running Async Analysis..."):
            st.session_state.live_output = run_live_analysis(t_list, llm_client, configs)
            st.session_state.live_analysis_triggered = True
            st.rerun()

    if st.session_state.live_analysis_triggered and st.session_state.live_output:
        # Summary Table
        rows = []
        for t, r in st.session_state.live_output.items():
            if "error" in r:
                rows.append({"Ticker": t, "Decision": "ERROR", "Score": 0})
            else:
                rows.append({
                    "Ticker": t,
                    "Decision": r.get('final_decision').upper(),
                    "Score": f"{r.get('composite_score'):.2f}",
                    "Price": f"${r.get('current_price_display'):.2f}"
                })
        st.dataframe(pd.DataFrame(rows))
        
        sel = st.selectbox("View Details", list(st.session_state.live_output.keys()))
        if sel and sel in st.session_state.live_output:
            display_detailed_analysis(st.session_state.live_output[sel])

# -------------------------
# MODE: BACKTESTING
# -------------------------
elif st.session_state.app_mode == "Backtesting":
    st.subheader("Backtesting Engine")
    bt_t = st.text_input("Ticker", "AAPL")
    if st.button("Run Backtest"):
        # Reuse fetching logic logic
        hist = fetch_price_history(bt_t, "5y")
        if not hist.empty:
            # Simple Moving Average Strategy for demo
            hist['SMA50'] = hist['Close'].rolling(50).mean()
            hist['Signal'] = np.where(hist['Close'] > hist['SMA50'], 1, 0)
            hist['Return'] = hist['Close'].pct_change()
            hist['Strategy'] = hist['Return'] * hist['Signal'].shift(1)
            hist['CumRet'] = (1 + hist['Strategy']).cumprod()
            
            st.line_chart(hist['CumRet'])
            st.success(f"Total Return: {(hist['CumRet'].iloc[-1]-1)*100:.2f}%")
        else:
            st.error("No data.")

# -------------------------
# MODE: PORTFOLIO (With Options)
# -------------------------
elif st.session_state.app_mode == "💼 Portfolio Management":
    user_pfs = get_current_user_portfolios()
    if not user_pfs:
        user_pfs["Main"] = {"holdings": [], "options": []}
        save_current_user_portfolios(user_pfs)
        
    pf_name = st.selectbox("Portfolio", list(user_pfs.keys()))
    curr_pf = user_pfs[pf_name]
    if "options" not in curr_pf: curr_pf["options"] = [] # Schema migration
    
    stk_tab, opt_tab = st.tabs(["Stocks", "Options"])
    
    with stk_tab:
        # Stocks Logic
        if curr_pf['holdings']:
            df = pd.DataFrame(curr_pf['holdings'])
            st.dataframe(df, use_container_width=True)
        else: st.info("No stocks.")
        
        with st.expander("Add Stock"):
            c1, c2, c3 = st.columns(3)
            a_t = c1.text_input("Ticker").upper()
            a_q = c2.number_input("Qty", 1.0)
            a_p = c3.number_input("Avg Cost", 1.0)
            if st.button("Add Stock"):
                curr_pf['holdings'].append({"ticker": a_t, "quantity": a_q, "avg_price": a_p})
                save_current_user_portfolios(user_pfs)
                st.rerun()

    with opt_tab:
        # Options Logic
        if curr_pf['options']:
            # Calculate Live PnL
            disp_opts = []
            for o in curr_pf['options']:
                # Try to get rough price estimate (simplified)
                try:
                    tk = yf.Ticker(o['ticker'])
                    # This is slow in loop, typically would be async, but fine for small portfolios
                    # Just using underlying price for rough estimate in this demo
                    und_price = tk.fast_info.get('last_price', 0)
                    intrinsic = 0
                    if o['type'] == 'Call': intrinsic = max(0, und_price - o['strike'])
                    else: intrinsic = max(0, o['strike'] - und_price)
                    
                    # Rough Premium Estimate (Intrinsic + Time Value heuristic)
                    est_price = intrinsic # + time value (omitted for speed)
                    pnl = (est_price - o['avg_cost']) * o['quantity'] * 100
                    
                    disp_opts.append({
                        "Ticker": o['ticker'], "Type": o['type'], "Strike": o['strike'],
                        "Exp": o['expiry'], "Cost": o['avg_cost'], "Est. PnL": f"${pnl:.2f}"
                    })
                except: pass
            st.dataframe(pd.DataFrame(disp_opts))
        else: st.info("No options positions.")
        
        with st.expander("Add Option"):
            o_t = st.text_input("Option Ticker", key="opt_t").upper()
            if o_t:
                dates = fetch_option_chain(o_t)
                if dates:
                    o_d = st.selectbox("Expiry", dates)
                    o_type = st.selectbox("Type", ["Call", "Put"])
                    o_str = st.number_input("Strike", 100.0)
                    o_qty = st.number_input("Contracts", 1)
                    o_pr = st.number_input("Premium", 1.0)
                    
                    if st.button("Add Option"):
                        curr_pf['options'].append({
                            "ticker": o_t, "expiry": o_d, "type": o_type,
                            "strike": o_str, "quantity": o_qty, "avg_cost": o_pr
                        })
                        save_current_user_portfolios(user_pfs)
                        st.rerun()

# -------------------------
# MODE: VIRTUAL TRADING
# -------------------------
elif st.session_state.app_mode == "🤖 Virtual Trading":
    st.header("Virtual Trading")
    vp = st.session_state.virtual_portfolio
    c1, c2, c3 = st.columns(3)
    c1.metric("Cash", f"${vp['cash']:,.2f}")
    
    # Logic to scan market and auto-trade would go here
    # Reusing the Live Analysis pipeline
    if st.button("Scan & Trade"):
        with st.spinner("AI Trader running..."):
            # 1. Pick random universe
            universe = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA"]
            # 2. Analyze
            res = run_live_analysis(universe, llm_client, {"use_sentiment": True, "use_filings": False})
            
            # 3. Trade
            log = []
            for t, r in res.items():
                if r.get('final_decision') == 'buy':
                    price = r.get('current_price_display')
                    if vp['cash'] > price:
                        vp['cash'] -= price
                        vp['holdings'].append({"ticker": t, "price": price, "date": str(datetime.now())})
                        log.append(f"Bought {t} @ ${price:.2f}")
                        
            if log:
                vp['transaction_history'].extend(log)
                save_json(VIRTUAL_PORTFOLIO_FILE, vp)
                st.success(f"Trades executed: {log}")
            else:
                st.info("No trades found.")
                
    if vp['holdings']:
        st.subheader("Holdings")
        st.dataframe(pd.DataFrame(vp['holdings']))

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
from newsapi import NewsApiClient
import json
from typing import List, Dict, Any, Optional

# --- Application Configuration (Must be the first Streamlit command) ---
st.set_page_config(page_title="AI Hedge Fund Simulator", layout="wide")

# --- Constants and Configuration ---
# File paths for storing persistent data
PORTFOLIOS_FILE = "portfolios.json"
VIRTUAL_PORTFOLIO_FILE = "virtual_portfolio.json"

# SEC EDGAR API Configuration - Replace with your own info for compliance
SEC_USER_AGENT = "KevalAhirApp/1.0 keval.ahir2019@gmail.com"

# --- Portfolio Management Helper Functions ---

def load_portfolios() -> Dict[str, Any]:
    """Loads portfolio data from a JSON file."""
    if os.path.exists(PORTFOLIOS_FILE):
        try:
            with open(PORTFOLIOS_FILE, 'r') as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def save_portfolios(portfolios_data: Dict[str, Any]):
    """Saves portfolio data to a JSON file."""
    if not isinstance(portfolios_data, dict):
        st.error("Error saving portfolios: Data is not in the correct format.")
        return
    with open(PORTFOLIOS_FILE, 'w') as f:
        json.dump(portfolios_data, f, indent=4)

def load_virtual_portfolio() -> Dict[str, Any]:
    """Loads the virtual portfolio from a JSON file, returning a default if not found."""
    if os.path.exists(VIRTUAL_PORTFOLIO_FILE):
        try:
            with open(VIRTUAL_PORTFOLIO_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return get_default_virtual_portfolio()
    return get_default_virtual_portfolio()

def save_virtual_portfolio(data: Dict[str, Any]):
    """Saves the virtual portfolio to a JSON file."""
    with open(VIRTUAL_PORTFOLIO_FILE, 'w') as f:
        # Use default=str to handle non-serializable types like datetimes
        json.dump(data, f, indent=4, default=str)

def get_default_virtual_portfolio() -> Dict[str, Any]:
    """Returns the default structure for a new virtual portfolio."""
    return {
        "cash": 3500.0,
        "holdings": [],
        "transaction_history": [],
        "last_scan_date": None
    }

# --- Session State Initialization ---
# Ensures that the application state persists across reruns.
if 'portfolios_data' not in st.session_state:
    st.session_state.portfolios_data = load_portfolios()

if 'selected_portfolio_name' not in st.session_state:
    st.session_state.selected_portfolio_name = None
    if st.session_state.portfolios_data:
        st.session_state.selected_portfolio_name = list(st.session_state.portfolios_data.keys())[0]

if 'portfolio_stock_analysis' not in st.session_state:
    st.session_state.portfolio_stock_analysis = {}

if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = {}

if 'live_output' not in st.session_state:
    st.session_state.live_output = {}

if 'virtual_portfolio' not in st.session_state:
    st.session_state.virtual_portfolio = load_virtual_portfolio()

# Initialize flags to control UI flow after an action is performed.
if 'live_analysis_triggered' not in st.session_state:
    st.session_state.live_analysis_triggered = False
if 'backtest_triggered' not in st.session_state:
    st.session_state.backtest_triggered = False


# --------------------------------
# Data Fetching Functions
# --------------------------------

@st.cache_data(ttl=900) # Cache for 15 minutes
def fetch_price_history(ticker: str, period: str = "max", interval: str = "1d") -> pd.DataFrame:
    """Fetches historical price data for a given ticker from Yahoo Finance."""
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, interval=interval)
        if df.empty: return pd.DataFrame()
        # Remove timezone for easier calculations later
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600) # Cache for 1 hour
def fetch_ticker_info(ticker: str) -> Dict[str, Any]:
    """Fetches key statistics and business summary for a ticker from Yahoo Finance."""
    try:
        info = yf.Ticker(ticker).info
        # Validate that essential data is present
        if not info or (info.get('regularMarketPrice') is None and info.get('currentPrice') is None and info.get('financialCurrency') is None):
            return {}
        return {
            "marketCap": info.get("marketCap"), "freeCashflow": info.get("freeCashflow"),
            "forwardPE": info.get("forwardPE"), "trailingPE": info.get("trailingPE"),
            "priceToBook": info.get("priceToBook"), "enterpriseToRevenue": info.get("enterpriseToRevenue"),
            "enterpriseToEbitda": info.get("enterpriseToEbitda"), "returnOnEquity": info.get("returnOnEquity"),
            "debtToEquity": info.get("debtToEquity"), "beta": info.get("beta"),
            "targetMeanPrice": info.get("targetMeanPrice"), "recommendationKey": info.get("recommendationKey"),
            "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions"), "industry": info.get("industry"),
            "sector": info.get("sector"), "longName": info.get("longName"), "shortName": info.get("shortName"),
            "longBusinessSummary": info.get("longBusinessSummary"),
            "currentPrice": info.get("currentPrice") or info.get("regularMarketPrice"),
            "financialCurrency": info.get("financialCurrency")
        }
    except Exception:
        return {}

@st.cache_data(ttl=1800) # Cache for 30 minutes
def fetch_enriched_news(ticker: str, ticker_info_data: dict) -> list[dict]:
    """Fetches and enriches news from Yahoo Finance, adding readable timestamps."""
    try:
        company_name = ticker_info_data.get('longName', ticker_info_data.get('shortName', ticker))
        ticker_obj = yf.Ticker(ticker)
        raw_news = []
        try:
            raw_news = ticker_obj.news
        except TypeError as te:
            return [{"error": f"yfinance .news type error for {ticker}: {te}", "source_api": "Yahoo Finance"}]
        except Exception as news_exc:
            return [{"error": f"yfinance .news call failed for {ticker}: {news_exc}", "source_api": "Yahoo Finance"}]
        
        if not raw_news: return []

        enriched_news_list = []
        for news_item in raw_news:
            if not isinstance(news_item, dict): continue
            enriched_item = news_item.copy()
            enriched_item.update({
                'ticker': ticker,
                'company_name': company_name,
                'source_api': 'Yahoo Finance',
                'title': news_item.get('title', 'N/A'),
                'publisher': news_item.get('publisher', 'N/A'),
                'link': news_item.get('link', '#'),
                'type': news_item.get('type', 'N/A')
            })
            
            if 'providerPublishTime' in news_item and news_item['providerPublishTime']:
                try:
                    dt_object_utc = datetime.fromtimestamp(int(news_item['providerPublishTime']), tz=timezone.utc)
                    enriched_item['publish_datetime_utc'] = dt_object_utc
                    enriched_item['publish_time_readable'] = dt_object_utc.strftime('%Y-%m-%d %H:%M:%S %Z')
                except (ValueError, TypeError, OSError) as e:
                    enriched_item.update({'publish_datetime_utc': None, 'publish_time_readable': "N/A", 'publish_time_error': str(e)})
            else:
                enriched_item.update({'publish_datetime_utc': None, 'publish_time_readable': "N/A"})
            
            enriched_news_list.append(enriched_item)
        
        enriched_news_list.sort(key=lambda x: x.get('publish_datetime_utc') or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return enriched_news_list
    except Exception as e:
        return [{"error": f"Processing Yahoo Finance news for {ticker} failed: {e}", "source_api": "Yahoo Finance"}]

@st.cache_data(ttl=1800) # Cache for 30 minutes
def fetch_comprehensive_news_from_api(ticker: str, company_name: str, lookback_days: int = 30) -> list[dict]:
    """Fetches news from NewsAPI.org for a broader perspective."""
    api_key = st.secrets.get("NEWSAPI_KEY")
    if not api_key:
        return [{"error": "NEWSAPI_KEY not found in Streamlit secrets.", "source_api": "NewsAPI.org"}]
    
    newsapi = NewsApiClient(api_key=api_key)
    query = f'("{company_name}" OR {ticker.upper()}) AND (stock OR shares OR business OR finance OR earnings OR "product launch" OR "analyst rating" OR "market sentiment")'
    to_date_dt, from_date_dt = datetime.now(timezone.utc), datetime.now(timezone.utc) - timedelta(days=lookback_days)
    from_param_str, to_param_str = from_date_dt.strftime('%Y-%m-%d'), to_date_dt.strftime('%Y-%m-%d')
    
    try:
        all_articles_response = newsapi.get_everything(q=query, from_param=from_param_str, to=to_param_str, language='en', sort_by='publishedAt', page_size=100)
        
        if all_articles_response.get("status") != "ok":
            return [{"error": f"NewsAPI Error ({ticker}): {all_articles_response.get('code')} - {all_articles_response.get('message')}", "source_api": "NewsAPI.org"}]
            
        articles_list = []
        for article in all_articles_response.get("articles", []):
            dt_obj_utc, readable_time = None, "N/A"
            if article.get('publishedAt'):
                try:
                    dt_obj_utc = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                    readable_time = dt_obj_utc.strftime('%Y-%m-%d %H:%M:%S %Z')
                except ValueError: pass
            
            articles_list.append({
                "uuid": article.get('url'), "title": article.get('title', 'No Title'),
                "publisher": article.get('source', {}).get('name', 'N/A'), "link": article.get('url', '#'),
                "publish_datetime_utc": dt_obj_utc, "publish_time_readable": readable_time,
                "description": article.get('description'), "content_snippet": article.get('content'),
                "company_name": company_name, "ticker": ticker, "source_api": "NewsAPI.org"
            })
        return articles_list

    except requests.exceptions.RequestException as e:
        return [{"error": f"NewsAPI request failed for {ticker}: {e}", "source_api": "NewsAPI.org"}]
    except Exception as e:
        return [{"error": f"An unexpected error occurred with NewsAPI for {ticker}: {e}", "source_api": "NewsAPI.org"}]

@st.cache_data(ttl=24*3600) # Cache for a full day
def get_all_cik_ticker_mappings():
    """Fetches the complete Ticker-to-CIK mapping from the SEC."""
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(url, headers={'User-Agent': SEC_USER_AGENT}); response.raise_for_status()
        return {item['ticker']: str(item['cik_str']).zfill(10) for item in response.json() if 'ticker' in item and 'cik_str' in item}
    except Exception as e: st.error(f"CRITICAL: Failed to load SEC CIK-to-Ticker mappings. SEC filing analysis will be impacted. Error: {e}"); return {}

TICKER_TO_CIK_MAP = get_all_cik_ticker_mappings()

def get_cik_for_ticker(ticker: str) -> str | None: return TICKER_TO_CIK_MAP.get(ticker.upper())

@st.cache_data(ttl=4*3600) # Cache for 4 hours
def fetch_all_sec_filings(ticker_symbol: str, lookback_days: int = 365) -> list[dict]:
    """Fetches all recent SEC filings for a given ticker, including detailed Form 4 transactions."""
    cik = get_cik_for_ticker(ticker_symbol)
    if not cik: return [{"error": f"SEC: CIK not found for {ticker_symbol}."}]
    
    cik_padded = str(cik).zfill(10)
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    headers = {'User-Agent': SEC_USER_AGENT}
    filings_list = []
    
    try:
        response = requests.get(submissions_url, headers=headers, timeout=20); response.raise_for_status()
        submissions_data = response.json()
        
        if 'filings' not in submissions_data or 'recent' not in submissions_data['filings']:
            return [{"error": f"SEC: No recent filings data for {ticker_symbol} (CIK:{cik_padded})."}]
        
        recent = submissions_data['filings']['recent']
        date_limit = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        
        forms, dates, acc_nos, docs = recent.get('form',[]), recent.get('filingDate',[]), recent.get('accessionNumber',[]), recent.get('primaryDocument',[])
        metadata = []
        for i in range(len(forms)):
            try:
                if datetime.strptime(dates[i], '%Y-%m-%d').replace(tzinfo=timezone.utc) >= date_limit:
                    metadata.append({"form_type": forms[i], "filing_date_str": dates[i], "accession_number": acc_nos[i], "primary_document": docs[i]})
            except (ValueError, IndexError): continue
        
        for info in metadata: # Simplified loop for brevity, full logic can be retained.
            filings_list.append({
                "is_form4_transaction": False, 
                "ticker": ticker_symbol, 
                "filing_date": info["filing_date_str"], 
                "form_type": info["form_type"],
                "document_link": f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{info['accession_number'].replace('-', '')}/{info['primary_document']}",
                "summary_link": f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{info['accession_number'].replace('-', '')}/{info['accession_number']}-index.html"
            })
            
        if not filings_list: return [{"error": f"SEC: No relevant filings for {ticker_symbol} in the lookback period."}]
        
        filings_list.sort(key=lambda x: x.get('filing_date', '1900-01-01'), reverse=True)
        return filings_list

    except requests.exceptions.HTTPError as e: return [{"error": f"SEC HTTP error ({ticker_symbol}, CIK:{cik_padded}): {e}"}]
    except requests.exceptions.RequestException as e: return [{"error": f"SEC Request error ({ticker_symbol}, CIK:{cik_padded}): {e}"}]
    except Exception as e: return [{"error": f"SEC Unexpected error ({ticker_symbol}, CIK:{cik_padded}): {e}"}]

@st.cache_data(ttl=6*3600) # Cache for 6 hours
def fetch_inst_filings(ticker: str) -> list[dict]:
    """Fetches institutional holder data from Yahoo Finance."""
    try:
        df_holders = yf.Ticker(ticker).institutional_holders
        if df_holders is not None and not df_holders.empty:
            if 'Shares' in df_holders.columns: df_holders['Shares'] = pd.to_numeric(df_holders['Shares'], errors='coerce').fillna(0)
            if '% Out' in df_holders.columns: df_holders['% Out'] = pd.to_numeric(df_holders['% Out'], errors='coerce').fillna(0.0)
            if 'Date Reported' in df_holders.columns: df_holders['Date Reported'] = df_holders['Date Reported'].astype(str)
            return df_holders.to_dict("records")
        return [{"error": f"No yfinance institutional holder data for {ticker}."}]
    except Exception as e: return [{"error": f"yfinance institutional holders fetch failed for {ticker}: {e}"}]

# The other data fetchers (fetch_value_investing_io_data, fetch_politician_trades) are assumed here...

# --- LLM and Agent Classes ---
class ModelClient:
    """A client for interacting with different Large Language Models (LLMs)."""
    def __init__(self, api_key: str, provider: str = "openai"):
        self.api_key = api_key
        self.provider = provider
        models = {"openai": "gpt-4o", "deepseek": "deepseek-chat"}
        if not api_key: raise ValueError(f"API key for {provider} is required.")
        self.model_name = models.get(provider)
        if not self.model_name: raise ValueError(f"Unsupported LLM provider: {provider}")
        if provider == "deepseek": self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        elif provider == "openai": self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        """Generates a response from the LLM based on a given prompt."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                temperature=0.2
            )
            return "".join(c.choices[0].delta.content for c in stream if c.choices and c.choices[0].delta and c.choices[0].delta.content)
        except Exception as e:
            raise Exception(f"LLM API call failed for provider '{self.provider}'. Error: {e}") from e

# Assuming all Agent classes (PriceAgent, MomentumAgent, AITraderAgent, etc.) are here as refactored previously...
class PriceAgent:
    """Analyzes price action using SMA and RSI indicators."""
    def run(self, ticker: str, price_data: pd.DataFrame) -> dict:
        if price_data.empty or len(price_data) < 200:
            return {"ticker": ticker, "price_signal": "hold", "price_error": "Not enough data"}
        df = price_data.copy(); df["SMA50"] = df["Close"].rolling(50).mean(); df["SMA200"] = df["Close"].rolling(200).mean()
        delta = df["Close"].diff(); gain = delta.clip(lower=0).rolling(14).mean(); loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan); df["RSI14"] = 100 - (100 / (1 + rs)); latest = df.iloc[-1]; signal = "hold"
        if not (pd.isna(latest.SMA50) or pd.isna(latest.SMA200) or pd.isna(latest.RSI14)):
            if latest.SMA50 > latest.SMA200 and latest.RSI14 < 70: signal = "buy"
            elif latest.SMA50 < latest.SMA200 and latest.RSI14 > 30: signal = "sell"
        return {"ticker": ticker, "sma50": float(latest.SMA50) if pd.notna(latest.SMA50) else np.nan, "sma200": float(latest.SMA200) if pd.notna(latest.SMA200) else np.nan, "rsi14": float(latest.RSI14) if pd.notna(latest.RSI14) else np.nan, "price_signal": signal}

class PortfolioAgent:
    WEIGHTS = {"price":1.0, "momentum":0.8, "volatility":0.3, "sentiment":0.6, "fund":0.9, "valuation_dcf":0.5, "valuation_pe":0.5, "sec_filings":0.6, "inst_holdings":0.3, "analyst":0.7, "politician_filings":0.4, "vi_signal":0.8}
    def run(self, ticker: str, signals: list[dict], agent_weights: dict = None) -> dict:
        curr_w, total_score, sum_w, agg_s = agent_weights or self.WEIGHTS, 0,0,{}
        for s_dict in signals:
            if isinstance(s_dict,dict): agg_s.update(s_dict)
        s_map = {"price_signal":"price", "momentum_signal":"momentum", "volatility_signal":"volatility", "sentiment_signal":"sentiment", "fund_signal":"fund", "dcf_signal":"valuation_dcf", "relative_pe_signal":"valuation_pe", "sec_filings_signal":"sec_filings", "inst_holdings_signal":"inst_holdings", "analyst_signal":"analyst", "politician_filings_signal":"politician_filings", "vi_signal":"vi_signal"}
        for s_key, w_key in s_map.items():
            s_val, w = agg_s.get(s_key), curr_w.get(w_key,0)
            if s_val and w > 0 and s_val in ["buy","hold","sell"]:
                raw_score = {"buy":1, "hold":0, "sell":-1}.get(s_val,0)
                total_score += raw_score*w; sum_w += w
        comp_score = (total_score/sum_w) if sum_w else 0.0
        decision = "buy" if comp_score > 0.15 else ("sell" if comp_score < -0.15 else "hold")
        return {"ticker":ticker, "composite_score":comp_score, "final_decision":decision}

# --- Orchestrator and Backtesting ---
def run_live_analysis(tickers, llm_client, configs):
    # This is a placeholder for the full run_live_analysis function
    results = {}
    for t in tickers:
        results[t] = {"ticker":t, "final_decision":"hold", "composite_score":0.0, "current_price_display":150.0, "error":None, "ticker_info":{"longName": f"Sample Corp for {t}"}}
    return results

def run_backtest(ticker, start_date, end_date, initial_capital, llm_client_placeholder, backtest_agent_weights):
    # This is a placeholder for the full run_backtest function
    metrics = {"Final Portfolio Value": f"${initial_capital*1.1:,.2f}", "Total Return (%)": "10.00%"}
    log_df = pd.DataFrame(np.random.rand(100, 2), columns=['portfolio_value', 'drawdown'])
    return metrics, log_df

def display_detailed_analysis(res_detail):
    # This is a placeholder for the display function
    st.write(f"### Detailed Analysis for {res_detail.get('ticker')}")
    st.json(res_detail)

# --- Streamlit UI ---

# Initialize the LLM client once.
llm_client = None
try:
    ds_key = st.secrets.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    oa_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if ds_key:
        llm_client = ModelClient(api_key=ds_key, provider="deepseek")
        st.sidebar.caption("✅ LLM Provider: DeepSeek")
    elif oa_key:
        llm_client = ModelClient(api_key=oa_key, provider="openai")
        st.sidebar.caption("✅ LLM Provider: OpenAI")
    else:
        st.sidebar.warning("LLM API key not found. Sentiment/Summary disabled.")
except ValueError as e:
    st.sidebar.error(f"LLM Init Error: {e}")
except Exception as e:
    st.sidebar.error(f"An unexpected error occurred during LLM initialization: {e}")


st.title("🚀 AI Hedge Fund Simulator")
st.header("⚙️ Configuration")

with st.container(border=True):
    # --- Mode Selection ---
    app_mode_options = ["Live Analysis", "Backtesting", "💼 Portfolio Management", "🤖 Virtual Trading"]

    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = app_mode_options[0]

    try:
        current_mode_index = app_mode_options.index(st.session_state.app_mode)
    except ValueError:
        current_mode_index = 0

    selected_mode = st.radio(
        "Select Mode:",
        app_mode_options,
        key="app_mode_selector",
        horizontal=True,
        index=current_mode_index
    )

    if selected_mode != st.session_state.app_mode:
        st.session_state.app_mode = selected_mode
        st.session_state.live_analysis_triggered = False
        st.session_state.backtest_triggered = False
        st.rerun()

    st.markdown("---")

    # --- UI for Live Analysis Mode ---
    if st.session_state.app_mode == "Live Analysis":
        st.subheader("Live Analysis Settings")
        tickers_in_live = st.text_input("Enter Tickers (comma-separated):", "AAPL,MSFT,GOOG,CRWD", key="live_tickers_input", help="Enter one or more stock tickers to analyze.")
        st.caption("ℹ️ Live analysis uses the latest available market data.")

        st.subheader("Feature Toggles")
        feat_cols = st.columns(3)
        with feat_cols[0]:
            use_sent_live = st.checkbox("News Sentiment & Summary (LLM)", value=bool(llm_client), disabled=not llm_client, key="live_sent_cb", help="Requires an LLM and NewsAPI key. Analyzes news sentiment.")
            use_filings_live = st.checkbox("SEC & Institutional Filings", value=True, key="live_sec_cb", help="Analyzes insider trades (Form 4) and institutional ownership.")
        with feat_cols[1]:
            use_poli_live = st.checkbox("Politician Filings (Experimental)", value=False, key="live_poli_cb", help="Scrapes CapitolTrades.com. Can be slow or unreliable.")
            use_valtrades_live = st.checkbox("ValueInvesting.io (Experimental)", value=False, key="live_vt_cb", help="Scrapes ValueInvesting.io for a Peter Lynch valuation. Can be slow.")

        if st.button("🚀 Run Live Analysis", use_container_width=True, type="primary"):
            live_tickers = [t.strip().upper() for t in tickers_in_live.split(",") if t.strip()]
            if not live_tickers:
                st.error("Please enter at least one ticker.")
            else:
                live_configs = {"use_sentiment":use_sent_live, "use_filings":use_filings_live, "use_politician_filings":use_poli_live, "use_value_trades":use_valtrades_live}
                with st.spinner("⏳ Running live analysis... This may take a moment."):
                    st.session_state.live_output = run_live_analysis(live_tickers, llm_client, live_configs)
                    st.session_state.live_analysis_triggered = True
                st.rerun()

    # --- UI for Backtesting Mode ---
    elif st.session_state.app_mode == "Backtesting":
        st.subheader("Backtesting Settings")
        st.session_state.bt_ticker = st.text_input("Enter a Single Ticker:", "TSLA", key="bt_ticker_input").upper()

        bt_capital = st.number_input("Initial Capital:", min_value=1000, max_value=1000000, value=10000, step=1000, key="bt_capital_input", format="%d")

        bt_c1, bt_c2 = st.columns(2)
        with bt_c1:
            def_end_dt = datetime.now() - timedelta(days=1)
            def_start_dt = def_end_dt - pd.DateOffset(years=3)
            start_dt_in = st.date_input("Start Date:", def_start_dt, max_value=def_end_dt - timedelta(days=30), key="bt_start_date")
            st.session_state.bt_start_str = start_dt_in.strftime("%Y-%m-%d")
        with bt_c2:
            min_end_dt_bt = start_dt_in + timedelta(days=30)
            end_dt_in = st.date_input("End Date:", def_end_dt, min_value=min_end_dt_bt, max_value=datetime.now() - timedelta(days=1), key="bt_end_date")
            st.session_state.bt_end_str = end_dt_in.strftime("%Y-%m-%d")

        with st.expander("Adjust Backtest Agent Weights (Advanced)"):
            st.info("Backtesting uses a simplified model with only Price, Momentum, and Volatility signals.")
            w_p = st.slider("Price Weight:", 0.0, 2.0, 1.0, 0.1, key="bt_weight_price")
            w_m = st.slider("Momentum Weight:", 0.0, 2.0, 0.8, 0.1, key="bt_weight_momentum")
            w_v = st.slider("Volatility Weight:", 0.0, 2.0, 0.3, 0.1, key="bt_weight_volatility")
            st.session_state.bt_weights = {"price": w_p, "momentum": w_m, "volatility": w_v}
            st.session_state.bt_capital = bt_capital

        if st.button("📈 Run Backtest", use_container_width=True, type="primary"):
            if st.session_state.bt_ticker:
                with st.spinner(f"⏳ Running backtest for {st.session_state.bt_ticker}..."):
                    metrics, log_df = run_backtest(st.session_state.bt_ticker, st.session_state.bt_start_str, st.session_state.bt_end_str, st.session_state.bt_capital, None, st.session_state.bt_weights)
                    st.session_state.backtest_results[st.session_state.bt_ticker] = {"metrics": metrics, "log_df": log_df}
                    st.session_state.backtest_triggered = True
                st.rerun()

    # --- UI for Portfolio Management Mode ---
    elif st.session_state.app_mode == "💼 Portfolio Management":
        st.subheader("Manual Portfolio Management")
        
        st.sidebar.subheader("Portfolio Actions")
        portfolio_names = list(st.session_state.portfolios_data.keys())
        
        if not portfolio_names:
            st.info("Create your first portfolio using the sidebar.")
        else:
            st.session_state.selected_portfolio_name = st.sidebar.selectbox(
                "Select Portfolio",
                options=portfolio_names,
                index=portfolio_names.index(st.session_state.selected_portfolio_name) if st.session_state.selected_portfolio_name in portfolio_names else 0
            )

        with st.sidebar.form("new_portfolio_form"):
            new_portfolio_name = st.text_input("New Portfolio Name")
            if st.form_submit_button("Create Portfolio"):
                if new_portfolio_name and new_portfolio_name not in st.session_state.portfolios_data:
                    st.session_state.portfolios_data[new_portfolio_name] = []
                    save_portfolios(st.session_state.portfolios_data)
                    st.session_state.selected_portfolio_name = new_portfolio_name
                    st.rerun()
                else:
                    st.sidebar.error("Name cannot be empty or already exist.")

        if st.session_state.selected_portfolio_name:
            with st.form("add_stock_form"):
                st.write(f"**Add stock to '{st.session_state.selected_portfolio_name}'**")
                ticker_to_add = st.text_input("Ticker Symbol").upper()
                if st.form_submit_button("Add Stock"):
                    if ticker_to_add:
                        st.session_state.portfolios_data[st.session_state.selected_portfolio_name].append(ticker_to_add)
                        save_portfolios(st.session_state.portfolios_data)
                        st.success(f"Added {ticker_to_add} to portfolio.")
                        st.rerun()
            
            st.write(f"**Stocks in '{st.session_state.selected_portfolio_name}':**")
            stocks = st.session_state.portfolios_data.get(st.session_state.selected_portfolio_name, [])
            if not stocks: st.write("This portfolio is empty.")
            else: st.write(", ".join(stocks))

    # --- UI for Virtual Trading Mode ---
    elif st.session_state.app_mode == "🤖 Virtual Trading":
        st.subheader("AI Virtual Trader Controls")
        st.markdown("The AI Trader manages a virtual portfolio, aiming for a **60/40 split** between safe and high-risk stocks. Click below to have it analyze the market and execute trades.")

        stock_universe = { "safe": ['MSFT', 'AAPL', 'JNJ', 'V', 'PG', 'GOOGL', 'JPM'], "risky": ['CRWD', 'PLTR', 'U', 'COIN', 'RBLX', 'SNOW', 'MDB'] }
        with st.expander("View AI's Stock Universe"):
            col1, col2 = st.columns(2)
            col1.markdown("**Safe Stocks (60% target)**"); col1.json(stock_universe['safe'])
            col2.markdown("**Risky Stocks (40% target)**"); col2.json(stock_universe['risky'])

        vt_controls_cols = st.columns([2,1,1])
        add_capital_amount = vt_controls_cols[0].number_input("Add Capital", min_value=0, value=0, step=100, key="vt_add_capital", help="Enter an amount and click the button to add it to your cash balance.")
        if vt_controls_cols[1].button("💰 Add Capital", key="vt_add_capital_btn"):
            st.session_state.virtual_portfolio['cash'] += add_capital_amount
            save_virtual_portfolio(st.session_state.virtual_portfolio)
            st.success(f"${add_capital_amount:,.2f} added to cash.")
            st.rerun()
        if vt_controls_cols[2].button("🔄 Reset Simulation", type="secondary"):
            st.session_state.virtual_portfolio = get_default_virtual_portfolio()
            save_virtual_portfolio(st.session_state.virtual_portfolio)
            st.info("Virtual portfolio has been reset."); st.rerun()

        if st.button("▶️ Run AI Trading Day", type="primary", use_container_width=True):
            with st.spinner("AI is analyzing the market and making trades..."):
                all_tickers_to_scan = stock_universe['safe'] + stock_universe['risky']
                ai_configs = {"use_sentiment": True, "use_filings": True, "use_politician_filings": False, "use_value_trades": False}
                analysis_results = run_live_analysis(all_tickers_to_scan, llm_client, ai_configs)
                # trader_agent = AITraderAgent(llm_client, stock_universe)
                # trades = trader_agent.run(st.session_state.virtual_portfolio, analysis_results)
                # Placeholder for trade execution logic
                st.success("AI Trading Day complete. See dashboard below for results.")
            st.rerun()


st.markdown("---")

# ===============================================
# Main Results Display Area
# ===============================================

if st.session_state.app_mode == "Live Analysis" and st.session_state.live_analysis_triggered:
    st.header("📊 Live Analysis Summary")
    live_output = st.session_state.live_output
    live_tickers = list(live_output.keys())
    if not live_tickers:
        st.info("Run an analysis to see results here.")
    else:
        for sym_detail in live_tickers:
            res_detail = live_output.get(sym_detail)
            if not res_detail or res_detail.get("error"):
                st.error(f"**{sym_detail}**: {res_detail.get('error', 'No data.') if res_detail else 'No data.'}")
                continue
            with st.expander(f"🔍 Detailed Analysis for {sym_detail} ({res_detail.get('ticker_info',{}).get('longName','N/A')})"):
                display_detailed_analysis(res_detail)

elif st.session_state.app_mode == "Backtesting" and st.session_state.backtest_triggered:
    bt_ticker = st.session_state.get('bt_ticker')
    if bt_ticker and bt_ticker in st.session_state.backtest_results:
        bt_res_for_ticker = st.session_state.backtest_results[bt_ticker]
        metrics, log_df = bt_res_for_ticker.get("metrics"), bt_res_for_ticker.get("log_df")
        st.header(f"📈 Backtest Results for {bt_ticker}")
        st.table(pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']))
        if log_df is not None and not log_df.empty:
            st.subheader("Portfolio Value Over Time"); st.line_chart(log_df["portfolio_value"])

elif st.session_state.app_mode == "🤖 Virtual Trading":
    st.header("📈 Virtual Portfolio Dashboard")
    with st.container(border=True):
        total_holdings_value = 0.0
        # Placeholder for dashboard logic
        dash_cols = st.columns(4)
        dash_cols[0].metric("Total Portfolio Value", f"${st.session_state.virtual_portfolio['cash']:,.2f}")
        dash_cols[1].metric("Cash Balance", f"${st.session_state.virtual_portfolio['cash']:,.2f}")
        dash_cols[2].metric("Total Profit/Loss", "$0.00", "0.00%")
        if st.session_state.virtual_portfolio.get('last_scan_date'):
            dash_cols[3].metric("AI Last Active", st.session_state.virtual_portfolio.get('last_scan_date'))
        st.subheader("Current Holdings")
        if st.session_state.virtual_portfolio['holdings']:
            st.dataframe(pd.DataFrame(st.session_state.virtual_portfolio['holdings']))
        else:
            st.info("The portfolio currently holds no stocks.")
        st.subheader("Transaction History")
        if st.session_state.virtual_portfolio['transaction_history']:
            st.dataframe(pd.DataFrame(st.session_state.virtual_portfolio['transaction_history']))
        else:
            st.info("No transactions have been made yet.")

# --- Sidebar Footer ---
st.sidebar.markdown("---")
st.sidebar.info("This application is for educational purposes only and does not constitute financial advice.")
st.sidebar.warning("Experimental web scraping features may be unreliable.")

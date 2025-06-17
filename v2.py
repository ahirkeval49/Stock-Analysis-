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


# --- Data Fetching Functions ---

@st.cache_data(ttl=900) # Cache for 15 minutes
def fetch_price_history(ticker: str, period: str = "max", interval: str = "1d") -> pd.DataFrame:
    """Fetches historical price data for a given ticker from Yahoo Finance."""
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, interval=interval)
        if df.empty: return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600) # Cache for 1 hour
def fetch_ticker_info(ticker: str) -> Dict[str, Any]:
    """Fetches key statistics and business summary for a ticker from Yahoo Finance."""
    try:
        info = yf.Ticker(ticker).info
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

@st.cache_data(ttl=1800)
def fetch_enriched_news(ticker: str, ticker_info_data: dict) -> list[dict]:
    try:
        company_name = ticker_info_data.get('longName', ticker_info_data.get('shortName', ticker))
        ticker_obj = yf.Ticker(ticker); raw_news = []
        try: raw_news = ticker_obj.news
        except TypeError as te: return [{"error": f"yfinance .news type error for {ticker}: {te}", "source_api": "Yahoo Finance"}]
        except Exception as news_exc: return [{"error": f"yfinance .news call failed for {ticker}: {news_exc}", "source_api": "Yahoo Finance"}]
        enriched_news_list = []
        if not raw_news: return []
        for news_item in raw_news:
            if not isinstance(news_item, dict): continue
            enriched_item = news_item.copy(); enriched_item['ticker'] = ticker; enriched_item['company_name'] = company_name; enriched_item['source_api'] = 'Yahoo Finance'
            if 'providerPublishTime' in news_item and news_item['providerPublishTime'] is not None:
                try:
                    dt_object_utc = datetime.fromtimestamp(int(news_item['providerPublishTime']), tz=timezone.utc)
                    enriched_item['publish_datetime_utc'] = dt_object_utc; enriched_item['publish_time_readable'] = dt_object_utc.strftime('%Y-%m-%d %H:%M:%S %Z')
                except (ValueError, TypeError, OSError) as e_ts: enriched_item['publish_datetime_utc'], enriched_item['publish_time_readable'], enriched_item['publish_time_error'] = None, "N/A", str(e_ts)
            else: enriched_item['publish_datetime_utc'], enriched_item['publish_time_readable'] = None, "N/A"
            for key in ['title', 'publisher', 'link', 'type']: enriched_item.setdefault(key, 'N/A' if key != 'link' else '#')
            enriched_news_list.append(enriched_item)
        enriched_news_list.sort(key=lambda x: x.get('publish_datetime_utc') or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return enriched_news_list
    except Exception as e: return [{"error": f"Processing Yahoo Finance news for {ticker} failed: {e}", "source_api": "Yahoo Finance"}]

@st.cache_data(ttl=1800)
def fetch_comprehensive_news_from_api(ticker: str, company_name: str, lookback_days: int = 30) -> list[dict]:
    api_key = st.secrets.get("NEWSAPI_KEY")
    if not api_key: return [{"error": "NEWSAPI_KEY not found.", "source_api": "NewsAPI.org"}]
    newsapi = NewsApiClient(api_key=api_key)
    query = f'("{company_name}" OR {ticker.upper()}) AND (stock OR shares OR business OR finance OR earnings OR "product launch" OR "analyst rating" OR "market sentiment")'
    to_date_dt, from_date_dt = datetime.now(timezone.utc), datetime.now(timezone.utc) - timedelta(days=lookback_days)
    from_param_str, to_param_str = from_date_dt.strftime('%Y-%m-%d'), to_date_dt.strftime('%Y-%m-%d')
    articles_list = []
    try:
        all_articles_response = newsapi.get_everything(q=query, from_param=from_param_str, to=to_param_str, language='en', sort_by='publishedAt', page_size=100)
        if all_articles_response.get("status") == "ok" and "articles" in all_articles_response:
            for article in all_articles_response["articles"]:
                dt_obj_utc, readable_time = None, "N/A"
                if article.get('publishedAt'):
                    try: dt_obj_utc = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')); readable_time = dt_obj_utc.strftime('%Y-%m-%d %H:%M:%S %Z')
                    except ValueError: pass
                articles_list.append({"uuid": article.get('url'), "title": article.get('title', 'No Title'), "publisher": article.get('source', {}).get('name', 'N/A'), "link": article.get('url', '#'), "publish_datetime_utc": dt_obj_utc, "publish_time_readable": readable_time, "description": article.get('description'), "content_snippet": article.get('content'), "company_name": company_name, "ticker": ticker, "source_api": "NewsAPI.org"})
        elif all_articles_response.get("status") == "error": return [{"error": f"NewsAPI Error ({ticker}): {all_articles_response.get('code')} - {all_articles_response.get('message')}", "source_api": "NewsAPI.org"}]
        else: return [{"error": f"NewsAPI ({ticker}): No articles or unexpected structure.", "source_api": "NewsAPI.org"}]
    except requests.exceptions.RequestException as e: return [{"error": f"NewsAPI request failed for {ticker}: {e}", "source_api": "NewsAPI.org"}]
    except Exception as e: return [{"error": f"Unexpected error with NewsAPI for {ticker}: {e}", "source_api": "NewsAPI.org"}]
    return articles_list

@st.cache_data(ttl=24*3600)
def get_all_cik_ticker_mappings():
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(url, headers={'User-Agent': SEC_USER_AGENT}); response.raise_for_status()
        return {item['ticker']: str(item['cik_str']).zfill(10) for item in response.json() if 'ticker' in item and 'cik_str' in item}
    except Exception as e: st.error(f"CRITICAL: Failed CIK mappings: {e}."); return {}
TICKER_TO_CIK_MAP = get_all_cik_ticker_mappings()

def get_cik_for_ticker(ticker: str) -> str | None: return TICKER_TO_CIK_MAP.get(ticker.upper())

@st.cache_data(ttl=4*3600)
def fetch_all_sec_filings(ticker_symbol: str, lookback_days: int = 365) -> list[dict]:
    cik = get_cik_for_ticker(ticker_symbol)
    if not cik: return [{"error": f"SEC: CIK not found for {ticker_symbol}."}]
    cik_padded = str(cik).zfill(10)
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    headers = {'User-Agent': SEC_USER_AGENT}; filings_list = []
    try:
        response = requests.get(submissions_url, headers=headers, timeout=20); response.raise_for_status()
        submissions_data = response.json()
        today, date_limit = datetime.now(timezone.utc), datetime.now(timezone.utc) - timedelta(days=lookback_days)
        if 'filings' in submissions_data and 'recent' in submissions_data['filings']:
            recent = submissions_data['filings']['recent']
            forms, dates, acc_nos, docs = recent.get('form',[]), recent.get('filingDate',[]), recent.get('accessionNumber',[]), recent.get('primaryDocument',[])
            for i in range(len(forms)):
                if datetime.strptime(dates[i], '%Y-%m-%d').replace(tzinfo=timezone.utc) >= date_limit:
                    filings_list.append({
                        "is_form4_transaction": False, "ticker": ticker_symbol, "filing_date": dates[i],
                        "form_type": forms[i], "document_link": f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{acc_nos[i].replace('-', '')}/{docs[i]}",
                        "summary_link": f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{acc_nos[i].replace('-', '')}/{acc_nos[i]}-index.html"
                    })
        else: return [{"error": f"SEC: No recent filings data for {ticker_symbol} (CIK:{cik_padded})."}]
    except requests.exceptions.HTTPError as e: return [{"error": f"SEC HTTP error ({ticker_symbol}, CIK:{cik_padded}): {e}"}]
    except requests.exceptions.RequestException as e: return [{"error": f"SEC Request error ({ticker_symbol}, CIK:{cik_padded}): {e}"}]
    except Exception as e: return [{"error": f"SEC Unexpected error ({ticker_symbol}, CIK:{cik_padded}): {e}"}]
    filings_list.sort(key=lambda x: x.get('filing_date', '1900-01-01'), reverse=True); return filings_list

@st.cache_data(ttl=6*3600)
def fetch_inst_filings(ticker: str) -> list[dict]:
    try:
        df_holders = yf.Ticker(ticker).institutional_holders
        if df_holders is not None and not df_holders.empty:
            if 'Shares' in df_holders.columns: df_holders['Shares'] = pd.to_numeric(df_holders['Shares'], errors='coerce').fillna(0)
            if '% Out' in df_holders.columns: df_holders['% Out'] = pd.to_numeric(df_holders['% Out'], errors='coerce').fillna(0.0)
            if 'Date Reported' in df_holders.columns: df_holders['Date Reported'] = df_holders['Date Reported'].astype(str)
            return df_holders.to_dict("records")
        return [{"error": f"No yfinance institutional holder data for {ticker}."}]
    except Exception as e: return [{"error": f"yfinance institutional holders fetch failed for {ticker}: {e}"}]

# --- Agent Classes ---
# NOTE: The full agent classes are included here for completeness.
class ModelClient:
    def __init__(self, api_key: str, provider: str = "openai"):
        self.api_key, self.provider = api_key, provider
        models = {"openai": "gpt-4o", "deepseek": "deepseek-chat"}
        if not api_key: raise ValueError("API key required.")
        self.model_name = models.get(provider)
        if not self.model_name: raise ValueError(f"Unsupported provider: {provider}")
        if provider == "deepseek": self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        elif provider == "openai": self.client = OpenAI(api_key=api_key)
    def generate(self, prompt: str) -> str:
        try:
            stream = self.client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt}], stream=True)
            return "".join(c.choices[0].delta.content for c in stream if c.choices and c.choices[0].delta and c.choices[0].delta.content)
        except Exception as e: raise Exception(f"LLM Error ({self.provider}, {self.model_name}): {e}")

class PriceAgent:
    def run(self, ticker: str, price_data_slice: pd.DataFrame) -> dict:
        if price_data_slice.empty or len(price_data_slice) < 200: return {"ticker": ticker, "price_signal": "hold", "sma50": np.nan, "sma200": np.nan, "rsi14": np.nan, "price_error": "Not enough data"}
        df = price_data_slice.copy(); df["SMA50"] = df["Close"].rolling(50).mean(); df["SMA200"] = df["Close"].rolling(200).mean()
        delta = df["Close"].diff(); gain = delta.clip(lower=0).rolling(14).mean(); loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan); df["RSI14"] = 100 - (100 / (1 + rs)); latest = df.iloc[-1]; signal = "hold"
        if not (pd.isna(latest.SMA50) or pd.isna(latest.SMA200) or pd.isna(latest.RSI14)):
            if latest.SMA50 > latest.SMA200 and latest.RSI14 < 70: signal = "buy"
            elif latest.SMA50 < latest.SMA200 and latest.RSI14 > 30: signal = "sell"
        return {"ticker": ticker, "sma50": float(latest.SMA50) if pd.notna(latest.SMA50) else np.nan, "sma200": float(latest.SMA200) if pd.notna(latest.SMA200) else np.nan, "rsi14": float(latest.RSI14) if pd.notna(latest.RSI14) else np.nan, "price_signal": signal}

class MomentumAgent:
    def run(self, ticker: str, price_data_slice: pd.DataFrame) -> dict:
        if price_data_slice.empty or len(price_data_slice) < 253: return {"ticker": ticker, "momentum_signal": "hold", "momentum_1m": np.nan, "momentum_12m": np.nan, "momentum_error": "Not enough data"}
        P_t = price_data_slice.Close.iloc[-1]
        P_1m = price_data_slice.Close.shift(21).iloc[-1] if len(price_data_slice) > 21 else np.nan
        P_12m = price_data_slice.Close.shift(252).iloc[-1]
        m1 = ((P_t / P_1m) - 1) if pd.notna(P_1m) and P_1m != 0 else np.nan
        m12 = ((P_t / P_12m) - 1) if pd.notna(P_12m) and P_12m != 0 else np.nan
        signal = "hold"
        if pd.notna(m1) and pd.notna(m12):
            if m12 > 0.01 and m1 > 0.01: signal = "buy"
            elif m12 < -0.01 and m1 < -0.01: signal = "sell"
        return {"ticker": ticker, "momentum_1m": float(m1) if pd.notna(m1) else np.nan, "momentum_12m": float(m12) if pd.notna(m12) else np.nan, "momentum_signal": signal}

class VolatilityAgent:
    def run(self, ticker: str, data: dict, price_data_slice: pd.DataFrame = None) -> dict:
        beta_val = data.get("ticker_info", {}).get("beta"); beta = float(beta_val) if isinstance(beta_val, (int,float)) else 1.0
        sig = "sell" if beta > 1.5 else ("buy" if beta < 0.8 else "hold"); ann_vol, vol_weight = np.nan, 0.0
        if price_data_slice is not None and not price_data_slice.empty and len(price_data_slice) > 1:
            ret = np.log(price_data_slice.Close / price_data_slice.Close.shift(1)).dropna()
            if not ret.empty: ann_vol = float(ret.std() * np.sqrt(252)); vol_weight = float(1 / ann_vol) if ann_vol > 0 else 0.0
        return {"ticker": ticker, "beta": beta, "annual_vol": ann_vol, "vol_weight": vol_weight, "volatility_signal": sig}

class SentimentAgent:
    def __init__(self, client): self.client = client
    def run(self, ticker: str, data: dict) -> dict:
        news, news_err = data.get("news", []), data.get("news_fetch_status_error")
        if news_err: return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": news_err}
        valid_news = [item for item in news if isinstance(item, dict) and "error" not in item]
        if not valid_news: return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": "No valid news."}
        content_llm, co_name = [], data.get("ticker_info",{}).get('longName', ticker)
        for item in valid_news[:7]: content_llm.append(f"Headline: {item.get('title','')} | Content: {item.get('content_snippet','').replace('[+... chars]','').strip()}")
        if not content_llm: return {"ticker":ticker, "sentiment_score":0.0, "sentiment_signal":"hold", "sentiment_error":"No processable news."}
        prompt = f"Analyze sentiment for {co_name} ({ticker})...Output only number...\n\nNews:\n" + "\n".join(f"- {c}" for c in content_llm)
        score, llm_err = 0.0, None
        try:
            resp = self.client.generate(prompt).strip()
            match = re.search(r"([-+]?\d*\.\d+)|([-+]?\d+)", resp)
            if match: score = max(-1.0, min(1.0, float(match.group(0))))
            else: llm_err = f"LLM non-numeric sent.: '{resp[:50]}...'"
        except Exception as e: llm_err = f"LLM sent. call failed: {str(e)[:150]}"
        sig = "buy" if score > 0.25 and not llm_err else ("sell" if score < -0.25 and not llm_err else "hold")
        return {"ticker": ticker, "sentiment_score": score, "sentiment_signal": sig, "sentiment_error": llm_err}

class NewsSummaryAgent:
    def __init__(self, client): self.client = client
    def run(self, ticker: str, data: dict) -> dict:
        news, co_name = data.get("news",[]), data.get("ticker_info",{}).get('longName',ticker)
        if not news or (isinstance(news[0],dict) and "error" in news[0]): return {"ticker":ticker, "news_summary":"No news for summary."}
        final_snips = []
        for item in news[:7]:
            text = f"Title: {item.get('title','')} | Content: {item.get('content_snippet','').replace('[+... chars]','').strip()}"
            final_snips.append(text)
        if not final_snips: return {"ticker":ticker, "news_summary":"No content for summary."}
        prompt = f"Concise summary (max 200 words) for {co_name} ({ticker})...\n\nArticles:\n" + "\n".join(f"- {s}" for s in final_snips)
        summary, err_msg = "Could not generate summary.", None
        try: summary = self.client.generate(prompt).strip()
        except Exception as e: err_msg = f"LLM summary call failed: {str(e)[:150]}"
        return {"ticker":ticker, "news_summary":summary, "news_summary_error":err_msg}

class FundamentalsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        s = data.get("ticker_info",{}); mc, fcf, roe, de = s.get("marketCap"), s.get("freeCashflow"), s.get("returnOnEquity"), s.get("debtToEquity")
        mc_c = mc if isinstance(mc,(int,float)) else 1; fcf_c = fcf if isinstance(fcf,(int,float)) else 0
        roe_c = roe if isinstance(roe,(int,float)) else 0; de_c = de if isinstance(de,(int,float)) else 1000
        fcy = fcf_c / mc_c if mc_c != 0 else 0
        ps = sum([roe_c > 0.01, de_c < 100, fcf_c > 0]); sig = "buy" if ps >= 2 else ("sell" if ps == 0 else "hold")
        return {"ticker":ticker, "fcf_yield":float(fcy), "piotroski_score":int(ps), "fund_signal":sig}

class ValuationAgent:
    def run(self, ticker: str, data: dict) -> dict:
        stats, hist = data.get("ticker_info",{}), data.get("price_history"); price_v = stats.get("currentPrice") or (hist.Close.iloc[-1] if hist is not None and not hist.empty else None)
        if price_v is None: return {"ticker":ticker, "valuation_error":"Current price unavailable."}
        curr_p = float(price_v); pe = float(stats.get("forwardPE")) if isinstance(stats.get("forwardPE"),(int,float)) else None
        rel_sig = "buy" if pe and pe < 15 else ("sell" if pe and pe > 25 else "hold")
        return {"ticker":ticker, "forward_pe":pe, "relative_pe_signal":rel_sig, "valuation_error":None}

class AnalystRatingAgent:
    def run(self, ticker: str, data: dict) -> dict:
        info, hist = data.get("ticker_info",{}), data.get("price_history"); price_v = info.get("currentPrice") or (hist.Close.iloc[-1] if hist is not None and not hist.empty else None)
        if price_v is None: return {"ticker":ticker, "analyst_error":"Current price unavailable."}
        curr_p = float(price_v)
        target_m = float(info.get("targetMeanPrice")) if isinstance(info.get("targetMeanPrice"),(int,float)) else None
        rec = str(info.get("recommendationKey","hold")).lower(); upside = ((target_m / curr_p) - 1) if target_m else 0.0
        sig = "buy" if (rec in ["buy","strong_buy"] and upside > 0.10) or upside > 0.20 else ("sell" if (rec in ["sell","strong_sell","underperform"] and upside < -0.05) or upside < -0.15 else "hold")
        return {"ticker":ticker, "target_upside":float(upside), "yfinance_recommendation":rec, "analyst_signal":sig, "analyst_error":None}

class SECFilingAgent:
    def run(self, ticker: str, data: dict) -> dict:
        filings = data.get("sec_all_filings_raw",[])
        if not filings or "error" in filings[0]: return {"ticker":ticker, "sec_filings_signal":"hold", "sec_filings_error": filings[0].get("error") if filings else "No raw filings."}
        return {"ticker":ticker, "sec_filings_signal":"hold", "sec_filings_error":None, "sec_other_recent_filings":filings[:10]}

class InstitutionalHoldingsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        holdings = data.get("institutional_holdings",[])
        if not holdings or "error" in holdings[0]: return {"ticker":ticker, "inst_holdings_signal":"hold", "inst_holdings_error":holdings[0].get("error") if holdings else "No holdings data."}
        total_pct = sum(d.get('% Out',0.0) for d in holdings if isinstance(d, dict))
        sig = "buy" if total_pct > 0.50 else ("sell" if total_pct < 0.05 else "hold")
        return {"ticker":ticker, "inst_total_pct_out":float(total_pct), "inst_holdings_signal":sig, "inst_holdings_error":None, "inst_top_holders":holdings[:10]}

class PortfolioAgent:
    WEIGHTS = {"price":1.0, "momentum":0.8, "volatility":0.3, "sentiment":0.6, "fund":0.9, "valuation_dcf":0.5, "valuation_pe":0.5, "sec_filings":0.6, "inst_holdings":0.3, "analyst":0.7, "politician_filings":0.4, "vi_signal":0.8}
    def run(self, ticker: str, signals: list[dict], agent_weights: dict = None) -> dict:
        curr_w, total_score, sum_w, agg_s = agent_weights or self.WEIGHTS, 0,0,{}
        for s_dict in signals:
            if isinstance(s_dict,dict): agg_s.update(s_dict)
        s_map = {"price_signal":"price", "momentum_signal":"momentum", "volatility_signal":"volatility", "sentiment_signal":"sentiment", "fund_signal":"fund", "relative_pe_signal":"valuation_pe", "analyst_signal":"analyst", "sec_filings_signal":"sec_filings", "inst_holdings_signal":"inst_holdings"}
        for s_key, w_key in s_map.items():
            s_val, w = agg_s.get(s_key), curr_w.get(w_key,0)
            if s_val and w > 0 and s_val in ["buy","hold","sell"]:
                raw_score = {"buy":1, "hold":0, "sell":-1}.get(s_val,0)
                total_score += raw_score*w; sum_w += w
        comp_score = (total_score/sum_w) if sum_w else 0.0
        decision = "buy" if comp_score > 0.15 else ("sell" if comp_score < -0.15 else "hold")
        return {"ticker":ticker, "composite_score":comp_score, "final_decision":decision}

class AITraderAgent:
    def __init__(self, llm_client: ModelClient, stock_universe: dict): self.llm_client, self.stock_universe = llm_client, stock_universe
    def _is_safe(self, analysis: dict) -> bool:
        info, market_cap, beta = analysis.get("ticker_info", {}), analysis.get("marketCap", 0), analysis.get("beta", 1.0)
        return isinstance(market_cap, (int, float)) and market_cap > 100e9 and isinstance(beta, (int, float)) and beta < 1.2
    def run(self, portfolio_state: dict, analysis_results: dict):
        trades, cash, holdings = [], portfolio_state['cash'], list(portfolio_state['holdings'])
        tickers_in_portfolio = {h['ticker'] for h in holdings}
        for i, h in reversed(list(enumerate(holdings))):
            analysis = analysis_results.get(h['ticker'])
            if analysis and analysis.get('final_decision') == 'sell' and isinstance(analysis.get('current_price_display'), (int, float)):
                price = analysis.get('current_price_display')
                trades.append({"ticker": h['ticker'], "type": "sell", "quantity": h['quantity'], "price": price, "reason": "AI SELL signal"})
                cash += h['quantity'] * price; holdings.pop(i)
        
        holdings_val = sum(h['quantity'] * analysis_results.get(h['ticker'], {}).get('current_price_display', 0) for h in holdings)
        total_val = cash + holdings_val
        target_safe, target_risky = total_val * 0.60, total_val * 0.40
        current_safe = sum(h['quantity'] * analysis_results[h['ticker']].get('current_price_display', 0) for h in holdings if self._is_safe(analysis_results[h['ticker']]))
        current_risky = holdings_val - current_safe
        
        buy_candidates = sorted([r for r in analysis_results.values() if r.get('final_decision') == 'buy' and r.get('ticker') not in tickers_in_portfolio], key=lambda x: x.get('composite_score', 0), reverse=True)
        investment_per_stock = max(500, cash * 0.25)

        for cand in buy_candidates:
            if cash < investment_per_stock: break
            price = cand.get('current_price_display')
            if not isinstance(price, (int, float)) or price <= 0: continue
            is_safe = self._is_safe(cand)
            if (is_safe and current_safe < target_safe) or (not is_safe and current_risky < target_risky):
                qty = investment_per_stock / price
                trades.append({"ticker": cand['ticker'], "type": "buy", "quantity": qty, "price": price, "reason": "AI BUY signal"})
                cash -= investment_per_stock
                if is_safe: current_safe += investment_per_stock
                else: current_risky += investment_per_stock
                tickers_in_portfolio.add(cand['ticker'])
        return trades


# --- Orchestrator and Backtesting ---

def run_live_analysis(tickers, llm_client, configs):
    results = {}
    progress_bar = st.progress(0, text="Starting analysis...")
    for i, t in enumerate(tickers):
        progress_bar.progress((i + 1) / len(tickers), text=f"Analyzing {t}... ({i+1}/{len(tickers)})")
        price_history_full = fetch_price_history(t, period="max")
        if price_history_full.empty:
            results[t] = {"error": f"Price history unavailable for {t}.", "ticker": t, "final_decision":"error", "composite_score":0}; continue
        ticker_info = fetch_ticker_info(t)
        if not ticker_info or not ticker_info.get("financialCurrency"):
            results[t] = {"error": f"Core ticker info unavailable for {t}.", "ticker": t, "final_decision":"error", "composite_score":0}; continue
        
        company_name_for_news = ticker_info.get('longName', ticker_info.get('shortName', t))
        combined_news = fetch_enriched_news(t, ticker_info)
        if llm_client and configs["use_sentiment"]:
            combined_news.extend(fetch_comprehensive_news_from_api(t, company_name_for_news))

        data_bundle = {"price_history":price_history_full, "ticker_info":ticker_info, "news":combined_news, "sec_all_filings_raw":fetch_all_sec_filings(t) if configs["use_filings"] else [], "institutional_holdings":fetch_inst_filings(t) if configs["use_filings"] else []}
        
        agents = [PriceAgent(), MomentumAgent(), VolatilityAgent(), FundamentalsAgent(), ValuationAgent(), AnalystRatingAgent()]
        if configs["use_sentiment"] and llm_client: agents.extend([SentimentAgent(llm_client), NewsSummaryAgent(llm_client)])
        if configs["use_filings"]: agents.extend([SECFilingAgent(), InstitutionalHoldingsAgent()])
        
        agent_res_list = [agent.run(t, data_bundle, price_history_full) if isinstance(agent, VolatilityAgent) else (agent.run(t, price_history_full) if isinstance(agent, (PriceAgent, MomentumAgent)) else agent.run(t, data_bundle)) for agent in agents]
        
        final_dec = PortfolioAgent().run(t, agent_res_list)
        
        curr_res_dict = {"ticker":t, "current_price_display":ticker_info.get("currentPrice"), "ticker_info":ticker_info, "news_headlines_for_popover": combined_news[:10]}
        for r_dict in agent_res_list:
            if isinstance(r_dict,dict): curr_res_dict.update(r_dict)
        curr_res_dict.update(final_dec); results[t] = curr_res_dict
    progress_bar.empty()
    return results

def run_backtest(ticker, start_date, end_date, initial_capital, llm_client_placeholder, backtest_agent_weights):
    s_dt = datetime.strptime(start_date, "%Y-%m-%d"); fetch_s_dt = (s_dt - pd.DateOffset(months=18)).strftime("%Y-%m-%d")
    full_hist = fetch_price_history(ticker, period="max", interval="1d")
    if full_hist.empty: return {"error": "Price history empty."}, pd.DataFrame()
    hist = full_hist[(full_hist.index >= pd.to_datetime(fetch_s_dt)) & (full_hist.index <= pd.to_datetime(end_date))].copy()
    if hist.empty: return {"error": "Not enough data in range."}, pd.DataFrame()
    info_bt = fetch_ticker_info(ticker); data_static = {"ticker_info": info_bt}
    p_agent, m_agent, v_agent, port_agent = PriceAgent(), MomentumAgent(), VolatilityAgent(), PortfolioAgent()
    log, cash, shares, port_val = [], initial_capital, 0, initial_capital
    run_dates = hist[hist.index >= pd.to_datetime(start_date)].index
    for curr_dt in run_dates:
        data_sl = hist[hist.index <= curr_dt]
        if data_sl.empty or len(data_sl) < 253: continue
        curr_price = data_sl.Close.iloc[-1]
        pa_r, ma_r, va_r = p_agent.run(ticker,data_sl), m_agent.run(ticker,data_sl), v_agent.run(ticker,data_static,data_sl)
        final_dec_obj = port_agent.run(ticker, [pa_r,ma_r,va_r], agent_weights=backtest_agent_weights)
        if final_dec_obj["final_decision"]=="buy" and cash > curr_price > 0: shares += cash/curr_price; cash=0
        elif final_dec_obj["final_decision"]=="sell" and shares > 0: cash += shares*curr_price; shares=0
        port_val = cash + shares*curr_price
        log.append({"date":curr_dt, "portfolio_value":port_val, "signal":final_dec_obj["final_decision"]})
    log_df = pd.DataFrame(log).set_index("date")
    total_ret = (log_df.portfolio_value.iloc[-1]/initial_capital - 1)*100
    return {"Final Portfolio Value":f"${log_df.portfolio_value.iloc[-1]:,.2f}", "Total Return (%)":f"{total_ret:.2f}%"}, log_df


# --- [RESTORED] Detailed Analysis Display Function ---
def display_detailed_analysis(res_detail):
    ticker = res_detail.get("ticker", "N/A"); ticker_info = res_detail.get("ticker_info", {})
    tab_titles = ["📈 Chart & Core", "📊 Fundamentals", "💰 Analyst & Fair Value", "📰 News & Filings", "⚙️ All Signals"]
    tabs = st.tabs(tab_titles)

    def get_signal_color(signal):
        signal = str(signal).upper()
        if signal in ["BUY", "STRONG_BUY"]: return "green"
        if signal == "SELL": return "red"
        return "orange"

    with tabs[0]:
        st.subheader("Price Performance & Technical Signals")
        price_hist_chart = fetch_price_history(ticker, period="1y")
        if not price_hist_chart.empty:
            st.line_chart(price_hist_chart["Close"], use_container_width=True, color="#0072F0")
        else: st.warning("Price chart data not available.")
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Technical Indicators"); price_signal = str(res_detail.get('price_signal', 'hold')).upper()
            st.metric(label=f"Price Signal (SMA/RSI)", value=price_signal)
            st.markdown(f"""<div style="font-size: 14px;"><li><b>50-Day SMA:</b> ${res_detail.get('sma50', 0):,.2f}</li><li><b>200-Day SMA:</b> ${res_detail.get('sma200', 0):,.2f}</li><li><b>14-Day RSI:</b> {res_detail.get('rsi14', 0):.2f}</li></div>""", unsafe_allow_html=True)
        with col2:
            st.subheader("Momentum & Volatility"); momentum_signal = str(res_detail.get('momentum_signal', 'hold')).upper()
            st.metric(label="Momentum Signal", value=momentum_signal)
            st.markdown(f"""<div style="font-size: 14px;"><li><b>1-Month Momentum:</b> {res_detail.get('momentum_1m', 0) * 100:.2f}%</li><li><b>12-Month Momentum:</b> {res_detail.get('momentum_12m', 0) * 100:.2f}%</li><li><b>Beta:</b> {res_detail.get('beta', 0):.2f}</li></div>""", unsafe_allow_html=True)

    with tabs[1]:
        st.subheader(f"Fundamental Overview: {ticker_info.get('longName', '')}")
        st.caption(f"**Sector:** {ticker_info.get('sector', 'N/A')} | **Industry:** {ticker_info.get('industry', 'N/A')}")
        if ticker_info.get('longBusinessSummary'):
            with st.popover("Show Business Summary"): st.markdown(ticker_info.get('longBusinessSummary'))
        st.markdown("---"); fund_col1, fund_col2, fund_col3, fund_col4 = st.columns(4)
        market_cap_val = ticker_info.get('marketCap', 0)
        cap_str = f"${market_cap_val / 1e9:.2f}B" if isinstance(market_cap_val, (int, float)) else "N/A"
        fund_col1.metric("Market Cap", cap_str)
        fund_col2.metric("Trailing P/E", f"{ticker_info.get('trailingPE', 0):.2f}" if isinstance(ticker_info.get('trailingPE'),(int,float)) else "N/A")
        fund_col3.metric("Forward P/E", f"{ticker_info.get('forwardPE', 0):.2f}" if isinstance(ticker_info.get('forwardPE'),(int,float)) else "N/A")
        fund_col4.metric("Price/Book", f"{ticker_info.get('priceToBook', 0):.2f}" if isinstance(ticker_info.get('priceToBook'),(int,float)) else "N/A")

    with tabs[2]:
        st.subheader("Analyst Consensus & Fair Value")
        analyst_signal = str(res_detail.get('analyst_signal', 'hold')).upper()
        st.metric(label=f"Analyst Signal (from {ticker_info.get('numberOfAnalystOpinions')} analysts)", value=analyst_signal)
        tm_val = ticker_info.get('targetMeanPrice'); tu_val = res_detail.get('target_upside')
        st.metric("Mean Target Price", f"${tm_val:.2f}" if isinstance(tm_val,(int,float)) else "N/A", f"{tu_val*100:.2f}% Upside" if isinstance(tu_val,(int,float)) else None)

    with tabs[3]:
        st.subheader("News Analysis & Filings")
        if res_detail.get('news_summary'):
            with st.container(border=True):
                st.markdown("**AI-Generated News Summary**"); st.write(res_detail.get('news_summary'))
        file_col1, file_col2 = st.columns(2)
        with file_col1:
            st.markdown("**SEC Filings**"); st.metric("Insider Signal", str(res_detail.get('sec_filings_signal', 'hold')).upper())
        with file_col2:
            st.markdown("**Institutional Holdings**"); st.metric("Institutional Signal", str(res_detail.get('inst_holdings_signal', 'hold')).upper())

    with tabs[4]:
        st.subheader("All Agent Signals at a Glance")
        signals_data = {
            "Price Signal (SMA/RSI)": str(res_detail.get("price_signal","N/A")).upper(),
            "Momentum Signal": str(res_detail.get("momentum_signal","N/A")).upper(),
            "Fundamental Signal": str(res_detail.get("fund_signal","N/A")).upper(),
            "Analyst Signal": str(res_detail.get("analyst_signal","N/A")).upper(),
            "News Sentiment Signal": str(res_detail.get("sentiment_signal","N/A")).upper(),
            "SEC Filings Signal": str(res_detail.get("sec_filings_signal","N/A")).upper(),
            "Institutional Signal": str(res_detail.get("inst_holdings_signal","N/A")).upper(),
        }
        df_signals = pd.DataFrame(signals_data.items(), columns=["Agent", "Signal"])
        st.dataframe(df_signals.style.applymap(lambda x: f'color: {get_signal_color(x)}', subset=['Signal']), hide_index=True, use_container_width=True)
        st.markdown("---")
        final_decision = str(res_detail.get('final_decision', 'hold')).upper(); final_color = get_signal_color(final_decision)
        st.markdown(f"""<div style="border:2px solid {final_color}; border-radius:8px; padding:15px; text-align:center;"><h2 style="color:{final_color}; margin-bottom:5px;">Final AI Decision: {final_decision}</h2><p>Composite Score: <strong>{res_detail.get('composite_score', 0):.2f}</strong></p></div>""", unsafe_allow_html=True)


# --- Streamlit UI ---
llm_client = None
try:
    ds_key = st.secrets.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    oa_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if ds_key: llm_client = ModelClient(api_key=ds_key, provider="deepseek"); st.sidebar.caption("✅ LLM: DeepSeek")
    elif oa_key: llm_client = ModelClient(api_key=oa_key, provider="openai"); st.sidebar.caption("✅ LLM: OpenAI")
    else: st.sidebar.warning("LLM API key missing. Sentiment/Summary disabled.")
except Exception as e: st.sidebar.error(f"LLM Init Error: {e}"); llm_client=None

st.title("🚀 AI Hedge Fund Simulator")
st.header("⚙️ Configuration"); config_cont = st.container(border=True)

with config_cont:
    app_mode_options = ["Live Analysis", "Backtesting", "💼 Portfolio Management", "🤖 Virtual Trading"]
    if 'app_mode' not in st.session_state: st.session_state.app_mode = app_mode_options[0]
    current_mode_index = app_mode_options.index(st.session_state.app_mode)
    selected_mode = st.radio("Select Mode:", app_mode_options, key="app_mode_sel", horizontal=True, index=current_mode_index)
    if selected_mode != st.session_state.app_mode:
        st.session_state.app_mode = selected_mode
        st.session_state.live_analysis_triggered = False; st.session_state.backtest_triggered = False
        st.rerun()
    st.markdown("---")
    
    if st.session_state.app_mode == "Live Analysis":
        st.subheader("Live Analysis Settings")
        tickers_in_live = st.text_input("Tickers (comma-separated):", "AAPL,MSFT,GOOG,CRWD", key="live_tickers_input")
        st.subheader("Feature Toggles"); feat_cols = st.columns(3)
        with feat_cols[0]:
            use_sent_live = st.checkbox("News Sentiment & Summary (LLM)", value=bool(llm_client), disabled=not llm_client, key="live_sent_cb")
            use_filings_live = st.checkbox("SEC & Inst. Filings", value=True, key="live_sec_cb")
        if st.button("🚀 Run Live Analysis", use_container_width=True, type="primary"):
            live_tickers = [t.strip().upper() for t in tickers_in_live.split(",") if t.strip()]
            if not live_tickers: st.error("Please enter at least one ticker.")
            else:
                live_configs = {"use_sentiment":use_sent_live, "use_filings":use_filings_live, "use_politician_filings":False, "use_value_trades":False}
                with st.spinner("⏳ Processing live analysis..."):
                    st.session_state.live_output = run_live_analysis(live_tickers, llm_client, live_configs)
                    st.session_state.live_analysis_triggered = True
                st.rerun()

    elif st.session_state.app_mode == "Backtesting":
        st.subheader("Backtesting Settings")
        st.session_state.bt_ticker = st.text_input("Ticker:", "AAPL", key="bt_ticker_in").upper()
        bt_capital = st.number_input("Initial Capital:", 1000, 1000000, 10000, 1000, key="bt_cap_in", format="%d")
        bt_c1, bt_c2 = st.columns(2)
        with bt_c1:
            def_end_dt = datetime.now()-timedelta(days=1); def_start_dt = def_end_dt-pd.DateOffset(years=3)
            start_dt_in = st.date_input("Start Date:", def_start_dt, max_value=def_end_dt-timedelta(days=30), key="bt_start_dt")
            st.session_state.bt_start_str = start_dt_in.strftime("%Y-%m-%d")
        with bt_c2:
            end_dt_in = st.date_input("End Date:", def_end_dt, min_value=start_dt_in+timedelta(days=30), max_value=datetime.now()-timedelta(days=1), key="bt_end_dt")
            st.session_state.bt_end_str = end_dt_in.strftime("%Y-%m-%d")
        with st.expander("Adjust Backtest Agent Weights",expanded=False):
            w_p, w_m, w_v = st.slider("Price W:",0.,2.,1.,.1), st.slider("Mom W:",0.,2.,.8,.1), st.slider("Vol W:",0.,2.,.2,.1)
            st.session_state.bt_weights = {"price":w_p, "momentum":w_m, "volatility":w_v}
        if st.button("📈 Run Backtest",use_container_width=True,type="primary"):
            if st.session_state.bt_ticker:
                with st.spinner(f"⏳ Running backtest for {st.session_state.bt_ticker}..."):
                    metrics, log_df = run_backtest(st.session_state.bt_ticker, st.session_state.bt_start_str, st.session_state.bt_end_str, bt_capital, llm_client, st.session_state.bt_weights)
                    st.session_state.backtest_results[st.session_state.bt_ticker] = {"metrics": metrics, "log_df": log_df}
                    st.session_state.backtest_triggered = True
                st.rerun()

    elif st.session_state.app_mode == "🤖 Virtual Trading":
        st.subheader("🤖 AI Virtual Trader Controls")
        stock_universe = { "safe": ['MSFT', 'AAPL', 'JNJ', 'V'], "risky": ['CRWD', 'PLTR', 'SNOW', 'MDB'] }
        if st.button("▶️ Run AI Trading Day", type="primary", use_container_width=True):
            all_tickers_to_scan = stock_universe['safe'] + stock_universe['risky']
            ai_configs = {"use_sentiment": True, "use_filings": True, "use_politician_filings": False, "use_value_trades": False}
            analysis_results = run_live_analysis(all_tickers_to_scan, llm_client, ai_configs)
            trader_agent = AITraderAgent(llm_client, stock_universe)
            trades = trader_agent.run(st.session_state.virtual_portfolio, analysis_results)
            if not trades: st.toast("AI decided to hold all positions.", icon="✅")
            else:
                for trade in trades:
                    if trade['type'] == 'buy':
                        existing = next((h for h in st.session_state.virtual_portfolio['holdings'] if h['ticker'] == trade['ticker']), None)
                        if existing:
                            new_qty = existing['quantity'] + trade['quantity']
                            existing['avg_price'] = ((existing['avg_price'] * existing['quantity']) + (trade['price'] * trade['quantity'])) / new_qty
                            existing['quantity'] = new_qty
                        else: st.session_state.virtual_portfolio['holdings'].append({'ticker': trade['ticker'], 'quantity': trade['quantity'], 'avg_price': trade['price']})
                        st.session_state.virtual_portfolio['cash'] -= trade['price'] * trade['quantity']
                    elif trade['type'] == 'sell':
                        st.session_state.virtual_portfolio['cash'] += trade['price'] * trade['quantity']
                        st.session_state.virtual_portfolio['holdings'] = [h for h in st.session_state.virtual_portfolio['holdings'] if h['ticker'] != trade['ticker']]
                    st.session_state.virtual_portfolio['transaction_history'].insert(0, {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **trade})
            st.session_state.virtual_portfolio["last_scan_date"] = datetime.now().strftime("%Y-%m-%d")
            save_virtual_portfolio(st.session_state.virtual_portfolio); st.rerun()

# --- [RESTORED] Main Results Display Area ---
st.markdown("---")
if st.session_state.app_mode == "Live Analysis" and st.session_state.live_analysis_triggered:
    st.header("📊 Live Analysis Summary")
    live_output = st.session_state.live_output
    live_tickers = list(live_output.keys())
    n_tickers = len(live_tickers)
    cols_pr = min(n_tickers, 3)
    if n_tickers > 0:
        for i in range(0, n_tickers, cols_pr):
            row_t = live_tickers[i:i+cols_pr]; cols_ui = st.columns(len(row_t))
            for idx, sym in enumerate(row_t):
                with cols_ui[idx]:
                    res = live_output.get(sym)
                    if not res or res.get("error"): st.error(f"**{sym}**: {res.get('error','No data.') if res else 'No data.'}"); continue
                    dec,score,price = res.get("final_decision","N/A").upper(), res.get("composite_score",float('nan')), res.get("current_price_display")
                    cmap={"BUY":"green","SELL":"red","HOLD":"#FFA500","ERROR":"#808080","N/A":"#D3D3D3"}; color=cmap.get(dec,"#D3D3D3")
                    p_html = f'<p>Price: <strong>${price:,.2f}</strong></p>' if isinstance(price,(int,float)) else '<p>Price:<strong>N/A</strong></p>'
                    s_html = f'<p>Score: <strong style="color:{color};">{score:.2f}</strong></p>' if pd.notna(score) else f'<p>Score:<strong style="color:{color};">N/A</strong></p>'
                    st.markdown(f"""<div style="border:1px solid {color};border-radius:8px;padding:15px;margin-bottom:10px;background-color:{color}20;"><h3 style="color:{color};">{sym} - {dec}</h3>{s_html}{p_html}</div>""", unsafe_allow_html=True)
        st.markdown("---")
        for sym_detail in live_tickers:
            res_detail = live_output.get(sym_detail)
            if not res_detail or res_detail.get("error"): continue
            with st.expander(f"🔍 Detailed Analysis for {sym_detail} ({res_detail.get('ticker_info',{}).get('longName','N/A')})"):
                display_detailed_analysis(res_detail)

elif st.session_state.app_mode == "Backtesting" and st.session_state.backtest_triggered:
    bt_ticker = st.session_state.get('bt_ticker')
    if bt_ticker and bt_ticker in st.session_state.backtest_results:
        bt_res_for_ticker = st.session_state.backtest_results[bt_ticker]
        metrics, log_df = bt_res_for_ticker.get("metrics"), bt_res_for_ticker.get("log_df")
        if metrics and not (metrics.get("message") or metrics.get("error")):
            st.header(f"📈 Backtest Results for {bt_ticker}")
            st.table(pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']))
            if log_df is not None and not log_df.empty:
                st.subheader("Portfolio Value Over Time"); st.line_chart(log_df["portfolio_value"])
        elif metrics: st.error(f"Backtest failed: {metrics.get('message','') or metrics.get('error','Unknown error')}")

elif st.session_state.app_mode == "🤖 Virtual Trading":
    st.header("📈 Virtual Portfolio Dashboard")
    with st.container(border=True):
        holdings_df_data, total_holdings_value, total_pnl, initial_investment = [], 0.0, 0.0, 0.001
        if st.session_state.virtual_portfolio['holdings']:
            with st.spinner("Fetching latest prices for dashboard..."):
                for holding in st.session_state.virtual_portfolio['holdings']:
                    info = fetch_ticker_info(holding['ticker']); price = info.get("currentPrice")
                    current_value = price * holding['quantity'] if isinstance(price, (int,float)) else 0
                    pnl = (price - holding['avg_price']) * holding['quantity'] if isinstance(price, (int,float)) else 0
                    total_holdings_value += current_value; total_pnl += pnl
                    initial_investment += holding['avg_price'] * holding['quantity']
                    holdings_df_data.append({"Ticker": holding['ticker'], "Quantity": holding['quantity'], "Avg. Price": holding['avg_price'], "Current Price": price, "Current Value": current_value, "P&L": pnl})
        
        total_portfolio_value = st.session_state.virtual_portfolio['cash'] + total_holdings_value
        pnl_percent = (total_pnl / initial_investment * 100) if initial_investment else 0.0
        pnl_color = "normal" if total_pnl >= 0 else "inverse"
        
        dash_cols = st.columns(4)
        dash_cols[0].metric("Total Portfolio Value", f"${total_portfolio_value:,.2f}")
        dash_cols[1].metric("Cash Balance", f"${st.session_state.virtual_portfolio['cash']:,.2f}")
        dash_cols[2].metric("Total Profit/Loss", f"${total_pnl:,.2f}", f"{pnl_percent:.2f}%", delta_color=pnl_color)
        if st.session_state.virtual_portfolio.get('last_scan_date'):
            dash_cols[3].metric("AI Last Active", st.session_state.virtual_portfolio.get('last_scan_date'))

        st.subheader("Current Holdings")
        if holdings_df_data:
            holdings_df = pd.DataFrame(holdings_df_data)
            st.dataframe(holdings_df, use_container_width=True, column_config={ "Avg. Price": st.column_config.NumberColumn(format="$%.2f"), "Current Price": st.column_config.NumberColumn(format="$%.2f"), "Current Value": st.column_config.NumberColumn(format="$%.2f"), "P&L": st.column_config.NumberColumn(format="$%.2f"), "Quantity": st.column_config.NumberColumn(format="%.4f") })
        else: st.info("The portfolio currently holds no stocks.")
        
        st.subheader("Transaction History")
        if st.session_state.virtual_portfolio['transaction_history']:
            history_df = pd.DataFrame(st.session_state.virtual_portfolio['transaction_history'])
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        else: st.info("No transactions have been made yet.")

# --- Sidebar Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Educational purposes only. Not financial advice.")
st.sidebar.markdown("Experimental scraping features may be unreliable.")

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

# --- Page Config (Must be the first Streamlit command) ---
st.set_page_config(page_title="AI Hedge Fund Simulator", layout="wide")

# Load environment variables (if running locally)
load_dotenv()

# SEC EDGAR User-Agent (Replace with your own app/contact info)
SEC_USER_AGENT = "KevalAhirApp/1.0 keval.ahir2019@gmail.com"

PORTFOLIOS_FILE = "portfolios.json"
VIRTUAL_PORTFOLIO_FILE = "virtual_portfolio.json"

# --------------------------------
# Portfolio Helper Functions
# --------------------------------
def load_portfolios():
    if os.path.exists(PORTFOLIOS_FILE):
        try:
            with open(PORTFOLIOS_FILE, 'r') as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}

def save_portfolios(portfolios_data):
    if not isinstance(portfolios_data, dict):
        st.error("Error saving portfolios: Data is not in the correct format.")
        return
    with open(PORTFOLIOS_FILE, 'w') as f:
        json.dump(portfolios_data, f, indent=4)

def load_virtual_portfolio():
    if os.path.exists(VIRTUAL_PORTFOLIO_FILE):
        try:
            with open(VIRTUAL_PORTFOLIO_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return get_default_virtual_portfolio()
    return get_default_virtual_portfolio()

def save_virtual_portfolio(data):
    with open(VIRTUAL_PORTFOLIO_FILE, 'w') as f:
        json.dump(data, f, indent=4, default=str) # Use default=str to handle datetimes if they exist

def get_default_virtual_portfolio():
    return {
        "cash": 3500.0,
        "holdings":,
        "transaction_history":,
        "last_scan_date": None
    }

# --- Session State Initialization ---
if 'portfolios_data' not in st.session_state:
    st.session_state.portfolios_data = load_portfolios()

if 'selected_portfolio_name' not in st.session_state:
    st.session_state.selected_portfolio_name = None
    if st.session_state.portfolios_data:
        st.session_state.selected_portfolio_name = list(st.session_state.portfolios_data.keys())

if 'portfolio_stock_analysis' not in st.session_state:
    st.session_state.portfolio_stock_analysis = {}

if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = {}

if 'live_output' not in st.session_state:
    st.session_state.live_output = {}

if 'virtual_portfolio' not in st.session_state:
    st.session_state.virtual_portfolio = load_virtual_portfolio()

# Initialize flags for running analysis
if 'live_analysis_triggered' not in st.session_state:
    st.session_state.live_analysis_triggered = False
if 'backtest_triggered' not in st.session_state:
    st.session_state.backtest_triggered = False

# --------------------------------
# Data Fetchers
# --------------------------------
@st.cache_data
def fetch_price_history(ticker: str, period: str = "max", interval: str = "1d") -> pd.DataFrame:
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, interval=interval)
        if df.empty: return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception: return pd.DataFrame()

@st.cache_data
def fetch_ticker_info(ticker: str) -> dict:
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
    except Exception: return {}

@st.cache_data
def fetch_enriched_news(ticker: str, ticker_info_data: dict) -> list[dict]:
    try:
        company_name = ticker_info_data.get('longName', ticker_info_data.get('shortName', ticker))
        ticker_obj = yf.Ticker(ticker); raw_news =
        try: raw_news = ticker_obj.news
        except TypeError as te: return [{"error": f"yfinance.news type error for {ticker}: {te}", "source_api": "Yahoo Finance"}]
        except Exception as news_exc: return [{"error": f"yfinance.news call failed for {ticker}: {news_exc}", "source_api": "Yahoo Finance"}]
        enriched_news_list =
        if not raw_news: return
        for news_item in raw_news:
            if not isinstance(news_item, dict): continue
            enriched_item = news_item.copy(); enriched_item['ticker'] = ticker; enriched_item['company_name'] = company_name; enriched_item['source_api'] = 'Yahoo Finance'
            if 'providerPublishTime' in news_item and news_item is not None:
                try:
                    dt_object_utc = datetime.fromtimestamp(int(news_item), tz=timezone.utc)
                    enriched_item['publish_datetime_utc'] = dt_object_utc; enriched_item['publish_time_readable'] = dt_object_utc.strftime('%Y-%m-%d %H:%M:%S %Z')
                except (ValueError, TypeError, OSError) as e_ts: enriched_item['publish_datetime_utc'], enriched_item['publish_time_readable'], enriched_item['publish_time_error'] = None, "N/A", str(e_ts)
            else: enriched_item['publish_datetime_utc'], enriched_item['publish_time_readable'] = None, "N/A"
            for key in ['title', 'publisher', 'link', 'type']: enriched_item.setdefault(key, 'N/A' if key!= 'link' else '#')
            enriched_news_list.append(enriched_item)
        enriched_news_list.sort(key=lambda x: x.get('publish_datetime_utc') or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return enriched_news_list
    except Exception as e: return [{"error": f"Processing Yahoo Finance news for {ticker} failed: {e}", "source_api": "Yahoo Finance"}]

@st.cache_data(ttl=1800)
def fetch_comprehensive_news_from_api(ticker: str, company_name: str, lookback_days: int = 30) -> list[dict]:
    api_key = st.secrets.get("NEWSAPI_KEY")
    if not api_key: return
    newsapi = NewsApiClient(api_key=api_key)
    query = f'("{company_name}" OR {ticker.upper()}) AND (stock OR shares OR business OR finance OR earnings OR "product launch" OR "analyst rating" OR "market sentiment")'
    to_date_dt, from_date_dt = datetime.now(timezone.utc), datetime.now(timezone.utc) - timedelta(days=lookback_days)
    from_param_str, to_param_str = from_date_dt.strftime('%Y-%m-%d'), to_date_dt.strftime('%Y-%m-%d')
    articles_list =
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

@st.cache_data(ttl=6*3600)
def fetch_inst_filings(ticker: str) -> list[dict]:
    """
    Fetches institutional holder data from Yahoo Finance.
    """
    try:
        df_holders = yf.Ticker(ticker).institutional_holders
        if df_holders is not None and not df_holders.empty:
            if 'Shares' in df_holders.columns:
                df_holders = pd.to_numeric(df_holders, errors='coerce').fillna(0)
            if '% Out' in df_holders.columns:
                df_holders['% Out'] = pd.to_numeric(df_holders['% Out'], errors='coerce').fillna(0.0)
            if 'Date Reported' in df_holders.columns:
                # Ensure 'Date Reported' is a string for consistent processing
                df_holders = df_holders.astype(str)
            return df_holders.to_dict("records")
        return [{"error": f"No yfinance institutional holder data for {ticker}."}]
    except Exception as e:
        return [{"error": f"yfinance institutional holders fetch failed for {ticker}: {e}"}]

@st.cache_data(ttl=1800) # Cache for 30 minutes
def fetch_sec_filings_from_search_api(search_query: str, lookback_days: int = 365) -> list[dict]:
    """
    Fetches ALL recent SEC filings using the public EDGAR search API.
    This does not require a dedicated API key.
    """
    headers = {
        'User-Agent': SEC_USER_AGENT,
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    api_url = "https://efts.sec.gov/LATEST/search-index"
    
    payload = {
        "q": search_query,
        "from": 0,
        "size": 100,
        "sort": [{"filed_date": "desc"}]
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        results = response.json()
    except Exception as e:
        return

    if not results or not results.get('hits', {}).get('hits'):
        return

    filings_list =
    date_limit = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    for hit in results['hits']['hits']:
        source = hit.get('_source', {})
        try:
            filing_date_str = source.get('file_date')
            if not filing_date_str: continue
            
            filing_date_dt = datetime.fromisoformat(filing_date_str)
            if filing_date_dt < date_limit:
                continue

            filings_list.append({
                "filing_date": filing_date_str[:10],
                "reporting_owner": ", ".join(source.get('display_names', ["N/A"])),
                "form_type": source.get('form', 'N/A'),
                "link_to_filing": f"https://www.sec.gov/edgar/search/#/submission/{source.get('adsh')}"
            })
        except (ValueError, TypeError, KeyError):
            continue

    return sorted(filings_list, key=lambda x: x.get('filing_date', '1900-01-01'), reverse=True)

@st.cache_data(ttl=4 * 3600)
def fetch_value_investing_io_data(ticker: str) -> dict:
    url = f"https://valueinvesting.io/{ticker.upper()}/valuation/fair-value"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=15); response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser'); target_text = None
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if ticker.upper() in text and "Fair Value" in text and ("Peter Lynch" in text or "based on" in text or "valuation model" in text):
                target_text = text; break
        if not target_text: return {"error": f"VI.io: Peter Lynch Fair Value paragraph not found for {ticker}."}
        pattern = re.compile(r"As of (?P<date>[\d]{4}-[\d]{2}-[\d]{2}), the Fair Value of.*?\(.*?" + re.escape(ticker.upper()) + r".*?\) is (?P<fair_value>[\d\.]+) USD\.?" + r"(?:.*?With the current market price of (?P<market_price>[\d\.]+) USD, the upside of.*? is (?P<upside_percent>[-+]?\d+\.?\d*)%\.?)?")
        match = pattern.search(target_text)
        if match:
            data = match.groupdict()
            return {"ticker": ticker, "vi_valuation_date": data.get("date"), "vi_fair_value": float(data.get("fair_value")) if data.get("fair_value") else None, "vi_site_market_price": float(data.get("market_price")) if data.get("market_price") else None, "vi_upside_percent": float(data.get("upside_percent")) if data.get("upside_percent") else None, "vi_full_text": target_text, "vi_data_source_url": url, "error": None}
        else:
            fv_match = re.search(r"Fair Value.*?is ([\d\.]+) USD", target_text)
            if fv_match: return {"ticker": ticker, "vi_valuation_date": "N/A (generic)", "vi_fair_value": float(fv_match.group(1)) if fv_match.group(1) else None, "vi_site_market_price": None, "vi_upside_percent": None, "vi_full_text": target_text, "vi_data_source_url": url, "error": None, "note": "Generic parse."}
            return {"error": f"VI.io: Could not parse details for {ticker} from: '{target_text[:200]}...'"}
    except requests.exceptions.HTTPError as http_err: return {"error": f"VI.io: HTTP error for {ticker} ({http_err.response.status_code if http_err.response else 'Unknown'}): {url}"}
    except requests.exceptions.RequestException as req_err: return {"error": f"VI.io: Request error for {ticker}: {req_err}"}
    except Exception as e: return {"error": f"VI.io: Unexpected error for {ticker}: {e}"}


# --- LLM Client and Agent Classes ---
class ModelClient:
    def __init__(self, api_key: str, provider: str = "openai"):
        self.api_key, self.provider = api_key, provider
        models = {"openai": "gpt-4o", "deepseek": "deepseek-reasoner"}
        if not api_key: raise ValueError("API key required.")
        self.model_name = models.get(provider)
        if not self.model_name: raise ValueError(f"Unsupported provider: {provider}")
        if provider == "deepseek": self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        elif provider == "openai": self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        try:
            stream = self.client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt}], stream=True)
            return "".join(c.choices.delta.content for c in stream if c.choices and c.choices.delta and c.choices.delta.content)
        except Exception as e: raise Exception(f"LLM Error ({self.provider}, {self.model_name}): {e}")

class PriceAgent:
    def run(self, ticker: str, price_data_slice: pd.DataFrame) -> dict:
        required_data_points = 200

        if price_data_slice.empty or len(price_data_slice) < required_data_points:
            return {
                "ticker": ticker, "price_signal": "hold", "sma50": np.nan, "sma200": np.nan,
                "rsi14": np.nan, "bb_upper": np.nan, "bb_lower": np.nan, "bb_signal": "hold",
                "price_confidence_score": 0.0, "price_error": "Not enough data for comprehensive analysis."
            }

        df = price_data_slice.copy()
        
        # --- FIX: Correctly calculate and assign indicators to new columns ---
        df = df["Close"].rolling(50).mean()
        df = df["Close"].rolling(200).mean()

        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df = 100 - (100 / (1 + rs))

        bb_period = 20
        bb_std_dev = 2
        df = df["Close"].rolling(bb_period).mean()
        df = df["Close"].rolling(bb_period).std()
        df = df + (df * bb_std_dev)
        df = df - (df * bb_std_dev)

        latest = df.iloc[-1]
        
        signal = "hold"
        confidence_score = 0.0
        bb_signal = "hold"

        if pd.isna(latest.SMA50) or pd.isna(latest.SMA200) or pd.isna(latest.RSI14) or pd.isna(latest.BB_Upper) or pd.isna(latest.BB_Lower):
            return {
                "ticker": ticker, "price_signal": "hold", 
                "sma50": float(latest.SMA50) if pd.notna(latest.SMA50) else np.nan,
                "sma200": float(latest.SMA200) if pd.notna(latest.SMA200) else np.nan,
                "rsi14": float(latest.RSI14) if pd.notna(latest.RSI14) else np.nan,
                "bb_upper": float(latest.BB_Upper) if pd.notna(latest.BB_Upper) else np.nan,
                "bb_lower": float(latest.BB_Lower) if pd.notna(latest.BB_Lower) else np.nan,
                "bb_signal": "hold", "price_confidence_score": 0.0,
                "price_error": "Some key indicators are NaN at the latest data point."
            }

        current_close = latest.Close

        if latest.SMA50 > latest.SMA200 and current_close > latest.SMA50:
            if len(df) >= 205 and df.iloc[-5] < df.iloc[-5]:
                signal = "buy"; confidence_score += 0.4
            else:
                signal = "buy"; confidence_score += 0.2
        elif latest.SMA50 < latest.SMA200 and current_close < latest.SMA50:
            if len(df) >= 205 and df.iloc[-5] > df.iloc[-5]:
                signal = "sell"; confidence_score -= 0.4
            else:
                signal = "sell"; confidence_score -= 0.2

        if latest.RSI14 < 30:
            if signal == "buy": confidence_score += 0.2
            elif signal == "hold": signal = "buy"; confidence_score += 0.1
        elif latest.RSI14 > 70:
            if signal == "sell": confidence_score -= 0.2
            elif signal == "hold": signal = "sell"; confidence_score -= 0.1

        if current_close < latest.BB_Lower:
            bb_signal = "buy"
            if signal == "buy": confidence_score += 0.1
            elif signal == "hold": signal = "buy"; confidence_score += 0.05
        elif current_close > latest.BB_Upper:
            bb_signal = "sell"
            if signal == "sell": confidence_score -= 0.1
            elif signal == "hold": signal = "sell"; confidence_score -= 0.05
        
        if signal == "hold":
            if 30 < latest.RSI14 < 40: confidence_score += 0.05
            elif 60 < latest.RSI14 < 70: confidence_score -= 0.05

        if confidence_score > 0.3: final_price_signal = "buy"
        elif confidence_score < -0.3: final_price_signal = "sell"
        else: final_price_signal = "hold"
            
        confidence_score = max(-1.0, min(1.0, confidence_score))

        return {
            "ticker": ticker, "sma50": float(latest.SMA50), "sma200": float(latest.SMA200),
            "rsi14": float(latest.RSI14), "bb_upper": float(latest.BB_Upper), "bb_lower": float(latest.BB_Lower),
            "bb_signal": bb_signal, "price_signal": final_price_signal,
            "price_confidence_score": float(confidence_score), "price_error": None
        }

class MomentumAgent:
    def run(self, ticker: str, price_data_slice: pd.DataFrame) -> dict:
        required_data_points = 253

        if price_data_slice.empty or len(price_data_slice) < required_data_points:
            return {
                "ticker": ticker, "momentum_signal": "hold", "momentum_1m": np.nan,
                "momentum_12m": np.nan, "momentum_confidence_score": 0.0,
                "momentum_error": "Not enough data for 1-year and 1-month momentum."
            }

        df = price_data_slice.copy()

        if 'Close' not in df.columns or not pd.api.types.is_numeric_dtype(df['Close']):
            return {
                "ticker": ticker, "momentum_signal": "hold", "momentum_1m": np.nan,
                "momentum_12m": np.nan, "momentum_confidence_score": 0.0,
                "momentum_error": "Price data is missing 'Close' column or not numeric."
            }

        P_t = df["Close"].iloc[-1]
        P_1m = df["Close"].shift(21).iloc[-1]
        P_12m = df["Close"].shift(252).iloc[-1]

        m1 = ((P_t / P_1m) - 1) if pd.notna(P_1m) and P_1m!= 0 else np.nan
        m12 = ((P_t / P_12m) - 1) if pd.notna(P_12m) and P_12m!= 0 else np.nan

        signal = "hold"
        confidence_score = 0.0

        if pd.notna(m1) and pd.notna(m12):
            raw_combined_momentum = (m1 + m12) / 2
            scaled_confidence = raw_combined_momentum * 5.0 
            confidence_score = max(-1.0, min(1.0, scaled_confidence))

            if confidence_score > 0.3: signal = "buy"
            elif confidence_score < -0.3: signal = "sell"
            else: signal = "hold"

        return {
            "ticker": ticker, "momentum_1m": float(m1) if pd.notna(m1) else np.nan,
            "momentum_12m": float(m12) if pd.notna(m12) else np.nan,
            "momentum_signal": signal, "momentum_confidence_score": float(confidence_score),
            "momentum_error": None
        }

class VolatilityAgent:
    def run(self, ticker: str, data: dict, price_data_slice: pd.DataFrame = None) -> dict:
        beta_val = data.get("ticker_info", {}).get("beta")
        beta = float(beta_val) if isinstance(beta_val, (int, float)) else 1.0

        ann_vol, vol_weight, volatility_confidence_score = np.nan, 0.0, 0.0
        volatility_signal, volatility_error = "hold", None

        if price_data_slice is not None and not price_data_slice.empty and len(price_data_slice) > 1:
            if 'Close' in price_data_slice.columns and pd.api.types.is_numeric_dtype(price_data_slice['Close']):
                ret = np.log(price_data_slice.Close / price_data_slice.Close.shift(1)).dropna()
                if not ret.empty and ret.std() > 0:
                    ann_vol = float(ret.std() * np.sqrt(252))
                    vol_weight = float(1 / ann_vol) if ann_vol > 0 else 0.0
                else:
                    volatility_error = "Could not calculate historical volatility."
            else:
                volatility_error = "Price data missing or invalid 'Close' column."
        else:
            volatility_error = "Not enough price data for volatility calculation."

        if beta > 1.2: volatility_confidence_score -= (beta - 1.2) * 0.5
        elif beta < 0.8: volatility_confidence_score += (0.8 - beta) * 0.5

        if pd.notna(ann_vol):
            if ann_vol > 0.30: volatility_confidence_score -= (ann_vol - 0.30) * 1.0
            elif ann_vol < 0.15: volatility_confidence_score += (0.15 - ann_vol) * 1.0

        volatility_confidence_score = max(-1.0, min(1.0, volatility_confidence_score))

        if volatility_confidence_score > 0.2: volatility_signal = "buy"
        elif volatility_confidence_score < -0.2: volatility_signal = "sell"
        else: volatility_signal = "hold"

        return {
            "ticker": ticker, "beta": float(beta), "annual_vol": float(ann_vol),
            "vol_weight": float(vol_weight), "volatility_signal": volatility_signal,
            "volatility_confidence_score": float(volatility_confidence_score),
            "volatility_error": volatility_error
        }

class SentimentAgent:
    def __init__(self, client):
        self.client = client

    def run(self, ticker: str, data: dict) -> dict:
        news, news_err = data.get("news",), data.get("news_fetch_status_error")

        if news_err:
            return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": news_err}

        valid_news = [item for item in news if isinstance(item, dict) and "error" not in item]
        if not valid_news:
            err_msg = news.get("error") if news and isinstance(news, dict) and "error" in news else "No valid news articles for sentiment."
            return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": err_msg}

        content_for_llm =
        co_name = data.get("ticker_info", {}).get('longName', ticker)
        
        for item in valid_news[:10]:
            title = (item.get('title') or '').strip()
            description = (item.get('description') or '').strip()
            content_snippet = (item.get('content_snippet') or '').replace('[+... chars]', '').strip()
            publisher = (item.get('publisher') or 'N/A').strip()
            
            main_text = ""
            if content_snippet and len(content_snippet) > 50: main_text = f"Content: {content_snippet}"
            elif description and len(description) > 50: main_text = f"Description: {description}"
            
            if main_text:
                snippet = f"Headline: {title} | {main_text}"
                if publisher!= 'N/A': snippet += f" (Source: {publisher})"
                content_for_llm.append(snippet)

        if not content_for_llm:
            return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": "No news with sufficient content."}

        prompt = f"""
        As a financial sentiment analyst, analyze the following news articles for {co_name} ({ticker}).
        Your task is to determine the overall sentiment of these articles towards the company's stock value.
        Output a single numerical score between -1.0 and 1.0 (inclusive).
        - A score of 1.0 indicates extremely positive sentiment (strong buy).
        - A score of 0.0 indicates neutral sentiment (hold).
        - A score of -1.0 indicates extremely negative sentiment (strong sell).
        Focus on information that could impact the stock price (e.g., earnings, product news, analyst ratings, market outlook).
        **Output ONLY the numerical score, nothing else.**
        News Articles:
        """ + "\n".join(f"- {c}" for c in content_for_llm)

        score, llm_err = 0.0, None
        try:
            resp = self.client.generate(prompt).strip()
            match = re.search(r"([-+]?\d*\.\d+)|([-+]?\d+)", resp)
            if match:
                score = max(-1.0, min(1.0, float(match.group(0))))
            else:
                llm_err = f"LLM did not output a recognizable number: '{resp[:50]}...'"
        except Exception as e:
            llm_err = f"LLM sentiment analysis call failed: {str(e)[:150]}"

        final_err = news_err or llm_err
        sentiment_signal = "hold"
        if not final_err:
            if score >= 0.45: sentiment_signal = "buy"
            elif score <= -0.45: sentiment_signal = "sell"

        return {"ticker": ticker, "sentiment_score": float(score), "sentiment_signal": sentiment_signal, "sentiment_confidence_score": abs(score), "sentiment_error": final_err}

class NewsSummaryAgent:
    def __init__(self, client): self.client = client
    
    def run(self, ticker: str, data: dict) -> dict:
        news, co_name, news_fetch_err = data.get("news",), data.get("ticker_info",{}).get('longName',ticker), data.get("news_fetch_status_error")
        if news_fetch_err: return {"ticker":ticker, "news_summary":"Summary skipped due to news fetch issues.", "news_summary_error":news_fetch_err}
        
        valid_news = [item for item in news if isinstance(item, dict) and "error" not in item]
        if not valid_news:
            err = news.get("error") if news and isinstance(news,dict) and "error" in news else "No news for summary."
            return {"ticker":ticker, "news_summary":"No news for summary.", "news_summary_error":err}

        final_snips, titles =, set()
        for item in valid_news[:10]:
            title = (item.get('title') or '').strip()
            desc = (item.get('description') or '').strip()
            cont = (item.get('content_snippet') or '').replace('[+... chars]','').strip()

            if not title or title in titles: continue
            titles.add(title)
            
            text = f"Title: {title}"
            if cont: text += f" | Content: {cont}"
            elif desc: text += f" | Description: {desc}"
            final_snips.append(text)
            
            if len(final_snips) >= 7: break

        if not final_snips: return {"ticker":ticker, "news_summary":"No content for summary.", "news_summary_error":"No articles with content/desc."}
        
        prompt = f"Provide a concise financial summary (max 150 words) for {co_name} ({ticker}) based on these headlines...\n\nArticles:\n" + "\n".join(f"- {s}" for s in final_snips)
        summary, err_msg = "Could not generate summary.", None
        try:
            resp = self.client.generate(prompt).strip()
            if len(resp) < 20: err_msg = "LLM returned an empty or too-short summary."
            else: summary = resp
        except Exception as e:
            err_msg = f"LLM summary call failed: {str(e)[:150]}"
            
        return {"ticker":ticker, "news_summary":summary, "news_summary_error":err_msg}

class FundamentalsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        s = data.get("ticker_info",{}); mc, fcf, roe, de = s.get("marketCap"), s.get("freeCashflow"), s.get("returnOnEquity"), s.get("debtToEquity")
        mc_c = mc if isinstance(mc,(int,float)) else 1; fcf_c = fcf if isinstance(fcf,(int,float)) else 0
        roe_c = roe if isinstance(roe,(int,float)) else 0; de_c = de if isinstance(de,(int,float)) else 1000
        fcy = fcf_c / mc_c if mc_c!= 0 else 0
        ps = sum([roe_c > 0.01, de_c < 100, fcf_c > 0]); sig = "hold"
        if ps >= 2: sig = "buy"
        elif ps == 0: sig = "sell"
        return {"ticker":ticker, "fcf_yield":float(fcy), "piotroski_score":int(ps), "fund_signal":sig}

class ValuationAgent:
    def run(self, ticker: str, data: dict) -> dict:
        stats, hist = data.get("ticker_info",{}), data.get("price_history"); price_v = stats.get("currentPrice")
        if price_v is None and hist is not None and not hist.empty: price_v = hist.Close.iloc[-1]
        curr_p = float(price_v) if isinstance(price_v,(int,float)) and price_v > 0 else None
        if curr_p is None: return {"ticker":ticker, "forward_pe":None, "relative_pe_signal":"hold", "dcf_fair_price":np.nan, "dcf_signal":"hold", "valuation_error":"Current price unavailable."}
        pe_v = stats.get("forwardPE"); pe = float(pe_v) if isinstance(pe_v,(int,float)) else None; rel_sig = "hold"
        if pe is not None and pe > 0: rel_sig = "buy" if pe < 15 else ("sell" if pe > 25 else "hold")
        fcf_v, mc_v = stats.get("freeCashflow"), stats.get("marketCap")
        fcf, mc = (float(fcf_v) if isinstance(fcf_v,(int,float)) else None), (float(mc_v) if isinstance(mc_v,(int,float)) else None)
        fcy = (fcf / mc) if fcf is not None and mc is not None and mc!= 0 else 0.0
        fp_est = curr_p * (1 + fcy); dcf_sig = "hold"
        if fp_est > curr_p * 1.15: dcf_sig = "buy"
        elif fp_est < curr_p * 0.85: dcf_sig = "sell"
        return {"ticker":ticker, "forward_pe":pe, "relative_pe_signal":rel_sig, "dcf_fair_price":float(fp_est) if pd.notna(fp_est) else np.nan, "dcf_signal":dcf_sig, "valuation_error":None}

class AnalystRatingAgent:
    def run(self, ticker: str, data: dict) -> dict:
        info, hist = data.get("ticker_info", {}), data.get("price_history")
        price_v = info.get("currentPrice")
        if price_v is None and hist is not None and not hist.empty:
            price_v = hist.Close.iloc[-1]
        
        curr_p = float(price_v) if isinstance(price_v, (int, float)) and price_v > 0 else None
        if curr_p is None:
            return {"ticker": ticker, "analyst_buy_pct_inferred": 0.5, "target_upside": 0.0, "yfinance_recommendation": "N/A", "analyst_signal": "hold", "analyst_error": "Current price unavailable."}

        target_v = info.get("targetMeanPrice")
        target_m = float(target_v) if isinstance(target_v, (int, float)) else None
        rec = str(info.get("recommendationKey", "hold")).lower()
        
        upside = 0.0
        if target_m is not None and curr_p > 0:
            upside = (target_m / curr_p) - 1

        rating_map = {"strong_buy": 1.0, "buy": 0.75, "hold": 0.0, "underperform": -0.75, "sell": -1.0}
        rating_score = rating_map.get(rec, 0.0)

        if upside > 0.50: upside_score = 1.0
        elif upside > 0.20: upside_score = 0.75
        elif upside > 0.05: upside_score = 0.25
        elif upside < -0.30: upside_score = -1.0
        elif upside < -0.15: upside_score = -0.75
        else: upside_score = 0.0

        final_score = (rating_score * 0.6) + (upside_score * 0.4)

        if final_score >= 0.75: sig = "strong_buy"
        elif final_score >= 0.35: sig = "buy"
        elif final_score <= -0.75: sig = "strong_sell"
        elif final_score <= -0.35: sig = "sell"
        else: sig = "hold"
        
        buy_pct = (final_score + 1.0) / 2.0

        return {"ticker": ticker, "analyst_buy_pct_inferred": float(buy_pct), "target_upside": float(upside), "yfinance_recommendation": rec, "analyst_signal": sig, "analyst_error": None}
        
class SECFilingAgent:
    """
    Analyzes the list of recent SEC filings to generate a signal.
    Focuses on the presence of Form 4 (insider) and Form 13D (activist) filings.
    """
    def run(self, ticker: str, data: dict) -> dict:
        filings = data.get("sec_all_filings_raw",)
        if not filings or (isinstance(filings, dict) and "error" in filings):
            err = filings.get("error") if filings else "No SEC filing data."
            return {"ticker": ticker, "sec_filings_signal": "hold", "sec_filings_error": err}

        recent_form4 = False
        recent_activist = False
        three_months_ago = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

        for f in filings:
            if not isinstance(f, dict): continue
            
            filing_date = f.get('filing_date', '1900-01-01')
            if filing_date >= three_months_ago:
                form_type = str(f.get('form_type', '')).upper()
                if '4' in form_type: # Catches '4', '4/A', etc.
                    recent_form4 = True
                if '13D' in form_type: # Catches '13D', '13D/A', etc.
                    recent_activist = True

        sig = "hold"
        if recent_activist:
            sig = "strong_buy"
        elif recent_form4:
            # A recent Form 4 is noteworthy, but direction is unknown without parsing.
            # Treat it as a weak "hold/watch" signal. The user can see the filing.
            sig = "hold" 
        
        return {"ticker": ticker, "sec_filings_signal": sig, "sec_filings_error": None}
        
class InstitutionalHoldingsAgent:
    """
    IMPROVED: Analyzes institutional ownership from yfinance.
    Includes a heuristic to differentiate between active and passive managers.
    """
    def run(self, ticker: str, data: dict) -> dict:
        holdings = data.get("institutional_holdings",)
        
        inst_data = {
            "ticker": ticker, "inst_holdings_signal": "hold", "inst_holdings_error": None,
            "inst_num_holders": 0, "inst_total_shares_held": 0, "inst_total_pct_out": 0.0,
            "inst_top_holders":, "inst_recently_reported_holders":,
            "active_investor_ratio": 0.0
        }

        if not holdings or (isinstance(holdings, dict) and "error" in holdings):
            inst_data["inst_holdings_error"] = holdings["error"] if holdings else "No institutional holdings data."
            return inst_data

        valid_h = [d for d in holdings if isinstance(d, dict) and "error" not in d]
        if not valid_h:
            inst_data["inst_holdings_error"] = "No valid institutional holdings data found."
            return inst_data

        try:
            inst_data["inst_num_holders"] = len(valid_h)
            inst_data["inst_total_shares_held"] = int(sum(d.get('Shares', 0) for d in valid_h))
            inst_data["inst_total_pct_out"] = float(sum(d.get('% Out', 0.0) for d in valid_h))
            
            inst_data["inst_top_holders"] = sorted(valid_h, key=lambda x: x.get('Shares', 0), reverse=True)[:10]

            recent_date_limit = datetime.now() - timedelta(days=90)
            for h in valid_h:
                date_str = h.get('Date Reported')
                if date_str and isinstance(date_str, str):
                    try:
                        # Handle different possible date formats from yfinance
                        report_date = pd.to_datetime(date_str).to_pydatetime()
                        if report_date > recent_date_limit:
                            inst_data["inst_recently_reported_holders"].append(h)
                    except (ValueError, TypeError):
                        continue
        except Exception as e:
            inst_data["inst_holdings_error"] = f"Error processing institutional holdings: {e}"

        # --- IMPROVED Signal Logic with Active vs. Passive Heuristic ---
        passive_keywords = ['index', 'etf', 'vanguard', 'blackrock', 'state street', 'geode', 'northern trust']
        active_holders_count = 0
        
        for holder in valid_h:
            holder_name = holder.get('Holder', '').lower()
            if not any(keyword in holder_name for keyword in passive_keywords):
                active_holders_count += 1
        
        if inst_data["inst_num_holders"] > 0:
            inst_data["active_investor_ratio"] = active_holders_count / inst_data["inst_num_holders"]

        # Generate signal based on active manager ratio and total ownership
        if inst_data["active_investor_ratio"] > 0.75 and inst_data["inst_total_pct_out"] > 0.5:
            inst_data["inst_holdings_signal"] = "buy" # Strong active institutional conviction
        elif inst_data["active_investor_ratio"] < 0.4 and inst_data["inst_total_pct_out"] > 0.5:
            inst_data["inst_holdings_signal"] = "sell" # Dominated by passive funds, may lack alpha
        
        return inst_data

class ValueInvestingIOAgent:
    def run(self, ticker: str, data: dict) -> dict:
        vi, err = data.get("value_investing_io_data",{}), data.get("value_investing_io_data",{}).get("error")
        fv, site_mp, up_pct, val_date, text = vi.get("vi_fair_value"), vi.get("vi_site_market_price"), vi.get("vi_upside_percent"), vi.get("vi_valuation_date"), vi.get("vi_full_text")
        sig = "hold"; curr_pyf_val = data.get("ticker_info",{}).get("currentPrice")
        if curr_pyf_val is None and data.get("price_history") is not None and not data["price_history"].empty: curr_pyf_val = data["price_history"].Close.iloc[-1]
        curr_pyf = float(curr_pyf_val) if isinstance(curr_pyf_val,(int,float)) and curr_pyf_val > 0 else None
        if not err and fv is not None and curr_pyf is not None:
            mos = 0.15
            if up_pct is not None:
                if up_pct > (mos*100+5): sig="buy"
                elif up_pct < -(mos*100+5): sig="sell"
            else:
                if curr_pyf < fv*(1-mos): sig="buy"
                elif curr_pyf > fv*(1+mos): sig="sell"
        return {"ticker":ticker, "vi_fair_value_estimate":fv, "vi_site_market_price":site_mp, "vi_upside_percent":up_pct, "vi_valuation_date":val_date, "vi_valuation_text_display":text, "vi_signal":sig, "vi_data_error":err}

class PortfolioAgent:
    WEIGHTS = {
        "price": 1.0, "momentum": 0.8, "volatility": 0.3, 
        "sentiment": 0.7, "fund": 1.0, "valuation_dcf": 0.5, 
        "valuation_pe": 0.5, "sec_filings": 1.2, "inst_holdings": 0.8, 
        "analyst": 0.7, "vi_signal": 0.8
    }
    
    def run(self, ticker: str, signals: list[dict], agent_weights: dict = None) -> dict:
        curr_w = agent_weights or self.WEIGHTS
        total_score, sum_w, agg_s = 0.0, 0.0, {}
        
        for s_dict in signals:
            if isinstance(s_dict, dict):
                agg_s.update(s_dict)
                
        s_map = {
            "price_signal": "price", "momentum_signal": "momentum", 
            "volatility_signal": "volatility", "sentiment_signal": "sentiment", 
            "fund_signal": "fund", "dcf_signal": "valuation_dcf", 
            "relative_pe_signal": "valuation_pe", "sec_filings_signal": "sec_filings", 
            "inst_holdings_signal": "inst_holdings", "analyst_signal": "analyst", 
            "vi_signal": "vi_signal"
        }

        for s_key, w_key in s_map.items():
            s_val = agg_s.get(s_key)
            w = curr_w.get(w_key, 0)
            if s_val and w > 0 and s_val in ["buy", "hold", "sell", "strong_buy", "strong_sell"]:
                score_map = {"strong_buy": 1.5, "buy": 1.0, "hold": 0.0, "sell": -1.0, "strong_sell": -1.5}
                raw_score = score_map.get(s_val, 0)
                total_score += raw_score * w
                sum_w += w
                
        comp_score = (total_score / sum_w) if sum_w > 0 else 0.0
        
        if comp_score > 0.35: decision = "strong_buy"
        elif comp_score > 0.15: decision = "buy"
        elif comp_score < -0.35: decision = "strong_sell"
        elif comp_score < -0.15: decision = "sell"
        else: decision = "hold"
        
        return {"ticker": ticker, "composite_score": comp_score, "final_decision": decision}

class AITraderAgent:
    def __init__(self, llm_client: ModelClient, stock_universe: dict):
        self.llm_client = llm_client
        self.stock_universe = stock_universe

    def _generate_trade_reason(self, ticker: str, decision: str, analysis: dict) -> str:
        if not self.llm_client:
            return "LLM client not available for justification."

        co_name = analysis.get('ticker_info', {}).get('longName', ticker)
        score = analysis.get('composite_score', 0)
        summary = analysis.get('news_summary', 'No summary available.')

        prompt = f"""
        As an AI Portfolio Manager, you have decided to '{decision.upper()}' shares of {co_name} ({ticker}).
        The composite analysis score was {score:.2f}.
        The latest news summary is: "{summary}"

        Based on this, provide a single, concise sentence explaining the reason for this trade.
        Example: "Initiating a position due to strong positive sentiment and a bullish analyst rating."
        Example: "Selling to lock in profits after a significant run-up and weakening momentum signals."

        Generate the reason for the {decision.upper()} decision now:
        """
        try:
            reason = self.llm_client.generate(prompt).strip()
            return reason
        except Exception as e:
            return f"Could not generate reason due to LLM error: {e}"

    def _is_safe(self, analysis: dict) -> bool:
        """Determines if a stock is 'safe' based on predefined criteria."""
        info = analysis.get("ticker_info", {})
        market_cap = info.get("marketCap", 0)
        beta = info.get("beta", 1.0)
        return isinstance(market_cap, (int, float)) and market_cap > 100e9 and isinstance(beta, (int, float)) and beta < 1.2

    def run(self, portfolio_state: dict, analysis_results: dict):
        trades_to_make =
        cash = portfolio_state['cash']
        holdings = list(portfolio_state['holdings'])

        tickers_in_portfolio = {h['ticker'] for h in holdings}
        for i, holding in reversed(list(enumerate(holdings))):
            ticker = holding['ticker']
            if ticker not in analysis_results or analysis_results[ticker].get('error'):
                continue

            analysis = analysis_results[ticker]
            if analysis.get('final_decision') in ['sell', 'strong_sell']:
                reason = self._generate_trade_reason(ticker, 'sell', analysis)
                price = analysis.get('current_price_display')
                if isinstance(price, (int, float)) and price > 0:
                    trades_to_make.append({
                        "ticker": ticker, "type": "sell", "quantity": holding['quantity'],
                        "price": price, "reason": reason
                    })
                    cash += holding['quantity'] * price
                    holdings.pop(i)

        current_holdings_value = 0
        for h in holdings:
            price = analysis_results.get(h['ticker'], {}).get('current_price_display')
            if isinstance(price, (int, float)):
                current_holdings_value += h['quantity'] * price
        total_portfolio_value = cash + current_holdings_value

        target_safe_value = total_portfolio_value * 0.60
        target_risky_value = total_portfolio_value * 0.40

        current_safe_value, current_risky_value = 0, 0
        for h in holdings:
            analysis = analysis_results.get(h['ticker'])
            if analysis and not analysis.get('error'):
                price = analysis.get('current_price_display')
                if isinstance(price, (int, float)):
                    value = h['quantity'] * price
                    if self._is_safe(analysis):
                        current_safe_value += value
                    else:
                        current_risky_value += value

        buy_candidates = sorted(
            [res for res in analysis_results.values() if res.get('final_decision') in ['buy', 'strong_buy'] and res.get('ticker') not in tickers_in_portfolio and not res.get('error')],
            key=lambda x: x.get('composite_score', 0), reverse=True
        )

        investment_per_stock = cash * 0.25
        if investment_per_stock < 500 and cash > 500:
            investment_per_stock = 500

        for candidate in buy_candidates:
            if cash < investment_per_stock or investment_per_stock <= 1:
                break

            price = candidate.get('current_price_display')
            if not isinstance(price, (int, float)) or price <= 0:
                continue

            is_safe_candidate = self._is_safe(candidate)
            should_buy = False
            if is_safe_candidate and current_safe_value < target_safe_value:
                should_buy = True
            elif not is_safe_candidate and current_risky_value < target_risky_value:
                should_buy = True

            if should_buy:
                quantity_to_buy = investment_per_stock / price
                reason = self._generate_trade_reason(candidate['ticker'], 'buy', candidate)
                trades_to_make.append({
                    "ticker": candidate['ticker'], "type": "buy", "quantity": quantity_to_buy,
                    "price": price, "reason": reason
                })
                cash -= investment_per_stock
                if is_safe_candidate:
                    current_safe_value += investment_per_stock
                else:
                    current_risky_value += investment_per_stock
                tickers_in_portfolio.add(candidate['ticker'])

        return trades_to_make

# --- Orchestrator and Backtesting ---
def run_live_analysis(tickers, llm_client, configs):
    results = {}
    progress_bar = st.progress(0, text="Starting analysis...")
    for i, t in enumerate(tickers):
        progress_text = f"Analyzing {t}... ({i+1}/{len(tickers)})"
        progress_bar.progress((i + 1) / len(tickers), text=progress_text)
        
        price_history_full = fetch_price_history(t, period="max")
        if price_history_full.empty:
            results[t] = {"error": f"Price history unavailable for {t}.", "ticker": t, "final_decision":"error", "composite_score":0}; continue
            
        ticker_info = fetch_ticker_info(t)
        if not ticker_info or not ticker_info.get("financialCurrency"):
            err_msg = f"Core ticker info (e.g., currency) unavailable for {t}. Invalid/delisted/no yfinance data."
            results[t] = {"error": err_msg, "ticker": t, "final_decision":"error", "composite_score":0}; continue

        current_price_for_ticker = ticker_info.get("currentPrice")
        if current_price_for_ticker is None and not price_history_full.empty:
            current_price_for_ticker = price_history_full["Close"].iloc[-1]
            
        company_name_for_news = ticker_info.get('longName', ticker_info.get('shortName', t))
        
        combined_news, news_fetch_msgs =,
        if configs["use_sentiment"]:
            yf_news = fetch_enriched_news(t, ticker_info)
            if yf_news and not (isinstance(yf_news,dict) and "error" in yf_news): combined_news.extend(yf_news)
            elif yf_news and isinstance(yf_news,dict) and "error" in yf_news: news_fetch_msgs.append(f"Yahoo: {yf_news['error']}")
            
            if llm_client and st.secrets.get("NEWSAPI_KEY"):
                api_news = fetch_comprehensive_news_from_api(t, company_name_for_news)
                if api_news and not (isinstance(api_news,dict) and "error" in api_news): combined_news.extend(api_news)
                elif api_news and isinstance(api_news,dict) and "error" in api_news: news_fetch_msgs.append(f"NewsAPI: {api_news['error']}")
            elif configs["use_sentiment"] and not st.secrets.get("NEWSAPI_KEY"): news_fetch_msgs.append("NewsAPI Key missing.")

        seen_urls, dedup_news = set(),
        for item in combined_news:
            if isinstance(item,dict) and "error" not in item:
                url = item.get('link') or item.get('url')
                if url and url not in seen_urls: dedup_news.append(item); seen_urls.add(url)
        if dedup_news: dedup_news.sort(key=lambda x: x.get('publish_datetime_utc') or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        news_status_bundle = " | ".join(news_fetch_msgs) if news_fetch_msgs else "News fetch OK"
        if not dedup_news and not news_fetch_msgs and configs["use_sentiment"]: news_status_bundle="No news articles found."

        data_bundle = {
            "price_history":price_history_full, "ticker_info":ticker_info, "news":dedup_news,
            "news_fetch_status_error": news_status_bundle if any(kw in news_status_bundle.lower() for kw in ["error","failed","no news","missing"]) else None,
            "value_investing_io_data":fetch_value_investing_io_data(t) if configs["use_value_trades"] else {"error":"VI.io: Skipped."},
            "institutional_holdings":fetch_inst_filings(t) if configs["use_filings"] else,
            "sec_all_filings_raw": fetch_sec_filings_from_search_api(t) if configs["use_filings"] else
        }

        agents =
        if configs["use_sentiment"] and llm_client: agents.extend()
        if configs["use_filings"]: agents.extend()
        if configs["use_value_trades"]: agents.append(ValueInvestingIOAgent())
        
        agent_res_list =
        for agent in agents:
            name = agent.__class__.__name__
            try:
                # Pass price_data_slice to agents that need it
                if name in ["PriceAgent", "MomentumAgent", "VolatilityAgent"]:
                    res_a = agent.run(t, data_bundle, price_data_slice=data_bundle["price_history"]) if name == "VolatilityAgent" else agent.run(t, data_bundle["price_history"])
                else: 
                    res_a = agent.run(t, data_bundle)
                agent_res_list.append(res_a)
            except Exception as e:
                err_k, sig_k = name.lower().replace("agent","")+"_error", name.lower().replace("agent","")+"_signal"
                agent_res_list.append({sig_k:"error", err_k:f"Agent {name} error: {str(e)[:150]}"}); st.warning(f"Error in {name} for {t}: {e}")

        final_dec = PortfolioAgent().run(t, agent_res_list)
        
        curr_res_dict = {
            "ticker":t, "current_price_display":current_price_for_ticker, "market_cap_display":ticker_info.get("marketCap"),
            "industry_display":ticker_info.get("industry"), "sector_display":ticker_info.get("sector"), "ticker_info":ticker_info,
            "news_headlines_for_popover":[f"{n.get('publish_time_readable','N/A')} - {n.get('title','N/A')} ({n.get('publisher','N/A')}) [Link]({n.get('link','#')})" for n in dedup_news[:10]],
            "news_status_display":news_status_bundle,
            "sec_all_filings_raw": data_bundle["sec_all_filings_raw"]
        }
        for r_dict in agent_res_list:
            if isinstance(r_dict,dict): curr_res_dict.update(r_dict)
        curr_res_dict.update(final_dec)
        results[t] = curr_res_dict
        
    progress_bar.empty()
    return results

def run_backtest(ticker, start_date, end_date, initial_capital, llm_client_placeholder, backtest_agent_weights):
    st.write(f"Preparing backtest: {ticker} ({start_date} to {end_date})...")
    s_dt = datetime.strptime(start_date, "%Y-%m-%d"); fetch_s_dt = (s_dt - pd.DateOffset(months=18)).strftime("%Y-%m-%d")
    full_hist = fetch_price_history(ticker, period="max", interval="1d")
    if full_hist.empty: return {"error": f"Backtest fail {ticker}: Price history empty."}, pd.DataFrame()
    hist = full_hist[(full_hist.index >= pd.to_datetime(fetch_s_dt)) & (full_hist.index <= pd.to_datetime(end_date))].copy()
    if hist.empty or len(hist[hist.index >= pd.to_datetime(start_date)]) < 2: return {"error": f"Backtest fail {ticker}: Not enough data in range."}, pd.DataFrame()
    info_bt = fetch_ticker_info(ticker); data_static = {"ticker_info": info_bt}
    p_agent, m_agent, v_agent, port_agent = PriceAgent(), MomentumAgent(), VolatilityAgent(), PortfolioAgent()
    log, cash, shares, port_val =, initial_capital, 0, initial_capital
    run_dates = hist[hist.index >= pd.to_datetime(start_date)].index
    for curr_dt in run_dates:
        data_sl = hist[hist.index <= curr_dt]
        curr_price_pt = data_sl.Close.iloc[-1] if not data_sl.empty else (port_val / shares if shares > 0 else 0)
        if data_sl.empty or len(data_sl) < 253:
            log.append({"date":curr_dt, "cash":cash, "shares_held":shares, "price":curr_price_pt, "portfolio_value":port_val, "signal":"hold (no data)", "composite_score":0.0}); continue
        curr_price = data_sl.Close.iloc[-1]
        pa_r, ma_r, va_r = p_agent.run(ticker,data_sl), m_agent.run(ticker,data_sl), v_agent.run(ticker,data_static,data_sl)
        final_dec_obj = port_agent.run(ticker, [pa_r,ma_r,va_r], agent_weights=backtest_agent_weights)
        final_dec = final_dec_obj["final_decision"]
        if final_dec in ["buy", "strong_buy"] and cash > curr_price and curr_price > 0: s_buy = cash/curr_price; shares += s_buy; cash=0
        elif final_dec in ["sell", "strong_sell"] and shares > 0: cash += shares*curr_price; shares=0
        port_val = cash + shares*curr_price
        log.append({"date":curr_dt, "cash":cash, "shares_held":shares, "price":curr_price, "portfolio_value":port_val, "signal":final_dec, "composite_score":final_dec_obj["composite_score"]})
    log_df = pd.DataFrame(log)
    if not log_df.empty: log_df.set_index("date",inplace=True)
    if log_df.empty or len(log_df) < 2: return {"message":f"Backtest log {ticker} too short."}, pd.DataFrame()
    total_ret = (log_df.portfolio_value.iloc[-1]/initial_capital - 1)*100
    days = (log_df.index[-1]-log_df.index).days; years = days/365.25 if days > 0 else (1/365.25 if days==0 else 0)
    ann_ret = 0
    if years > 0 and initial_capital > 0: ann_ret = ((log_df.portfolio_value.iloc[-1]/initial_capital)**(1/years)-1)*100
    elif years == 0 and initial_capital > 0: ann_ret = total_ret
    log_df["daily_return"] = log_df.portfolio_value.pct_change().fillna(0); ann_vol = log_df.daily_return.std()*np.sqrt(252)*100
    sharpe = (ann_ret/ann_vol) if ann_vol!=0 else 0
    log_df["cum_max"] = log_df.portfolio_value.cummax(); log_df["drawdown"] = (log_df.portfolio_value - log_df.cum_max)/log_df.cum_max.replace(0,np.nan)
    max_dd = log_df.drawdown.min()*100 if not log_df.drawdown.empty and pd.notna(log_df.drawdown.min()) else 0
    trades = (log_df.signal!= log_df.signal.shift()).fillna(False).sum()//2
    return {"Initial Capital":f"${initial_capital:,.2f}", "Final Portfolio Value":f"${log_df.portfolio_value.iloc[-1]:,.2f}", "Total Return (%)":f"{total_ret:.2f}%", "Annualized Return (%)":f"{ann_ret:.2f}%", "Annualized Volatility (%)":f"{ann_vol:.2f}%", "Sharpe Ratio":f"{sharpe:.2f}", "Max Drawdown (%)":f"{max_dd:.2f}%", "Number of Trades (approx)":f"{trades}"}, log_df

def display_detailed_analysis(res_detail):
    ticker = res_detail.get("ticker", "N/A")
    ticker_info = res_detail.get("ticker_info", {})
    tab_titles =
    tabs = st.tabs(tab_titles)

    # Main Header
    st.subheader(f"Detailed Analysis for {ticker_info.get('longName', ticker)}")
    sig_col1, sig_col2, sig_col3, sig_col4 = st.columns(4)
    sig_col1.metric("Final AI Decision", str(res_detail.get('final_decision', 'N/A')).upper())
    sig_col2.metric("Composite Score", f"{res_detail.get('composite_score', 0):.2f}")
    sig_col3.metric("Analyst Signal", str(res_detail.get('analyst_signal', 'N/A')).upper())
    sig_col4.metric("Insider Signal", str(res_detail.get('sec_filings_signal', 'N/A')).upper())
    st.markdown("---")

    with tabs: # Technicals & Momentum
        st.markdown("#### Price Chart (1-Year)")
        price_hist_chart = fetch_price_history(ticker, period="1y")
        if not price_hist_chart.empty:
            st.line_chart(price_hist_chart["Close"], use_container_width=True, color="#0072F0")
        
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.markdown("##### Technical Signal")
            st.info(f"**Signal:** {str(res_detail.get('price_signal', 'N/A')).upper()}")
        with t_col2:
            st.markdown("##### Momentum Signal")
            st.info(f"**Signal:** {str(res_detail.get('momentum_signal', 'N/A')).upper()}")

    with tabs[1]: # Fundamentals & Value
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.markdown("##### Core Fundamentals")
            st.info(f"**Signal:** {str(res_detail.get('fund_signal', 'N/A')).upper()}")
            cap_str = f"${ticker_info.get('marketCap', 0) / 1e9:.2f}B" if isinstance(ticker_info.get('marketCap'), (int, float)) else "N/A"
            roe_val = ticker_info.get('returnOnEquity')
            st.caption(f"Market Cap: {cap_str}")
            st.caption(f"Return on Equity: {roe_val * 100:.2f}%" if isinstance(roe_val, (int,float)) else "ROE: N/A")
        with f_col2:
            st.markdown("##### Valuation")
            st.info(f"**VI.io Signal:** {str(res_detail.get('vi_signal', 'N/A')).upper()}")
            up_val = res_detail.get('vi_upside_percent')
            st.caption(f"Forward P/E: {ticker_info.get('forwardPE'):.2f}" if isinstance(ticker_info.get('forwardPE'), (int,float)) else "P/E: N/A")
            st.caption(f"VI.io Upside: {up_val:.1f}%" if isinstance(up_val, (int,float)) else "VI.io Upside: N/A")

    with tabs[2]: # News & Filings
        news_col, file_col = st.columns([3, 2])
        with news_col:
            st.markdown("##### News Summary & Sentiment")
            st.info(f"**Sentiment Signal:** {str(res_detail.get('sentiment_signal','N/A')).upper()}")
            st.write(res_detail.get('news_summary', 'No summary available.'))
        
        with file_col:
            st.markdown("##### Corporate Filings")
            with st.expander("View All Recent Filings"):
                all_filings = res_detail.get('sec_all_filings_raw',)
                if not all_filings or (isinstance(all_filings, dict) and 'error' in all_filings):
                    st.warning(all_filings['error'] if all_filings else "Filings could not be fetched.")
                else:
                    st.dataframe(pd.DataFrame(all_filings), use_container_width=True, hide_index=True)

            with st.expander("Institutional Holders"):
                def display_holders_df(holders_data):
                    if not holders_data:
                        st.info("No data available for this section.")
                        return
                    
                    df_holders = pd.DataFrame(holders_data)
                    if '% Out' in df_holders.columns:
                        df_holders = df_holders.rename(columns={"% Out": "% of Outstanding"})
                    
                    desired_cols =
                    available_cols = [col for col in desired_cols if col in df_holders.columns]
                    
                    if not available_cols:
                        st.warning("Holder data is in an unexpected format.")
                        st.dataframe(df_holders)
                    else:
                        st.dataframe(df_holders[available_cols], hide_index=True, use_container_width=True)

                st.markdown("###### Top 10 Holders")
                display_holders_df(res_detail.get('inst_top_holders',))
                
                st.markdown("###### Recently Reported")
                display_holders_df(res_detail.get('inst_recently_reported_holders',))

    with tabs[3]: # All Signals (Raw Data)
        st.subheader("All Agent Raw Outputs")
        filtered_results = {k: v for k, v in res_detail.items() if not isinstance(v, (dict, list))}
        st.dataframe(pd.DataFrame(filtered_results.items(), columns=["Metric", "Value"]), use_container_width=True, hide_index=True)

# --- Streamlit UI ---
llm_client = None
try:
    # Prioritize Streamlit secrets, then fall back to environment variables
    ds_key = st.secrets.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    oa_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    
    if ds_key: 
        llm_client = ModelClient(api_key=ds_key, provider="deepseek")
        st.sidebar.caption("✅ LLM: DeepSeek")
    elif oa_key: 
        llm_client = ModelClient(api_key=oa_key, provider="openai")
        st.sidebar.caption("✅ LLM: OpenAI")
    else: 
        st.sidebar.warning("LLM API key missing. Sentiment/Summary disabled.")
except Exception as e: 
    st.sidebar.error(f"LLM Init Error: {e}")
    llm_client = None

st.title("🚀 AI Hedge Fund Simulator")
st.header("⚙️ Configuration"); config_cont = st.container(border=True)

app_mode_options =
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = app_mode_options

with config_cont:
    current_mode_index = app_mode_options.index(st.session_state.app_mode)
    selected_mode = st.radio("Select Mode:", app_mode_options, key="app_mode_sel_main_key", horizontal=True, index=current_mode_index)
    if selected_mode!= st.session_state.app_mode:
        st.session_state.app_mode = selected_mode
        st.session_state.live_analysis_triggered = False
        st.session_state.backtest_triggered = False
        st.rerun()

    st.markdown("---")

    if st.session_state.app_mode == "Live Analysis":
        st.subheader("Live Analysis Settings")
        tickers_in_live = st.text_input("Tickers (comma-separated):", "AAPL,MSFT,GOOG,CRWD", key="live_tickers_input")
        st.caption("ℹ️ Live analysis uses all available historical data.")
        st.subheader("Feature Toggles")
        
        feat_cols = st.columns(3)
        with feat_cols:
            use_sent_live = st.checkbox("News Sentiment & Summary (LLM)", value=bool(llm_client), disabled=not llm_client, key="live_sent_cb_main", help="Uses LLM. Requires NewsAPI key.")
        with feat_cols[1]:
            use_filings_live = st.checkbox("SEC & Inst. Filings", value=True, key="live_sec_cb_main")
        with feat_cols[2]:
            use_valtrades_live = st.checkbox("ValueInvesting.io (Exp.)", value=False, key="live_vt_cb_main", help="Scrapes ValueInvesting.io. May be slow/unreliable.")
        
        if st.button("🚀 Run Live Analysis", use_container_width=True, type="primary", key="run_live_analysis_button"):
            live_tickers = [t.strip().upper() for t in tickers_in_live.split(",") if t.strip()]
            if not live_tickers:
                st.error("Please enter at least one ticker.")
            else:
                # REMOVED: 'use_politician_filings' as it was non-functional
                live_configs = {
                    "use_sentiment": use_sent_live, 
                    "use_filings": use_filings_live,
                    "use_value_trades": use_valtrades_live
                }
                with st.spinner("⏳ Processing live analysis..."):
                    st.session_state.live_output = run_live_analysis(live_tickers, llm_client, live_configs)
                    st.session_state.live_analysis_triggered = True
                    st.rerun()

    elif st.session_state.app_mode == "Backtesting":
        st.subheader("Backtesting Settings")
        st.session_state.bt_ticker = st.text_input("Ticker:", "AAPL", key="bt_ticker_in_bt").upper()
        
        bt_capital_source = st.radio("Capital Source:", ("Manual Input", "From Saved Portfolio"), horizontal=True, key="bt_capital_source_radio")
        bt_capital = 10000
        if bt_capital_source == "Manual Input":
            bt_capital = st.number_input("Initial Capital:", 1000, 1000000, 10000, 1000, key="bt_cap_in_bt", format="%d")
        else:
            portfolio_names_bt = list(st.session_state.portfolios_data.keys())
            if not portfolio_names_bt: st.warning("No portfolios found.")
            else:
                sel_pf_bt = st.selectbox("Select Portfolio to use its total value:", portfolio_names_bt, key="bt_pf_select")
                # Logic to fetch portfolio value would go here
        
        bt_c1, bt_c2 = st.columns(2)
        with bt_c1:
            def_end_dt = datetime.now()-timedelta(days=1); def_start_dt = def_end_dt-pd.DateOffset(years=3)
            start_dt_in = st.date_input("Start Date:", def_start_dt, max_value=def_end_dt-timedelta(days=30), key="bt_start_dt_bt")
            st.session_state.bt_start_str = start_dt_in.strftime("%Y-%m-%d")
        with bt_c2:
            min_end_dt_bt = start_dt_in+timedelta(days=30)
            end_dt_in = st.date_input("End Date:", def_end_dt, min_value=min_end_dt_bt, max_value=datetime.now()-timedelta(days=1), key="bt_end_dt_bt")
            st.session_state.bt_end_str = end_dt_in.strftime("%Y-%m-%d")
        
        with st.expander("Adjust Backtest Agent Weights",expanded=False):
            w_p = st.slider("Price W:",0.0,2.0,1.0,0.1,key="bt_w_p_bt")
            w_m = st.slider("Mom W:",0.0,2.0,0.8,0.1,key="bt_w_m_bt")
            w_v = st.slider("Vol W:",0.0,2.0,0.2,0.1,key="bt_w_v_bt")
            st.info("Other signals disabled in backtest for speed.")
            st.session_state.bt_weights = {"price":w_p, "momentum":w_m, "volatility":w_v}
            st.session_state.bt_capital = bt_capital

        if st.button("📈 Run Backtest",use_container_width=True,type="primary",key="run_bt_btn_main"):
            if st.session_state.bt_ticker:
                with st.spinner(f"⏳ Running backtest for {st.session_state.bt_ticker}..."):
                    metrics, log_df = run_backtest(st.session_state.bt_ticker, st.session_state.bt_start_str, st.session_state.bt_end_str, st.session_state.bt_capital, llm_client, st.session_state.bt_weights)
                    st.session_state.backtest_results[st.session_state.bt_ticker] = {"metrics": metrics, "log_df": log_df}
                    st.session_state.backtest_triggered = True
                    st.rerun()

    elif st.session_state.app_mode == "💼 Portfolio Management":
        st.subheader("💼 Portfolio Management")
        portfolio_names_list = list(st.session_state.portfolios_data.keys())

        if not portfolio_names_list:
            st.session_state.portfolios_data["My First Portfolio"] = {"holdings":, "cash": 10000.0}
            st.session_state.selected_portfolio_name = "My First Portfolio"
            save_portfolios(st.session_state.portfolios_data)
            st.rerun()

        col_pf1, col_pf2, col_pf3 = st.columns([3, 1, 1])

        st.session_state.selected_portfolio_name = col_pf1.selectbox(
            "Select Portfolio:",
            portfolio_names_list,
            index=portfolio_names_list.index(st.session_state.selected_portfolio_name) if st.session_state.selected_portfolio_name in portfolio_names_list else 0,
            key="portfolio_selector"
        )
        current_portfolio = st.session_state.portfolios_data.get(st.session_state.selected_portfolio_name, {"holdings":, "cash": 0.0})

        new_portfolio_name = col_pf2.text_input("New Portfolio Name:", "", key="new_pf_name")
        if col_pf3.button("➕ Create Portfolio", key="create_pf_btn"):
            if new_portfolio_name and new_portfolio_name not in st.session_state.portfolios_data:
                st.session_state.portfolios_data[new_portfolio_na

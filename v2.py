import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
import openai
from openai import OpenAI
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
import json

# --- Page Config (Must be the first Streamlit command) ---
st.set_page_config(page_title="AI Hedge Fund Simulator", layout="wide")

# Load environment variables (if running locally)
load_dotenv()

# --- Application Configuration ---
CONFIG = {
    "SEC_USER_AGENT": "KevalAhirApp/1.0 keval.ahir2019@gmail.com",
    "PORTFOLIOS_FILE": "portfolios.json",
    "VIRTUAL_PORTFOLIO_FILE": "virtual_portfolio.json",
    "AGENT_WEIGHTS": {
        "price": 1.0, "momentum": 0.8, "volatility": 0.3, "sentiment": 0.6,
        "fund": 0.9, "valuation_dcf": 0.5, "valuation_pe": 0.5, "sec_filings": 0.6,
        "inst_holdings": 0.3, "analyst": 0.7, "politician_filings": 0.4, "vi_signal": 0.8
    },
    "AI_TRADER_UNIVERSE": {
        "safe": ['MSFT', 'AAPL', 'JNJ', 'V', 'PG', 'GOOGL', 'JPM'],
        "risky": ['CRWD', 'PLTR', 'U', 'COIN', 'RBLX', 'SNOW', 'MDB']
    }
}

# --- Portfolio Helper Functions ---
def load_portfolios() -> Dict:
    if os.path.exists(CONFIG["PORTFOLIOS_FILE"]):
        try:
            with open(CONFIG["PORTFOLIOS_FILE"], 'r') as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}

def save_portfolios(portfolios_data: Dict) -> None:
    if not isinstance(portfolios_data, dict):
        st.error("Error saving portfolios: Data is not in the correct format.")
        return
    with open(CONFIG["PORTFOLIOS_FILE"], 'w') as f:
        json.dump(portfolios_data, f, indent=4)

def load_virtual_portfolio() -> Dict:
    if os.path.exists(CONFIG["VIRTUAL_PORTFOLIO_FILE"]):
        try:
            with open(CONFIG["VIRTUAL_PORTFOLIO_FILE"], 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return get_default_virtual_portfolio()
    return get_default_virtual_portfolio()

def save_virtual_portfolio(data: Dict) -> None:
    with open(CONFIG["VIRTUAL_PORTFOLIO_FILE"], 'w') as f:
        json.dump(data, f, indent=4, default=str)

def get_default_virtual_portfolio() -> Dict:
    return {
        "cash": 3500.0,
        "holdings": [],
        "transaction_history": [],
        "last_scan_date": None
    }

# --- Session State Initialization ---
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
if 'live_analysis_triggered' not in st.session_state:
    st.session_state.live_analysis_triggered = False
if 'backtest_triggered' not in st.session_state:
    st.session_state.backtest_triggered = False

# --- Data Fetchers ---
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
def fetch_ticker_info(ticker: str) -> Dict[str, Any]:
    try:
        info = yf.Ticker(ticker).info
        if not info or info.get('quoteType') == "MUTUALFUND" or (info.get('regularMarketPrice') is None and info.get('currentPrice') is None):
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
def fetch_enriched_news(ticker: str, ticker_info_data: dict) -> List[dict]:
    try:
        company_name = ticker_info_data.get('longName', ticker_info_data.get('shortName', ticker))
        ticker_obj = yf.Ticker(ticker)
        raw_news = []
        try:
            raw_news = ticker_obj.news
        except Exception as news_exc:
            return [{"error": f"yfinance .news call failed for {ticker}: {news_exc}", "source_api": "Yahoo Finance"}]
        
        enriched_news_list = []
        if not raw_news: return []
        for news_item in raw_news:
            if not isinstance(news_item, dict): continue
            enriched_item = news_item.copy()
            enriched_item.update({'ticker': ticker, 'company_name': company_name, 'source_api': 'Yahoo Finance'})
            if 'providerPublishTime' in news_item and news_item['providerPublishTime'] is not None:
                try:
                    dt_object_utc = datetime.fromtimestamp(int(news_item['providerPublishTime']), tz=timezone.utc)
                    enriched_item['publish_datetime_utc'] = dt_object_utc
                    enriched_item['publish_time_readable'] = dt_object_utc.strftime('%Y-%m-%d %H:%M:%S %Z')
                except (ValueError, TypeError, OSError) as e_ts:
                    enriched_item.update({'publish_datetime_utc': None, 'publish_time_readable': "N/A", 'publish_time_error': str(e_ts)})
            else:
                enriched_item.update({'publish_datetime_utc': None, 'publish_time_readable': "N/A"})
            
            for key in ['title', 'publisher', 'link', 'type']:
                enriched_item.setdefault(key, 'N/A' if key != 'link' else '#')
            enriched_news_list.append(enriched_item)
            
        enriched_news_list.sort(key=lambda x: x.get('publish_datetime_utc') or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return enriched_news_list
    except Exception as e:
        return [{"error": f"Processing Yahoo Finance news for {ticker} failed: {e}", "source_api": "Yahoo Finance"}]

@st.cache_data(ttl=1800)
def fetch_comprehensive_news_from_api(ticker: str, company_name: str, lookback_days: int = 30) -> List[dict]:
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
                    try:
                        dt_obj_utc = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                        readable_time = dt_obj_utc.strftime('%Y-%m-%d %H:%M:%S %Z')
                    except ValueError:
                        pass
                articles_list.append({
                    "uuid": article.get('url'), "title": article.get('title', 'No Title'),
                    "publisher": article.get('source', {}).get('name', 'N/A'), "link": article.get('url', '#'),
                    "publish_datetime_utc": dt_obj_utc, "publish_time_readable": readable_time,
                    "description": article.get('description'), "content_snippet": article.get('content'),
                    "company_name": company_name, "ticker": ticker, "source_api": "NewsAPI.org"
                })
        elif all_articles_response.get("status") == "error":
            return [{"error": f"NewsAPI Error ({ticker}): {all_articles_response.get('code')} - {all_articles_response.get('message')}", "source_api": "NewsAPI.org"}]
        else:
            return [{"error": f"NewsAPI ({ticker}): No articles or unexpected structure.", "source_api": "NewsAPI.org"}]
    except requests.exceptions.RequestException as e:
        return [{"error": f"NewsAPI request failed for {ticker}: {e}", "source_api": "NewsAPI.org"}]
    except Exception as e:
        return [{"error": f"Unexpected error with NewsAPI for {ticker}: {e}", "source_api": "NewsAPI.org"}]
    return articles_list

@st.cache_data(ttl=24 * 3600)
def get_all_cik_ticker_mappings() -> Dict[str, str]:
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(url, headers={'User-Agent': CONFIG["SEC_USER_AGENT"]})
        response.raise_for_status()
        return {item['ticker']: str(item['cik_str']).zfill(10) for item in response.json() if 'ticker' in item and 'cik_str' in item}
    except Exception as e:
        st.error(f"CRITICAL: Failed to fetch CIK mappings: {e}.")
        return {}
TICKER_TO_CIK_MAP = get_all_cik_ticker_mappings()

def get_cik_for_ticker(ticker: str) -> Optional[str]:
    return TICKER_TO_CIK_MAP.get(ticker.upper())

@st.cache_data(ttl=4 * 3600)
def fetch_all_sec_filings(ticker_symbol: str, lookback_days: int = 365) -> List[dict]:
    cik = get_cik_for_ticker(ticker_symbol)
    if not cik:
        return [{"error": f"SEC: CIK not found for {ticker_symbol} in local map."}]
    
    headers = {'User-Agent': CONFIG["SEC_USER_AGENT"]}
    submissions_url = f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json"
    
    try:
        response = requests.get(submissions_url, headers=headers, timeout=20)
        response.raise_for_status()
        submissions_data = response.json()
    except requests.exceptions.RequestException as e:
        return [{"error": f"SEC Request error for {ticker_symbol}: {e}"}]
    except Exception as e:
        return [{"error": f"SEC Unexpected error for {ticker_symbol}: {e}"}]
        
    filings_list = []
    if 'filings' in submissions_data and 'recent' in submissions_data['filings']:
        recent = submissions_data['filings']['recent']
        date_limit = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        
        for i in range(len(recent.get('form', []))):
            try:
                filing_date = datetime.strptime(recent['filingDate'][i], '%Y-%m-%d').replace(tzinfo=timezone.utc)
                if filing_date < date_limit: continue
                
                form_type = recent['form'][i]
                acc_no = recent['accessionNumber'][i]
                cik_padded = str(cik).zfill(10)
                acc_no_dashless = acc_no.replace('-', '')
                idx_link = f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{acc_no_dashless}/{acc_no}-index.html"
                
                filing_data = {
                    "ticker": ticker_symbol,
                    "filing_date_str": recent['filingDate'][i],
                    "form_type": form_type,
                    "summary_link": idx_link,
                }
                filings_list.append(filing_data)

            except (ValueError, IndexError):
                continue
    else:
        return [{"error": f"SEC: No recent filings data found for {ticker_symbol}."}]
        
    filings_list.sort(key=lambda x: x.get('filing_date_str', '1900-01-01'), reverse=True)
    return filings_list

@st.cache_data(ttl=6 * 3600)
def fetch_inst_filings(ticker: str) -> List[dict]:
    try:
        df_holders = yf.Ticker(ticker).institutional_holders
        if df_holders is not None and not df_holders.empty:
            for col in ['Shares', '% Out']:
                if col in df_holders.columns:
                    df_holders[col] = pd.to_numeric(df_holders[col], errors='coerce').fillna(0)
            if 'Date Reported' in df_holders.columns:
                df_holders['Date Reported'] = df_holders['Date Reported'].astype(str)
            return df_holders.to_dict("records")
        return [{"error": f"No institutional holder data for {ticker}."}]
    except Exception as e:
        return [{"error": f"yfinance institutional holders fetch failed for {ticker}: {e}"}]

@st.cache_data(ttl=6 * 3600)
def fetch_recommendations(ticker: str) -> pd.DataFrame:
    try:
        recs = yf.Ticker(ticker).recommendations
        if recs is not None and not recs.empty:
            recs.index = pd.to_datetime(recs.index).tz_localize(None)
            last_12_months = datetime.now() - pd.DateOffset(months=12)
            recs = recs[recs.index >= last_12_months].sort_index(ascending=False)
            recs.rename(columns={"Firm": "Firm", "To Grade": "Rating"}, inplace=True, errors='ignore')
            return recs
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

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
        if not valid_news:
            err_msg = news[0].get("error") if news and isinstance(news[0],dict) and "error" in news[0] else "No valid news."
            return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": err_msg}
        content_llm, co_name = [], data.get("ticker_info",{}).get('longName', ticker)
        for item in valid_news[:7]:
            title, pub, desc, cont = item.get('title',''), item.get('publisher',''), item.get('description',''), item.get('content_snippet','')
            snippet = f"Headline: {title}"
            if cont and isinstance(cont,str) and len(cont)>10: snippet += f" | Content: {cont.replace('[+... chars]','').strip()}"
            elif desc and isinstance(desc,str): snippet += f" | Description: {desc.strip()}"
            if pub and pub!='N/A': snippet += f" (Source: {pub} via {item.get('source_api','Unknown')})"
            content_llm.append(snippet)
        if not content_llm: return {"ticker":ticker, "sentiment_score":0.0, "sentiment_signal":"hold", "sentiment_error":"No processable news."}
        prompt = f"Analyze sentiment for {co_name} ({ticker})...Output only number...\n\nNews:\n" + "\n".join(f"- {c}" for c in content_llm)
        score, llm_err = 0.0, None
        try:
            resp = self.client.generate(prompt).strip()
            if resp.startswith("Error:"): llm_err = resp
            else:
                match = re.search(r"([-+]?\d*\.\d+)|([-+]?\d+)", resp)
                if match: score = max(-1.0, min(1.0, float(match.group(0))))
                else: llm_err = f"LLM non-numeric sent.: '{resp[:50]}...'"
        except Exception as e: llm_err = f"LLM sent. call failed: {str(e)[:150]}"
        final_err = llm_err
        if news_err and ("Error" in news_err or "failed" in news_err.lower()): final_err = f"News: {news_err}" + (f" | LLM: {llm_err}" if llm_err else "")
        sig = "buy" if score > 0.25 and not llm_err else ("sell" if score < -0.25 and not llm_err else "hold")
        return {"ticker": ticker, "sentiment_score": score, "sentiment_signal": sig, "sentiment_error": final_err}

class NewsSummaryAgent:
    def __init__(self, client): self.client = client
    def run(self, ticker: str, data: dict) -> dict:
        news, co_name, news_fetch_err = data.get("news",[]), data.get("ticker_info",{}).get('longName',ticker), data.get("news_fetch_status_error")
        if news_fetch_err: return {"ticker":ticker, "news_summary":"Summary skipped due to news fetch issues.", "news_summary_error":news_fetch_err}
        if not news or (isinstance(news[0],dict) and "error" in news[0] and not any("error" not in item for item in news)):
            err = news[0]["error"] if news and isinstance(news[0],dict) and "error" in news[0] else "No news for summary."
            return {"ticker":ticker, "news_summary":"No news for summary.", "news_summary_error":err}
        y_news = [item for item in news if item.get('source_api')=='Yahoo Finance' and "error" not in item][:5]
        n_news = [item for item in news if item.get('source_api')=='NewsAPI.org' and "error" not in item][:5]
        sel_news, y_len, n_len = [], len(y_news), len(n_news)
        for i in range(max(y_len, n_len)):
            if i < y_len: sel_news.append(y_news[i])
            if i < n_len: sel_news.append(n_news[i])
        final_snips, titles = [], set()
        for item in sel_news:
            if len(final_snips) >= 7: break
            title, desc, cont = item.get('title',''), item.get('description',''), item.get('content_snippet','').replace('[+... chars]','').strip()
            if title in titles: continue; titles.add(title)
            text = f"Title: {title}"
            if cont: text += f" | Content: {cont}"
            elif desc: text += f" | Description: {desc}"
            final_snips.append(text)
        if not final_snips: return {"ticker":ticker, "news_summary":"No content for summary.", "news_summary_error":"No articles with content/desc."}
        prompt = f"Concise summary (max 200 words) for {co_name} ({ticker})...\n\nArticles:\n" + "\n".join(f"- {s}" for s in final_snips)
        summary, err_msg = "Could not generate summary.", None
        try:
            resp = self.client.generate(prompt).strip()
            if resp.startswith("Error:"): err_msg = resp
            else: summary = resp
        except Exception as e: err_msg = f"LLM summary call failed: {str(e)[:150]}"
        return {"ticker":ticker, "news_summary":summary, "news_summary_error":err_msg}

class FundamentalsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        s = data.get("ticker_info",{}); mc, fcf, roe, de = s.get("marketCap"), s.get("freeCashflow"), s.get("returnOnEquity"), s.get("debtToEquity")
        mc_c = mc if isinstance(mc,(int,float)) and mc > 0 else 1
        fcf_c = fcf if isinstance(fcf,(int,float)) else 0
        roe_c = roe if isinstance(roe,(int,float)) else 0
        de_c = de if isinstance(de,(int,float)) else 1000
        fcy = fcf_c / mc_c
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
        fcy = (fcf / mc) if fcf is not None and mc is not None and mc != 0 else 0.0
        fp_est = curr_p * (1 + fcy); dcf_sig = "hold"
        if fp_est > curr_p * 1.15: dcf_sig = "buy"
        elif fp_est < curr_p * 0.85: dcf_sig = "sell"
        return {"ticker":ticker, "forward_pe":pe, "relative_pe_signal":rel_sig, "dcf_fair_price":float(fp_est) if pd.notna(fp_est) else np.nan, "dcf_signal":dcf_sig, "valuation_error":None}

class AnalystRatingAgent:
    def run(self, ticker: str, data: dict) -> dict:
        info, hist = data.get("ticker_info",{}), data.get("price_history"); price_v = info.get("currentPrice")
        if price_v is None and hist is not None and not hist.empty: price_v = hist.Close.iloc[-1]
        curr_p = float(price_v) if isinstance(price_v,(int,float)) and price_v > 0 else None
        if curr_p is None: return {"ticker":ticker, "analyst_buy_pct_inferred":0.5, "target_upside":0.0, "yfinance_recommendation":"N/A", "analyst_signal":"hold", "analyst_error":"Current price unavailable."}
        target_v = info.get("targetMeanPrice"); target_m = float(target_v) if isinstance(target_v,(int,float)) else None
        rec = str(info.get("recommendationKey","hold")).lower(); upside = 0.0
        if target_m is not None and curr_p > 0: upside = (target_m / curr_p) -1
        sig = "hold"
        if rec in ["buy","strong_buy"] and upside > 0.10: sig = "buy"
        elif rec == "buy" and upside > 0.05: sig = "buy"
        elif rec in ["sell","strong_sell","underperform"] and upside < -0.05: sig = "sell"
        elif upside > 0.20: sig = "buy"
        elif upside < -0.15: sig = "sell"
        buy_pct = {"strong_buy":0.9, "buy":0.7, "hold":0.5, "underperform":0.3, "sell":0.1}.get(rec,0.5)
        return {"ticker":ticker, "analyst_buy_pct_inferred":float(buy_pct), "target_upside":float(upside), "yfinance_recommendation":rec, "analyst_signal":sig, "analyst_error":None}

class SECFilingAgent:
    def run(self, ticker: str, data: dict) -> dict:
        filings = data.get("sec_all_filings_raw", [])
        if not filings or (isinstance(filings[0], dict) and "error" in filings[0]):
            error_msg = filings[0].get("error") if filings else f"SEC: No raw filings for {ticker}."
            return {
                "ticker": ticker, "sec_filings_signal": "hold", "sec_filings_error": error_msg,
                "sec_recent_form4_transactions": [], "sec_other_recent_filings": []
            }
        
        form4_filings = [f for f in filings if f.get("form_type") == '4']
        other_filings = [f for f in filings if f.get("form_type") != '4']

        # Signal logic can be enhanced here, for now it's a placeholder
        insider_signal = "hold"
        if any(f.get('form_type') == '4' for f in form4_filings): # Basic signal if any insider activity
            insider_signal = "hold" 

        return {
            "ticker": ticker,
            "sec_filings_signal": insider_signal,
            "sec_filings_error": None,
            "sec_recent_form4_transactions": form4_filings[:10],
            "sec_other_recent_filings": other_filings[:10]
        }

class InstitutionalHoldingsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        holdings = data.get("institutional_holdings", [])
        if not holdings or (isinstance(holdings[0], dict) and "error" in holdings[0]):
            error_msg = holdings[0].get("error") if holdings else "No institutional holdings data."
            return {"ticker":ticker, "inst_num_holders":0, "inst_total_shares_held":0, "inst_total_pct_out":0.0, "inst_holdings_signal":"hold", "inst_holdings_error":error_msg, "inst_top_holders":[]}
        
        valid_holders = [h for h in holdings if "error" not in h]
        num_holders = len(valid_holders)
        total_shares = sum(h.get('Shares', 0) for h in valid_holders)
        total_pct = sum(h.get('% Out', 0.0) for h in valid_holders)
        top_holders = sorted(valid_holders, key=lambda x: x.get('Shares', 0), reverse=True)[:10]

        signal = "hold"
        if total_pct > 0.50:
            signal = "buy"
        elif total_pct < 0.05 and num_holders > 0:
            signal = "sell"
            
        return {"ticker":ticker, "inst_num_holders":num_holders, "inst_total_shares_held":int(total_s), "inst_total_pct_out":float(total_pct), "inst_holdings_signal":signal, "inst_holdings_error":None, "inst_top_holders":top_holders}

class PortfolioAgent:
    def run(self, ticker: str, signals: List[dict], agent_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        weights = agent_weights if agent_weights is not None else CONFIG["AGENT_WEIGHTS"]
        total_score, sum_w, aggregated_signals = 0.0, 0.0, {}
        for s_dict in signals:
            if isinstance(s_dict, dict): aggregated_signals.update(s_dict)
        
        signal_map = {
            "price_signal": "price", "momentum_signal": "momentum", "volatility_signal": "volatility",
            "sentiment_signal": "sentiment", "fund_signal": "fund", "dcf_signal": "valuation_dcf",
            "relative_pe_signal": "valuation_pe", "sec_filings_signal": "sec_filings",
            "inst_holdings_signal": "inst_holdings", "analyst_signal": "analyst",
        }
        
        for signal_key, weight_key in signal_map.items():
            signal_value = aggregated_signals.get(signal_key)
            weight = weights.get(weight_key, 0.0)
            if signal_value and weight > 0 and signal_value in ["buy", "hold", "sell"]:
                raw_score = {"buy": 1, "hold": 0, "sell": -1}.get(signal_value, 0)
                total_score += raw_score * weight
                sum_w += weight
        
        composite_score = (total_score / sum_w) if sum_w > 0 else 0.0
        decision = "buy" if composite_score > 0.15 else ("sell" if composite_score < -0.15 else "hold")
        
        return {"ticker": ticker, "composite_score": composite_score, "final_decision": decision}

class AITraderAgent:
    def __init__(self, llm_client: Optional[ModelClient], stock_universe: Dict[str, List[str]]):
        self.llm_client = llm_client
        self.stock_universe = stock_universe

    def _generate_trade_reason(self, ticker: str, decision: str, analysis: dict) -> str:
        if not self.llm_client:
            return "Automated trade based on composite score."
        # ... (rest of the function is unchanged)
    
    def _is_safe(self, analysis: dict) -> bool:
        # ... (unchanged)
        return False
        
    def run(self, portfolio_state: dict, analysis_results: dict) -> List[dict]:
        # ... (unchanged)
        return []


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
            err_msg = f"Core ticker info unavailable. Invalid/delisted ticker for {t}."
            results[t] = {"error": err_msg, "ticker": t, "final_decision":"error", "composite_score":0}; continue
        
        current_price_for_ticker = ticker_info.get("currentPrice") or price_history_full["Close"].iloc[-1]
        
        data_bundle = {
            "price_history": price_history_full,
            "ticker_info": ticker_info,
            "news": fetch_enriched_news(t, ticker_info) if configs.get("use_sentiment") else [],
            "sec_all_filings_raw": fetch_all_sec_filings(t) if configs.get("use_filings") else [],
            "institutional_holdings": fetch_inst_filings(t) if configs.get("use_filings") else [],
            "recommendations": fetch_recommendations(t)
        }
        
        agents_to_run = [PriceAgent(), MomentumAgent(), VolatilityAgent(), FundamentalsAgent(), ValuationAgent(), AnalystRatingAgent()]
        if configs.get("use_sentiment") and llm_client:
            agents_to_run.extend([SentimentAgent(llm_client), NewsSummaryAgent(llm_client)])
        if configs.get("use_filings"):
            agents_to_run.append(SECFilingAgent())
            agents_to_run.append(InstitutionalHoldingsAgent())

        agent_results_list = []
        for agent in agents_to_run:
            try:
                if isinstance(agent, (PriceAgent, MomentumAgent, VolatilityAgent)):
                    res = agent.run(t, data_bundle, price_history_full) if isinstance(agent, VolatilityAgent) else agent.run(t, price_history_full)
                else:
                    res = agent.run(t, data_bundle)
                agent_results_list.append(res)
            except Exception as e:
                st.warning(f"Error running {agent.__class__.__name__} for {t}: {e}")
        
        portfolio_agent = PortfolioAgent()
        final_decision = portfolio_agent.run(t, agent_results_list)

        final_bundle = {
            "ticker": t,
            "current_price_display": current_price_for_ticker
        }
        final_bundle.update(data_bundle)
        for res in agent_results_list:
            final_bundle.update(res)
        final_bundle.update(final_decision)
        results[t] = final_bundle

    progress_bar.empty()
    return results

def display_detailed_analysis(res_detail):
    ticker = res_detail.get("ticker", "N/A")
    ticker_info = res_detail.get("ticker_info", {})
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
        else:
            st.warning("Price chart data not available.")
        # ... (Placeholder for the rest of this tab's content)

    with tabs[1]:
        st.subheader(f"Fundamental Overview: {ticker_info.get('longName', '')}")
        st.caption(f"**Sector:** {ticker_info.get('sector', 'N/A')} | **Industry:** {ticker_info.get('industry', 'N/A')}")
        if ticker_info.get('longBusinessSummary'):
            with st.popover("Show Business Summary"):
                st.markdown(ticker_info.get('longBusinessSummary'))
        # ... (Placeholder for the rest of this tab's content)

    with tabs[2]:
        st.subheader("Analyst & Fair Value Analysis")
        # ... (Placeholder for the rest of this tab's content)
        st.markdown("---")
        st.subheader("Recent Analyst Rating Changes (1-Year)")
        recommendations_df = res_detail.get('recommendations')
        if recommendations_df is not None and not recommendations_df.empty:
            st.dataframe(recommendations_df.head(10), use_container_width=True)
        else:
            st.info("No recent analyst rating changes found.")

    with tabs[3]:
        st.subheader("News, Filings & Ownership")
        st.markdown("---")
        st.subheader("All Recent Company Filings (1-Year)")
        all_filings = res_detail.get('sec_all_filings_raw', [])
        if all_filings and isinstance(all_filings, list) and not (isinstance(all_filings[0], dict) and all_filings[0].get("error")):
            df_all_filings = pd.DataFrame(all_filings)
            df_display = df_all_filings.rename(columns={"filing_date_str": "Filing Date", "form_type": "Form Type", "summary_link": "SEC Link"})
            
            cols_to_show = ["Filing Date", "Form Type", "SEC Link"]
            final_cols = [col for col in cols_to_show if col in df_display.columns]
            
            if final_cols:
                st.dataframe(df_display[final_cols], use_container_width=True, hide_index=True,
                             column_config={"SEC Link": st.column_config.LinkColumn("🔗 Link", validate=True)})
            else:
                st.info("Found filings, but could not display them due to missing column data.")
        else:
            st.info("No major company filings found in the last year.")
            
        st.markdown("---")
        st.subheader("Top Institutional Holdings")
        holders = res_detail.get('institutional_holdings', [])
        if holders and isinstance(holders, list) and not (isinstance(holders[0], dict) and holders[0].get("error")):
            df_holders = pd.DataFrame(holders)
            df_holders_display = df_holders.rename(columns={"% Out": "% of Outstanding", "Date Reported": "As Of Date"})
            
            column_config = {"Shares": st.column_config.NumberColumn(format="%.0f")}
            if "% of Outstanding" in df_holders_display.columns:
                max_val = df_holders_display["% of Outstanding"].max()
                if pd.notna(max_val):
                    column_config["% of Outstanding"] = st.column_config.ProgressColumn(format="%.2f%%", min_value=0, max_value=max(0.10, max_val))
            
            cols_to_display = ["Holder", "Shares", "% of Outstanding", "As Of Date"]
            available_cols = [col for col in cols_to_display if col in df_holders_display.columns]
            st.dataframe(df_holders_display[available_cols], hide_index=True, use_container_width=True, column_config=column_config)
        else:
            st.info("No institutional holder data available.")

    with tabs[4]:
        st.subheader("All Agent Signals at a Glance")
        # ... (Placeholder for the rest of this tab's content)


# --- Main UI Logic ---
st.title("🚀 AI Hedge Fund Simulator")
st.header("⚙️ Configuration")
with st.container(border=True):
    app_mode = st.radio("Select Mode:", ["Live Analysis", "Backtesting", "Virtual Trading"], horizontal=True)
    st.markdown("---")

    if app_mode == "Live Analysis":
        st.subheader("Live Analysis Settings")
        tickers_in_live = st.text_input("Tickers (comma-separated):", "AAPL,MSFT,GOOG,CRWD")
        use_filings = st.checkbox("SEC & Inst. Filings", value=True)
        use_sentiment = st.checkbox("News & Sentiment", value=True)
        
        if st.button("🚀 Run Live Analysis", use_container_width=True, type="primary"):
            live_tickers = [t.strip().upper() for t in tickers_in_live.split(",") if t.strip()]
            if live_tickers:
                with st.spinner("⏳ Processing live analysis..."):
                    st.session_state.live_output = run_live_analysis(live_tickers, None, {"use_filings": use_filings, "use_sentiment": use_sentiment})
                    st.session_state.live_analysis_triggered = True
                    st.rerun()

st.markdown("---")
if st.session_state.get('live_analysis_triggered'):
    st.header("📊 Live Analysis Summary")
    for ticker, res_detail in st.session_state.live_output.items():
        if res_detail and not res_detail.get("error"):
            with st.expander(f"🔍 Detailed Analysis for {ticker}", expanded=True):
                display_detailed_analysis(res_detail)
        else:
            st.error(f"Could not retrieve data for {ticker}: {res_detail.get('error', 'Unknown error')}")

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

# ... (All Agent classes are unchanged and should be pasted here)

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
        
        final_analysis = {
            "ticker": t,
            "current_price_display": current_price_for_ticker,
            "final_decision": "HOLD",
            "composite_score": 0.0
        }
        final_analysis.update(data_bundle)
        results[t] = final_analysis
        
    progress_bar.empty()
    return results

def display_detailed_analysis(res_detail):
    ticker = res_detail.get("ticker", "N/A")
    ticker_info = res_detail.get("ticker_info", {})
    tab_titles = ["📈 Chart & Core", "📊 Fundamentals", "💰 Analyst & Fair Value", "📰 News & Filings", "⚙️ All Signals"]
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        st.subheader("Price Performance & Technical Signals")
        price_hist_chart = fetch_price_history(ticker, period="1y")
        if not price_hist_chart.empty:
            st.line_chart(price_hist_chart["Close"], use_container_width=True, color="#0072F0")
        else:
            st.warning("Price chart data not available.")

    with tabs[1]:
        st.subheader(f"Fundamental Overview: {ticker_info.get('longName', '')}")
        st.caption(f"**Sector:** {ticker_info.get('sector', 'N/A')} | **Industry:** {ticker_info.get('industry', 'N/A')}")
        if ticker_info.get('longBusinessSummary'):
            with st.popover("Show Business Summary"):
                st.markdown(ticker_info.get('longBusinessSummary'))

    with tabs[2]:
        st.subheader("Analyst & Fair Value Analysis")
        st.subheader("Recent Analyst Rating Changes (1-Year)")
        recommendations_df = res_detail.get('recommendations')
        if recommendations_df is not None and not recommendations_df.empty:
            st.dataframe(recommendations_df.head(10), use_container_width=True)
        else:
            st.info("No recent analyst rating changes found.")

    with tabs[3]:
        st.subheader("News, Filings & Ownership")
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
            st.info("No major company filings found in the last year.")
        
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
                    st.session_state.live_output = run_live_analysis(None, None, {"use_filings": use_filings, "use_sentiment": use_sentiment, "tickers": live_tickers})
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

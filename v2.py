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

# SEC EDGAR User-Agent
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

# Initialize flags for running analysis
if 'live_analysis_triggered' not in st.session_state:
    st.session_state.live_analysis_triggered = False
if 'backtest_triggered' not in st.session_state:
    st.session_state.backtest_triggered = False

# --------------------------------
# Data Fetchers (No changes)
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
    if not cik:
        try:
            headers = {'User-Agent': SEC_USER_AGENT}
            lookup_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker_symbol.upper()}&owner=exclude&count=10"
            response = requests.get(lookup_url, headers=headers, timeout=10); response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            cik_anchor = soup.find('a', href=re.compile(r"CIK=(\d{10})"))
            if cik_anchor:
                match = re.search(r"CIK=(\d{10})", cik_anchor['href'])
                if match: cik = match.group(1)
            if not cik:
                cik_text_match = re.search(r"CIK:\s*(\d{10})", soup.get_text(), re.IGNORECASE)
                if cik_text_match: cik = cik_text_match.group(1)
        except Exception: pass
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
            metadata = []
            for i in range(len(forms)):
                try:
                    if datetime.strptime(dates[i], '%Y-%m-%d').replace(tzinfo=timezone.utc) >= date_limit:
                        metadata.append({"form_type": forms[i], "filing_date_str": dates[i], "accession_number": acc_nos[i], "primary_document": docs[i]})
                except (ValueError, IndexError): continue
            xml_fetches, max_xml, max_other = 0, 20, 15
            for info in metadata:
                form, date_str, acc_no, doc_name = info["form_type"], info["filing_date_str"], info["accession_number"], info["primary_document"]
                acc_no_dashless = acc_no.replace('-', '')
                idx_link = f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{acc_no_dashless}/{acc_no}-index.html"
                if form == '4' and doc_name.lower().endswith(('.xml', '.xsd')):
                    if xml_fetches >= max_xml: continue
                    xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{acc_no_dashless}/{doc_name}"
                    try:
                        filing_resp = requests.get(xml_url, headers=headers, timeout=10)
                        if filing_resp.status_code != 200: continue
                        soup_xml = BeautifulSoup(filing_resp.content, 'xml'); xml_fetches += 1
                        owner_tag = soup_xml.find('reportingOwner'); owner_name, owner_rel = "N/A", "N/A"
                        if owner_tag:
                            if owner_tag.find('reportingOwnerId') and owner_tag.find('reportingOwnerId').find('rptOwnerName'): owner_name = owner_tag.find('reportingOwnerId').find('rptOwnerName').text.strip()
                            rel_tag = owner_tag.find('reportingOwnerRelationship'); rels = []
                            if rel_tag:
                                if rel_tag.find('isDirector') and rel_tag.find('isDirector').text in ['1','true']: rels.append("Director")
                                if rel_tag.find('isOfficer') and rel_tag.find('isOfficer').text in ['1','true']: rels.append(f"Officer ({rel_tag.find('officerTitle').text.strip() if rel_tag.find('officerTitle') and rel_tag.find('officerTitle').text else ''})")
                                if rel_tag.find('isTenPercentOwner') and rel_tag.find('isTenPercentOwner').text in ['1','true']: rels.append(">10% Owner")
                                if rels: owner_rel = ", ".join(filter(None, rels))
                        for table_name in ['nonDerivativeTable', 'derivativeTable']:
                            table = soup_xml.find(table_name)
                            if not table: continue
                            for tx in table.find_all(['nonDerivativeTransaction', 'derivativeTransaction']):
                                tx_date_tag, tx_code_tag = tx.find('transactionDate'), tx.find('transactionCoding')
                                tx_date = tx_date_tag.find('value').text.strip() if tx_date_tag and tx_date_tag.find('value') else "N/A"
                                tx_code = tx_code_tag.find('transactionCode').text.strip().upper() if tx_code_tag and tx_code_tag.find('transactionCode') else "N/A"
                                shares, price = 0.0, None
                                amounts = tx.find('transactionAmounts')
                                if amounts and amounts.find('transactionShares') and amounts.find('transactionShares').find('value'):
                                    try: shares = float(amounts.find('transactionShares').find('value').text.strip())
                                    except ValueError: continue
                                price_node = tx.find('transactionPricePerShare')
                                if price_node and price_node.find('value'):
                                    try: price = float(price_node.find('value').text.strip())
                                    except ValueError: price = None
                                ad_node = tx.find('transactionAcquiredDisposedCode'); ad_code = ad_node.find('value').text.strip().upper() if ad_node and ad_node.find('value') else "N/A"
                                if shares != 0: filings_list.append({"is_form4_transaction": True, "ticker": ticker_symbol, "filing_date": date_str, "transaction_date": tx_date, "reporting_owner": owner_name, "owner_relationship": owner_rel, "transaction_code": tx_code, "acq_disp_code": ad_code, "shares": shares, "price_per_share": price, "link_to_filing": idx_link})
                    except: pass
                elif len([f for f in filings_list if not f.get("is_form4_transaction")]) < max_other:
                    filings_list.append({"is_form4_transaction": False, "ticker": ticker_symbol, "filing_date": date_str, "form_type": form, "document_link": f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{acc_no_dashless}/{doc_name}", "summary_link": idx_link})
            if not filings_list and xml_fetches > 0: return [{"error": f"SEC: {xml_fetches} Form 4s for {ticker_symbol}, but no tx parsed."}]
            if not filings_list: return [{"error": f"SEC: No relevant filings for {ticker_symbol} (CIK:{cik_padded})."}]
        else: return [{"error": f"SEC: No recent filings data for {ticker_symbol} (CIK:{cik_padded})."}]
    except requests.exceptions.HTTPError as e: return [{"error": f"SEC HTTP error ({ticker_symbol}, CIK:{cik_padded}): {e}"}]
    except requests.exceptions.RequestException as e: return [{"error": f"SEC Request error ({ticker_symbol}, CIK:{cik_padded}): {e}"}]
    except Exception as e: return [{"error": f"SEC Unexpected error ({ticker_symbol}, CIK:{cik_padded}): {e}"}]
    filings_list.sort(key=lambda x: x.get('filing_date', '1900-01-01'), reverse=True); return filings_list
# --- REPLACE your old fetch_all_sec_filings function with this one ---

@st.cache_data(ttl=3600)
def fetch_sec_filings_from_search_api(search_query: str, lookback_days: int = 365) -> list[dict]:
    """
    Fetches ALL recent SEC filings using the new EDGAR search API.
    This version uses a standard browser User-Agent to avoid 403 errors.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36',
        'Accept': 'application/json',
    }
    api_url = f"https://efts.sec.gov/LATEST/search-index"
    payload = {"q": search_query.lower(), "from": 0, "size": 100, "sort": [{"filed_date": "desc"}]}
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        results = response.json()
        if not results or not results.get('hits', {}).get('hits'): return []
        
        filings_list = []
        date_limit = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        for hit in results['hits']['hits']:
            source = hit.get('_source', {})
            try:
                if datetime.fromisoformat(source.get('file_date')) < date_limit: continue
                filings_list.append({
                    "filing_date": source.get('file_date', 'N/A')[:10],
                    "form_type": source.get('form', 'N/A'),
                    "link_to_filing": f"https://www.sec.gov/edgar/browse/?CIK={source.get('ciks')[0]}"
                })
            except (ValueError, TypeError, KeyError): continue
        return sorted(filings_list, key=lambda x: x.get('filing_date'), reverse=True)
    except Exception as e:
        return [{"error": f"SEC Search API request failed for '{search_query}': {e}"}]
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
        pattern = re.compile(r"As of (?P<date>[\d]{4}-[\d]{2}-[\d]{2}), the Fair Value of .*?\(.*?" + re.escape(ticker.upper()) + r".*?\) is (?P<fair_value>[\d\.]+) USD\.?" + r"(?:.*?With the current market price of (?P<market_price>[\d\.]+) USD, the upside of .*? is (?P<upside_percent>[-+]?\d+\.?\d*)%\.?)?")
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
            return "".join(c.choices[0].delta.content for c in stream if c.choices and c.choices[0].delta and c.choices[0].delta.content)
        except Exception as e: raise Exception(f"LLM Error ({self.provider}, {self.model_name}): {e}")

class PriceAgent:
    def run(self, ticker: str, price_data_slice: pd.DataFrame) -> dict:
        # Minimum data required for SMA200 and Bollinger Bands (approx. 20 periods)
        # RSI needs 14 periods, SMA50 needs 50. SMA200 needs 200. Bollinger Bands typically use 20 periods.
        # Let's ensure enough data for all, so 200 is a good base for SMA200, and 20 for Bollinger.
        # Ensure at least 200 periods for all indicators to be meaningful.
        required_data_points = 200 # For SMA200, which is the longest period currently.

        if price_data_slice.empty or len(price_data_slice) < required_data_points:
            return {
                "ticker": ticker,
                "price_signal": "hold",
                "sma50": np.nan,
                "sma200": np.nan,
                "rsi14": np.nan,
                "bb_upper": np.nan,
                "bb_lower": np.nan,
                "bb_signal": "hold",
                "price_confidence_score": 0.0,
                "price_error": "Not enough data for comprehensive analysis."
            }

        df = price_data_slice.copy()
        
        # --- Calculate Indicators ---
        # Simple Moving Averages
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["SMA200"] = df["Close"].rolling(200).mean()

        # Relative Strength Index (RSI)
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        # Handle division by zero for RS by replacing 0 with np.nan for robust calculation
        rs = gain / loss.replace(0, np.nan)
        df["RSI14"] = 100 - (100 / (1 + rs))

        # Bollinger Bands (typically 20-period SMA, 2 standard deviations)
        bb_period = 20
        bb_std_dev = 2
        df["BB_SMA"] = df["Close"].rolling(bb_period).mean()
        df["BB_STD"] = df["Close"].rolling(bb_period).std()
        df["BB_Upper"] = df["BB_SMA"] + (df["BB_STD"] * bb_std_dev)
        df["BB_Lower"] = df["BB_SMA"] - (df["BB_STD"] * bb_std_dev)

        # Get latest valid data point
        latest = df.iloc[-1]
        
        # Initialize signals and score
        signal = "hold"
        confidence_score = 0.0
        bb_signal = "hold"

        # --- Apply Signal Logic ---
        
        # Check for NaN values in latest indicators before making decisions
        if pd.isna(latest.SMA50) or pd.isna(latest.SMA200) or pd.isna(latest.RSI14) or \
           pd.isna(latest.BB_Upper) or pd.isna(latest.BB_Lower):
            return {
                "ticker": ticker,
                "price_signal": "hold", # Fallback to hold if indicators are NaN at latest point
                "sma50": float(latest.SMA50) if pd.notna(latest.SMA50) else np.nan,
                "sma200": float(latest.SMA200) if pd.notna(latest.SMA200) else np.nan,
                "rsi14": float(latest.RSI14) if pd.notna(latest.RSI14) else np.nan,
                "bb_upper": float(latest.BB_Upper) if pd.notna(latest.BB_Upper) else np.nan,
                "bb_lower": float(latest.BB_Lower) if pd.notna(latest.BB_Lower) else np.nan,
                "bb_signal": "hold",
                "price_confidence_score": 0.0,
                "price_error": "Some key indicators are NaN at the latest data point."
            }

        current_close = latest.Close

        # 1. SMA Crossover + Price Confirmation
        # Golden Cross (Bullish)
        if latest.SMA50 > latest.SMA200 and current_close > latest.SMA50:
            # Check if recent cross occurred for stronger signal (optional, requires looking back)
            # Example: Was SMA50 below SMA200 5 days ago, and now it's above?
            if len(df) >= 205 and df["SMA50"].iloc[-5] < df["SMA200"].iloc[-5]: # Check 5 days ago for cross
                 signal = "buy"
                 confidence_score += 0.4 # Higher confidence for confirmed cross and price above SMAs
            else:
                 signal = "buy" # Weaker buy if no recent cross but alignment
                 confidence_score += 0.2
        # Death Cross (Bearish)
        elif latest.SMA50 < latest.SMA200 and current_close < latest.SMA50:
            if len(df) >= 205 and df["SMA50"].iloc[-5] > df["SMA200"].iloc[-5]: # Check 5 days ago for cross
                signal = "sell"
                confidence_score -= 0.4
            else:
                signal = "sell"
                confidence_score -= 0.2

        # 2. RSI Signal (Secondary Confirmation/Overbought/Oversold)
        if latest.RSI14 < 30: # Oversold
            if signal == "buy": confidence_score += 0.2 # RSI confirms buy
            elif signal == "hold":
                signal = "buy" # RSI alone triggers a buy (reversal play)
                confidence_score += 0.1
        elif latest.RSI14 > 70: # Overbought
            if signal == "sell": confidence_score -= 0.2 # RSI confirms sell
            elif signal == "hold":
                signal = "sell" # RSI alone triggers a sell (reversal play)
                confidence_score -= 0.1

        # 3. Bollinger Bands Signal
        # Price crossing below lower band (often buy opportunity after pullback)
        if current_close < latest.BB_Lower:
            bb_signal = "buy"
            if signal == "buy": confidence_score += 0.1 # Adds to existing buy
            elif signal == "hold":
                signal = "buy" # BB alone triggers buy
                confidence_score += 0.05
        # Price crossing above upper band (often sell opportunity, or strong momentum)
        elif current_close > latest.BB_Upper:
            bb_signal = "sell"
            if signal == "sell": confidence_score -= 0.1 # Adds to existing sell
            elif signal == "hold":
                signal = "sell" # BB alone triggers sell
                confidence_score -= 0.05
        
        # If signal is still 'hold', but RSI is extreme (and not yet acted upon)
        if signal == "hold":
            if latest.RSI14 < 40 and latest.RSI14 > 30: # Approaching oversold
                confidence_score += 0.05
            elif latest.RSI14 > 60 and latest.RSI14 < 70: # Approaching overbought
                confidence_score -= 0.05

        # Final decision based on refined logic and confidence
        if confidence_score > 0.3:
            final_price_signal = "buy"
        elif confidence_score < -0.3:
            final_price_signal = "sell"
        else:
            final_price_signal = "hold"
            
        # Cap confidence score between -1.0 and 1.0 (or whatever range makes sense)
        confidence_score = max(-1.0, min(1.0, confidence_score))

        return {
            "ticker": ticker,
            "sma50": float(latest.SMA50),
            "sma200": float(latest.SMA200),
            "rsi14": float(latest.RSI14),
            "bb_upper": float(latest.BB_Upper) if pd.notna(latest.BB_Upper) else np.nan,
            "bb_lower": float(latest.BB_Lower) if pd.notna(latest.BB_Lower) else np.nan,
            "bb_signal": bb_signal, # Individual BB signal
            "price_signal": final_price_signal, # Overall aggregated signal
            "price_confidence_score": float(confidence_score), # New: Quantified confidence
            "price_error": None # Clear error if successful
        }

class MomentumAgent:
    def run(self, ticker: str, price_data_slice: pd.DataFrame) -> dict:
        # Minimum data required for 12-month momentum (252 trading days for a year)
        # We need at least 253 points to calculate a 252-period shift (current price + 252 past prices)
        required_data_points = 253

        if price_data_slice.empty or len(price_data_slice) < required_data_points:
            return {
                "ticker": ticker,
                "momentum_signal": "hold",
                "momentum_1m": np.nan,
                "momentum_12m": np.nan,
                "momentum_confidence_score": 0.0, # Add a confidence score initialized to 0
                "momentum_error": "Not enough data for 1-year and 1-month momentum."
            }

        df = price_data_slice.copy()

        # Ensure 'Close' column exists and is numeric
        if 'Close' not in df.columns or not pd.api.types.is_numeric_dtype(df['Close']):
            return {
                "ticker": ticker,
                "momentum_signal": "hold",
                "momentum_1m": np.nan,
                "momentum_12m": np.nan,
                "momentum_confidence_score": 0.0,
                "momentum_error": "Price data is missing 'Close' column or not numeric."
            }

        # Calculate momentum for 1 month (approx. 21 trading days) and 12 months (approx. 252 trading days)
        # Using .iloc[-1] for current price and .iloc[-22] for 1-month ago (21 trading days prior)
        # and .iloc[-253] for 12-month ago (252 trading days prior)
        # This assumes the index is ordered by time, and there are no missing dates that would break iloc indexing.
        P_t = df["Close"].iloc[-1]

        # Use .shift() and .iloc[-1] to get values from specific past *trading* days,
        # and then dropna() to ensure valid numerical prices for calculation.
        # This handles cases where data might be sparse at the beginning of the slice.
        # We fetch enough initial data to allow for shifting.
        
        # Shifted prices. Use .dropna() to ensure previous_price is a scalar and not NaN if data is insufficient for that specific shift.
        # If the shift results in NaN, the subsequent pd.notna(P_1m) will handle it.
        P_1m_series = df["Close"].shift(21)
        P_12m_series = df["Close"].shift(252)

        # Get the latest shifted values. If the shift went beyond available data, these will be NaN.
        P_1m = P_1m_series.iloc[-1]
        P_12m = P_12m_series.iloc[-1]

        # Calculate momentum percentages
        # Ensure division by zero is explicitly handled or avoided for P_1m, P_12m
        m1 = ((P_t / P_1m) - 1) if pd.notna(P_1m) and P_1m != 0 else np.nan
        m12 = ((P_t / P_12m) - 1) if pd.notna(P_12m) and P_12m != 0 else np.nan

        signal = "hold"
        confidence_score = 0.0 # Initial confidence score

        # Define momentum thresholds (can be optimized via backtesting)
        STRONG_POSITIVE_MOMENTUM_THRESHOLD = 0.10 # 10% gain over period
        MODERATE_POSITIVE_MOMENTUM_THRESHOLD = 0.03 # 3% gain over period
        STRONG_NEGATIVE_MOMENTUM_THRESHOLD = -0.10 # 10% loss over period
        MODERATE_NEGATIVE_MOMENTUM_THRESHOLD = -0.03 # 3% loss over period

        # Signal Logic: Combination of short-term and long-term momentum
        if pd.notna(m1) and pd.notna(m12):
            # Strong Buy: Both short and long-term strong positive momentum
            if m12 > STRONG_POSITIVE_MOMENTUM_THRESHOLD and m1 > MODERATE_POSITIVE_MOMENTUM_THRESHOLD:
                signal = "buy"
                confidence_score = 0.8 # High confidence
            # Moderate Buy: Long-term positive, short-term positive (but maybe not strong)
            elif m12 > MODERATE_POSITIVE_MOMENTUM_THRESHOLD and m1 > 0:
                signal = "buy"
                confidence_score = 0.5 # Moderate confidence
            # Strong Sell: Both short and long-term strong negative momentum
            elif m12 < STRONG_NEGATIVE_MOMENTUM_THRESHOLD and m1 < MODERATE_NEGATIVE_MOMENTUM_THRESHOLD:
                signal = "sell"
                confidence_score = -0.8 # High negative confidence
            # Moderate Sell: Long-term negative, short-term negative (but maybe not strong)
            elif m12 < MODERATE_NEGATIVE_MOMENTUM_THRESHOLD and m1 < 0:
                signal = "sell"
                confidence_score = -0.5 # Moderate negative confidence
            else:
                signal = "hold"
                confidence_score = 0.0 # Neutral

        # Adjust confidence based on how far from 0.0 the momentum values are
        # This gives a continuous score, not just discrete levels
        if pd.notna(m1) and pd.notna(m12):
            # Combined momentum score (could be weighted or simple average)
            # Normalize to roughly -1 to 1 range for consistency with other agents
            raw_combined_momentum = (m1 + m12) / 2 # Simple average, adjust if needed
            
            # Simple linear scaling for confidence based on raw_combined_momentum
            # Example: -0.1 to 0.1 range for raw_combined_momentum maps to -0.5 to 0.5 confidence
            # Adjust scaling factor (e.g., 5.0) based on desired sensitivity
            scaled_confidence = raw_combined_momentum * 5.0 
            confidence_score = max(-1.0, min(1.0, scaled_confidence)) # Clamp between -1 and 1

            # Override discrete signal based on clamped confidence
            if confidence_score > 0.3: # Threshold for a buy signal based on confidence
                signal = "buy"
            elif confidence_score < -0.3: # Threshold for a sell signal based on confidence
                signal = "sell"
            else:
                signal = "hold"


        return {
            "ticker": ticker,
            "momentum_1m": float(m1) if pd.notna(m1) else np.nan,
            "momentum_12m": float(m12) if pd.notna(m12) else np.nan,
            "momentum_signal": signal,
            "momentum_confidence_score": float(confidence_score), # New: Quantified confidence
            "momentum_error": None # Clear error if successful
        }

class VolatilityAgent:
    def run(self, ticker: str, data: dict, price_data_slice: pd.DataFrame = None) -> dict:
        beta_val = data.get("ticker_info", {}).get("beta")
        # Ensure beta is a float; default to 1.0 (market-like) if unavailable or invalid
        beta = float(beta_val) if isinstance(beta_val, (int, float)) else 1.0

        ann_vol = np.nan
        vol_weight = 0.0 # Will be higher for lower volatility
        volatility_signal = "hold"
        volatility_confidence_score = 0.0
        volatility_error = None

        # --- Calculate Annualized Historical Volatility ---
        if price_data_slice is not None and not price_data_slice.empty and len(price_data_slice) > 1:
            # Ensure 'Close' column is present and numeric
            if 'Close' not in price_data_slice.columns or not pd.api.types.is_numeric_dtype(price_data_slice['Close']):
                volatility_error = "Price data is missing 'Close' column or not numeric for volatility calculation."
            else:
                # Calculate daily log returns
                ret = np.log(price_data_slice.Close / price_data_slice.Close.shift(1)).dropna()

                if not ret.empty:
                    daily_std = ret.std()
                    if daily_std > 0:
                        ann_vol = float(daily_std * np.sqrt(252)) # Annualized volatility
                        vol_weight = float(1 / ann_vol) # Inverse volatility (higher for lower vol)
                    else:
                        volatility_error = "Daily returns standard deviation is zero (no price movement)."
                else:
                    volatility_error = "Not enough valid returns to calculate historical volatility."
        else:
            volatility_error = "Not enough price data for historical volatility calculation."


        # --- Generate Volatility Signal and Confidence Score ---

        # Base signal from Beta
        # Lower beta is generally 'safer' -> buy. Higher beta is 'riskier' -> sell.
        if beta > 1.2: # More volatile than market
            volatility_signal = "sell" # Risk-off for high beta
            volatility_confidence_score -= (beta - 1.2) * 0.5 # Score decreases as beta increases
        elif beta < 0.8: # Less volatile than market
            volatility_signal = "buy" # Risk-on for low beta
            volatility_confidence_score += (0.8 - beta) * 0.5 # Score increases as beta decreases
        else:
            volatility_signal = "hold" # Market-like volatility
            # confidence_score remains 0 from beta for hold range

        # Incorporate Annualized Volatility into confidence
        # Typical interpretation: high volatility is risky (negative signal), low volatility is stable (positive signal)
        if pd.notna(ann_vol):
            # Define thresholds for what's considered high/low volatility (these are examples)
            HIGH_VOL_THRESHOLD = 0.30 # 30% annualized volatility
            LOW_VOL_THRESHOLD = 0.15  # 15% annualized volatility

            if ann_vol > HIGH_VOL_THRESHOLD:
                # Strongly negative impact on confidence for high volatility
                volatility_confidence_score -= (ann_vol - HIGH_VOL_THRESHOLD) * 1.0 # Scale by 1.0
            elif ann_vol < LOW_VOL_THRESHOLD:
                # Positive impact on confidence for low volatility
                volatility_confidence_score += (LOW_VOL_THRESHOLD - ann_vol) * 1.0 # Scale by 1.0

        # Clamp confidence score to the range [-1.0, 1.0]
        volatility_confidence_score = max(-1.0, min(1.0, volatility_confidence_score))

        # Re-evaluate signal based on combined confidence score
        if volatility_confidence_score > 0.2: # Tunable threshold for a "buy" signal
            volatility_signal = "buy"
        elif volatility_confidence_score < -0.2: # Tunable threshold for a "sell" signal
            volatility_signal = "sell"
        else:
            volatility_signal = "hold"


        return {
            "ticker": ticker,
            "beta": float(beta),
            "annual_vol": float(ann_vol) if pd.notna(ann_vol) else np.nan,
            "vol_weight": float(vol_weight) if pd.notna(vol_weight) else np.nan,
            "volatility_signal": volatility_signal, # Overall combined signal
            "volatility_confidence_score": float(volatility_confidence_score), # New: Quantified confidence
            "volatility_error": volatility_error
        }

class SentimentAgent:
    def __init__(self, client):
        self.client = client

    def run(self, ticker: str, data: dict) -> dict:
        news, news_err = data.get("news", []), data.get("news_fetch_status_error")

        if news_err:
            return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": news_err}

        # Filter out news items that are explicitly marked with an "error"
        valid_news = [item for item in news if isinstance(item, dict) and "error" not in item]

        if not valid_news:
            # More precise error message if the entire news list was empty or only contained errors
            err_msg = news[0].get("error") if news and isinstance(news[0], dict) and "error" in news[0] else "No valid news articles found for sentiment analysis."
            return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": err_msg}

        content_for_llm = []
        co_name = data.get("ticker_info", {}).get('longName', ticker)
        
        # Sort news by recency and prioritize longer snippets
        # News are already sorted by publish_datetime_utc descending, which is good.
        # We'll just iterate and build content.
        
        # --- MODIFICATION HERE: Set MAX_NEWS_ARTICLES_FOR_LLM to 10 ---
        MAX_NEWS_ARTICLES_FOR_LLM = 10 

        for item in valid_news[:MAX_NEWS_ARTICLES_FOR_LLM]:
            title = item.get('title', '').strip()
            description = item.get('description', '').strip()
            content_snippet = item.get('content_snippet', '').replace('[+... chars]', '').strip()
            publisher = item.get('publisher', 'N/A').strip()
            source_api = item.get('source_api', 'Unknown').strip()
            publish_time = item.get('publish_time_readable', 'N/A').strip()

            # Prefer content_snippet if substantial, otherwise use description
            main_text = ""
            if content_snippet and len(content_snippet) > 50: # Require a minimum length for content
                main_text = f"Content: {content_snippet}"
            elif description and len(description) > 50: # Require a minimum length for description
                main_text = f"Description: {description}"
            
            # Only add to LLM input if there's substantial text
            if main_text:
                snippet = f"Headline: {title}"
                if main_text:
                    snippet += f" | {main_text}"
                if publisher != 'N/A':
                    snippet += f" (Source: {publisher} via {source_api})"
                if publish_time != 'N/A':
                    snippet += f" (Published: {publish_time})"
                
                content_for_llm.append(snippet)

        if not content_for_llm:
            return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": "No processable news articles with sufficient content for sentiment analysis."}

        # --- LLM Prompt Engineering ---
        prompt = f"""
        As a financial sentiment analyst, analyze the following news articles for {co_name} ({ticker}).
        Your task is to determine the overall sentiment of these articles towards the company's stock value.

        Output a single numerical score between -1.0 and 1.0 (inclusive).
        - A score of 1.0 indicates extremely positive sentiment (strong buy).
        - A score of 0.5 indicates moderately positive sentiment (buy).
        - A score of 0.0 indicates neutral sentiment (hold).
        - A score of -0.5 indicates moderately negative sentiment (sell).
        - A score of -1.0 indicates extremely negative sentiment (strong sell).

        Focus on information that could impact the stock price (e.g., earnings, product news, analyst ratings, market outlook).
        **Output ONLY the numerical score, nothing else.**

        News Articles:
        """ + "\n".join(f"- {c}" for c in content_for_llm)

        score = 0.0
        llm_err = None

        try:
            resp = self.client.generate(prompt).strip()
            
            # --- Robust LLM Response Parsing ---
            match = re.search(r"([-+]?\d*\.\d+)|([-+]?\d+)", resp)
            
            if match:
                extracted_score = float(match.group(0))
                score = max(-1.0, min(1.0, extracted_score))
                
                if len(resp.split()) > 5 and not resp.strip().replace('-', '').replace('.', '').isdigit():
                    llm_err = f"LLM responded with extra text: '{resp[:50]}...'"
            else:
                llm_err = f"LLM did not output a recognizable number: '{resp[:50]}...'"
                score = 0.0

        except Exception as e:
            llm_err = f"LLM sentiment analysis call failed: {str(e)[:150]}"
            score = 0.0

        # Consolidate error messages
        final_err = None
        if news_err:
            final_err = f"News fetch issues: {news_err}"
        if llm_err:
            final_err = (f"{final_err} | LLM issues: {llm_err}" if final_err else f"LLM issues: {llm_err}")


        # --- Sentiment Confidence and Signal Generation ---
        sentiment_confidence_score = abs(score)

        BUY_THRESHOLD = 0.45
        SELL_THRESHOLD = -0.45

        sentiment_signal = "hold"
        if score >= BUY_THRESHOLD and not final_err:
            sentiment_signal = "buy"
        elif score <= SELL_THRESHOLD and not final_err:
            sentiment_signal = "sell"

        return {
            "ticker": ticker,
            "sentiment_score": float(score),
            "sentiment_signal": sentiment_signal,
            "sentiment_confidence_score": float(sentiment_confidence_score),
            "sentiment_error": final_err
        }

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
        mc_c = mc if isinstance(mc,(int,float)) else 1; fcf_c = fcf if isinstance(fcf,(int,float)) else 0
        roe_c = roe if isinstance(roe,(int,float)) else 0; de_c = de if isinstance(de,(int,float)) else 1000
        fcy = fcf_c / mc_c if mc_c != 0 else 0
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
        filings, err = data.get("sec_all_filings_raw",[]), None
        if not filings or (isinstance(filings[0],dict) and "error" in filings[0]):
            err = filings[0].get("error") if filings and isinstance(filings[0],dict) else f"SEC: No raw filings for {ticker}."
            return {"ticker":ticker, "sec_net_insider_shares_1y":0, "sec_insider_buy_value_1y":0, "sec_insider_sell_value_1y":0, "sec_filings_signal":"hold", "sec_filings_error":err, "sec_recent_form4_transactions":[], "sec_other_recent_filings":[]}
        net_s, buy_v, sell_v, form4, others = 0,0,0,[],[]
        for f in filings:
            if not isinstance(f,dict) or "error" in f: continue
            if f.get("is_form4_transaction"):
                form4.append(f); s,p = f.get("shares",0.0), f.get("price_per_share")
                if not isinstance(s,(int,float)): s = 0.0
                if f.get("transaction_code")=="P" and f.get("acq_disp_code")=="A":
                    net_s += s;
                    if isinstance(p,(int,float)) and s!=0: buy_v += s*p
                elif f.get("transaction_code")=="S" and f.get("acq_disp_code")=="D":
                    net_s -= s;
                    if isinstance(p,(int,float)) and s!=0: sell_v += s*p
            else: others.append(f)
        sig = "hold"
        if net_s > 2000 or buy_v > 200000: sig = "buy"
        elif net_s < -2000 or sell_v > 200000: sig = "sell"
        return {"ticker":ticker, "sec_net_insider_shares_1y":int(net_s), "sec_insider_buy_value_1y":round(buy_v,2), "sec_insider_sell_value_1y":round(sell_v,2), "sec_filings_signal":sig, "sec_filings_error":None, "sec_recent_form4_transactions":form4[:10], "sec_other_recent_filings":others[:10]}

class InstitutionalHoldingsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        holdings, err = data.get("institutional_holdings",[]), None
        if holdings and isinstance(holdings[0],dict) and "error" in holdings[0]:
            err = holdings[0]["error"]
            return {"ticker":ticker, "inst_num_holders":0, "inst_total_shares_held":0, "inst_total_pct_out":0.0, "inst_holdings_signal":"hold", "inst_holdings_error":err, "inst_top_holders":[]}
        num_h, total_s, total_pct, top_h = 0,0,0.0,[]
        if holdings:
            valid_h = [d for d in holdings if isinstance(d,dict) and "error" not in d]
            if valid_h:
                num_h = len(valid_h)
                try:
                    total_s = sum(d.get('Shares',0) for d in valid_h); total_pct = sum(d.get('% Out',0.0) for d in valid_h)
                    top_h = sorted(valid_h, key=lambda x: x.get('Shares',0), reverse=True)[:10]
                except Exception as e: err = f"Error processing inst holdings: {e}"
            elif not err: err = "No valid inst holdings."
        sig = "hold"
        if total_pct > 0.50: sig = "buy"
        elif total_pct < 0.05 and num_h > 0: sig = "sell"
        return {"ticker":ticker, "inst_num_holders":num_h, "inst_total_shares_held":int(total_s), "inst_total_pct_out":float(total_pct), "inst_holdings_signal":sig, "inst_holdings_error":err, "inst_top_holders":top_h}

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
        "price": 1.0, "momentum": 0.8, "volatility": 0.3, "sentiment": 0.7, 
        "fund": 1.0, "valuation_dcf": 0.5, "valuation_pe": 0.5, 
        "sec_filings": 1.0, "inst_holdings": 0.6, "analyst": 0.7, 
        "vi_signal": 0.8, "enhanced_inst_signal": 1.2 # Weight for the new agent's signal
    }
    def run(self, ticker: str, signals: list[dict], agent_weights: dict = None) -> dict:
        # ... (the rest of the function remains the same, just ensure the WEIGHTS dict is updated)
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
        trades_to_make = []
        cash = portfolio_state['cash']
        holdings = list(portfolio_state['holdings'])

        tickers_in_portfolio = {h['ticker'] for h in holdings}
        for i, holding in reversed(list(enumerate(holdings))):
            ticker = holding['ticker']
            if ticker not in analysis_results or analysis_results[ticker].get('error'):
                continue

            analysis = analysis_results[ticker]
            if analysis.get('final_decision') == 'sell':
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
            [res for res in analysis_results.values() if res.get('final_decision') == 'buy' and res.get('ticker') not in tickers_in_portfolio and not res.get('error')],
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
        
class SECReportAnalysisAgent:
    """
    An advanced agent that uses an LLM to "read" and synthesize insights from key SEC filings.
    """
    def __init__(self, client: ModelClient):
        self.client = client

    def _fetch_filing_text(self, url: str) -> str:
        """Helper to scrape the text from a filing URL."""
        if not url: return ""
        try:
            # The Browse tool is essential for reading content from the SEC's document viewer
            browse_result = Browse(
                url=url, 
                query="Extract all text from 'Management\'s Discussion and Analysis (MD&A)' and 'Risk Factors' sections."
            )
            return browse_result if isinstance(browse_result, str) else ""
        except Exception:
            # Fallback for environments where Browse tool might not be available
            return "Browse to fetch filing text failed."

    def run(self, ticker: str, data: dict) -> dict:
        if not self.client:
            return {"sec_analysis": {"error": "LLM client not available."}}
            
        filings = data.get("sec_all_filings_raw", [])
        if not filings or (isinstance(filings[0], dict) and "error" in filings[0]):
            return {"sec_analysis": {"error": "No filings data to analyze."}}

        latest_report = next((f for f in filings if f.get('form_type') in ['10-K', '10-Q']), None)
        
        if not latest_report:
            return {"sec_analysis": {"error": "No recent 10-K or 10-Q found to analyze."}}

        report_text = self._fetch_filing_text(latest_report.get('link_to_filing'))
        
        if len(report_text) < 500:
            return {"sec_analysis": {"error": f"Could not extract sufficient text from {latest_report['form_type']}."}}

        prompt = f"""
        As a senior financial analyst, analyze the following text from the Form {latest_report['form_type']} for {ticker}.
        Distill the text into a structured JSON object with the keys: "summary" (2-3 sentence executive summary), "key_risks" (a Python list of 2-3 risks), "key_opportunities" (a Python list of 2-3 opportunities), and "management_tone" (a single adjective like "Optimistic", "Cautious", etc.).
        **Output ONLY the raw JSON object.**

        FILING TEXT:
        ---
        {report_text[:20000]}
        ---
        """
        try:
            response = self.client.generate(prompt).strip()
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                analysis = json.loads(match.group())
                analysis['source_filing'] = f"{latest_report['form_type']} ({latest_report['filing_date']})"
                return {"sec_analysis": analysis}
            return {"sec_analysis": {"error": "LLM did not return valid JSON."}}
        except Exception as e:
            return {"sec_analysis": {"error": f"LLM analysis failed: {e}"}}

class EnhancedInstitutionalHoldingsAgent:
    """
    An enhanced agent that identifies recently reporting holders as a proxy for new activity.
    """
    def run(self, ticker: str, data: dict) -> dict:
        holdings = data.get("institutional_holdings", [])
        inst_data = {
            "enhanced_inst_signal": "hold",
            "inst_recently_reported_holders": []
        }

        if not holdings or (isinstance(holdings[0], dict) and "error" in holdings[0]):
            return inst_data

        valid_h = [d for d in holdings if isinstance(d, dict) and "error" not in d]
        if valid_h:
            try:
                recent_date_limit = datetime.now() - timedelta(days=45)
                for h in valid_h:
                    if 'Date Reported' in h and isinstance(h.get('Date Reported'), str):
                        try:
                            # Accommodate different possible date formats from yfinance
                            report_date_str = h['Date Reported'].split(' ')[0]
                            report_date = datetime.strptime(report_date_str, '%Y-%m-%d')
                            if report_date > recent_date_limit:
                                inst_data["inst_recently_reported_holders"].append(h)
                        except (ValueError, TypeError):
                            continue
            except Exception:
                pass # Fail silently if date processing has an issue
        
        # Simple signal based on recent reporting activity
        if len(inst_data["inst_recently_reported_holders"]) > 5:
             inst_data["enhanced_inst_signal"] = "buy"
        
        return inst_data
        
# --- Orchestrator and Backtesting ---

def run_live_analysis(tickers, llm_client, configs):
    results = {}
    progress_bar = st.progress(0, text="Starting analysis...")
    for i, t in enumerate(tickers):
        progress_text = f"Analyzing {t}... ({i+1}/{len(tickers)})"
        progress_bar.progress((i + 1) / len(tickers), text=progress_text)
        
        ticker_info = fetch_ticker_info(t)
        if not ticker_info:
            results[t] = {"error": f"Core info for {t} unavailable."}
            continue

        data_bundle = {
            "price_history": fetch_price_history(t, period="max"),
            "ticker_info": ticker_info,
            "news": fetch_enriched_news(t, ticker_info) if configs["use_sentiment"] else [],
            # --- FIX: Call the correct function here ---
            "sec_all_filings_raw": fetch_sec_filings_from_search_api(t) if configs["use_filings"] else [],
            "institutional_holdings": fetch_inst_filings(t) if configs["use_filings"] else [],
            "value_investing_io_data": fetch_value_investing_io_data(t) if configs.get("use_value_trades") else {}
        }
        
        # --- FIX: Running ALL agents (old and new) for comparison ---
        agents = [
            PriceAgent(), MomentumAgent(), VolatilityAgent(), FundamentalsAgent(), 
            ValuationAgent(), AnalystRatingAgent(), 
            SECFilingAgent(), # <-- OLD SEC Agent
            InstitutionalHoldingsAgent() # <-- OLD Institutional Agent
        ]
        if llm_client:
            agents.extend([SentimentAgent(llm_client), NewsSummaryAgent(llm_client)])
            if configs["use_filings"]:
                # ADDING the new agents to run alongside the old ones
                agents.append(SECReportAnalysisAgent(llm_client))
                agents.append(EnhancedInstitutionalHoldingsAgent())
        
        if configs.get("use_value_trades"): 
            agents.append(ValueInvestingIOAgent())
        
        agent_res_list = [agent.run(t, data_bundle) for agent in agents]
        
        final_dec = PortfolioAgent().run(t, agent_res_list)
        
        curr_res_dict = {"ticker": t}
        for r_dict in [data_bundle, *agent_res_list, final_dec]:
            if isinstance(r_dict, dict): curr_res_dict.update(r_dict)
        results[t] = curr_res_dict
        
    progress_bar.empty()
    return results

class PortfolioAgent:
    WEIGHTS = {
        "price": 1.0, "momentum": 0.8, "volatility": 0.3, "sentiment": 0.7, 
        "fund": 1.0, "valuation_dcf": 0.5, "valuation_pe": 0.5, 
        "sec_filings": 1.0, "inst_holdings": 0.6, "analyst": 0.7, 
        "vi_signal": 0.8, 
        "enhanced_inst_signal": 1.2 # Weight for the new agent's signal
    }

    def run(self, ticker: str, signals: list[dict], agent_weights: dict = None) -> dict:
        curr_w, total_score, sum_w, agg_s = agent_weights or self.WEIGHTS, 0.0, 0.0, {}
        for s_dict in signals:
            if isinstance(s_dict, dict): agg_s.update(s_dict)
        
        s_map = {
            "price_signal": "price", "momentum_signal": "momentum", "volatility_signal": "volatility",
            "sentiment_signal": "sentiment", "fund_signal": "fund", "dcf_signal": "valuation_dcf",
            "relative_pe_signal": "valuation_pe", "sec_filings_signal": "sec_filings", 
            "inst_holdings_signal": "inst_holdings", "analyst_signal": "analyst", 
            "vi_signal": "vi_signal", "enhanced_inst_signal": "enhanced_inst_signal"
        }
        for s_key, w_key in s_map.items():
            s_val, w = agg_s.get(s_key), curr_w.get(w_key, 0)
            if s_val and w > 0:
                score_map = {"strong_buy": 1.5, "buy": 1.0, "hold": 0.0, "sell": -1.0, "strong_sell": -1.5}
                raw_score = score_map.get(str(s_val).lower(), 0)
                total_score += raw_score * w; sum_w += w
        
        comp_score = (total_score / sum_w) if sum_w else 0.0
        
        if comp_score > 0.4: decision = "strong_buy"
        elif comp_score > 0.15: decision = "buy"
        elif comp_score < -0.4: decision = "strong_sell"
        elif comp_score < -0.15: decision = "sell"
        else: decision = "hold"
        
        return {"ticker": ticker, "composite_score": comp_score, "final_decision": decision}
    
def run_backtest(ticker, start_date, end_date, initial_capital, llm_client_placeholder, backtest_agent_weights):
    st.write(f"Preparing backtest: {ticker} ({start_date} to {end_date})...")
    s_dt = datetime.strptime(start_date, "%Y-%m-%d"); fetch_s_dt = (s_dt - pd.DateOffset(months=18)).strftime("%Y-%m-%d")
    full_hist = fetch_price_history(ticker, period="max", interval="1d")
    if full_hist.empty: return {"error": f"Backtest fail {ticker}: Price history empty."}, pd.DataFrame()
    hist = full_hist[(full_hist.index >= pd.to_datetime(fetch_s_dt)) & (full_hist.index <= pd.to_datetime(end_date))].copy()
    if hist.empty or len(hist[hist.index >= pd.to_datetime(start_date)]) < 2: return {"error": f"Backtest fail {ticker}: Not enough data in range."}, pd.DataFrame()
    info_bt = fetch_ticker_info(ticker); data_static = {"ticker_info": info_bt}
    p_agent, m_agent, v_agent, port_agent = PriceAgent(), MomentumAgent(), VolatilityAgent(), PortfolioAgent()
    log, cash, shares, port_val = [], initial_capital, 0, initial_capital
    run_dates = hist[hist.index >= pd.to_datetime(start_date)].index
    for curr_dt in run_dates:
        data_sl = hist[hist.index <= curr_dt]
        curr_price_pt = data_sl.Close.iloc[-1] if not data_sl.empty else (port_val / shares if shares else 0)
        if data_sl.empty or len(data_sl) < 253:
            log.append({"date":curr_dt, "cash":cash, "shares_held":shares, "price":curr_price_pt, "portfolio_value":port_val, "signal":"hold (no data)", "composite_score":0.0}); continue
        curr_price = data_sl.Close.iloc[-1]
        pa_r, ma_r, va_r = p_agent.run(ticker,data_sl), m_agent.run(ticker,data_sl), v_agent.run(ticker,data_static,data_sl)
        final_dec_obj = port_agent.run(ticker, [pa_r,ma_r,va_r], agent_weights=backtest_agent_weights)
        final_dec = final_dec_obj["final_decision"]
        if final_dec=="buy" and cash > curr_price and curr_price > 0: s_buy = cash/curr_price; shares += s_buy; cash=0
        elif final_dec=="sell" and shares > 0: cash += shares*curr_price; shares=0
        port_val = cash + shares*curr_price
        log.append({"date":curr_dt, "cash":cash, "shares_held":shares, "price":curr_price, "portfolio_value":port_val, "signal":final_dec, "composite_score":final_dec_obj["composite_score"]})
    log_df = pd.DataFrame(log)
    if not log_df.empty: log_df.set_index("date",inplace=True)
    if log_df.empty or len(log_df) < 2: return {"message":f"Backtest log {ticker} too short."}, pd.DataFrame()
    total_ret = (log_df.portfolio_value.iloc[-1]/initial_capital - 1)*100
    days = (log_df.index[-1]-log_df.index[0]).days; years = days/365.25 if days > 0 else (1/365.25 if days==0 else 0)
    ann_ret = 0
    if years > 0 and initial_capital > 0: ann_ret = ((log_df.portfolio_value.iloc[-1]/initial_capital)**(1/years)-1)*100
    elif years == 0 and initial_capital > 0: ann_ret = total_ret
    log_df["daily_return"] = log_df.portfolio_value.pct_change().fillna(0); ann_vol = log_df.daily_return.std()*np.sqrt(252)*100
    sharpe = (ann_ret/ann_vol) if ann_vol!=0 else 0
    log_df["cum_max"] = log_df.portfolio_value.cummax(); log_df["drawdown"] = (log_df.portfolio_value - log_df.cum_max)/log_df.cum_max.replace(0,np.nan)
    max_dd = log_df.drawdown.min()*100 if not log_df.drawdown.empty and pd.notna(log_df.drawdown.min()) else 0
    trades = (log_df.signal != log_df.signal.shift()).fillna(False).sum()//2
    return {"Initial Capital":f"${initial_capital:,.2f}", "Final Portfolio Value":f"${log_df.portfolio_value.iloc[-1]:,.2f}", "Total Return (%)":f"{total_ret:.2f}%", "Annualized Return (%)":f"{ann_ret:.2f}%", "Annualized Volatility (%)":f"{ann_vol:.2f}%", "Sharpe Ratio":f"{sharpe:.2f}", "Max Drawdown (%)":f"{max_dd:.2f}%", "Number of Trades (approx)":f"{trades}"}, log_df

# --- Detailed Analysis Display Function ---
def display_detailed_analysis(res_detail):
    ticker = res_detail.get("ticker", "N/A")
    ticker_info = res_detail.get("ticker_info", {})
    tab_titles = ["📈 Chart & Core", "📊 Fundamentals & Value", "📰 News & Filings", "⚙️ All Signals"]
    tabs = st.tabs(tab_titles)

    def get_signal_color(signal):
        signal = str(signal).upper()
        if "BUY" in signal or "STRONG_BUY" in signal: return "green"
        if "SELL" in signal or "STRONG_SELL" in signal: return "red"
        return "orange"

    # --- Header Metrics ---
    st.subheader(f"Detailed Analysis for {ticker_info.get('longName', ticker)}")
    sig_col1, sig_col2, sig_col3, sig_col4 = st.columns(4)
    sig_col1.metric("Final AI Decision", str(res_detail.get('final_decision', 'N/A')).upper())
    sig_col2.metric("Composite Score", f"{res_detail.get('composite_score', 0):.2f}")
    sig_col3.metric("Analyst Signal", str(res_detail.get('analyst_signal', 'N/A')).upper())
    # The sec_filings_signal will now come from the advanced agent if it runs
    sig_col4.metric("Filings Signal", str(res_detail.get('sec_filings_signal', 'N/A')).upper())
    st.markdown("---")

    # --- Tab 1: Chart & Core ---
    with tabs[0]:
        st.subheader("Price Performance & Technical Signals")
        price_hist_chart = fetch_price_history(ticker, period="1y")
        if not price_hist_chart.empty:
            st.line_chart(price_hist_chart["Close"], use_container_width=True, color="#0072F0")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Technical Indicators")
            st.metric(label="Price Signal (SMA/RSI)", value=str(res_detail.get('price_signal', 'hold')).upper())
            st.caption(f"50D SMA: ${res_detail.get('sma50', 0):,.2f} | 200D SMA: ${res_detail.get('sma200', 0):,.2f} | RSI: {res_detail.get('rsi14', 0):.2f}")
        with col2:
            st.subheader("Momentum & Volatility")
            st.metric(label="Momentum Signal", value=str(res_detail.get('momentum_signal', 'hold')).upper())
            mom_12m = res_detail.get('momentum_12m')
            beta = res_detail.get('beta')
            st.caption(f"12-Mo Momentum: {mom_12m * 100:.2f}%" if isinstance(mom_12m, (int, float)) else "12-Mo: N/A")
            st.caption(f"Beta: {beta:.2f}" if isinstance(beta, (int, float)) else "Beta: N/A")

    # --- Tab 2: Fundamentals & Value ---
    with tabs[1]:
        st.subheader(f"Fundamental & Value Overview")
        if ticker_info.get('longBusinessSummary'):
            with st.expander("Show Business Summary"):
                st.markdown(ticker_info.get('longBusinessSummary'))
        
        st.markdown("---")
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        cap_str = f"${ticker_info.get('marketCap', 0) / 1e9:.2f}B" if isinstance(ticker_info.get('marketCap'), (int, float)) else "N/A"
        f_col1.metric("Market Cap", cap_str)
        f_col2.metric("Forward P/E", f"{ticker_info.get('forwardPE'):.2f}" if isinstance(ticker_info.get('forwardPE'), (int, float)) else "N/A")
        roe_val = ticker_info.get('returnOnEquity')
        f_col3.metric("Return on Equity", f"{roe_val * 100:.2f}%" if isinstance(roe_val, (int, float)) else "N/A")
        fcy_val = res_detail.get('fcf_yield')
        f_col4.metric("FCF Yield", f"{fcy_val * 100:.2f}%" if isinstance(fcy_val, (int, float)) else "N/A")

    # --- Tab 3: News & Filings ---
    with tabs[2]:
        st.subheader("News, Filings, and Institutional Ownership")
        
        # --- AI Analysis of SEC Filings ---
        with st.expander("**[NEW] AI-Powered Filing Analysis**", expanded=True):
            analysis = res_detail.get("sec_analysis", {})
            if analysis and not analysis.get("error"):
                st.success(f"**Source:** {analysis.get('source_filing', 'N/A')} | **Management Tone:** {analysis.get('management_tone', 'N/A')}")
                st.write(analysis.get('summary', "No summary available."))
                o_col, r_col = st.columns(2)
                with o_col:
                    st.markdown("**Key Opportunities**"); [st.markdown(f"• {item}") for item in analysis.get('key_opportunities', [])]
                with r_col:
                    st.markdown("**Key Risks**"); [st.markdown(f"• {item}") for item in analysis.get('key_risks', [])]
            else:
                st.warning(f"AI analysis could not be performed. Reason: {analysis.get('error', 'Not available')}")
        
        st.markdown("---")
        
        # --- Source Data Popovers ---
        c1, c2, c3 = st.columns(3)
        with c1:
            with st.popover("View Raw Filings List"):
                st.dataframe(pd.DataFrame(res_detail.get('sec_all_filings_raw', [])), hide_index=True)
        with c2:
            with st.popover("View Institutional Holders"):
                st.markdown("##### Top 10 Holders (Snapshot)")
                st.dataframe(pd.DataFrame(res_detail.get('inst_top_holders', [])), hide_index=True)
                st.markdown("---")
                st.markdown("##### [NEW] Recently Reported (Last 45 Days)")
                recent_holders = res_detail.get('inst_recently_reported_holders', [])
                st.dataframe(pd.DataFrame(recent_holders), hide_index=True) if
    # --- Tab 4: All Signals ---
    with tabs[3]:
        st.subheader("All Agent Signals & Raw Data")
        
        # Display signals in a more readable format
        signals_data = {
            "Price Signal": str(res_detail.get("price_signal","N/A")).upper(),
            "Momentum Signal": str(res_detail.get("momentum_signal","N/A")).upper(),
            "Fundamental Signal": str(res_detail.get("fund_signal","N/A")).upper(),
            "Analyst Signal": str(res_detail.get("analyst_signal","N/A")).upper(),
            "VI.io Signal": str(res_detail.get("vi_signal","N/A")).upper(),
            "News Sentiment Signal": str(res_detail.get("sentiment_signal","N/A")).upper(),
            "Original SEC Signal": str(res_detail.get("sec_filings_signal","N/A")).upper(),
            "Original Inst. Signal": str(res_detail.get("inst_holdings_signal","N/A")).upper(),
            "[NEW] Enhanced Inst. Signal": str(res_detail.get("enhanced_inst_signal","N/A")).upper()
        }
        df_signals = pd.DataFrame(signals_data.items(), columns=["Agent", "Signal"])
        st.dataframe(df_signals.style.applymap(lambda x: f'color: {get_signal_color(x)}', subset=['Signal']), hide_index=True)

        with st.expander("View Full Raw Data Dictionary"):
            st.json(res_detail)

# --- Streamlit UI ---
llm_client = None
try:
    ds_key, oa_key = st.secrets.get("DEEPSEEK_API_KEY"), st.secrets.get("OPENAI_API_KEY")
    if not ds_key: ds_key = os.environ.get("DEEPSEEK_API_KEY")
    if not oa_key: oa_key = os.environ.get("OPENAI_API_KEY")
    if ds_key: llm_client = ModelClient(api_key=ds_key, provider="deepseek"); st.sidebar.caption("✅ LLM: DeepSeek")
    elif oa_key: llm_client = ModelClient(api_key=oa_key, provider="openai"); st.sidebar.caption("✅ LLM: OpenAI")
    else: st.sidebar.warning("LLM API key missing. Sentiment/Summary disabled.")
except ValueError as e: st.sidebar.error(f"LLM Init Error: {e}"); llm_client=None
except Exception as e: st.sidebar.error(f"LLM Unexpected Init Error: {e}"); llm_client=None

st.title("🚀 AI Hedge Fund Simulator")
st.header("⚙️ Configuration"); config_cont = st.container(border=True)

app_mode_options = ["Live Analysis", "Backtesting", "💼 Portfolio Management", "🤖 Virtual Trading"]
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = app_mode_options[0]

with config_cont:
    # Logic to reset triggers if mode is changed
    current_mode_index = app_mode_options.index(st.session_state.app_mode)
    selected_mode = st.radio("Select Mode:", app_mode_options, key="app_mode_sel_main_key", horizontal=True, index=current_mode_index)
    if selected_mode != st.session_state.app_mode:
        st.session_state.app_mode = selected_mode
        st.session_state.live_analysis_triggered = False
        st.session_state.backtest_triggered = False
        st.rerun()

    st.markdown("---")

    if st.session_state.app_mode == "Live Analysis":
        st.subheader("Live Analysis Settings")
        tickers_in_live = st.text_input("Tickers (comma-separated):", "AAPL,MSFT,GOOG,CRWD", key="live_tickers_input")
        st.caption("ℹ️ Live analysis uses all available historical data.")
        st.subheader("Feature Toggles"); feat_cols = st.columns(3)
        with feat_cols[0]:
            use_sent_live = st.checkbox("News Sentiment & Summary (LLM)", value=bool(llm_client), disabled=not llm_client, key="live_sent_cb_main", help="Uses LLM. Requires NewsAPI key.")
            use_filings_live = st.checkbox("SEC & Inst. Filings", value=True, key="live_sec_cb_main")
            use_valtrades_live = st.checkbox("ValueInvesting.io (Exp.)", value=False, key="live_vt_cb_main", help="Scrapes ValueInvesting.io. May be slow/unreliable.")
        
        if st.button("🚀 Run Live Analysis", use_container_width=True, type="primary", key="run_live_analysis_button"):
            live_tickers = [t.strip().upper() for t in tickers_in_live.split(",") if t.strip()]
            if not live_tickers:
                st.error("Please enter at least one ticker.")
            else:
                live_configs = {"use_sentiment":use_sent_live, "use_filings":use_filings_live,"use_value_trades":use_valtrades_live}
                with st.spinner("⏳ Processing live analysis..."):
                    st.session_state.live_output = run_live_analysis(live_tickers, llm_client, live_configs)
                    st.session_state.live_analysis_triggered = True # Set flag
                    st.rerun()

    elif st.session_state.app_mode == "Backtesting":
        st.subheader("Backtesting Settings")
        st.session_state.bt_ticker = st.text_input("Ticker:", "AAPL", key="bt_ticker_in_bt").upper()
        # ... (rest of backtesting config is the same)
        bt_capital_source = st.radio("Capital Source:", ("Manual Input", "From Saved Portfolio"), horizontal=True, key="bt_capital_source_radio")
        bt_capital = 10000
        if bt_capital_source == "Manual Input":
            bt_capital = st.number_input("Initial Capital:", 1000, 1000000, 10000, 1000, key="bt_cap_in_bt", format="%d")
        else:
            portfolio_names_bt = list(st.session_state.portfolios_data.keys())
            if not portfolio_names_bt: st.warning("No portfolios found.")
            else:
                sel_pf_bt = st.selectbox("Select Portfolio to use its total value:", portfolio_names_bt, key="bt_pf_select")
                # ... rest of logic
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
            w_p, w_m, w_v = st.slider("Price W:",0.,2.,1.,.1,key="bt_w_p_bt"), st.slider("Mom W:",0.,2.,.8,.1,key="bt_w_m_bt"), st.slider("Vol W:",0.,2.,.2,.1,key="bt_w_v_bt")
            st.info("Other signals disabled in backtest.")
            st.session_state.bt_weights = {"price":w_p, "momentum":w_m, "volatility":w_v, "sentiment":0.,"fund":0.,"valuation_dcf":0.,"valuation_pe":0.,"sec_filings":0.,"inst_holdings":0.,"analyst":0.,"politician_filings":0.,"vi_signal":0.}
            st.session_state.bt_capital = bt_capital

        if st.button("📈 Run Backtest",use_container_width=True,type="primary",key="run_bt_btn_main"):
            if st.session_state.bt_ticker:
                with st.spinner(f"⏳ Running backtest for {st.session_state.bt_ticker}..."):
                    metrics, log_df = run_backtest(st.session_state.bt_ticker, st.session_state.bt_start_str, st.session_state.bt_end_str, st.session_state.bt_capital, llm_client, st.session_state.bt_weights)
                    st.session_state.backtest_results[st.session_state.bt_ticker] = {"metrics": metrics, "log_df": log_df}
                    st.session_state.backtest_triggered = True # Set flag
                    st.rerun()

    elif st.session_state.app_mode == "💼 Portfolio Management":
        # This section remains self-contained and does not need a separate results area.
        # The logic is unchanged.
        st.subheader("💼 Portfolio Management")
        # --- Portfolio Selection ---
        portfolio_names_list = list(st.session_state.portfolios_data.keys())

        if not portfolio_names_list:
            # Create a default portfolio if none exist
            st.session_state.portfolios_data["My First Portfolio"] = {"holdings": [], "cash": 10000.0} # Initialize with cash
            st.session_state.selected_portfolio_name = "My First Portfolio"
            save_portfolios(st.session_state.portfolios_data)
            st.rerun() # Rerun to show the newly created portfolio

        col_pf1, col_pf2, col_pf3 = st.columns([3, 1, 1])

        st.session_state.selected_portfolio_name = col_pf1.selectbox(
            "Select Portfolio:",
            portfolio_names_list,
            index=portfolio_names_list.index(st.session_state.selected_portfolio_name) if st.session_state.selected_portfolio_name in portfolio_names_list else 0,
            key="portfolio_selector"
        )
        current_portfolio = st.session_state.portfolios_data.get(st.session_state.selected_portfolio_name, {"holdings": [], "cash": 0.0})

        new_portfolio_name = col_pf2.text_input("New Portfolio Name:", "", key="new_pf_name")
        if col_pf3.button("➕ Create Portfolio", key="create_pf_btn"):
            if new_portfolio_name and new_portfolio_name not in st.session_state.portfolios_data:
                st.session_state.portfolios_data[new_portfolio_name] = {"holdings": [], "cash": 10000.0} # New portfolios start with cash
                save_portfolios(st.session_state.portfolios_data)
                st.session_state.selected_portfolio_name = new_portfolio_name
                st.success(f"Portfolio '{new_portfolio_name}' created!")
                st.rerun()
            else:
                st.error("Portfolio name is empty or already exists.")

        st.markdown("---")
        st.subheader(f"Holdings for '{st.session_state.selected_portfolio_name}'")

        # Display current cash
        st.metric("Cash Balance", f"${current_portfolio['cash']:,.2f}")

        # Fetch current prices for holdings to calculate market value and P&L
        holdings_display_data = []
        total_market_value = 0.0
        total_unrealized_pnl = 0.0
        total_invested_cost = 0.0

        if current_portfolio['holdings']:
            with st.spinner(f"Fetching live prices for {st.session_state.selected_portfolio_name} holdings..."):
                for holding in current_portfolio['holdings']:
                    ticker = holding['ticker']
                    quantity = holding['quantity']
                    avg_price = holding['avg_price']

                    info = fetch_ticker_info(ticker)
                    current_price = info.get("currentPrice") or (fetch_price_history(ticker, period="1d").iloc[-1]["Close"] if not fetch_price_history(ticker, period="1d").empty else None)

                    if isinstance(current_price, (int, float)):
                        market_value = current_price * quantity
                        unrealized_pnl = (current_price - avg_price) * quantity
                        total_market_value += market_value
                        total_unrealized_pnl += unrealized_pnl
                        total_invested_cost += avg_price * quantity # Track total invested for overall P&L %
                        holdings_display_data.append({
                            "Ticker": ticker,
                            "Quantity": quantity,
                            "Avg. Cost": avg_price,
                            "Current Price": current_price,
                            "Market Value": market_value,
                            "Unrealized P&L": unrealized_pnl,
                            "P&L (%)": (unrealized_pnl / (avg_price * quantity) * 100) if (avg_price * quantity) != 0 else 0.0
                        })
                    else:
                        holdings_display_data.append({
                            "Ticker": ticker,
                            "Quantity": quantity,
                            "Avg. Cost": avg_price,
                            "Current Price": "N/A",
                            "Market Value": "N/A",
                            "Unrealized P&L": "N/A",
                            "P&L (%)": "N/A"
                        })
        
        # Display overall portfolio metrics
        overall_total_value = current_portfolio['cash'] + total_market_value
        overall_pnl_percent = (total_unrealized_pnl / total_invested_cost * 100) if total_invested_cost != 0 else 0.0
        overall_pnl_color = "normal" if total_unrealized_pnl >= 0 else "inverse"

        st.columns(3)[0].metric("Total Portfolio Value", f"${overall_total_value:,.2f}")
        st.columns(3)[1].metric("Total Holdings Value", f"${total_market_value:,.2f}")
        st.columns(3)[2].metric("Total Unrealized P&L", f"${total_unrealized_pnl:,.2f}", f"{overall_pnl_percent:.2f}%", delta_color=overall_pnl_color)


        if holdings_display_data:
            holdings_df = pd.DataFrame(holdings_display_data)
            st.dataframe(holdings_df, use_container_width=True, hide_index=True,
                         column_config={
                             "Quantity": st.column_config.NumberColumn(format="%.4f"),
                             "Avg. Cost": st.column_config.NumberColumn(format="$%.2f"),
                             "Current Price": st.column_config.NumberColumn(format="$%.2f"),
                             "Market Value": st.column_config.NumberColumn(format="$%.2f"),
                             "Unrealized P&L": st.column_config.NumberColumn(format="$%.2f", help="Unrealized Profit & Loss"),
                             "P&L (%)": st.column_config.ProgressColumn(format="%.2f%%", min_value=-100, max_value=100)
                         })
        else:
            st.info("This portfolio currently has no stock holdings. Use the 'Add Stock' section below.")

        st.markdown("---")
        st.subheader("Add/Remove Stocks")
        col_add1, col_add2, col_add3 = st.columns(3)
        add_ticker = col_add1.text_input("Ticker to Add:", "", key="add_ticker_input").upper()
        add_quantity = col_add2.number_input("Quantity:", min_value=0.01, value=1.0, step=0.1, key="add_quantity_input")
        add_price = col_add3.number_input("Purchase Price (optional, current if 0):", min_value=0.0, value=0.0, step=0.01, key="add_price_input")

        if st.button("➕ Add Stock to Portfolio", key="add_stock_btn"):
            if add_ticker and add_quantity > 0:
                # Fetch current price if not provided
                if add_price == 0:
                    info = fetch_ticker_info(add_ticker)
                    current_price_for_add = info.get("currentPrice") or (fetch_price_history(add_ticker, period="1d").iloc[-1]["Close"] if not fetch_price_history(add_ticker, period="1d").empty else None)
                    if not current_price_for_add:
                        st.error(f"Could not fetch current price for {add_ticker}. Please enter a purchase price manually.")
                        st.stop() # Stop execution to allow user to input price
                    purchase_price = current_price_for_add
                else:
                    purchase_price = add_price

                # Check if holding already exists
                existing_holding_index = -1
                for i, h in enumerate(current_portfolio['holdings']):
                    if h['ticker'] == add_ticker:
                        existing_holding_index = i
                        break

                if current_portfolio['cash'] >= (purchase_price * add_quantity):
                    current_portfolio['cash'] -= (purchase_price * add_quantity)
                    if existing_holding_index != -1:
                        # Update existing holding
                        existing_holding = current_portfolio['holdings'][existing_holding_index]
                        new_total_quantity = existing_holding['quantity'] + add_quantity
                        new_avg_price = ((existing_holding['avg_price'] * existing_holding['quantity']) + (purchase_price * add_quantity)) / new_total_quantity
                        existing_holding['quantity'] = new_total_quantity
                        existing_holding['avg_price'] = new_avg_price
                    else:
                        # Add new holding
                        current_portfolio['holdings'].append({"ticker": add_ticker, "quantity": add_quantity, "avg_price": purchase_price})
                    
                    save_portfolios(st.session_state.portfolios_data)
                    st.success(f"Added {add_quantity:.2f} shares of {add_ticker} to '{st.session_state.selected_portfolio_name}'.")
                    st.rerun()
                else:
                    st.error(f"Insufficient cash to buy {add_quantity:.2f} shares of {add_ticker} at ${purchase_price:.2f}. Available cash: ${current_portfolio['cash']:.2f}")
            else:
                st.error("Please enter a valid ticker and quantity.")

        col_rem1, col_rem2 = st.columns([1,2])
        remove_ticker = col_rem1.text_input("Ticker to Remove:", "", key="remove_ticker_input").upper()
        if col_rem2.button("➖ Remove Stock from Portfolio", key="remove_stock_btn"):
            if remove_ticker:
                initial_holdings_count = len(current_portfolio['holdings'])
                # Find the holding to remove and process sale to cash
                removed_holding = None
                for i, h in enumerate(current_portfolio['holdings']):
                    if h['ticker'] == remove_ticker:
                        removed_holding = current_portfolio['holdings'].pop(i)
                        break

                if removed_holding:
                    info = fetch_ticker_info(removed_holding['ticker'])
                    current_price_for_remove = info.get("currentPrice") or (fetch_price_history(removed_holding['ticker'], period="1d").iloc[-1]["Close"] if not fetch_price_history(removed_holding['ticker'], period="1d").empty else None)
                    
                    if isinstance(current_price_for_remove, (int,float)):
                        sale_value = current_price_for_remove * removed_holding['quantity']
                        current_portfolio['cash'] += sale_value
                        st.success(f"Removed {removed_holding['quantity']:.2f} shares of {remove_ticker} from '{st.session_state.selected_portfolio_name}'. Sold for ${sale_value:,.2f}.")
                    else:
                        st.warning(f"Removed {removed_holding['quantity']:.2f} shares of {remove_ticker}. Could not fetch current price, so cash balance not updated for sale.")
                    save_portfolios(st.session_state.portfolios_data)
                    st.rerun()
                else:
                    st.error(f"{remove_ticker} not found in '{st.session_state.selected_portfolio_name}' holdings.")
            else:
                st.error("Please enter a ticker to remove.")
        
        st.markdown("---")
        if st.button("🗑️ Delete Current Portfolio", key="delete_portfolio_btn", type="secondary"):
            if st.session_state.selected_portfolio_name and st.session_state.selected_portfolio_name in st.session_state.portfolios_data:
                del st.session_state.portfolios_data[st.session_state.selected_portfolio_name]
                save_portfolios(st.session_state.portfolios_data)
                st.session_state.selected_portfolio_name = None # Reset selected portfolio
                st.success(f"Portfolio '{st.session_state.selected_portfolio_name}' deleted.")
                st.rerun()
            else:
                st.error("No portfolio selected or found to delete.")


    elif st.session_state.app_mode == "🤖 Virtual Trading":
        st.subheader("🤖 AI Virtual Trader Controls")
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
            all_tickers_to_scan = stock_universe['safe'] + stock_universe['risky']
            ai_configs = {"use_sentiment": True, "use_filings": True, "use_value_trades": False}
            
            # This is where the analysis and trading happens
            analysis_results = run_live_analysis(all_tickers_to_scan, llm_client, ai_configs)
            trader_agent = AITraderAgent(llm_client, stock_universe)
            trades = trader_agent.run(st.session_state.virtual_portfolio, analysis_results)
            
            if not trades:
                st.toast("AI analyzed the market and decided to hold all positions.", icon="✅")
            else:
                for trade in trades:
                    if trade['type'] == 'buy':
                        st.toast(f"AI is buying {trade['ticker']}...", icon="📈")
                        existing_holding = next((h for h in st.session_state.virtual_portfolio['holdings'] if h['ticker'] == trade['ticker']), None)
                        if existing_holding:
                            new_total_quantity = existing_holding['quantity'] + trade['quantity']
                            new_avg_price = ((existing_holding['avg_price'] * existing_holding['quantity']) + (trade['price'] * trade['quantity'])) / new_total_quantity
                            existing_holding['quantity'] = new_total_quantity; existing_holding['avg_price'] = new_avg_price
                        else:
                            st.session_state.virtual_portfolio['holdings'].append({'ticker': trade['ticker'], 'quantity': trade['quantity'], 'avg_price': trade['price']})
                        st.session_state.virtual_portfolio['cash'] -= trade['price'] * trade['quantity']
                    elif trade['type'] == 'sell':
                        st.toast(f"AI is selling {trade['ticker']}...", icon="📉")
                        st.session_state.virtual_portfolio['cash'] += trade['price'] * trade['quantity']
                        # Ensure we remove only the exact quantity if partial sale or remove the whole holding
                        st.session_state.virtual_portfolio['holdings'] = [h for h in st.session_state.virtual_portfolio['holdings'] if h['ticker'] != trade['ticker']]
                    st.session_state.virtual_portfolio['transaction_history'].insert(0, {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "ticker": trade['ticker'], "type": trade['type'].upper(), "quantity": f"{trade['quantity']:.4f}", "price": f"${trade['price']:.2f}", "reason": trade['reason']})
            
            st.session_state.virtual_portfolio["last_scan_date"] = datetime.now().strftime("%Y-%m-%d")
            save_virtual_portfolio(st.session_state.virtual_portfolio)
            st.rerun()

st.markdown("---")

# ===============================================
# Main Results Display Area
# This block now correctly handles the display for each mode.
# ===============================================

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
                    # ... (rest of live analysis display logic is unchanged)
                    if not res or res.get("error"): st.error(f"**{sym}**: {res.get('error','No data.') if res else 'No data.'}"); continue
                    dec,score,price = res.get("final_decision","N/A").upper(), res.get("composite_score",float('nan')), res.get("current_price_display")
                    cmap={"BUY":"green","SELL":"red","HOLD":"#FFA500","ERROR":"#808080","N/A":"#D3D3D3"}; color=cmap.get(dec,"#D3D3D3")
                    p_html = f'<p style="font-size:0.9em;">Price:<strong>${price:,.2f}</strong></p>' if isinstance(price,(int,float)) else '<p style="font-size:0.9em;">Price:<strong>N/A</strong></p>'
                    s_html = f'<p style="font-size:0.9em;">Score:<strong style="color:{color};">{score:.2f}</strong></p>' if pd.notna(score) else f'<p style="font-size:0.9em;">Score:<strong style="color:{color};">N/A</strong></p>'
                    st.markdown(f"""<div style="border:1px solid {color};border-radius:8px;padding:15px;margin-bottom:10px;background-color:{color}20;"><h3 style="margin-bottom:5px;color:{color};">{sym}</h3><p style="font-size:1.6em;font-weight:bold;color:{color};margin-bottom:5px;">{dec}</p>{s_html}{p_html}</div>""", unsafe_allow_html=True)
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
            metrics_df_bt = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
            st.table(metrics_df_bt)
            if log_df is not None and not log_df.empty:
                st.subheader("Portfolio Value Over Time"); st.line_chart(log_df["portfolio_value"])
                st.subheader("Drawdown Over Time"); st.area_chart(log_df["drawdown"].fillna(0))
                with st.expander("View Raw Backtest Log (Last 1000)"): st.dataframe(log_df.tail(1000))
            else: st.warning("Backtest log empty.")
        elif metrics:
            st.error(f"Backtest failed: {metrics.get('message','') or metrics.get('error','Unknown error')}")

elif st.session_state.app_mode == "🤖 Virtual Trading":
    st.header("📈 Virtual Portfolio Dashboard")
    with st.container(border=True):
        holdings_df_data = []
        total_holdings_value, total_pnl, initial_investment = 0.0, 0.0, 0.001 # Avoid division by zero
        
        # Use a single spinner for all price fetches
        if st.session_state.virtual_portfolio['holdings']:
            with st.spinner("Fetching latest prices for dashboard..."):
                for holding in st.session_state.virtual_portfolio['holdings']:
                    info = fetch_ticker_info(holding['ticker'])
                    price = info.get("currentPrice")
                    current_value = price * holding['quantity'] if isinstance(price, (int,float)) else 0
                    pnl = (price - holding['avg_price']) * holding['quantity'] if isinstance(price, (int,float)) else 0
                    total_holdings_value += current_value
                    total_pnl += pnl
                    initial_investment += holding['avg_price'] * holding['quantity']
                    holdings_df_data.append({"Ticker": holding['ticker'], "Quantity": holding['quantity'], "Avg. Price": holding['avg_price'], "Current Price": price, "Current Value": current_value, "P&L": pnl})

        total_portfolio_value = st.session_state.virtual_portfolio['cash'] + total_holdings_value
        pnl_percent = (total_pnl / initial_investment * 100)

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
        else:
            st.info("The portfolio currently holds no stocks. Run the AI Trader to start investing.")

        st.subheader("Transaction History")
        if st.session_state.virtual_portfolio['transaction_history']:
            history_df = pd.DataFrame(st.session_state.virtual_portfolio['transaction_history'])
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            st.info("No transactions have been made yet.")



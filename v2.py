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
import altair as alt
import time
import random

# --- Page Config (Must be the first Streamlit command) ---
st.set_page_config(page_title="AI Hedge Fund Simulator", layout="wide")

# Load environment variables
load_dotenv()

# SEC EDGAR User-Agent (important for compliance)
SEC_USER_AGENT = "KevalAhirApp/1.0 keval.ahir2019@gmail.com"

# File paths for portfolio persistence
PORTFOLIOS_FILE = "portfolios.json"
VIRTUAL_PORTFOLIO_FILE = "virtual_portfolio.json"

# --------------------------------
# Utility Functions (Global Scope)
# --------------------------------

def get_signal_color(signal):
    """Returns a color (green, red, orange) based on the signal for UI display."""
    signal = str(signal).upper()
    if signal in ["BUY", "STRONG_BUY"]: return "green"
    if signal == "SELL": return "red"
    return "orange"

# --------------------------------
# Portfolio Helper Functions
# --------------------------------
def load_portfolios():
    """Loads all saved portfolios from the JSON file."""
    if os.path.exists(PORTFOLIOS_FILE):
        try:
            with open(PORTFOLIOS_FILE, 'r') as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {} # Ensure it's a dict
        except json.JSONDecodeError:
            st.warning("Error decoding portfolios.json. Starting with empty portfolios.")
            return {}
    return {}

def save_portfolios(portfolios_data):
    """Saves all portfolios to the JSON file."""
    if not isinstance(portfolios_data, dict):
        st.error("Error saving portfolios: Data is not in the correct format.")
        return
    with open(PORTFOLIOS_FILE, 'w') as f:
        json.dump(portfolios_data, f, indent=4)

def load_virtual_portfolio():
    """Loads the virtual trading portfolio state."""
    if os.path.exists(VIRTUAL_PORTFOLIO_FILE):
        try:
            with open(VIRTUAL_PORTFOLIO_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.warning("Error decoding virtual_portfolio.json. Starting with default virtual portfolio.")
            return get_default_virtual_portfolio()
    return get_default_virtual_portfolio()

def save_virtual_portfolio(data):
    """Saves the virtual trading portfolio state."""
    with open(VIRTUAL_PORTFOLIO_FILE, 'w') as f:
        json.dump(data, f, indent=4, default=str) # Use default=str to handle datetimes if they exist

def get_default_virtual_portfolio():
    """Returns the default structure for a new virtual portfolio."""
    return {
        "cash": 3500.0,
        "holdings": [],
        "transaction_history": [],
        "last_scan_date": None
    }

# --- Session State Initialization ---
# Initialize session state variables if they don't exist, loading from files
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

# Flags to control when to display analysis results (to avoid re-displaying on every rerun)
if 'live_analysis_triggered' not in st.session_state:
    st.session_state.live_analysis_triggered = False
if 'backtest_triggered' not in st.session_state:
    st.session_state.backtest_triggered = False

# --------------------------------
# Data Fetchers (All using yfinance and direct scraping for non-standard data)
# --------------------------------

@st.cache_data(ttl=300) # Cache for 5 minutes
def fetch_price_history(ticker: str, period: str = "max", interval: str = "1d") -> pd.DataFrame:
    """Fetches historical price data for a given ticker using yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, interval=interval)
        if df.empty:
            st.warning(f"No price history returned from yfinance for {ticker} (period={period}). This might be an invalid ticker or data issue.")
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None) # Remove timezone info for consistency
        return df
    except Exception as e:
        st.error(f"Critical error fetching price history for {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300) # Cache for 5 minutes
def fetch_ticker_info(ticker: str) -> dict:
    """
    Fetches comprehensive ticker information from yfinance with robust error handling.
    Returns a dictionary of relevant info or an error message within the dict.
    """
    max_retries = 3
    base_delay = 1.0 # seconds
    
    for attempt in range(max_retries):
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info

            # Validate if essential information is present
            if not info or not info.get('financialCurrency'): # financialCurrency is a good indicator of valid stock data
                missing_keys = []
                if not info:
                    missing_keys.append("empty_info_dict")
                if info and info.get('financialCurrency') is None:
                    missing_keys.append("financialCurrency")
                
                error_detail = f"Missing essential info: {', '.join(missing_keys)}. Raw keys: {list(info.keys()) if info else 'None'}."
                st.warning(f"Attempt {attempt + 1}/{max_retries}: Failed to get complete info for {ticker}: {error_detail}")
                
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt) + random.uniform(0.1, 0.5))
                    continue # Retry on incomplete data
                else:
                    final_error_msg = f"Failed to get complete info for {ticker} after {max_retries} attempts. Reason: {error_detail}. Invalid ticker, delisted, or persistent yfinance issues."
                    st.error(final_error_msg)
                    return {"_error": final_error_msg}
            
            # Return a cleaned dictionary with relevant fields
            return {
                "marketCap": info.get("marketCap"),
                "freeCashflow": info.get("freeCashflow"),
                "forwardPE": info.get("forwardPE"),
                "trailingPE": info.get("trailingPE"),
                "priceToBook": info.get("priceToBook"),
                "enterpriseToRevenue": info.get("enterpriseToRevenue"),
                "enterpriseToEbitda": info.get("enterpriseToEbitda"),
                "returnOnEquity": info.get("returnOnEquity"),
                "debtToEquity": info.get("debtToEquity"),
                "beta": info.get("beta"),
                "targetMeanPrice": info.get("targetMeanPrice"),
                "recommendationKey": info.get("recommendationKey"),
                "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions"),
                "industry": info.get("industry"),
                "sector": info.get("sector"),
                "longName": info.get("longName"),
                "shortName": info.get("shortName"),
                "longBusinessSummary": info.get("longBusinessSummary"),
                "currentPrice": info.get("currentPrice") or info.get("regularMarketPrice"), # Fallback to regularMarketPrice
                "financialCurrency": info.get("financialCurrency"),
                "_error": None # Indicate no error if successful
            }
        except Exception as e:
            st.error(f"Attempt {attempt + 1}/{max_retries}: Exception fetching ticker info for {ticker}: {e}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt) + random.uniform(0.1, 0.5))
                continue
            else:
                final_error_msg = f"Final exception fetching info for {ticker} after {max_retries} attempts: {e}. This may indicate an invalid ticker, a delisted stock, or persistent upstream data issues."
                st.error(final_error_msg)
                return {"_error": final_error_msg}

@st.cache_data(ttl=300) # Cache for 5 minutes
def fetch_enriched_news(ticker: str, ticker_info_data: dict) -> list[dict]:
    """Fetches news from Yahoo Finance for a given ticker and enriches it."""
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
    """Fetches news from NewsAPI.org for a given company and enriches it."""
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
    """Fetches the CIK to Ticker mappings from SEC EDGAR."""
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(url, headers={'User-Agent': SEC_USER_AGENT}); response.raise_for_status()
        return {item['ticker']: str(item['cik_str']).zfill(10) for item in response.json() if 'ticker' in item and 'cik_str' in item}
    except Exception as e: st.error(f"CRITICAL: Failed CIK mappings: {e}. SEC filing features may be affected."); return {}
TICKER_TO_CIK_MAP = get_all_cik_ticker_mappings()

def get_cik_for_ticker(ticker: str) -> str | None: return TICKER_TO_CIK_MAP.get(ticker.upper())

@st.cache_data(ttl=4*3600)
def fetch_all_sec_filings(ticker_symbol: str, lookback_days: int = 365) -> list[dict]:
    """
    Fetches recent SEC filings (including Form 4s for insider trading) for a given ticker.
    Includes a placeholder for document content extraction for LLM summarization.
    """
    cik = get_cik_for_ticker(ticker_symbol)
    if not cik:
        # Fallback lookup if not in initial map (can be slow)
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
        except Exception: # Broad catch for lookup issues
            pass
    if not cik: return [{"error": f"SEC: CIK not found for {ticker_symbol}."}]
    
    cik_padded = str(cik).zfill(10)
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    headers = {'User-Agent': SEC_USER_AGENT}; filings_list = []
    try:
        response = requests.get(submissions_url, headers=headers, timeout=20); response.raise_for_status()
        submissions_data = response.json()

        today_utc = datetime.now(timezone.utc)
        date_limit = today_utc - timedelta(days=lookback_days)
        
        if 'filings' in submissions_data and 'recent' in submissions_data['filings']:
            recent = submissions_data['filings']['recent']
            forms, dates, acc_nos, docs = recent.get('form',[]), recent.get('filingDate',[]), recent.get('accessionNumber',[]), recent.get('primaryDocument',[])
            
            # Collect metadata for relevant filings within the lookback period
            metadata = []
            for i in range(len(forms)):
                try:
                    filing_date = datetime.strptime(dates[i], '%Y-%m-%d').replace(tzinfo=timezone.utc)
                    if filing_date >= date_limit:
                        metadata.append({"form_type": forms[i], "filing_date_str": dates[i], "accession_number": acc_nos[i], "primary_document": docs[i]})
                except (ValueError, IndexError): continue # Skip if date parsing fails or index out of bounds
            
            xml_fetches = 0
            max_xml_fetches = 20 # Limit number of XML (Form 4) fetches to avoid overwhelming API/memory
            max_other_filings = 15 # Limit number of other filings metadata

            for info in metadata:
                form, date_str, acc_no, doc_name = info["form_type"], info["filing_date_str"], info["accession_number"], info["primary_document"]
                acc_no_dashless = acc_no.replace('-', '')
                idx_link = f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{acc_no_dashless}/{acc_no}-index.html"
                
                # Placeholder for fetching document content for LLM summary.
                # IMPORTANT: For a real application, you would need to fetch the actual document
                # (usually HTML, sometimes XML/TXT) and parse specific sections for summarization.
                # This is a complex task due to varied document structures.
                # Libraries like `sec-api.io` or custom scraping logic would be needed.
                document_content_for_llm = ""
                if form in ['10-K', '10-Q', '8-K']:
                    document_content_for_llm = f"Content for {form} on {date_str}. (Automated scraping and parsing of document content for LLM summary is a complex task requiring specific logic for each form type or external APIs like sec-api.io. This is currently a placeholder for integration.)"
                
                if form == '4' and doc_name.lower().endswith(('.xml', '.xsd')):
                    if xml_fetches >= max_xml_fetches: continue # Respect limit
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
                                if shares != 0:
                                    filings_list.append({"is_form4_transaction": True, "ticker": ticker_symbol, "filing_date": date_str, "transaction_date": tx_date, "reporting_owner": owner_name, "owner_relationship": owner_rel, "transaction_code": tx_code, "acq_disp_code": ad_code, "shares": shares, "price_per_share": price, "link_to_filing": idx_link})
                    except Exception: # Broad catch for parsing issues in individual Form 4s
                        continue
                elif len([f for f in filings_list if not f.get("is_form4_transaction")]) < max_other_filings: # Limit other filings too
                    filings_list.append({"is_form4_transaction": False, "ticker": ticker_symbol, "filing_date": date_str, "form_type": form, "document_link": f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{acc_no_dashless}/{doc_name}", "summary_link": idx_link, "document_content_for_llm": document_content_for_llm})
            
            if not filings_list and xml_fetches > 0: return [{"error": f"SEC: {xml_fetches} Form 4s for {ticker_symbol}, but no transactions parsed or other issues."}]
            if not filings_list: return [{"error": f"SEC: No relevant filings for {ticker_symbol} (CIK:{cik_padded})."}]
        else: return [{"error": f"SEC: No recent filings data for {ticker_symbol} (CIK:{cik_padded})."}]
    except requests.exceptions.HTTPError as e: return [{"error": f"SEC HTTP error ({ticker_symbol}, CIK:{cik_padded}): {e}"}]
    except requests.exceptions.RequestException as e: return [{"error": f"SEC Request error ({ticker_symbol}, CIK:{cik_padded}): {e}"}]
    except Exception as e: return [{"error": f"SEC Unexpected error ({ticker_symbol}, CIK:{cik_padded}): {e}"}]
    
    # Ensure sorting handles potential missing 'filing_date' (though it should be present by now)
    filings_list.sort(key=lambda x: x.get('filing_date', '1900-01-01'), reverse=True)
    return filings_list

@st.cache_data(ttl=6*3600)
def fetch_inst_filings(ticker: str) -> list[dict]:
    """Fetches institutional holder data for a given ticker using yfinance."""
    try:
        df_holders = yf.Ticker(ticker).institutional_holders
        if df_holders is not None and not df_holders.empty:
            # Ensure columns exist before trying to convert them
            if 'Shares' in df_holders.columns: df_holders['Shares'] = pd.to_numeric(df_holders['Shares'], errors='coerce').fillna(0)
            if '% Out' in df_holders.columns: df_holders['% Out'] = pd.to_numeric(df_holders['% Out'], errors='coerce').fillna(0.0)
            if 'Date Reported' in df_holders.columns: df_holders['Date Reported'] = df_holders['Date Reported'].astype(str)
            return df_holders.to_dict("records")
        return [{"error": f"No yfinance institutional holder data for {ticker}."}]
    except Exception as e: return [{"error": f"yfinance institutional holders fetch failed for {ticker}: {e}"}]

@st.cache_data(ttl=4 * 3600)
def fetch_value_investing_io_data(ticker: str) -> dict:
    """Scrapes Peter Lynch Fair Value from ValueInvesting.io (experimental)."""
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

# Removed fetch_politician_trades as requested

# --- LLM Client and Agent Classes ---
class ModelClient:
    """Manages interaction with various LLM providers."""
    def __init__(self, api_key: str, provider: str = "openai"):
        self.api_key, self.provider = api_key, provider
        models = {"openai": "gpt-4o", "deepseek": "deepseek-reasoner"}
        if not api_key: raise ValueError("API key required for LLM.")
        self.model_name = models.get(provider)
        if not self.model_name: raise ValueError(f"Unsupported LLM provider: {provider}")
        if provider == "deepseek": self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        elif provider == "openai": self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        try:
            stream = self.client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt}], stream=True)
            response_content = "".join(c.choices[0].delta.content for c in stream if c.choices and c.choices[0].delta and c.choices[0].delta.content)
            return response_content
        except Exception as e: raise Exception(f"LLM Error ({self.provider}, {self.model_name}): {e}")

class PriceAgent:
    """Analyzes price action using SMAs, RSI, and Bollinger Bands."""
    def run(self, ticker: str, price_data_slice: pd.DataFrame) -> dict:
        required_data_points = 200

        if price_data_slice.empty or len(price_data_slice) < required_data_points:
            error_msg = f"Not enough data ({len(price_data_slice)} rows) for comprehensive PriceAgent analysis (requires {required_data_points})."
            return {
                "ticker": ticker, "price_signal": "hold", "sma50": np.nan, "sma200": np.nan,
                "rsi14": np.nan, "bb_upper": np.nan, "bb_lower": np.nan, "bb_signal": "hold",
                "price_confidence_score": 0.0, "price_error": error_msg
            }

        df = price_data_slice.copy()
        
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["SMA200"] = df["Close"].rolling(200).mean()

        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI14"] = 100 - (100 / (1 + rs))

        bb_period = 20
        bb_std_dev = 2
        df["BB_SMA"] = df["Close"].rolling(bb_period).mean()
        df["BB_STD"] = df["Close"].rolling(bb_period).std()
        df["BB_Upper"] = df["BB_SMA"] + (df["BB_STD"] * bb_std_dev)
        df["BB_Lower"] = df["BB_SMA"] - (df["BB_STD"] * bb_std_dev)

        latest = df.iloc[-1]
        
        signal = "hold"
        confidence_score = 0.0
        bb_signal = "hold"

        if pd.isna(latest.SMA50) or pd.isna(latest.SMA200) or pd.isna(latest.RSI14) or \
           pd.isna(latest.BB_Upper) or pd.isna(latest.BB_Lower):
            error_msg = "Some key indicators are NaN at the latest data point after calculation."
            return {
                "ticker": ticker, "price_signal": "hold",
                "sma50": float(latest.SMA50) if pd.notna(latest.SMA50) else np.nan,
                "sma200": float(latest.SMA200) if pd.notna(latest.SMA200) else np.nan,
                "rsi14": float(latest.RSI14) if pd.notna(latest.RSI14) else np.nan,
                "bb_upper": float(latest.BB_Upper) if pd.notna(latest.BB_Upper) else np.nan,
                "bb_lower": float(latest.BB_Lower) if pd.notna(latest.BB_Lower) else np.nan,
                "bb_signal": "hold", "price_confidence_score": 0.0, "price_error": error_msg
            }

        current_close = latest.Close

        if latest.SMA50 > latest.SMA200 and current_close > latest.SMA50:
            if len(df) >= 205 and df["SMA50"].iloc[-5] < df["SMA200"].iloc[-5]:
                signal = "buy"; confidence_score += 0.4
            else:
                signal = "buy"; confidence_score += 0.2
        elif latest.SMA50 < latest.SMA200 and current_close < latest.SMA50:
            if len(df) >= 205 and df["SMA50"].iloc[-5] > df["SMA200"].iloc[-5]:
                signal = "sell"; confidence_score -= 0.4
            else:
                signal = "sell"; confidence_score -= 0.2

        if latest.RSI14 < 30:
            if signal == "buy": confidence_score += 0.2
            elif signal == "hold":
                signal = "buy"; confidence_score += 0.1
        elif latest.RSI14 > 70:
            if signal == "sell": confidence_score -= 0.2
            elif signal == "hold":
                signal = "sell"; confidence_score -= 0.1

        if current_close < latest.BB_Lower:
            bb_signal = "buy"
            if signal == "buy": confidence_score += 0.1
            elif signal == "hold":
                signal = "buy"; confidence_score += 0.05
        elif current_close > latest.BB_Upper:
            bb_signal = "sell"
            if signal == "sell": confidence_score -= 0.1
            elif signal == "hold":
                signal = "sell"; confidence_score -= 0.05
        
        if signal == "hold":
            if latest.RSI14 < 40 and latest.RSI14 > 30:
                confidence_score += 0.05
            elif latest.RSI14 > 60 and latest.RSI14 < 70:
                confidence_score -= 0.05

        if confidence_score > 0.3:
            final_price_signal = "buy"
        elif confidence_score < -0.3:
            final_price_signal = "sell"
        else:
            final_price_signal = "hold"
            
        confidence_score = max(-1.0, min(1.0, confidence_score))
        return {
            "ticker": ticker,
            "sma50": float(latest.SMA50), "sma200": float(latest.SMA200), "rsi14": float(latest.RSI14),
            "bb_upper": float(latest.BB_Upper) if pd.notna(latest.BB_Upper) else np.nan,
            "bb_lower": float(latest.BB_Lower) if pd.notna(latest.BB_Lower) else np.nan,
            "bb_signal": bb_signal,
            "price_signal": final_price_signal,
            "price_confidence_score": float(confidence_score),
            "price_error": None
        }

class MomentumAgent:
    """Calculates 1-month and 12-month momentum and provides a signal."""
    def run(self, ticker: str, price_data_slice: pd.DataFrame) -> dict:
        required_data_points = 253 # Approx 1 year of trading days + 1 month

        if price_data_slice.empty or len(price_data_slice) < required_data_points:
            error_msg = f"Not enough data ({len(price_data_slice)} rows) for 1-year and 1-month momentum (requires {required_data_points})."
            return {
                "ticker": ticker, "momentum_signal": "hold", "momentum_1m": np.nan, "momentum_12m": np.nan,
                "momentum_confidence_score": 0.0, "momentum_error": error_msg
            }

        df = price_data_slice.copy()

        if 'Close' not in df.columns or not pd.api.types.is_numeric_dtype(df['Close']):
            error_msg = "Price data is missing 'Close' column or not numeric for momentum calculation."
            return {
                "ticker": ticker, "momentum_signal": "hold",
                "momentum_1m": np.nan, "momentum_12m": np.nan,
                "momentum_confidence_score": 0.0, "momentum_error": error_msg
            }
        
        P_t = df["Close"].iloc[-1]
        P_1m_series = df["Close"].shift(21) # Approx 1 month
        P_12m_series = df["Close"].shift(252) # Approx 1 year

        P_1m = P_1m_series.iloc[-1]
        P_12m = P_12m_series.iloc[-1]

        m1 = ((P_t / P_1m) - 1) if pd.notna(P_1m) and P_1m != 0 else np.nan
        m12 = ((P_t / P_12m) - 1) if pd.notna(P_12m) and P_12m != 0 else np.nan

        signal = "hold"
        confidence_score = 0.0

        STRONG_POSITIVE_MOMENTUM_THRESHOLD = 0.10
        MODERATE_POSITIVE_MOMENTUM_THRESHOLD = 0.03
        STRONG_NEGATIVE_MOMENTUM_THRESHOLD = -0.10
        MODERATE_NEGATIVE_MOMENTUM_THRESHOLD = -0.03

        if pd.notna(m1) and pd.notna(m12):
            raw_combined_momentum = (m1 + m12) / 2
            scaled_confidence = raw_combined_momentum * 5.0 # Scale to roughly -1.0 to 1.0 range
            confidence_score = max(-1.0, min(1.0, scaled_confidence))

            if confidence_score > 0.3:
                signal = "buy"
            elif confidence_score < -0.3:
                signal = "sell"
            else:
                signal = "hold"

        return {
            "ticker": ticker,
            "momentum_1m": float(m1) if pd.notna(m1) else np.nan,
            "momentum_12m": float(m12) if pd.notna(m12) else np.nan,
            "momentum_signal": signal,
            "momentum_confidence_score": float(confidence_score),
            "momentum_error": None
        }

class VolatilityAgent:
    """Analyzes stock volatility (Beta and Annualized Volatility) to gauge risk."""
    def run(self, ticker: str, data: dict, price_data_slice: pd.DataFrame = None) -> dict:
        beta_val = data.get("ticker_info", {}).get("beta")
        beta = float(beta_val) if isinstance(beta_val, (int, float)) else 1.0 # Default beta to 1.0 if not available

        ann_vol = np.nan
        vol_weight = 0.0
        volatility_signal = "hold"
        volatility_confidence_score = 0.0
        volatility_error = None

        if price_data_slice is not None and not price_data_slice.empty and len(price_data_slice) > 1:
            if 'Close' not in price_data_slice.columns or not pd.api.types.is_numeric_dtype(price_data_slice['Close']):
                volatility_error = "Price data is missing 'Close' column or not numeric for volatility calculation."
            else:
                ret = np.log(price_data_slice.Close / price_data_slice.Close.shift(1)).dropna()

                if not ret.empty:
                    daily_std = ret.std()
                    if daily_std > 0:
                        ann_vol = float(daily_std * np.sqrt(252)) # Annualize daily volatility
                        vol_weight = float(1 / ann_vol) # Inverse volatility weighting for portfolio construction
                    else:
                        volatility_error = "Daily returns standard deviation is zero (no price movement)."
                else:
                    volatility_error = "Not enough valid returns to calculate historical volatility."
        else:
            volatility_error = "Not enough price data for historical volatility calculation."

        # Factor in Beta for signal and confidence
        if beta > 1.2: # High beta implies higher risk, potentially a sell signal from volatility perspective
            volatility_signal = "sell"; volatility_confidence_score -= (beta - 1.2) * 0.5
        elif beta < 0.8: # Low beta implies lower risk, potentially a buy signal from volatility perspective
            volatility_signal = "buy"; volatility_confidence_score += (0.8 - beta) * 0.5
        else:
            volatility_signal = "hold"

        # Factor in Annualized Volatility
        if pd.notna(ann_vol):
            HIGH_VOL_THRESHOLD = 0.30 # Example: 30% annualized vol
            LOW_VOL_THRESHOLD = 0.15 # Example: 15% annualized vol

            if ann_vol > HIGH_VOL_THRESHOLD: # Higher volatility can be negative
                volatility_confidence_score -= (ann_vol - HIGH_VOL_THRESHOLD) * 1.0
            elif ann_vol < LOW_VOL_THRESHOLD: # Lower volatility can be positive (more stable)
                volatility_confidence_score += (LOW_VOL_THRESHOLD - ann_vol) * 1.0

        volatility_confidence_score = max(-1.0, min(1.0, volatility_confidence_score)) # Clamp score

        # Refine signal based on combined confidence
        if volatility_confidence_score > 0.2:
            volatility_signal = "buy"
        elif volatility_confidence_score < -0.2:
            volatility_signal = "sell"
        else:
            volatility_signal = "hold"

        return {
            "ticker": ticker,
            "beta": float(beta),
            "annual_vol": float(ann_vol) if pd.notna(ann_vol) else np.nan,
            "vol_weight": float(vol_weight) if pd.notna(vol_weight) else np.nan,
            "volatility_signal": volatility_signal,
            "volatility_confidence_score": float(volatility_confidence_score),
            "volatility_error": volatility_error
        }

class SentimentAgent:
    """Analyzes news sentiment for a ticker using an LLM."""
    def __init__(self, client):
        self.client = client

    def run(self, ticker: str, data: dict) -> dict:
        news, news_err = data.get("news", []), data.get("news_fetch_status_error")

        if news_err:
            return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": news_err}

        valid_news = [item for item in news if isinstance(item, dict) and "error" not in item]

        if not valid_news:
            err_msg = news[0].get("error") if news and isinstance(news[0], dict) and "error" in news[0] else "No valid news articles found for sentiment analysis."
            return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": err_msg}

        content_for_llm = []
        co_name = data.get("ticker_info", {}).get('longName', ticker)
        
        MAX_NEWS_ARTICLES_FOR_LLM = 10 # Limit number of articles for LLM context window

        for item in valid_news[:MAX_NEWS_ARTICLES_FOR_LLM]:
            title = item.get('title', '').strip()
            description = item.get('description', '').strip()
            content_snippet = item.get('content_snippet', '').replace('[+... chars]', '').strip()
            publisher = item.get('publisher', 'N/A').strip()
            source_api = item.get('source_api', 'Unknown').strip()
            publish_time = item.get('publish_time_readable', 'N/A').strip()

            main_text = ""
            if content_snippet and len(content_snippet) > 50:
                main_text = f"Content: {content_snippet}"
            elif description and len(description) > 50:
                main_text = f"Description: {description}"
            
            if main_text:
                snippet = f"Headline: {title}"
                if main_text: snippet += f" | {main_text}"
                if publisher != 'N/A': snippet += f" (Source: {publisher} via {source_api})"
                if publish_time != 'N/A': snippet += f" (Published: {publish_time})"
                content_for_llm.append(snippet)

        if not content_for_llm:
            return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": "No processable news articles with sufficient content for sentiment analysis."}

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

        final_err = None
        if news_err: final_err = f"News fetch issues: {news_err}"
        if llm_err: final_err = (f"{final_err} | LLM issues: {llm_err}" if final_err else f"LLM issues: {llm_err}")

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
    """Generates a concise summary of recent news articles using an LLM."""
    def __init__(self, client): self.client = client
    def run(self, ticker: str, data: dict) -> dict:
        news, co_name, news_fetch_err = data.get("news",[]), data.get("ticker_info",{}).get('longName',ticker), data.get("news_fetch_status_error")
        if news_fetch_err: return {"ticker":ticker, "news_summary":"Summary skipped due to news fetch issues.", "news_summary_error":news_fetch_err}
        if not news or (isinstance(news[0],dict) and "error" in news[0] and not any("error" not in item for item in news)):
            err = news[0]["error"] if news and isinstance(news[0],dict) and "error" in news[0] else "No news for summary."
            return {"ticker":ticker, "news_summary":"No news for summary.", "news_summary_error":err}
        
        # Prioritize Yahoo Finance news, then NewsAPI, for diversity and relevance
        y_news = [item for item in news if item.get('source_api')=='Yahoo Finance' and "error" not in item][:5]
        n_news = [item for item in news if item.get('source_api')=='NewsAPI.org' and "error" not in item][:5]
        
        # Combine and interleave for a mix of sources
        sel_news = []
        for i in range(max(len(y_news), len(n_news))):
            if i < len(y_news): sel_news.append(y_news[i])
            if i < len(n_news): sel_news.append(n_news[i])
        
        final_snips, titles = [], set()
        for item in sel_news:
            if len(final_snips) >= 7: break # Limit total snippets for LLM
            title = item.get('title','').strip()
            # Prefer content_snippet if available and substantial, otherwise use description
            content_or_desc = item.get('content_snippet', '').replace('[+... chars]','').strip()
            if not content_or_desc and item.get('description'):
                content_or_desc = item.get('description', '').strip()
            
            if title in titles: continue; # Avoid duplicate titles
            titles.add(title) # Add to set after check
            
            text = f"Title: {title}"
            if content_or_desc: text += f" | Content: {content_or_desc}"
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
    """Analyzes fundamental data (Market Cap, FCF Yield, Piotroski Score)."""
    def run(self, ticker: str, data: dict) -> dict:
        s = data.get("ticker_info",{});
        
        # Safely get and convert numerical values
        mc = s.get("marketCap")
        fcf = s.get("freeCashflow")
        roe = s.get("returnOnEquity")
        de = s.get("debtToEquity")

        mc_c = float(mc) if isinstance(mc,(int,float)) and mc is not None else 1 # Avoid division by zero
        fcf_c = float(fcf) if isinstance(fcf,(int,float)) and fcf is not None else 0
        roe_c = float(roe) if isinstance(roe,(int,float)) and roe is not None else 0
        de_c = float(de) if isinstance(de,(int,float)) and de is not None else 1000 # High default to penalize missing

        fcy = (fcf_c / mc_c) if mc_c != 0 else 0.0 # Free Cash Flow Yield
        
        # Simple Piotroski-like score based on available fundamental metrics
        ps_score = 0
        if roe_c > 0.01: ps_score += 1 # Positive Return on Equity
        if de_c < 100: ps_score += 1 # Low Debt to Equity
        if fcf_c > 0: ps_score += 1 # Positive Free Cash Flow
        
        sig = "hold"
        if ps_score >= 2: sig = "buy"
        elif ps_score == 0: sig = "sell"
        
        return {"ticker":ticker, "fcf_yield":float(fcy), "piotroski_score":int(ps_score), "fund_signal":sig}

class ValuationAgent:
    """Provides valuation insights based on PE ratio and a simple DCF estimate."""
    def run(self, ticker: str, data: dict) -> dict:
        stats = data.get("ticker_info",{})
        hist = data.get("price_history")
        
        price_v = stats.get("currentPrice")
        if price_v is None and hist is not None and not hist.empty:
            price_v = hist.Close.iloc[-1] # Fallback to last close from history
        
        curr_p = float(price_v) if isinstance(price_v,(int,float)) and price_v > 0 else None
        
        if curr_p is None:
            return {"ticker":ticker, "forward_pe":None, "relative_pe_signal":"hold", "dcf_fair_price":np.nan, "dcf_signal":"hold", "valuation_error":"Current price unavailable."}
        
        pe_v = stats.get("forwardPE"); pe = float(pe_v) if isinstance(pe_v,(int,float)) else None; rel_sig = "hold"
        if pe is not None and pe > 0: rel_sig = "buy" if pe < 15 else ("sell" if pe > 25 else "hold") # Simple PE heuristic
        
        fcf_v = stats.get("freeCashflow")
        mc_v = stats.get("marketCap")
        fcf = (float(fcf_v) if isinstance(fcf_v,(int,float)) and fcf_v is not None else None)
        mc = (float(mc_v) if isinstance(mc_v,(int,float)) and mc_v is not None else None)

        # Simple DCF-like estimate: Current Price * (1 + FCF Yield)
        # This assumes FCF grows company value, very simplistic
        fcy = (fcf / mc) if fcf is not None and mc is not None and mc != 0 else 0.0
        fp_est = curr_p * (1 + fcy) if curr_p is not None else np.nan # Estimated Fair Price
        
        dcf_sig = "hold"
        if pd.notna(fp_est) and curr_p is not None:
            if fp_est > curr_p * 1.15: dcf_sig = "buy" # 15% upside
            elif fp_est < curr_p * 0.85: dcf_sig = "sell" # 15% downside

        return {"ticker":ticker, "forward_pe":pe, "relative_pe_signal":rel_sig, "dcf_fair_price":float(fp_est) if pd.notna(fp_est) else np.nan, "dcf_signal":dcf_sig, "valuation_error":None}

class AnalystRatingAgent:
    """Processes analyst recommendations and target price upside."""
    def run(self, ticker: str, data: dict) -> dict:
        info = data.get("ticker_info",{})
        hist = data.get("price_history")
        
        price_v = info.get("currentPrice")
        if price_v is None and hist is not None and not hist.empty:
            price_v = hist.Close.iloc[-1] # Fallback to last close from history
        
        curr_p = float(price_v) if isinstance(price_v,(int,float)) and price_v > 0 else None
        
        if curr_p is None:
            return {"ticker":ticker, "analyst_buy_pct_inferred":0.5, "target_upside":0.0, "yfinance_recommendation":"N/A", "analyst_signal":"hold", "analyst_error":"Current price unavailable."}
        
        target_v = info.get("targetMeanPrice"); target_m = float(target_v) if isinstance(target_v,(int,float)) else None
        rec = str(info.get("recommendationKey","hold")).lower(); # Lowercase for easier comparison
        upside = 0.0
        if target_m is not None and curr_p > 0: upside = (target_m / curr_p) -1
        
        sig = "hold"
        if rec in ["buy","strong_buy"] and upside > 0.10: sig = "buy"
        elif rec == "buy" and upside > 0.05: sig = "buy"
        elif rec in ["sell","strong_sell","underperform"] and upside < -0.05: sig = "sell"
        elif upside > 0.20: sig = "buy" # Strong upside even if recommendation is just 'hold'
        elif upside < -0.15: sig = "sell" # Significant downside
        
        # Infer a buy percentage from Yahoo Finance's recommendation key (heuristic)
        buy_pct = {"strong_buy":0.9, "buy":0.7, "hold":0.5, "underperform":0.3, "sell":0.1}.get(rec,0.5)
        
        return {"ticker":ticker, "analyst_buy_pct_inferred":float(buy_pct), "target_upside":float(upside), "yfinance_recommendation":rec, "analyst_signal":sig, "analyst_error":None}

class SECFilingAgent:
    """Analyzes SEC filings, particularly Form 4 for insider trading."""
    def run(self, ticker: str, data: dict) -> dict:
        filings, err = data.get("sec_all_filings_raw",[]), None
        if not filings or (isinstance(filings[0],dict) and "error" in filings[0]):
            err = filings[0].get("error") if filings and isinstance(filings[0],dict) else f"SEC: No raw filings for {ticker}."
            return {"ticker":ticker, "sec_net_insider_shares_1y":0, "sec_insider_buy_value_1y":0, "sec_insider_sell_value_1y":0, "sec_filings_signal":"hold", "sec_filings_error":err, "sec_recent_form4_transactions":[], "sec_other_recent_filings":[]}
        
        net_s, buy_v, sell_v, form4, others = 0,0,0,[],[]
        for f in filings:
            if not isinstance(f,dict) or "error" in f: continue
            if f.get("is_form4_transaction"):
                form4.append(f)
                s, p = f.get("shares", 0.0), f.get("price_per_share")
                if not isinstance(s, (int, float)): s = 0.0
                
                # Check for transaction codes 'P' (purchase) and 'S' (sale) and acquisition/disposition codes
                if f.get("transaction_code")=="P" and f.get("acq_disp_code")=="A": # Purchase (Acquired)
                    net_s += s
                    if isinstance(p,(int,float)) and s!=0: buy_v += s*p
                elif f.get("transaction_code")=="S" and f.get("acq_disp_code")=="D": # Sale (Disposed)
                    net_s -= s
                    if isinstance(p,(int,float)) and s!=0: sell_v += s*p
            else:
                others.append(f)
        
        sig = "hold"
        # Heuristic for insider trading signal
        if net_s > 2000 or buy_v > 200000: # Significant net shares bought or large value of buys
            sig = "buy"
        elif net_s < -2000 or sell_v > 200000: # Significant net shares sold or large value of sells
            sig = "sell"
        
        return {"ticker":ticker, "sec_net_insider_shares_1y":int(net_s), "sec_insider_buy_value_1y":round(buy_v,2), "sec_insider_sell_value_1y":round(sell_v,2), "sec_filings_signal":sig, "sec_filings_error":None, "sec_recent_form4_transactions":form4[:10], "sec_other_recent_filings":others[:10]}

class SECSummaryAgent:
    """Generates an LLM-powered summary of key SEC filings (10-K, 10-Q, 8-K)."""
    def __init__(self, client):
        self.client = client

    def run(self, ticker: str, data: dict) -> dict:
        filings_raw = data.get("sec_all_filings_raw", [])
        co_name = data.get("ticker_info", {}).get('longName', ticker)

        if not self.client:
            return {"ticker": ticker, "sec_summary": "LLM client not available for SEC summary.", "sec_summary_error": "LLM not configured."}
        
        # Filter for non-Form 4 filings with content placeholders (10-K, 10-Q, 8-K)
        relevant_filings = [
            f for f in filings_raw if not f.get("is_form4_transaction") and f.get("document_content_for_llm")
        ]
        
        if not relevant_filings:
            return {"ticker": ticker, "sec_summary": "No relevant non-Form 4 filings with content found for summary.", "sec_summary_error": None}

        # Sort filings to prioritize 8-K, then 10-Q, then 10-K, and then by most recent date
        sorted_filings = sorted(
            relevant_filings,
            key=lambda x: (
                0 if x.get('form_type') == '8-K' else # Highest priority
                1 if x.get('form_type') == '10-Q' else
                2 if x.get('form_type') == '10-K' else
                3, # Lowest priority for other forms
                x.get('filing_date_str', '1900-01-01') # Secondary sort by date (oldest first if multiple of same form_type)
            ),
            reverse=False # Keep oldest of highest priority forms
        )
        
        # Take up to 3 most relevant filings for the LLM prompt to manage token limits and focus
        filings_for_llm = sorted_filings[:3]

        content_for_llm = []
        for f in filings_for_llm:
            content_for_llm.append(
                f"Form Type: {f.get('form_type', 'N/A')} (Filed: {f.get('filing_date_str', 'N/A')})\n"
                f"Content: {f.get('document_content_for_llm', 'No content placeholder.')}\n"
                f"Link: {f.get('summary_link', '#')}\n"
            )
        
        prompt = f"""
        As an expert financial analyst, summarize the key highlights and any significant positive or negative developments related to {co_name} ({ticker}) based on the following SEC filing information.
        Pay particular attention to:
        - Any mentions of stock issuance, buybacks, or changes in capital structure.
        - Major operational changes, strategic shifts, or significant events (e.g., acquisitions, litigation, product failures/successes).
        - Any forward-looking statements that imply significant financial impact.
        - Important financial performance highlights (positive or negative trends).

        Provide a concise summary (max 250 words) and identify any clear implications for the stock. If no significant events related to stock movement are found, state that.

        SEC Filings:
        """ + "\n---\n".join(content_for_llm) + "\n---"

        summary = "Could not generate summary."
        llm_err = None

        try:
            resp = self.client.generate(prompt).strip()
            if resp.startswith("Error:"):
                llm_err = resp
            else:
                summary = resp
        except Exception as e:
            llm_err = f"LLM SEC summary call failed: {str(e)[:150]}"
            
        return {
            "ticker": ticker,
            "sec_summary_llm": summary,
            "sec_summary_error": llm_err
        }


class InstitutionalHoldingsAgent:
    """Analyzes institutional ownership data from yfinance."""
    def run(self, ticker: str, data: dict) -> dict:
        holdings, err = data.get("institutional_holdings",[]), None
        if holdings and isinstance(holdings[0],dict) and "error" in holdings[0]:
            err = holdings[0]["error"]
            return {"ticker":ticker, "inst_num_holders":0, "inst_total_shares_held":0, "inst_total_pct_out":0.0, "inst_holdings_signal":"hold", "inst_holdings_error":err, "inst_top_holders":[]}
        
        num_h, total_s, total_pct, top_h = 0,0,0.0,[]
        if holdings:
            valid_h = [d for d d in holdings if isinstance(d,dict) and "error" not in d]
            if valid_h:
                num_h = len(valid_h)
                try:
                    total_s = sum(d.get('Shares',0) for d in valid_h)
                    total_pct = sum(d.get('% Out',0.0) for d in valid_h)
                    # Ensure 'Shares' for sorting are numeric and handle potential NaNs before sorting
                    for d in valid_h:
                        d['Shares'] = pd.to_numeric(d.get('Shares'), errors='coerce').fillna(0)
                    # Get top 10 holders by shares for display
                    top_h = sorted(valid_h, key=lambda x: x.get('Shares',0), reverse=True)[:10]
                except Exception as e:
                    err = f"Error processing institutional holdings data: {e}"
            elif not err:
                err = "No valid institutional holdings data found after filtering."
        
        sig = "hold"
        # Heuristic for institutional holdings signal
        inst_confidence_score = 0.0
        if not err:
            if total_pct > 0.60: sig = "buy" # High institutional concentration
            elif total_pct < 0.10 and num_h > 0: sig = "sell" # Low institutional interest for a public company
            
            # Refine confidence based on percentage of shares held by institutions
            if total_pct > 0.75: inst_confidence_score = 0.8 # Very high conviction
            elif total_pct > 0.60: inst_confidence_score = 0.5
            elif total_pct < 0.05 and num_h > 0: inst_confidence_score = -0.8 # Very low interest
            elif total_pct < 0.10 and num_h > 0: inst_confidence_score = -0.5
        
        return {"ticker":ticker, "inst_num_holders":num_h, "inst_total_shares_held":int(total_s), "inst_total_pct_out":float(total_pct), "inst_holdings_signal":sig, "inst_holdings_error":err, "inst_top_holders":top_h, "inst_confidence_score": float(inst_confidence_score)}

# Removed PoliticianFilingsAgent as requested

class ValueInvestingIOAgent:
    """Scrapes Peter Lynch Fair Value from ValueInvesting.io and provides a signal (experimental)."""
    def run(self, ticker: str, data: dict) -> dict:
        vi, err = data.get("value_investing_io_data",{}), data.get("value_investing_io_data",{}).get("error")
        fv, site_mp, up_pct, val_date, text = vi.get("vi_fair_value"), vi.get("vi_site_market_price"), vi.get("vi_upside_percent"), vi.get("vi_valuation_date"), vi.get("vi_full_text")
        
        # Get current price from primary data source (yfinance) for comparison
        curr_pyf_val = data.get("ticker_info",{}).get("currentPrice")
        if curr_pyf_val is None and data.get("price_history") is not None and not data["price_history"].empty:
            curr_pyf_val = data["price_history"].Close.iloc[-1]
        curr_pyf = float(curr_pyf_val) if isinstance(curr_pyf_val,(int,float)) and curr_pyf_val > 0 else None
        
        sig = "hold"
        vi_confidence_score = 0.0
        
        if not err and fv is not None and curr_pyf is not None:
            mos = 0.15 # Margin of safety threshold (15%)
            
            if up_pct is not None: # Use direct upside % from site if available
                if up_pct > (mos*100+5): sig="buy"; vi_confidence_score = 0.8 # e.g., > 20% upside
                elif up_pct < -(mos*100+5): sig="sell"; vi_confidence_score = -0.8 # e.g., > 20% downside
                elif up_pct > mos*100: sig="buy"; vi_confidence_score = 0.5 # e.g., > 15% upside
                elif up_pct < -mos*100: sig="sell"; vi_confidence_score = -0.5 # e.g., > 15% downside
            else: # Fallback: calculate upside using fair value vs current price
                if curr_pyf < fv*(1-mos): sig="buy"; vi_confidence_score = 0.6
                elif curr_pyf > fv*(1+mos): sig="sell"; vi_confidence_score = -0.6
        
        return {"ticker":ticker, "vi_fair_value_estimate":fv, "vi_site_market_price":site_mp, "vi_upside_percent":up_pct, "vi_valuation_date":val_date, "vi_valuation_text_display":text, "vi_signal":sig, "vi_data_error":err, "vi_confidence_score": float(vi_confidence_score)}


class PortfolioAgent:
    """Aggregates signals from various agents and provides a final buy/sell/hold decision."""
    WEIGHTS = {
        "price": 1.0, "momentum": 0.8, "volatility": 0.2, 
        "sentiment": 0.7, "fund": 0.9, "valuation_dcf": 0.6, "valuation_pe": 0.4,
        "sec_filings": 0.6, "sec_summary": 0.7,
        "inst_holdings": 0.3, "analyst": 0.5,
        "vi_signal": 0.8 # Politician filings removed
    }

    def run(self, ticker: str, signals: list[dict], agent_weights: dict = None) -> dict:
        curr_w, total_score, sum_w, agg_s = agent_weights or self.WEIGHTS, 0,0,{}
        for s_dict in signals:
            if isinstance(s_dict, dict): agg_s.update(s_dict)
        
        # Mapping signals to their corresponding weights and confidence scores (if available)
        s_map = {
            "price_signal": ("price", "price_confidence_score"),
            "momentum_signal": ("momentum", "momentum_confidence_score"),
            "volatility_signal": ("volatility", "volatility_confidence_score"),
            "sentiment_signal": ("sentiment", "sentiment_confidence_score"),
            "fund_signal": ("fund", None), # No direct confidence score in current FundamentalsAgent output
            "dcf_signal": ("valuation_dcf", None),
            "relative_pe_signal": ("valuation_pe", None),
            "sec_filings_signal": ("sec_filings", None),
            "sec_summary_llm": ("sec_summary", None), # Infer score from summary text later or use default contribution
            "inst_holdings_signal": ("inst_holdings", "inst_confidence_score"),
            "analyst_signal": ("analyst", None),
            "vi_signal": ("vi_signal", "vi_confidence_score")
        }

        for s_key, (w_key, conf_key) in s_map.items():
            s_val = agg_s.get(s_key)
            w = curr_w.get(w_key, 0)
            
            # Special handling for SEC Summary if it's a text summary, or use default signal value
            if s_key == "sec_summary_llm" and s_val and w > 0:
                # If LLM summary generation had an error, treat it neutrally
                if agg_s.get("sec_summary_error"):
                    raw_score = 0
                # Basic sentiment analysis of the summary text (can be improved by more sophisticated NLP)
                elif "negative" in s_val.lower() and "no significant events" not in s_val.lower():
                    raw_score = -0.5
                elif "positive" in s_val.lower() and "no significant events" not in s_val.lower():
                    raw_score = 0.5
                else: # Default small positive if summary exists but no strong sentiment words
                    raw_score = 0.1
                total_score += raw_score * w
                sum_w += w
            elif s_val and w > 0 and s_val in ["buy", "hold", "sell"]:
                raw_score = {"buy":1, "hold":0, "sell":-1}.get(s_val,0)
                
                # Incorporate confidence score from agents if available
                if conf_key and pd.notna(agg_s.get(conf_key)):
                    agent_confidence = agg_s.get(conf_key)
                    # Weight the raw signal by its own agent's confidence
                    total_score += (raw_score * agent_confidence) * w
                    sum_w += w * agent_confidence # Sum weights by effective confidence
                else:
                    total_score += raw_score * w
                    sum_w += w

        comp_score = (total_score/sum_w) if sum_w else 0.0
        decision = "buy" if comp_score > 0.15 else ("sell" if comp_score < -0.15 else "hold")
        return {"ticker":ticker, "composite_score":comp_score, "final_decision":decision}

class AITraderAgent:
    """Decides actual buy/sell trades based on composite analysis and portfolio allocation rules."""
    def __init__(self, llm_client: ModelClient, stock_universe: dict):
        self.llm_client = llm_client
        self.stock_universe = stock_universe

    def _generate_trade_reason(self, ticker: str, decision: str, analysis: dict) -> str:
        """Generates an LLM-powered justification for a trade decision."""
        if not self.llm_client:
            return "LLM client not available for justification."

        co_name = analysis.get('ticker_info', {}).get('longName', ticker)
        score = analysis.get('composite_score', 0)
        news_summary = analysis.get('news_summary', 'No news summary available.')
        sec_summary = analysis.get('sec_summary_llm', 'No SEC filing summary available.')

        prompt = f"""
        As an AI Portfolio Manager, you have decided to '{decision.upper()}' shares of {co_name} ({ticker}).
        The composite analysis score was {score:.2f}.
        Here is a summary of recent news: "{news_summary}"
        Here is a summary of recent SEC filings: "{sec_summary}"

        Based on this, provide a single, concise sentence explaining the primary reason(s) for this trade.
        Example: "Initiating a position due to strong positive news sentiment, bullish analyst ratings, and insider buying activity."
        Example: "Selling to reduce exposure after a significant price run-up, weakening momentum signals, and recent negative SEC filing disclosures."

        Generate the reason for the {decision.upper()} decision now:
        """
        try:
            reason = self.llm_client.generate(prompt).strip()
            return reason
        except Exception as e:
            return f"Could not generate reason due to LLM error: {e}"

    def _is_safe(self, analysis: dict) -> bool:
        """Determines if a stock is 'safe' based on predefined criteria (e.g., Mega-cap and low beta)."""
        info = analysis.get("ticker_info", {})
        market_cap = info.get("marketCap", 0)
        beta = info.get("beta", 1.0)
        # Define "safe" as Mega-cap (>$200B) and low beta (<1.0)
        return isinstance(market_cap, (int, float)) and market_cap > 200e9 and isinstance(beta, (int, float)) and beta < 1.0

    def run(self, portfolio_state: dict, analysis_results: dict):
        """
        Executes trading decisions based on AI analysis and portfolio allocation rules.
        Prioritizes selling over buying and attempts to rebalance between 'safe' and 'risky' assets.
        """
        trades_to_make = []
        cash = portfolio_state['cash']
        holdings = list(portfolio_state['holdings']) # Create a mutable copy

        tickers_in_portfolio = {h['ticker'] for h in holdings}
        
        # --- 1. Process Sells: Liquidate positions based on AI signal ---
        # Iterate in reverse to safely remove items from a list while iterating
        for i, holding in reversed(list(enumerate(holdings))):
            ticker = holding['ticker']
            analysis = analysis_results.get(ticker)
            if not analysis or analysis.get('error'):
                continue # Skip if no analysis results or there was an error for this holding

            if analysis.get('final_decision') == 'sell':
                reason = self._generate_trade_reason(ticker, 'sell', analysis)
                price = analysis.get('current_price_display')
                if isinstance(price, (int, float)) and price > 0:
                    trades_to_make.append({
                        "ticker": ticker, "type": "sell", "quantity": holding['quantity'],
                        "price": price, "reason": reason
                    })
                    cash += holding['quantity'] * price # Add cash from sale
                    holdings.pop(i) # Remove holding from portfolio
                # else: If price is invalid, we don't try to sell

        # --- 2. Calculate current portfolio value after sells for rebalancing ---
        current_holdings_value = 0
        for h in holdings:
            price = analysis_results.get(h['ticker'], {}).get('current_price_display')
            if isinstance(price, (int, float)):
                current_holdings_value += h['quantity'] * price
        total_portfolio_value = cash + current_holdings_value

        # Define target allocation values (e.g., 60% safe, 40% risky)
        target_safe_value = total_portfolio_value * 0.60
        target_risky_value = total_portfolio_value * 0.40

        # Calculate current values of safe and risky holdings after sells
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

        # --- 3. Process Buys: Allocate remaining cash based on AI signal and portfolio balance ---
        buy_candidates = sorted(
            [res for res in analysis_results.values() 
             if res.get('final_decision') == 'buy' 
             and res.get('ticker') not in tickers_in_portfolio # Don't buy what's already held (for simplicity, avoid averaging in)
             and not res.get('error')],
            key=lambda x: x.get('composite_score', 0), reverse=True # Prioritize highest composite scores
        )
        
        # Define a base investment size. This prevents buying tiny fractions of shares
        # and ensures enough cash for multiple trades.
        MIN_INVESTMENT_PER_STOCK = 200 # Minimum dollar amount to invest in a single stock
        investment_per_stock_target = max(MIN_INVESTMENT_PER_STOCK, cash * 0.20) # Try to invest up to 20% of available cash per stock

        for candidate in buy_candidates:
            if cash < MIN_INVESTMENT_PER_STOCK: # Stop if not enough cash left for even a minimum trade
                break

            price = candidate.get('current_price_display')
            if not isinstance(price, (int, float)) or price <= 0:
                continue # Skip if price is invalid

            # Determine category (safe/risky) for allocation
            is_safe_candidate = self._is_safe(candidate)
            
            should_buy = False
            investment_amount = 0

            # Allocate towards under-represented categories first
            if is_safe_candidate and current_safe_value < target_safe_value:
                should_buy = True
                investment_amount = min(investment_per_stock_target, target_safe_value - current_safe_value, cash)
            elif not is_safe_candidate and current_risky_value < target_risky_value:
                should_buy = True
                investment_amount = min(investment_per_stock_target, target_risky_value - current_risky_value, cash)
            else:
                continue # If allocation is already met for this category, or it doesn't fit criteria, skip

            if should_buy and investment_amount > price: # Ensure we can buy at least one share
                quantity_to_buy = investment_amount / price
                reason = self._generate_trade_reason(candidate['ticker'], 'buy', candidate)
                trades_to_make.append({
                    "ticker": candidate['ticker'], "type": "buy", "quantity": quantity_to_buy,
                    "price": price, "reason": reason
                })
                cash -= investment_amount # Deduct the allocated investment amount
                
                # Update current category values for subsequent allocations in the same run
                if is_safe_candidate:
                    current_safe_value += investment_amount
                else:
                    current_risky_value += investment_amount
                
                tickers_in_portfolio.add(candidate['ticker']) # Add to set to avoid re-buying in the same run

        return trades_to_make

# --- Orchestrator and Backtesting ---
def run_live_analysis(tickers, llm_client, configs):
    """
    Orchestrates the data fetching and agent analysis for a list of tickers.
    Returns a dictionary of analysis results per ticker.
    """
    results = {}
    progress_bar = st.progress(0, text="Starting analysis...")
    
    # Weights for the simple backtest simulation (only uses Price, Momentum, Volatility agents)
    default_live_backtest_weights = {
        "price": 1.0, "momentum": 0.8, "volatility": 0.3, 
        "sentiment": 0., "fund": 0., "valuation_dcf": 0., "valuation_pe": 0.,
        "sec_filings": 0., "sec_summary": 0.,
        "inst_holdings": 0., "analyst": 0.,
        "politician_filings": 0., "vi_signal": 0.
    }

    for i, t in enumerate(tickers):
        progress_text = f"Analyzing {t}... ({i+1}/{len(tickers)})"
        progress_bar.progress((i + 1) / len(tickers), text=progress_text)
        
        # --- Fetch Price History ---
        price_history_full = fetch_price_history(t, period="max")
        if price_history_full.empty:
            results[t] = {
                "error": f"Price history unavailable for {t}. This can happen for invalid tickers, delisted stocks, or temporary data provider issues.",
                "ticker": t, "final_decision": "error", "composite_score": 0
            }
            continue

        # --- Fetch Ticker Info ---
        ticker_info = fetch_ticker_info(t)
        if ticker_info.get("_error"): # Check for errors from the fetch function itself
            results[t] = {"error": ticker_info["_error"], "ticker": t, "final_decision": "error", "composite_score": 0}
            continue
        elif not ticker_info or not ticker_info.get("financialCurrency"): # Check if essential info is missing from valid response
             err_msg = f"Core ticker info (e.g., currency) unavailable for {t}. This likely indicates an invalid ticker, a delisted stock, or persistent issues with yfinance data for this symbol."
             results[t] = {"error": err_msg, "ticker": t, "final_decision": "error", "composite_score": 0}
             continue
        
        # Determine current price, preferring live data if available, else last close from history
        current_price_for_ticker = ticker_info.get("currentPrice")
        if current_price_for_ticker is None and not price_history_full.empty:
            current_price_for_ticker = price_history_full["Close"].iloc[-1]
        
        if current_price_for_ticker is None: # If still no current price, cannot proceed meaningfully
            results[t] = {
                "error": f"Current price could not be determined for {t}. Missing essential market data.",
                "ticker": t, "final_decision": "error", "composite_score": 0
            }
            continue

        company_name_for_news = ticker_info.get('longName', ticker_info.get('shortName', t))
        
        # --- Fetch News ---
        combined_news, news_fetch_msgs = [], []
        if configs["use_sentiment"]:
            yf_news = fetch_enriched_news(t, ticker_info)
            if yf_news and not (isinstance(yf_news[0],dict) and "error" in yf_news[0]): combined_news.extend(yf_news)
            elif yf_news and isinstance(yf_news[0],dict) and "error" in yf_news[0]: news_fetch_msgs.append(f"Yahoo News Error: {yf_news[0]['error']}")
            
            if llm_client and st.secrets.get("NEWSAPI_KEY"):
                api_news = fetch_comprehensive_news_from_api(t, company_name_for_news)
                if api_news and not (isinstance(api_news[0],dict) and "error" in api_news[0]): combined_news.extend(api_news)
                elif api_news and isinstance(api_news[0],dict) and "error" in api_news[0]: news_fetch_msgs.append(f"NewsAPI Error: {api_news[0]['error']}")
            elif configs["use_sentiment"] and not st.secrets.get("NEWSAPI_KEY"):
                news_fetch_msgs.append("NewsAPI Key missing.")

        # Deduplicate news by URL and sort by date
        seen_urls, dedup_news = set(), []
        for item in combined_news:
            if isinstance(item,dict) and "error" not in item:
                url = item.get('link') or item.get('url')
                if url and url not in seen_urls: dedup_news.append(item); seen_urls.add(url)
        if dedup_news: dedup_news.sort(key=lambda x: x.get('publish_datetime_utc') or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        
        news_status_bundle = " | ".join(news_fetch_msgs) if news_fetch_msgs else "News fetch OK"
        if not dedup_news and not news_fetch_msgs and configs["use_sentiment"]: news_status_bundle="No news articles found."

        # --- Assemble Data Bundle for Agents ---
        data_bundle = {
            "price_history":price_history_full,
            "ticker_info":ticker_info,
            "news":dedup_news,
            "news_fetch_status_error": news_status_bundle if any(kw in news_status_bundle.lower() for kw in ["error","failed","no news","missing","issue"]) else None,
            # Removed "politician_trades"
            "value_investing_io_data":fetch_value_investing_io_data(t) if configs["use_value_trades"] else {"error":"VI.io: Skipped."},
            "institutional_holdings":fetch_inst_filings(t) if configs["use_filings"] else [],
            "sec_all_filings_raw":fetch_all_sec_filings(t) if configs["use_filings"] else []
        }

        # --- Run Agents ---
        agents = [PriceAgent(), MomentumAgent(), VolatilityAgent(), FundamentalsAgent(), ValuationAgent(), AnalystRatingAgent()]
        if configs["use_sentiment"] and llm_client: agents.extend([SentimentAgent(llm_client), NewsSummaryAgent(llm_client)])
        if configs["use_filings"]: agents.extend([SECFilingAgent(), InstitutionalHoldingsAgent()])
        # Only add SECSummaryAgent if explicitly enabled in configs AND LLM is available
        if configs["use_filings"] and llm_client and configs.get("use_sec_summary", False):
             agents.append(SECSummaryAgent(llm_client))
        # Removed PoliticianFilingsAgent
        if configs["use_value_trades"]: agents.append(ValueInvestingIOAgent())
        
        agent_res_list = []
        for agent in agents:
            name = agent.__class__.__name__
            try:
                # Special handling for agents that require specific arguments
                if isinstance(agent,(PriceAgent,MomentumAgent)): res_a = agent.run(t, data_bundle["price_history"])
                elif isinstance(agent,VolatilityAgent): res_a = agent.run(t, data_bundle, data_bundle["price_history"])
                # SECSummaryAgent needs the full data_bundle to get raw filings and ticker_info
                elif name == "SECSummaryAgent":
                    res_a = agent.run(t, data_bundle)
                else: # Generic run for other agents
                    res_a = agent.run(t, data_bundle)
                agent_res_list.append(res_a)
            except Exception as e:
                # Generic error handling for agents
                err_k = name.lower().replace("agent","")+"_error"
                sig_k = name.lower().replace("agent","")+"_signal" # Many agents provide a signal
                
                # Default error structure, adapted for SECSummaryAgent specifically
                error_dict = {err_k:f"Agent {name} error: {str(e)[:150]}"}
                if name == "SECSummaryAgent":
                    error_dict["sec_summary"] = "Error generating summary."
                else:
                    error_dict[sig_k] = "error" # Assign 'error' signal if a signal is expected
                
                agent_res_list.append(error_dict)
                st.error(f"Error in {name} for {t}: {e}. Results for this agent might be incomplete.")

        # --- Run Portfolio Agent for Final Decision ---
        final_dec = PortfolioAgent().run(t, agent_res_list, agent_weights=default_live_backtest_weights)
        
        # --- Consolidate all results for display ---
        curr_res_dict = {
            "ticker":t,
            "current_price_display":current_price_for_ticker,
            "market_cap_display":ticker_info.get("marketCap"),
            "industry_display":ticker_info.get("industry"),
            "sector_display":ticker_info.get("sector"),
            "ticker_info":ticker_info, # Keep full info for detailed display
            "news_headlines_for_popover":[f"{n.get('publish_time_readable','N/A')} - {n.get('title','N/A')} ({n.get('publisher','N/A')} via {n.get('source_api','Unk')}) [Link]({n.get('link','#')})" + (f" - {n.get('content_snippet',n.get('description',''))[:150]}..." if n.get('content_snippet') or n.get('description') else "") for n in dedup_news[:10]],
            # Removed "politician_trades_for_popover"
            "news_status_display":news_status_bundle
        }
        
        # Merge all agent results into the current ticker's result dictionary
        for r_dict in agent_res_list:
            if isinstance(r_dict,dict): curr_res_dict.update(r_dict)
        curr_res_dict.update(final_dec) # Add final decision and composite score

        # --- Run Simulated Backtest (simplified) ---
        backtest_end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        backtest_start_date = (datetime.now() - pd.DateOffset(years=1, days=1)).strftime("%Y-%m-%d")
        
        sim_bt_metrics, sim_bt_log_df = run_backtest(t, backtest_start_date, backtest_end_date, 10000, llm_client, default_live_backtest_weights)
        curr_res_dict["simulated_backtest_results"] = {"metrics": sim_bt_metrics, "log_df": sim_bt_log_df.to_dict('records') if not sim_bt_log_df.empty else []}

        results[t] = curr_res_dict
    progress_bar.empty()
    return results

def run_backtest(ticker, start_date, end_date, initial_capital, llm_client_placeholder, backtest_agent_weights):
    """
    Performs a simplified historical backtest for a single ticker.
    Note: This backtest only uses Price, Momentum, and Volatility agents for speed.
    """
    st.write(f"Preparing backtest: {ticker} ({start_date} to {end_date})...")
    
    # Fetch enough historical data for the longest lookback period (253 days for momentum)
    s_dt_obj = datetime.strptime(start_date, "%Y-%m-%d")
    # Fetch 18 months prior to start_date to ensure enough data for 200/252 day SMAs/Mom.
    fetch_start_dt_obj = (s_dt_obj - pd.DateOffset(months=18)) 
    fetch_s_dt_str = fetch_start_dt_obj.strftime("%Y-%m-%d")

    full_hist = fetch_price_history(ticker, period="max", interval="1d") # Use max period to get all available data
    
    if full_hist.empty:
        return {"error": f"Backtest fail {ticker}: Price history empty."}, pd.DataFrame()
    
    # Filter history to the relevant period for backtesting (including lookback for indicators)
    hist = full_hist[(full_hist.index >= pd.to_datetime(fetch_s_dt_str)) & (full_hist.index <= pd.to_datetime(end_date))].copy()
    
    # Ensure there's enough data within the *actual backtest window* for meaningful simulation
    if hist.empty or len(hist[hist.index >= pd.to_datetime(start_date)]) < 2:
        return {"error": f"Backtest fail {ticker}: Not enough historical data in the specified range ({start_date} to {end_date})."}, pd.DataFrame()
    
    info_bt = fetch_ticker_info(ticker); data_static = {"ticker_info": info_bt} # Ticker info is static for backtest duration
    
    # Initialize only relevant agents for backtesting (Price, Momentum, Volatility)
    p_agent, m_agent, v_agent, port_agent = PriceAgent(), MomentumAgent(), VolatilityAgent(), PortfolioAgent()
    
    log, cash, shares, port_val = [], initial_capital, 0, initial_capital
    
    # Dates on which to run the simulation (from start_date onwards)
    run_dates = hist[hist.index >= pd.to_datetime(start_date)].index

    for curr_dt in run_dates:
        # Slice data up to current date for agent calculations
        data_sl = hist[hist.index <= curr_dt]
        
        # Get current price for portfolio value calculation
        curr_price_pt = data_sl.Close.iloc[-1] if not data_sl.empty else (port_val / shares if shares else 0)
        
        # Skip if not enough historical data for agents on this specific day (e.g., at the very start of backtest)
        if data_sl.empty or len(data_sl) < 253: # Must have enough data for MomentumAgent at least
            log.append({"date":curr_dt, "cash":cash, "shares_held":shares, "price":curr_price_pt, "portfolio_value":port_val, "signal":"hold (not enough data)", "composite_score":0.0})
            continue # Continue to next day
        
        curr_price = data_sl.Close.iloc[-1]
        
        # Run agents (only the ones relevant for backtesting and using supplied weights)
        pa_r = p_agent.run(ticker, data_sl)
        ma_r = m_agent.run(ticker, data_sl)
        va_r = v_agent.run(ticker, data_static, data_sl) # Volatility agent also uses price data slice
        
        # Get final decision from the Portfolio Agent using backtest-specific weights
        final_dec_obj = port_agent.run(ticker, [pa_r, ma_r, va_r], agent_weights=backtest_agent_weights)
        final_dec = final_dec_obj["final_decision"]

        # Execute trades
        if final_dec=="buy" and cash > curr_price and curr_price > 0:
            s_buy = cash / curr_price
            shares += s_buy
            cash = 0 # Invest all cash for simplicity in this model
        elif final_dec=="sell" and shares > 0:
            cash += shares * curr_price
            shares = 0 # Sell all shares for simplicity
        
        # Update portfolio value
        port_val = cash + shares * curr_price
        
        # Log daily state
        log.append({"date":curr_dt, "cash":cash, "shares_held":shares, "price":curr_price, "portfolio_value":port_val, "signal":final_dec, "composite_score":final_dec_obj["composite_score"]})
    
    log_df = pd.DataFrame(log)
    if not log_df.empty:
        log_df.set_index("date", inplace=True)
    
    # Check again if log_df is empty or too short after processing, before calculating metrics
    if log_df.empty or len(log_df) < 2:
        return {"error":f"Backtest simulation for {ticker} resulted in insufficient data for performance metrics. (Log length: {len(log_df)})."}, pd.DataFrame()
    
    # --- Calculate Performance Metrics ---
    total_ret = (log_df.portfolio_value.iloc[-1]/initial_capital - 1)*100
    
    # Calculate years duration correctly, handling potential single-day scenarios
    days = (log_df.index[-1]-log_df.index[0]).days
    years = days/365.25 if days > 0 else (1/365.25 if len(log_df)>1 else 0) # If only one day, use 1/365.25 to avoid zero division, else 0
    
    ann_ret = 0
    if years > 0 and initial_capital > 0:
        # Annualized return formula: ((Ending Value / Beginning Value)^(1/Years)) - 1
        ann_ret = ((log_df.portfolio_value.iloc[-1]/initial_capital)**(1/years)-1)*100
    elif years == 0 and initial_capital > 0: # If it's a single day or no time passed
        ann_ret = total_ret

    log_df["daily_return"] = log_df.portfolio_value.pct_change().fillna(0)
    ann_vol = log_df.daily_return.std()*np.sqrt(252)*100 # Annualized volatility
    
    sharpe = (ann_ret/ann_vol) if ann_vol!=0 else 0 # Sharpe ratio
    
    log_df["cum_max"] = log_df.portfolio_value.cummax()
    log_df["drawdown"] = (log_df.portfolio_value - log_df.cum_max)/log_df.cum_max.replace(0,np.nan)
    max_dd = log_df.drawdown.min()*100 if not log_df.drawdown.empty and pd.notna(log_df.drawdown.min()) else 0 # Max Drawdown
    
    trades = (log_df.signal != log_df.signal.shift()).fillna(False).sum()//2 # Approximate number of trades (buy/sell pairs)
    
    # Calculate Buy and Hold return for comparison
    # Get the exact data used for backtest's hist for B&H calculation
    bh_start_price = hist[hist.index >= pd.to_datetime(start_date)].iloc[0]["Close"]
    bh_end_price = hist[hist.index >= pd.to_datetime(start_date)].iloc[-1]["Close"]
    
    buy_hold_value = (bh_end_price / bh_start_price) * initial_capital if bh_start_price != 0 else initial_capital
    buy_hold_ret = (buy_hold_value / initial_capital - 1) * 100 if initial_capital != 0 else 0

    metrics = {
        "Initial Capital":f"${initial_capital:,.2f}",
        "Final Portfolio Value":f"${log_df.portfolio_value.iloc[-1]:,.2f}",
        "Total Return (%)":f"{total_ret:.2f}%",
        "Buy & Hold Return (%)":f"{buy_hold_ret:.2f}%",
        "Annualized Return (%)":f"{ann_ret:.2f}%",
        "Annualized Volatility (%)":f"{ann_vol:.2f}%",
        "Sharpe Ratio":f"{sharpe:.2f}",
        "Max Drawdown (%)":f"{max_dd:.2f}%",
        "Number of Trades (approx)":f"{trades}"
    }
    return metrics, log_df

# --- Detailed Analysis Display Function ---
def display_detailed_analysis(res_detail):
    """Displays comprehensive analysis results for a single ticker in a tabbed interface."""
    ticker = res_detail.get("ticker", "N/A"); ticker_info = res_detail.get("ticker_info", {})
    
    tab_titles = ["📈 Chart & Core", "📊 Fundamentals", "💰 Analyst & Fair Value", "📰 News & Filings", "⚙️ All Signals", "🧪 Simulated Backtest"]
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        st.subheader("Price Performance & Technical Signals")
        # Fetch 1 year of price history specifically for the chart
        price_hist_chart_data = fetch_price_history(ticker, period="1y")
        if not price_hist_chart_data.empty:
            st.line_chart(price_hist_chart_data["Close"], use_container_width=True, color="#0072F0")
        else: st.warning("Price chart data not available for the last year.")
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Technical Indicators"); price_signal = str(res_detail.get('price_signal', 'hold')).upper()
            st.metric(label=f"Price Signal (SMA/RSI)", value=price_signal)
            st.markdown(f"""
                <div style="font-size: 14px;">
                    <li><b>50-Day SMA:</b> ${res_detail.get('sma50', np.nan):,.2f}</li>
                    <li><b>200-Day SMA:</b> ${res_detail.get('sma200', np.nan):,.2f}</li>
                    <li><b>14-Day RSI:</b> {res_detail.get('rsi14', np.nan):.2f}</li>
                </div>
            """, unsafe_allow_html=True)
            if pd.isna(res_detail.get('sma200')):
                st.info("SMA200 may not be available if less than 200 days of price data.")
        with col2:
            st.subheader("Momentum & Volatility"); momentum_signal = str(res_detail.get('momentum_signal', 'hold')).upper()
            st.metric(label="Momentum Signal", value=momentum_signal)
            st.markdown(f"""
                <div style="font-size: 14px;">
                    <li><b>1-Month Momentum:</b> {res_detail.get('momentum_1m', np.nan) * 100:.2f}%</li>
                    <li><b>12-Month Momentum:</b> {res_detail.get('momentum_12m', np.nan) * 100:.2f}%</li>
                    <li><b>Beta:</b> {res_detail.get('beta', np.nan):.2f}</li>
                </div>
            """, unsafe_allow_html=True)
            if pd.isna(res_detail.get('momentum_12m')):
                st.info("12-Month Momentum may not be available if less than ~252 days of price data.")


    with tabs[1]:
        st.subheader(f"Fundamental Overview: {ticker_info.get('longName', ticker)}")
        st.caption(f"**Sector:** {ticker_info.get('sector', 'N/A')} | **Industry:** {ticker_info.get('industry', 'N/A')}")
        if ticker_info.get('longBusinessSummary'):
            with st.popover("Show Business Summary"):
                st.markdown(ticker_info.get('longBusinessSummary'))
        st.markdown("---")
        fund_col1, fund_col2, fund_col3, fund_col4 = st.columns(4)
        market_cap_val = ticker_info.get('marketCap')
        if isinstance(market_cap_val, (int, float)):
            if market_cap_val >= 1e12: cap_str = f"${market_cap_val / 1e12:,.2f}T"
            elif market_cap_val >= 1e9: cap_str = f"${market_cap_val / 1e9:,.2f}B"
            elif market_cap_val >= 1e6: cap_str = f"${market_cap_val / 1e6:,.2f}M"
            else: cap_str = f"${market_cap_val:,.0f}"
        else: cap_str = "N/A"
        fund_col1.metric("Market Cap", cap_str)
        fund_col2.metric("Trailing P/E", f"{ticker_info.get('trailingPE', np.nan):.2f}" if isinstance(ticker_info.get('trailingPE'),(int,float)) else "N/A")
        fund_col3.metric("Forward P/E", f"{ticker_info.get('forwardPE', np.nan):.2f}" if isinstance(ticker_info.get('forwardPE'),(int,float)) else "N/A")
        fund_col4.metric("Price/Book", f"{ticker_info.get('priceToBook', np.nan):.2f}" if isinstance(ticker_info.get('priceToBook'),(int,float)) else "N/A")
        
        st.markdown("---")
        st.subheader("Financial Health")
        fund_sig = str(res_detail.get('fund_signal', 'hold')).upper()
        f_col1, f_col2, f_col3 = st.columns(3)
        f_col1.metric("Fundamental Signal", fund_sig)
        f_col2.metric("Piotroski Score (0-3)", f"{res_detail.get('piotroski_score', 'N/A')}/3")
        fcy_val = res_detail.get('fcf_yield'); f_col3.metric("FCF Yield", f"{fcy_val * 100:.2f}%" if isinstance(fcy_val,(int,float)) else "N/A")
        
        roe_val = ticker_info.get('returnOnEquity'); de_val = ticker_info.get('debtToEquity'); etr_val = ticker_info.get('enterpriseToRevenue'); ete_val = ticker_info.get('enterpriseToEbitda')
        health_data = {
            "Return on Equity (ROE)": f"{roe_val * 100:.2f}%" if isinstance(roe_val,(int,float)) else "N/A",
            "Debt to Equity": f"{de_val:.2f}" if isinstance(de_val,(int,float)) else "N/A",
            "EV/Revenue": f"{etr_val:.2f}" if isinstance(etr_val,(int,float)) else "N/A",
            "EV/EBITDA": f"{ete_val:.2f}" if isinstance(ete_val,(int,float)) else "N/A"
        }
        st.table(pd.DataFrame(health_data.items(), columns=["Metric", "Value"]))

    with tabs[2]:
        val_col1, val_col2 = st.columns(2)
        with val_col1:
            st.subheader("Analyst Consensus")
            analyst_signal = str(res_detail.get('analyst_signal', 'hold')).upper()
            num_analysts = ticker_info.get('numberOfAnalystOpinions', 'N/A')
            st.metric(label=f"Analyst Signal (from {num_analysts} opinions)", value=analyst_signal)
            abp_val = res_detail.get('analyst_buy_pct_inferred',0.5)
            st.progress(abp_val, text=f"{abp_val*100:.0f}% Buy Rating")
            tm_val = ticker_info.get('targetMeanPrice'); tu_val = res_detail.get('target_upside')
            st.metric("Mean Target Price", f"${tm_val:,.2f}" if isinstance(tm_val,(int,float)) else "N/A", f"{tu_val*100:.2f}% Upside" if isinstance(tu_val,(int,float)) else None)
        with val_col2:
            st.subheader("Peter Lynch Fair Value (via VI.io)")
            vi_signal = str(res_detail.get('vi_signal', 'hold')).upper()
            vi_fv = res_detail.get('vi_fair_value_estimate'); up_val = res_detail.get('vi_upside_percent')
            vi_fv_label = f"${vi_fv:,.2f}" if isinstance(vi_fv, (int, float)) else "N/A"
            st.metric(label=f"VI.io Signal (Fair Value: {vi_fv_label})", value=vi_signal, delta=f"{up_val:.2f}% Upside" if isinstance(up_val,(int,float)) else None, delta_color="inverse")
            if res_detail.get('vi_valuation_text_display'): st.markdown(f"> *{res_detail.get('vi_valuation_text_display')}*")

    with tabs[3]:
        st.subheader("News Analysis & Filings")
        if res_detail.get('news_summary'):
            with st.container(border=True):
                st.markdown("**AI-Generated News Summary (from news articles)**")
                st.write(res_detail.get('news_summary'))
                headlines = res_detail.get('news_headlines_for_popover', [])
                if headlines:
                    with st.expander("View News Sources & Links"):
                        for line in headlines: st.markdown(f"- {line}")
                if res_detail.get('sentiment_error'): st.warning(f"Sentiment Analysis Note: {res_detail.get('sentiment_error')}")
        else:
            st.info("News summary not available (LLM disabled, no articles found, or error during summary generation).")
        
        st.markdown("---")
        if res_detail.get('sec_summary_llm'):
            with st.container(border=True):
                st.markdown("**AI-Generated SEC Filings Summary**")
                st.write(res_detail.get('sec_summary_llm'))
                if res_detail.get('sec_summary_error'): st.warning(f"SEC Summary Note: {res_detail.get('sec_summary_error')}")
        else:
            if res_detail.get('sec_summary_error'):
                 st.warning(f"Could not generate SEC Filings Summary: {res_detail.get('sec_summary_error')}")
            else:
                 st.info("No relevant SEC filings found or processed for summary (check 'SEC & Inst. Filings' toggle and LLM availability).")

        file_col1, file_col2 = st.columns(2)
        with file_col1:
            st.markdown("**Insider Trading (Form 4 Filings)**")
            sec_signal = str(res_detail.get('sec_filings_signal', 'hold')).upper()
            st.metric("Insider Trading Signal", sec_signal)
            st.markdown(f"""
                <div style="font-size: 14px;">
                    <li><b>Net Insider Shares (1Y):</b> {res_detail.get('sec_net_insider_shares_1y', 0):,}</li>
                    <li><b>Insider Buys (1Y Value):</b> ${res_detail.get('sec_insider_buy_value_1y', 0):,.2f}</li>
                    <li><b>Insider Sells (1Y Value):</b> ${res_detail.get('sec_insider_sell_value_1y', 0):,.2f}</li>
                </div>
            """, unsafe_allow_html=True)
            with st.expander("View Recent Insider Transactions (Form 4s)"):
                form4_txns = res_detail.get('sec_recent_form4_transactions', [])
                if form4_txns:
                    for tx in form4_txns:
                        tx_type_display = "Bought" if tx.get("transaction_code") == "P" else "Sold"
                        price_display = f" @ ${tx.get('price_per_share'):,.2f}" if tx.get('price_per_share') else ""
                        st.write(f"**{tx.get('transaction_date')}**: {tx.get('reporting_owner', 'N/A')} ({tx.get('owner_relationship', 'N/A')}) {tx_type_display} {tx.get('shares', 0):,.0f} shares{price_display} - [Link]({tx.get('link_to_filing', '#')})")
                else:
                    st.info("No recent Form 4 insider transactions found.")
            with st.expander("View Other Recent SEC Filings (Metadata only)"):
                other_filings_meta = [f for f in res_detail.get('sec_other_recent_filings', []) if f.get("form_type") not in ['10-K', '10-Q', '8-K']]
                if other_filings_meta:
                    for f in other_filings_meta: st.write(f"**{f.get('filing_date', 'N/A')}**: Form {f.get('form_type', 'N/A')} - [Link]({f.get('summary_link', '#')})")
                else: st.info("No other relevant SEC filings metadata found.")

        with file_col2:
            st.markdown("**Institutional Holdings**")
            inst_signal = str(res_detail.get('inst_holdings_signal', 'hold')).upper()
            st.metric("Institutional Ownership Signal", inst_signal)
            st.markdown(f"""
                <div style="font-size: 14px;">
                    <li><b>Total Holders:</b> {res_detail.get('inst_num_holders', 0):,}</li>
                    <li><b>Total Shares Held:</b> {res_detail.get('inst_total_shares_held', 0):,}</li>
                    <li><b>% of Outstanding:</b> {res_detail.get('inst_total_pct_out', 0.0) * 100:.2f}%</li>
                </div>
            """, unsafe_allow_html=True)
            
            holders = res_detail.get('inst_top_holders', [])
            if holders:
                holders_df = pd.DataFrame(holders)
                # Ensure 'Shares' column is numeric and filter out zero shares for charting
                holders_df['Shares'] = pd.to_numeric(holders_df.get('Shares'), errors='coerce').fillna(0)
                holders_df = holders_df[holders_df['Shares'] > 0] # Filter out zero share holdings

                if not holders_df.empty and 'Holder' in holders_df.columns and 'Shares' in holders_df.columns:
                    top_n_holders = holders_df.nlargest(10, 'Shares').copy() # Ensure copy to avoid SettingWithCopyWarning
                    # Add a 'display_value' column for tooltip clarity if 'Value' is available
                    if 'Value' in top_n_holders.columns:
                        top_n_holders['Display Value'] = top_n_holders['Value'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
                    
                    chart = alt.Chart(top_n_holders).mark_arc(outerRadius=120).encode(
                        theta=alt.Theta(field="Shares", type="quantitative"),
                        color=alt.Color(field="Holder", type="nominal", title="Top Holders"),
                        order=alt.Order("Shares", sort="descending"),
                        tooltip=["Holder", "Shares", alt.Tooltip("% Out", format=".2%"), alt.Tooltip("Display Value", title="Value Held")] # Added Display Value to tooltip
                    ).properties(
                        title="Top Institutional Holders (by Shares)"
                    )
                    try:
                        st.altair_chart(chart, use_container_width=True)
                        st.markdown("**Top 10 Institutional Holders:**")
                        for holder_data in top_n_holders.to_dict('records'):
                            st.write(f"- {holder_data.get('Holder', 'N/A')}: {holder_data.get('Shares', 0):,.0f} shares ({holder_data.get('% Out', 0):.2%}) as of {holder_data.get('Date Reported', 'N/A')}")
                    except Exception as chart_err:
                        st.error(f"Error rendering institutional holdings chart for {ticker}: {chart_err}")
                        st.json(top_n_holders.to_dict('records')) # Show raw data if chart fails
                else:
                    st.info(f"Not enough valid data points for institutional holdings chart for {ticker}.")
            else:
                st.info("No institutional holder data available.")

    with tabs[4]: # All Signals
        st.subheader("All Agent Signals at a Glance")
        signals_data = {
            "Price Signal (SMA/RSI)": str(res_detail.get("price_signal","N/A")).upper(),
            "Momentum Signal": str(res_detail.get("momentum_signal","N/A")).upper(),
            "Volatility Signal": str(res_detail.get("volatility_signal","N/A")).upper(),
            "Fundamental Signal": str(res_detail.get("fund_signal","N/A")).upper(),
            "Analyst Signal": str(res_detail.get("analyst_signal","N/A")).upper(),
            "ValueInvesting.io Signal": str(res_detail.get("vi_signal","N/A")).upper(),
            "News Sentiment Signal": str(res_detail.get("sentiment_signal","N/A")).upper(),
            "SEC Insider Trading Signal": str(res_detail.get("sec_filings_signal","N/A")).upper(),
            "SEC Filings Summary (LLM)": "Generated" if res_detail.get("sec_summary_llm") and not res_detail.get("sec_summary_error") else "N/A (Error)" if res_detail.get("sec_summary_error") else "Skipped/No Data",
            "Institutional Signal": str(res_detail.get("inst_holdings_signal","N/A")).upper(),
        }
        df_signals = pd.DataFrame(signals_data.items(), columns=["Agent", "Signal"])
        st.dataframe(df_signals.style.applymap(lambda x: f'color: {get_signal_color(x)}', subset=['Signal']), use_container_width=True, hide_index=True)
        st.markdown("---")
        final_decision = str(res_detail.get('final_decision', 'hold')).upper(); final_color = get_signal_color(final_decision)
        st.markdown(f"""<div style="border:2px solid {final_color}; border-radius:8px; padding:15px; text-align:center;"><p style="font-size:1.2em; margin-bottom:5px;">Final AI Decision</p><h2 style="color:{final_color}; margin-bottom:5px;">{final_decision}</h2><p style="font-size:1em;">Composite Score: <strong>{res_detail.get('composite_score', np.nan):.2f}</strong></p></div>""", unsafe_allow_html=True)
    
    with tabs[5]: # Simulated Backtest
        st.subheader(f"Simulated Backtest for {res_detail.get('ticker')}")
        sim_bt_data = res_detail.get("simulated_backtest_results", {})
        sim_bt_metrics = sim_bt_data.get("metrics")
        sim_bt_log_df_raw = sim_bt_data.get("log_df") # This is a list of dicts

        if sim_bt_metrics and not (sim_bt_metrics.get("message") or sim_bt_metrics.get("error")):
            st.markdown("This section shows a quick simulated backtest for the last year using the **Price, Momentum, and Volatility** agents with standard weights. This is a simplified simulation for quick insight, not a full backtest.")
            metrics_df_sim_bt = pd.DataFrame.from_dict(sim_bt_metrics, orient='index', columns=['Value'])
            st.table(metrics_df_sim_bt)
            
            if sim_bt_log_df_raw:
                try:
                    sim_bt_log_df = pd.DataFrame(sim_bt_log_df_raw)
                    if not sim_bt_log_df.empty and 'date' in sim_bt_log_df.columns:
                        sim_bt_log_df['date'] = pd.to_datetime(sim_bt_log_df['date'])
                        # Ensure 'date' is timezone-naive to match expected index type for plotting
                        sim_bt_log_df.set_index(sim_bt_log_df['date'].dt.tz_localize(None), inplace=True)
                        
                        st.subheader("Portfolio Value Over Time (Simulated)"); st.line_chart(sim_bt_log_df["portfolio_value"])
                        st.subheader("Drawdown Over Time (Simulated)"); st.area_chart(sim_bt_log_df["drawdown"].fillna(0))
                    else:
                        st.warning("Simulated backtest log data is empty or missing 'date' column for charting.")
                except Exception as e:
                    st.error(f"Error processing simulated backtest log data for charting: {e}")
            else:
                st.info("No log data available for simulated backtest.")
        elif sim_bt_metrics:
            st.error(f"Simulated Backtest Error: {sim_bt_metrics.get('error','Unknown error')}")
        else:
            st.info("Simulated backtest results not available for this ticker.")


# --- Streamlit UI Main App Flow ---
llm_client = None
try:
    ds_key = st.secrets.get("DEEPSEEK_API_KEY")
    oa_key = st.secrets.get("OPENAI_API_KEY")
    
    if not ds_key: ds_key = os.environ.get("DEEPSEEK_API_KEY")
    if not oa_key: oa_key = os.environ.get("OPENAI_API_KEY")

    if ds_key:
        llm_client = ModelClient(api_key=ds_key, provider="deepseek")
        st.sidebar.caption("✅ LLM: DeepSeek")
    elif oa_key:
        llm_client = ModelClient(api_key=oa_key, provider="openai")
        st.sidebar.caption("✅ LLM: OpenAI")
    else:
        st.sidebar.warning("LLM API key missing. Sentiment/Summary disabled.")
except ValueError as e:
    st.sidebar.error(f"LLM Init Error: {e}")
    llm_client = None
except Exception as e:
    st.sidebar.error(f"LLM Unexpected Init Error: {e}")
    llm_client = None

st.title("🚀 AI Hedge Fund Simulator")
st.header("⚙️ Configuration");
config_cont = st.container(border=True)

# Radio buttons for selecting app mode
app_mode_options = ["Live Analysis", "Backtesting", "💼 Portfolio Management", "🤖 Virtual Trading"]
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = app_mode_options[0]

with config_cont:
    current_mode_index = app_mode_options.index(st.session_state.app_mode)
    selected_mode = st.radio("Select Mode:", app_mode_options, key="app_mode_sel_main_key", horizontal=True, index=current_mode_index)
    if selected_mode != st.session_state.app_mode:
        st.session_state.app_mode = selected_mode
        # Reset flags when switching modes to ensure fresh run if button is clicked
        st.session_state.live_analysis_triggered = False
        st.session_state.backtest_triggered = False
        st.rerun() # Rerun to update the UI based on the new mode

    st.markdown("---")

    if st.session_state.app_mode == "Live Analysis":
        st.subheader("Live Analysis Settings")
        tickers_in_live = st.text_input("Tickers (comma-separated):", "AAPL,MSFT,GOOG,CRWD", help="Enter comma-separated stock tickers (e.g., AAPL,MSFT,GOOG,NVDA).", key="live_tickers_input")
        st.caption("ℹ️ Live analysis fetches the latest available data from various sources.")
        
        st.subheader("Feature Toggles")
        feat_cols = st.columns(3)
        with feat_cols[0]:
            use_sent_live = st.checkbox("News Sentiment & Summary (LLM)", value=bool(llm_client), disabled=not llm_client, key="live_sent_cb_main", help="Uses LLM to analyze news. Requires NewsAPI key and LLM API key.")
            use_filings_live = st.checkbox("SEC & Inst. Filings", value=True, key="live_sec_cb_main", help="Fetches insider trades (Form 4) and institutional holdings. Enables SEC Summary if LLM is active.")
        with feat_cols[1]:
            # Removed Politician Filings checkbox
            use_valtrades_live = st.checkbox("ValueInvesting.io (Exp.)", value=False, key="live_vt_cb_main", help="Scrapes Peter Lynch fair value from ValueInvesting.io. Experimental, may be slow or break.")
            # New checkbox for LLM SEC summary, dependent on LLM and filings
            use_sec_summary_live = st.checkbox("SEC Filings Summary (LLM)", value=bool(llm_client) and use_filings_live, disabled=not (llm_client and use_filings_live), key="live_sec_summary_cb", help="Uses LLM to summarize recent 10-K, 10-Q, 8-K filings. Requires LLM and 'SEC & Inst. Filings' to be enabled.")

        if st.button("🚀 Run Live Analysis", use_container_width=True, type="primary", key="run_live_analysis_button"):
            live_tickers = [t.strip().upper() for t in tickers_in_live.split(",") if t.strip()]
            if not live_tickers:
                st.error("Please enter at least one ticker to analyze.")
            else:
                live_configs = {
                    "use_sentiment":use_sent_live,
                    "use_filings":use_filings_live,
                    # Removed "use_politician_filings"
                    "use_value_trades":use_valtrades_live,
                    "use_sec_summary":use_sec_summary_live
                }
                with st.spinner("⏳ Processing live analysis... This might take a while for multiple tickers or if scraping external sites."):
                    st.session_state.live_output = run_live_analysis(live_tickers, llm_client, live_configs)
                    st.session_state.live_analysis_triggered = True
                    st.rerun() # Trigger a rerun to display the results

    # Display Live Analysis Results (after the script reruns due to the button click)
    if st.session_state.app_mode == "Live Analysis" and st.session_state.live_analysis_triggered:
        st.subheader("Live Analysis Results")
        
        summary_data = []
        if st.session_state.live_output:
            for ticker, res in st.session_state.live_output.items():
                if res.get("error"):
                    summary_data.append({"Ticker": ticker, "AI Decision": "ERROR", "Composite Score": "N/A", "Market Cap": "N/A", "Industry": "N/A", "News Status": "N/A", "Error Message": res["error"]})
                else:
                    market_cap_val = res.get('market_cap_display', 0)
                    if isinstance(market_cap_val, (int, float)):
                        if market_cap_val >= 1e12: cap_str = f"${market_cap_val / 1e12:,.2f}T"
                        elif market_cap_val >= 1e9: cap_str = f"${market_cap_val / 1e9:,.2f}B"
                        elif market_cap_val >= 1e6: cap_str = f"${market_cap_val / 1e6:,.2f}M"
                        else: cap_str = f"${market_cap_val:,.0f}"
                    else: cap_str = "N/A"

                    summary_data.append({
                        "Ticker": ticker,
                        "AI Decision": res.get('final_decision', 'N/A').upper(),
                        "Composite Score": f"{res.get('composite_score', np.nan):.2f}",
                        "Market Cap": cap_str,
                        "Industry": res.get('industry_display', 'N/A'),
                        "News Status": res.get('news_status_display', 'N/A')
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            # Apply color based on AI Decision using the global get_signal_color function
            st.dataframe(summary_df.style.applymap(lambda x: f'color: {get_signal_color(x)}', subset=['AI Decision']), use_container_width=True, hide_index=True)
            st.markdown("---")
            
            # Dropdown to select a ticker for detailed analysis
            selected_ticker_for_detail = st.selectbox("Select Ticker for Detailed Analysis:", [""] + list(st.session_state.live_output.keys()), help="Choose a ticker from the table above to see a detailed breakdown of its analysis.")
            if selected_ticker_for_detail:
                # Display detailed analysis for the selected ticker
                display_detailed_analysis(st.session_state.live_output[selected_ticker_for_detail])
        else:
            st.info("No analysis results to display. Please run a live analysis above.")

    elif st.session_state.app_mode == "Backtesting":
        st.subheader("Backtesting Settings")
        st.session_state.bt_ticker = st.text_input("Ticker:", "AAPL", help="Enter a single stock ticker for backtesting.", key="bt_ticker_in_bt").upper()
        
        # Option to use manual capital or from a saved portfolio (fixed to 10k for simplicity)
        bt_capital_source = st.radio("Capital Source:", ("Manual Input", "From Saved Portfolio"), horizontal=True, help="For simplicity, using a fixed $10,000 for backtesting regardless of saved portfolio value.", key="bt_capital_source_radio")
        bt_capital = 10000 # Default/Fixed capital for backtesting
        if bt_capital_source == "Manual Input":
            bt_capital = st.number_input("Initial Capital:", 1000, 1000000, 10000, 1000, format="%d", help="Set the initial capital for the backtest simulation.", key="bt_cap_in_bt")
        else:
            portfolio_names_bt = list(st.session_state.portfolios_data.keys())
            if not portfolio_names_bt: st.warning("No portfolios found to choose from.")
            else:
                sel_pf_bt = st.selectbox("Select Portfolio (value is fixed at $10,000):", portfolio_names_bt, help="Selecting a portfolio here uses its name, but the initial capital for backtesting remains fixed at $10,000.", key="bt_pf_select")
                # bt_capital remains 10000 as stated in info

        bt_c1, bt_c2 = st.columns(2)
        with bt_c1:
            def_end_dt = datetime.now(timezone.utc)-timedelta(days=1)
            def_start_dt = def_end_dt-pd.DateOffset(years=3)
            start_dt_in = st.date_input("Start Date:", def_start_dt, max_value=def_end_dt-timedelta(days=30), help="Select the start date for the backtest. Must be at least 30 days before end date.", key="bt_start_dt_bt")
            st.session_state.bt_start_str = start_dt_in.strftime("%Y-%m-%d")
        with bt_c2:
            min_end_dt_bt = start_dt_in+timedelta(days=30)
            end_dt_in = st.date_input("End Date:", def_end_dt, min_value=min_end_dt_bt, max_value=datetime.now(timezone.utc)-timedelta(days=1), help="Select the end date for the backtest. Must be at least 30 days after start date and before today.", key="bt_end_dt_bt")
            st.session_state.bt_end_str = end_dt_in.strftime("%Y-%m-%d")
            
        with st.expander("Adjust Backtest Agent Weights",expanded=False):
            st.info("These weights only apply to the simulated backtest, which uses only Price, Momentum, and Volatility agents for simplicity and speed.")
            w_p = st.slider("Price Agent Weight:",0.,2.,1.,.1,key="bt_w_p_bt")
            w_m = st.slider("Momentum Agent Weight:",0.,2.,.8,.1,key="bt_w_m_bt")
            w_v = st.slider("Volatility Agent Weight:",0.,2.,.2,.1,key="bt_w_v_bt")
            
            st.session_state.bt_weights = {
                "price":w_p, "momentum":w_m, "volatility":w_v,
                "sentiment":0.,"fund":0.,"valuation_dcf":0.,"valuation_pe":0.,
                "sec_filings":0.,"sec_summary":0., 
                "inst_holdings":0.,"analyst":0.,
                # Removed "politician_filings" key
                "vi_signal":0.
            }
            st.session_state.bt_capital = bt_capital # Store capital in session state for backtest

        if st.button("📈 Run Backtest",use_container_width=True,type="primary",key="run_bt_btn_main"):
            if st.session_state.bt_ticker:
                with st.spinner(f"⏳ Running backtest for {st.session_state.bt_ticker} from {st.session_state.bt_start_str} to {st.session_state.bt_end_str}..."):
                    metrics, log_df = run_backtest(st.session_state.bt_ticker, st.session_state.bt_start_str, st.session_state.bt_end_str, st.session_state.bt_capital, llm_client, st.session_state.bt_weights)
                    st.session_state.backtest_results[st.session_state.bt_ticker] = {"metrics": metrics, "log_df": log_df.to_dict('records') if not log_df.empty else []}
                    st.session_state.backtest_triggered = True
                    st.rerun()
            else:
                st.error("Please enter a ticker for backtesting.")
        
        # Display Backtest Results after rerun
        if st.session_state.app_mode == "Backtesting" and st.session_state.backtest_triggered:
            st.subheader(f"Backtest Results for {st.session_state.bt_ticker}")
            
            bt_res = st.session_state.backtest_results.get(st.session_state.bt_ticker, {})
            bt_metrics = bt_res.get("metrics")
            bt_log_df_raw = bt_res.get("log_df")

            if bt_metrics and not (bt_metrics.get("message") or bt_metrics.get("error")):
                st.markdown("This shows a simulated backtest.")
                metrics_df = pd.DataFrame.from_dict(bt_metrics, orient='index', columns=['Value'])
                st.table(metrics_df)
                
                if bt_log_df_raw:
                    try:
                        bt_log_df = pd.DataFrame(bt_log_df_raw)
                        if not bt_log_df.empty and 'date' in bt_log_df.columns:
                            bt_log_df['date'] = pd.to_datetime(bt_log_df['date'])
                            # Ensure 'date' column is timezone-naive to match plotting library expectations if it was originally local time or lost timezone.
                            # If yfinance df is tz-naive, this is safe.
                            bt_log_df.set_index(bt_log_df['date'].dt.tz_localize(None), inplace=True)
                            
                            st.subheader("Portfolio Value Over Time"); st.line_chart(bt_log_df["portfolio_value"])
                            st.subheader("Drawdown Over Time"); st.area_chart(bt_log_df["drawdown"].fillna(0))
                        else:
                            st.warning("Backtest log data is empty or missing 'date' column for charting. Cannot display charts.")
                    except Exception as e:
                        st.error(f"Error processing backtest log data for charting: {e}. Raw data might be corrupted.")
                        st.json(bt_log_df_raw) # Display raw data for debugging
                else:
                    st.info("No detailed log data available for backtest charts.")
            elif bt_metrics:
                st.error(f"Backtest Error: {bt_metrics.get('error','Unknown error')}. Please check ticker and date range.")
            else:
                st.info("No backtest results available. Run a backtest using the settings above.")

    elif st.session_state.app_mode == "💼 Portfolio Management":
        st.subheader("💼 Portfolio Management")
        portfolio_names_list = list(st.session_state.portfolios_data.keys())

        if not portfolio_names_list:
            # Auto-create a default portfolio if none exist
            st.session_state.portfolios_data["My First Portfolio"] = {"holdings": []} 
            st.session_state.selected_portfolio_name = "My First Portfolio"
            save_portfolios(st.session_state.portfolios_data)
            st.success("No portfolios found. Created a default portfolio 'My First Portfolio'.")
            st.rerun() # Rerun once a default portfolio is created to update the selectbox

        col_pf1, col_pf2, col_pf3 = st.columns([3, 1, 1])

        st.session_state.selected_portfolio_name = col_pf1.selectbox(
            "Select Portfolio:",
            portfolio_names_list,
            index=portfolio_names_list.index(st.session_state.selected_portfolio_name) if st.session_state.selected_portfolio_name in portfolio_names_list else 0,
            key="portfolio_selector"
        )
        current_portfolio = st.session_state.portfolios_data.get(st.session_state.selected_portfolio_name, {"holdings": []})

        new_portfolio_name = col_pf2.text_input("New Portfolio Name:", "", help="Enter a name to create a new, empty portfolio.", key="new_pf_name")
        if col_pf3.button("➕ Create Portfolio", key="create_pf_btn"):
            if new_portfolio_name and new_portfolio_name not in st.session_state.portfolios_data:
                st.session_state.portfolios_data[new_portfolio_name] = {"holdings": []}
                save_portfolios(st.session_state.portfolios_data)
                st.session_state.selected_portfolio_name = new_portfolio_name
                st.success(f"Portfolio '{new_portfolio_name}' created!")
                st.rerun()
            else:
                st.error("Portfolio name is empty or already exists. Please choose a different name.")

        st.markdown("---")
        st.subheader(f"Holdings & Analysis for '{st.session_state.selected_portfolio_name}'")

        holdings_display_data = []
        tickers_to_analyze = [h['ticker'] for h in current_portfolio['holdings']]
        
        if tickers_to_analyze:
            portfolio_analysis_configs = {
                "use_sentiment": True, "use_filings": True, # Politician filings removed
                "use_value_trades": True, "use_sec_summary": True 
            }
            # Only re-run analysis if the portfolio name changes or results for this portfolio are not yet cached in session_state
            if 'portfolio_analysis_results' not in st.session_state or \
               st.session_state.portfolio_analysis_results.get('portfolio_name') != st.session_state.selected_portfolio_name:
                with st.spinner(f"⏳ Running AI analysis for '{st.session_state.selected_portfolio_name}' holdings... This may take a while."):
                    st.session_state.portfolio_analysis_results = {
                        'portfolio_name': st.session_state.selected_portfolio_name,
                        'analysis': run_live_analysis(tickers_to_analyze, llm_client, portfolio_analysis_configs)
                    }
            
            analysis_results = st.session_state.portfolio_analysis_results['analysis']
            st.markdown("---")

            total_market_value = 0.0
            total_unrealized_pnl = 0.0
            total_invested_cost = 0.001 # Initialize to small non-zero to avoid division by zero

            for holding in current_portfolio['holdings']:
                ticker = holding['ticker']
                quantity = holding['quantity']
                avg_price = holding['avg_price']

                analysis_res = analysis_results.get(ticker, {})

                current_price = analysis_res.get('current_price_display')
                final_decision = analysis_res.get('final_decision', 'N/A').upper()
                composite_score = analysis_res.get('composite_score', np.nan) # Use np.nan for numeric score if not found

                if isinstance(current_price, (int, float)) and current_price > 0:
                    market_value = current_price * quantity
                    unrealized_pnl = (current_price - avg_price) * quantity
                    total_market_value += market_value
                    total_unrealized_pnl += unrealized_pnl
                    total_invested_cost += avg_price * quantity # Accumulate actual cost basis
                    
                    holdings_display_data.append({
                        "Ticker": ticker,
                        "Quantity": quantity,
                        "Avg. Cost": avg_price,
                        "Current Price": current_price,
                        "Market Value": market_value,
                        "Unrealized P&L": unrealized_pnl,
                        "P&L (%)": (unrealized_pnl / (avg_price * quantity) * 100) if (avg_price * quantity) != 0 else 0.0,
                        "AI Decision": final_decision,
                        "Composite Score": composite_score
                    })
                else:
                    holdings_display_data.append({
                        "Ticker": ticker,
                        "Quantity": quantity,
                        "Avg. Cost": avg_price,
                        "Current Price": "N/A",
                        "Market Value": "N/A",
                        "Unrealized P&L": "N/A",
                        "P&L (%)": "N/A",
                        "AI Decision": final_decision,
                        "Composite Score": composite_score
                    })
            
            overall_total_value = total_market_value
            overall_pnl_percent = (total_unrealized_pnl / total_invested_cost * 100) if total_invested_cost != 0 else 0.0
            overall_pnl_color = "normal" if total_unrealized_pnl >= 0 else "inverse"

            st.columns(2)[0].metric("Total Portfolio Value (Holdings)", f"${overall_total_value:,.2f}")
            st.columns(2)[1].metric("Total Unrealized P&L", f"${total_unrealized_pnl:,.2f}", f"{overall_pnl_percent:.2f}%", delta_color=overall_pnl_color)

            if holdings_display_data:
                holdings_df = pd.DataFrame(holdings_display_data)
                # Ensure formatting for composite score is consistent
                holdings_df['Composite Score'] = holdings_df['Composite Score'].apply(lambda x: f"{x:.2f}" if isinstance(x, (float, int)) and not np.isnan(x) else x)

                st.dataframe(holdings_df.style.applymap(lambda x: f'color: {get_signal_color(x)}', subset=['AI Decision']), use_container_width=True, hide_index=True,
                                 column_config={
                                     "Quantity": st.column_config.NumberColumn(format="%.4f"),
                                     "Avg. Cost": st.column_config.NumberColumn(format="$%.2f"),
                                     "Current Price": st.column_config.NumberColumn(format="$%.2f"),
                                     "Market Value": st.column_config.NumberColumn(format="$%.2f"),
                                     "Unrealized P&L": st.column_config.NumberColumn(format="$%.2f", help="Unrealized Profit & Loss"),
                                     "P&L (%)": st.column_config.ProgressColumn(format="%.2f%%", min_value=-100, max_value=100),
                                     "Composite Score": st.column_config.TextColumn(help="Aggregated AI score (-1.0 to 1.0)") # Use TextColumn since we formatted it as string
                                 })
                st.markdown("---")
                selected_ticker_for_detail = st.selectbox("Select Holding for Detailed Analysis:", [""] + [h['Ticker'] for h in holdings_display_data], help="Choose a stock from your portfolio holdings to view its detailed AI analysis.")
                if selected_ticker_for_detail:
                    detail_res = analysis_results.get(selected_ticker_for_detail)
                    if detail_res:
                        display_detailed_analysis(detail_res)
                    else:
                        st.warning(f"Analysis results not found for {selected_ticker_for_detail}. Please re-run portfolio analysis if recent changes were made or data fetch failed.")
            else:
                st.info("This portfolio currently has no stock holdings. Use the 'Add Stock' section below.")

        else:
            st.info("This portfolio currently has no stock holdings. Add stocks to analyze them.")
        
        st.markdown("---")
        st.subheader("Add/Remove Stocks")
        col_add1, col_add2, col_add3 = st.columns(3)
        add_ticker = col_add1.text_input("Ticker to Add:", "", help="Enter the ticker symbol (e.g., MSFT).", key="add_ticker_input_pf").upper()
        add_quantity = col_add2.number_input("Quantity:", min_value=0.01, value=1.0, step=0.1, help="Number of shares to add (can be fractional).", key="add_quantity_input_pf")
        add_price = col_add3.number_input("Purchase Price (required):", min_value=0.01, value=0.01, step=0.01, help="The price at which you 'purchased' these shares.", key="add_price_input_pf")

        if st.button("➕ Add Stock to Portfolio", key="add_stock_btn_pf"):
            if add_ticker and add_quantity > 0 and add_price > 0:
                existing_holding_index = -1
                for i, h in enumerate(current_portfolio['holdings']):
                    if h['ticker'] == add_ticker:
                        existing_holding_index = i
                        break
                
                if existing_holding_index != -1:
                    existing_holding = current_portfolio['holdings'][existing_holding_index]
                    new_total_quantity = existing_holding['quantity'] + add_quantity
                    new_avg_price = ((existing_holding['avg_price'] * existing_holding['quantity']) + (add_price * add_quantity)) / new_total_quantity
                    existing_holding['quantity'] = new_total_quantity
                    existing_holding['avg_price'] = new_avg_price
                    st.success(f"Updated {add_ticker} in '{st.session_state.selected_portfolio_name}'. New quantity: {new_total_quantity:.2f}, Avg. Price: ${new_avg_price:.2f}.")
                else:
                    current_portfolio['holdings'].append({"ticker": add_ticker, "quantity": add_quantity, "avg_price": add_price})
                    st.success(f"Added {add_quantity:.2f} shares of {add_ticker} at ${add_price:.2f} to '{st.session_state.selected_portfolio_name}'.")
                
                save_portfolios(st.session_state.portfolios_data)
                st.session_state.portfolio_analysis_results = None # Invalidate cached analysis results for this portfolio to trigger re-analysis
                st.rerun()
            else:
                st.error("Please enter a valid ticker, quantity, and purchase price for the stock you wish to add.")

        col_rem1, col_rem2 = st.columns([1,2])
        tickers_in_current_portfolio = [h['ticker'] for h in current_portfolio['holdings']]
        remove_ticker_selection = col_rem1.selectbox("Select Ticker to Remove:", [""] + tickers_in_current_portfolio, help="Select a stock to completely remove all its shares from this portfolio.", key="remove_ticker_select_pf")

        if col_rem2.button("➖ Remove Stock from Portfolio", key="remove_stock_btn_pf"):
            if remove_ticker_selection:
                initial_holdings_count = len(current_portfolio['holdings'])
                current_portfolio['holdings'] = [h for h in current_portfolio['holdings'] if h['ticker'] != remove_ticker_selection]
                
                if len(current_portfolio['holdings']) < initial_holdings_count:
                    save_portfolios(st.session_state.portfolios_data)
                    st.success(f"Removed {remove_ticker_selection} from '{st.session_state.selected_portfolio_name}'.")
                    st.session_state.portfolio_analysis_results = None # Invalidate cached analysis results
                    st.rerun()
                else:
                    st.error(f"{remove_ticker_selection} not found in '{st.session_state.selected_portfolio_name}' holdings.")
            else:
                st.error("Please select a ticker to remove from the dropdown.")
        
        st.markdown("---")
        if st.button("🗑️ Delete Current Portfolio", key="delete_portfolio_btn_pf", type="secondary"):
            if st.session_state.selected_portfolio_name and st.session_state.selected_portfolio_name in st.session_state.portfolios_data:
                if st.session_state.selected_portfolio_name == "My First Portfolio" and len(st.session_state.portfolios_data) == 1:
                    st.error("Cannot delete the last portfolio. Please create a new one first if you wish to replace it.")
                else:
                    del st.session_state.portfolios_data[st.session_state.selected_portfolio_name]
                    save_portfolios(st.session_state.portfolios_data)
                    # Reset selected portfolio to the first available if current one is deleted
                    if st.session_state.portfolios_data:
                        st.session_state.selected_portfolio_name = list(st.session_state.portfolios_data.keys())[0]
                    else:
                        st.session_state.selected_portfolio_name = None # No portfolios left
                    st.session_state.portfolio_analysis_results = None # Clear analysis for deleted portfolio
                    st.success(f"Portfolio '{st.session_state.selected_portfolio_name}' deleted.")
                    st.rerun()
            else:
                st.error("No portfolio selected or found to delete.")


    elif st.session_state.app_mode == "🤖 Virtual Trading":
        st.header("📈 Virtual Portfolio Dashboard")
        # Container for dashboard overview
        with st.container(border=True):
            holdings_df_data = []
            total_holdings_value, total_pnl, initial_investment = 0.0, 0.0, 0.001 # Initial investment to prevent div by zero
            
            # Reconstruct portfolio value history from transactions for plotting
            chronological_transactions = list(st.session_state.virtual_portfolio['transaction_history']) # No reverse needed if processing chronologically
            
            # Fetch all necessary historical prices for all tickers involved in transactions
            all_tickers_in_history = list(set([t['ticker'] for t in chronological_transactions] + [h['ticker'] for h in st.session_state.virtual_portfolio['holdings']]))
            price_data_for_history = {}
            with st.spinner("Pre-fetching historical prices for portfolio value chart (this may take a moment for many transactions)..."):
                for t in all_tickers_in_history:
                    price_data_for_history[t] = fetch_price_history(t, period="max") # Fetch max period from yfinance

            daily_portfolio_values_df = pd.DataFrame() # Initialize empty DataFrame for history chart

            if chronological_transactions:
                # Determine the date range for the portfolio value history
                # Ensure all dates are timezone-aware if comparing to timezone.utc dates
                now_utc = datetime.now(timezone.utc)
                earliest_tx_date_str = chronological_transactions[0]['date'].split(" ")[0] # Assuming 'YYYY-MM-DD HH:MM:SS' format
                earliest_tx_date = datetime.strptime(earliest_tx_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                
                # Start recording portfolio value a bit before the first transaction
                earliest_data_point = earliest_tx_date - timedelta(days=365*2) # Look back 2 years before first transaction for context
                if earliest_data_point < datetime(1990, 1, 1, tzinfo=timezone.utc): # Prevent extremely old dates
                    earliest_data_point = datetime(1990, 1, 1, tzinfo=timezone.utc)

                full_date_range = pd.date_range(start=earliest_data_point, end=now_utc, freq='D', tz=timezone.utc)
                
                daily_portfolio_values = pd.Series(index=full_date_range, dtype=float)
                # Initialize with initial cash at the start of the range
                daily_portfolio_values[full_date_range[0]] = get_default_virtual_portfolio()["cash"]

                current_holdings_for_chart = {} # Keep track of holdings over time for history calculation
                current_cash_for_chart = get_default_virtual_portfolio()["cash"]
                last_date_processed = full_date_range[0]

                for tx in chronological_transactions:
                    tx_date_naive = datetime.strptime(tx['date'].split(" ")[0], "%Y-%m-%d")
                    tx_date = tx_date_naive.replace(tzinfo=timezone.utc) # Make transaction date timezone-aware
                    
                    tx_quantity = float(tx['quantity'])
                    tx_price = float(tx['price'].replace('$', ''))
                    tx_type = tx['type']

                    # Fill in portfolio value for days between last processed date and current transaction date
                    for day in pd.date_range(start=last_date_processed + timedelta(days=1), end=tx_date, freq='D', tz=timezone.utc):
                        portfolio_val_on_day = current_cash_for_chart
                        for ticker_chart, qty_chart in current_holdings_for_chart.items():
                            day_price_series = price_data_for_history.get(ticker_chart, pd.DataFrame())
                            day_price = None
                            if not day_price_series.empty:
                                try:
                                    # Ensure comparison with timezone-naive index for yfinance data
                                    day_price = day_price_series.loc[day.date()].get("Close")
                                except KeyError:
                                    # Fallback to nearest day if exact date not found
                                    nearest_idx = day_price_series.index.asof(day.date())
                                    if nearest_idx is not pd.NaT:
                                        day_price = day_price_series.loc[nearest_idx].get("Close")

                            if day_price is not None and pd.notna(day_price):
                                portfolio_val_on_day += qty_chart * day_price
                            # else: if price not found, that holding is not valued for that day.

                        if pd.notna(portfolio_val_on_day):
                            daily_portfolio_values.loc[day] = portfolio_val_on_day
                    
                    last_date_processed = tx_date

                    # Apply the transaction
                    if tx_type == 'BUY':
                        current_holdings_for_chart[tx['ticker']] = current_holdings_for_chart.get(tx['ticker'], 0) + tx_quantity
                        current_cash_for_chart -= tx_quantity * tx_price
                    elif tx_type == 'SELL':
                        current_holdings_for_chart[tx['ticker']] = current_holdings_for_chart.get(tx['ticker'], 0) - tx_quantity
                        if current_holdings_for_chart[tx['ticker']] <= 0.0001: # Remove if quantity is near zero after sale
                            del current_holdings_for_chart[tx['ticker']]
                        current_cash_for_chart += tx_quantity * tx_price
                    
                    # Update portfolio value on the transaction day itself (after applying transaction)
                    portfolio_val_on_tx_day = current_cash_for_chart
                    for ticker_chart, qty_chart in current_holdings_for_chart.items():
                        tx_day_price_series = price_data_for_history.get(ticker_chart, pd.DataFrame())
                        tx_day_price = None
                        if not tx_day_price_series.empty:
                            try:
                                tx_day_price = tx_day_price_series.loc[tx_date.date()].get("Close")
                            except KeyError:
                                nearest_idx = tx_day_price_series.index.asof(tx_date.date())
                                if nearest_idx is not pd.NaT:
                                    tx_day_price = tx_day_price_series.loc[nearest_idx].get("Close")
                        if tx_day_price is not None and pd.notna(tx_day_price):
                            portfolio_val_on_tx_day += qty_chart * tx_day_price
                    if pd.notna(portfolio_val_on_tx_day):
                        daily_portfolio_values.loc[tx_date] = portfolio_val_on_tx_day
                
                # Fill any remaining days up to today with the last known value
                daily_portfolio_values = daily_portfolio_values.fillna(method='ffill').fillna(method='bfill')

                # Ensure the final DataFrame index is also timezone-aware for consistency
                daily_portfolio_values_df = pd.DataFrame(daily_portfolio_values, columns=['Portfolio Value'])
                daily_portfolio_values_df.index.name = 'Date'

            else: # If no transactions, show a flat portfolio value for a short period
                daily_portfolio_values_df = pd.DataFrame({
                    'Date': [datetime.now(timezone.utc) - timedelta(days=7), datetime.now(timezone.utc)],
                    'Portfolio Value': [get_default_virtual_portfolio()["cash"], get_default_virtual_portfolio()["cash"]]
                }).set_index('Date')
                daily_portfolio_values_df.index = daily_portfolio_values_df.index.tz_localize(timezone.utc)


            # Display current holdings and overall metrics
            if st.session_state.virtual_portfolio['holdings']:
                with st.spinner("Fetching latest prices for current holdings display..."):
                    for holding in st.session_state.virtual_portfolio['holdings']:
                        info = fetch_ticker_info(holding['ticker'])
                        price = info.get("currentPrice")
                        # Fallback to last known historical price if live price isn't available
                        if price is None and not price_data_for_history.get(holding['ticker'], pd.DataFrame()).empty:
                            price = price_data_for_history[holding['ticker']].iloc[-1].get("Close")
                        
                        if price is None:
                            st.warning(f"Could not get live price for {holding['ticker']} in virtual portfolio. Displaying based on average cost.")
                            current_value = holding['avg_price'] * holding['quantity'] # Use avg cost as fallback for display
                            pnl = 0 # Cannot calculate P&L without live price
                        else:
                            current_value = price * holding['quantity']
                            pnl = (price - holding['avg_price']) * holding['quantity']
                        
                        total_holdings_value += current_value
                        total_pnl += pnl
                        initial_investment += holding['avg_price'] * holding['quantity']
                        holdings_df_data.append({"Ticker": holding['ticker'], "Quantity": holding['quantity'], "Avg. Price": holding['avg_price'], "Current Price": price, "Current Value": current_value, "P&L": pnl})
            
            total_portfolio_value = st.session_state.virtual_portfolio['cash'] + total_holdings_value
            pnl_percent = (total_pnl / initial_investment * 100) if initial_investment != 0 else 0.0

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
                st.dataframe(holdings_df, use_container_width=True, column_config={
                    "Avg. Price": st.column_config.NumberColumn(format="$%.2f"),
                    "Current Price": st.column_config.NumberColumn(format="$%.2f"),
                    "Current Value": st.column_config.NumberColumn(format="$%.2f"),
                    "P&L": st.column_config.NumberColumn(format="$%.2f"),
                    "Quantity": st.column_config.NumberColumn(format="%.4f")
                })
            else:
                st.info("The portfolio currently holds no stocks. Run the AI Trader to start investing.")

            st.subheader("Portfolio Value Over Time")
            if not daily_portfolio_values_df.empty:
                st.line_chart(daily_portfolio_values_df["Portfolio Value"], use_container_width=True, color="#0072F0")
            else:
                st.info("No sufficient data to plot portfolio value history.")

            st.subheader("Transaction History")
            if st.session_state.virtual_portfolio['transaction_history']:
                history_df = pd.DataFrame(st.session_state.virtual_portfolio['transaction_history'])
                st.dataframe(history_df, use_container_width=True, hide_index=True)
            else:
                st.info("No transactions have been made yet.")

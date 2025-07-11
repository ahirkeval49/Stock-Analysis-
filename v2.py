import os
import streamlit as st
import yfinance as yf # Keep yfinance for all data fetching for now
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
import altair as alt # Import altair for charting
import time # Import time for sleeps
import random # Import random for random delays

# --- REMOVED ALPHA VANTAGE IMPORTS (no longer needed) ---
# from alpha_vantage.timeseries import TimeSeries
# from alpha_vantage.fundamentaldata import FundamentalData

# --- Page Config (Must be the first Streamlit command) ---
st.set_page_config(page_title="AI Hedge Fund Simulator", layout="wide")

st.write("DEBUG: App started and page config set.") # DEBUG

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
            print("DEBUG: JSONDecodeError loading portfolios. Returning empty dict.") # DEBUG
            return {}
    print("DEBUG: Portfolios file not found. Returning empty dict.") # DEBUG
    return {}

def save_portfolios(portfolios_data):
    if not isinstance(portfolios_data, dict):
        st.error("Error saving portfolios: Data is not in the correct format.")
        print("DEBUG: Error saving portfolios: Data is not in the correct format.") # DEBUG
        return
    with open(PORTFOLIOS_FILE, 'w') as f:
        json.dump(portfolios_data, f, indent=4)
    print(f"DEBUG: Portfolios data saved to {PORTFOLIOS_FILE}") # DEBUG

def load_virtual_portfolio():
    if os.path.exists(VIRTUAL_PORTFOLIO_FILE):
        try:
            with open(VIRTUAL_PORTFOLIO_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("DEBUG: JSONDecodeError loading virtual portfolio. Returning default.") # DEBUG
            return get_default_virtual_portfolio()
    print("DEBUG: Virtual portfolio file not found. Returning default.") # DEBUG
    return get_default_virtual_portfolio()

def save_virtual_portfolio(data):
    with open(VIRTUAL_PORTFOLIO_FILE, 'w') as f:
        json.dump(data, f, indent=4, default=str) # Use default=str to handle datetimes if they exist
    print(f"DEBUG: Virtual portfolio saved to {VIRTUAL_PORTFOLIO_FILE}") # DEBUG

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
    print(f"DEBUG: Initialized portfolios_data from file: {st.session_state.portfolios_data.keys()}") # DEBUG

if 'selected_portfolio_name' not in st.session_state:
    st.session_state.selected_portfolio_name = None
    if st.session_state.portfolios_data:
        st.session_state.selected_portfolio_name = list(st.session_state.portfolios_data.keys())[0]
    print(f"DEBUG: Initial selected_portfolio_name: {st.session_state.selected_portfolio_name}") # DEBUG

if 'portfolio_stock_analysis' not in st.session_state:
    st.session_state.portfolio_stock_analysis = {}
    print("DEBUG: Initialized portfolio_stock_analysis.") # DEBUG

if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = {}
    print("DEBUG: Initialized backtest_results.") # DEBUG

if 'live_output' not in st.session_state:
    st.session_state.live_output = {}
    print("DEBUG: Initialized live_output.") # DEBUG

if 'virtual_portfolio' not in st.session_state:
    st.session_state.virtual_portfolio = load_virtual_portfolio()
    print(f"DEBUG: Initialized virtual_portfolio. Cash: {st.session_state.virtual_portfolio['cash']}") # DEBUG

# Initialize flags for running analysis
if 'live_analysis_triggered' not in st.session_state:
    st.session_state.live_analysis_triggered = False
    print("DEBUG: Initialized live_analysis_triggered to False.") # DEBUG
if 'backtest_triggered' not in st.session_state:
    st.session_state.backtest_triggered = False
    print("DEBUG: Initialized backtest_triggered to False.") # DEBUG

# --------------------------------
# Data Fetchers (All reverted to yfinance)
# --------------------------------

@st.cache_data(ttl=300) # Cache for 5 minutes
def fetch_price_history(ticker: str, period: str = "max", interval: str = "1d") -> pd.DataFrame:
    print(f"DEBUG: Calling fetch_price_history (yfinance) for {ticker} with period={period}, interval={interval}") # DEBUG
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, interval=interval)
        if df.empty:
            st.warning(f"No price history returned from yfinance for {ticker} (period={period}).")
            print(f"DEBUG: No price history returned from yfinance for {ticker}. DataFrame is empty.") # DEBUG
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        print(f"DEBUG: Successfully fetched {len(df)} rows of price history for {ticker}.") # DEBUG
        return df
    except Exception as e:
        st.error(f"Critical error fetching price history for {ticker}: {e}")
        print(f"DEBUG: CRITICAL ERROR in fetch_price_history for {ticker}: {e}") # DEBUG
        return pd.DataFrame()


@st.cache_data(ttl=300) # Cache for 5 minutes
def fetch_ticker_info(ticker: str) -> dict:
    """
    Fetches ticker information from yfinance with enhanced error handling and retries.
    Returns {..., "_error": "error message"} on final failure.
    """
    print(f"DEBUG: Calling fetch_ticker_info (yfinance) for {ticker}") # DEBUG
    max_retries = 3
    base_delay = 1.0 # seconds
    
    for attempt in range(max_retries):
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info

            if not info or not info.get('financialCurrency'):
                missing_keys = []
                if not info:
                    missing_keys.append("empty_info_dict")
                if info and info.get('financialCurrency') is None:
                    missing_keys.append("financialCurrency")
                
                error_detail = f"Missing essential info: {', '.join(missing_keys)}. Raw keys: {list(info.keys()) if info else 'None'}."
                st.warning(f"Attempt {attempt + 1}/{max_retries}: Failed to get complete info for {ticker}: {error_detail}")
                print(f"DEBUG: Attempt {attempt + 1}/{max_retries}: Incomplete info for {ticker}: {error_detail}") # DEBUG
                
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt) + random.uniform(0.1, 0.5))
                    continue
                else:
                    final_error_msg = f"Failed to get complete info for {ticker} after {max_retries} attempts. Reason: {error_detail}. This may indicate an invalid ticker, a delisted stock, or persistent upstream data issues."
                    st.error(final_error_msg)
                    print(f"DEBUG: FINAL ERROR in fetch_ticker_info for {ticker}: {final_error_msg}") # DEBUG
                    return {"_error": final_error_msg}
            
            print(f"DEBUG: Successfully fetched ticker info for {ticker}. Keys: {list(info.keys())[:5]}...") # DEBUG
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
                "financialCurrency": info.get("financialCurrency"),
                "_error": None
            }
        except Exception as e:
            st.error(f"Attempt {attempt + 1}/{max_retries}: Exception fetching ticker info for {ticker}: {e}")
            print(f"DEBUG: Attempt {attempt + 1}/{max_retries}: EXCEPTION in fetch_ticker_info for {ticker}: {e}") # DEBUG
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt) + random.uniform(0.1, 0.5))
                continue
            else:
                final_error_msg = f"Final exception fetching info for {ticker} after {max_retries} attempts: {e}. This may indicate an invalid ticker, a delisted stock, or persistent upstream data issues."
                st.error(final_error_msg)
                print(f"DEBUG: FINAL EXCEPTION in fetch_ticker_info for {ticker}: {final_error_msg}") # DEBUG
                return {"_error": final_error_msg}

@st.cache_data(ttl=300) # Cache for 5 minutes
def fetch_enriched_news(ticker: str, ticker_info_data: dict) -> list[dict]:
    print(f"DEBUG: Calling fetch_enriched_news (yfinance) for {ticker}") # DEBUG
    try:
        company_name = ticker_info_data.get('longName', ticker_info_data.get('shortName', ticker))
        ticker_obj = yf.Ticker(ticker); raw_news = []
        try:
            raw_news = ticker_obj.news
            print(f"DEBUG: Fetched {len(raw_news)} raw news items for {ticker}.") # DEBUG
        except TypeError as te:
            print(f"DEBUG: Yfinance .news type error for {ticker}: {te}") # DEBUG
            return [{"error": f"yfinance .news type error for {ticker}: {te}", "source_api": "Yahoo Finance"}]
        except Exception as news_exc:
            print(f"DEBUG: Yfinance .news call failed for {ticker}: {news_exc}") # DEBUG
            return [{"error": f"yfinance .news call failed for {ticker}: {news_exc}", "source_api": "Yahoo Finance"}]
        
        enriched_news_list = []
        if not raw_news:
            print(f"DEBUG: No raw news to process for {ticker}.") # DEBUG
            return []
        
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
        print(f"DEBUG: Processed {len(enriched_news_list)} enriched news items for {ticker}.") # DEBUG
        return enriched_news_list
    except Exception as e:
        print(f"DEBUG: Overall error in fetch_enriched_news for {ticker}: {e}") # DEBUG
        return [{"error": f"Processing Yahoo Finance news for {ticker} failed: {e}", "source_api": "Yahoo Finance"}]

@st.cache_data(ttl=1800)
def fetch_comprehensive_news_from_api(ticker: str, company_name: str, lookback_days: int = 30) -> list[dict]:
    print(f"DEBUG: Calling fetch_comprehensive_news_from_api (NewsAPI.org) for {ticker}") # DEBUG
    api_key = st.secrets.get("NEWSAPI_KEY")
    if not api_key:
        print("DEBUG: NEWSAPI_KEY not found.") # DEBUG
        return [{"error": "NEWSAPI_KEY not found.", "source_api": "NewsAPI.org"}]
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
            print(f"DEBUG: Found {len(articles_list)} NewsAPI.org articles for {ticker}.") # DEBUG
        elif all_articles_response.get("status") == "error":
            print(f"DEBUG: NewsAPI Error for {ticker}: {all_articles_response.get('code')} - {all_articles_response.get('message')}") # DEBUG
            return [{"error": f"NewsAPI Error ({ticker}): {all_articles_response.get('code')} - {all_articles_response.get('message')}", "source_api": "NewsAPI.org"}]
        else:
            print(f"DEBUG: NewsAPI ({ticker}): No articles or unexpected structure.") # DEBUG
            return [{"error": f"NewsAPI ({ticker}): No articles or unexpected structure.", "source_api": "NewsAPI.org"}]
    except requests.exceptions.RequestException as e:
        print(f"DEBUG: NewsAPI request failed for {ticker}: {e}") # DEBUG
        return [{"error": f"NewsAPI request failed for {ticker}: {e}", "source_api": "NewsAPI.org"}]
    except Exception as e:
        print(f"DEBUG: Unexpected error with NewsAPI for {ticker}: {e}") # DEBUG
        return [{"error": f"Unexpected error with NewsAPI for {ticker}: {e}", "source_api": "NewsAPI.org"}]
    return articles_list

@st.cache_data(ttl=24*3600)
def get_all_cik_ticker_mappings():
    print("DEBUG: Calling get_all_cik_ticker_mappings.") # DEBUG
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(url, headers={'User-Agent': SEC_USER_AGENT}); response.raise_for_status()
        mappings = {item['ticker']: str(item['cik_str']).zfill(10) for item in response.json() if 'ticker' in item and 'cik_str' in item}
        print(f"DEBUG: Fetched {len(mappings)} CIK mappings.") # DEBUG
        return mappings
    except Exception as e:
        st.error(f"CRITICAL: Failed CIK mappings: {e}.");
        print(f"DEBUG: CRITICAL ERROR in get_all_cik_ticker_mappings: {e}") # DEBUG
        return {}
TICKER_TO_CIK_MAP = get_all_cik_ticker_mappings()

def get_cik_for_ticker(ticker: str) -> str | None: return TICKER_TO_CIK_MAP.get(ticker.upper())

@st.cache_data(ttl=4*3600)
def fetch_all_sec_filings(ticker_symbol: str, lookback_days: int = 365) -> list[dict]:
    print(f"DEBUG: Calling fetch_all_sec_filings for {ticker_symbol}") # DEBUG
    cik = get_cik_for_ticker(ticker_symbol)
    if not cik:
        print(f"DEBUG: No CIK found for {ticker_symbol} during initial lookup.") # DEBUG
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
            print(f"DEBUG: CIK found via lookup for {ticker_symbol}: {cik}") # DEBUG
        except Exception as e:
            print(f"DEBUG: Error during CIK lookup for {ticker_symbol}: {e}") # DEBUG
            pass
    if not cik:
        print(f"DEBUG: CIK still not found for {ticker_symbol}.") # DEBUG
        return [{"error": f"SEC: CIK not found for {ticker_symbol}."}]
    
    cik_padded = str(cik).zfill(10)
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    headers = {'User-Agent': SEC_USER_AGENT}; filings_list = []
    try:
        response = requests.get(submissions_url, headers=headers, timeout=20); response.raise_for_status()
        submissions_data = response.json()
        print(f"DEBUG: Fetched submissions data for CIK {cik_padded}.") # DEBUG

        today_utc = datetime.now(timezone.utc) # Use timezone-aware datetime
        date_limit = today_utc - timedelta(days=lookback_days)
        
        if 'filings' in submissions_data and 'recent' in submissions_data['filings']:
            recent = submissions_data['filings']['recent']
            forms, dates, acc_nos, docs = recent.get('form',[]), recent.get('filingDate',[]), recent.get('accessionNumber',[]), recent.get('primaryDocument',[])
            metadata = []
            for i in range(len(forms)):
                try:
                    filing_date = datetime.strptime(dates[i], '%Y-%m-%d').replace(tzinfo=timezone.utc)
                    if filing_date >= date_limit:
                        metadata.append({"form_type": forms[i], "filing_date_str": dates[i], "accession_number": acc_nos[i], "primary_document": docs[i]})
                except (ValueError, IndexError) as e:
                    print(f"DEBUG: Error processing filing metadata for {ticker_symbol} at index {i}: {e}") # DEBUG
                    continue
            
            print(f"DEBUG: Found {len(metadata)} recent relevant filings metadata for {ticker_symbol}.") # DEBUG
            xml_fetches, max_xml, max_other = 0, 20, 15 
            for info in metadata:
                form, date_str, acc_no, doc_name = info["form_type"], info["filing_date_str"], info["accession_number"], info["primary_document"]
                acc_no_dashless = acc_no.replace('-', '')
                idx_link = f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{acc_no_dashless}/{acc_no}-index.html"
                
                document_content_placeholder = ""
                if form in ['10-K', '10-Q', '8-K']:
                    document_content_placeholder = f"Content for {form} on {date_str} (Placeholder for actual scraped text from {doc_name})."
                
                if form == '4' and doc_name.lower().endswith(('.xml', '.xsd')):
                    if xml_fetches >= max_xml:
                        print(f"DEBUG: Max XML fetches reached for {ticker_symbol}. Skipping Form 4.") # DEBUG
                        continue
                    xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{acc_no_dashless}/{doc_name}"
                    try:
                        filing_resp = requests.get(xml_url, headers=headers, timeout=10)
                        if filing_resp.status_code != 200:
                            print(f"DEBUG: Failed to fetch Form 4 XML {xml_url}: Status {filing_resp.status_code}") # DEBUG
                            continue
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
                                    print(f"DEBUG: Parsed Form 4 TX for {ticker_symbol}: {tx_code} {shares} @ {price}") # DEBUG
                    except Exception as e: # Broad catch for parsing issues in individual Form 4s
                        print(f"DEBUG: Error parsing Form 4 for {ticker_symbol} from {xml_url}: {e}") # DEBUG
                        continue # continue to next filing in metadata
                elif len([f for f in filings_list if not f.get("is_form4_transaction")]) < max_other:
                    filings_list.append({"is_form4_transaction": False, "ticker": ticker_symbol, "filing_date": date_str, "form_type": form, "document_link": f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{acc_no_dashless}/{doc_name}", "summary_link": idx_link, "document_content_for_llm": document_content_placeholder})
                    print(f"DEBUG: Added other filing for {ticker_symbol}: {form} on {date_str}") # DEBUG
            
            if not filings_list and xml_fetches > 0:
                print(f"DEBUG: SEC: {xml_fetches} Form 4s for {ticker_symbol}, but no tx parsed.") # DEBUG
                return [{"error": f"SEC: {xml_fetches} Form 4s for {ticker_symbol}, but no tx parsed."}]
            if not filings_list:
                print(f"DEBUG: SEC: No relevant filings for {ticker_symbol} (CIK:{cik_padded}).") # DEBUG
                return [{"error": f"SEC: No relevant filings for {ticker_symbol} (CIK:{cik_padded})."}]
        else:
            print(f"DEBUG: SEC: No recent filings data for {ticker_symbol} (CIK:{cik_padded}).") # DEBUG
            return [{"error": f"SEC: No recent filings data for {ticker_symbol} (CIK:{cik_padded})."}]
    except requests.exceptions.HTTPError as e:
        print(f"DEBUG: SEC HTTP error ({ticker_symbol}, CIK:{cik_padded}): {e}") # DEBUG
        return [{"error": f"SEC HTTP error ({ticker_symbol}, CIK:{cik_padded}): {e}"}]
    except requests.exceptions.RequestException as e:
        print(f"DEBUG: SEC Request error ({ticker_symbol}, CIK:{cik_padded}): {e}") # DEBUG
        return [{"error": f"SEC Request error ({ticker_symbol}, CIK:{cik_padded}): {e}"}]
    except Exception as e:
        print(f"DEBUG: SEC Unexpected error ({ticker_symbol}, CIK:{cik_padded}): {e}") # DEBUG
        return [{"error": f"SEC Unexpected error ({ticker_symbol}, CIK:{cik_padded}): {e}"}]
    filings_list.sort(key=lambda x: x.get('filing_date', '1900-01-01'), reverse=True);
    print(f"DEBUG: Returned {len(filings_list)} SEC filings for {ticker_symbol}.") # DEBUG
    return filings_list

@st.cache_data(ttl=6*3600)
def fetch_inst_filings(ticker: str) -> list[dict]:
    print(f"DEBUG: Calling fetch_inst_filings (yfinance) for {ticker}") # DEBUG
    try:
        df_holders = yf.Ticker(ticker).institutional_holders
        if df_holders is not None and not df_holders.empty:
            if 'Shares' in df_holders.columns: df_holders['Shares'] = pd.to_numeric(df_holders['Shares'], errors='coerce').fillna(0)
            if '% Out' in df_holders.columns: df_holders['% Out'] = pd.to_numeric(df_holders['% Out'], errors='coerce').fillna(0.0)
            if 'Date Reported' in df_holders.columns: df_holders['Date Reported'] = df_holders['Date Reported'].astype(str)
            print(f"DEBUG: Fetched {len(df_holders)} institutional holders for {ticker}.") # DEBUG
            return df_holders.to_dict("records")
        print(f"DEBUG: No yfinance institutional holder data for {ticker}.") # DEBUG
        return [{"error": f"No yfinance institutional holder data for {ticker}."}]
    except Exception as e:
        print(f"DEBUG: Error fetching institutional holders for {ticker}: {e}") # DEBUG
        return [{"error": f"yfinance institutional holders fetch failed for {ticker}: {e}"}]

@st.cache_data(ttl=4 * 3600)
def fetch_value_investing_io_data(ticker: str) -> dict:
    print(f"DEBUG: Calling fetch_value_investing_io_data for {ticker}") # DEBUG
    url = f"https://valueinvesting.io/{ticker.upper()}/valuation/fair-value"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=15); response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser'); target_text = None
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if ticker.upper() in text and "Fair Value" in text and ("Peter Lynch" in text or "based on" in text or "valuation model" in text):
                target_text = text; break
        if not target_text:
            print(f"DEBUG: VI.io: Peter Lynch Fair Value paragraph not found for {ticker}.") # DEBUG
            return {"error": f"VI.io: Peter Lynch Fair Value paragraph not found for {ticker}."}
        pattern = re.compile(r"As of (?P<date>[\d]{4}-[\d]{2}-[\d]{2}), the Fair Value of .*?\(.*?" + re.escape(ticker.upper()) + r".*?\) is (?P<fair_value>[\d\.]+) USD\.?" + r"(?:.*?With the current market price of (?P<market_price>[\d\.]+) USD, the upside of .*? is (?P<upside_percent>[-+]?\d+\.?\d*)%\.?)?")
        match = pattern.search(target_text)
        if match:
            data = match.groupdict()
            print(f"DEBUG: VI.io: Successfully parsed data for {ticker}.") # DEBUG
            return {"ticker": ticker, "vi_valuation_date": data.get("date"), "vi_fair_value": float(data.get("fair_value")) if data.get("fair_value") else None, "vi_site_market_price": float(data.get("market_price")) if data.get("market_price") else None, "vi_upside_percent": float(data.get("upside_percent")) if data.get("upside_percent") else None, "vi_full_text": target_text, "vi_data_source_url": url, "error": None}
        else:
            fv_match = re.search(r"Fair Value.*?is ([\d\.]+) USD", target_text)
            if fv_match:
                print(f"DEBUG: VI.io: Generic parse for {ticker}.") # DEBUG
                return {"ticker": ticker, "vi_valuation_date": "N/A (generic)", "vi_fair_value": float(fv_match.group(1)) if fv_match.group(1) else None, "vi_site_market_price": None, "vi_upside_percent": None, "vi_full_text": target_text, "vi_data_source_url": url, "error": None, "note": "Generic parse."}
            print(f"DEBUG: VI.io: Could not parse details for {ticker}.") # DEBUG
            return {"error": f"VI.io: Could not parse details for {ticker} from: '{target_text[:200]}...'"}
    except requests.exceptions.HTTPError as http_err:
        print(f"DEBUG: VI.io: HTTP error for {ticker}: {http_err.response.status_code}") # DEBUG
        return {"error": f"VI.io: HTTP error for {ticker} ({http_err.response.status_code if http_err.response else 'Unknown'}): {url}"}
    except requests.exceptions.RequestException as req_err:
        print(f"DEBUG: VI.io: Request error for {ticker}: {req_err}") # DEBUG
        return {"error": f"VI.io: Request error for {ticker}: {req_err}"}
    except Exception as e:
        print(f"DEBUG: VI.io: Unexpected error for {ticker}: {e}") # DEBUG
        return {"error": f"VI.io: Unexpected error for {ticker}: {e}"}

@st.cache_data(ttl=3600)
def fetch_politician_trades(ticker: str, days_back: int = 365) -> list[dict]:
    print(f"DEBUG: Calling fetch_politician_trades for {ticker}") # DEBUG
    url = f"https://www.capitoltrades.com/trades?asset={ticker.upper()}&pageSize=100&perPage=100"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36', 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8', 'Accept-Language': 'en-US,en;q=0.5', 'Referer': 'https://www.capitoltrades.com/'}
    trades_list = []
    try:
        response = requests.get(url, headers=headers, timeout=20); response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        trade_rows = soup.select("a[href^='/trades/'][class*='trade-row'], a[href^='/trades/'][class*='issuer-trade-row']")
        if not trade_rows: trade_rows = soup.find_all('a', href=lambda href: href and href.startswith('/trades/'))
        if not trade_rows:
            print(f"DEBUG: CT: No trade rows found for {ticker}.") # DEBUG
            return [{"error": f"CT: No trade rows found for {ticker}. Website HTML structure may have changed or scraping blocked. This feature is experimental."}]
        
        print(f"DEBUG: Found {len(trade_rows)} potential trade rows for {ticker}.") # DEBUG
        for row in trade_rows[:20]: # Limit to first 20 for performance/relevance
            name_tag = row.find(['div','span'], class_=lambda x: x and ('politician-name' in x or 'filer-name' in x))
            type_tag = row.find(['div','span'], class_=lambda x: x and ('tx-type' in x or 'transaction-type' in x))
            val_tag = row.find(['div','span'], class_=lambda x: x and ('tx-value' in x or 'transaction-value' in x))
            date_tag = row.find(['div','span'], class_=lambda x: x and ('tx-date' in x or 'transaction-date' in x))
            if all([name_tag, type_tag, val_tag, date_tag]):
                name, tx_type_text = name_tag.text.strip(), type_tag.text.strip().lower()
                tx_type = "purchase" if "purchase" in tx_type_text else ("sale" if "sale" in tx_type_text else "other")
                val_range, date_str, val_est = val_tag.text.strip(), date_tag.text.strip(), 0
                matches = re.findall(r'\$([\d,]+)', val_range)
                if matches:
                    try: val_est = int(matches[0].replace(',',''))
                    except ValueError: pass
                trades_list.append({"politician_name": name, "transaction_type": tx_type, "value_range": val_range, "value_estimate_lower": val_est, "date_str": date_str, "source_url": urljoin("https://www.capitoltrades.com", row['href'])})
        
        if not trades_list and trade_rows:
            print(f"DEBUG: CT: Found rows for {ticker}, but failed parsing.") # DEBUG
            return [{"error": f"CT: Found rows for {ticker}, but failed parsing. HTML details may have changed."}]
        print(f"DEBUG: Processed {len(trades_list)} politician trades for {ticker}.") # DEBUG
        return trades_list
    except requests.exceptions.Timeout:
        print(f"DEBUG: CT: Timeout fetching {ticker}.") # DEBUG
        return [{"error": f"CT: Timeout fetching {ticker}."}]
    except requests.exceptions.HTTPError as e:
        print(f"DEBUG: CT: HTTP error {e.response.status_code if e.response else ''} for {ticker}.") # DEBUG
        return [{"error": f"CT: HTTP error {e.response.status_code if e.response else ''} for {ticker}."}]
    except requests.exceptions.RequestException as e:
        print(f"DEBUG: CT: Request error for {ticker}: {e}.") # DEBUG
        return [{"error": f"CT: Request error for {ticker}: {e}."}]
    except Exception as e:
        print(f"DEBUG: CT: Parsing error for {ticker}: {e}.") # DEBUG
        return [{"error": f"CT: Parsing error for {ticker}: {e}."}]

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
        print(f"DEBUG: ModelClient initialized for {provider} with model {self.model_name}.") # DEBUG

    def generate(self, prompt: str) -> str:
        try:
            print(f"DEBUG: Generating LLM response (first 100 chars of prompt): {prompt[:100]}...") # DEBUG
            stream = self.client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt}], stream=True)
            response_content = "".join(c.choices[0].delta.content for c in stream if c.choices and c.choices[0].delta and c.choices[0].delta.content)
            print(f"DEBUG: LLM response generated (first 100 chars): {response_content[:100]}...") # DEBUG
            return response_content
        except Exception as e:
            print(f"DEBUG: LLM Error in generate: {e}") # DEBUG
            raise Exception(f"LLM Error ({self.provider}, {self.model_name}): {e}")

class PriceAgent:
    def run(self, ticker: str, price_data_slice: pd.DataFrame) -> dict:
        print(f"DEBUG: PriceAgent running for {ticker}. Data shape: {price_data_slice.shape if not price_data_slice.empty else 'empty'}") # DEBUG
        required_data_points = 200

        if price_data_slice.empty or len(price_data_slice) < required_data_points:
            error_msg = f"Not enough data ({len(price_data_slice)} rows) for comprehensive PriceAgent analysis (requires {required_data_points})." # DEBUG
            print(f"DEBUG: PriceAgent error: {error_msg}") # DEBUG
            return {
                "ticker": ticker,
                "price_signal": "hold",
                "sma50": np.nan, "sma200": np.nan, "rsi14": np.nan,
                "bb_upper": np.nan, "bb_lower": np.nan, "bb_signal": "hold",
                "price_confidence_score": 0.0,
                "price_error": error_msg
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
            error_msg = "Some key indicators are NaN at the latest data point after calculation." # DEBUG
            print(f"DEBUG: PriceAgent calculation error: {error_msg}") # DEBUG
            return {
                "ticker": ticker,
                "price_signal": "hold",
                "sma50": float(latest.SMA50) if pd.notna(latest.SMA50) else np.nan,
                "sma200": float(latest.SMA200) if pd.notna(latest.SMA200) else np.nan,
                "rsi14": float(latest.RSI14) if pd.notna(latest.RSI14) else np.nan,
                "bb_upper": float(latest.BB_Upper) if pd.notna(latest.BB_Upper) else np.nan,
                "bb_lower": float(latest.BB_Lower) if pd.notna(latest.BB_Lower) else np.nan,
                "bb_signal": "hold",
                "price_confidence_score": 0.0,
                "price_error": error_msg
            }

        current_close = latest.Close

        # ... (rest of PriceAgent logic unchanged) ...
        if confidence_score > 0.3:
            final_price_signal = "buy"
        elif confidence_score < -0.3:
            final_price_signal = "sell"
        else:
            final_price_signal = "hold"
            
        confidence_score = max(-1.0, min(1.0, confidence_score))
        print(f"DEBUG: PriceAgent for {ticker} signal: {final_price_signal}, confidence: {confidence_score:.2f}") # DEBUG
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
    def run(self, ticker: str, price_data_slice: pd.DataFrame) -> dict:
        print(f"DEBUG: MomentumAgent running for {ticker}. Data shape: {price_data_slice.shape if not price_data_slice.empty else 'empty'}") # DEBUG
        required_data_points = 253

        if price_data_slice.empty or len(price_data_slice) < required_data_points:
            error_msg = f"Not enough data ({len(price_data_slice)} rows) for 1-year and 1-month momentum (requires {required_data_points})." # DEBUG
            print(f"DEBUG: MomentumAgent error: {error_msg}") # DEBUG
            return {
                "ticker": ticker,
                "momentum_signal": "hold",
                "momentum_1m": np.nan, "momentum_12m": np.nan,
                "momentum_confidence_score": 0.0,
                "momentum_error": error_msg
            }

        df = price_data_slice.copy()

        if 'Close' not in df.columns or not pd.api.types.is_numeric_dtype(df['Close']):
            error_msg = "Price data is missing 'Close' column or not numeric for momentum." # DEBUG
            print(f"DEBUG: MomentumAgent data quality error: {error_msg}") # DEBUG
            return {
                "ticker": ticker, "momentum_signal": "hold",
                "momentum_1m": np.nan, "momentum_12m": np.nan,
                "momentum_confidence_score": 0.0,
                "momentum_error": error_msg
            }
        
        # ... (rest of MomentumAgent logic unchanged) ...
        
        P_t = df["Close"].iloc[-1]
        P_1m_series = df["Close"].shift(21)
        P_12m_series = df["Close"].shift(252)

        P_1m = P_1m_series.iloc[-1]
        P_12m = P_12m_series.iloc[-1]

        m1 = ((P_t / P_1m) - 1) if pd.notna(P_1m) and P_1m != 0 else np.nan
        m12 = ((P_t / P_12m) - 1) if pd.notna(P_12m) and P_12m != 0 else np.nan

        signal = "hold"
        confidence_score = 0.0

        if pd.notna(m1) and pd.notna(m12):
            raw_combined_momentum = (m1 + m12) / 2
            scaled_confidence = raw_combined_momentum * 5.0 
            confidence_score = max(-1.0, min(1.0, scaled_confidence))

            if confidence_score > 0.3:
                signal = "buy"
            elif confidence_score < -0.3:
                signal = "sell"
            else:
                signal = "hold"
        else:
            print(f"DEBUG: MomentumAgent: m1 or m12 is NaN. Cannot generate strong signal.") # DEBUG

        print(f"DEBUG: MomentumAgent for {ticker} signal: {signal}, confidence: {confidence_score:.2f}") # DEBUG
        return {
            "ticker": ticker,
            "momentum_1m": float(m1) if pd.notna(m1) else np.nan,
            "momentum_12m": float(m12) if pd.notna(m12) else np.nan,
            "momentum_signal": signal,
            "momentum_confidence_score": float(confidence_score),
            "momentum_error": None
        }


class VolatilityAgent:
    def run(self, ticker: str, data: dict, price_data_slice: pd.DataFrame = None) -> dict:
        print(f"DEBUG: VolatilityAgent running for {ticker}. Price data shape: {price_data_slice.shape if price_data_slice is not None and not price_data_slice.empty else 'empty'}") # DEBUG
        beta_val = data.get("ticker_info", {}).get("beta")
        beta = float(beta_val) if isinstance(beta_val, (int, float)) else 1.0

        ann_vol = np.nan
        vol_weight = 0.0
        volatility_signal = "hold"
        volatility_confidence_score = 0.0
        volatility_error = None

        if price_data_slice is not None and not price_data_slice.empty and len(price_data_slice) > 1:
            if 'Close' not in price_data_slice.columns or not pd.api.types.is_numeric_dtype(price_data_slice['Close']):
                volatility_error = "Price data is missing 'Close' column or not numeric for volatility calculation."
                print(f"DEBUG: VolatilityAgent data quality error: {volatility_error}") # DEBUG
            else:
                ret = np.log(price_data_slice.Close / price_data_slice.Close.shift(1)).dropna()

                if not ret.empty:
                    daily_std = ret.std()
                    if daily_std > 0:
                        ann_vol = float(daily_std * np.sqrt(252))
                        vol_weight = float(1 / ann_vol)
                        print(f"DEBUG: VolatilityAgent calculated ann_vol: {ann_vol:.2f}") # DEBUG
                    else:
                        volatility_error = "Daily returns standard deviation is zero (no price movement)."
                        print(f"DEBUG: VolatilityAgent calculation error: {volatility_error}") # DEBUG
                else:
                    volatility_error = "Not enough valid returns to calculate historical volatility."
                    print(f"DEBUG: VolatilityAgent data insufficiency: {volatility_error}") # DEBUG
        else:
            volatility_error = "Not enough price data for historical volatility calculation."
            print(f"DEBUG: VolatilityAgent data insufficiency: {volatility_error}") # DEBUG

        # ... (rest of VolatilityAgent logic unchanged) ...
        print(f"DEBUG: VolatilityAgent for {ticker} signal: {volatility_signal}, confidence: {volatility_confidence_score:.2f}") # DEBUG
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
    def __init__(self, client):
        self.client = client
        print("DEBUG: SentimentAgent initialized.") # DEBUG

    def run(self, ticker: str, data: dict) -> dict:
        print(f"DEBUG: SentimentAgent running for {ticker}.") # DEBUG
        news, news_err = data.get("news", []), data.get("news_fetch_status_error")

        if news_err:
            print(f"DEBUG: SentimentAgent: News fetch error: {news_err}") # DEBUG
            return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": news_err}

        valid_news = [item for item in news if isinstance(item, dict) and "error" not in item]

        if not valid_news:
            err_msg = news[0].get("error") if news and isinstance(news[0], dict) and "error" in news[0] else "No valid news articles found for sentiment analysis."
            print(f"DEBUG: SentimentAgent: No valid news: {err_msg}") # DEBUG
            return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": err_msg}

        content_for_llm = []
        co_name = data.get("ticker_info", {}).get('longName', ticker)
        
        MAX_NEWS_ARTICLES_FOR_LLM = 10 

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
            error_msg = "No processable news articles with sufficient content for sentiment analysis." # DEBUG
            print(f"DEBUG: SentimentAgent: {error_msg}") # DEBUG
            return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": error_msg}

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
                    print(f"DEBUG: Sentiment LLM extra text: {llm_err}") # DEBUG
            else:
                llm_err = f"LLM did not output a recognizable number: '{resp[:50]}...'"
                print(f"DEBUG: Sentiment LLM no number: {llm_err}") # DEBUG
                score = 0.0

        except Exception as e:
            llm_err = f"LLM sentiment analysis call failed: {str(e)[:150]}"
            print(f"DEBUG: Sentiment LLM call failed: {llm_err}") # DEBUG
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
        
        print(f"DEBUG: SentimentAgent for {ticker} signal: {sentiment_signal}, score: {score:.2f}") # DEBUG
        return {
            "ticker": ticker,
            "sentiment_score": float(score),
            "sentiment_signal": sentiment_signal,
            "sentiment_confidence_score": float(sentiment_confidence_score),
            "sentiment_error": final_err
        }

class NewsSummaryAgent:
    def __init__(self, client):
        self.client = client
        print("DEBUG: NewsSummaryAgent initialized.") # DEBUG
    def run(self, ticker: str, data: dict) -> dict:
        print(f"DEBUG: NewsSummaryAgent running for {ticker}.") # DEBUG
        news, co_name, news_fetch_err = data.get("news",[]), data.get("ticker_info",{}).get('longName',ticker), data.get("news_fetch_status_error")
        if news_fetch_err:
            print(f"DEBUG: NewsSummaryAgent: News fetch issues: {news_fetch_err}") # DEBUG
            return {"ticker":ticker, "news_summary":"Summary skipped due to news fetch issues.", "news_summary_error":news_fetch_err}
        if not news or (isinstance(news[0],dict) and "error" in news[0] and not any("error" not in item for item in news)):
            err = news[0]["error"] if news and isinstance(news[0],dict) and "error" in news[0] else "No news for summary."
            print(f"DEBUG: NewsSummaryAgent: No news for summary: {err}") # DEBUG
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
        
        if not final_snips:
            error_msg = "No content for summary." # DEBUG
            print(f"DEBUG: NewsSummaryAgent: {error_msg}") # DEBUG
            return {"ticker":ticker, "news_summary":"No content for summary.", "news_summary_error":"No articles with content/desc."}
        
        prompt = f"Concise summary (max 200 words) for {co_name} ({ticker})...\n\nArticles:\n" + "\n".join(f"- {s}" for s in final_snips)
        summary, err_msg = "Could not generate summary.", None
        try:
            resp = self.client.generate(prompt).strip()
            if resp.startswith("Error:"):
                err_msg = resp
                print(f"DEBUG: NewsSummaryAgent LLM error: {err_msg}") # DEBUG
            else:
                summary = resp
                print(f"DEBUG: NewsSummaryAgent LLM summary generated.") # DEBUG
        except Exception as e:
            err_msg = f"LLM summary call failed: {str(e)[:150]}"
            print(f"DEBUG: NewsSummaryAgent LLM call failed: {err_msg}") # DEBUG
        return {"ticker":ticker, "news_summary":summary, "news_summary_error":err_msg}

class FundamentalsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        print(f"DEBUG: FundamentalsAgent running for {ticker}.") # DEBUG
        s = data.get("ticker_info",{}); mc, fcf, roe, de = s.get("marketCap"), s.get("freeCashflow"), s.get("returnOnEquity"), s.get("debtToEquity")
        mc_c = mc if isinstance(mc,(int,float)) else 1; fcf_c = fcf if isinstance(fcf,(int,float)) else 0
        roe_c = roe if isinstance(roe,(int,float)) else 0; de_c = de if isinstance(de,(int,float)) else 1000
        fcy = fcf_c / mc_c if mc_c != 0 else 0
        ps = sum([roe_c > 0.01, de_c < 100, fcf_c > 0]); sig = "hold"
        if ps >= 2: sig = "buy"
        elif ps == 0: sig = "sell"
        print(f"DEBUG: FundamentalsAgent for {ticker} signal: {sig}, Piotroski: {ps}") # DEBUG
        return {"ticker":ticker, "fcf_yield":float(fcy), "piotroski_score":int(ps), "fund_signal":sig}

class ValuationAgent:
    def run(self, ticker: str, data: dict) -> dict:
        print(f"DEBUG: ValuationAgent running for {ticker}.") # DEBUG
        stats, hist = data.get("ticker_info",{}), data.get("price_history"); price_v = stats.get("currentPrice")
        if price_v is None and hist is not None and not hist.empty: price_v = hist.Close.iloc[-1]
        curr_p = float(price_v) if isinstance(price_v,(int,float)) and price_v > 0 else None
        if curr_p is None:
            error_msg = "Current price unavailable for valuation." # DEBUG
            print(f"DEBUG: ValuationAgent error: {error_msg}") # DEBUG
            return {"ticker":ticker, "forward_pe":None, "relative_pe_signal":"hold", "dcf_fair_price":np.nan, "dcf_signal":"hold", "valuation_error":error_msg}
        pe_v = stats.get("forwardPE"); pe = float(pe_v) if isinstance(pe_v,(int,float)) else None; rel_sig = "hold"
        if pe is not None and pe > 0: rel_sig = "buy" if pe < 15 else ("sell" if pe > 25 else "hold")
        fcf_v, mc_v = stats.get("freeCashflow"), stats.get("marketCap")
        fcf, mc = (float(fcf_v) if isinstance(fcf_v,(int,float)) else None), (float(mc_v) if isinstance(mc_v,(int,float)) else None)
        fcy = (fcf / mc) if fcf is not None and mc is not None and mc != 0 else 0.0
        fp_est = curr_p * (1 + fcy); dcf_sig = "hold"
        if fp_est > curr_p * 1.15: dcf_sig = "buy"
        elif fp_est < curr_p * 0.85: dcf_sig = "sell"
        print(f"DEBUG: ValuationAgent for {ticker} PE sig: {rel_sig}, DCF sig: {dcf_sig}") # DEBUG
        return {"ticker":ticker, "forward_pe":pe, "relative_pe_signal":rel_sig, "dcf_fair_price":float(fp_est) if pd.notna(fp_est) else np.nan, "dcf_signal":dcf_sig, "valuation_error":None}

class AnalystRatingAgent:
    def run(self, ticker: str, data: dict) -> dict:
        print(f"DEBUG: AnalystRatingAgent running for {ticker}.") # DEBUG
        info, hist = data.get("ticker_info",{}), data.get("price_history"); price_v = info.get("currentPrice")
        if price_v is None and hist is not None and not hist.empty: price_v = hist.Close.iloc[-1]
        curr_p = float(price_v) if isinstance(price_v,(int,float)) and price_v > 0 else None
        if curr_p is None:
            error_msg = "Current price unavailable for analyst rating." # DEBUG
            print(f"DEBUG: AnalystRatingAgent error: {error_msg}") # DEBUG
            return {"ticker":ticker, "analyst_buy_pct_inferred":0.5, "target_upside":0.0, "yfinance_recommendation":"N/A", "analyst_signal":"hold", "analyst_error":error_msg}
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
        print(f"DEBUG: AnalystRatingAgent for {ticker} signal: {sig}, upside: {upside:.2%}") # DEBUG
        return {"ticker":ticker, "analyst_buy_pct_inferred":float(buy_pct), "target_upside":float(upside), "yfinance_recommendation":rec, "analyst_signal":sig, "analyst_error":None}

class SECFilingAgent:
    def run(self, ticker: str, data: dict) -> dict:
        print(f"DEBUG: SECFilingAgent running for {ticker}.") # DEBUG
        filings, err = data.get("sec_all_filings_raw",[]), None
        if not filings or (isinstance(filings[0],dict) and "error" in filings[0]):
            err = filings[0].get("error") if filings and isinstance(filings[0],dict) else f"SEC: No raw filings for {ticker}."
            print(f"DEBUG: SECFilingAgent error: {err}") # DEBUG
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
        print(f"DEBUG: SECFilingAgent for {ticker} signal: {sig}, net shares: {net_s}") # DEBUG
        return {"ticker":ticker, "sec_net_insider_shares_1y":int(net_s), "sec_insider_buy_value_1y":round(buy_v,2), "sec_insider_sell_value_1y":round(sell_v,2), "sec_filings_signal":sig, "sec_filings_error":None, "sec_recent_form4_transactions":form4[:10], "sec_other_recent_filings":others[:10]}

class SECSummaryAgent:
    def __init__(self, client):
        self.client = client
        print("DEBUG: SECSummaryAgent initialized.") # DEBUG

    def run(self, ticker: str, data: dict) -> dict:
        print(f"DEBUG: SECSummaryAgent running for {ticker}.") # DEBUG
        filings_raw = data.get("sec_all_filings_raw", [])
        co_name = data.get("ticker_info", {}).get('longName', ticker)

        if not self.client:
            print("DEBUG: SECSummaryAgent LLM not configured.") # DEBUG
            return {"ticker": ticker, "sec_summary": "LLM client not available for SEC summary.", "sec_summary_error": "LLM not configured."}
        
        relevant_filings = [
            f for f in filings_raw if not f.get("is_form4_transaction") and f.get("document_content_for_llm")
        ]
        print(f"DEBUG: SECSummaryAgent found {len(relevant_filings)} relevant filings for LLM.") # DEBUG
        
        if not relevant_filings:
            print("DEBUG: SECSummaryAgent: No relevant non-Form 4 filings with content found for summary.") # DEBUG
            return {"ticker": ticker, "sec_summary": "No relevant non-Form 4 filings with content found for summary.", "sec_summary_error": None}

        sorted_filings = sorted(
            relevant_filings,
            key=lambda x: (
                0 if x['form_type'] == '8-K' else
                1 if x['form_type'] == '10-Q' else
                2 if x['form_type'] == '10-K' else
                3,
                x['filing_date_str']
            ),
            reverse=False
        )
        
        filings_for_llm = sorted_filings[:3]
        print(f"DEBUG: SECSummaryAgent sending {len(filings_for_llm)} filings to LLM.") # DEBUG

        content_for_llm = []
        for f in filings_for_llm:
            content_for_llm.append(
                f"Form Type: {f.get('form_type')} (Filed: {f.get('filing_date_str')})\n"
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
                print(f"DEBUG: SECSummaryAgent LLM error: {llm_err}") # DEBUG
            else:
                summary = resp
                print(f"DEBUG: SECSummaryAgent LLM summary generated.") # DEBUG
        except Exception as e:
            llm_err = f"LLM SEC summary call failed: {str(e)[:150]}"
            print(f"DEBUG: SECSummaryAgent LLM call failed: {llm_err}") # DEBUG
            
        return {
            "ticker": ticker,
            "sec_summary_llm": summary,
            "sec_summary_error": llm_err
        }


class InstitutionalHoldingsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        print(f"DEBUG: InstitutionalHoldingsAgent running for {ticker}.") # DEBUG
        holdings, err = data.get("institutional_holdings",[]), None
        if holdings and isinstance(holdings[0],dict) and "error" in holdings[0]:
            err = holdings[0]["error"]
            print(f"DEBUG: InstitutionalHoldingsAgent error: {err}") # DEBUG
            return {"ticker":ticker, "inst_num_holders":0, "inst_total_shares_held":0, "inst_total_pct_out":0.0, "inst_holdings_signal":"hold", "inst_holdings_error":err, "inst_top_holders":[]}
        
        num_h, total_s, total_pct, top_h = 0,0,0.0,[]
        if holdings:
            valid_h = [d for d in holdings if isinstance(d,dict) and "error" not in d]
            if valid_h:
                num_h = len(valid_h)
                try:
                    total_s = sum(d.get('Shares',0) for d in valid_h); total_pct = sum(d.get('% Out',0.0) for d in valid_h)
                    for d in valid_h:
                        d['Shares'] = pd.to_numeric(d.get('Shares'), errors='coerce').fillna(0)
                    top_h = sorted(valid_h, key=lambda x: x.get('Shares',0), reverse=True)[:10]
                    print(f"DEBUG: InstitutionalHoldingsAgent processed {num_h} holders for {ticker}.") # DEBUG
                except Exception as e:
                    err = f"Error processing inst holdings: {e}"
                    print(f"DEBUG: InstitutionalHoldingsAgent processing error: {e}") # DEBUG
            elif not err:
                err = "No valid inst holdings."
                print(f"DEBUG: InstitutionalHoldingsAgent: No valid holdings found.") # DEBUG
        
        sig = "hold"
        inst_confidence_score = 0.0
        if not err:
            if total_pct > 0.60: sig = "buy"
            elif total_pct < 0.10 and num_h > 0: sig = "sell"
            if total_pct > 0.75: inst_confidence_score = 0.8
            elif total_pct > 0.60: inst_confidence_score = 0.5
            elif total_pct < 0.05 and num_h > 0: inst_confidence_score = -0.8
            elif total_pct < 0.10 and num_h > 0: inst_confidence_score = -0.5
        
        print(f"DEBUG: InstitutionalHoldingsAgent for {ticker} signal: {sig}, pct_out: {total_pct:.2%}") # DEBUG
        return {"ticker":ticker, "inst_num_holders":num_h, "inst_total_shares_held":int(total_s), "inst_total_pct_out":float(total_pct), "inst_holdings_signal":sig, "inst_holdings_error":err, "inst_top_holders":top_h, "inst_confidence_score": float(inst_confidence_score)}

class PoliticianFilingsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        print(f"DEBUG: PoliticianFilingsAgent running for {ticker}.") # DEBUG
        trades, err = data.get("politician_trades",[]), None
        net_val, buys, sells = 0,0,0
        if trades and isinstance(trades,list) and len(trades)>0 and isinstance(trades[0],dict) and "error" in trades[0]:
            err = trades[0]["error"]
            print(f"DEBUG: PoliticianFilingsAgent error: {err}") # DEBUG
        elif trades:
            for trade in trades:
                if isinstance(trade,dict):
                    val = trade.get("value_estimate_lower",0)
                    if trade.get("transaction_type")=="purchase": net_val += val; buys +=1
                    elif trade.get("transaction_type")=="sale": net_val -= val; sells +=1
            print(f"DEBUG: PoliticianFilingsAgent processed {buys} buys, {sells} sells for {ticker}.") # DEBUG
        sig = "hold"
        politician_confidence = 0.0
        if not err:
            if buys > sells and buys >= 2 and net_val > 50000:
                sig = "buy"; politician_confidence = 0.6
            elif sells > buys and sells >= 2 and net_val < -50000:
                sig = "sell"; politician_confidence = -0.6
            elif buys > sells and buys >= 1:
                sig = "hold"
                politician_confidence = 0.1
            elif sells > buys and sells >= 1:
                sig = "hold"
                politician_confidence = -0.1
            
        print(f"DEBUG: PoliticianFilingsAgent for {ticker} signal: {sig}, net_val: {net_val}") # DEBUG
        return {"ticker":ticker, "politician_net_trade_value_estimate":net_val, "politician_buy_tx_count":buys, "politician_sell_tx_count":sells, "politician_filings_signal":sig, "politician_data_error":err, "politician_confidence_score": float(politician_confidence)}

class ValueInvestingIOAgent:
    def run(self, ticker: str, data: dict) -> dict:
        print(f"DEBUG: ValueInvestingIOAgent running for {ticker}.") # DEBUG
        vi, err = data.get("value_investing_io_data",{}), data.get("value_investing_io_data",{}).get("error")
        fv, site_mp, up_pct, val_date, text = vi.get("vi_fair_value"), vi.get("vi_site_market_price"), vi.get("vi_upside_percent"), vi.get("vi_valuation_date"), vi.get("vi_full_text")
        sig = "hold"; curr_pyf_val = data.get("ticker_info",{}).get("currentPrice")
        if curr_pyf_val is None and data.get("price_history") is not None and not data["price_history"].empty: curr_pyf_val = data["price_history"].Close.iloc[-1]
        curr_pyf = float(curr_pyf_val) if isinstance(curr_pyf_val,(int,float)) and curr_pyf_val > 0 else None
        
        vi_confidence_score = 0.0
        if not err and fv is not None and curr_pyf is not None:
            mos = 0.15 # Margin of safety
            if up_pct is not None:
                if up_pct > (mos*100+5):
                    sig="buy"; vi_confidence_score = 0.8
                elif up_pct < -(mos*100+5):
                    sig="sell"; vi_confidence_score = -0.8
                elif up_pct > mos*100:
                    sig="buy"; vi_confidence_score = 0.5
                elif up_pct < -mos*100:
                    sig="sell"; vi_confidence_score = -0.5
            else:
                if curr_pyf < fv*(1-mos):
                    sig="buy"; vi_confidence_score = 0.6
                elif curr_pyf > fv*(1+mos):
                    sig="sell"; vi_confidence_score = -0.6
        
        print(f"DEBUG: ValueInvestingIOAgent for {ticker} signal: {sig}, FV: {fv}") # DEBUG
        return {"ticker":ticker, "vi_fair_value_estimate":fv, "vi_site_market_price":site_mp, "vi_upside_percent":up_pct, "vi_valuation_date":val_date, "vi_valuation_text_display":text, "vi_signal":sig, "vi_data_error":err, "vi_confidence_score": float(vi_confidence_score)}


class PortfolioAgent:
    WEIGHTS = {
        "price": 1.0, "momentum": 0.8, "volatility": 0.2, 
        "sentiment": 0.7, "fund": 0.9, "valuation_dcf": 0.6, "valuation_pe": 0.4,
        "sec_filings": 0.6, "sec_summary": 0.7,
        "inst_holdings": 0.3, "analyst": 0.5,
        "politician_filings": 0.2, "vi_signal": 0.8
    }

    def run(self, ticker: str, signals: list[dict], agent_weights: dict = None) -> dict:
        print(f"DEBUG: PortfolioAgent running for {ticker}.") # DEBUG
        curr_w, total_score, sum_w, agg_s = agent_weights or self.WEIGHTS, 0,0,{}
        for s_dict in signals:
            if isinstance(s_dict, dict): agg_s.update(s_dict)
        
        s_map = {
            "price_signal": ("price", "price_confidence_score"),
            "momentum_signal": ("momentum", "momentum_confidence_score"),
            "volatility_signal": ("volatility", "volatility_confidence_score"),
            "sentiment_signal": ("sentiment", "sentiment_confidence_score"),
            "fund_signal": ("fund", None),
            "dcf_signal": ("valuation_dcf", None),
            "relative_pe_signal": ("valuation_pe", None),
            "sec_filings_signal": ("sec_filings", None),
            "sec_summary_llm": ("sec_summary", None),
            "inst_holdings_signal": ("inst_holdings", "inst_confidence_score"),
            "analyst_signal": ("analyst", None),
            "politician_filings_signal": ("politician_filings", "politician_confidence_score"),
            "vi_signal": ("vi_signal", "vi_confidence_score")
        }

        for s_key, (w_key, conf_key) in s_map.items():
            s_val = agg_s.get(s_key)
            w = curr_w.get(w_key, 0)
            
            if s_key == "sec_summary_llm" and s_val and w > 0:
                if agg_s.get("sec_summary_error"):
                    raw_score = 0
                elif "negative" in s_val.lower() and "no significant events" not in s_val.lower():
                    raw_score = -0.5
                elif "positive" in s_val.lower() and "no significant events" not in s_val.lower():
                    raw_score = 0.5
                else:
                    raw_score = 0.1
                total_score += raw_score * w
                sum_w += w
            elif s_val and w > 0 and s_val in ["buy", "hold", "sell"]:
                raw_score = {"buy":1, "hold":0, "sell":-1}.get(s_val,0)
                
                if conf_key and pd.notna(agg_s.get(conf_key)):
                    agent_confidence = agg_s.get(conf_key)
                    total_score += (raw_score * agent_confidence) * w
                    sum_w += w * agent_confidence
                else:
                    total_score += raw_score * w
                    sum_w += w

        comp_score = (total_score/sum_w) if sum_w else 0.0
        decision = "buy" if comp_score > 0.15 else ("sell" if comp_score < -0.15 else "hold")
        print(f"DEBUG: PortfolioAgent for {ticker} composite score: {comp_score:.2f}, decision: {decision}") # DEBUG
        return {"ticker":ticker, "composite_score":comp_score, "final_decision":decision}

class AITraderAgent:
    def __init__(self, llm_client: ModelClient, stock_universe: dict):
        self.llm_client = llm_client
        self.stock_universe = stock_universe
        print("DEBUG: AITraderAgent initialized.") # DEBUG

    def _generate_trade_reason(self, ticker: str, decision: str, analysis: dict) -> str:
        if not self.llm_client:
            print("DEBUG: AITraderAgent LLM client not available for trade reason.") # DEBUG
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
            print(f"DEBUG: AITraderAgent generated trade reason for {ticker}: {reason[:50]}...") # DEBUG
            return reason
        except Exception as e:
            print(f"DEBUG: AITraderAgent failed to generate reason: {e}") # DEBUG
            return f"Could not generate reason due to LLM error: {e}"

    def _is_safe(self, analysis: dict) -> bool:
        info = analysis.get("ticker_info", {})
        market_cap = info.get("marketCap", 0)
        beta = info.get("beta", 1.0)
        is_safe_flag = isinstance(market_cap, (int, float)) and market_cap > 200e9 and isinstance(beta, (int, float)) and beta < 1.0
        print(f"DEBUG: AITraderAgent for {analysis.get('ticker')}: is_safe={is_safe_flag} (MC: {market_cap}, Beta: {beta})") # DEBUG
        return is_safe_flag

    def run(self, portfolio_state: dict, analysis_results: dict):
        print("DEBUG: AITraderAgent running to decide trades.") # DEBUG
        trades_to_make = []
        cash = portfolio_state['cash']
        holdings = list(portfolio_state['holdings'])

        tickers_in_portfolio = {h['ticker'] for h in holdings}
        
        for i, holding in reversed(list(enumerate(holdings))):
            ticker = holding['ticker']
            analysis = analysis_results.get(ticker)
            if not analysis or analysis.get('error'):
                print(f"DEBUG: AITraderAgent: Skipping {ticker} for sell decision due to missing or errored analysis.") # DEBUG
                continue

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
                    print(f"DEBUG: AITraderAgent: Decided to SELL {holding['quantity']:.2f} of {ticker} at ${price:.2f}.") # DEBUG
                else:
                    print(f"DEBUG: AITraderAgent: Cannot sell {ticker}, price invalid: {price}") # DEBUG

        current_holdings_value = 0
        for h in holdings:
            price = analysis_results.get(h['ticker'], {}).get('current_price_display')
            if isinstance(price, (int, float)):
                current_holdings_value += h['quantity'] * price
        total_portfolio_value = cash + current_holdings_value
        print(f"DEBUG: AITraderAgent: Current portfolio value after sells: ${total_portfolio_value:,.2f}") # DEBUG

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
        print(f"DEBUG: AITraderAgent: Current safe value: ${current_safe_value:,.2f}, risky value: ${current_risky_value:,.2f}") # DEBUG

        buy_candidates = sorted(
            [res for res in analysis_results.values() 
             if res.get('final_decision') == 'buy' 
             and res.get('ticker') not in tickers_in_portfolio
             and not res.get('error')],
            key=lambda x: x.get('composite_score', 0), reverse=True
        )
        print(f"DEBUG: AITraderAgent: Found {len(buy_candidates)} buy candidates.") # DEBUG
        
        MIN_INVESTMENT_PER_STOCK = 200
        investment_per_stock = max(MIN_INVESTMENT_PER_STOCK, cash * 0.20)

        for candidate in buy_candidates:
            if cash < MIN_INVESTMENT_PER_STOCK:
                print(f"DEBUG: AITraderAgent: Not enough cash left (${cash:,.2f}) for minimum trade.") # DEBUG
                break

            price = candidate.get('current_price_display')
            if not isinstance(price, (int, float)) or price <= 0:
                print(f"DEBUG: AITraderAgent: Skipping buy candidate {candidate.get('ticker')} due to invalid price: {price}") # DEBUG
                continue

            is_safe_candidate = self._is_safe(candidate)
            
            should_buy = False
            investment_amount = 0
            if is_safe_candidate and current_safe_value < target_safe_value:
                should_buy = True
                investment_amount = min(investment_per_stock, target_safe_value - current_safe_value, cash)
            elif not is_safe_candidate and current_risky_value < target_risky_value:
                should_buy = True
                investment_amount = min(investment_per_stock, target_risky_value - current_risky_value, cash)
            else:
                print(f"DEBUG: AITraderAgent: Skipping {candidate.get('ticker')}. Allocation met or criteria not fit.") # DEBUG
                continue

            if should_buy and investment_amount > price:
                quantity_to_buy = investment_amount / price
                reason = self._generate_trade_reason(candidate['ticker'], 'buy', candidate)
                trades_to_make.append({
                    "ticker": candidate['ticker'], "type": "buy", "quantity": quantity_to_buy,
                    "price": price, "reason": reason
                })
                cash -= investment_amount
                if is_safe_candidate:
                    current_safe_value += investment_amount
                else:
                    current_risky_value += investment_amount
                tickers_in_portfolio.add(candidate['ticker'])
                print(f"DEBUG: AITraderAgent: Decided to BUY {quantity_to_buy:.2f} of {candidate.get('ticker')} at ${price:.2f}.") # DEBUG
            else:
                print(f"DEBUG: AITraderAgent: Cannot buy {candidate.get('ticker')}. Not enough funds allocated or price too high. Inv Amount: {investment_amount:,.2f}, Price: {price:,.2f}") # DEBUG

        print(f"DEBUG: AITraderAgent finished. Total trades to make: {len(trades_to_make)}") # DEBUG
        return trades_to_make

# --- Orchestrator and Backtesting ---
def run_live_analysis(tickers, llm_client, configs):
    results = {}
    progress_bar = st.progress(0, text="Starting analysis...")
    
    st.write(f"DEBUG: Entering run_live_analysis for tickers: {tickers}") # DEBUG

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
        st.write(f"DEBUG: Starting analysis for ticker: {t}") # DEBUG
        
        # --- Using yfinance for price history ---
        price_history_full = fetch_price_history(t, period="max") # Calls yfinance
        st.write(f"DEBUG: Price history for {t} empty: {price_history_full.empty}, rows: {len(price_history_full)}") # DEBUG
        if price_history_full.empty:
            results[t] = {
                "error": f"Price history unavailable for {t}. This can happen for invalid tickers, delisted stocks, or temporary data provider issues.",
                "ticker": t, "final_decision": "error", "composite_score": 0
            }
            st.write(f"DEBUG: Price history empty for {t}. Skipping analysis.") # DEBUG
            continue

        # --- Handle Ticker Info Fetch (still using yfinance) ---
        ticker_info = fetch_ticker_info(t) # This still calls yfinance
        st.write(f"DEBUG: Ticker info for {t} error: {ticker_info.get('_error')}") # DEBUG
        if ticker_info.get("_error"):
            results[t] = {"error": ticker_info["_error"], "ticker": t, "final_decision": "error", "composite_score": 0}
            st.write(f"DEBUG: Ticker info error for {t}. Skipping analysis.") # DEBUG
            continue
        elif not ticker_info or not ticker_info.get("financialCurrency"):
             err_msg = f"Core ticker info (e.g., currency) unavailable for {t}. This likely indicates an invalid ticker, a delisted stock, or persistent issues with yfinance data for this symbol."
             results[t] = {"error": err_msg, "ticker": t, "final_decision": "error", "composite_score": 0}
             st.write(f"DEBUG: Core ticker info missing for {t}. Skipping analysis.") # DEBUG
             continue
        
        current_price_for_ticker = ticker_info.get("currentPrice")
        st.write(f"DEBUG: Current price for {t}: {current_price_for_ticker}") # DEBUG
        if current_price_for_ticker is None and not price_history_full.empty:
            current_price_for_ticker = price_history_full["Close"].iloc[-1]
            st.write(f"DEBUG: Using last close price from history as current price for {t}: {current_price_for_ticker}") # DEBUG
        
        if current_price_for_ticker is None:
            results[t] = {
                "error": f"Current price could not be determined for {t}. Missing essential market data.",
                "ticker": t, "final_decision": "error", "composite_score": 0
            }
            st.write(f"DEBUG: Current price could not be determined for {t}. Skipping analysis.") # DEBUG
            continue

        company_name_for_news = ticker_info.get('longName', ticker_info.get('shortName', t))
        
        # News fetching section
        combined_news, news_fetch_msgs = [], []
        if configs["use_sentiment"]:
            st.write(f"DEBUG: Fetching yfinance news for {t}") # DEBUG
            yf_news = fetch_enriched_news(t, ticker_info)
            if yf_news and not (isinstance(yf_news[0],dict) and "error" in yf_news[0]):
                combined_news.extend(yf_news)
                st.write(f"DEBUG: Found {len(yf_news)} yfinance news for {t}") # DEBUG
            elif yf_news and isinstance(yf_news[0],dict) and "error" in yf_news[0]:
                news_fetch_msgs.append(f"Yahoo News Error: {yf_news[0]['error']}")
                st.write(f"DEBUG: Yfinance news error for {t}: {yf_news[0]['error']}") # DEBUG
            
            if llm_client and st.secrets.get("NEWSAPI_KEY"):
                st.write(f"DEBUG: Fetching NewsAPI news for {t}") # DEBUG
                api_news = fetch_comprehensive_news_from_api(t, company_name_for_news)
                if api_news and not (isinstance(api_news[0],dict) and "error" in api_news[0]):
                    combined_news.extend(api_news)
                    st.write(f"DEBUG: Found {len(api_news)} NewsAPI articles for {t}") # DEBUG
                elif api_news and isinstance(api_news[0],dict) and "error" in api_news[0]:
                    news_fetch_msgs.append(f"NewsAPI Error: {api_news[0]['error']}")
                    st.write(f"DEBUG: NewsAPI error for {t}: {api_news[0]['error']}") # DEBUG
            elif configs["use_sentiment"] and not st.secrets.get("NEWSAPI_KEY"):
                news_fetch_msgs.append("NewsAPI Key missing.")
                st.write("DEBUG: NewsAPI Key missing.") # DEBUG

        seen_urls, dedup_news = set(), []
        for item in combined_news:
            if isinstance(item,dict) and "error" not in item:
                url = item.get('link') or item.get('url')
                if url and url not in seen_urls: dedup_news.append(item); seen_urls.add(url)
        if dedup_news: dedup_news.sort(key=lambda x: x.get('publish_datetime_utc') or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        news_status_bundle = " | ".join(news_fetch_msgs) if news_fetch_msgs else "News fetch OK"
        if not dedup_news and not news_fetch_msgs and configs["use_sentiment"]: news_status_bundle="No news articles found."
        st.write(f"DEBUG: Deduped news for {t}: {len(dedup_news)} articles. Status: {news_status_bundle}") # DEBUG

        data_bundle = {
            "price_history":price_history_full, "ticker_info":ticker_info, "news":dedup_news,
            "news_fetch_status_error": news_status_bundle if any(kw in news_status_bundle.lower() for kw in ["error","failed","no news","missing","issue"]) else None, # Added 'issue'
            "politician_trades":fetch_politician_trades(t) if configs["use_politician_filings"] else [],
            "value_investing_io_data":fetch_value_investing_io_data(t) if configs["use_value_trades"] else {"error":"VI.io: Skipped."},
            "institutional_holdings":fetch_inst_filings(t) if configs["use_filings"] else [],
            "sec_all_filings_raw":fetch_all_sec_filings(t) if configs["use_filings"] else []
        }
        st.write(f"DEBUG: Data bundle created for {t}.") # DEBUG

        agents = [PriceAgent(), MomentumAgent(), VolatilityAgent(), FundamentalsAgent(), ValuationAgent(), AnalystRatingAgent()]
        if configs["use_sentiment"] and llm_client: agents.extend([SentimentAgent(llm_client), NewsSummaryAgent(llm_client)])
        if configs["use_filings"]: agents.extend([SECFilingAgent(), InstitutionalHoldingsAgent()])
        if configs["use_filings"] and llm_client and configs.get("use_sec_summary", False):
             agents.append(SECSummaryAgent(llm_client))
        if configs["use_politician_filings"]: agents.append(PoliticianFilingsAgent())
        if configs["use_value_trades"]: agents.append(ValueInvestingIOAgent())
        
        agent_res_list = []
        for agent in agents:
            name = agent.__class__.__name__
            st.write(f"DEBUG: Running agent: {name} for {t}") # DEBUG
            try:
                if isinstance(agent,(PriceAgent,MomentumAgent)): res_a = agent.run(t, data_bundle["price_history"])
                elif isinstance(agent,VolatilityAgent): res_a = agent.run(t, data_bundle, data_bundle["price_history"])
                elif name == "SECSummaryAgent":
                    res_a = agent.run(t, data_bundle)
                else: res_a = agent.run(t, data_bundle)
                agent_res_list.append(res_a)
                st.write(f"DEBUG: Agent {name} for {t} completed.") # DEBUG
            except Exception as e:
                err_k, sig_k = name.lower().replace("agent","")+"_error", name.lower().replace("agent","")+"_signal"
                if name == "SECSummaryAgent":
                    agent_res_list.append({"sec_summary": "Error generating summary.", "sec_summary_error":f"Agent {name} error: {str(e)[:150]}"})
                else:
                    agent_res_list.append({sig_k:"error", err_k:f"Agent {name} error: {str(e)[:150]}"});
                st.error(f"Error in {name} for {t}: {e}") # This warning should show in Streamlit app
                st.write(f"DEBUG: Agent {name} for {t} FAILED with: {e}") # DEBUG

        final_dec = PortfolioAgent().run(t, agent_res_list)
        st.write(f"DEBUG: PortfolioAgent final decision for {t}: {final_dec['final_decision']}, Score: {final_dec['composite_score']:.2f}") # DEBUG
        
        curr_res_dict = {"ticker":t, "current_price_display":current_price_for_ticker, "market_cap_display":ticker_info.get("marketCap"), "industry_display":ticker_info.get("industry"), "sector_display":ticker_info.get("sector"), "ticker_info":ticker_info,
                             "news_headlines_for_popover":[f"{n.get('publish_time_readable','N/A')} - {n.get('title','N/A')} ({n.get('publisher','N/A')} via {n.get('source_api','Unk')}) [Link]({n.get('link','#')})" + (f" - {n.get('content_snippet',n.get('description',''))[:150]}..." if n.get('content_snippet') or n.get('description') else "") for n in dedup_news[:10]],
                             "politician_trades_for_popover":[pt for pt in data_bundle["politician_trades"][:5] if isinstance(pt,dict) and "error" not in pt],
                             "news_status_display":news_status_bundle}
        
        for r_dict in agent_res_list:
            if isinstance(r_dict,dict): curr_res_dict.update(r_dict)
        curr_res_dict.update(final_dec)
        
        # Ensure backtest start date is within valid range for yfinance data
        backtest_end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        backtest_start_date = (datetime.now() - pd.DateOffset(years=1, days=1)).strftime("%Y-%m-%d")
        
        st.write(f"DEBUG: Running simulated backtest for {t}.") # DEBUG
        sim_bt_metrics, sim_bt_log_df = run_backtest(t, backtest_start_date, backtest_end_date, 10000, llm_client, default_live_backtest_weights)
        curr_res_dict["simulated_backtest_results"] = {"metrics": sim_bt_metrics, "log_df": sim_bt_log_df.to_dict('records') if not sim_bt_log_df.empty else []}
        st.write(f"DEBUG: Simulated backtest for {t} completed.") # DEBUG

        results[t] = curr_res_dict
    progress_bar.empty()
    st.write("DEBUG: All live analysis finished for all tickers.") # DEBUG
    return results

def run_backtest(ticker, start_date, end_date, initial_capital, llm_client_placeholder, backtest_agent_weights):
    st.write(f"DEBUG: Preparing backtest: {ticker} ({start_date} to {end_date})...") # DEBUG
    s_dt = datetime.strptime(start_date, "%Y-%m-%d"); fetch_s_dt = (s_dt - pd.DateOffset(months=18)).strftime("%Y-%m-%d")
    
    full_hist = fetch_price_history(ticker, period="max", interval="1d") # Calls yfinance
    st.write(f"DEBUG: Backtest price history for {ticker} empty: {full_hist.empty}") # DEBUG
    
    if full_hist.empty:
        st.write(f"DEBUG: Backtest fail {ticker}: Price history empty.") # DEBUG
        return {"error": f"Backtest fail {ticker}: Price history empty."}, pd.DataFrame()
    
    hist = full_hist[(full_hist.index >= pd.to_datetime(fetch_s_dt)) & (full_hist.index <= pd.to_datetime(end_date))].copy()
    st.write(f"DEBUG: Backtest filtered history for {ticker} rows: {len(hist)}") # DEBUG
    if hist.empty or len(hist[hist.index >= pd.to_datetime(start_date)]) < 2:
        st.write(f"DEBUG: Backtest fail {ticker}: Not enough data in range.") # DEBUG
        return {"error": f"Backtest fail {ticker}: Not enough data in range."}, pd.DataFrame()
    
    info_bt = fetch_ticker_info(ticker); data_static = {"ticker_info": info_bt} # This still calls yfinance
    p_agent, m_agent, v_agent, port_agent = PriceAgent(), MomentumAgent(), VolatilityAgent(), PortfolioAgent()
    log, cash, shares, port_val = [], initial_capital, 0, initial_capital
    run_dates = hist[hist.index >= pd.to_datetime(start_date)].index
    
    st.write(f"DEBUG: Backtest running simulation for {ticker} over {len(run_dates)} dates.") # DEBUG
    for curr_dt in run_dates:
        data_sl = hist[hist.index <= curr_dt]
        curr_price_pt = data_sl.Close.iloc[-1] if not data_sl.empty else (port_val / shares if shares else 0)
        if data_sl.empty or len(data_sl) < 253:
            log.append({"date":curr_dt, "cash":cash, "shares_held":shares, "price":curr_price_pt, "portfolio_value":port_val, "signal":"hold (no data)", "composite_score":0.0});
            # print(f"DEBUG: Backtest {ticker} on {curr_dt}: Skipping data_sl empty/short") # DEBUG - too verbose for loop
            continue
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
    
    if log_df.empty or len(log_df) < 2:
        st.write(f"DEBUG: Backtest log for {ticker} too short or empty.") # DEBUG
        return {"message":f"Backtest log {ticker} too short."}, pd.DataFrame()
    
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
    
    buy_hold_value = hist["Close"].iloc[-1] / hist["Close"].iloc[0] * initial_capital if not hist.empty and hist["Close"].iloc[0] != 0 else initial_capital
    buy_hold_ret = (buy_hold_value / initial_capital - 1) * 100 if initial_capital != 0 else 0

    metrics = {"Initial Capital":f"${initial_capital:,.2f}", "Final Portfolio Value":f"${log_df.portfolio_value.iloc[-1]:,.2f}", "Total Return (%)":f"{total_ret:.2f}%", "Buy & Hold Return (%)":f"{buy_hold_ret:.2f}%", "Annualized Return (%)":f"{ann_ret:.2f}%", "Annualized Volatility (%)":f"{ann_vol:.2f}%", "Sharpe Ratio":f"{sharpe:.2f}", "Max Drawdown (%)":f"{max_dd:.2f}%", "Number of Trades (approx)":f"{trades}"}
    st.write(f"DEBUG: Backtest metrics for {ticker}: {metrics}") # DEBUG
    return metrics, log_df

# --- Detailed Analysis Display Function ---
def display_detailed_analysis(res_detail):
    print(f"DEBUG: Entering display_detailed_analysis for {res_detail.get('ticker')}") # DEBUG
    ticker = res_detail.get("ticker", "N/A"); ticker_info = res_detail.get("ticker_info", {})
    
    tab_titles = ["📈 Chart & Core", "📊 Fundamentals", "💰 Analyst & Fair Value", "📰 News & Filings", "⚙️ All Signals", "🧪 Simulated Backtest"]
    tabs = st.tabs(tab_titles)

    def get_signal_color(signal):
        signal = str(signal).upper()
        if signal in ["BUY", "STRONG_BUY"]: return "green"
        if signal == "SELL": return "red"
        return "orange"

    with tabs[0]:
        st.subheader("Price Performance & Technical Signals")
        # Using yfinance for charting
        price_hist_chart_data = fetch_price_history(ticker, period="1y") # Calls yfinance
        st.write(f"DEBUG: Chart data for {ticker} empty: {price_hist_chart_data.empty}, rows: {len(price_hist_chart_data)}") # DEBUG
        if not price_hist_chart_data.empty:
            st.line_chart(price_hist_chart_data["Close"], use_container_width=True, color="#0072F0")
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
        st.subheader(f"Fundamental Overview: {ticker_info.get('longName', '')}"); st.caption(f"**Sector:** {ticker_info.get('sector', 'N/A')} | **Industry:** {ticker_info.get('industry', 'N/A')}")
        if ticker_info.get('longBusinessSummary'):
            with st.popover("Show Business Summary"):
                st.markdown(ticker_info.get('longBusinessSummary'))
        st.markdown("---"); fund_col1, fund_col2, fund_col3, fund_col4 = st.columns(4)
        market_cap_val = ticker_info.get('marketCap', 0)
        if isinstance(market_cap_val, (int, float)):
            if market_cap_val > 1e12: cap_str = f"${market_cap_val / 1e12:.2f}T"
            else: cap_str = f"${market_cap_val / 1e9:.2f}B"
        else: cap_str = "N/A"
        fund_col1.metric("Market Cap", cap_str)
        fund_col2.metric("Trailing P/E", f"{ticker_info.get('trailingPE', 0):.2f}" if isinstance(ticker_info.get('trailingPE'),(int,float)) else "N/A")
        fund_col3.metric("Forward P/E", f"{ticker_info.get('forwardPE', 0):.2f}" if isinstance(ticker_info.get('forwardPE'),(int,float)) else "N/A")
        fund_col4.metric("Price/Book", f"{ticker_info.get('priceToBook', 0):.2f}" if isinstance(ticker_info.get('priceToBook'),(int,float)) else "N/A")
        st.markdown("---"); st.subheader("Financial Health"); fund_sig = str(res_detail.get('fund_signal', 'hold')).upper()
        f_col1, f_col2, f_col3 = st.columns(3)
        f_col1.metric("Fundamental Signal", fund_sig); f_col2.metric("Piotroski Score (0-3)", f"{res_detail.get('piotroski_score', 'N/A')}/3")
        fcy_val = res_detail.get('fcf_yield'); f_col3.metric("FCF Yield", f"{fcy_val * 100:.2f}%" if isinstance(fcy_val,(int,float)) else "N/A")
        roe_val = ticker_info.get('returnOnEquity'); de_val = ticker_info.get('debtToEquity'); etr_val = ticker_info.get('enterpriseToRevenue'); ete_val = ticker_info.get('enterpriseToEbitda')
        health_data = {"Return on Equity (ROE)": f"{roe_val * 100:.2f}%" if isinstance(roe_val,(int,float)) else "N/A", "Debt to Equity": f"{de_val:.2f}" if isinstance(de_val,(int,float)) else "N/A", "EV/Revenue": f"{etr_val:.2f}" if isinstance(etr_val,(int,float)) else "N/A", "EV/EBITDA": f"{ete_val:.2f}" if isinstance(ete_val,(int,float)) else "N/A"}
        st.table(pd.DataFrame(health_data.items(), columns=["Metric", "Value"]))

    with tabs[2]:
        val_col1, val_col2 = st.columns(2)
        with val_col1:
            st.subheader("Analyst Consensus"); analyst_signal = str(res_detail.get('analyst_signal', 'hold')).upper()
            st.metric(label=f"Analyst Signal (from {ticker_info.get('numberOfAnalystOpinions')} analysts)", value=analyst_signal)
            abp_val = res_detail.get('analyst_buy_pct_inferred',0.5); st.progress(abp_val, text=f"{abp_val*100:.0f}% Buy Rating")
            tm_val = ticker_info.get('targetMeanPrice'); tu_val = res_detail.get('target_upside')
            st.metric("Mean Target Price", f"${tm_val:.2f}" if isinstance(tm_val,(int,float)) else "N/A", f"{tu_val*100:.2f}% Upside" if isinstance(tu_val,(int,float)) else None)
        with val_col2:
            st.subheader("Peter Lynch Fair Value (via VI.io)"); vi_signal = str(res_detail.get('vi_signal', 'hold')).upper()
            vi_fv = res_detail.get('vi_fair_value_estimate'); up_val = res_detail.get('vi_upside_percent')
            vi_fv_label = f"${vi_fv:,.2f}" if isinstance(vi_fv, (int, float)) else "N/A"
            st.metric(label=f"VI.io Signal (Fair Value: {vi_fv_label})", value=vi_signal, delta=f"{up_val:.2f}% Upside" if isinstance(up_val,(int,float)) else None, delta_color="inverse")
            if res_detail.get('vi_valuation_text_display'): st.markdown(f"> *{res_detail.get('vi_valuation_text_display')}*")

    with tabs[3]:
        st.subheader("News Analysis & Filings")
        if res_detail.get('news_summary'):
            with st.container(border=True):
                st.markdown("**AI-Generated News Summary (from news articles)**"); st.write(res_detail.get('news_summary'))
                headlines = res_detail.get('news_headlines_for_popover', [])
                if headlines:
                    with st.expander("View News Sources & Links"):
                        for line in headlines: st.markdown(f"- {line}")
                if res_detail.get('sentiment_error'): st.warning(f"Sentiment Analysis Note: {res_detail.get('sentiment_error')}")
        
        st.markdown("---")
        if res_detail.get('sec_summary_llm'):
            with st.container(border=True):
                st.markdown("**AI-Generated SEC Filings Summary**"); st.write(res_detail.get('sec_summary_llm'))
                if res_detail.get('sec_summary_error'): st.warning(f"SEC Summary Note: {res_detail.get('sec_summary_error')}")
        else:
            if res_detail.get('sec_summary_error'):
                 st.warning(f"Could not generate SEC Filings Summary: {res_detail.get('sec_summary_error')}")
            else:
                 st.info("No relevant SEC filings found or processed for summary.")

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
                        st.write(f"**{tx.get('transaction_date')}**: {tx.get('reporting_owner')} ({tx.get('owner_relationship')}) {tx_type_display} {tx.get('shares'):,.0f} shares{price_display} - [Link]({tx.get('link_to_filing')})")
                else:
                    st.info("No recent Form 4 insider transactions found.")
            with st.expander("View Other Recent SEC Filings (Metadata only)"):
                other_filings_meta = [f for f in res_detail.get('sec_other_recent_filings', []) if f.get("form_type") not in ['10-K', '10-Q', '8-K']]
                if other_filings_meta:
                    for f in other_filings_meta: st.write(f"**{f.get('filing_date')}**: Form {f.get('form_type')} - [Link]({f.get('summary_link')})")
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
                holders_df['Shares'] = pd.to_numeric(holders_df['Shares'], errors='coerce')
                holders_df = holders_df.dropna(subset=['Shares'])
                holders_df = holders_df[holders_df['Shares'] > 0]

                if not holders_df.empty and 'Holder' in holders_df.columns and 'Shares' in holders_df.columns:
                    top_n_holders = holders_df.nlargest(5, 'Shares') if len(holders_df) > 5 else holders_df
                    
                    if not top_n_holders.empty:
                        chart = alt.Chart(top_n_holders).mark_arc(outerRadius=120).encode(
                            theta=alt.Theta(field="Shares", type="quantitative"),
                            color=alt.Color(field="Holder", type="nominal", title="Top Holders"),
                            order=alt.Order("Shares", sort="descending"),
                            tooltip=["Holder", "Shares", alt.Tooltip("% Out", format=".1%")]
                        ).properties(
                            title="Top Institutional Holders (by Shares)"
                        )
                        try:
                            st.altair_chart(chart, use_container_width=True)
                        except Exception as chart_err:
                            st.error(f"Error rendering institutional holdings chart for {ticker}: {chart_err}")
                            st.json(top_n_holders.to_dict('records'))
                    else:
                        st.info(f"Not enough positive share data for institutional holdings chart for {ticker}.")
                else:
                    st.info(f"Not enough valid data points for institutional holdings chart for {ticker}.")
            else:
                st.info("No institutional holder data available.")

        st.markdown("---")
        st.subheader("Politician Trading Activity")
        politician_signal = str(res_detail.get('politician_filings_signal', 'hold')).upper()
        st.metric("Politician Trading Signal", politician_signal)
        st.markdown(f"""
            <div style="font-size: 14px;">
                <li><b>Net Estimated Value:</b> ${res_detail.get('politician_net_trade_value_estimate', 0):,.2f}</li>
                <li><b>Buy Transactions:</b> {res_detail.get('politician_buy_tx_count', 0)}</li>
                <li><b>Sell Transactions:</b> {res_detail.get('politician_sell_tx_count', 0)}</li>
            </div>
        """, unsafe_allow_html=True)
        if res_detail.get('politician_data_error'):
            st.warning(f"Politician Data Note: {res_detail.get('politician_data_error')}")
        else:
            with st.expander("View Recent Politician Trades"):
                poli_trades = res_detail.get('politician_trades_for_popover', [])
                if poli_trades:
                    for trade in poli_trades:
                        st.write(f"**{trade.get('date_str')}**: {trade.get('politician_name')} {trade.get('transaction_type').capitalize()} {trade.get('value_range')} - [Link]({trade.get('source_url')})")
                else:
                    st.info("No recent politician trades found.")

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
            "Politician Filings Signal": str(res_detail.get("politician_filings_signal","N/A")).upper()
        }
        df_signals = pd.DataFrame(signals_data.items(), columns=["Agent", "Signal"])
        st.dataframe(df_signals.style.applymap(lambda x: f'color: {get_signal_color(x)}', subset=['Signal']), hide_index=True, use_container_width=True)
        st.markdown("---")
        final_decision = str(res_detail.get('final_decision', 'hold')).upper(); final_color = get_signal_color(final_decision)
        st.markdown(f"""<div style="border:2px solid {final_color}; border-radius:8px; padding:15px; text-align:center;"><p style="font-size:1.2em; margin-bottom:5px;">Final AI Decision</p><h2 style="color:{final_color}; margin-bottom:5px;">{final_decision}</h2><p style="font-size:1em;">Composite Score: <strong>{res_detail.get('composite_score', 0):.2f}</strong></p></div>""", unsafe_allow_html=True)
    
    with tabs[5]: # New tab for Simulated Backtest
        st.subheader(f"Simulated Backtest for {res_detail.get('ticker')}")
        sim_bt_data = res_detail.get("simulated_backtest_results", {})
        sim_bt_metrics = sim_bt_data.get("metrics")
        sim_bt_log_df_raw = sim_bt_data.get("log_df")

        if sim_bt_metrics and not (sim_bt_metrics.get("message") or sim_bt_metrics.get("error")):
            st.markdown("This section shows a quick simulated backtest for the last year using the **Price, Momentum, and Volatility** agents with standard weights. This is a simplified simulation for quick insight, not a full backtest.")
            metrics_df_sim_bt = pd.DataFrame.from_dict(sim_bt_metrics, orient='index', columns=['Value'])
            st.table(metrics_df_sim_bt)
            
            if sim_bt_log_df_raw:
                sim_bt_log_df = pd.DataFrame(sim_bt_log_df_raw)
                if not sim_bt_log_df.empty and 'date' in sim_bt_log_df.columns:
                    sim_bt_log_df['date'] = pd.to_datetime(sim_bt_log_df['date'])
                    sim_bt_log_df.set_index("date", inplace=True)
                    st.subheader("Portfolio Value Over Time (Simulated)"); st.line_chart(sim_bt_log_df["portfolio_value"])
                    st.subheader("Drawdown Over Time (Simulated)"); st.area_chart(sim_bt_log_df["drawdown"].fillna(0))
                else:
                    st.warning("Simulated backtest log data is not in the expected format or is empty.")
            else:
                st.info("No log data available for simulated backtest.")
        elif sim_bt_metrics:
            st.error(f"Simulated Backtest Error: {sim_bt_metrics.get('error','Unknown error')}")
        else:
            st.info("Simulated backtest results not available for this ticker.")


# --- Streamlit UI ---
llm_client = None
try:
    ds_key = st.secrets.get("DEEPSEEK_API_KEY")
    oa_key = st.secrets.get("OPENAI_API_KEY") # Prioritize secrets for security
    
    if not ds_key: ds_key = os.environ.get("DEEPSEEK_API_KEY")
    if not oa_key: oa_key = os.environ.get("OPENAI_API_KEY")

    if ds_key:
        llm_client = ModelClient(api_key=ds_key, provider="deepseek")
        st.sidebar.caption("✅ LLM: DeepSeek")
        st.write("DEBUG: DeepSeek LLM client initialized.") # DEBUG
    elif oa_key:
        llm_client = ModelClient(api_key=oa_key, provider="openai")
        st.sidebar.caption("✅ LLM: OpenAI")
        st.write("DEBUG: OpenAI LLM client initialized.") # DEBUG
    else:
        st.sidebar.warning("LLM API key missing. Sentiment/Summary disabled.")
        st.write("DEBUG: LLM API key missing. Sentiment/Summary disabled.") # DEBUG
except ValueError as e:
    st.sidebar.error(f"LLM Init Error: {e}")
    st.write(f"DEBUG: LLM Init ValueError: {e}") # DEBUG
    llm_client = None
except Exception as e:
    st.sidebar.error(f"LLM Unexpected Init Error: {e}")
    st.write(f"DEBUG: LLM Unexpected Init Error: {e}") # DEBUG
    llm_client = None

st.title("🚀 AI Hedge Fund Simulator")
st.header("⚙️ Configuration"); config_cont = st.container(border=True)

app_mode_options = ["Live Analysis", "Backtesting", "💼 Portfolio Management", "🤖 Virtual Trading"]
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = app_mode_options[0]
    st.write(f"DEBUG: Initial app_mode: {st.session_state.app_mode}") # DEBUG

with config_cont:
    current_mode_index = app_mode_options.index(st.session_state.app_mode)
    selected_mode = st.radio("Select Mode:", app_mode_options, key="app_mode_sel_main_key", horizontal=True, index=current_mode_index)
    if selected_mode != st.session_state.app_mode:
        st.session_state.app_mode = selected_mode
        st.session_state.live_analysis_triggered = False
        st.session_state.backtest_triggered = False
        st.write(f"DEBUG: Mode changed to {selected_mode}. Rerunning.") # DEBUG
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
        with feat_cols[1]:
            use_poli_live = st.checkbox("Politician Filings (Exp.)", value=False, key="live_poli_cb_main", help="Scrapes CapitolTrades. May be slow/unreliable.")
            use_valtrades_live = st.checkbox("ValueInvesting.io (Exp.)", value=False, key="live_vt_cb_main", help="Scrapes ValueInvesting.io. May be slow/unreliable.")
            use_sec_summary_live = st.checkbox("SEC Filings Summary (LLM)", value=bool(llm_client) and use_filings_live, disabled=not (llm_client and use_filings_live), key="live_sec_summary_cb", help="Uses LLM to summarize recent 10-K, 10-Q, 8-K filings. Requires LLM and 'SEC & Inst. Filings' to be enabled.")

        if st.button("🚀 Run Live Analysis", use_container_width=True, type="primary", key="run_live_analysis_button"):
            live_tickers = [t.strip().upper() for t in tickers_in_live.split(",") if t.strip()]
            if not live_tickers:
                st.error("Please enter at least one ticker.")
                st.write("DEBUG: No tickers entered.") # DEBUG
            else:
                st.write(f"DEBUG: Triggering live analysis for tickers: {live_tickers}") # DEBUG
                live_configs = {
                    "use_sentiment":use_sent_live,
                    "use_filings":use_filings_live,
                    "use_politician_filings":use_poli_live,
                    "use_value_trades":use_valtrades_live,
                    "use_sec_summary":use_sec_summary_live
                }
                with st.spinner("⏳ Processing live analysis..."):
                    st.session_state.live_output = run_live_analysis(live_tickers, llm_client, live_configs)
                    st.session_state.live_analysis_triggered = True
                    st.write("DEBUG: Live analysis complete. Triggering rerun to display results.") # DEBUG
                    st.rerun()

    # Display Live Analysis Results after rerun
    if st.session_state.app_mode == "Live Analysis" and st.session_state.live_analysis_triggered:
        st.subheader("Live Analysis Results")
        st.write(f"DEBUG: Attempting to display {len(st.session_state.live_output)} live analysis results.") # DEBUG
        
        summary_data = []
        for ticker, res in st.session_state.live_output.items():
            if res.get("error"):
                summary_data.append({"Ticker": ticker, "AI Decision": "ERROR", "Composite Score": "N/A", "Market Cap": "N/A", "Error Message": res["error"]})
            else:
                market_cap_val = res.get('market_cap_display', 0)
                if isinstance(market_cap_val, (int, float)):
                    if market_cap_val > 1e12: cap_str = f"${market_cap_val / 1e12:.2f}T"
                    elif market_cap_val > 1e9: cap_str = f"${market_cap_val / 1e9:.2f}B"
                    elif market_cap_val > 1e6: cap_str = f"${market_cap_val / 1e6:.2f}M"
                    else: cap_str = f"${market_cap_val:,.2f}"
                else: cap_str = "N/A"

                summary_data.append({
                    "Ticker": ticker,
                    "AI Decision": res.get('final_decision', 'N/A').upper(),
                    "Composite Score": f"{res.get('composite_score', 0):.2f}",
                    "Market Cap": cap_str,
                    "Industry": res.get('industry_display', 'N/A'),
                    "News Status": res.get('news_status_display', 'N/A')
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df.style.applymap(lambda x: f'color: {get_signal_color(x)}', subset=['AI Decision']), use_container_width=True, hide_index=True)
            st.markdown("---")
            
            selected_ticker_for_detail = st.selectbox("Select Ticker for Detailed Analysis:", [""] + list(st.session_state.live_output.keys()))
            if selected_ticker_for_detail:
                st.write(f"DEBUG: Displaying detailed analysis for {selected_ticker_for_detail}.") # DEBUG
                display_detailed_analysis(st.session_state.live_output[selected_ticker_for_detail])
            else:
                st.write("DEBUG: No ticker selected for detailed analysis yet.") # DEBUG
        else:
            st.info("No analysis results to display.")
            st.write("DEBUG: No summary data to display (all tickers might have errors).") # DEBUG

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
                st.info("Using a fixed $10,000 for backtesting from saved portfolio. Live value fetching is not implemented here for performance.")
                bt_capital = 10000 

        bt_c1, bt_c2 = st.columns(2)
        with bt_c1:
            def_end_dt = datetime.now(timezone.utc)-timedelta(days=1); def_start_dt = def_end_dt-pd.DateOffset(years=3)
            start_dt_in = st.date_input("Start Date:", def_start_dt, max_value=def_end_dt-timedelta(days=30), key="bt_start_dt_bt")
            st.session_state.bt_start_str = start_dt_in.strftime("%Y-%m-%d")
        with bt_c2:
            min_end_dt_bt = start_dt_in+timedelta(days=30)
            end_dt_in = st.date_input("End Date:", def_end_dt, min_value=min_end_dt_bt, max_value=datetime.now(timezone.utc)-timedelta(days=1), key="bt_end_dt_bt")
            st.session_state.bt_end_str = end_dt_in.strftime("%Y-%m-%d")
            
        with st.expander("Adjust Backtest Agent Weights",expanded=False):
            w_p, w_m, w_v = st.slider("Price W:",0.,2.,1.,.1,key="bt_w_p_bt"), st.slider("Mom W:",0.,2.,.8,.1,key="bt_w_m_bt"), st.slider("Vol W:",0.,2.,.2,.1,key="bt_w_v_bt")
            st.info("Only Price, Momentum, and Volatility agents are used in backtesting for speed and simpler demonstration. Other signals are disabled.")
            st.session_state.bt_weights = {
                "price":w_p, "momentum":w_m, "volatility":w_v,
                "sentiment":0.,"fund":0.,"valuation_dcf":0.,"valuation_pe":0.,
                "sec_filings":0.,"sec_summary":0., 
                "inst_holdings":0.,"analyst":0.,"politician_filings":0.,"vi_signal":0.
            }
            st.session_state.bt_capital = bt_capital

        if st.button("📈 Run Backtest",use_container_width=True,type="primary",key="run_bt_btn_main"):
            if st.session_state.bt_ticker:
                st.write(f"DEBUG: Triggering backtest for {st.session_state.bt_ticker}") # DEBUG
                with st.spinner(f"⏳ Running backtest for {st.session_state.bt_ticker}..."):
                    metrics, log_df = run_backtest(st.session_state.bt_ticker, st.session_state.bt_start_str, st.session_state.bt_end_str, st.session_state.bt_capital, llm_client, st.session_state.bt_weights)
                    st.session_state.backtest_results[st.session_state.bt_ticker] = {"metrics": metrics, "log_df": log_df.to_dict('records') if not log_df.empty else []} # Convert DataFrame to list of dicts for session state
                    st.session_state.backtest_triggered = True
                    st.write("DEBUG: Backtest complete. Triggering rerun to display results.") # DEBUG
                    st.rerun()
        
        # Display Backtest Results after rerun
        if st.session_state.app_mode == "Backtesting" and st.session_state.backtest_triggered:
            st.subheader(f"Backtest Results for {st.session_state.bt_ticker}")
            st.write(f"DEBUG: Attempting to display backtest results for {st.session_state.bt_ticker}.") # DEBUG
            
            bt_res = st.session_state.backtest_results.get(st.session_state.bt_ticker, {})
            bt_metrics = bt_res.get("metrics")
            bt_log_df_raw = bt_res.get("log_df") # This is now list of dicts

            if bt_metrics and not (bt_metrics.get("message") or bt_metrics.get("error")):
                st.markdown("This shows a simulated backtest.")
                metrics_df = pd.DataFrame.from_dict(bt_metrics, orient='index', columns=['Value'])
                st.table(metrics_df)
                
                if bt_log_df_raw:
                    bt_log_df = pd.DataFrame(bt_log_df_raw)
                    if not bt_log_df.empty and 'date' in bt_log_df.columns:
                        bt_log_df['date'] = pd.to_datetime(bt_log_df['date'])
                        bt_log_df.set_index("date", inplace=True)
                        st.subheader("Portfolio Value Over Time"); st.line_chart(bt_log_df["portfolio_value"])
                        st.subheader("Drawdown Over Time"); st.area_chart(bt_log_df["drawdown"].fillna(0))
                    else:
                        st.warning("Backtest log data is not in the expected format or is empty for charting.")
                        st.write("DEBUG: Backtest log df empty or wrong format for chart.") # DEBUG
                else:
                    st.info("No log data available for backtest chart.")
                    st.write("DEBUG: No backtest log data raw.") # DEBUG
            elif bt_metrics:
                st.error(f"Backtest Error: {bt_metrics.get('error','Unknown error')}")
                st.write(f"DEBUG: Backtest error reported: {bt_metrics.get('error')}") # DEBUG
            else:
                st.info("No backtest results available. Run a backtest above.")
                st.write("DEBUG: No backtest metrics to display.") # DEBUG

    elif st.session_state.app_mode == "💼 Portfolio Management":
        st.subheader("💼 Portfolio Management")
        portfolio_names_list = list(st.session_state.portfolios_data.keys())

        if not portfolio_names_list:
            st.session_state.portfolios_data["My First Portfolio"] = {"holdings": []} 
            st.session_state.selected_portfolio_name = "My First Portfolio"
            save_portfolios(st.session_state.portfolios_data)
            st.rerun() # Rerun once a default portfolio is created

        col_pf1, col_pf2, col_pf3 = st.columns([3, 1, 1])

        st.session_state.selected_portfolio_name = col_pf1.selectbox(
            "Select Portfolio:",
            portfolio_names_list,
            index=portfolio_names_list.index(st.session_state.selected_portfolio_name) if st.session_state.selected_portfolio_name in portfolio_names_list else 0,
            key="portfolio_selector"
        )
        current_portfolio = st.session_state.portfolios_data.get(st.session_state.selected_portfolio_name, {"holdings": []})

        new_portfolio_name = col_pf2.text_input("New Portfolio Name:", "", key="new_pf_name")
        if col_pf3.button("➕ Create Portfolio", key="create_pf_btn"):
            if new_portfolio_name and new_portfolio_name not in st.session_state.portfolios_data:
                st.session_state.portfolios_data[new_portfolio_name] = {"holdings": []}
                save_portfolios(st.session_state.portfolios_data)
                st.session_state.selected_portfolio_name = new_portfolio_name
                st.success(f"Portfolio '{new_portfolio_name}' created!")
                st.rerun()
            else:
                st.error("Portfolio name is empty or already exists.")

        st.markdown("---")
        st.subheader(f"Holdings & Analysis for '{st.session_state.selected_portfolio_name}'")

        holdings_display_data = []
        tickers_to_analyze = [h['ticker'] for h in current_portfolio['holdings']]
        
        # Only run analysis if tickers exist and haven't been analyzed for this portfolio
        if tickers_to_analyze:
            st.write(f"DEBUG: Portfolio Management: {len(tickers_to_analyze)} tickers in portfolio to analyze.") # DEBUG
            portfolio_analysis_configs = {
                "use_sentiment": True, "use_filings": True, "use_politician_filings": True,
                "use_value_trades": True, "use_sec_summary": True 
            }
            # Only re-run analysis if the portfolio name changes or results are not cached
            if 'portfolio_analysis_results' not in st.session_state or \
               st.session_state.portfolio_analysis_results.get('portfolio_name') != st.session_state.selected_portfolio_name:
                with st.spinner(f"⏳ Running AI analysis for '{st.session_state.selected_portfolio_name}' holdings..."):
                    st.session_state.portfolio_analysis_results = {
                        'portfolio_name': st.session_state.selected_portfolio_name,
                        'analysis': run_live_analysis(tickers_to_analyze, llm_client, portfolio_analysis_configs)
                    }
                st.write("DEBUG: Portfolio analysis results updated.") # DEBUG
            
            analysis_results = st.session_state.portfolio_analysis_results['analysis']
            st.markdown("---")

            total_market_value = 0.0
            total_unrealized_pnl = 0.0
            total_invested_cost = 0.001 

            for holding in current_portfolio['holdings']:
                ticker = holding['ticker']
                quantity = holding['quantity']
                avg_price = holding['avg_price']

                analysis_res = analysis_results.get(ticker, {})

                current_price = analysis_res.get('current_price_display')
                final_decision = analysis_res.get('final_decision', 'N/A').upper()
                composite_score = analysis_res.get('composite_score', 0.0)

                if isinstance(current_price, (int, float)):
                    market_value = current_price * quantity
                    unrealized_pnl = (current_price - avg_price) * quantity
                    total_market_value += market_value
                    total_unrealized_pnl += unrealized_pnl
                    total_invested_cost += avg_price * quantity
                    
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
                def color_ai_decision(val):
                    color = get_signal_color(val)
                    return f'color: {color}; font-weight: bold;'

                st.dataframe(holdings_df.style.applymap(color_ai_decision, subset=['AI Decision']), use_container_width=True, hide_index=True,
                                 column_config={
                                     "Quantity": st.column_config.NumberColumn(format="%.4f"),
                                     "Avg. Cost": st.column_config.NumberColumn(format="$%.2f"),
                                     "Current Price": st.column_config.NumberColumn(format="$%.2f"),
                                     "Market Value": st.column_config.NumberColumn(format="$%.2f"),
                                     "Unrealized P&L": st.column_config.NumberColumn(format="$%.2f", help="Unrealized Profit & Loss"),
                                     "P&L (%)": st.column_config.ProgressColumn(format="%.2f%%", min_value=-100, max_value=100),
                                     "Composite Score": st.column_config.NumberColumn(format="%.2f", help="Aggregated AI score (-1.0 to 1.0)")
                                 })
                st.markdown("---")
                selected_ticker_for_detail = st.selectbox("Select Holding for Detailed Analysis:", [""] + [h['Ticker'] for h in holdings_display_data])
                if selected_ticker_for_detail:
                    detail_res = analysis_results.get(selected_ticker_for_detail)
                    if detail_res:
                        st.write(f"DEBUG: Displaying detailed analysis for {selected_ticker_for_detail} from Portfolio Management tab.") # DEBUG
                        display_detailed_analysis(detail_res)
                    else:
                        st.warning(f"Analysis results not found for {selected_ticker_for_detail}. Please re-run analysis if recent changes were made.")
                        st.write(f"DEBUG: Analysis results not found for {selected_ticker_for_detail} in portfolio mode.") # DEBUG
            else:
                st.info("This portfolio currently has no stock holdings. Use the 'Add Stock' section below.")
                st.write("DEBUG: Portfolio holdings_display_data is empty.") # DEBUG

        else:
            st.info("This portfolio currently has no stock holdings. Add stocks to analyze them.")
            st.write("DEBUG: Tickers to analyze is empty in portfolio management.") # DEBUG
        
        st.markdown("---")
        st.subheader("Add/Remove Stocks")
        col_add1, col_add2, col_add3 = st.columns(3)
        add_ticker = col_add1.text_input("Ticker to Add:", "", key="add_ticker_input_pf").upper()
        add_quantity = col_add2.number_input("Quantity:", min_value=0.01, value=1.0, step=0.1, key="add_quantity_input_pf")
        add_price = col_add3.number_input("Purchase Price (required):", min_value=0.01, value=0.01, step=0.01, key="add_price_input_pf")

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
                    st.success(f"Updated {add_ticker} in '{st.session_state.selected_portfolio_name}'. New quantity: {new_total_quantity:.2f}, Avg. Price: ${new_avg_price:.2f}.") # DEBUG
                else:
                    current_portfolio['holdings'].append({"ticker": add_ticker, "quantity": add_quantity, "avg_price": add_price})
                    st.success(f"Added {add_quantity:.2f} shares of {add_ticker} at ${add_price:.2f} to '{st.session_state.selected_portfolio_name}'.") # DEBUG
                
                save_portfolios(st.session_state.portfolios_data)
                st.session_state.portfolio_analysis_results = None # Invalidate cached analysis results for this portfolio
                st.rerun()
            else:
                st.error("Please enter a valid ticker, quantity, and purchase price.")
                st.write("DEBUG: Invalid input for Add Stock.") # DEBUG

        col_rem1, col_rem2 = st.columns([1,2])
        tickers_in_current_portfolio = [h['ticker'] for h in current_portfolio['holdings']]
        remove_ticker_selection = col_rem1.selectbox("Select Ticker to Remove:", [""] + tickers_in_current_portfolio, key="remove_ticker_select_pf")

        if col_rem2.button("➖ Remove Stock from Portfolio", key="remove_stock_btn_pf"):
            if remove_ticker_selection:
                initial_holdings_count = len(current_portfolio['holdings'])
                current_portfolio['holdings'] = [h for h in current_portfolio['holdings'] if h['ticker'] != remove_ticker_selection]
                
                if len(current_portfolio['holdings']) < initial_holdings_count:
                    save_portfolios(st.session_state.portfolios_data)
                    st.success(f"Removed {remove_ticker_selection} from '{st.session_state.selected_portfolio_name}'.") # DEBUG
                    st.session_state.portfolio_analysis_results = None # Invalidate cached analysis results
                    st.rerun()
                else:
                    st.error(f"{remove_ticker_selection} not found in '{st.session_state.selected_portfolio_name}' holdings.")
                    st.write(f"DEBUG: Ticker {remove_ticker_selection} not found for removal.") # DEBUG
            else:
                st.error("Please select a ticker to remove.")
                st.write("DEBUG: No ticker selected for removal.") # DEBUG
        
        st.markdown("---")
        if st.button("🗑️ Delete Current Portfolio", key="delete_portfolio_btn_pf", type="secondary"):
            if st.session_state.selected_portfolio_name and st.session_state.selected_portfolio_name in st.session_state.portfolios_data:
                st.write(f"DEBUG: Deleting portfolio {st.session_state.selected_portfolio_name}.") # DEBUG
                del st.session_state.portfolios_data[st.session_state.selected_portfolio_name]
                save_portfolios(st.session_state.portfolios_data)
                st.session_state.selected_portfolio_name = None
                st.session_state.portfolio_analysis_results = None # Clear analysis for deleted portfolio
                st.success(f"Portfolio '{st.session_state.selected_portfolio_name}' deleted.")
                st.rerun()
            else:
                st.error("No portfolio selected or found to delete.")
                st.write("DEBUG: No portfolio selected or found for deletion.") # DEBUG


    elif st.session_state.app_mode == "🤖 Virtual Trading":
        st.header("📈 Virtual Portfolio Dashboard")
        with st.container(border=True):
            holdings_df_data = []
            total_holdings_value, total_pnl, initial_investment = 0.0, 0.0, 0.001
            
            portfolio_value_history = []
            
            temp_cash = get_default_virtual_portfolio()["cash"]
            
            chronological_transactions = list(reversed(st.session_state.virtual_portfolio['transaction_history']))
            
            all_tickers_in_history = list(set([t['ticker'] for t in st.session_state.virtual_portfolio['transaction_history']]))
            price_data_for_history = {}
            with st.spinner("Pre-fetching historical prices for portfolio value chart..."):
                st.write("DEBUG: Virtual Trading: Pre-fetching historical prices for chart.") # DEBUG
                for t in all_tickers_in_history:
                    price_data_for_history[t] = fetch_price_history(t, period="max") # Calls yfinance
                    st.write(f"DEBUG: Virtual Trading: Fetched {len(price_data_for_history[t])} historical rows for {t}.") # DEBUG

            
            if chronological_transactions:
                st.write(f"DEBUG: Virtual Trading: Processing {len(chronological_transactions)} transactions for chart history.") # DEBUG
                earliest_tx_date = datetime.strptime(chronological_transactions[0]['date'].split(" ")[0], "%Y-%m-%d") if chronological_transactions else datetime.now()
                earliest_data_point = earliest_tx_date - timedelta(days=365*2) # Look back 2 years before first transaction
                
                full_date_range = pd.date_range(start=earliest_data_point, end=datetime.now(timezone.utc), freq='D')
                
                daily_portfolio_values = pd.Series(index=full_date_range, dtype=float).fillna(np.nan)
                daily_portfolio_values[full_date_range[0]] = get_default_virtual_portfolio()["cash"]

                current_holdings_for_chart = {}
                current_cash_for_chart = get_default_virtual_portfolio()["cash"]
                last_date_processed = full_date_range[0]

                for tx in chronological_transactions:
                    tx_date = datetime.strptime(tx['date'].split(" ")[0], "%Y-%m-%d")
                    tx_quantity = float(tx['quantity'])
                    tx_price = float(tx['price'].replace('$', ''))
                    tx_type = tx['type']

                    # Fill in portfolio value for days between last processed date and current transaction date
                    for day in pd.date_range(start=last_date_processed + timedelta(days=1), end=tx_date, freq='D'):
                        portfolio_val_on_day = current_cash_for_chart
                        for ticker_chart, qty_chart in current_holdings_for_chart.items():
                            day_price_series = price_data_for_history.get(ticker_chart, pd.DataFrame())
                            day_price = None
                            if not day_price_series.empty:
                                try:
                                    day_price = day_price_series.loc[day.strftime("%Y-%m-%d")].get("Close")
                                except KeyError:
                                    # Fallback to nearest day if exact date not found
                                    nearest_idx = day_price_series.index.asof(day)
                                    if nearest_idx is not pd.NaT:
                                        day_price = day_price_series.loc[nearest_idx].get("Close")

                            if day_price:
                                portfolio_val_on_day += qty_chart * day_price
                        if pd.notna(portfolio_val_on_day):
                            daily_portfolio_values.loc[day] = portfolio_val_on_day
                    
                    last_date_processed = tx_date

                    if tx_type == 'BUY':
                        current_holdings_for_chart[tx['ticker']] = current_holdings_for_chart.get(tx['ticker'], 0) + tx_quantity
                        current_cash_for_chart -= tx_quantity * tx_price
                    elif tx_type == 'SELL':
                        current_holdings_for_chart[tx['ticker']] = current_holdings_for_chart.get(tx['ticker'], 0) - tx_quantity
                        if current_holdings_for_chart[tx['ticker']] <= 0.0001: # Handle floating point near zero
                            del current_holdings_for_chart[tx['ticker']]
                        current_cash_for_chart += tx_quantity * tx_price
                    
                    # Update portfolio value on the transaction day itself
                    portfolio_val_on_tx_day = current_cash_for_chart
                    for ticker_chart, qty_chart in current_holdings_for_chart.items():
                        tx_day_price_series = price_data_for_history.get(ticker_chart, pd.DataFrame())
                        tx_day_price = None
                        if not tx_day_price_series.empty:
                            try:
                                tx_day_price = tx_day_price_series.loc[tx_date.strftime("%Y-%m-%d")].get("Close")
                            except KeyError:
                                nearest_idx = tx_day_price_series.index.asof(tx_date)
                                if nearest_idx is not pd.NaT:
                                    tx_day_price = tx_day_price_series.loc[nearest_idx].get("Close")
                        if tx_day_price:
                            portfolio_val_on_tx_day += qty_chart * tx_day_price
                    if pd.notna(portfolio_val_on_tx_day):
                        daily_portfolio_values.loc[tx_date] = portfolio_val_on_tx_day
                
                # Fill any remaining days up to today
                daily_portfolio_values = daily_portfolio_values.fillna(method='ffill').fillna(method='bfill')

                today_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
                if today_utc not in daily_portfolio_values.index:
                    last_known_value = daily_portfolio_values.iloc[-1] if not daily_portfolio_values.empty else get_default_virtual_portfolio()["cash"]
                    daily_portfolio_values.loc[today_utc] = last_known_value
                
                daily_portfolio_values = daily_portfolio_values.reindex(pd.date_range(start=daily_portfolio_values.index.min(), end=today_utc, freq='D')).fillna(method='ffill').fillna(method='bfill')
                daily_portfolio_values_df = pd.DataFrame(daily_portfolio_values, columns=['Portfolio Value'])
                daily_portfolio_values_df.index.name = 'Date'
                st.write("DEBUG: Virtual Trading: Portfolio value history calculated.") # DEBUG

            else:
                st.write("DEBUG: Virtual Trading: No transactions to build history from. Using default range.") # DEBUG
                daily_portfolio_values_df = pd.DataFrame({
                    'Date': [datetime.now(timezone.utc) - timedelta(days=7), datetime.now(timezone.utc)],
                    'Portfolio Value': [get_default_virtual_portfolio()["cash"], get_default_virtual_portfolio()["cash"]]
                }).set_index('Date')


            if st.session_state.virtual_portfolio['holdings']:
                st.write("DEBUG: Virtual Trading: Fetching latest prices for current holdings display.") # DEBUG
                with st.spinner("Fetching latest prices for dashboard..."):
                    for holding in st.session_state.virtual_portfolio['holdings']:
                        info = fetch_ticker_info(holding['ticker']) # This calls yfinance for info
                        price = info.get("currentPrice")
                        if price is None and not price_data_for_history.get(holding['ticker'], pd.DataFrame()).empty:
                            price = price_data_for_history[holding['ticker']].iloc[-1].get("Close")
                        
                        if price is None:
                            st.warning(f"Could not get live price for {holding['ticker']} in virtual portfolio.") # DEBUG
                            print(f"DEBUG: Virtual Trading: No live price for {holding['ticker']}") # DEBUG
                            current_value = holding['avg_price'] * holding['quantity'] # Use avg cost as fallback
                            pnl = 0 # No P&L if no live price
                        else:
                            current_value = price * holding['quantity']
                            pnl = (price - holding['avg_price']) * holding['quantity']
                        
                        total_holdings_value += current_value
                        total_pnl += pnl
                        initial_investment += holding['avg_price'] * holding['quantity']
                        holdings_df_data.append({"Ticker": holding['ticker'], "Quantity": holding['quantity'], "Avg. Price": holding['avg_price'], "Current Price": price, "Current Value": current_value, "P&L": pnl})
            st.write("DEBUG: Virtual Trading: Holdings data prepared for display.") # DEBUG
            
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
                st.dataframe(holdings_df, use_container_width=True, column_config={ "Avg. Price": st.column_config.NumberColumn(format="$%.2f"), "Current Price": st.column_config.NumberColumn(format="$%.2f"), "Current Value": st.column_config.NumberColumn(format="$%.2f"), "P&L": st.column_config.NumberColumn(format="$%.2f"), "Quantity": st.column_config.NumberColumn(format="%.4f") })
            else:
                st.info("The portfolio currently holds no stocks. Run the AI Trader to start investing.")

            st.subheader("Portfolio Value Over Time")
            st.line_chart(daily_portfolio_values_df, use_container_width=True, color="#0072F0")

            st.subheader("Transaction History")
            if st.session_state.virtual_portfolio['transaction_history']:
                history_df = pd.DataFrame(st.session_state.virtual_portfolio['transaction_history'])
                st.dataframe(history_df, use_container_width=True, hide_index=True)
            else:
                st.info("No transactions have been made yet.")

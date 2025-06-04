import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import openai
from openai import OpenAI # Ensure this is the correct import for your openai version
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
from newsapi import NewsApiClient
import json

# --- Page Config (Must be the first Streamlit command) ---
st.set_page_config(page_title="AI Hedge Fund Simulator", layout="wide")

# Load environment variables (if running locally)
load_dotenv()

# SEC EDGAR User-Agent
SEC_USER_AGENT = "KevalAhirApp/1.0 keval.ahir2019@gmail.com"

# --------------------------------
# Data Fetchers
# --------------------------------
@st.cache_data
def fetch_price_history(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Fetches historical price data for a given ticker."""
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        # Ensure timezone-naive for consistency
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception as e:
        # st.error(f"Error fetching price history for {ticker}: {e}") # Error displayed by orchestrator
        return pd.DataFrame()

@st.cache_data
def fetch_ticker_info(ticker: str) -> dict:
    """Fetches comprehensive info from yfinance for a ticker."""
    try:
        info = yf.Ticker(ticker).info
        if not info or (info.get('regularMarketPrice') is None and info.get('currentPrice') is None and info.get('financialCurrency') is None):
            # Warning handled by orchestrator if this returns {}
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
    except Exception as e:
        # st.error(f"Error fetching ticker info for {ticker}: {e}") # Error displayed by orchestrator
        return {}

@st.cache_data
def fetch_enriched_news(ticker: str, ticker_info_data: dict) -> list[dict]:
    """
    Fetches news for a given ticker from yfinance and enriches it.
    Uses pre-fetched ticker_info_data for company name.
    """
    try:
        company_name = ticker_info_data.get('longName', ticker_info_data.get('shortName', ticker))
        ticker_obj = yf.Ticker(ticker)
        raw_news = []
        try:
            raw_news = ticker_obj.news
        except TypeError as te:
            return [{"error": f"yfinance .news call failed for {ticker} with TypeError: {te}", "source_api": "Yahoo Finance"}]
        except Exception as news_exc: # Catch other yfinance news errors
            return [{"error": f"yfinance .news call failed for {ticker}: {news_exc}", "source_api": "Yahoo Finance"}]

        enriched_news_list = []
        if not raw_news:
            return []

        for news_item in raw_news:
            if not isinstance(news_item, dict):
                continue

            enriched_item = news_item.copy()
            enriched_item['ticker'] = ticker
            enriched_item['company_name'] = company_name
            enriched_item['source_api'] = 'Yahoo Finance'

            if 'providerPublishTime' in news_item and news_item['providerPublishTime'] is not None:
                try:
                    timestamp = int(news_item['providerPublishTime'])
                    dt_object_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                    enriched_item['publish_datetime_utc'] = dt_object_utc
                    enriched_item['publish_time_readable'] = dt_object_utc.strftime('%Y-%m-%d %H:%M:%S %Z')
                except (ValueError, TypeError, OSError) as e_ts:
                    enriched_item['publish_datetime_utc'] = None
                    enriched_item['publish_time_readable'] = "N/A"
                    enriched_item['publish_time_error'] = str(e_ts)
            else:
                enriched_item['publish_datetime_utc'] = None
                enriched_item['publish_time_readable'] = "N/A"

            enriched_item.setdefault('title', 'No Title')
            enriched_item.setdefault('publisher', 'N/A')
            enriched_item.setdefault('link', '#')
            enriched_item.setdefault('type', 'N/A')
            enriched_news_list.append(enriched_item)

        enriched_news_list.sort(key=lambda x: x.get('publish_datetime_utc') or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return enriched_news_list
    except Exception as e: # General processing error
        return [{"error": f"Failed to process Yahoo Finance news for {ticker}: {e}", "source_api": "Yahoo Finance"}]

@st.cache_data(ttl=1800)
def fetch_comprehensive_news_from_api(ticker: str, company_name: str, lookback_days: int = 30) -> list[dict]:
    api_key = st.secrets.get("NEWSAPI_KEY")
    if not api_key:
        return [{"error": "NEWSAPI_KEY not found in secrets for NewsAPI.org.", "source_api": "NewsAPI.org"}]

    newsapi = NewsApiClient(api_key=api_key)
    query = f'("{company_name}" OR {ticker.upper()}) AND (stock OR shares OR business OR finance OR earnings OR "product launch" OR "analyst rating" OR "market sentiment")'
    
    to_date_dt = datetime.now(timezone.utc)
    from_date_dt = to_date_dt - timedelta(days=lookback_days)
    from_param_str = from_date_dt.strftime('%Y-%m-%d')
    to_param_str = to_date_dt.strftime('%Y-%m-%d')

    articles_list = []
    try:
        all_articles_response = newsapi.get_everything(
            q=query, from_param=from_param_str, to=to_param_str,
            language='en', sort_by='publishedAt', page_size=100
        )

        if all_articles_response.get("status") == "ok" and "articles" in all_articles_response:
            for article in all_articles_response["articles"]:
                dt_object_utc = None; readable_time = "N/A"
                if article.get('publishedAt'):
                    try:
                        dt_object_utc = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                        readable_time = dt_object_utc.strftime('%Y-%m-%d %H:%M:%S %Z')
                    except ValueError: # Handle malformed date string from API
                        pass
                articles_list.append({
                    "uuid": article.get('url'), "title": article.get('title', 'No Title Provided'),
                    "publisher": article.get('source', {}).get('name', 'N/A'),
                    "link": article.get('url', '#'), "publish_datetime_utc": dt_object_utc,
                    "publish_time_readable": readable_time, "description": article.get('description'),
                    "content_snippet": article.get('content'), "company_name": company_name,
                    "ticker": ticker, "source_api": "NewsAPI.org"
                })
        elif all_articles_response.get("status") == "error":
            return [{"error": f"NewsAPI Error ({ticker}): {all_articles_response.get('code')} - {all_articles_response.get('message')}", "source_api": "NewsAPI.org"}]
        else:
            return [{"error": f"NewsAPI ({ticker}): No articles found or unexpected response structure.", "source_api": "NewsAPI.org"}]
    except requests.exceptions.RequestException as e:
        return [{"error": f"NewsAPI request failed for {ticker}: {e}", "source_api": "NewsAPI.org"}]
    except Exception as e: # Catch any other unexpected error
        return [{"error": f"Unexpected error fetching news from NewsAPI for {ticker}: {e}", "source_api": "NewsAPI.org"}]
    return articles_list

@st.cache_data(ttl=24*3600)
def get_all_cik_ticker_mappings():
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(url, headers={'User-Agent': SEC_USER_AGENT})
        response.raise_for_status()
        data = response.json()
        ticker_to_cik = {item['ticker']: str(item['cik_str']).zfill(10) for item in data if 'ticker' in item and 'cik_str' in item}
        return ticker_to_cik
    except Exception as e:
        st.error(f"CRITICAL: Failed to fetch CIK ticker mappings: {e}. SEC features might be impaired.")
        return {}
TICKER_TO_CIK_MAP = get_all_cik_ticker_mappings()

def get_cik_for_ticker(ticker: str) -> str | None:
    return TICKER_TO_CIK_MAP.get(ticker.upper())

@st.cache_data(ttl=4*3600)
def fetch_all_sec_filings(ticker_symbol: str, lookback_days: int = 365) -> list[dict]:
    cik = get_cik_for_ticker(ticker_symbol)
    if not cik:
        try:
            headers = {'User-Agent': SEC_USER_AGENT}
            lookup_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker_symbol.upper()}&owner=exclude&count=10"
            response = requests.get(lookup_url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            cik_anchor = soup.find('a', href=re.compile(r"CIK=(\d{10})"))
            if cik_anchor:
                match = re.search(r"CIK=(\d{10})", cik_anchor['href'])
                if match: cik = match.group(1)
            if not cik:
                cik_text_match = re.search(r"CIK:\s*(\d{10})", soup.get_text(), re.IGNORECASE)
                if cik_text_match: cik = cik_text_match.group(1)
        except Exception: # Suppress error for this non-critical fallback
            pass 
    
    if not cik:
        return [{"error": f"SEC Filings: CIK could not be determined for {ticker_symbol}."}]

    cik_padded = str(cik).zfill(10)
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    headers = {'User-Agent': SEC_USER_AGENT}

    filings_list = []
    try:
        response = requests.get(submissions_url, headers=headers, timeout=20)
        response.raise_for_status()
        submissions_data = response.json()

        today = datetime.now(timezone.utc)
        date_limit = today - timedelta(days=lookback_days)

        if 'filings' in submissions_data and 'recent' in submissions_data['filings']:
            recent_filings = submissions_data['filings']['recent']
            forms = recent_filings.get('form',[])
            filing_dates = recent_filings.get('filingDate',[])
            accession_numbers = recent_filings.get('accessionNumber',[])
            primary_documents = recent_filings.get('primaryDocument',[])

            filings_to_process_metadata = []
            for i in range(len(forms)): # Iterate safely up to the shortest list length
                try:
                    filing_date_dt = datetime.strptime(filing_dates[i], '%Y-%m-%d').replace(tzinfo=timezone.utc)
                    if filing_date_dt >= date_limit:
                        filings_to_process_metadata.append({
                            "form_type": forms[i], "filing_date_str": filing_dates[i],
                            "accession_number": accession_numbers[i], "primary_document": primary_documents[i]
                        })
                except (ValueError, IndexError): # Handle parsing errors or mismatched list lengths
                    continue

            form4_xml_fetches = 0; max_form4_xml_fetches = 20; max_other_filings_to_list = 15

            for filing_info in filings_to_process_metadata:
                form_type = filing_info["form_type"]; filing_date_str = filing_info["filing_date_str"]
                accession_number = filing_info["accession_number"]; primary_document_name = filing_info["primary_document"]
                accession_number_no_dashes = accession_number.replace('-', '')
                sec_filing_index_link = f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{accession_number_no_dashes}/{accession_number}-index.html"

                if form_type == '4' and primary_document_name.lower().endswith(('.xml', '.xsd')):
                    if form4_xml_fetches >= max_form4_xml_fetches: continue
                    xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{accession_number_no_dashes}/{primary_document_name}"
                    try:
                        filing_response = requests.get(xml_url, headers=headers, timeout=10)
                        if filing_response.status_code != 200: continue
                        soup_xml = BeautifulSoup(filing_response.content, 'xml'); form4_xml_fetches += 1
                        reporting_owner_tag = soup_xml.find('reportingOwner')
                        owner_name = "N/A"; owner_relationship_str = "N/A"
                        if reporting_owner_tag:
                            owner_id_tag = reporting_owner_tag.find('reportingOwnerId')
                            if owner_id_tag and owner_id_tag.find('rptOwnerName'): owner_name = owner_id_tag.find('rptOwnerName').text.strip()
                            rel_tag = reporting_owner_tag.find('reportingOwnerRelationship')
                            if rel_tag:
                                rels = []
                                if rel_tag.find('isDirector') and rel_tag.find('isDirector').text in ['1', 'true']: rels.append("Director")
                                if rel_tag.find('isOfficer') and rel_tag.find('isOfficer').text in ['1', 'true']:
                                    title_tag = rel_tag.find('officerTitle')
                                    rels.append(f"Officer ({title_tag.text.strip() if title_tag and title_tag.text else ''})")
                                if rel_tag.find('isTenPercentOwner') and rel_tag.find('isTenPercentOwner').text in ['1', 'true']: rels.append(">10% Owner")
                                if rels: owner_relationship_str = ", ".join(filter(None, rels))
                        for transaction_table_name in ['nonDerivativeTable', 'derivativeTable']:
                            table = soup_xml.find(transaction_table_name)
                            if not table: continue
                            for transaction in table.find_all(['nonDerivativeTransaction', 'derivativeTransaction']):
                                trans_date_tag = transaction.find('transactionDate')
                                trans_date = trans_date_tag.find('value').text.strip() if trans_date_tag and trans_date_tag.find('value') else "N/A"
                                trans_coding_tag = transaction.find('transactionCoding')
                                trans_code = trans_coding_tag.find('transactionCode').text.strip().upper() if trans_coding_tag and trans_coding_tag.find('transactionCode') else "N/A"
                                shares_val = 0.0; price_val = None
                                amounts_tag = transaction.find('transactionAmounts')
                                if amounts_tag and amounts_tag.find('transactionShares') and amounts_tag.find('transactionShares').find('value'):
                                    try: shares_val = float(amounts_tag.find('transactionShares').find('value').text.strip())
                                    except ValueError: continue
                                price_node = transaction.find('transactionPricePerShare')
                                if price_node and price_node.find('value'):
                                    try: price_val = float(price_node.find('value').text.strip())
                                    except ValueError: price_val = None
                                acq_disp_node = transaction.find('transactionAcquiredDisposedCode')
                                acq_disp_code = acq_disp_node.find('value').text.strip().upper() if acq_disp_node and acq_disp_node.find('value') else "N/A"
                                if shares_val != 0:
                                    filings_list.append({
                                        "is_form4_transaction": True, "ticker": ticker_symbol, "filing_date": filing_date_str,
                                        "transaction_date": trans_date, "reporting_owner": owner_name,
                                        "owner_relationship": owner_relationship_str, "transaction_code": trans_code,
                                        "acq_disp_code": acq_disp_code, "shares": shares_val, "price_per_share": price_val,
                                        "link_to_filing": sec_filing_index_link
                                    })
                    except requests.exceptions.RequestException: pass 
                    except Exception: pass
                elif len([f for f in filings_list if not f.get("is_form4_transaction")]) < max_other_filings_to_list :
                    filings_list.append({
                        "is_form4_transaction": False, "ticker": ticker_symbol, "filing_date": filing_date_str,
                        "form_type": form_type,
                        "document_link": f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{accession_number_no_dashes}/{primary_document_name}",
                        "summary_link": sec_filing_index_link
                    })
            if not filings_list and form4_xml_fetches > 0 :
                return [{"error": f"SEC: Found {form4_xml_fetches} Form 4s for {ticker_symbol}, but failed to parse transaction details."}]
            if not filings_list:
                return [{"error": f"SEC: No relevant filings found/parsable for {ticker_symbol} in last {lookback_days} days (CIK: {cik_padded})."}]
        else:
            return [{"error": f"SEC: No recent filings data in submissions JSON for {ticker_symbol} (CIK: {cik_padded})."}]
    except requests.exceptions.HTTPError as e:
        return [{"error": f"SEC: HTTP error for {ticker_symbol} (CIK: {cik_padded}, submissions.json): {e}"}]
    except requests.exceptions.RequestException as e:
        return [{"error": f"SEC: Request error for {ticker_symbol} (CIK: {cik_padded}, submissions.json): {e}"}]
    except Exception as e:
        return [{"error": f"SEC: Unexpected error for {ticker_symbol} (CIK: {cik_padded}): {e}"}]
    
    filings_list.sort(key=lambda x: x.get('filing_date', '1900-01-01'), reverse=True)
    return filings_list

@st.cache_data(ttl=6*3600)
def fetch_inst_filings(ticker: str) -> list[dict]:
    try:
        ticker_obj = yf.Ticker(ticker)
        df_holders = ticker_obj.institutional_holders 
        if df_holders is not None and not df_holders.empty:
            if 'Shares' in df_holders.columns: df_holders['Shares'] = pd.to_numeric(df_holders['Shares'], errors='coerce').fillna(0)
            if '% Out' in df_holders.columns: df_holders['% Out'] = pd.to_numeric(df_holders['% Out'], errors='coerce').fillna(0.0)
            if 'Date Reported' in df_holders.columns: df_holders['Date Reported'] = df_holders['Date Reported'].astype(str)
            return df_holders.to_dict("records")
        return [{"error": f"No institutional holder data found for {ticker} via yfinance."}]
    except Exception as e:
        return [{"error": f"Failed to fetch institutional holders for {ticker} via yfinance: {e}"}]

@st.cache_data(ttl=4 * 3600)
def fetch_value_investing_io_data(ticker: str) -> dict:
    url = f"https://valueinvesting.io/{ticker.upper()}/valuation/fair-value"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() 
        soup = BeautifulSoup(response.content, 'html.parser')
        target_paragraph_text = None
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if ticker.upper() in text and "Fair Value" in text and ("Peter Lynch" in text or "based on" in text or "valuation model" in text):
                target_paragraph_text = text; break
        if not target_paragraph_text:
            return {"error": f"VI.io: Target paragraph (Peter Lynch Fair Value) not found for {ticker}."}
        pattern = re.compile(
            r"As of (?P<date>[\d]{4}-[\d]{2}-[\d]{2}), the Fair Value of .*?\(.*?" + re.escape(ticker.upper()) + 
            r".*?\) is (?P<fair_value>[\d\.]+) USD\.?" +
            r"(?:.*?With the current market price of (?P<market_price>[\d\.]+) USD, the upside of .*? is (?P<upside_percent>[-+]?\d+\.?\d*)%\.?)?"
        )
        match = pattern.search(target_paragraph_text)
        if match:
            data = match.groupdict()
            return {"ticker": ticker, "vi_valuation_date": data.get("date"),
                    "vi_fair_value": float(data.get("fair_value")) if data.get("fair_value") else None,
                    "vi_site_market_price": float(data.get("market_price")) if data.get("market_price") else None,
                    "vi_upside_percent": float(data.get("upside_percent")) if data.get("upside_percent") else None,
                    "vi_full_text": target_paragraph_text, "vi_data_source_url": url, "error": None}
        else:
            fair_value_match = re.search(r"Fair Value.*?is ([\d\.]+) USD", target_paragraph_text) # Generic fallback
            if fair_value_match:
                 return {"ticker": ticker, "vi_valuation_date": "N/A (generic parse)",
                         "vi_fair_value": float(fair_value_match.group(1)) if fair_value_match.group(1) else None,
                         "vi_site_market_price": None, "vi_upside_percent": None,
                         "vi_full_text": target_paragraph_text, "vi_data_source_url": url, "error": None,
                         "note": "Parsed with generic fair value regex."}
            return {"error": f"VI.io: Could not parse specific details for {ticker} from: '{target_paragraph_text[:200]}...'"}
    except requests.exceptions.HTTPError as http_err:
        return {"error": f"VI.io: HTTP error for {ticker} ({http_err.response.status_code if http_err.response else 'Unknown'}) at {url}."}
    except requests.exceptions.RequestException as req_err:
        return {"error": f"VI.io: Request error for {ticker}: {req_err}"}
    except Exception as e:
        return {"error": f"VI.io: Unexpected error for {ticker}: {e}"}

@st.cache_data(ttl=3600)
def fetch_politician_trades(ticker: str, days_back: int = 365) -> list[dict]:
    url = f"https://www.capitoltrades.com/trades?asset={ticker.upper()}&pageSize=100&perPage=100"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
               'Accept-Language': 'en-US,en;q=0.5', 'Referer': 'https://www.capitoltrades.com/'}
    politician_trades_list = []
    try:
        response = requests.get(url, headers=headers, timeout=20); response.raise_for_status() 
        soup = BeautifulSoup(response.content, 'html.parser')
        trade_rows = soup.select("a[href^='/trades/'][class*='trade-row'], a[href^='/trades/'][class*='issuer-trade-row']")
        if not trade_rows: trade_rows = soup.find_all('a', href=lambda href: href and href.startswith('/trades/'))
        if not trade_rows:
            return [{"error": f"CT: No trade rows found for {ticker}. Website HTML structure may have changed or scraping blocked. This feature is experimental."}]
        for row_link_tag in trade_rows[:20]: 
            name_tag = row_link_tag.find(['div','span'], class_=lambda x: x and ('politician-name' in x or 'filer-name' in x))
            type_tag = row_link_tag.find(['div','span'], class_=lambda x: x and ('tx-type' in x or 'transaction-type' in x))
            val_tag = row_link_tag.find(['div','span'], class_=lambda x: x and ('tx-value' in x or 'transaction-value' in x))
            date_tag = row_link_tag.find(['div','span'], class_=lambda x: x and ('tx-date' in x or 'transaction-date' in x))
            if all([name_tag, type_tag, val_tag, date_tag]):
                name = name_tag.text.strip(); tx_type_text = type_tag.text.strip().lower()
                tx_type = "purchase" if "purchase" in tx_type_text else ("sale" if "sale" in tx_type_text else "other")
                val_range = val_tag.text.strip(); date_str = date_tag.text.strip(); val_est = 0 
                val_matches = re.findall(r'\$([\d,]+)', val_range)
                if val_matches:
                    try: val_est = int(val_matches[0].replace(',','')) 
                    except ValueError: pass 
                politician_trades_list.append({
                    "politician_name": name, "transaction_type": tx_type, "value_range": val_range,
                    "value_estimate_lower": val_est, "date_str": date_str,
                    "source_url": urljoin("https://www.capitoltrades.com", row_link_tag['href'])
                })
        if not politician_trades_list and trade_rows:
            return [{"error": f"CT: Found trade rows for {ticker}, but failed to parse fields. HTML structure of details might have changed."}]
        return politician_trades_list
    except requests.exceptions.Timeout: return [{"error": f"CT: Timeout fetching data for {ticker}."}]
    except requests.exceptions.HTTPError as http_err: return [{"error": f"CT: HTTP error {http_err.response.status_code if http_err.response else ''} for {ticker}. Website may be blocking or down."}]
    except requests.exceptions.RequestException as e: return [{"error": f"CT: Request error for {ticker}: {e}."}]
    except Exception as e: return [{"error": f"CT: General parsing error for {ticker}: {e}. Website structure likely changed."}]

# -------------------------------- LLM Client & Agents (Largely Unchanged Internally, only minor fixes if any)
# LLM Client
# --------------------------------
class ModelClient:
    def __init__(self, api_key: str, provider: str = "openai"):
        self.api_key = api_key; self.provider = provider
        OPENAI_DEFAULT_MODEL = "gpt-4o" ; DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"
        if not api_key: raise ValueError("API key required for ModelClient.")
        if provider == "deepseek":
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com/v1")
            self.model_name = DEEPSEEK_DEFAULT_MODEL
        elif provider == "openai":
            self.client = OpenAI(api_key=self.api_key); self.model_name = OPENAI_DEFAULT_MODEL
        else: raise ValueError(f"Unsupported LLM provider: {provider}")
    def generate(self, prompt: str) -> str:
        try:
            stream = self.client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt}], stream=True)
            return "".join(chunk.choices[0].delta.content for chunk in stream if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content)
        except Exception as e: raise Exception(f"LLM Generation Error ({self.provider}, {self.model_name}): {e}")

# PriceAgent, MomentumAgent, VolatilityAgent, SentimentAgent, NewsSummaryAgent, FundamentalsAgent,
# ValuationAgent, AnalystRatingAgent, SECFilingAgent, InstitutionalHoldingsAgent,
# PoliticianFilingsAgent, ValueInvestingIOAgent, PortfolioAgent classes are assumed to be the same as in the previous version
# with the internal robustness fixes already applied (e.g., handling of None for numeric types before calculation).
# For brevity, I am not re-listing them here if their core logic hasn't changed beyond what was discussed for robustness.
# Ensure these agent classes are present in your full script.
# I will paste them here just to be complete, assuming the prior robustness fixes were good.

class PriceAgent:
    def run(self, ticker: str, price_data_slice: pd.DataFrame) -> dict:
        if price_data_slice.empty or len(price_data_slice) < 200:
            return {"ticker": ticker, "price_signal": "hold", "sma50": np.nan, "sma200": np.nan, "rsi14": np.nan, "price_error": "Not enough data for indicators"}
        df = price_data_slice.copy(); df["SMA50"] = df["Close"].rolling(50).mean(); df["SMA200"] = df["Close"].rolling(200).mean()
        delta = df["Close"].diff(); gain = delta.clip(lower=0).rolling(14).mean(); loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan); df["RSI14"] = 100 - (100 / (1 + rs)); latest = df.iloc[-1]; signal = "hold"
        if pd.isna(latest.SMA50) or pd.isna(latest.SMA200) or pd.isna(latest.RSI14): signal = "hold" 
        elif latest.SMA50 > latest.SMA200 and latest.RSI14 < 70: signal = "buy"
        elif latest.SMA50 < latest.SMA200 and latest.RSI14 > 30: signal = "sell"
        return {"ticker": ticker, "sma50": float(latest.SMA50) if pd.notna(latest.SMA50) else np.nan,
                "sma200": float(latest.SMA200) if pd.notna(latest.SMA200) else np.nan,
                "rsi14": float(latest.RSI14) if pd.notna(latest.RSI14) else np.nan, "price_signal": signal}

class MomentumAgent:
    def run(self, ticker: str, price_data_slice: pd.DataFrame) -> dict:
        if price_data_slice.empty or len(price_data_slice) < 252 + 1: # Min days for 12-month momentum + current
             return {"ticker": ticker, "momentum_signal": "hold", "momentum_1m": np.nan, "momentum_12m": np.nan, "momentum_error": "Not enough data for 12M momentum"}
        df = price_data_slice; P_t = df.Close.iloc[-1]
        P_1m = df.Close.shift(21).iloc[-1] if len(df) > 21 else np.nan
        P_12m = df.Close.shift(252).iloc[-1] if len(df) > 252 else np.nan
        m1 = ((P_t / P_1m) - 1) if pd.notna(P_1m) and P_1m != 0 else np.nan
        m12 = ((P_t / P_12m) - 1) if pd.notna(P_12m) and P_12m != 0 else np.nan
        signal = "hold"
        if pd.notna(m1) and pd.notna(m12):
            if m12 > 0.01 and m1 > 0.01: signal = "buy"
            elif m12 < -0.01 and m1 < -0.01: signal = "sell"
        return {"ticker": ticker, "momentum_1m": float(m1) if pd.notna(m1) else np.nan,
                "momentum_12m": float(m12) if pd.notna(m12) else np.nan, "momentum_signal": signal}

class VolatilityAgent:
    def run(self, ticker: str, data: dict, price_data_slice: pd.DataFrame = None) -> dict:
        beta_val = data.get("ticker_info", {}).get("beta") 
        beta = float(beta_val) if isinstance(beta_val, (int,float)) else 1.0
        sig = "sell" if beta > 1.5 else ("buy" if beta < 0.8 else "hold"); ann_vol = np.nan; vol_weight = 0.0
        if price_data_slice is not None and not price_data_slice.empty and len(price_data_slice) > 1:
            ret = np.log(price_data_slice.Close / price_data_slice.Close.shift(1)).dropna()
            if not ret.empty:
                ann_vol = float(ret.std() * np.sqrt(252)) 
                vol_weight = float(1 / ann_vol) if ann_vol > 0 else 0.0
        return {"ticker": ticker, "beta": beta, "annual_vol": ann_vol, "vol_weight": vol_weight, "volatility_signal": sig}

class SentimentAgent: # Assuming previous version was robust
    def __init__(self, client): self.client = client
    def run(self, ticker: str, data: dict) -> dict:
        news_items_from_bundle = data.get("news", []); overall_news_fetch_error = data.get("news_fetch_status_error")
        if overall_news_fetch_error: return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": overall_news_fetch_error}
        valid_news_items = [item for item in news_items_from_bundle if isinstance(item, dict) and "error" not in item]
        if not valid_news_items:
            err = news_items_from_bundle[0].get("error") if news_items_from_bundle and isinstance(news_items_from_bundle[0], dict) and "error" in news_items_from_bundle[0] else "No valid news articles."
            return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": err}
        content_for_llm = []; company_name_overall = data.get("ticker_info",{}).get('longName', ticker)
        for item in valid_news_items[:7]: 
            title = item.get('title', ''); publisher = item.get('publisher', ''); description = item.get('description', ''); content = item.get('content_snippet', '')
            text_snippet = f"Headline: {title}"
            if content and isinstance(content, str) and len(content) > 10: text_snippet += f" | Content Snippet: {content.replace('[+... chars]', '').strip()}"
            elif description and isinstance(description, str): text_snippet += f" | Description: {description.strip()}"
            if publisher and publisher != 'N/A': text_snippet += f" (Source: {publisher} via {item.get('source_api', 'Unknown')})"
            content_for_llm.append(text_snippet)
        if not content_for_llm: return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": "No processable news content."}
        prompt = (f"Analyze sentiment for {company_name_overall} ({ticker}) ... Output only the number ...\n\nNews:\n" + "\n".join(f"- {c}" for c in content_for_llm))
        score = 0.0; llm_error_msg = None
        try:
            response_text = self.client.generate(prompt).strip()
            if response_text.startswith("Error:"): llm_error_msg = response_text
            else:
                match = re.search(r"([-+]?\d*\.\d+)|([-+]?\d+)", response_text)
                if match: score = max(-1.0, min(1.0, float(match.group(0))))
                else: llm_error_msg = f"LLM non-numeric sentiment: '{response_text[:50]}...'"
        except Exception as e: llm_error_msg = f"LLM sentiment call failed: {str(e)[:150]}"
        final_error_message_for_sentiment = llm_error_msg
        if overall_news_fetch_error and ("Error" in overall_news_fetch_error or "failed" in overall_news_fetch_error.lower()) :
            final_error_message_for_sentiment = f"News: {overall_news_fetch_error}" + (f" | LLM: {llm_error_msg}" if llm_error_msg else "")
        sig = "buy" if score > 0.25 and not llm_error_msg else ("sell" if score < -0.25 and not llm_error_msg else "hold")
        return {"ticker": ticker, "sentiment_score": score, "sentiment_signal": sig, "sentiment_error": final_error_message_for_sentiment}

class NewsSummaryAgent: # Assuming previous version was robust
    def __init__(self, client): self.client = client
    def run(self, ticker: str, data: dict) -> dict:
        news_items = data.get("news", []); company_name = data.get("ticker_info", {}).get('longName', ticker); news_fetch_error = data.get("news_fetch_status_error")
        if news_fetch_error: return {"ticker": ticker, "news_summary": "News summary skipped due to news fetching issues.", "news_summary_error": news_fetch_error}
        if not news_items or (isinstance(news_items[0], dict) and "error" in news_items[0] and not any("error" not in item for item in news_items)):
            err = news_items[0]["error"] if news_items and isinstance(news_items[0], dict) and "error" in news_items[0] else "No news for summary."
            return {"ticker": ticker, "news_summary": "No news for summary.", "news_summary_error": err}
        yahoo_news = [item for item in news_items if item.get('source_api') == 'Yahoo Finance' and "error" not in item][:5]
        newsapi_news = [item for item in news_items if item.get('source_api') == 'NewsAPI.org' and "error" not in item][:5]
        selected_news = []; len_y = len(yahoo_news); len_n = len(newsapi_news)
        for i in range(max(len_y, len_n)):
            if i < len_y: selected_news.append(yahoo_news[i])
            if i < len_n: selected_news.append(newsapi_news[i])
        final_snippets = []; seen_titles = set()
        for item in selected_news:
            if len(final_snippets) >= 7: break
            title = item.get('title', '');_ = item.get('description', ''); content = item.get('content_snippet', '').replace('[+... chars]', '').strip()
            if title in seen_titles: continue; seen_titles.add(title)
            text_to_add = f"Title: {title}"
            if content: text_to_add += f" | Content: {content}"
            elif _: text_to_add += f" | Description: {_}" # Using _ for description var
            final_snippets.append(text_to_add)
        if not final_snippets: return {"ticker": ticker, "news_summary": "No content for summary.", "news_summary_error": "No articles with content/desc."}
        prompt = (f"Provide a concise summary paragraph (max 200 words) about {company_name} ({ticker})...\n\nNews Articles:\n" + "\n".join(f"- {s}" for s in final_snippets))
        summary = "Could not generate summary."; error_msg = None
        try:
            response_text = self.client.generate(prompt).strip()
            if response_text.startswith("Error:"): error_msg = response_text
            else: summary = response_text
        except Exception as e: error_msg = f"LLM summary call failed: {str(e)[:150]}"
        return {"ticker": ticker, "news_summary": summary, "news_summary_error": error_msg}

class FundamentalsAgent: # Assuming previous version was robust
    def run(self, ticker: str, data: dict) -> dict:
        s = data.get("ticker_info", {}); mcap = s.get("marketCap"); fcf = s.get("freeCashflow"); roe = s.get("returnOnEquity"); de = s.get("debtToEquity")
        mcap_calc = mcap if isinstance(mcap, (int, float)) else 1; fcf_calc = fcf if isinstance(fcf, (int, float)) else 0
        roe_calc = roe if isinstance(roe, (int, float)) else 0; de_calc = de if isinstance(de, (int, float)) else 1000
        fcy = fcf_calc / mcap_calc if mcap_calc != 0 else 0
        piotroski_score = sum([roe_calc > 0.01, de_calc < 100, fcf_calc > 0]); signal = "hold"
        if piotroski_score >= 2: signal = "buy"
        elif piotroski_score == 0: signal = "sell"
        return {"ticker": ticker, "fcf_yield": float(fcy), "piotroski_score": int(piotroski_score), "fund_signal": signal}

class ValuationAgent: # Assuming previous version was robust
    def run(self, ticker: str, data: dict) -> dict:
        stats = data.get("ticker_info", {}); hist = data.get("price_history"); price_val = stats.get("currentPrice")
        if price_val is None and hist is not None and not hist.empty: price_val = hist["Close"].iloc[-1]
        current_price = float(price_val) if isinstance(price_val, (int, float)) and price_val > 0 else None
        if current_price is None: return {"ticker": ticker, "forward_pe": None, "relative_pe_signal": "hold", "dcf_fair_price": np.nan, "dcf_signal": "hold", "valuation_error": "Current price unavailable."}
        pe_val = stats.get("forwardPE"); pe = float(pe_val) if isinstance(pe_val, (int, float)) else None; rel_sig = "hold"
        if pe is not None and pe > 0: rel_sig = "buy" if pe < 15 else ("sell" if pe > 25 else "hold")
        fcf_val = stats.get("freeCashflow"); mcap_val = stats.get("marketCap")
        fcf = float(fcf_val) if isinstance(fcf_val, (int, float)) else None
        mcap = float(mcap_val) if isinstance(mcap_val, (int, float)) else None
        fcy = (fcf / mcap) if fcf is not None and mcap is not None and mcap != 0 else 0.0
        fair_price_est = current_price * (1 + fcy); dcf_sig = "hold"
        if fair_price_est > current_price * 1.15: dcf_sig = "buy"
        elif fair_price_est < current_price * 0.85: dcf_sig = "sell"
        return {"ticker": ticker, "forward_pe": pe, "relative_pe_signal": rel_sig,
                "dcf_fair_price": float(fair_price_est) if pd.notna(fair_price_est) else np.nan,
                "dcf_signal": dcf_sig, "valuation_error": None}

class AnalystRatingAgent: # Assuming previous version was robust
    def run(self, ticker: str, data: dict) -> dict:
        info = data.get("ticker_info", {}); hist = data.get("price_history"); price_val = info.get("currentPrice")
        if price_val is None and hist is not None and not hist.empty: price_val = hist["Close"].iloc[-1]
        current_price = float(price_val) if isinstance(price_val, (int, float)) and price_val > 0 else None
        if current_price is None: return {"ticker": ticker, "analyst_buy_pct_inferred": 0.5, "target_upside": 0.0, "yfinance_recommendation": "N/A", "analyst_signal": "hold", "analyst_error": "Current price unavailable."}
        target_val = info.get("targetMeanPrice"); target_mean = float(target_val) if isinstance(target_val, (int,float)) else None
        rec = str(info.get("recommendationKey", "hold")).lower(); upside = 0.0
        if target_mean is not None and current_price > 0: upside = (target_mean / current_price) - 1
        sig = "hold"
        if rec in ["buy", "strong_buy"] and upside > 0.10: sig = "buy"
        elif rec == "buy" and upside > 0.05: sig = "buy"
        elif rec in ["sell", "strong_sell", "underperform"] and upside < -0.05: sig = "sell"
        elif upside > 0.20: sig = "buy"; elif upside < -0.15: sig = "sell"
        buy_pct = {"strong_buy": 0.9, "buy": 0.7, "hold": 0.5, "underperform": 0.3, "sell": 0.1}.get(rec, 0.5)
        return {"ticker": ticker, "analyst_buy_pct_inferred": float(buy_pct), "target_upside": float(upside),
                "yfinance_recommendation": rec, "analyst_signal": sig, "analyst_error": None}

class SECFilingAgent: # Assuming previous version was robust
    def run(self, ticker: str, data: dict) -> dict:
        filings = data.get("sec_all_filings_raw", []); err = None
        if not filings: err = f"SEC: No raw filings for {ticker}."
        elif isinstance(filings[0], dict) and "error" in filings[0]: err = filings[0].get("error")
        if err: return {"ticker": ticker, "sec_net_insider_shares_1y": 0, "sec_insider_buy_value_1y": 0, "sec_insider_sell_value_1y": 0, "sec_filings_signal": "hold", "sec_filings_error": err, "sec_recent_form4_transactions": [], "sec_other_recent_filings": []}
        net_shares = 0; buy_val = 0; sell_val = 0; form4 = []; others = []
        for f in filings:
            if not isinstance(f, dict) or "error" in f: continue
            if f.get("is_form4_transaction"):
                form4.append(f); s = f.get("shares", 0.0); p = f.get("price_per_share")
                if not isinstance(s, (int, float)): s = 0.0
                if f.get("transaction_code") == "P" and f.get("acq_disp_code") == "A":
                    net_shares += s;
                    if isinstance(p, (int,float)) and s != 0: buy_val += s * p
                elif f.get("transaction_code") == "S" and f.get("acq_disp_code") == "D":
                    net_shares -= s;
                    if isinstance(p, (int,float)) and s != 0: sell_val += s * p
            else: others.append(f)
        sig = "hold"
        if net_shares > 2000 or buy_val > 200000: sig = "buy" 
        elif net_shares < -2000 or sell_val > 200000: sig = "sell"
        return {"ticker": ticker, "sec_net_insider_shares_1y": int(net_shares), "sec_insider_buy_value_1y": round(buy_val, 2),
                "sec_insider_sell_value_1y": round(sell_val, 2), "sec_filings_signal": sig, "sec_filings_error": None,
                "sec_recent_form4_transactions": form4[:10], "sec_other_recent_filings": others[:10]}

class InstitutionalHoldingsAgent: # Assuming previous version was robust
    def run(self, ticker: str, data: dict) -> dict:
        holdings = data.get("institutional_holdings", []); err = None
        if holdings and isinstance(holdings[0], dict) and "error" in holdings[0]:
            err = holdings[0]["error"]
            return {"ticker": ticker, "inst_num_holders": 0, "inst_total_shares_held": 0, "inst_total_pct_out": 0.0, "inst_holdings_signal": "hold", "inst_holdings_error": err, "inst_top_holders": []}
        num_h = 0; total_s = 0; total_pct = 0.0; top_h = []
        if holdings:
            valid_h = [d for d in holdings if isinstance(d, dict) and "error" not in d]
            if valid_h:
                num_h = len(valid_h)
                try:
                    total_s = sum(d.get('Shares', 0) for d in valid_h)
                    total_pct = sum(d.get('% Out', 0.0) for d in valid_h)
                    top_h = sorted(valid_h, key=lambda x: x.get('Shares', 0), reverse=True)[:10]
                except Exception as e: err = f"Error processing inst holdings: {e}"
            elif not err: err = "No valid inst holdings data."
        sig = "hold"
        if total_pct > 0.50: sig = "buy"
        elif total_pct < 0.05 and num_h > 0: sig = "sell"
        return {"ticker": ticker, "inst_num_holders": num_h, "inst_total_shares_held": int(total_s),
                "inst_total_pct_out": float(total_pct), "inst_holdings_signal": sig,
                "inst_holdings_error": err, "inst_top_holders": top_h}

class PoliticianFilingsAgent: # Assuming previous version was robust
    def run(self, ticker: str, data: dict) -> dict:
        trades = data.get("politician_trades", []); net_val = 0; buys = 0; sells = 0; err = None
        if trades and isinstance(trades, list) and len(trades)>0 and isinstance(trades[0], dict) and "error" in trades[0]: err = trades[0]["error"]
        elif trades:
            for trade in trades:
                if isinstance(trade, dict): 
                    val = trade.get("value_estimate_lower", 0)
                    if trade.get("transaction_type") == "purchase": net_val += val; buys += 1
                    elif trade.get("transaction_type") == "sale": net_val -= val; sells += 1
        sig = "hold"
        if not err: 
            if buys > sells and buys > 1 : sig = "buy"
            elif sells > buys and sells > 1: sig = "sell"
        return {"ticker": ticker, "politician_net_trade_value_estimate": net_val, "politician_buy_tx_count": buys,
                "politician_sell_tx_count": sells, "politician_filings_signal": sig, "politician_data_error": err}

class ValueInvestingIOAgent: # Assuming previous version was robust
    def run(self, ticker: str, data: dict) -> dict:
        vi = data.get("value_investing_io_data", {}); err = vi.get("error"); fv = vi.get("vi_fair_value"); site_mp = vi.get("vi_site_market_price"); up_pct = vi.get("vi_upside_percent"); val_date = vi.get("vi_valuation_date"); text = vi.get("vi_full_text"); sig = "hold"
        curr_pyf_val = data.get("ticker_info", {}).get("currentPrice")
        if curr_pyf_val is None and data.get("price_history") is not None and not data["price_history"].empty: curr_pyf_val = data["price_history"]["Close"].iloc[-1]
        curr_pyf = float(curr_pyf_val) if isinstance(curr_pyf_val, (int,float)) and curr_pyf_val > 0 else None
        if not err and fv is not None and curr_pyf is not None:
            mos = 0.15 
            if up_pct is not None:
                if up_pct > (mos * 100 + 5): sig = "buy"
                elif up_pct < -(mos * 100 + 5): sig = "sell"
            else:
                if curr_pyf < fv * (1 - mos): sig = "buy"
                elif curr_pyf > fv * (1 + mos): sig = "sell"
        return {"ticker": ticker, "vi_fair_value_estimate": fv, "vi_site_market_price": site_mp, "vi_upside_percent": up_pct,
                "vi_valuation_date": val_date, "vi_valuation_text_display": text, "vi_signal": sig, "vi_data_error": err}

class PortfolioAgent: # Assuming previous version was robust
    WEIGHTS = {"price": 1.0, "momentum": 0.8, "volatility": 0.3, "sentiment": 0.6, "fund": 0.9, "valuation_dcf": 0.5, "valuation_pe": 0.5, "sec_filings": 0.6, "inst_holdings": 0.3, "analyst": 0.7, "politician_filings": 0.4, "vi_signal": 0.8}
    def run(self, ticker: str, signals: list[dict], agent_weights: dict = None) -> dict:
        curr_w = agent_weights or self.WEIGHTS; total_score = 0; sum_w = 0; agg_s = {}
        for s_dict in signals:
            if isinstance(s_dict, dict): agg_s.update(s_dict)
        s_map = {"price_signal": "price", "momentum_signal": "momentum", "volatility_signal": "volatility", "sentiment_signal": "sentiment", "fund_signal": "fund", "dcf_signal": "valuation_dcf", "relative_pe_signal": "valuation_pe", "sec_filings_signal": "sec_filings", "inst_holdings_signal": "inst_holdings", "analyst_signal": "analyst", "politician_filings_signal": "politician_filings", "vi_signal": "vi_signal"}
        for s_key, w_key in s_map.items():
            s_val = agg_s.get(s_key); w = curr_w.get(w_key, 0)
            if s_val and w > 0 and s_val in ["buy", "hold", "sell"]:
                raw_score = {"buy": 1, "hold": 0, "sell": -1}.get(s_val, 0)
                total_score += raw_score * w; sum_w += w
        comp_score = (total_score / sum_w) if sum_w else 0.0
        decision = "buy" if comp_score > 0.15 else ("sell" if comp_score < -0.15 else "hold")
        return {"ticker": ticker, "composite_score": comp_score, "final_decision": decision}

# --------------------------------
# Orchestrator for Live Analysis
# --------------------------------
def run_live_analysis(tickers, llm_client, configs): # history_years removed
    results = {}
    for t in tickers:
        st.write(f"▶️ Running analysis for {t}...")
        price_history_full = fetch_price_history(t, period="max") # Use "max" period
        if price_history_full.empty:
            results[t] = {"error": f"Price history unavailable for {t}.", "ticker": t, "final_decision":"error", "composite_score":0}
            st.error(f"Skipping {t}: Price history error.")
            continue
        
        ticker_info = fetch_ticker_info(t)
        if not ticker_info or not ticker_info.get("financialCurrency"): # More robust check
            err_msg = f"Core ticker info (e.g., currency) unavailable for {t}. May be invalid, delisted, or lack yfinance data."
            results[t] = {"error": err_msg, "ticker": t, "final_decision":"error", "composite_score":0}
            st.error(f"Skipping {t}: {err_msg}")
            continue

        current_price_for_ticker = ticker_info.get("currentPrice")
        if current_price_for_ticker is None and not price_history_full.empty:
             current_price_for_ticker = price_history_full["Close"].iloc[-1]
        company_name_for_news = ticker_info.get('longName', ticker_info.get('shortName', t))
        
        combined_news_data_list = []; news_fetch_status_messages = []
        if configs["use_sentiment"]:
            yfinance_news = fetch_enriched_news(t, ticker_info)
            if yfinance_news and not (isinstance(yfinance_news[0], dict) and "error" in yfinance_news[0]): combined_news_data_list.extend(yfinance_news)
            elif yfinance_news and isinstance(yfinance_news[0], dict) and "error" in yfinance_news[0]: news_fetch_status_messages.append(f"Yahoo News: {yfinance_news[0]['error']}")
            if llm_client and st.secrets.get("NEWSAPI_KEY"):
                newsapi_articles = fetch_comprehensive_news_from_api(t, company_name_for_news, lookback_days=30)
                if newsapi_articles and not (isinstance(newsapi_articles[0], dict) and "error" in newsapi_articles[0]): combined_news_data_list.extend(newsapi_articles)
                elif newsapi_articles and isinstance(newsapi_articles[0], dict) and "error" in newsapi_articles[0]: news_fetch_status_messages.append(f"NewsAPI: {newsapi_articles[0]['error']}")
            elif configs["use_sentiment"] and not st.secrets.get("NEWSAPI_KEY"): news_fetch_status_messages.append("NewsAPI Key not configured; using Yahoo news only.")
        
        seen_urls = set(); deduplicated_news = []
        for item in combined_news_data_list:
            if isinstance(item, dict) and "error" not in item:
                url = item.get('link') or item.get('url') 
                if url and url not in seen_urls: deduplicated_news.append(item); seen_urls.add(url)
        if deduplicated_news: deduplicated_news.sort(key=lambda x: x.get('publish_datetime_utc') or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        news_fetch_status_for_bundle = " | ".join(news_fetch_status_messages) if news_fetch_status_messages else "News fetch OK"
        if not deduplicated_news and not news_fetch_status_messages and configs["use_sentiment"]: news_fetch_status_for_bundle = "No news articles found from enabled sources."

        data_bundle = {
            "price_history": price_history_full, "ticker_info": ticker_info, "news": deduplicated_news, 
            "news_fetch_status_error": news_fetch_status_for_bundle if any(kw in news_fetch_status_for_bundle.lower() for kw in ["error", "failed", "no news"]) else None,
            "politician_trades": fetch_politician_trades(t) if configs["use_politician_filings"] else [],
            "value_investing_io_data": fetch_value_investing_io_data(t) if configs["use_value_trades"] else {"error": "VI.io: Skipped by user config."},
            "institutional_holdings": fetch_inst_filings(t) if configs["use_filings"] else [],
            "sec_all_filings_raw": fetch_all_sec_filings(t, lookback_days=365) if configs["use_filings"] else []
        }
        
        all_agents_instances = [PriceAgent(), MomentumAgent(), VolatilityAgent(), FundamentalsAgent(), ValuationAgent(), AnalystRatingAgent()]
        if configs["use_sentiment"] and llm_client: all_agents_instances.extend([SentimentAgent(llm_client), NewsSummaryAgent(llm_client)])
        if configs["use_filings"]: all_agents_instances.extend([SECFilingAgent(), InstitutionalHoldingsAgent()])
        if configs["use_politician_filings"]: all_agents_instances.append(PoliticianFilingsAgent())
        if configs["use_value_trades"]: all_agents_instances.append(ValueInvestingIOAgent())

        agent_results_list = []
        for agent_instance in all_agents_instances:
            agent_name = agent_instance.__class__.__name__
            try:
                if isinstance(agent_instance, (PriceAgent, MomentumAgent)): res_agent = agent_instance.run(t, data_bundle["price_history"])
                elif isinstance(agent_instance, VolatilityAgent): res_agent = agent_instance.run(t, data_bundle, data_bundle["price_history"])
                else: res_agent = agent_instance.run(t, data_bundle)
                agent_results_list.append(res_agent)
            except Exception as e:
                err_key = agent_name.lower().replace("agent","") + "_error"; sig_key = agent_name.lower().replace("agent","") + "_signal"
                agent_results_list.append({sig_key: "error", err_key: f"Agent {agent_name} critical error: {str(e)[:150]}"})
                st.warning(f"Critical error in {agent_name} for {t}: {e}")

        final_decision_obj = PortfolioAgent().run(t, agent_results_list)
        current_result_dict = {"ticker": t, "current_price_display": current_price_for_ticker, "market_cap_display": ticker_info.get("marketCap"),
                               "industry_display": ticker_info.get("industry"), "sector_display": ticker_info.get("sector"), "ticker_info": ticker_info,
                               "news_headlines_for_popover": [f"{n.get('publish_time_readable','N/A')} - {n.get('title', 'N/A')} ({n.get('publisher','N/A')} via {n.get('source_api','Unknown')}) [Link]({n.get('link','#')})" + (f" - {n.get('content_snippet', n.get('description', ''))[:150]}..." if n.get('content_snippet') or n.get('description') else "") for n in deduplicated_news[:10]],
                               "politician_trades_for_popover": [pt for pt in data_bundle["politician_trades"][:5] if isinstance(pt, dict) and "error" not in pt], # Use from data_bundle
                               "news_status_display": news_fetch_status_for_bundle}
        for res_dict in agent_results_list:
            if isinstance(res_dict, dict): current_result_dict.update(res_dict)
        current_result_dict.update(final_decision_obj); results[t] = current_result_dict
    return results

# -------------------------------- Backtesting Engine (Largely unchanged)
def run_backtest(ticker, start_date, end_date, initial_capital, llm_client_placeholder, backtest_agent_weights): # Assumed robust from prior
    st.write(f"Preparing backtest for {ticker} from {start_date} to {end_date}...")
    s_date_obj = datetime.strptime(start_date, "%Y-%m-%d"); fetch_start_date = (s_date_obj - pd.DateOffset(months=18)).strftime("%Y-%m-%d")
    full_price_history = fetch_price_history(ticker, period="max", interval="1d")
    if full_price_history.empty: return {"error": f"Backtest failed for {ticker}: Price history empty."}, pd.DataFrame()
    price_history = full_price_history[(full_price_history.index >= pd.to_datetime(fetch_start_date)) & (full_price_history.index <= pd.to_datetime(end_date))].copy()
    if price_history.empty or len(price_history[price_history.index >= pd.to_datetime(start_date)]) < 2: return {"error": f"Backtest failed for {ticker}: Not enough data in selected range."}, pd.DataFrame()
    ticker_info_for_backtest = fetch_ticker_info(ticker) # Might be {} but agents should handle
    data_bundle_static = {"ticker_info": ticker_info_for_backtest}
    price_agent, momentum_agent, volatility_agent, portfolio_agent = PriceAgent(), MomentumAgent(), VolatilityAgent(), PortfolioAgent()
    portfolio_log, cash, shares_held, portfolio_value = [], initial_capital, 0, initial_capital
    backtest_run_dates = price_history[price_history.index >= pd.to_datetime(start_date)].index
    for current_date in backtest_run_dates:
        data_slice = price_history[price_history.index <= current_date]
        current_price_point = data_slice.Close.iloc[-1] if not data_slice.empty else (portfolio_value / shares_held if shares_held else 0)
        if data_slice.empty or len(data_slice) < 252 + 1: # Min days for indicators
            portfolio_log.append({"date": current_date, "cash": cash, "shares_held": shares_held, "price": current_price_point, "portfolio_value": portfolio_value, "signal": "hold (insufficient data)", "composite_score": 0.0}); continue
        current_price_val = data_slice.Close.iloc[-1]
        pa_res = price_agent.run(ticker, data_slice); ma_res = momentum_agent.run(ticker, data_slice); va_res = volatility_agent.run(ticker, data_bundle_static, data_slice)
        final_decision_obj = portfolio_agent.run(ticker, [pa_res, ma_res, va_res], agent_weights=backtest_agent_weights)
        final_decision = final_decision_obj["final_decision"]
        if final_decision == "buy" and cash > current_price_val and current_price_val > 0: shares_to_buy = cash / current_price_val; shares_held += shares_to_buy; cash = 0
        elif final_decision == "sell" and shares_held > 0: cash += shares_held * current_price_val; shares_held = 0
        portfolio_value = cash + shares_held * current_price_val
        portfolio_log.append({"date": current_date, "cash": cash, "shares_held": shares_held, "price": current_price_val, "portfolio_value": portfolio_value, "signal": final_decision, "composite_score": final_decision_obj["composite_score"]})
    log_df = pd.DataFrame(portfolio_log);
    if not log_df.empty: log_df.set_index("date", inplace=True)
    if log_df.empty or len(log_df) < 2: return {"message":f"Backtest log for {ticker} too short."}, pd.DataFrame()
    total_return = (log_df["portfolio_value"].iloc[-1] / initial_capital - 1) * 100
    num_days = (log_df.index[-1] - log_df.index[0]).days; num_years = num_days / 365.25 if num_days > 0 else (1/365.25 if num_days == 0 else 0)
    ann_ret = 0
    if num_years > 0 and initial_capital > 0: ann_ret = ((log_df["portfolio_value"].iloc[-1] / initial_capital) ** (1 / num_years) - 1) * 100
    elif num_years == 0 and initial_capital > 0: ann_ret = total_return
    log_df["daily_return"] = log_df["portfolio_value"].pct_change().fillna(0); ann_vol = log_df["daily_return"].std() * np.sqrt(252) * 100
    sharpe = (ann_ret / ann_vol) if ann_vol != 0 else 0
    log_df["cumulative_max"] = log_df["portfolio_value"].cummax(); log_df["drawdown"] = (log_df["portfolio_value"] - log_df["cumulative_max"]) / log_df["cumulative_max"].replace(0, np.nan)
    max_dd = log_df["drawdown"].min() * 100 if not log_df["drawdown"].empty and pd.notna(log_df["drawdown"].min()) else 0
    num_trades = (log_df['signal'] != log_df['signal'].shift()).fillna(False).sum() // 2
    return {"Initial Capital": f"${initial_capital:,.2f}", "Final Portfolio Value": f"${log_df['portfolio_value'].iloc[-1]:,.2f}", "Total Return (%)": f"{total_return:.2f}%", "Annualized Return (%)": f"{ann_ret:.2f}%", "Annualized Volatility (%)": f"{ann_vol:.2f}%", "Sharpe Ratio": f"{sharpe:.2f}", "Max Drawdown (%)": f"{max_dd:.2f}%", "Number of Trades (approx)": f"{num_trades}"}, log_df

# -------------------------------- Streamlit UI
llm_client = None
try:
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY") or st.secrets.get("DEEPSEEK_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if deepseek_key: llm_client = ModelClient(api_key=deepseek_key, provider="deepseek"); st.sidebar.caption("✅ LLM: DeepSeek Initialized")
    elif openai_key: llm_client = ModelClient(api_key=openai_key, provider="openai"); st.sidebar.caption("✅ LLM: OpenAI Initialized")
    else: st.sidebar.warning("LLM API key missing. Sentiment/Summary disabled.")
except ValueError as e: st.sidebar.error(f"LLM Init Error: {e}"); llm_client = None
except Exception as e: st.sidebar.error(f"LLM Init Unexpected Error: {e}"); llm_client = None

st.title("🚀 AI Hedge Fund Simulator")
st.header("⚙️ Configuration"); config_container = st.container(border=True)
app_mode = "Live Analysis" 
with config_container:
    app_mode = st.radio("Select Mode:", ["Live Analysis", "Backtesting"], key="app_mode_select_main", horizontal=True, index=0); st.markdown("---")
    if app_mode == "Live Analysis":
        st.subheader("Live Analysis Settings")
        tickers_in_main = st.text_input("Tickers (comma-separated):", "AAPL,MSFT,GOOG,CRWV", key="live_tickers_input_main")
        st.caption("ℹ️ Live analysis now uses all available historical data for indicators.")
        st.subheader("Feature Toggles (Live Analysis)"); cols_features = st.columns(3)
        with cols_features[0]:
            use_sentiment_live_main = st.checkbox("News Sentiment & Summary (LLM)", value=True if llm_client else False, disabled=not llm_client, key="live_sentiment_cb_main", help="Uses LLM. Requires NewsAPI key for comprehensive news.")
            use_filings_live_main = st.checkbox("SEC & Institutional Filings", value=True, key="live_sec_filings_cb_main")
        with cols_features[1]:
            use_politician_filings_main = st.checkbox("Politician Filings (Experimental)", value=False, key="live_politician_cb_main", help="Scrapes CapitolTrades.com. May be slow/unreliable.")
            use_value_trades_main = st.checkbox("ValueInvesting.io Fair Value (Experimental)", value=False, key="live_vt_cb_main", help="Scrapes ValueInvesting.io. May be slow/unreliable.")
        st.markdown(""); run_button_live_main = st.button("🚀 Run Live Analysis", use_container_width=True, type="primary", key="run_live_btn_main")
    elif app_mode == "Backtesting":
        st.subheader("Backtesting Settings"); bt_ticker_main = st.text_input("Ticker for Backtest:", "AAPL", key="bt_ticker_input_main").upper()
        col1_bt, col2_bt = st.columns(2)
        with col1_bt:
            default_bt_end_main = datetime.now() - timedelta(days=1); default_bt_start_main = default_bt_end_main - pd.DateOffset(years=3)
            bt_start_date_main_dt = st.date_input("Start Date:", default_bt_start_main, max_value=default_bt_end_main - timedelta(days=30), key="bt_start_date_main")
            bt_start_date_main = bt_start_date_main_dt.strftime("%Y-%m-%d")
        with col2_bt:
            min_end_dt = bt_start_date_main_dt + timedelta(days=30)
            bt_end_date_main = st.date_input("End Date:", default_bt_end_main, min_value=min_end_dt, max_value=datetime.now() - timedelta(days=1), key="bt_end_date_main").strftime("%Y-%m-%d")
        bt_initial_capital_main = st.number_input("Initial Capital:", 1000, 1000000, 10000, 1000, key="bt_capital_input_main", format="%d")
        with st.expander("Adjust Backtest Agent Weights (Simplified Strategy)", expanded=False):
            st.caption("Backtesting uses a simplified strategy. Adjust weights:")
            bt_w_price = st.slider("Price Sig Weight:", 0.0, 2.0, 1.0, 0.1, key="bt_w_price_main")
            bt_w_mom = st.slider("Momentum Sig Weight:", 0.0, 2.0, 0.8, 0.1, key="bt_w_momentum_main")
            bt_w_vol = st.slider("Volatility Sig Weight:", 0.0, 2.0, 0.2, 0.1, key="bt_w_vol_main")
            st.info("Other signals disabled in backtesting for performance.")
        backtest_portfolio_weights_main = {"price": bt_w_price, "momentum": bt_w_mom, "volatility": bt_w_vol, "sentiment": 0.0, "fund": 0.0, "valuation_dcf":0.0, "valuation_pe":0.0, "sec_filings": 0.0, "inst_holdings": 0.0, "analyst": 0.0, "politician_filings": 0.0, "vi_signal": 0.0}
        st.markdown(""); run_button_backtest_main = st.button("📈 Run Backtest", use_container_width=True, type="primary", key="run_bt_btn_main")
st.markdown("---")

if app_mode == "Live Analysis":
    if 'run_button_live_main' in locals() and run_button_live_main and 'tickers_in_main' in locals() and tickers_in_main:
        live_tickers_list_main = [t.strip().upper() for t in tickers_in_main.split(",") if t.strip()]
        if not live_tickers_list_main: st.error("Please enter at least one valid ticker.")
        else:
            live_configs_main = {"use_sentiment": use_sentiment_live_main, "use_filings": use_filings_live_main, "use_politician_filings": use_politician_filings_main, "use_value_trades": use_value_trades_main}
            if 'live_output' not in st.session_state: st.session_state.live_output = {}
            with st.spinner("⏳ Processing live analysis..."):
                st.session_state.live_output = run_live_analysis(live_tickers_list_main, llm_client, live_configs_main) # No history_years
            st.header("📊 Live Analysis Summary"); num_tickers = len(live_tickers_list_main); cols_per_row = min(num_tickers, 3)
            for i in range(0, num_tickers, cols_per_row):
                row_tickers = live_tickers_list_main[i:i+cols_per_row]; cols = st.columns(len(row_tickers))
                for idx, t_symbol in enumerate(row_tickers):
                    with cols[idx]:
                        res = st.session_state.live_output.get(t_symbol)
                        if not res or res.get("error"): st.error(f"**{t_symbol}**: {res.get('error', 'No data.') if res else 'No data.'}"); continue
                        dec = res.get("final_decision", "N/A").upper(); score = res.get("composite_score", float('nan')); price = res.get("current_price_display")
                        color_map = {"BUY": "green", "SELL": "red", "HOLD": "#FFA500", "ERROR": "#808080", "N/A": "#D3D3D3"}; color = color_map.get(dec, "#D3D3D3")
                        price_html = f'<p style="font-size: 0.9em;">Price: <strong>${price:,.2f}</strong></p>' if isinstance(price, (int,float)) else '<p style="font-size: 0.9em;">Price: <strong>N/A</strong></p>'
                        score_html = f'<p style="font-size: 0.9em;">Score: <strong style="color: {color};">{score:.2f}</strong></p>' if pd.notna(score) else f'<p style="font-size: 0.9em;">Score: <strong style="color: {color};">N/A</strong></p>'
                        st.markdown(f"""<div style="border:1px solid {color};border-radius:8px;padding:15px;margin-bottom:10px;background-color:{color}20;"><h3 style="margin-bottom:5px;color:{color};">{t_symbol}</h3><p style="font-size:1.6em;font-weight:bold;color:{color};margin-bottom:5px;">{dec}</p>{score_html}{price_html}</div>""", unsafe_allow_html=True)
            st.markdown("---")
            for t_symbol in live_tickers_list_main:
                res = st.session_state.live_output.get(t_symbol)
                if not res or res.get("error"): continue
                with st.expander(f"🔍 Detailed Analysis for {t_symbol} ({res.get('ticker_info',{}).get('longName', 'N/A')})"):
                    tabs = st.tabs(["📈 Chart & Core", " फंड Fundamentals", "💰 Valuation & Fair Value", "📰 News & Filings", "⚙️ All Signals"])
                    with tabs[0]: # Chart & Core
                        st.subheader("Price Performance & Core Signals")
                        # CORRECTED: Fetch price history directly for the chart.
                        price_history_for_chart = fetch_price_history(t_symbol, period="max")
                        if not price_history_for_chart.empty:
                            plot_df_chart = price_history_for_chart.copy()
                            if len(plot_df_chart) > 5*252: plot_df_chart = plot_df_chart.tail(5*252) # Plot last 5 years if very long
                            st.line_chart(plot_df_chart["Close"], use_container_width=True)
                        else: st.warning("Price chart data not available for display.")
                        core_s = {"Price Sig (SMA/RSI)": res.get("price_signal", "N/A").upper(), "SMA50/SMA200": f"{res.get('sma50', np.nan):.2f}/{res.get('sma200', np.nan):.2f}", "RSI14": f"{res.get('rsi14', np.nan):.2f}", "Momentum Sig (1M/12M)": res.get("momentum_signal", "N/A").upper(), "Momentum 1M/12M (%)": f"{res.get('momentum_1m', np.nan)*100:.1f}%/{res.get('momentum_12m', np.nan)*100:.1f}%", "Volatility Sig (Beta)": res.get("volatility_signal", "N/A").upper(), "Beta/Ann.Vol (%)": f"{res.get('beta', np.nan):.2f}/{res.get('annual_vol', np.nan)*100:.1f}%"}
                        st.dataframe(pd.Series(core_s, name="Value"), use_container_width=True)
                        if res.get("price_error"): st.caption(f"Price Note: {res.get('price_error')}")
                        if res.get("momentum_error"): st.caption(f"Momentum Note: {res.get('momentum_error')}")
                    with tabs[1]: # Fundamentals
                        st.subheader(f"Fundamentals - {res.get('industry_display', 'N/A')} ({res.get('sector_display', 'N/A')})")
                        info_exp = res.get("ticker_info", {}); fund_s_exp = {}
                        mcap_exp = res.get('market_cap_display'); fund_s_exp["Market Cap"] = f"${mcap_exp:,.0f}" if isinstance(mcap_exp, (int,float)) else "N/A"
                        fcfy_exp = res.get('fcf_yield'); fund_s_exp["FCF Yield"] = f"{fcfy_exp*100:.2f}%" if isinstance(fcfy_exp, (int,float)) else "N/A"
                        piot_exp = res.get('piotroski_score'); fund_s_exp["Piotroski Score"] = piot_exp if piot_exp is not None else "N/A"
                        roe_exp = info_exp.get('returnOnEquity'); fund_s_exp["ROE"] = f"{roe_exp*100:.1f}%" if isinstance(roe_exp, (int,float)) else "N/A"
                        de_exp = info_exp.get('debtToEquity'); fund_s_exp["Debt/Equity"] = f"{de_exp:.1f}" if isinstance(de_exp, (int,float)) else "N/A"
                        fund_s_exp["Fund. Signal"] = res.get("fund_signal", "N/A").upper()
                        st.dataframe(pd.Series(fund_s_exp, name="Value"), use_container_width=True)
                        if info_exp.get("longBusinessSummary"):
                            with st.popover("Business Summary"): st.markdown(info_exp.get("longBusinessSummary"))
                        else: st.info("No business summary.")
                    with tabs[2]: # Valuation
                        st.subheader("Valuation (yfinance)"); val_err = res.get("valuation_error")
                        if val_err: st.warning(f"Valuation (yf): {val_err}")
                        val_s_exp = {}; fwdpe = res.get('forward_pe'); val_s_exp["Fwd P/E"] = f"{fwdpe:.1f}" if isinstance(fwdpe,(int,float)) else "N/A"
                        val_s_exp["Rel. P/E Sig"] = res.get('relative_pe_signal', "N/A").upper()
                        dcf_fp = res.get('dcf_fair_price'); val_s_exp["DCF Fair Price (Est)"] = f"${dcf_fp:,.2f}" if pd.notna(dcf_fp) and isinstance(dcf_fp,(int,float)) else "N/A"
                        val_s_exp["DCF Sig"] = res.get('dcf_signal', "N/A").upper(); st.dataframe(pd.Series(val_s_exp, name="Value"), use_container_width=True)
                        st.subheader("Analyst Ratings"); an_s_exp = {}
                        an_s_exp["YF Rec"] = res.get("yfinance_recommendation", "N/A").replace("_"," ").title()
                        targ_up = res.get('target_upside'); an_s_exp["Target Upside (%)"] = f"{targ_up*100:.2f}%" if isinstance(targ_up,(int,float)) else "N/A"
                        buy_pct_inf = res.get('analyst_buy_pct_inferred'); an_s_exp["Inf. Buy %"] = f"{buy_pct_inf*100:.0f}%" if isinstance(buy_pct_inf,(int,float)) else "N/A"
                        an_s_exp["Analyst Sig"] = res.get("analyst_signal", "N/A").upper(); st.dataframe(pd.Series(an_s_exp, name="Value"), use_container_width=True)
                        if res.get("analyst_error"): st.caption(f"Analyst Note: {res.get('analyst_error')}")
                        if live_configs_main["use_value_trades"]:
                            st.subheader("ValueInvesting.io (Experimental)"); vi_err = res.get('vi_data_error'); vi_text = res.get('vi_valuation_text_display')
                            if not vi_err and (res.get('vi_fair_value_estimate') is not None or vi_text):
                                st.markdown("**VI.io Analysis:**"); 
                                if vi_text: st.markdown(f"> *{vi_text}*")
                                if res.get('vi_fair_value_estimate') is not None: st.markdown(f"- **Fair Value (VI.io):** ${res.get('vi_fair_value_estimate'):,.2f}")
                                if res.get('vi_site_market_price') is not None: st.markdown(f"- **Market Price (VI.io):** ${res.get('vi_site_market_price'):,.2f}")
                                price_disp_vi = res.get('current_price_display')
                                if price_disp_vi is not None and isinstance(price_disp_vi, (int,float)): st.markdown(f"- **Current Yahoo Price:** ${price_disp_vi:,.2f}")
                                if res.get('vi_upside_percent') is not None: st.markdown(f"- **Upside (VI.io):** {res.get('vi_upside_percent'):.2f}%")
                                st.markdown(f"- **VI.io Signal:** {res.get('vi_signal', 'N/A').upper()}")
                            elif vi_err: st.warning(f"VI.io Status: {vi_err}")
                            else: st.info("VI.io: No specific fair value analysis parsed.")
                    with tabs[3]: # News & Filings
                        if live_configs_main["use_sentiment"]:
                            st.subheader("News Sentiment (LLM)"); sent_status = res.get("news_status_display", "OK") 
                            if res.get("sentiment_error"): sent_status += f" | LLM Sent Err: {res.get('sentiment_error')}"
                            sent_s = {"Sent. Score": f"{res.get('sentiment_score',0.0):.2f}", "Sent. Signal": res.get("sentiment_signal", "N/A").upper(), "News/LLM Status": sent_status}
                            st.dataframe(pd.Series(sent_s, name="Value"), use_container_width=True)
                            st.subheader("News Summary (LLM)"); news_sum_err = res.get("news_summary_error")
                            if news_sum_err: st.error(f"News Summary Err: {news_sum_err}")
                            st.markdown(f"*{res.get('news_summary', 'No summary.')}*")
                            news_pop = res.get("news_headlines_for_popover")
                            if news_pop:
                                with st.popover("Recent News (Top 10)"):
                                    for title in news_pop: st.markdown(f"- {title}")
                            elif "Error" not in sent_status and "No news" not in sent_status : st.caption("No headlines for summary.")
                        else: st.info("News Sentiment/Summary disabled.")
                        st.markdown("---")
                        if live_configs_main["use_filings"]:
                            st.subheader("SEC Insider Tx (Form 4 - 1Y)"); sec_err = res.get("sec_filings_error")
                            if sec_err: st.caption(f"SEC Status: {sec_err}")
                            sec_data = {"Net Insider Shares (1Y)": f"{res.get('sec_net_insider_shares_1y',0):,}", "Insider Buy Val (1Y Est)": f"${res.get('sec_insider_buy_value_1y',0):,.0f}", "Insider Sell Val (1Y Est)": f"${res.get('sec_insider_sell_value_1y',0):,.0f}", "SEC Filings Sig": res.get("sec_filings_signal", "N/A").upper()}
                            st.dataframe(pd.Series(sec_data, name="Value"), use_container_width=True)
                            form4_pop = res.get("sec_recent_form4_transactions")
                            if form4_pop:
                                with st.popover("Recent SEC Form 4 Tx (Max 10)"):
                                    for tx in form4_pop:
                                        direction = "Acq" if tx.get('acq_disp_code')=='A' else ("Disp" if tx.get('acq_disp_code')=='D' else tx.get('acq_disp_code','N/A'))
                                        price_info = f"@ ${tx.get('price_per_share'):.2f}" if isinstance(tx.get('price_per_share'),(int,float)) else "(price N/A)"
                                        st.markdown(f"- **{tx.get('transaction_date')}**: {tx.get('reporting_owner')} ({tx.get('owner_relationship','')}) {direction} {tx.get('shares',0):,.0f} shares {price_info}. Code:{tx.get('transaction_code')}. [Link]({tx.get('link_to_filing')})")
                            elif not sec_err: st.caption("No recent Form 4 tx.")
                            other_f_pop = res.get("sec_other_recent_filings")
                            if other_f_pop:
                                st.subheader("Other Recent SEC Filings (1Y - Max 10)")
                                for f_item in other_f_pop: st.markdown(f"- **{f_item.get('filing_date')}**: Form {f_item.get('form_type')} - [View]({f_item.get('summary_link')})")
                            elif not sec_err: st.caption("No other recent SEC filings.")
                            st.subheader("Institutional Holdings (yfinance)"); inst_err = res.get("inst_holdings_error")
                            if inst_err: st.caption(f"Inst. Holdings Status: {inst_err}")
                            inst_data = {"# Inst. Holding": res.get('inst_num_holders',0), "Total Shares Held by Inst.": f"{res.get('inst_total_shares_held',0):,}", "% Outstanding Held by Inst.": f"{res.get('inst_total_pct_out',0.0)*100:.2f}%", "Inst. Holdings Sig": res.get("inst_holdings_signal", "N/A").upper()}
                            st.dataframe(pd.Series(inst_data, name="Value"), use_container_width=True)
                            top_h_pop = res.get("inst_top_holders")
                            if top_h_pop:
                                with st.popover("Top Inst. Holders (Max 10 yf)"):
                                    for i, h in enumerate(top_h_pop):
                                        shares_d = f"{h.get('Shares',0):,}" if isinstance(h.get('Shares'),(int,float)) else h.get('Shares','N/A')
                                        pct_d = f"{h.get('% Out',0.0)*100:.2f}%" if isinstance(h.get('% Out'),(int,float)) else h.get('% Out','N/A')
                                        st.markdown(f"{i+1}. **{h.get('Holder')}**: Sh:{shares_d} (%Out:{pct_d}) Rept:{h.get('Date Reported','N/A')}")
                            elif not inst_err: st.caption("No top inst. holder data.")
                        else: st.info("SEC/Inst. Filings disabled.")
                        if live_configs_main["use_politician_filings"]:
                            st.subheader("Politician Trading (Experimental)"); pol_err = res.get("politician_data_error")
                            if pol_err: st.warning(f"Poli. Trades Status: {pol_err}")
                            pol_data = {"Net Poli. Trade Val Est": f"${res.get('politician_net_trade_value_estimate',0):,.0f}", "Poli. Buy Tx": res.get('politician_buy_tx_count',0), "Poli. Sell Tx": res.get('politician_sell_tx_count',0), "Poli. Filings Sig": res.get("politician_filings_signal", "N/A").upper()}
                            st.dataframe(pd.Series(pol_data, name="Value"), use_container_width=True)
                            pol_pop = res.get("politician_trades_for_popover")
                            if pol_pop:
                                with st.popover("Recent Poli. Trades (Max 5)"):
                                    for trade in pol_pop: st.markdown(f"- **{trade.get('date_str')}**: {trade.get('politician_name')} - {trade.get('transaction_type','N/A').title()} - {trade.get('value_range')} [Link]({trade.get('source_url')})")
                            elif not pol_err: st.caption("No recent poli. trades.")
                        else: st.info("Poli. Filings disabled.")
                    with tabs[4]: # All Signals
                        st.subheader("Aggregated Signals & Final Decision")
                        all_s_keys = [k for k in res if k.endswith("_signal")]; all_s_tab = {k.replace("_signal","").replace("_"," ").title(): str(res[k]).upper() for k in all_s_keys}
                        all_s_tab["Composite Score"] = f"{res.get('composite_score',0.0):.2f}"; all_s_tab["Final Decision"] = res.get('final_decision',"").upper()
                        st.dataframe(pd.Series(all_s_tab, name="Signal Value"), use_container_width=True)
                        with st.popover("View Full Raw Analysis Data (JSON)"): st.json(res)
    with st.sidebar.expander("Portfolio Agent Weights (Live Analysis)", expanded=False):
        st.caption("Weights for PortfolioAgent combining signals."); st.json(dict(sorted(PortfolioAgent.WEIGHTS.items())))
elif app_mode == "Backtesting": # Backtesting UI display logic unchanged, assumed robust
    if 'run_button_backtest_main' in locals() and run_button_backtest_main and 'bt_ticker_main' in locals() and bt_ticker_main:
        if 'bt_metrics' not in st.session_state: st.session_state.bt_metrics = None
        if 'bt_log_df' not in st.session_state: st.session_state.bt_log_df = pd.DataFrame()
        with st.spinner(f"⏳ Running backtest for {bt_ticker_main}..."):
            st.session_state.bt_metrics, st.session_state.bt_log_df = run_backtest(bt_ticker_main, bt_start_date_main, bt_end_date_main, bt_initial_capital_main, llm_client, backtest_portfolio_weights_main)
        if st.session_state.bt_metrics and not (st.session_state.bt_metrics.get("message") or st.session_state.bt_metrics.get("error")):
            st.header(f"📈 Backtest Results for {bt_ticker_main}"); metrics_df = pd.DataFrame.from_dict(st.session_state.bt_metrics, orient='index', columns=['Value']); st.table(metrics_df)
            if not st.session_state.bt_log_df.empty:
                st.subheader("Portfolio Value Over Time"); st.line_chart(st.session_state.bt_log_df["portfolio_value"])
                st.subheader("Drawdown Over Time"); drawdown_s = st.session_state.bt_log_df["drawdown"].fillna(0); st.area_chart(drawdown_s)
                with st.expander("View Raw Backtest Log (Last 1000 rows)"): st.dataframe(st.session_state.bt_log_df[["price", "signal", "composite_score", "portfolio_value", "cash", "shares_held"]].tail(1000))
            else: st.warning("Backtest log empty.")
        else:
            err_msg_bt = "Unknown backtest error."
            if st.session_state.bt_metrics: err_msg_bt = st.session_state.bt_metrics.get('message', '') or st.session_state.bt_metrics.get('error', 'Unknown error')
            st.error(f"Backtest failed: {err_msg_bt}")
st.sidebar.markdown("---")
st.sidebar.info("Educational purposes only. Not financial advice.")
st.sidebar.markdown("Experimental scraping features may be unreliable.")

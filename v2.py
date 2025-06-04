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
from bs.bs4 import BeautifulSoup
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
        st.error(f"Error fetching price history for {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data
def fetch_ticker_info(ticker: str) -> dict:
    """Fetches comprehensive info from yfinance for a ticker."""
    try:
        info = yf.Ticker(ticker).info
        if not info or (info.get('regularMarketPrice') is None and info.get('currentPrice') is None):
            return {}
        # Ensure common keys are always present, even if with None or default values
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
        }
    except Exception as e:
        st.error(f"Error fetching ticker info for {ticker}: {e}")
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
            # Handle cases where .news might return non-iterable or None elements
            return [{"error": f"yfinance .news call failed for {ticker} with TypeError: {te}", "source_api": "Yahoo Finance"}]
        except Exception as news_exc:
            return [{"error": f"yfinance .news call failed for {ticker}: {news_exc}", "source_api": "Yahoo Finance"}]

        enriched_news_list = []
        if not raw_news:
            return []

        for news_item in raw_news:
            if not isinstance(news_item, dict):
                continue # Skip malformed items

            enriched_item = news_item.copy()
            enriched_item['ticker'] = ticker
            enriched_item['company_name'] = company_name
            enriched_item['source_api'] = 'Yahoo Finance'

            if 'providerPublishTime' in news_item and news_item['providerPublishTime'] is not None:
                try:
                    # Convert timestamp to datetime object in UTC
                    timestamp = int(news_item['providerPublishTime'])
                    dt_object_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                    enriched_item['publish_datetime_utc'] = dt_object_utc
                    enriched_item['publish_time_readable'] = dt_object_utc.strftime('%Y-%m-%d %H:%M:%S %Z')
                except (ValueError, TypeError, OSError) as e_ts:
                    # Handle invalid timestamp values
                    enriched_item['publish_datetime_utc'] = None
                    enriched_item['publish_time_readable'] = "N/A"
                    enriched_item['publish_time_error'] = str(e_ts)
            else:
                enriched_item['publish_datetime_utc'] = None
                enriched_item['publish_time_readable'] = "N/A"

            # Ensure essential keys are present
            enriched_item.setdefault('title', 'No Title')
            enriched_item.setdefault('publisher', 'N/A')
            enriched_item.setdefault('link', '#')
            enriched_item.setdefault('type', 'N/A')
            enriched_news_list.append(enriched_item)

        # Sort by publish date, handling None gracefully (None dates will go to the end)
        enriched_news_list.sort(key=lambda x: x.get('publish_datetime_utc') or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return enriched_news_list
    except Exception as e:
        # Catch any other unexpected errors during processing
        return [{"error": f"Failed to process Yahoo Finance news for {ticker}: {e}", "source_api": "Yahoo Finance"}]

@st.cache_data(ttl=1800)
def fetch_comprehensive_news_from_api(ticker: str, company_name: str, lookback_days: int = 30) -> list[dict]:
    """
    Fetches news from NewsAPI.org for a given ticker and company name.
    """
    api_key = st.secrets.get("NEWSAPI_KEY")
    if not api_key:
        return [{"error": "NEWSAPI_KEY not found in secrets for NewsAPI.org.", "source_api": "NewsAPI.org"}]

    newsapi = NewsApiClient(api_key=api_key)
    # Refined query to be more relevant and include stock/finance terms
    query = f'("{company_name}" OR {ticker.upper()}) AND (stock OR shares OR business OR finance OR earnings OR "product launch" OR "analyst rating" OR "market sentiment")'
    
    to_date_dt = datetime.now(timezone.utc)
    from_date_dt = to_date_dt - timedelta(days=lookback_days)
    from_param_str = from_date_dt.strftime('%Y-%m-%d')
    to_param_str = to_date_dt.strftime('%Y-%m-%d')

    articles_list = []
    try:
        all_articles_response = newsapi.get_everything(
            q=query, from_param=from_param_str, to=to_param_str,
            language='en', sort_by='publishedAt', page_size=100 # Increased page_size to max 100
        )

        if all_articles_response.get("status") == "ok" and "articles" in all_articles_response:
            for article in all_articles_response["articles"]:
                dt_object_utc = None; readable_time = "N/A"
                if article.get('publishedAt'):
                    try:
                        # NewsAPI returns ISO 8601 format, handle 'Z' for UTC
                        dt_object_utc = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                        readable_time = dt_object_utc.strftime('%Y-%m-%d %H:%M:%S %Z')
                    except ValueError:
                        # In case of malformed date string
                        pass
                articles_list.append({
                    "uuid": article.get('url'), # Using URL as a unique ID for deduplication
                    "title": article.get('title', 'No Title Provided'),
                    "publisher": article.get('source', {}).get('name', 'N/A'),
                    "link": article.get('url', '#'), # Explicitly include link
                    "publish_datetime_utc": dt_object_utc,
                    "publish_time_readable": readable_time,
                    "description": article.get('description'),
                    "content_snippet": article.get('content'), # This is the 'content' field, often truncated by NewsAPI
                    "company_name": company_name,
                    "ticker": ticker,
                    "source_api": "NewsAPI.org"
                })
        elif all_articles_response.get("status") == "error":
            # Propagate NewsAPI specific errors
            return [{"error": f"NewsAPI Error ({ticker}): {all_articles_response.get('code')} - {all_articles_response.get('message')}", "source_api": "NewsAPI.org"}]
        else:
            return [{"error": f"NewsAPI ({ticker}): No articles found or unexpected data structure.", "source_api": "NewsAPI.org"}]
    except requests.exceptions.RequestException as e:
        return [{"error": f"NewsAPI request failed for {ticker}: {e}", "source_api": "NewsAPI.org"}]
    except Exception as e:
        return [{"error": f"Unexpected error fetching news from NewsAPI for {ticker}: {e}", "source_api": "NewsAPI.org"}]
    return articles_list

@st.cache_data(ttl=24*3600)
def get_all_cik_ticker_mappings():
    """Fetches and caches the CIK to ticker mappings from SEC.gov."""
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(url, headers={'User-Agent': SEC_USER_AGENT})
        response.raise_for_status() # Raise an exception for HTTP errors
        data = response.json()
        ticker_to_cik = {item['ticker']: str(item['cik_str']).zfill(10) for item in data if 'ticker' in item and 'cik_str' in item}
        return ticker_to_cik
    except Exception as e:
        st.error(f"CRITICAL: Failed to fetch CIK ticker mappings: {e}")
        return {}
TICKER_TO_CIK_MAP = get_all_cik_ticker_mappings() # Load once globally

def get_cik_for_ticker(ticker: str) -> str | None:
    """Retrieves CIK for a given ticker from the pre-loaded map."""
    return TICKER_TO_CIK_MAP.get(ticker.upper())

@st.cache_data(ttl=4*3600)
def fetch_all_sec_filings(ticker_symbol: str, lookback_days: int = 365) -> list[dict]:
    """
    Fetches recent SEC filings (including Form 4 insider trades) for a given ticker.
    Parses Form 4 XML for transaction details.
    """
    cik = get_cik_for_ticker(ticker_symbol)
    # Fallback CIK lookup if not in map (less efficient but useful for new tickers)
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
            # Another attempt if direct anchor not found
            if not cik:
                cik_text_match = re.search(r"CIK:\s*(\d{10})", soup.get_text(), re.IGNORECASE)
                if cik_text_match: cik = cik_text_match.group(1)
        except Exception as e:
            # st.warning(f"Secondary CIK lookup failed for {ticker_symbol}: {e}") # Optional: for debugging
            pass # Suppress detailed error for less critical fallback
    
    if not cik:
        return [{"error": f"SEC Filings: CIK could not be determined for {ticker_symbol}"}]

    cik_padded = str(cik).zfill(10) # Ensure CIK is 10 digits, zero-padded
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    headers = {'User-Agent': SEC_USER_AGENT} # Required by SEC for programmatic access

    filings_list = []
    try:
        response = requests.get(submissions_url, headers=headers, timeout=20)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        submissions_data = response.json()

        today = datetime.now(timezone.utc)
        date_limit = today - timedelta(days=lookback_days)

        if 'filings' in submissions_data and 'recent' in submissions_data['filings']:
            recent_filings = submissions_data['filings']['recent']
            
            # Extract lists for easier iteration
            forms = recent_filings.get('form',[])
            filing_dates = recent_filings.get('filingDate',[])
            accession_numbers = recent_filings.get('accessionNumber',[])
            primary_documents = recent_filings.get('primaryDocument',[])

            filings_to_process_metadata = []
            for i in range(len(forms)):
                try:
                    filing_date = datetime.strptime(filing_dates[i], '%Y-%m-%d').replace(tzinfo=timezone.utc)
                    # Only process filings within the lookback period
                    if filing_date >= date_limit:
                        filings_to_process_metadata.append({
                            "form_type": forms[i],
                            "filing_date_str": filing_dates[i],
                            "accession_number": accession_numbers[i],
                            "primary_document": primary_documents[i]
                        })
                except ValueError:
                    # Skip if date parsing fails
                    continue

            form4_xml_fetches = 0
            max_form4_xml_fetches = 20 # Limit XML fetches to avoid rate limits
            max_other_filings_to_list = 15 # Limit general filings listed

            for filing_info in filings_to_process_metadata:
                form_type = filing_info["form_type"]
                filing_date_str = filing_info["filing_date_str"]
                accession_number = filing_info["accession_number"]
                primary_document_name = filing_info["primary_document"]
                
                accession_number_no_dashes = accession_number.replace('-', '')
                sec_filing_link = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number_no_dashes}/{accession_number}-index.html"

                if form_type == '4' and primary_document_name.lower().endswith(('.xml', '.xsd')):
                    if form4_xml_fetches >= max_form4_xml_fetches:
                        continue # Skip if we've fetched too many XMLs
                    
                    xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number_no_dashes}/{primary_document_name}"
                    
                    try:
                        filing_response = requests.get(xml_url, headers=headers, timeout=10)
                        if filing_response.status_code != 200:
                            continue # Skip if fetching XML failed
                        
                        soup_xml = BeautifulSoup(filing_response.content, 'xml')
                        form4_xml_fetches += 1

                        # Extract reporting owner details
                        reporting_owner_tag = soup_xml.find('reportingOwner')
                        owner_name = "N/A"
                        owner_relationship_str = "N/A"
                        if reporting_owner_tag:
                            owner_id_tag = reporting_owner_tag.find('reportingOwnerId')
                            if owner_id_tag and owner_id_tag.find('rptOwnerName'):
                                owner_name = owner_id_tag.find('rptOwnerName').text.strip()
                            rel_tag = reporting_owner_tag.find('reportingOwnerRelationship')
                            if rel_tag:
                                rels = []
                                if rel_tag.find('isDirector') and rel_tag.find('isDirector').text in ['1', 'true']:
                                    rels.append("Director")
                                if rel_tag.find('isOfficer') and rel_tag.find('isOfficer').text in ['1', 'true']:
                                    title_tag = rel_tag.find('officerTitle')
                                    rels.append(f"Officer ({title_tag.text.strip() if title_tag and title_tag.text else ''})")
                                if rel_tag.find('isTenPercentOwner') and rel_tag.find('isTenPercentOwner').text in ['1', 'true']:
                                    rels.append(">10% Owner")
                                if rels:
                                    owner_relationship_str = ", ".join(filter(None, rels))

                        # Parse transactions
                        for transaction_table_name in ['nonDerivativeTable', 'derivativeTable']:
                            table = soup_xml.find(transaction_table_name)
                            if not table:
                                continue

                            for transaction in table.find_all(['nonDerivativeTransaction', 'derivativeTransaction']):
                                trans_date_tag = transaction.find('transactionDate')
                                trans_date = trans_date_tag.find('value').text.strip() if trans_date_tag and trans_date_tag.find('value') else "N/A"

                                trans_coding_tag = transaction.find('transactionCoding')
                                trans_code = trans_coding_tag.find('transactionCode').text.strip().upper() if trans_coding_tag and trans_coding_tag.find('transactionCode') else "N/A"

                                shares_val = 0.0
                                price_val = None

                                amounts_tag = transaction.find('transactionAmounts')
                                if amounts_tag and amounts_tag.find('transactionShares') and amounts_tag.find('transactionShares').find('value'):
                                    try:
                                        shares_val = float(amounts_tag.find('transactionShares').find('value').text.strip())
                                    except ValueError:
                                        continue # Skip if shares value is not numeric
                                
                                price_node = transaction.find('transactionPricePerShare')
                                if price_node and price_node.find('value'):
                                    try:
                                        price_val = float(price_node.find('value').text.strip())
                                    except ValueError:
                                        price_val = None # Keep as None if price is not numeric

                                acq_disp_node = transaction.find('transactionAcquiredDisposedCode')
                                acq_disp_code = acq_disp_node.find('value').text.strip().upper() if acq_disp_node and acq_disp_node.find('value') else "N/A"

                                # Only include transactions with actual share movement
                                if shares_val != 0:
                                    filings_list.append({
                                        "is_form4_transaction": True,
                                        "ticker": ticker_symbol,
                                        "filing_date": filing_date_str,
                                        "transaction_date": trans_date,
                                        "reporting_owner": owner_name,
                                        "owner_relationship": owner_relationship_str,
                                        "transaction_code": trans_code,
                                        "acq_disp_code": acq_disp_code,
                                        "shares": shares_val,
                                        "price_per_share": price_val,
                                        "link_to_filing": xml_url.replace(primary_document_name, "FilingSummary.xml") # Link to summary for context
                                    })
                    except requests.exceptions.RequestException as req_e:
                        # Log/handle request errors for XML fetches, but continue
                        # st.warning(f"Error fetching Form 4 XML for {ticker_symbol} ({accession_number}): {req_e}") # Optional: for debugging
                        pass
                    except Exception as e:
                        # Catch any other parsing errors for Form 4 XML
                        # st.warning(f"Error parsing Form 4 XML for {ticker_symbol} ({accession_number}): {e}") # Optional: for debugging
                        pass
                
                # Add other filing types (e.g., 10-K, 10-Q, 8-K) as metadata, limit count
                elif len([f for f in filings_list if not f.get("is_form4_transaction")]) < max_other_filings_to_list :
                    filings_list.append({
                        "is_form4_transaction": False,
                        "ticker": ticker_symbol,
                        "filing_date": filing_date_str,
                        "form_type": form_type,
                        "document_link": f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number_no_dashes}/{primary_document_name}",
                        "summary_link": sec_filing_link # Link to the index page for full context
                    })
            
            if not filings_list and form4_xml_fetches > 0 :
                return [{"error": f"SEC: Found {form4_xml_fetches} Form 4s for {ticker_symbol} but failed to parse transaction details."}]
            if not filings_list:
                return [{"error": f"SEC: No relevant filings found or parsable for {ticker_symbol} in last {lookback_days} days."}]
        else:
            return [{"error": f"SEC: No recent filings data for {ticker_symbol} (CIK: {cik})"}]
    except requests.exceptions.HTTPError as e:
        # Handle HTTP errors from the main submissions JSON URL
        return [{"error": f"SEC: HTTP error for {ticker_symbol} (submissions.json): {e}"}]
    except requests.exceptions.RequestException as e:
        # Handle general request errors (e.g., network issues)
        return [{"error": f"SEC: Request error for {ticker_symbol} (submissions.json): {e}"}]
    except Exception as e:
        # Catch any other unexpected errors
        return [{"error": f"SEC: Unexpected error for {ticker_symbol}: {e}"}]
    
    # Sort all filings by date
    filings_list.sort(key=lambda x: x.get('filing_date', '1900-01-01'), reverse=True)
    return filings_list

@st.cache_data(ttl=6*3600)
def fetch_inst_filings(ticker: str) -> list[dict]:
    """
    Fetches institutional holder data from yfinance. This typically summarizes 13F filings.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        # institutional_holders provides a snapshot, not historical changes easily.
        df_holders = ticker_obj.institutional_holders 
        
        if df_holders is not None and not df_holders.empty:
            # Ensure numeric conversion for calculations if needed
            if 'Shares' in df_holders.columns:
                df_holders['Shares'] = pd.to_numeric(df_holders['Shares'], errors='coerce').fillna(0)
            if '% Out' in df_holders.columns:
                df_holders['% Out'] = pd.to_numeric(df_holders['% Out'], errors='coerce').fillna(0.0)
            return df_holders.to_dict("records")
        return [{"error": f"No institutional holder data found for {ticker} via yfinance."}]
    except Exception as e:
        return [{"error": f"Failed to fetch institutional holders for {ticker} via yfinance: {e}"}]

@st.cache_data(ttl=4 * 3600)
def fetch_value_investing_io_data(ticker: str) -> dict:
    """
    Scrapes fair value data from valueinvesting.io based on Peter Lynch's formula.
    """
    url = f"https://valueinvesting.io/{ticker.upper()}/valuation/fair-value"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        target_paragraph_text = None
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            text = p.get_text(strip=True)
            # Make the regex more robust to variations in surrounding text
            # Looking for text containing ticker, "Fair Value", and indications of Peter Lynch or valuation basis
            if ticker.upper() in text and "Fair Value" in text and ("Peter Lynch" in text or "based on" in text):
                target_paragraph_text = text
                break

        if not target_paragraph_text:
            return {"error": f"VI.io: Target paragraph not found for {ticker} on {url}. No specific Peter Lynch Fair Value found."}

        # More flexible pattern to capture the details
        # Pattern structure: date, fair_value, market_price, upside_percent.
        # Made parts optional where they might not always be present or vary in wording.
        pattern = re.compile(
            r"As of (?P<date>[\d]{4}-[\d]{2}-[\d]{2}), the Fair Value of .*?\(.*?" + re.escape(ticker.upper()) + 
            r".*?\) is (?P<fair_value>[\d\.]+) USD\.?" + # Fair value in USD, optional period
            r"(?:.*?With the current market price of (?P<market_price>[\d\.]+) USD, the upside of .*? is (?P<upside_percent>[-+]?\d+\.?\d*)%\.?)?" # Optional market price and upside
        )
        match = pattern.search(target_paragraph_text)

        if match:
            data = match.groupdict()
            return {
                "ticker": ticker,
                "vi_valuation_date": data.get("date"),
                "vi_fair_value": float(data.get("fair_value")) if data.get("fair_value") else None,
                "vi_site_market_price": float(data.get("market_price")) if data.get("market_price") else None,
                "vi_upside_percent": float(data.get("upside_percent")) if data.get("upside_percent") else None,
                "vi_full_text": target_paragraph_text,
                "vi_data_source_url": url,
                "error": None
            }
        else:
            return {"error": f"VI.io: Could not parse specific numerical details for {ticker} from text: '{target_paragraph_text[:200]}...'"}
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 404:
            return {"error": f"VI.io: Page not found for {ticker} (404) at {url}."}
        return {"error": f"VI.io: HTTP error for {ticker}: {http_err}"}
    except requests.exceptions.RequestException as req_err:
        return {"error": f"VI.io: Request error for {ticker}: {req_err}"}
    except Exception as e:
        return {"error": f"VI.io: Unexpected error for {ticker}: {e}"}

@st.cache_data(ttl=3600)
def fetch_politician_trades(ticker: str, days_back: int = 365) -> list[dict]:
    """
    Scrapes politician trading data for a given ticker from CapitolTrades.com.
    Note: This is an experimental feature and may be unreliable due to website changes or anti-scraping measures.
    """
    url = f"https://www.capitoltrades.com/trades?asset={ticker.upper()}&pageSize=100&perPage=100"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.capitoltrades.com/'
    }
    politician_trades_list = []
    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for trade rows (often contained in anchor tags with specific classes or href patterns)
        trade_rows = soup.select("a[href^='/trades/'][class*='trade-row']");
        if not trade_rows: # Fallback if initial selector fails
            trade_rows = soup.find_all('a', href=lambda href: href and href.startswith('/trades/'))
        
        if not trade_rows:
            return [{"error": f"CT: No trade rows found for {ticker}."}]

        # Process a limited number of trade rows to avoid excessive parsing/issues
        for row_link_tag in trade_rows[:20]: # Limit to top 20 trades
            name_tag = row_link_tag.find(['div','span'], class_=lambda x: x and 'politician-name' in x)
            type_tag = row_link_tag.find(['div','span'], class_=lambda x: x and 'tx-type' in x)
            val_tag = row_link_tag.find(['div','span'], class_=lambda x: x and 'tx-value' in x)
            date_tag = row_link_tag.find(['div','span'], class_=lambda x: x and 'tx-date' in x)
            
            if all([name_tag, type_tag, val_tag, date_tag]):
                name = name_tag.text.strip()
                tx_type_text = type_tag.text.strip().lower()
                tx_type = "purchase" if "purchase" in tx_type_text else ("sale" if "sale" in tx_type_text else "other")
                val_range = val_tag.text.strip()
                date_str = date_tag.text.strip()
                val_est = 0 # Default estimated value

                # Extract numerical value from range (e.g., "$1,001 - $15,000" -> 1001)
                val_matches = re.findall(r'\$([\d,]+)', val_range)
                if val_matches:
                    try:
                        val_est = int(val_matches[0].replace(',','')) # Take the lower bound of range
                    except ValueError:
                        pass # Keep 0 if conversion fails
                
                politician_trades_list.append({
                    "politician_name": name,
                    "transaction_type": tx_type,
                    "value_range": val_range,
                    "value_estimate_lower": val_est,
                    "date_str": date_str,
                    "source_url": "https://www.capitoltrades.com" + row_link_tag['href']
                })
        
        if not politician_trades_list and trade_rows:
            return [{"error": f"CT: Found rows for {ticker} but failed to parse individual fields (HTML structure might have changed)."}]
        
        return politician_trades_list
    except requests.exceptions.Timeout:
        return [{"error": f"CT: Timeout fetching data for {ticker}."}]
    except requests.exceptions.RequestException as e:
        return [{"error": f"CT Error fetching data for {ticker}: {e}"}]
    except Exception as e:
        return [{"error": f"CT Parsing error for {ticker}: {e}"}]

# --------------------------------
# LLM Client
# --------------------------------
class ModelClient:
    def __init__(self, api_key: str, provider: str = "openai"):
        self.api_key = api_key
        self.provider = provider
        
        OPENAI_DEFAULT_MODEL = "gpt-4o"
        DEEPSEEK_DEFAULT_MODEL = "deepseek-reasoner" # Or "deepseek-coder" if preferred

        if not api_key:
            raise ValueError("API key required for ModelClient.")

        if provider == "deepseek":
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com/v1")
            self.model_name = DEEPSEEK_DEFAULT_MODEL
        elif provider == "openai":
            self.client = OpenAI(api_key=self.api_key)
            self.model_name = OPENAI_DEFAULT_MODEL
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def generate(self, prompt: str) -> str:
        """Generates a response from the configured LLM."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            final_content = "".join(chunk.choices[0].delta.content for chunk in stream if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content)
            return final_content
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"LLM Generation Error ({self.provider}, {self.model_name}): {e}")

# --------------------------------
# Agents
# --------------------------------
class PriceAgent:
    def run(self, ticker: str, price_data_slice: pd.DataFrame) -> dict:
        """Calculates SMA, RSI, and generates a price signal."""
        if price_data_slice.empty or len(price_data_slice) < 200:
            return {"ticker": ticker, "price_signal": "hold", "sma50": np.nan, "sma200": np.nan, "rsi14": np.nan}

        df = price_data_slice.copy()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["SMA200"] = df["Close"].rolling(200).mean()

        # RSI Calculation
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        
        # Avoid division by zero for RS calculation
        rs = gain / loss.replace(0, np.nan) 
        df["RSI14"] = 100 - (100 / (1 + rs))

        latest = df.iloc[-1]
        signal = "hold"

        # Signal logic based on MA crossover and RSI
        if pd.isna(latest.SMA50) or pd.isna(latest.SMA200) or pd.isna(latest.RSI14):
            signal = "hold" # Not enough data for indicators
        elif latest.SMA50 > latest.SMA200 and latest.RSI14 < 70:
            signal = "buy"
        elif latest.SMA50 < latest.SMA200 and latest.RSI14 > 30:
            signal = "sell"
        
        return {
            "ticker": ticker,
            "sma50": float(latest.SMA50) if pd.notna(latest.SMA50) else np.nan,
            "sma200": float(latest.SMA200) if pd.notna(latest.SMA200) else np.nan,
            "rsi14": float(latest.RSI14) if pd.notna(latest.RSI14) else np.nan,
            "price_signal": signal
        }

class MomentumAgent:
    def run(self, ticker: str, price_data_slice: pd.DataFrame) -> dict:
        """Calculates 1-month and 12-month momentum and generates a signal."""
        if price_data_slice.empty or len(price_data_slice) < 252: # Need at least 1 year of data for 12-month momentum
            return {"ticker": ticker, "momentum_signal": "hold", "momentum_1m": 0, "momentum_12m": 0}

        df = price_data_slice
        P_t = df.Close.iloc[-1] # Current price
        
        # Price 1 month ago (approx 21 trading days)
        P_1m = df.Close.shift(21).iloc[-1] if len(df) > 21 else np.nan
        # Price 12 months ago (approx 252 trading days)
        P_12m = df.Close.shift(252).iloc[-1] if len(df) > 252 else np.nan

        m1 = (P_t / P_1m) - 1 if pd.notna(P_1m) and P_1m != 0 else 0.0
        m12 = (P_t / P_12m) - 1 if pd.notna(P_12m) and P_12m != 0 else 0.0

        signal = "hold"
        # Simple momentum strategy: buy if both short and long term momentum are positive, sell if negative
        if m12 > 0.01 and m1 > 0.01: # Small positive threshold
            signal = "buy"
        elif m12 < -0.01 and m1 < -0.01: # Small negative threshold
            signal = "sell"
        
        return {
            "ticker": ticker,
            "momentum_1m": float(m1),
            "momentum_12m": float(m12),
            "momentum_signal": signal
        }

class VolatilityAgent:
    def run(self, ticker: str, data: dict, price_data_slice: pd.DataFrame = None) -> dict:
        """Analyzes stock beta and historical volatility to generate a signal and weight."""
        beta = data.get("ticker_info", {}).get("beta", 1.0)
        if beta is None: beta = 1.0 # Default to 1 if beta is not found

        # Signal based on beta: higher beta means more volatile, potentially "sell" if seeking stability
        sig = "sell" if beta > 1.5 else ("buy" if beta < 0.8 else "hold")

        ann_vol = np.nan
        vol_weight = 0.0 # Weight for portfolio allocation (inverse of volatility)

        if price_data_slice is not None and not price_data_slice.empty and len(price_data_slice) > 1:
            # Calculate daily returns for volatility
            ret = np.log(price_data_slice.Close / price_data_slice.Close.shift(1)).dropna()
            if not ret.empty:
                ann_vol = float(ret.std() * np.sqrt(252)) # Annualized daily volatility
                vol_weight = float(1 / ann_vol) if ann_vol > 0 else 0.0 # Inverse volatility weighting
        
        return {
            "ticker": ticker,
            "beta": beta,
            "annual_vol": ann_vol,
            "vol_weight": vol_weight,
            "volatility_signal": sig
        }

class SentimentAgent:
    def __init__(self, client):
        self.client = client

    def run(self, ticker: str, data: dict) -> dict:
        """
        Analyzes the sentiment of recent news articles using an LLM.
        Scores from -1.0 (very negative) to +1.0 (very positive).
        """
        news_items_from_bundle = data.get("news", [])
        overall_news_fetch_error = data.get("news_fetch_status_error")

        if overall_news_fetch_error:
            return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": overall_news_fetch_error}

        valid_news_items = [item for item in news_items_from_bundle if isinstance(item, dict) and "error" not in item]
        if not valid_news_items:
            # Check if the news list itself contained an error message
            if news_items_from_bundle and isinstance(news_items_from_bundle[0], dict) and "error" in news_items_from_bundle[0]:
                return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": news_items_from_bundle[0].get("error")}
            return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": "No valid news articles to process."}

        content_for_llm = []
        company_name_overall = data.get("ticker_info",{}).get('longName', ticker)

        # Prepare news snippets for the LLM
        for item in valid_news_items[:7]: # Limit to top 7 articles for token efficiency
            title = item.get('title', '')
            publisher = item.get('publisher', '')
            description = item.get('description', '')
            content = item.get('content_snippet', '') # Use the content_snippet from NewsAPI

            text_snippet = f"Headline: {title}"
            if content and isinstance(content, str) and len(content) > 10: # Prefer content if available and substantial
                text_snippet += f" | Content Snippet: {content.replace('[+... chars]', '').strip()}" # Remove common truncation markers
            elif description and isinstance(description, str):
                text_snippet += f" | Description: {description.strip()}"
            
            if publisher and publisher != 'N/A':
                text_snippet += f" (Source: {publisher} via {item.get('source_api', 'Unknown')})"
            
            content_for_llm.append(text_snippet)

        if not content_for_llm:
            return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": "No processable news content for LLM."}

        prompt = (f"Analyze sentiment for {company_name_overall} ({ticker}) based on the following recent news snippets. "
                  "Provide a single numeric score between -1.0 (very negative), 0.0 (neutral), and +1.0 (very positive). "
                  "Output only the number, with up to two decimal places, e.g., '0.75' or '-0.30'.\n\nNews:\n" + 
                  "\n".join(f"- {c}" for c in content_for_llm))
        
        score = 0.0
        llm_error_msg = None
        try:
            response_text = self.client.generate(prompt).strip()
            if response_text.startswith("Error:"):
                llm_error_msg = response_text
            else:
                # Attempt to parse the number from the response
                match = re.search(r"[-+]?\d*\.\d+|\d+", response_text)
                if match:
                    score = float(match.group(0))
                    score = max(-1.0, min(1.0, score)) # Clamp score between -1.0 and 1.0
                else:
                    llm_error_msg = f"LLM did not return a parsable number: '{response_text[:50]}...'"
        except Exception as e:
            llm_error_msg = f"LLM call failed for sentiment: {str(e)[:150]}"

        final_error_message_for_sentiment = llm_error_msg
        # Aggregate news fetch errors with LLM errors for clarity
        if overall_news_fetch_error and ("Error" in overall_news_fetch_error or "failed" in overall_news_fetch_error.lower()) :
            final_error_message_for_sentiment = f"News: {overall_news_fetch_error}" + (f" | LLM: {llm_error_msg}" if llm_error_msg else "")

        # Signal logic based on sentiment score
        sig = "buy" if score > 0.25 and not llm_error_msg else ("sell" if score < -0.25 and not llm_error_msg else "hold")
        
        return {
            "ticker": ticker,
            "sentiment_score": score,
            "sentiment_signal": sig,
            "sentiment_error": final_error_message_for_sentiment
        }

class NewsSummaryAgent:
    def __init__(self, client):
        self.client = client

    def run(self, ticker: str, data: dict) -> dict:
        """
        Generates a concise summary of recent news using an LLM.
        Pulls from both Yahoo Finance and NewsAPI sources.
        """
        news_items = data.get("news", [])
        company_name = data.get("ticker_info", {}).get('longName', ticker)
        
        if not news_items or (isinstance(news_items[0], dict) and "error" in news_items[0]):
            return {"ticker": ticker, "news_summary": "No news available for summary.", "news_summary_error": data.get("news_fetch_status_error", "No news fetched.")}

        # Select a balanced set of recent articles for summary
        # Aim for up to 5 from Yahoo and up to 5 from NewsAPI
        yahoo_news_for_summary = [item for item in news_items if item.get('source_api') == 'Yahoo Finance' and "error" not in item][:5]
        newsapi_news_for_summary = [item for item in news_items if item.get('source_api') == 'NewsAPI.org' and "error" not in item][:5]
        
        selected_news_for_summary = []
        # Alternate picking from each list to get a mix of sources, limited to total 7
        for i in range(max(len(yahoo_news_for_summary), len(newsapi_news_for_summary))):
            if i < len(yahoo_news_for_summary):
                selected_news_for_summary.append(yahoo_news_for_summary[i])
            if i < len(newsapi_news_for_summary):
                selected_news_for_summary.append(newsapi_news_for_summary[i])
        
        # Limit to top 7 unique articles for the LLM prompt to manage token limits
        final_news_snippets = []
        seen_titles = set()
        for item in selected_news_for_summary:
            if len(final_news_snippets) >= 7: # Hard limit for LLM prompt
                break
            title = item.get('title', '')
            if title in seen_titles: # Basic deduplication by title
                continue
            seen_titles.add(title)
            
            description = item.get('description', '')
            content_snippet = item.get('content_snippet', '').replace('[+... chars]', '').strip()
            
            text_to_add = f"Title: {title}"
            if content_snippet: # Prefer content if available and substantial
                text_to_add += f" | Content: {content_snippet}"
            elif description: # Fallback to description
                text_to_add += f" | Description: {description}"
            
            final_news_snippets.append(text_to_add)

        if not final_news_snippets:
            return {"ticker": ticker, "news_summary": "No sufficient news content for summary.", "news_summary_error": "No articles with content/description for summary."}

        prompt = (f"Provide a concise summary paragraph (max 200 words) about what is happening with {company_name} ({ticker}) based on the following recent news articles. "
                  "Focus on key events, trends, and impacts. If no significant events or if the news is generic, state that clearly.\n\nNews Articles:\n" + 
                  "\n".join(f"- {s}" for s in final_news_snippets))
        
        summary = "Could not generate news summary."
        error_msg = None
        try:
            response_text = self.client.generate(prompt).strip()
            if response_text.startswith("Error:"):
                error_msg = response_text
            else:
                summary = response_text
        except Exception as e:
            error_msg = f"LLM call for news summary failed: {str(e)[:150]}"
        
        return {"ticker": ticker, "news_summary": summary, "news_summary_error": error_msg}


class FundamentalsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        """Calculates fundamental metrics (FCF Yield, Piotroski Score) and generates a signal."""
        s = data.get("ticker_info", {})
        mcap = s.get("marketCap") or 1 # Avoid division by zero
        fcf = s.get("freeCashflow") or 0
        roe = s.get("returnOnEquity") or 0 # Return on Equity
        de = s.get("debtToEquity") # Debt to Equity. Can be None.
        de = 1000 if de is None else de # Assign a high value if None to make it "bad" for score

        fcy = fcf / mcap if mcap != 0 else 0 # Free Cash Flow Yield

        # Simplified Piotroski-like score (can be expanded)
        # 1 point for positive ROE, 1 point for low Debt/Equity, 1 point for positive FCF
        piotroski_score = sum([
            roe > 0.01, # Positive Return on Equity
            de < 100,   # Debt to Equity ratio less than 100 (less debt)
            fcf > 0     # Positive Free Cash Flow
        ])

        signal = "hold"
        if piotroski_score >= 2: # At least 2 out of 3 positive indicators
            signal = "buy"
        elif piotroski_score == 0: # Zero positive indicators
            signal = "sell"
        
        return {
            "ticker": ticker,
            "fcf_yield": float(fcy),
            "piotroski_score": piotroski_score,
            "fund_signal": signal
        }

class ValuationAgent:
    def run(self, ticker: str, data: dict) -> dict:
        """
        Provides valuation signals based on P/E ratio and a simple DCF-like free cash flow model.
        """
        stats = data.get("ticker_info", {})
        price_history_df = data.get("price_history")
        
        # Get current price, prioritizing ticker_info then price history
        price = stats.get("currentPrice")
        if price is None and price_history_df is not None and not price_history_df.empty:
            price = price_history_df["Close"].iloc[-1]
        
        if price is None or price <= 0:
            return {"ticker": ticker, "forward_pe": None, "relative_pe_signal": "hold", "dcf_fair_price": np.nan, "dcf_signal": "hold", "valuation_error": "Current price not available for valuation."}

        # Forward P/E based signal
        pe = stats.get("forwardPE")
        rel_sig = "hold"
        if pe is not None and pe > 0:
            rel_sig = "buy" if pe < 15 else "sell" if pe > 25 else "hold" # Example thresholds

        # Simple DCF-like valuation based on Free Cash Flow Yield
        fcf = stats.get("freeCashflow")
        mcap = stats.get("marketCap")

        fcy = (fcf / mcap) if fcf is not None and mcap is not None and mcap != 0 else 0.0
        
        # Very simplistic "fair price" based on FCF yield growth implied
        # Assuming FCF is a proxy for earnings power, a higher yield might imply undervaluation.
        # This is a very rough heuristic.
        fair_price = price * (1 + fcy) # If FCF is positive, fair price is higher than current price.
        dcf_sig = "hold"
        if fair_price > price * 1.15: # 15% upside potential
            dcf_sig = "buy"
        elif fair_price < price * 0.85: # 15% downside potential
            dcf_sig = "sell"
            
        return {
            "ticker": ticker,
            "forward_pe": pe,
            "relative_pe_signal": rel_sig,
            "dcf_fair_price": float(fair_price),
            "dcf_signal": dcf_sig,
            "valuation_error": None
        }

class AnalystRatingAgent:
    def run(self, ticker: str, data: dict) -> dict:
        """Processes analyst recommendations and target prices from yfinance."""
        ticker_info_res = data.get("ticker_info", {})
        price_history_df = data.get("price_history")

        current_price = ticker_info_res.get("currentPrice")
        if current_price is None and price_history_df is not None and not price_history_df.empty:
            current_price = price_history_df["Close"].iloc[-1]
        
        if current_price is None or current_price <= 0:
            return {"ticker": ticker, "analyst_buy_pct_inferred": 0.5, "target_upside": 0.0,
                    "yfinance_recommendation": "N/A", "analyst_signal": "hold", "analyst_error": "Current price not available."}

        target_mean_price = ticker_info_res.get("targetMeanPrice")
        recommendation = str(ticker_info_res.get("recommendationKey", "hold")).lower()
        
        upside = 0.0
        if target_mean_price and current_price and current_price > 0:
            try:
                upside = (float(target_mean_price) / float(current_price)) - 1
            except (ValueError, TypeError):
                upside = 0.0 # Handle cases where values might not be convertible to float

        sig = "hold"
        # Logic for analyst signal
        if recommendation in ["buy", "strong_buy"] and upside > 0.10: # Buy if recommendation is positive and significant upside
            sig = "buy"
        elif recommendation == "buy" and upside > 0.05: # Smaller upside for plain "buy"
            sig = "buy"
        elif recommendation in ["sell", "strong_sell", "underperform"] and upside < -0.05: # Sell if recommendation is negative and significant downside
            sig = "sell"
        elif upside > 0.20: # Strong buy signal if target price implies very high upside, regardless of explicit recommendation key
            sig = "buy"
        elif upside < -0.15: # Strong sell signal if target price implies very high downside
            sig = "sell"
            
        # Infer a "buy percentage" for display or further use (e.g., portfolio weighting)
        buy_pct_inferred = {"strong_buy": 0.9, "buy": 0.7, "hold": 0.5, "underperform": 0.3, "sell": 0.1}.get(recommendation, 0.5)

        return {
            "ticker": ticker,
            "analyst_buy_pct_inferred": buy_pct_inferred,
            "target_upside": float(upside),
            "yfinance_recommendation": recommendation,
            "analyst_signal": sig
        }

class SECFilingAgent:
    def run(self, ticker: str, data: dict) -> dict:
        """
        Analyzes SEC Form 4 (insider transactions) for significant buy/sell activity.
        Also provides metadata on other recent filings.
        """
        # Data from `sec_all_filings_raw` which is already fetched by the orchestrator
        all_filings_raw = data.get("sec_all_filings_raw", [])
        
        error_from_fetch = None
        if not all_filings_raw: 
            error_from_fetch = f"SEC Filings: No filings data returned for {ticker}."
        elif isinstance(all_filings_raw[0], dict) and "error" in all_filings_raw[0]:
            error_from_fetch = all_filings_raw[0].get("error")
            
        if error_from_fetch:
            return {
                "ticker": ticker,
                "sec_net_insider_shares_1y": 0,
                "sec_insider_buy_value_1y": 0,
                "sec_insider_sell_value_1y": 0,
                "sec_filings_signal": "hold",
                "sec_filings_error": error_from_fetch,
                "sec_recent_form4_transactions": [], # Ensure these are always present, even if empty
                "sec_other_recent_filings": []
            }

        net_shares = 0
        buy_value = 0
        sell_value = 0
        form4_transactions_processed = []
        other_filings_metadata = []

        for filing in all_filings_raw:
            if not isinstance(filing, dict) or "error" in filing:
                continue # Skip non-dictionary items or error dictionaries within the list

            if filing.get("is_form4_transaction"):
                form4_transactions_processed.append(filing)
                # 'P' for purchase, 'S' for sale, 'A' for acquired, 'D' for disposed
                if filing.get("transaction_code") == "P" and filing.get("acq_disp_code") == "A":
                    shares = filing.get("shares", 0.0)
                    price = filing.get("price_per_share")
                    net_shares += shares
                    if price is not None and shares != 0:
                        buy_value += shares * price
                elif filing.get("transaction_code") == "S" and filing.get("acq_disp_code") == "D":
                    shares = filing.get("shares", 0.0)
                    price = filing.get("price_per_share")
                    net_shares -= shares
                    if price is not None and shares != 0:
                        sell_value += shares * price
            else:
                other_filings_metadata.append(filing)
            
        signal = "hold"
        # Simple signal: significant net insider buying or selling volume/value
        if net_shares > 2000 or buy_value > 200000: # Thresholds for "significant" insider activity
            signal = "buy" 
        elif net_shares < -2000 or sell_value > 200000:
            signal = "sell"
            
        return {
            "ticker": ticker,
            "sec_net_insider_shares_1y": int(net_shares),
            "sec_insider_buy_value_1y": round(buy_value, 2),
            "sec_insider_sell_value_1y": round(sell_value, 2),
            "sec_filings_signal": signal,
            "sec_filings_error": None, # Clear error if processing was successful
            "sec_recent_form4_transactions": form4_transactions_processed[:10], # Limit to top 10 for display
            "sec_other_recent_filings": other_filings_metadata[:10] # Limit to top 10 for display
        }

class InstitutionalHoldingsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        """
        Summarizes current institutional holdings data from yfinance.
        Note: This agent provides a snapshot of holdings, not historical changes in detail.
        Changes would require parsing historical 13F filings, which is more complex.
        """
        institutional_holdings_data = data.get("institutional_holdings", [])
        
        # Check for errors from the fetcher function
        error = None
        if institutional_holdings_data and isinstance(institutional_holdings_data[0], dict) and "error" in institutional_holdings_data[0]:
            error = institutional_holdings_data[0]["error"]
            return {
                "ticker": ticker,
                "inst_num_holders": 0,
                "inst_total_shares_held": 0,
                "inst_total_pct_out": 0.0,
                "inst_holdings_signal": "hold",
                "inst_holdings_error": error,
                "inst_top_holders": []
            }

        num_holders = 0
        total_shares_held = 0
        total_pct_out = 0.0
        top_holders = []

        if institutional_holdings_data:
            num_holders = len(institutional_holdings_data)
            try:
                # Sum up shares and percentage out from valid dictionaries
                total_shares_held = sum(d.get('Shares', 0) for d in institutional_holdings_data if isinstance(d, dict))
                total_pct_out = sum(d.get('% Out', 0.0) for d in institutional_holdings_data if isinstance(d, dict))
                
                # Sort to get top holders
                top_holders = sorted(institutional_holdings_data, key=lambda x: x.get('Shares', 0), reverse=True)[:10]

            except Exception as e:
                error = f"Error processing institutional holdings data: {e}"

        signal = "hold"
        # Example signal logic: highly held by institutions might be a buy, very low might be a sell
        if total_pct_out > 0.50: # More than 50% held by institutions
            signal = "buy"
        elif total_pct_out < 0.05 and num_holders > 0: # Very low institutional ownership but some holders exist
            signal = "sell"

        return {
            "ticker": ticker,
            "inst_num_holders": num_holders,
            "inst_total_shares_held": int(total_shares_held),
            "inst_total_pct_out": float(total_pct_out),
            "inst_holdings_signal": signal,
            "inst_holdings_error": error,
            "inst_top_holders": top_holders
        }


class PoliticianFilingsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        """
        Analyzes politician trading data for a given ticker from CapitolTrades.com (scraped data).
        """
        trades = data.get("politician_trades", [])
        net_value_estimate = 0
        buy_count = 0
        sell_count = 0
        error = None

        # Check if the trades data itself contains an error message from the fetcher
        if trades and isinstance(trades, list) and len(trades)>0 and isinstance(trades[0], dict) and "error" in trades[0]:
            error = trades[0]["error"]
        elif trades:
            for trade in trades:
                if isinstance(trade, dict): # Ensure it's a dictionary
                    value = trade.get("value_estimate_lower", 0)
                    if trade.get("transaction_type") == "purchase":
                        net_value_estimate += value
                        buy_count += 1
                    elif trade.get("transaction_type") == "sale":
                        net_value_estimate -= value
                        sell_count += 1
        
        signal = "hold"
        # Simple signal based on net count of buys vs. sells (requires more than 1 trade to signal)
        if not error: # Only signal if no data errors
            if buy_count > sell_count and buy_count > 1 : # More buys than sells, and at least 2 buys
                signal = "buy"
            elif sell_count > buy_count and sell_count > 1: # More sells than buys, and at least 2 sells
                signal = "sell"
                
        return {
            "ticker": ticker,
            "politician_net_trade_value_estimate": net_value_estimate,
            "politician_buy_tx_count": buy_count,
            "politician_sell_tx_count": sell_count,
            "politician_filings_signal": signal,
            "politician_data_error": error # Propagate error from fetcher
        }

class ValueInvestingIOAgent:
    def run(self, ticker: str, data: dict) -> dict:
        """
        Analyzes fair value data scraped from ValueInvesting.io and generates a signal.
        """
        vi_data = data.get("value_investing_io_data", {})
        error = vi_data.get("error")
        fair_value = vi_data.get("vi_fair_value")
        site_market_price = vi_data.get("vi_site_market_price")
        upside_percent = vi_data.get("vi_upside_percent")
        valuation_date = vi_data.get("vi_valuation_date")
        full_text = vi_data.get("vi_full_text")
        
        signal = "hold"

        # Get current price from primary ticker info or price history for comparison
        current_price_yf = data.get("ticker_info", {}).get("currentPrice")
        if current_price_yf is None and data.get("price_history") is not None and not data["price_history"].empty:
            current_price_yf = data["price_history"]["Close"].iloc[-1]

        if not error and fair_value is not None and current_price_yf is not None and current_price_yf > 0:
            margin_of_safety = 0.15 # 15% margin of safety
            
            # Use upside_percent if available (it's often directly calculated by the site)
            if upside_percent is not None:
                if upside_percent > (margin_of_safety * 100 + 5): # e.g., > 20% upside
                    signal = "buy"
                elif upside_percent < -(margin_of_safety * 100 + 5): # e.g., < -20% downside
                    signal = "sell"
            else: # Fallback to calculating from fair_value and current price
                if current_price_yf < fair_value * (1 - margin_of_safety): # Current price significantly below fair value
                    signal = "buy"
                elif current_price_yf > fair_value * (1 + margin_of_safety): # Current price significantly above fair value
                    signal = "sell"
        
        return {
            "ticker": ticker,
            "vi_fair_value_estimate": fair_value,
            "vi_site_market_price": site_market_price,
            "vi_upside_percent": upside_percent,
            "vi_valuation_date": valuation_date,
            "vi_valuation_text_display": full_text,
            "vi_signal": signal,
            "vi_data_error": error
        }

class PortfolioAgent:
    """
    Combines signals from various agents using predefined weights to produce a final decision.
    """
    WEIGHTS = {
        "price": 1.0,           # Price action (SMA, RSI)
        "momentum": 0.8,        # Price momentum
        "volatility": 0.3,      # Beta, Inverse Annual Volatility
        "sentiment": 0.6,       # LLM-based news sentiment
        "fund": 0.9,            # Piotroski-like score, FCF Yield
        "valuation_dcf": 0.5,   # Simple DCF-like estimate
        "valuation_pe": 0.5,    # Relative P/E valuation
        "sec_filings": 0.6,     # SEC Form 4 insider activity
        "inst_holdings": 0.3,   # Institutional ownership snapshot
        "analyst": 0.7,         # Analyst ratings and target price upside
        "politician_filings": 0.4, # Politician trade activity (experimental)
        "vi_signal": 0.8        # ValueInvesting.io fair value (experimental)
    }

    def run(self, ticker: str, signals: list[dict], agent_weights: dict = None) -> dict:
        """
        Aggregates signals from individual agents into a composite score and a final decision.
        """
        current_weights = agent_weights or self.WEIGHTS
        total_weighted_score = 0
        sum_of_weights_used = 0
        
        # Flatten list of signal dictionaries into one dictionary for easy lookup
        agg_signals = {}
        for s_dict in signals:
            if isinstance(s_dict, dict):
                agg_signals.update(s_dict)
        
        # Map signal keys to their corresponding weights
        signal_map = {
            "price_signal": "price",
            "momentum_signal": "momentum",
            "volatility_signal": "volatility",
            "sentiment_signal": "sentiment",
            "fund_signal": "fund",
            "dcf_signal": "dcf_valuation", # Changed from valuation_dcf to align with key
            "relative_pe_signal": "valuation_pe",
            "sec_filings_signal": "sec_filings", 
            "inst_holdings_signal": "inst_holdings",
            "analyst_signal": "analyst", 
            "politician_filings_signal": "politician_filings",
            "vi_signal": "vi_signal"
        }

        for signal_key, weight_key in signal_map.items():
            signal_value = agg_signals.get(signal_key)
            weight = current_weights.get(weight_key, 0) # Get weight, default to 0 if not found

            # Only consider signals that are 'buy', 'hold', or 'sell' and have a positive weight
            if signal_value and weight > 0 and signal_value in ["buy", "hold", "sell"]:
                raw_score = {"buy": 1, "hold": 0, "sell": -1}.get(signal_value, 0)
                total_weighted_score += raw_score * weight
                sum_of_weights_used += weight
        
        # Calculate composite score
        composite_score = (total_weighted_score / sum_of_weights_used) if sum_of_weights_used else 0.0

        # Determine final decision based on composite score thresholds
        final_decision = "buy" if composite_score > 0.15 else ("sell" if composite_score < -0.15 else "hold")
        
        return {
            "ticker": ticker,
            "composite_score": composite_score,
            "final_decision": final_decision
        }

# --------------------------------
# Orchestrator for Live Analysis
# --------------------------------
def run_live_analysis(tickers, history_years, llm_client, configs):
    """
    Orchestrates the fetching of data and running of all agents for live analysis.
    """
    results = {}
    for t in tickers:
        st.write(f"Running analysis for {t}...") # Debugging output

        price_history_full = fetch_price_history(t, period=f"{history_years}y")
        if price_history_full.empty:
            results[t] = {"error": f"Price history unavailable for {t}.", "ticker": t, "final_decision":"error", "composite_score":0}
            st.error(f"Skipping {t} due to price history error.")
            continue
        
        ticker_info = fetch_ticker_info(t)
        if not ticker_info:
            results[t] = {"error": f"Ticker info unavailable for {t}.", "ticker": t, "final_decision":"error", "composite_score":0}
            st.error(f"Skipping {t} due to ticker info error.")
            continue

        current_price_for_ticker = ticker_info.get("currentPrice") or (price_history_full["Close"].iloc[-1] if not price_history_full.empty else None)
        company_name_for_news = ticker_info.get('longName', ticker_info.get('shortName', t))
        
        combined_news_data_list = []
        news_fetch_status_messages = []

        # Fetch news if sentiment analysis is enabled
        if configs["use_sentiment"]:
            yfinance_news = fetch_enriched_news(t, ticker_info)
            if yfinance_news and not (isinstance(yfinance_news[0], dict) and "error" in yfinance_news[0]):
                combined_news_data_list.extend(yfinance_news)
            elif yfinance_news and isinstance(yfinance_news[0], dict) and "error" in yfinance_news[0]:
                news_fetch_status_messages.append(f"Yahoo News: {yfinance_news[0]['error']}")
            
            if llm_client and st.secrets.get("NEWSAPI_KEY"): # Only try NewsAPI if key is present
                newsapi_articles = fetch_comprehensive_news_from_api(t, company_name_for_news, lookback_days=30)
                if newsapi_articles and not (isinstance(newsapi_articles[0], dict) and "error" in newsapi_articles[0]):
                    combined_news_data_list.extend(newsapi_articles)
                elif newsapi_articles and isinstance(newsapi_articles[0], dict) and "error" in newsapi_articles[0]:
                    news_fetch_status_messages.append(f"NewsAPI: {newsapi_articles[0]['error']}")
            elif configs["use_sentiment"]: # If sentiment is on but NewsAPI key is missing
                news_fetch_status_messages.append("NewsAPI Key not configured.")
        
        # Deduplicate news articles based on URL
        seen_urls = set()
        deduplicated_news = []
        for news_item in combined_news_data_list:
            if isinstance(news_item, dict) and "error" not in news_item:
                url = news_item.get('link') or news_item.get('url') # Try both 'link' and 'url'
                if url and url not in seen_urls:
                    deduplicated_news.append(news_item)
                    seen_urls.add(url)
        # Sort deduplicated news by date
        if deduplicated_news:
            deduplicated_news.sort(key=lambda x: x.get('publish_datetime_utc') or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        
        news_fetch_status_for_bundle = " | ".join(news_fetch_status_messages) if news_fetch_status_messages else "News fetch OK"
        if not deduplicated_news and not news_fetch_status_messages and configs["use_sentiment"]:
            news_fetch_status_for_bundle = "No news articles found from enabled sources."

        # Fetch optional data based on configs
        politician_trades_list = fetch_politician_trades(t) if configs["use_politician_filings"] else []
        
        # Bundle all fetched data for agents
        data_bundle = {
            "price_history": price_history_full,
            "ticker_info": ticker_info,
            "news": deduplicated_news, # Pass deduplicated and sorted news
            "news_fetch_status_error": news_fetch_status_for_bundle if "Error" in news_fetch_status_for_bundle or "failed" in news_fetch_status_for_bundle.lower() or "No news" in news_fetch_status_for_bundle else None,
            "politician_trades": politician_trades_list,
            "value_investing_io_data": fetch_value_investing_io_data(t) if configs["use_value_trades"] else \
                                        {"error": "VI.io: Skipped by user config."},
            "institutional_holdings": fetch_inst_filings(t) if configs["use_filings"] else [], # Yahoo institutional holdings
            "sec_all_filings_raw": fetch_all_sec_filings(t, lookback_days=365) if configs["use_filings"] else [] # SEC filings (for Form 4 and others)
        }
        
        # Initialize and run agents based on enabled features
        all_agents_instances = [
            PriceAgent(),
            MomentumAgent(),
            VolatilityAgent(),
            FundamentalsAgent(),
            ValuationAgent(),
            AnalystRatingAgent()
        ]
        if configs["use_sentiment"] and llm_client:
            all_agents_instances.append(SentimentAgent(llm_client))
            all_agents_instances.append(NewsSummaryAgent(llm_client)) # Add the News Summary Agent
        if configs["use_filings"]: 
            all_agents_instances.append(SECFilingAgent()) # Handles Form 4 and other SEC filings
            all_agents_instances.append(InstitutionalHoldingsAgent()) # Handles Yahoo institutional holdings
        if configs["use_politician_filings"]:
            all_agents_instances.append(PoliticianFilingsAgent())
        if configs["use_value_trades"]: 
            all_agents_instances.append(ValueInvestingIOAgent())

        agent_results_list = []
        for agent_instance in all_agents_instances:
            agent_name = agent_instance.__class__.__name__
            try:
                # Special handling for agents that need price_history_full or data_bundle directly
                if isinstance(agent_instance, (PriceAgent, MomentumAgent)):
                    res_agent = agent_instance.run(t, data_bundle["price_history"])
                elif isinstance(agent_instance, VolatilityAgent):
                    res_agent = agent_instance.run(t, data_bundle, data_bundle["price_history"])
                elif isinstance(agent_instance, (SECFilingAgent, InstitutionalHoldingsAgent, PoliticianFilingsAgent, ValueInvestingIOAgent, NewsSummaryAgent, SentimentAgent)):
                    res_agent = agent_instance.run(t, data_bundle) # Pass data_bundle as is
                else: # Generic case for other agents that just need the bundle
                    res_agent = agent_instance.run(t, data_bundle)
                
                agent_results_list.append(res_agent)
            except Exception as e:
                # Log agent-specific errors
                agent_error_key = agent_name.lower().replace("agent","") + "_error"
                default_signal_key_name = agent_name.lower().replace("agent","") + "_signal" # Attempt to create a signal key
                agent_results_list.append({default_signal_key_name: "error", agent_error_key: f"Agent {agent_name} error: {str(e)[:150]}"})
                st.warning(f"Error running {agent_name} for {t}: {e}") # Show warning in UI for agent errors

        final_decision = PortfolioAgent().run(t, agent_results_list)

        # Compile results for display
        current_result_dict = {
            "ticker": t,
            "current_price_display": current_price_for_ticker,
            "market_cap_display": ticker_info.get("marketCap"),
            "industry_display": ticker_info.get("industry"),
            "sector_display": ticker_info.get("sector"),
            "ticker_info": ticker_info, # Full ticker info can be useful for debugging
            "news_headlines_for_popover": [
                # Display format: Date - Title (Publisher via Source API) [Link] - Content Snippet...
                f"{n.get('publish_time_readable','N/A')} - {n.get('title', 'N/A')} ({n.get('publisher','N/A')} via {n.get('source_api','Unknown')}) [Link]({n.get('link','#')})"
                + (f" - {n.get('content_snippet', n.get('description', ''))[:150]}..." if n.get('content_snippet') or n.get('description') else "")
                for n in deduplicated_news[:10] # Limit news headlines for popover
            ],
            "politician_trades_for_popover": [pt for pt in politician_trades_list[:5] if isinstance(pt, dict) and "error" not in pt], # Limit for display
            "news_status_display": news_fetch_status_for_bundle # Overall status of news fetching
        }
        
        # Merge all agent results into the main dictionary
        for res_dict in agent_results_list:
            if isinstance(res_dict, dict):
                current_result_dict.update(res_dict)
        
        current_result_dict.update(final_decision) # Add composite score and final decision
        results[t] = current_result_dict
    return results

# --------------------------------
# Backtesting Engine 
# --------------------------------
def run_backtest(ticker, start_date, end_date, initial_capital, llm_client_placeholder, backtest_agent_weights):
    """
    Simulates a trading strategy for a single ticker over a historical period.
    Note: LLM-based agents (Sentiment, News Summary) are disabled in backtest for performance/consistency.
    """
    st.write(f"Preparing backtest for {ticker} from {start_date} to {end_date}...") # Debugging output

    s_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    # Fetch enough history for all indicators (e.g., 200-day SMA, 12-month momentum)
    fetch_start_date = (s_date_obj - pd.DateOffset(months=18)).strftime("%Y-%m-%d") # 18 months for 200-day SMA and 12-month momentum

    full_price_history = fetch_price_history(ticker, period=None, interval="1d") # Fetch all available daily data
    if full_price_history.empty:
        return {"error": "Backtest failed: Price history empty."}, pd.DataFrame()
    
    # Filter price history to the relevant backtest period plus lookback for indicators
    price_history = full_price_history[(full_price_history.index >= pd.to_datetime(fetch_start_date)) & (full_price_history.index <= pd.to_datetime(end_date))].copy()

    # Check if there's enough data within the actual backtest period for trading decisions
    if price_history.empty or len(price_history[price_history.index >= pd.to_datetime(start_date)]) < 2:
        return {"error": "Backtest failed: Not enough data in selected range for meaningful trading."}, pd.DataFrame()
    
    # Static data for backtest (info that doesn't change daily)
    ticker_info_for_backtest = fetch_ticker_info(ticker)
    data_bundle_static = {"ticker_info": ticker_info_for_backtest} # Used by agents like VolatilityAgent for beta

    # Initialize agents used in backtesting
    price_agent = PriceAgent()
    momentum_agent = MomentumAgent()
    volatility_agent = VolatilityAgent() # Volatility is calculated on current slice, but beta comes from static info
    portfolio_agent = PortfolioAgent() # Uses the provided backtest_agent_weights

    portfolio_log = []
    cash = initial_capital
    shares_held = 0
    portfolio_value = initial_capital

    # Iterate through each trading day within the specified backtest period
    backtest_run_dates = price_history[price_history.index >= pd.to_datetime(start_date)].index

    for current_date in backtest_run_dates:
        # Slice data up to the current date for indicator calculation
        data_slice = price_history[price_history.index <= current_date]
        
        current_price_point = data_slice.Close.iloc[-1] if not data_slice.empty else (portfolio_value / shares_held if shares_held else 0)

        # Ensure enough historical data for indicators to be calculated
        if data_slice.empty or len(data_slice) < 252: # Need enough data for 12-month momentum and 200-SMA
            portfolio_log.append({
                "date": current_date,
                "cash": cash,
                "shares_held": shares_held,
                "price": current_price_point,
                "portfolio_value": portfolio_value,
                "signal": "hold (insufficient data)",
                "composite_score": 0.0
            })
            continue # Skip trading if not enough historical data for indicators

        current_price = data_slice.Close.iloc[-1]

        # Run selected agents
        pa_res = price_agent.run(ticker, data_slice)
        ma_res = momentum_agent.run(ticker, data_slice)
        va_res = volatility_agent.run(ticker, data_bundle_static, data_slice) # Pass data_slice for volatility calc

        # Combine signals using the portfolio agent with backtest-specific weights
        final_decision_obj = portfolio_agent.run(ticker, [pa_res, ma_res, va_res], agent_weights=backtest_agent_weights)
        final_decision = final_decision_obj["final_decision"]

        # Execute trades
        if final_decision == "buy" and cash > current_price:
            shares_to_buy = cash / current_price
            shares_held += shares_to_buy
            cash = 0 # All cash invested
        elif final_decision == "sell" and shares_held > 0:
            cash += shares_held * current_price
            shares_held = 0 # All shares sold
        
        # Update portfolio value
        portfolio_value = cash + shares_held * current_price

        # Log daily portfolio state
        portfolio_log.append({
            "date": current_date,
            "cash": cash,
            "shares_held": shares_held,
            "price": current_price,
            "portfolio_value": portfolio_value,
            "signal": final_decision,
            "composite_score": final_decision_obj["composite_score"]
        })
    
    log_df = pd.DataFrame(portfolio_log)
    if not log_df.empty:
        log_df.set_index("date", inplace=True)

    if log_df.empty or len(log_df) < 2:
        return {"message":"Backtest log too short to calculate performance metrics. Check data range and ticker."}, pd.DataFrame()

    # Calculate performance metrics
    total_return = (log_df["portfolio_value"].iloc[-1] / initial_capital - 1) * 100
    
    num_days = (log_df.index[-1] - log_df.index[0]).days
    num_years = num_days / 365.25 if num_days > 0 else 1/365.25 # Avoid div by zero if only one day
    
    annualized_return = ((log_df["portfolio_value"].iloc[-1] / initial_capital) ** (1 / num_years) - 1) * 100 if num_years > 0 else (total_return if num_days > 0 else 0)

    log_df["daily_return"] = log_df["portfolio_value"].pct_change().fillna(0)
    annualized_volatility = log_df["daily_return"].std() * np.sqrt(252) * 100 # Assuming 252 trading days

    sharpe_ratio = (annualized_return / annualized_volatility) if annualized_volatility != 0 else 0

    log_df["cumulative_max"] = log_df["portfolio_value"].cummax()
    log_df["drawdown"] = (log_df["portfolio_value"] - log_df["cumulative_max"]) / log_df["cumulative_max"]
    max_drawdown = log_df["drawdown"].min() * 100 if not log_df["drawdown"].empty else 0

    # Count approximate number of trades (buy followed by sell, or vice versa)
    num_trades = (log_df['signal'] != log_df['signal'].shift()).fillna(False).sum() // 2

    metrics = {
        "Initial Capital": f"${initial_capital:,.2f}",
        "Final Portfolio Value": f"${log_df['portfolio_value'].iloc[-1]:,.2f}",
        "Total Return (%)": f"{total_return:.2f}%",
        "Annualized Return (%)": f"{annualized_return:.2f}%",
        "Annualized Volatility (%)": f"{annualized_volatility:.2f}%",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Max Drawdown (%)": f"{max_drawdown:.2f}%",
        "Number of Trades (approx)": f"{num_trades}"
    }
    return metrics, log_df

# --------------------------------
# Streamlit UI
# --------------------------------
llm_client = None
try:
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY") or getattr(st.secrets, "DEEPSEEK_API_KEY", None)
    openai_key = os.environ.get("OPENAI_API_KEY") or getattr(st.secrets, "OPENAI_API_KEY", None)

    if deepseek_key:
        llm_client = ModelClient(api_key=deepseek_key, provider="deepseek")
        st.sidebar.caption("✅ LLM: DeepSeek Initialized")
    elif openai_key:
        llm_client = ModelClient(api_key=openai_key, provider="openai")
        st.sidebar.caption("✅ LLM: OpenAI Initialized")
    else:
        st.sidebar.warning("LLM API key missing in environment variables or Streamlit secrets. Sentiment analysis and News Summary disabled.")
except ValueError as e:
    st.sidebar.error(f"LLM Initialization Error: {e}. Check API Key.")
except Exception as e:
    st.sidebar.error(f"LLM Initialization Unexpected Error: {e}")

st.title("🚀 AI Hedge Fund Simulator")

st.header("⚙️ Configuration")
config_container = st.container(border=True)
app_mode = "Live Analysis" # Default mode

with config_container:
    app_mode = st.radio("Select Mode:", ["Live Analysis", "Backtesting"], key="app_mode_select_main", horizontal=True, index=0)
    st.markdown("---") # Separator

    if app_mode == "Live Analysis":
        st.subheader("Live Analysis Settings")
        tickers_in_main = st.text_input("Tickers (comma-separated):", "AAPL,MSFT,GOOG", key="live_tickers_input_main")
        history_years_live_main = st.slider("Historical Data for Analysis (Years):", 1, 10, 5, key="live_history_slider_main")
        
        st.subheader("Feature Toggles (Live Analysis)")
        cols_features = st.columns(3)
        with cols_features[0]:
            use_sentiment_live_main = st.checkbox("News Sentiment & Summary (LLM)", value=True if llm_client else False, disabled=not llm_client, key="live_sentiment_cb_main", help="Uses LLM for news sentiment and summary. Requires NewsAPI key for comprehensive news, else uses Yahoo Finance news.")
            use_filings_live_main = st.checkbox("SEC & Institutional Filings", value=True, key="live_sec_filings_cb_main", help="Analyzes SEC Form 4 insider trades and aggregates institutional holdings from Yahoo Finance.")
        with cols_features[1]:
            use_politician_filings_main = st.checkbox("Politician Filings", value=False, key="live_politician_cb_main", help="EXPERIMENTAL: Attempts to scrape CapitolTrades.com for politician trading data. May be slow/unreliable and prone to breaking.")
            use_value_trades_main = st.checkbox("ValueInvesting.io Fair Value", value=False, key="live_vt_cb_main", help="EXPERIMENTAL: Scrapes fair value estimates from ValueInvesting.io based on Peter Lynch's formula. May be slow/unreliable.")
        
        st.markdown("") # Spacer
        run_button_live_main = st.button("🚀 Run Live Analysis", use_container_width=True, type="primary", key="run_live_btn_main")

    elif app_mode == "Backtesting":
        st.subheader("Backtesting Settings")
        bt_ticker_main = st.text_input("Ticker for Backtest:", "AAPL", key="bt_ticker_input_main").upper()
        
        col1_bt, col2_bt = st.columns(2)
        with col1_bt:
            default_bt_end_date_main = datetime.now() - timedelta(days=1)
            default_bt_start_date_main = default_bt_end_date_main - pd.DateOffset(years=3)
            bt_start_date_main = st.date_input("Start Date:", default_bt_start_date_main, max_value=default_bt_end_date_main - timedelta(days=1), key="bt_start_date_main").strftime("%Y-%m-%d")
        with col2_bt:
            # Ensure end date is after start date
            min_end_date = datetime.strptime(bt_start_date_main, "%Y-%m-%d") + timedelta(days=1)
            bt_end_date_main = st.date_input("End Date:", default_bt_end_date_main, min_value=min_end_date, key="bt_end_date_main").strftime("%Y-%m-%d")
        
        bt_initial_capital_main = st.number_input("Initial Capital:", 1000, 1000000, 10000, 1000, key="bt_capital_input_main", format="%d")
        
        with st.expander("Adjust Backtest Agent Weights (Simplified Strategy)", expanded=False):
            st.caption("Backtesting uses a simplified strategy focusing on core technical signals for performance. Adjust weights below:")
            bt_weights_price_main = st.slider("Price Signal Weight (SMA/RSI):", 0.0, 2.0, 1.0, 0.1, key="bt_w_price_main")
            bt_weights_momentum_main = st.slider("Momentum Signal Weight (1M/12M):", 0.0, 2.0, 0.8, 0.1, key="bt_w_momentum_main")
            bt_weights_volatility_main = st.slider("Volatility Signal Weight (Beta/Vol):", 0.0, 2.0, 0.2, 0.1, key="bt_w_vol_main")
            st.info("Other signals (Sentiment, Filings, Valuation) are disabled in backtesting to focus on quantifiable historical price-based strategies.")
            
        # These weights are passed to the PortfolioAgent in backtest_run, keeping other weights 0
        backtest_portfolio_weights_main = {
            "price": bt_weights_price_main,
            "momentum": bt_weights_momentum_main,
            "volatility": bt_weights_volatility_main,
            "sentiment": 0.0, "fund": 0.0, "valuation_dcf":0.0, "valuation_pe":0.0,
            "sec_filings": 0.0, "inst_holdings": 0.0, "analyst": 0.0, 
            "politician_filings": 0.0, "vi_signal": 0.0
        }
        st.markdown("") # Spacer
        run_button_backtest_main = st.button("📈 Run Backtest", use_container_width=True, type="primary", key="run_bt_btn_main")

st.markdown("---")

# --- Live Analysis Results Display ---
if app_mode == "Live Analysis":
    # Only run analysis if the button was clicked and tickers are provided
    if 'run_button_live_main' in locals() and run_button_live_main and 'tickers_in_main' in locals() and tickers_in_main:
        live_tickers_list_main = [t.strip().upper() for t in tickers_in_main.split(",") if t.strip()]
        if not live_tickers_list_main:
            st.error("Please enter at least one valid ticker to run live analysis.")
        else:
            # Define live_configs_main here, immediately before its usage
            live_configs_main = {
                "use_sentiment": use_sentiment_live_main,
                "use_filings": use_filings_live_main,
                "use_politician_filings": use_politician_filings_main,
                "use_value_trades": use_value_trades_main
            }

            # Store results in session_state to persist across reruns
            if 'live_output' not in st.session_state:
                st.session_state.live_output = {}
            
            with st.spinner("⏳ Processing live analysis... This may take a moment for multiple tickers and data fetches."):
                st.session_state.live_output = run_live_analysis(live_tickers_list_main, history_years_live_main, llm_client, live_configs_main)
            
            st.header("📊 Live Analysis Summary")
            num_tickers = len(live_tickers_list_main)
            cols_per_row = min(num_tickers, 3) # Max 3 columns per row for summary cards

            # Display summary cards
            for i in range(0, num_tickers, cols_per_row):
                row_tickers = live_tickers_list_main[i:i+cols_per_row]
                cols = st.columns(len(row_tickers))
                for idx, t_symbol in enumerate(row_tickers):
                    with cols[idx]:
                        res = st.session_state.live_output.get(t_symbol)
                        if not res or ("error" in res and res["error"] is not None and "Failed to fetch price history" in res["error"]):
                            st.error(f"**{t_symbol}**: {res.get('error', 'Critical error, no data to display.')}")
                            continue # Skip to next ticker if critical error
                        elif not res :
                            st.error(f"**{t_symbol}**: No analysis data returned."); continue

                        dec = res.get("final_decision", "N/A").upper()
                        score = res.get("composite_score", float('nan'))
                        price_disp = res.get("current_price_display")

                        # Color coding for decision cards
                        card_color_map = {"BUY": "green", "SELL": "red", "HOLD": "#FFA500", "ERROR": "#808080", "N/A": "#D3D3D3"}
                        card_color = card_color_map.get(dec, "#D3D3D3")
                        
                        st.markdown(f"""
                            <div style="border: 1px solid {card_color}; border-radius: 8px; padding: 15px; margin-bottom: 10px; background-color: {card_color}20;">
                                <h3 style="margin-bottom: 5px; color: {card_color};">{t_symbol}</h3>
                                <p style="font-size: 1.6em; font-weight: bold; color: {card_color}; margin-bottom: 5px;">{dec}</p>
                                <p style="font-size: 0.9em; margin-bottom: 3px;">Composite Score: <strong style="color: {card_color};">{score:.2f}</strong></p>
                                {f'<p style="font-size: 0.9em;">Current Price: <strong>${price_disp:,.2f}</strong></p>' if price_disp is not None else ""}
                            </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---") # Separator before detailed results

            # Display detailed results for each ticker in expanders
            for t_symbol in live_tickers_list_main:
                res = st.session_state.live_output.get(t_symbol)
                # Skip if critical error already handled above
                if not res or ("error" in res and res["error"] is not None and "Failed to fetch price history" in res["error"]):
                    continue 
                elif not res : continue

                with st.expander(f"🔍 Detailed Analysis for {t_symbol} ({res.get('ticker_info',{}).get('longName', 'N/A')})"):
                    # Use tabs for organized display
                    tab_titles = ["📈 Chart & Core", "펀 Fundamentals", "💰 Valuation & Fair Value", "📰 News & Filings", "⚙️ All Signals"]
                    tabs = st.tabs(tab_titles)

                    with tabs[0]: # Chart & Core Tab
                        st.subheader("Price Performance & Core Signals")
                        price_history_df_display_exp = fetch_price_history(t_symbol, period=f"{history_years_live_main}y")
                        if not price_history_df_display_exp.empty:
                            st.line_chart(price_history_df_display_exp["Close"], use_container_width=True)
                        else:
                            st.warning("Price chart not available.")
                        
                        core_s = {
                            "Price Signal (SMA/RSI)": res.get("price_signal", "N/A").upper(),
                            "SMA50 / SMA200": f"{res.get('sma50',0):.2f} / {res.get('sma200',0):.2f}",
                            "RSI14": f"{res.get('rsi14',0):.2f}",
                            "Momentum Signal (1M/12M)": res.get("momentum_signal", "N/A").upper(),
                            "Momentum 1M / 12M (%)": f"{res.get('momentum_1m',0)*100:.1f}% / {res.get('momentum_12m',0)*100:.1f}%",
                            "Volatility Signal (Beta)": res.get("volatility_signal", "N/A").upper(),
                            "Beta / Annual Volatility (%)": f"{res.get('beta',0):.2f} / {res.get('annual_vol',0)*100:.1f}%",
                        }
                        st.dataframe(pd.Series(core_s, name="Value"), use_container_width=True)

                    with tabs[1]: # Fundamentals Tab
                        st.subheader(f"Fundamental Snapshot - {res.get('industry_display', 'N/A')} ({res.get('sector_display', 'N/A')})")
                        ticker_info_res = res.get("ticker_info", {})
                        
                        fund_s = {
                            "Market Cap": f"${res.get('market_cap_display',0):,}" if res.get('market_cap_display') else "N/A",
                            "Free Cash Flow Yield": f"{res.get('fcf_yield',0)*100:.2f}%",
                            "Piotroski Score (Simple)": res.get('piotroski_score'),
                            "Return on Equity (ROE)": f"{ticker_info_res.get('returnOnEquity',0)*100:.1f}%",
                            "Debt to Equity": f"{ticker_info_res.get('debtToEquity',0):.1f}",
                            "Fundamental Signal": res.get("fund_signal", "N/A").upper()
                        }
                        st.dataframe(pd.Series(fund_s, name="Value"), use_container_width=True)
                        
                        business_summary = ticker_info_res.get("longBusinessSummary")
                        if business_summary:
                            with st.popover("View Business Summary"):
                                st.markdown(business_summary)
                        else:
                            st.info("No detailed business summary available from Yahoo Finance.")

                    with tabs[2]: # Valuation & Fair Value Tab
                        st.subheader("Valuation Metrics (from yfinance)")
                        val_error_yf = res.get("valuation_error")
                        if val_error_yf:
                            st.warning(f"Valuation (yfinance): {val_error_yf}")
                        val_s = {
                            "Forward P/E": f"{res.get('forward_pe',0):.1f}",
                            "Relative P/E Signal": res.get('relative_pe_signal', "N/A").upper(),
                            "DCF Fair Price (Simple Est.)": f"${res.get('dcf_fair_price',0):,.2f}" if pd.notna(res.get('dcf_fair_price')) else "N/A",
                            "DCF Signal": res.get('dcf_signal', "N/A").upper()
                        }
                        st.dataframe(pd.Series(val_s, name="Value"), use_container_width=True)

                        st.subheader("Analyst Ratings & Target Price")
                        analyst_s = {
                            "YFinance Recommendation": res.get("yfinance_recommendation", "N/A").replace("_", " ").title(),
                            "Analyst Target Upside (%)": f"{res.get('target_upside',0)*100:.2f}%",
                            "Inferred Analyst Buy %": f"{res.get('analyst_buy_pct_inferred',0)*100:.0f}%",
                            "Analyst Signal": res.get("analyst_signal", "N/A").upper()
                        }
                        st.dataframe(pd.Series(analyst_s, name="Value"), use_container_width=True)

                        if live_configs_main["use_value_trades"]:
                            st.subheader("ValueInvesting.io Fair Value Analysis (Experimental)")
                            vi_error = res.get('vi_data_error')
                            vi_full_text = res.get('vi_valuation_text_display')

                            if not vi_error and (res.get('vi_fair_value_estimate') is not None or vi_full_text):
                                st.markdown(f"**Analysis from ValueInvesting.io:**")
                                if vi_full_text:
                                    st.markdown(f"> *{vi_full_text}*")
                                st.caption("This analysis is based on Peter Lynch's Fair Value formula as per ValueInvesting.io.")
                                
                                if res.get('vi_fair_value_estimate') is not None:
                                    st.markdown(f"- **Fair Value (VI.io):** ${res.get('vi_fair_value_estimate'):,.2f}")
                                if res.get('vi_site_market_price') is not None:
                                    st.markdown(f"- **Market Price (VI.io Site):** ${res.get('vi_site_market_price'):,.2f}")
                                if res.get('current_price_display') is not None:
                                     st.markdown(f"- **Current Yahoo Price:** ${res.get('current_price_display'):,.2f}")
                                if res.get('vi_upside_percent') is not None:
                                    st.markdown(f"- **Upside/Downside (VI.io):** {res.get('vi_upside_percent'):.2f}%")
                                st.markdown(f"- **VI.io Signal:** {res.get('vi_signal', 'N/A').upper()}")
                            elif vi_error:
                                st.warning(f"ValueInvesting.io Status: {vi_error}")
                                st.caption("Could not retrieve or parse fair value details from ValueInvesting.io. This feature is experimental and may be unreliable.")
                            else:
                                st.info("ValueInvesting.io: No specific fair value analysis text found or parsed for this ticker.")
                                st.caption("This feature is experimental and may be unreliable.")

                    with tabs[3]: # News & Filings Tab
                        if live_configs_main["use_sentiment"]:
                            st.subheader("News Sentiment (LLM Analysis)")
                            llm_status_message = res.get("news_status_display", "Status: OK") 
                            if res.get("sentiment_error"): llm_status_message += f" | LLM Sentiment Error: {res.get('sentiment_error')}"
                            sent_s = {
                                "Sentiment Score": f"{res.get('sentiment_score',0):.2f}",
                                "Sentiment Signal": res.get("sentiment_signal", "N/A").upper(),
                                "News Fetch & LLM Status": llm_status_message
                            }
                            st.dataframe(pd.Series(sent_s, name="Value"), use_container_width=True)
                            
                            st.subheader("News Summary (Generated by LLM)")
                            news_summary_text = res.get("news_summary", "No news summary generated.")
                            news_summary_error = res.get("news_summary_error")
                            if news_summary_error:
                                st.error(f"News Summary Error: {news_summary_error}")
                            st.markdown(f"*{news_summary_text}*")

                            news_headlines = res.get("news_headlines_for_popover")
                            if news_headlines:
                                with st.popover("View Recent News Headlines (Top 10)"):
                                    for title_info in news_headlines:
                                        st.markdown(f"- {title_info}") # These now include links
                            elif "Error" not in llm_status_message and "No news articles found" not in llm_status_message and "No valid news" not in llm_status_message:
                                st.caption("No news headlines available or processed.")
                        else:
                            st.info("News Sentiment and Summary are disabled in configuration.")
                        
                        st.markdown("---")

                        if live_configs_main["use_filings"]:
                            st.subheader("SEC Insider Transactions (Form 4 - Past Year)")
                            sec_filings_error = res.get("sec_filings_error")
                            if sec_filings_error:
                                st.caption(f"Status: {sec_filings_error}")
                            
                            sec_data_display = {
                                "Net Insider Shares (1Y)": f"{res.get('sec_net_insider_shares_1y',0):,}",
                                "Total Insider Buy Value (1Y Est.)": f"${res.get('sec_insider_buy_value_1y',0):,.0f}",
                                "Total Insider Sell Value (1Y Est.)": f"${res.get('sec_insider_sell_value_1y',0):,.0f}",
                                "SEC Filings Signal": res.get("sec_filings_signal", "N/A").upper()
                            }
                            st.dataframe(pd.Series(sec_data_display, name="Value"), use_container_width=True)
                            
                            recent_form4_txs = res.get("sec_recent_form4_transactions")
                            if recent_form4_txs:
                                with st.popover("View Recent SEC Form 4 Transactions (Max 10)"):
                                    for tx in recent_form4_txs:
                                        direction = "Acquired" if tx.get('acq_disp_code') == 'A' else "Disposed"
                                        price_info = f"@ ${tx.get('price_per_share'):.2f}" if tx.get('price_per_share') is not None else "(price N/A)"
                                        st.markdown(f"- **{tx.get('transaction_date')}**: {tx.get('reporting_owner')} ({tx.get('owner_relationship', '')}) "
                                                    f"{direction} {tx.get('shares'):,.0f} shares {price_info}. Code: {tx.get('transaction_code')}. "
                                                    f"[Link]({tx.get('link_to_filing')})")
                            elif not sec_filings_error:
                                st.caption("No recent Form 4 transactions parsed or found within the last year.")

                            other_filings_display = res.get("sec_other_recent_filings")
                            if other_filings_display:
                                st.subheader("Other Recent SEC Filings (Past Year - Max 10)")
                                for filing_item in other_filings_display:
                                    st.markdown(f"- **{filing_item.get('filing_date')}**: Form {filing_item.get('form_type')} - [View Filing]({filing_item.get('summary_link')})")
                            elif not sec_filings_error: 
                                st.caption("No other recent SEC filings found within the last year.")
                            
                            st.subheader("Institutional Holdings (Snapshot via Yahoo Finance)")
                            inst_holdings_error = res.get("inst_holdings_error")
                            if inst_holdings_error:
                                st.caption(f"Status: {inst_holdings_error}")
                            
                            inst_data_display = {
                                "Number of Institutions Holding": res.get('inst_num_holders', 0),
                                "Total Shares Held by Institutions": f"{res.get('inst_total_shares_held',0):,}",
                                "% Outstanding Held by Institutions": f"{res.get('inst_total_pct_out',0.0)*100:.2f}%",
                                "Institutional Holdings Signal": res.get("inst_holdings_signal", "N/A").upper()
                            }
                            st.dataframe(pd.Series(inst_data_display, name="Value"), use_container_width=True)
                            st.caption("Note: This data represents a snapshot of institutional ownership from Yahoo Finance, often based on the latest available 13F filings. It does not show historical changes in institutional holdings directly.")

                            top_holders_display = res.get("inst_top_holders")
                            if top_holders_display:
                                with st.popover("View Top Institutional Holders (Max 10 from yfinance)"):
                                    for i, holder in enumerate(top_holders_display):
                                        shares_display = f"{holder.get('Shares',0):,}" if isinstance(holder.get('Shares'), (int,float)) else holder.get('Shares', 'N/A')
                                        pct_out_display = f"{holder.get('% Out',0.0)*100:.2f}%" if isinstance(holder.get('% Out'), (int,float)) else holder.get('% Out', 'N/A')
                                        st.markdown(f"{i+1}. **{holder.get('Holder')}**: Shares: {shares_display} (% Out: {pct_out_display}) - Reported: {holder.get('Date Reported')}")
                            elif not inst_holdings_error:
                                st.caption("No top institutional holder data processed or found.")
                        else:
                            st.info("SEC and Institutional Filings are disabled in configuration.")

                        if live_configs_main["use_politician_filings"]:
                            st.subheader("Politician Trading Activity (Experimental)")
                            poli_error = res.get("politician_data_error")
                            if poli_error:
                                st.warning(f"Politician Trades Status: {poli_error}")
                            poli_data_display = {
                                "Net Politician Trade Value Est. (Total)": f"${res.get('politician_net_trade_value_estimate',0):,.0f}",
                                "Politician Buy Transactions": res.get('politician_buy_tx_count',0),
                                "Politician Sell Transactions": res.get('politician_sell_tx_count',0),
                                "Politician Filings Signal": res.get("politician_filings_signal", "N/A").upper()
                            }
                            st.dataframe(pd.Series(poli_data_display, name="Value"), use_container_width=True)
                            
                            politician_trades_detail = res.get("politician_trades_for_popover")
                            if politician_trades_detail:
                                with st.popover("View Recent Politician Trades (Max 5)"):
                                    for trade in politician_trades_detail:
                                        st.markdown(f"- **{trade.get('date_str')}**: {trade.get('politician_name')} - {trade.get('transaction_type').title()} - {trade.get('value_range')} [Link]({trade.get('source_url')})")
                            elif not poli_error:
                                st.caption("No recent politician trades found or parsed for this ticker.")
                        else:
                            st.info("Politician Filings are disabled in configuration.")


                    with tabs[4]: # All Signals Tab
                        st.subheader("Aggregated Agent Signals & Final Decision")
                        # Filter for keys ending with "_signal" and format for display
                        all_s_keys = [k for k in res if k.endswith("_signal")]
                        all_s_table = {k.replace("_signal","").replace("_"," ").title(): str(res[k]).upper() for k in all_s_keys}
                        
                        all_s_table["Composite Score"] = f"{res.get('composite_score',0.0):.2f}"
                        all_s_table["Final Decision"] = res.get('final_decision',"").upper()
                        
                        st.dataframe(pd.Series(all_s_table, name="Signal Value"), use_container_width=True)
                        
                        # Display full raw JSON for advanced users/debugging
                        with st.popover("View Full Raw Analysis Data (JSON)"):
                            st.json(res)

    # Sidebar for displaying Portfolio Agent Weights (Live Analysis)
    with st.sidebar.expander("Portfolio Agent Weights (Live Analysis)", expanded=False):
        st.caption("These weights are used by the PortfolioAgent to combine individual signals into the Composite Score and Final Decision.")
        st.json(PortfolioAgent.WEIGHTS) # Display the default weights


# --- Backtesting Results Display ---
elif app_mode == "Backtesting":
    if 'run_button_backtest_main' in locals() and run_button_backtest_main and 'bt_ticker_main' in locals() and bt_ticker_main:
        # Initialize session state variables for backtest results
        if 'bt_metrics' not in st.session_state:
            st.session_state.bt_metrics = None
        if 'bt_log_df' not in st.session_state:
            st.session_state.bt_log_df = pd.DataFrame()
        
        with st.spinner(f"⏳ Running backtest for {bt_ticker_main} from {bt_start_date_main} to {bt_end_date_main}... This may take a while."):
            st.session_state.bt_metrics, st.session_state.bt_log_df = run_backtest(
                bt_ticker_main, bt_start_date_main, bt_end_date_main,
                bt_initial_capital_main, llm_client, backtest_portfolio_weights_main # Pass LLM client as placeholder, it's not used in BT
            )
        
        # Display backtest results
        if st.session_state.bt_metrics and "message" not in st.session_state.bt_metrics and "error" not in st.session_state.bt_metrics:
            st.header(f"📈 Backtest Results for {bt_ticker_main}")
            
            # Convert metrics dict to DataFrame for nice display
            metrics_df = pd.DataFrame.from_dict(st.session_state.bt_metrics, orient='index', columns=['Value'])
            st.table(metrics_df)
            
            if not st.session_state.bt_log_df.empty:
                st.subheader("Portfolio Value Over Time")
                st.line_chart(st.session_state.bt_log_df["portfolio_value"])
                
                st.subheader("Drawdown Over Time")
                st.area_chart(st.session_state.bt_log_df["drawdown"])
                
                with st.expander("View Raw Backtest Log and Signals (Last 1000 rows)"):
                    st.dataframe(st.session_state.bt_log_df[["price", "signal", "composite_score", "portfolio_value", "cash", "shares_held"]].tail(1000))
            else:
                st.warning("Backtest log is empty, no charts can be displayed.")
        else:
            # Display backtest errors
            st.error(f"Backtest failed: {st.session_state.bt_metrics.get('message', '') or st.session_state.bt_metrics.get('error', 'Unknown error') if st.session_state.bt_metrics else 'Unknown error'}")

# --- Sidebar for global info / disclaimers ---
st.sidebar.markdown("---")
st.sidebar.info("This simulator is for educational purposes only and does not constitute financial advice. Investment decisions should be made based on your own research and professional advice.")
st.sidebar.markdown("Experimental scraping features (Politician Trades, ValueInvesting.io Fair Value) are prone to breaking due to website changes and may be slow or unreliable. They are provided 'as-is' and use public web scraping, which may be against some websites' Terms of Service.")

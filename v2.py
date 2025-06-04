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
from urllib.parse import urljoin, urlparse # Added urlparse
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
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception: # Simplified error handling, orchestrator will report
        return pd.DataFrame()

@st.cache_data
def fetch_ticker_info(ticker: str) -> dict:
    """Fetches comprehensive info from yfinance for a ticker."""
    try:
        info = yf.Ticker(ticker).info
        if not info or (info.get('regularMarketPrice') is None and info.get('currentPrice') is None and info.get('financialCurrency') is None):
            return {} # Orchestrator will handle this
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
    except Exception: # Simplified error handling
        return {}

@st.cache_data
def fetch_enriched_news(ticker: str, ticker_info_data: dict) -> list[dict]:
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

        enriched_news_list = []
        if not raw_news: return []
        for news_item in raw_news:
            if not isinstance(news_item, dict): continue
            enriched_item = news_item.copy()
            enriched_item['ticker'] = ticker; enriched_item['company_name'] = company_name
            enriched_item['source_api'] = 'Yahoo Finance'
            if 'providerPublishTime' in news_item and news_item['providerPublishTime'] is not None:
                try:
                    timestamp = int(news_item['providerPublishTime'])
                    dt_object_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                    enriched_item['publish_datetime_utc'] = dt_object_utc
                    enriched_item['publish_time_readable'] = dt_object_utc.strftime('%Y-%m-%d %H:%M:%S %Z')
                except (ValueError, TypeError, OSError) as e_ts:
                    enriched_item['publish_datetime_utc'] = None; enriched_item['publish_time_readable'] = "N/A"
                    enriched_item['publish_time_error'] = str(e_ts)
            else:
                enriched_item['publish_datetime_utc'] = None; enriched_item['publish_time_readable'] = "N/A"
            for key in ['title', 'publisher', 'link', 'type']: enriched_item.setdefault(key, 'N/A' if key != 'link' else '#')
            enriched_news_list.append(enriched_item)
        enriched_news_list.sort(key=lambda x: x.get('publish_datetime_utc') or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return enriched_news_list
    except Exception as e:
        return [{"error": f"Processing Yahoo Finance news for {ticker} failed: {e}", "source_api": "Yahoo Finance"}]

@st.cache_data(ttl=1800)
def fetch_comprehensive_news_from_api(ticker: str, company_name: str, lookback_days: int = 30) -> list[dict]:
    api_key = st.secrets.get("NEWSAPI_KEY")
    if not api_key: return [{"error": "NEWSAPI_KEY not found.", "source_api": "NewsAPI.org"}]
    newsapi = NewsApiClient(api_key=api_key)
    query = f'("{company_name}" OR {ticker.upper()}) AND (stock OR shares OR business OR finance OR earnings OR "product launch" OR "analyst rating" OR "market sentiment")'
    to_date_dt = datetime.now(timezone.utc); from_date_dt = to_date_dt - timedelta(days=lookback_days)
    from_param_str = from_date_dt.strftime('%Y-%m-%d'); to_param_str = to_date_dt.strftime('%Y-%m-%d')
    articles_list = []
    try:
        all_articles_response = newsapi.get_everything(q=query, from_param=from_param_str, to=to_param_str, language='en', sort_by='publishedAt', page_size=100)
        if all_articles_response.get("status") == "ok" and "articles" in all_articles_response:
            for article in all_articles_response["articles"]:
                dt_object_utc = None; readable_time = "N/A"
                if article.get('publishedAt'):
                    try:
                        dt_object_utc = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                        readable_time = dt_object_utc.strftime('%Y-%m-%d %H:%M:%S %Z')
                    except ValueError: pass
                articles_list.append({"uuid": article.get('url'), "title": article.get('title', 'No Title'), "publisher": article.get('source', {}).get('name', 'N/A'), "link": article.get('url', '#'), "publish_datetime_utc": dt_object_utc, "publish_time_readable": readable_time, "description": article.get('description'), "content_snippet": article.get('content'), "company_name": company_name, "ticker": ticker, "source_api": "NewsAPI.org"})
        elif all_articles_response.get("status") == "error": return [{"error": f"NewsAPI Error ({ticker}): {all_articles_response.get('code')} - {all_articles_response.get('message')}", "source_api": "NewsAPI.org"}]
        else: return [{"error": f"NewsAPI ({ticker}): No articles or unexpected structure.", "source_api": "NewsAPI.org"}]
    except requests.exceptions.RequestException as e: return [{"error": f"NewsAPI request failed for {ticker}: {e}", "source_api": "NewsAPI.org"}]
    except Exception as e: return [{"error": f"Unexpected error with NewsAPI for {ticker}: {e}", "source_api": "NewsAPI.org"}]
    return articles_list

@st.cache_data(ttl=24*3600)
def get_all_cik_ticker_mappings():
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(url, headers={'User-Agent': SEC_USER_AGENT})
        response.raise_for_status()
        data = response.json()
        return {item['ticker']: str(item['cik_str']).zfill(10) for item in data if 'ticker' in item and 'cik_str' in item}
    except Exception as e:
        st.error(f"CRITICAL: Failed to fetch CIK ticker mappings: {e}. SEC features may be impaired."); return {}
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
        today = datetime.now(timezone.utc); date_limit = today - timedelta(days=lookback_days)
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
                    except: pass # Suppress individual Form 4 fetch/parse errors
                elif len([f for f in filings_list if not f.get("is_form4_transaction")]) < max_other:
                    filings_list.append({"is_form4_transaction": False, "ticker": ticker_symbol, "filing_date": date_str, "form_type": form, "document_link": f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{acc_no_dashless}/{doc_name}", "summary_link": idx_link})
            if not filings_list and xml_fetches > 0: return [{"error": f"SEC: {xml_fetches} Form 4s found for {ticker_symbol}, but no transactions parsed."}]
            if not filings_list: return [{"error": f"SEC: No relevant filings for {ticker_symbol} in last {lookback_days} days (CIK:{cik_padded})."}]
        else: return [{"error": f"SEC: No recent filings data in submissions JSON for {ticker_symbol} (CIK:{cik_padded})."}]
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

@st.cache_data(ttl=3600)
def fetch_politician_trades(ticker: str, days_back: int = 365) -> list[dict]:
    url = f"https://www.capitoltrades.com/trades?asset={ticker.upper()}&pageSize=100&perPage=100"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36', 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8', 'Accept-Language': 'en-US,en;q=0.5', 'Referer': 'https://www.capitoltrades.com/'}
    trades_list = []
    try:
        response = requests.get(url, headers=headers, timeout=20); response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        trade_rows = soup.select("a[href^='/trades/'][class*='trade-row'], a[href^='/trades/'][class*='issuer-trade-row']")
        if not trade_rows: trade_rows = soup.find_all('a', href=lambda href: href and href.startswith('/trades/'))
        if not trade_rows: return [{"error": f"CT: No trade rows found for {ticker}. Website HTML structure may have changed or scraping blocked. This feature is experimental."}]
        for row in trade_rows[:20]:
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
        if not trades_list and trade_rows: return [{"error": f"CT: Found rows for {ticker}, but failed parsing. HTML details may have changed."}]
        return trades_list
    except requests.exceptions.Timeout: return [{"error": f"CT: Timeout fetching {ticker}."}]
    except requests.exceptions.HTTPError as e: return [{"error": f"CT: HTTP error {e.response.status_code if e.response else ''} for {ticker}."}]
    except requests.exceptions.RequestException as e: return [{"error": f"CT: Request error for {ticker}: {e}."}]
    except Exception as e: return [{"error": f"CT: Parsing error for {ticker}: {e}."}]

# --- LLM Client and Agent Classes (assumed robust from previous versions) ---
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
        P_12m = price_data_slice.Close.shift(252).iloc[-1] # Already checked len >= 253
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

class PoliticianFilingsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        trades, err = data.get("politician_trades",[]), None
        net_val, buys, sells = 0,0,0
        if trades and isinstance(trades,list) and len(trades)>0 and isinstance(trades[0],dict) and "error" in trades[0]: err = trades[0]["error"]
        elif trades:
            for trade in trades:
                if isinstance(trade,dict):
                    val = trade.get("value_estimate_lower",0)
                    if trade.get("transaction_type")=="purchase": net_val += val; buys +=1
                    elif trade.get("transaction_type")=="sale": net_val -= val; sells +=1
        sig = "hold"
        if not err:
            if buys > sells and buys > 1: sig = "buy"
            elif sells > buys and sells > 1: sig = "sell"
        return {"ticker":ticker, "politician_net_trade_value_estimate":net_val, "politician_buy_tx_count":buys, "politician_sell_tx_count":sells, "politician_filings_signal":sig, "politician_data_error":err}

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

# --- Orchestrator and Backtesting (structure assumed robust from prior, focusing on the NameError fix impact) ---
def run_live_analysis(tickers, llm_client, configs):
    results = {}
    for t in tickers:
        st.write(f"▶️ Running analysis for {t}...")
        price_history_full = fetch_price_history(t, period="max")
        if price_history_full.empty:
            results[t] = {"error": f"Price history unavailable for {t}.", "ticker": t, "final_decision":"error", "composite_score":0}; st.error(f"Skipping {t}: Price history error."); continue
        ticker_info = fetch_ticker_info(t)
        if not ticker_info or not ticker_info.get("financialCurrency"):
            err_msg = f"Core ticker info (e.g., currency) unavailable for {t}. Invalid/delisted/no yfinance data."
            results[t] = {"error": err_msg, "ticker": t, "final_decision":"error", "composite_score":0}; st.error(f"Skipping {t}: {err_msg}"); continue
        current_price_for_ticker = ticker_info.get("currentPrice")
        if current_price_for_ticker is None and not price_history_full.empty: current_price_for_ticker = price_history_full["Close"].iloc[-1]
        company_name_for_news = ticker_info.get('longName', ticker_info.get('shortName', t))
        combined_news, news_fetch_msgs = [], []
        if configs["use_sentiment"]:
            yf_news = fetch_enriched_news(t, ticker_info)
            if yf_news and not (isinstance(yf_news[0],dict) and "error" in yf_news[0]): combined_news.extend(yf_news)
            elif yf_news and isinstance(yf_news[0],dict) and "error" in yf_news[0]: news_fetch_msgs.append(f"Yahoo: {yf_news[0]['error']}")
            if llm_client and st.secrets.get("NEWSAPI_KEY"):
                api_news = fetch_comprehensive_news_from_api(t, company_name_for_news)
                if api_news and not (isinstance(api_news[0],dict) and "error" in api_news[0]): combined_news.extend(api_news)
                elif api_news and isinstance(api_news[0],dict) and "error" in api_news[0]: news_fetch_msgs.append(f"NewsAPI: {api_news[0]['error']}")
            elif configs["use_sentiment"] and not st.secrets.get("NEWSAPI_KEY"): news_fetch_msgs.append("NewsAPI Key missing.")
        seen_urls, dedup_news = set(), []
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
            "politician_trades":fetch_politician_trades(t) if configs["use_politician_filings"] else [],
            "value_investing_io_data":fetch_value_investing_io_data(t) if configs["use_value_trades"] else {"error":"VI.io: Skipped."},
            "institutional_holdings":fetch_inst_filings(t) if configs["use_filings"] else [],
            "sec_all_filings_raw":fetch_all_sec_filings(t) if configs["use_filings"] else []
        }
        agents = [PriceAgent(), MomentumAgent(), VolatilityAgent(), FundamentalsAgent(), ValuationAgent(), AnalystRatingAgent()]
        if configs["use_sentiment"] and llm_client: agents.extend([SentimentAgent(llm_client), NewsSummaryAgent(llm_client)])
        if configs["use_filings"]: agents.extend([SECFilingAgent(), InstitutionalHoldingsAgent()])
        if configs["use_politician_filings"]: agents.append(PoliticianFilingsAgent())
        if configs["use_value_trades"]: agents.append(ValueInvestingIOAgent())
        agent_res_list = []
        for agent in agents:
            name = agent.__class__.__name__
            try:
                if isinstance(agent,(PriceAgent,MomentumAgent)): res_a = agent.run(t, data_bundle["price_history"])
                elif isinstance(agent,VolatilityAgent): res_a = agent.run(t, data_bundle, data_bundle["price_history"])
                else: res_a = agent.run(t, data_bundle)
                agent_res_list.append(res_a)
            except Exception as e:
                err_k, sig_k = name.lower().replace("agent","")+"_error", name.lower().replace("agent","")+"_signal"
                agent_res_list.append({sig_k:"error", err_k:f"Agent {name} error: {str(e)[:150]}"}); st.warning(f"Error in {name} for {t}: {e}")
        final_dec = PortfolioAgent().run(t, agent_res_list)
        curr_res_dict = {"ticker":t, "current_price_display":current_price_for_ticker, "market_cap_display":ticker_info.get("marketCap"), "industry_display":ticker_info.get("industry"), "sector_display":ticker_info.get("sector"), "ticker_info":ticker_info,
                         "news_headlines_for_popover":[f"{n.get('publish_time_readable','N/A')} - {n.get('title','N/A')} ({n.get('publisher','N/A')} via {n.get('source_api','Unk')}) [Link]({n.get('link','#')})" + (f" - {n.get('content_snippet',n.get('description',''))[:150]}..." if n.get('content_snippet') or n.get('description') else "") for n in dedup_news[:10]],
                         "politician_trades_for_popover":[pt for pt in data_bundle["politician_trades"][:5] if isinstance(pt,dict) and "error" not in pt],
                         "news_status_display":news_status_bundle}
        for r_dict in agent_res_list:
            if isinstance(r_dict,dict): curr_res_dict.update(r_dict)
        curr_res_dict.update(final_dec); results[t] = curr_res_dict
    return results

def run_backtest(ticker, start_date, end_date, initial_capital, llm_client_placeholder, backtest_agent_weights):
    st.write(f"Preparing backtest: {ticker} ({start_date} to {end_date})...")
    s_dt = datetime.strptime(start_date, "%Y-%m-%d"); fetch_s_dt = (s_dt - pd.DateOffset(months=18)).strftime("%Y-%m-%d")
    full_hist = fetch_price_history(ticker, period="max", interval="1d")
    if full_hist.empty: return {"error": f"Backtest fail {ticker}: Price history empty."}, pd.DataFrame()
    hist = full_hist[(full_hist.index >= pd.to_datetime(fetch_s_dt)) & (full_hist.index <= pd.to_datetime(end_date))].copy()
    if hist.empty or len(hist[hist.index >= pd.to_datetime(start_date)]) < 2: return {"error": f"Backtest fail {ticker}: Not enough data in range."}, pd.DataFrame()
    info_bt = fetch_ticker_info(ticker) # Might be {}
    data_static = {"ticker_info": info_bt}
    p_agent, m_agent, v_agent, port_agent = PriceAgent(), MomentumAgent(), VolatilityAgent(), PortfolioAgent()
    log, cash, shares, port_val = [], initial_capital, 0, initial_capital
    run_dates = hist[hist.index >= pd.to_datetime(start_date)].index
    for curr_dt in run_dates:
        data_sl = hist[hist.index <= curr_dt]
        curr_price_pt = data_sl.Close.iloc[-1] if not data_sl.empty else (port_val / shares if shares else 0)
        if data_sl.empty or len(data_sl) < 253: # Min days for indicators
            log.append({"date":curr_dt, "cash":cash, "shares_held":shares, "price":curr_price_pt, "portfolio_value":port_val, "signal":"hold (no data)", "composite_score":0.0}); continue
        curr_price = data_sl.Close.iloc[-1]
        pa_r, ma_r, va_r = p_agent.run(ticker,data_sl), m_agent.run(ticker,data_sl), v_agent.run(ticker,data_static,data_sl)
        final_dec_obj = port_agent.run(ticker, [pa_r,ma_r,va_r], agent_weights=backtest_agent_weights)
        final_dec = final_dec_obj["final_decision"]
        if final_dec=="buy" and cash > curr_price and curr_price > 0:
            s_buy = cash/curr_price; shares += s_buy; cash=0
        elif final_dec=="sell" and shares > 0:
            cash += shares*curr_price; shares=0
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

# --- Streamlit UI ---
llm_client = None
try:
    ds_key, oa_key = st.secrets.get("DEEPSEEK_API_KEY"), st.secrets.get("OPENAI_API_KEY") # Prefer secrets
    if not ds_key: ds_key = os.environ.get("DEEPSEEK_API_KEY") # Fallback to env
    if not oa_key: oa_key = os.environ.get("OPENAI_API_KEY")

    if ds_key: llm_client = ModelClient(api_key=ds_key, provider="deepseek"); st.sidebar.caption("✅ LLM: DeepSeek")
    elif oa_key: llm_client = ModelClient(api_key=oa_key, provider="openai"); st.sidebar.caption("✅ LLM: OpenAI")
    else: st.sidebar.warning("LLM API key missing. Sentiment/Summary disabled.")
except ValueError as e: st.sidebar.error(f"LLM Init Error: {e}"); llm_client=None
except Exception as e: st.sidebar.error(f"LLM Unexpected Init Error: {e}"); llm_client=None

st.title("🚀 AI Hedge Fund Simulator")
st.header("⚙️ Configuration"); config_cont = st.container(border=True)
app_mode = "Live Analysis"
with config_cont:
    app_mode = st.radio("Select Mode:", ["Live Analysis", "Backtesting"], key="app_mode_sel", horizontal=True)
    st.markdown("---")
    if app_mode == "Live Analysis":
        st.subheader("Live Analysis Settings")
        tickers_in = st.text_input("Tickers (comma-separated):", "AAPL,MSFT,GOOG,CRWV", key="live_tickers_in")
        st.caption("ℹ️ Live analysis uses all available historical data.")
        st.subheader("Feature Toggles"); feat_cols = st.columns(3)
        with feat_cols[0]:
            use_sent = st.checkbox("News Sentiment & Summary (LLM)", value=bool(llm_client), disabled=not llm_client, key="live_sent_cb", help="Uses LLM. Requires NewsAPI key.")
            use_filings = st.checkbox("SEC & Inst. Filings", value=True, key="live_sec_cb")
        with feat_cols[1]:
            use_poli = st.checkbox("Politician Filings (Exp.)", value=False, key="live_poli_cb", help="Scrapes CapitolTrades. May be slow/unreliable.")
            use_valtrades = st.checkbox("ValueInvesting.io (Exp.)", value=False, key="live_vt_cb", help="Scrapes ValueInvesting.io. May be slow/unreliable.")
        st.markdown(""); run_live_btn = st.button("🚀 Run Live Analysis", use_container_width=True, type="primary", key="run_live_btn")
    elif app_mode == "Backtesting":
        st.subheader("Backtesting Settings"); bt_ticker = st.text_input("Ticker:", "AAPL", key="bt_ticker_in").upper()
        bt_c1, bt_c2 = st.columns(2)
        with bt_c1:
            def_end_dt = datetime.now()-timedelta(days=1); def_start_dt = def_end_dt-pd.DateOffset(years=3)
            start_dt_in = st.date_input("Start Date:", def_start_dt, max_value=def_end_dt-timedelta(days=30), key="bt_start_dt")
            bt_start_str = start_dt_in.strftime("%Y-%m-%d")
        with bt_c2:
            min_end_dt_bt = start_dt_in+timedelta(days=30)
            bt_end_str = st.date_input("End Date:", def_end_dt, min_value=min_end_dt_bt, max_value=datetime.now()-timedelta(days=1), key="bt_end_dt").strftime("%Y-%m-%d")
        bt_capital = st.number_input("Initial Capital:",1000,1000000,10000,1000,key="bt_cap_in",format="%d")
        with st.expander("Adjust Backtest Weights",expanded=False):
            st.caption("Backtesting uses a simplified strategy. Adjust weights:")
            w_p, w_m, w_v = st.slider("Price W:",0.,2.,1.,.1,key="bt_w_p"), st.slider("Mom W:",0.,2.,.8,.1,key="bt_w_m"), st.slider("Vol W:",0.,2.,.2,.1,key="bt_w_v")
            st.info("Other signals disabled in backtest.")
        bt_weights = {"price":w_p, "momentum":w_m, "volatility":w_v, "sentiment":0.,"fund":0.,"valuation_dcf":0.,"valuation_pe":0.,"sec_filings":0.,"inst_holdings":0.,"analyst":0.,"politician_filings":0.,"vi_signal":0.}
        st.markdown(""); run_bt_btn = st.button("📈 Run Backtest",use_container_width=True,type="primary",key="run_bt_btn")
st.markdown("---")

if app_mode == "Live Analysis":
    if 'run_live_btn' in locals() and run_live_btn and 'tickers_in' in locals() and tickers_in:
        live_tickers = [t.strip().upper() for t in tickers_in.split(",") if t.strip()]
        if not live_tickers: st.error("Please enter at least one ticker.")
        else:
            live_configs = {"use_sentiment":use_sent, "use_filings":use_filings, "use_politician_filings":use_poli, "use_value_trades":use_valtrades}
            if 'live_output' not in st.session_state: st.session_state.live_output = {}
            with st.spinner("⏳ Processing live analysis..."):
                st.session_state.live_output = run_live_analysis(live_tickers, llm_client, live_configs)
            st.header("📊 Live Analysis Summary"); n_tickers = len(live_tickers); cols_pr = min(n_tickers,3)
            for i in range(0,n_tickers,cols_pr):
                row_t = live_tickers[i:i+cols_pr]; cols_ui = st.columns(len(row_t))
                for idx, sym in enumerate(row_t):
                    with cols_ui[idx]:
                        res = st.session_state.live_output.get(sym)
                        if not res or res.get("error"): st.error(f"**{sym}**: {res.get('error','No data.') if res else 'No data.'}"); continue
                        dec,score,price = res.get("final_decision","N/A").upper(), res.get("composite_score",float('nan')), res.get("current_price_display")
                        cmap={"BUY":"green","SELL":"red","HOLD":"#FFA500","ERROR":"#808080","N/A":"#D3D3D3"}; color=cmap.get(dec,"#D3D3D3")
                        p_html = f'<p style="font-size:0.9em;">Price:<strong>${price:,.2f}</strong></p>' if isinstance(price,(int,float)) else '<p style="font-size:0.9em;">Price:<strong>N/A</strong></p>'
                        s_html = f'<p style="font-size:0.9em;">Score:<strong style="color:{color};">{score:.2f}</strong></p>' if pd.notna(score) else f'<p style="font-size:0.9em;">Score:<strong style="color:{color};">N/A</strong></p>'
                        st.markdown(f"""<div style="border:1px solid {color};border-radius:8px;padding:15px;margin-bottom:10px;background-color:{color}20;"><h3 style="margin-bottom:5px;color:{color};">{sym}</h3><p style="font-size:1.6em;font-weight:bold;color:{color};margin-bottom:5px;">{dec}</p>{s_html}{p_html}</div>""", unsafe_allow_html=True)
            st.markdown("---")
            for sym_detail in live_tickers: # Changed variable name to avoid conflict
                res_detail = st.session_state.live_output.get(sym_detail)
                if not res_detail or res_detail.get("error"): continue
                with st.expander(f"🔍 Detailed Analysis for {sym_detail} ({res_detail.get('ticker_info',{}).get('longName','N/A')})"):
                    ui_tabs = st.tabs(["📈 Chart & Core", " फंड Fundamentals", "💰 Valuation & Fair Value", "📰 News & Filings", "⚙️ All Signals"])
                    with ui_tabs[0]: # Chart & Core
                        st.subheader("Price Performance & Core Signals")
                        # CORRECTED: Directly fetch price history for the chart.
                        price_hist_chart = fetch_price_history(sym_detail, period="max") # Use sym_detail
                        if not price_hist_chart.empty:
                            plot_df = price_hist_chart.copy()
                            if len(plot_df) > 5*252: plot_df = plot_df.tail(5*252)
                            st.line_chart(plot_df["Close"], use_container_width=True)
                        else: st.warning("Price chart data unavailable.")
                        core_s_data = {"Price Sig (SMA/RSI)":res_detail.get("price_signal","N/A").upper(), "SMA50/SMA200":f"{res_detail.get('sma50',np.nan):.2f}/{res_detail.get('sma200',np.nan):.2f}", "RSI14":f"{res_detail.get('rsi14',np.nan):.2f}", "Momentum Sig (1M/12M)":res_detail.get("momentum_signal","N/A").upper(), "Momentum 1M/12M (%)":f"{res_detail.get('momentum_1m',np.nan)*100:.1f}%/{res_detail.get('momentum_12m',np.nan)*100:.1f}%", "Volatility Sig (Beta)":res_detail.get("volatility_signal","N/A").upper(), "Beta/Ann.Vol (%)":f"{res_detail.get('beta',np.nan):.2f}/{res_detail.get('annual_vol',np.nan)*100:.1f}%"}
                        st.dataframe(pd.Series(core_s_data,name="Value"),use_container_width=True)
                        if res_detail.get("price_error"): st.caption(f"Price Note: {res_detail.get('price_error')}")
                        if res_detail.get("momentum_error"): st.caption(f"Momentum Note: {res_detail.get('momentum_error')}")
                    with ui_tabs[1]: # Fundamentals
                        st.subheader(f"Fundamentals - {res_detail.get('industry_display','N/A')} ({res_detail.get('sector_display','N/A')})")
                        info_res_exp = res_detail.get("ticker_info",{}); fund_s_exp = {}
                        mcap_exp = res_detail.get('market_cap_display'); fund_s_exp["Market Cap"] = f"${mcap_exp:,.0f}" if isinstance(mcap_exp,(int,float)) else "N/A"
                        fcfy_exp = res_detail.get('fcf_yield'); fund_s_exp["FCF Yield"] = f"{fcfy_exp*100:.2f}%" if isinstance(fcfy_exp,(int,float)) else "N/A"
                        piot_exp = res_detail.get('piotroski_score'); fund_s_exp["Piotroski Score"] = piot_exp if piot_exp is not None else "N/A"
                        roe_exp = info_res_exp.get('returnOnEquity'); fund_s_exp["ROE"] = f"{roe_exp*100:.1f}%" if isinstance(roe_exp,(int,float)) else "N/A"
                        de_exp = info_res_exp.get('debtToEquity'); fund_s_exp["Debt/Equity"] = f"{de_exp:.1f}" if isinstance(de_exp,(int,float)) else "N/A"
                        fund_s_exp["Fund. Signal"] = res_detail.get("fund_signal","N/A").upper(); st.dataframe(pd.Series(fund_s_exp,name="Value"),use_container_width=True)
                        if info_res_exp.get("longBusinessSummary"):
                            with st.popover("Business Summary"): st.markdown(info_res_exp.get("longBusinessSummary"))
                        else: st.info("No business summary.")
                    with ui_tabs[2]: # Valuation
                        st.subheader("Valuation (yfinance)"); val_err_yf = res_detail.get("valuation_error")
                        if val_err_yf: st.warning(f"Valuation (yf): {val_err_yf}")
                        val_s_exp_data = {}; fwdpe_d = res_detail.get('forward_pe'); val_s_exp_data["Fwd P/E"] = f"{fwdpe_d:.1f}" if isinstance(fwdpe_d,(int,float)) else "N/A"
                        val_s_exp_data["Rel. P/E Sig"] = res_detail.get('relative_pe_signal',"N/A").upper()
                        dcf_fp_d = res_detail.get('dcf_fair_price'); val_s_exp_data["DCF Fair Price (Est)"] = f"${dcf_fp_d:,.2f}" if pd.notna(dcf_fp_d) and isinstance(dcf_fp_d,(int,float)) else "N/A"
                        val_s_exp_data["DCF Sig"] = res_detail.get('dcf_signal',"N/A").upper(); st.dataframe(pd.Series(val_s_exp_data,name="Value"),use_container_width=True)
                        st.subheader("Analyst Ratings"); an_s_exp_data = {}
                        an_s_exp_data["YF Rec"] = res_detail.get("yfinance_recommendation","N/A").replace("_"," ").title()
                        targ_up_d = res_detail.get('target_upside'); an_s_exp_data["Target Upside (%)"] = f"{targ_up_d*100:.2f}%" if isinstance(targ_up_d,(int,float)) else "N/A"
                        buy_pct_inf_d = res_detail.get('analyst_buy_pct_inferred'); an_s_exp_data["Inf. Buy %"] = f"{buy_pct_inf_d*100:.0f}%" if isinstance(buy_pct_inf_d,(int,float)) else "N/A"
                        an_s_exp_data["Analyst Sig"] = res_detail.get("analyst_signal","N/A").upper(); st.dataframe(pd.Series(an_s_exp_data,name="Value"),use_container_width=True)
                        if res_detail.get("analyst_error"): st.caption(f"Analyst Note: {res_detail.get('analyst_error')}")
                        if live_configs["use_value_trades"]:
                            st.subheader("ValueInvesting.io (Exp.)"); vi_err_d = res_detail.get('vi_data_error'); vi_text_d = res_detail.get('vi_valuation_text_display')
                            if not vi_err_d and (res_detail.get('vi_fair_value_estimate') is not None or vi_text_d):
                                st.markdown("**VI.io Analysis:**");
                                if vi_text_d: st.markdown(f"> *{vi_text_d}*")
                                if res_detail.get('vi_fair_value_estimate') is not None: st.markdown(f"- FV (VI.io): ${res_detail.get('vi_fair_value_estimate'):,.2f}")
                                if res_detail.get('vi_site_market_price') is not None: st.markdown(f"- MP (VI.io): ${res_detail.get('vi_site_market_price'):,.2f}")
                                price_disp_vi_d = res_detail.get('current_price_display')
                                if price_disp_vi_d is not None and isinstance(price_disp_vi_d,(int,float)): st.markdown(f"- Curr YF Price: ${price_disp_vi_d:,.2f}")
                                if res_detail.get('vi_upside_percent') is not None: st.markdown(f"- Upside (VI.io): {res_detail.get('vi_upside_percent'):.2f}%")
                                st.markdown(f"- VI.io Sig: {res_detail.get('vi_signal','N/A').upper()}")
                            elif vi_err_d: st.warning(f"VI.io Status: {vi_err_d}")
                            else: st.info("VI.io: No specific fair value analysis parsed.")
                    with ui_tabs[3]: # News & Filings
                        if live_configs["use_sentiment"]:
                            st.subheader("News Sentiment (LLM)"); sent_status_d = res_detail.get("news_status_display","OK")
                            if res_detail.get("sentiment_error"): sent_status_d += f" | LLM Sent Err: {res_detail.get('sentiment_error')}"
                            sent_s_d = {"Sent. Score":f"{res_detail.get('sentiment_score',0.0):.2f}", "Sent. Signal":res_detail.get("sentiment_signal","N/A").upper(), "News/LLM Status":sent_status_d}
                            st.dataframe(pd.Series(sent_s_d,name="Value"),use_container_width=True)
                            st.subheader("News Summary (LLM)"); news_sum_err_d = res_detail.get("news_summary_error")
                            if news_sum_err_d: st.error(f"News Summary Err: {news_sum_err_d}")
                            st.markdown(f"*{res_detail.get('news_summary','No summary.')}*")
                            news_pop_d = res_detail.get("news_headlines_for_popover")
                            if news_pop_d:
                                with st.popover("Recent News (Top 10)"):
                                    for title in news_pop_d: st.markdown(f"- {title}")
                            elif "Error" not in sent_status_d and "No news" not in sent_status_d: st.caption("No headlines for summary.")
                        else: st.info("News Sentiment/Summary disabled.")
                        st.markdown("---")
                        if live_configs["use_filings"]:
                            st.subheader("SEC Insider Tx (Form 4 - 1Y)"); sec_err_d = res_detail.get("sec_filings_error")
                            if sec_err_d: st.caption(f"SEC Status: {sec_err_d}")
                            sec_data_d = {"Net Insider Shares (1Y)":f"{res_detail.get('sec_net_insider_shares_1y',0):,}", "Insider Buy Val (1Y Est)":f"${res_detail.get('sec_insider_buy_value_1y',0):,.0f}", "Insider Sell Val (1Y Est)":f"${res_detail.get('sec_insider_sell_value_1y',0):,.0f}", "SEC Filings Sig":res_detail.get("sec_filings_signal","N/A").upper()}
                            st.dataframe(pd.Series(sec_data_d,name="Value"),use_container_width=True)
                            form4_pop_d = res_detail.get("sec_recent_form4_transactions")
                            if form4_pop_d:
                                with st.popover("Recent SEC Form 4 Tx (Max 10)"):
                                    for tx in form4_pop_d:
                                        direction = "Acq" if tx.get('acq_disp_code')=='A' else ("Disp" if tx.get('acq_disp_code')=='D' else tx.get('acq_disp_code','N/A'))
                                        price_info = f"@ ${tx.get('price_per_share'):.2f}" if isinstance(tx.get('price_per_share'),(int,float)) else "(price N/A)"
                                        st.markdown(f"- **{tx.get('transaction_date')}**: {tx.get('reporting_owner')} ({tx.get('owner_relationship','')}) {direction} {tx.get('shares',0):,.0f} sh {price_info}. Code:{tx.get('transaction_code')}. [Link]({tx.get('link_to_filing')})")
                            elif not sec_err_d: st.caption("No recent Form 4 tx.")
                            other_f_pop_d = res_detail.get("sec_other_recent_filings")
                            if other_f_pop_d:
                                st.subheader("Other Recent SEC Filings (1Y - Max 10)")
                                for f_item in other_f_pop_d: st.markdown(f"- **{f_item.get('filing_date')}**: Form {f_item.get('form_type')} - [View]({f_item.get('summary_link')})")
                            elif not sec_err_d: st.caption("No other recent SEC filings.")
                            st.subheader("Institutional Holdings (yfinance)"); inst_err_d = res_detail.get("inst_holdings_error")
                            if inst_err_d: st.caption(f"Inst. Holdings Status: {inst_err_d}")
                            inst_data_d = {"# Inst. Holding":res_detail.get('inst_num_holders',0), "Total Shares Held by Inst.":f"{res_detail.get('inst_total_shares_held',0):,}", "% Out Held by Inst.":f"{res_detail.get('inst_total_pct_out',0.0)*100:.2f}%", "Inst. Holdings Sig":res_detail.get("inst_holdings_signal","N/A").upper()}
                            st.dataframe(pd.Series(inst_data_d,name="Value"),use_container_width=True)
                            top_h_pop_d = res_detail.get("inst_top_holders")
                            if top_h_pop_d:
                                with st.popover("Top Inst. Holders (Max 10 yf)"):
                                    for i,h in enumerate(top_h_pop_d):
                                        s_d = f"{h.get('Shares',0):,}" if isinstance(h.get('Shares'),(int,float)) else h.get('Shares','N/A')
                                        p_d = f"{h.get('% Out',0.0)*100:.2f}%" if isinstance(h.get('% Out'),(int,float)) else h.get('% Out','N/A')
                                        st.markdown(f"{i+1}. **{h.get('Holder')}**: Sh:{s_d} (%Out:{p_d}) Rept:{h.get('Date Reported','N/A')}")
                            elif not inst_err_d: st.caption("No top inst. holder data.")
                        else: st.info("SEC/Inst. Filings disabled.")
                        if live_configs["use_politician_filings"]:
                            st.subheader("Politician Trading (Exp.)"); pol_err_d = res_detail.get("politician_data_error")
                            if pol_err_d: st.warning(f"Poli. Trades Status: {pol_err_d}")
                            pol_data_d = {"Net Poli. Trade Val Est":f"${res_detail.get('politician_net_trade_value_estimate',0):,.0f}", "Poli. Buy Tx":res_detail.get('politician_buy_tx_count',0), "Poli. Sell Tx":res_detail.get('politician_sell_tx_count',0), "Poli. Filings Sig":res_detail.get("politician_filings_signal","N/A").upper()}
                            st.dataframe(pd.Series(pol_data_d,name="Value"),use_container_width=True)
                            pol_pop_d = res_detail.get("politician_trades_for_popover")
                            if pol_pop_d:
                                with st.popover("Recent Poli. Trades (Max 5)"):
                                    for trade in pol_pop_d: st.markdown(f"- **{trade.get('date_str')}**: {trade.get('politician_name')} - {trade.get('transaction_type','N/A').title()} - {trade.get('value_range')} [Link]({trade.get('source_url')})")
                            elif not pol_err_d: st.caption("No recent poli. trades.")
                        else: st.info("Poli. Filings disabled.")
                    with ui_tabs[4]: # All Signals
                        st.subheader("Aggregated Signals & Final Decision")
                        all_s_keys_d = [k for k in res_detail if k.endswith("_signal")]; all_s_tab_d = {k.replace("_signal","").replace("_"," ").title(): str(res_detail[k]).upper() for k in all_s_keys_d}
                        all_s_tab_d["Composite Score"] = f"{res_detail.get('composite_score',0.0):.2f}"; all_s_tab_d["Final Decision"] = res_detail.get('final_decision',"").upper()
                        st.dataframe(pd.Series(all_s_tab_d,name="Signal Value"),use_container_width=True)
                        with st.popover("View Full Raw Analysis Data (JSON)"): st.json(res_detail)
    with st.sidebar.expander("Portfolio Agent Weights (Live Analysis)",expanded=False):
        st.caption("Weights for PortfolioAgent combining signals."); st.json(dict(sorted(PortfolioAgent.WEIGHTS.items())))
elif app_mode == "Backtesting":
    if 'run_bt_btn' in locals() and run_bt_btn and 'bt_ticker' in locals() and bt_ticker: # use bt_ticker from UI
        if 'bt_metrics' not in st.session_state: st.session_state.bt_metrics = None
        if 'bt_log_df' not in st.session_state: st.session_state.bt_log_df = pd.DataFrame()
        with st.spinner(f"⏳ Running backtest for {bt_ticker} from {bt_start_str} to {bt_end_str}..."): # use UI vars
            st.session_state.bt_metrics, st.session_state.bt_log_df = run_backtest(bt_ticker, bt_start_str, bt_end_str, bt_capital, llm_client, bt_weights)
        if st.session_state.bt_metrics and not (st.session_state.bt_metrics.get("message") or st.session_state.bt_metrics.get("error")):
            st.header(f"📈 Backtest Results for {bt_ticker}") # use UI var
            metrics_df_bt = pd.DataFrame.from_dict(st.session_state.bt_metrics,orient='index',columns=['Value']); st.table(metrics_df_bt)
            if not st.session_state.bt_log_df.empty:
                st.subheader("Portfolio Value Over Time"); st.line_chart(st.session_state.bt_log_df["portfolio_value"])
                st.subheader("Drawdown Over Time"); drawdown_s_bt = st.session_state.bt_log_df["drawdown"].fillna(0); st.area_chart(drawdown_s_bt)
                with st.expander("View Raw Backtest Log (Last 1000)"): st.dataframe(st.session_state.bt_log_df[["price","signal","composite_score","portfolio_value","cash","shares_held"]].tail(1000))
            else: st.warning("Backtest log empty.")
        else:
            err_msg_bt_res = "Unknown backtest error."
            if st.session_state.bt_metrics: err_msg_bt_res = st.session_state.bt_metrics.get('message','') or st.session_state.bt_metrics.get('error','Unknown error')
            st.error(f"Backtest failed: {err_msg_bt_res}")
st.sidebar.markdown("---")
st.sidebar.info("Educational purposes only. Not financial advice.")
st.sidebar.markdown("Experimental scraping features may be unreliable.")

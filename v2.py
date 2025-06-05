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

# --------------------------------
# Data Fetchers
# --------------------------------
@st.cache_data
def fetch_price_history(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
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

# --- LLM Client and Agent Classes ---
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

# --- Orchestrator and Backtesting ---
def run_live_analysis(tickers, llm_client, configs):
    results = {}
    for t in tickers:
        st.write(f"▶️ Running analysis for {t}...")
        price_history_full = fetch_price_history(t, period="max")
        if price_history_full.empty:
            results[t] = {"error": f"Price history unavailable for {t}.", "ticker": t, "final_decision":"error", "composite_score":0}; continue
        ticker_info = fetch_ticker_info(t)
        if not ticker_info or not ticker_info.get("financialCurrency"):
            err_msg = f"Core ticker info (e.g., currency) unavailable for {t}. Invalid/delisted/no yfinance data."
            results[t] = {"error": err_msg, "ticker": t, "final_decision":"error", "composite_score":0}; continue
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
    ticker = res_detail.get("ticker", "N/A"); ticker_info = res_detail.get("ticker_info", {})
    tab_titles = ["📈 Chart & Core", "📊 Fundamentals", "💰 Analyst & Fair Value", "📰 News & Filings", "⚙️ All Signals"]
    tabs = st.tabs(tab_titles)
    def get_signal_color(signal):
        if signal == "BUY" or signal == "STRONG_BUY": return "green"
        elif signal == "SELL": return "red"
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
            st.subheader("Technical Indicators"); price_signal = res_detail.get('price_signal', 'hold').upper()
            st.metric(label=f"Price Signal (SMA/RSI)", value=price_signal)
            st.markdown(f"""<div style="font-size: 14px;"><li><b>50-Day SMA:</b> ${res_detail.get('sma50', 0):,.2f}</li><li><b>200-Day SMA:</b> ${res_detail.get('sma200', 0):,.2f}</li><li><b>14-Day RSI:</b> {res_detail.get('rsi14', 0):.2f}</li></div>""", unsafe_allow_html=True)
        with col2:
            st.subheader("Momentum & Volatility"); momentum_signal = res_detail.get('momentum_signal', 'hold').upper()
            st.metric(label="Momentum Signal", value=momentum_signal)
            st.markdown(f"""<div style="font-size: 14px;"><li><b>1-Month Momentum:</b> {res_detail.get('momentum_1m', 0) * 100:.2f}%</li><li><b>12-Month Momentum:</b> {res_detail.get('momentum_12m', 0) * 100:.2f}%</li><li><b>Beta:</b> {res_detail.get('beta', 0):.2f}</li></div>""", unsafe_allow_html=True)

    with tabs[1]:
        st.subheader(f"Fundamental Overview: {ticker_info.get('longName', '')}"); st.caption(f"**Sector:** {ticker_info.get('sector', 'N/A')} | **Industry:** {ticker_info.get('industry', 'N/A')}")
        with st.expander("Show Business Summary"): st.write(ticker_info.get('longBusinessSummary', 'No summary available.'))
        st.markdown("---"); fund_col1, fund_col2, fund_col3, fund_col4 = st.columns(4)
        fund_col1.metric("Market Cap", f"${ticker_info.get('marketCap', 0) / 1e12:.2f}T" if isinstance(ticker_info.get('marketCap'),(int,float)) else "N/A")
        fund_col2.metric("Trailing P/E", f"{ticker_info.get('trailingPE', 0):.2f}" if isinstance(ticker_info.get('trailingPE'),(int,float)) else "N/A")
        fund_col3.metric("Forward P/E", f"{ticker_info.get('forwardPE', 0):.2f}" if isinstance(ticker_info.get('forwardPE'),(int,float)) else "N/A")
        fund_col4.metric("Price/Book", f"{ticker_info.get('priceToBook', 0):.2f}" if isinstance(ticker_info.get('priceToBook'),(int,float)) else "N/A")
        st.markdown("---"); st.subheader("Financial Health"); fund_sig = res_detail.get('fund_signal', 'hold').upper()
        f_col1, f_col2, f_col3 = st.columns(3)
        f_col1.metric("Fundamental Signal", fund_sig); f_col2.metric("Piotroski Score (0-3)", f"{res_detail.get('piotroski_score', 'N/A')}/3")
        fcy_val = res_detail.get('fcf_yield'); f_col3.metric("FCF Yield", f"{fcy_val * 100:.2f}%" if isinstance(fcy_val,(int,float)) else "N/A")
        roe_val = ticker_info.get('returnOnEquity'); de_val = ticker_info.get('debtToEquity'); etr_val = ticker_info.get('enterpriseToRevenue'); ete_val = ticker_info.get('enterpriseToEbitda')
        health_data = {"Return on Equity (ROE)": f"{roe_val * 100:.2f}%" if isinstance(roe_val,(int,float)) else "N/A", "Debt to Equity": f"{de_val:.2f}" if isinstance(de_val,(int,float)) else "N/A", "EV/Revenue": f"{etr_val:.2f}" if isinstance(etr_val,(int,float)) else "N/A", "EV/EBITDA": f"{ete_val:.2f}" if isinstance(ete_val,(int,float)) else "N/A"}
        st.table(pd.DataFrame(health_data.items(), columns=["Metric", "Value"]))

    with tabs[2]:
        val_col1, val_col2 = st.columns(2)
        with val_col1:
            st.subheader("Analyst Consensus"); analyst_signal = res_detail.get('analyst_signal', 'hold').upper()
            st.metric(label=f"Analyst Signal (from {ticker_info.get('numberOfAnalystOpinions')} analysts)", value=analyst_signal)
            abp_val = res_detail.get('analyst_buy_pct_inferred',0.5); st.progress(abp_val, text=f"{abp_val*100:.0f}% Buy Rating")
            tm_val = ticker_info.get('targetMeanPrice'); tu_val = res_detail.get('target_upside')
            st.metric("Mean Target Price", f"${tm_val:.2f}" if isinstance(tm_val,(int,float)) else "N/A", f"{tu_val*100:.2f}% Upside" if isinstance(tu_val,(int,float)) else None)
        with val_col2:
            st.subheader("Peter Lynch Fair Value (via VI.io)"); vi_signal = res_detail.get('vi_signal', 'hold').upper()
            vi_fv = res_detail.get('vi_fair_value_estimate'); up_val = res_detail.get('vi_upside_percent')
            st.metric(label=f"VI.io Signal (Fair Value: ${vi_fv:,.2f})", value=vi_signal, delta=f"{up_val:.2f}% Upside" if isinstance(up_val,(int,float)) else None)
            if res_detail.get('vi_valuation_text_display'): st.markdown(f"> *{res_detail.get('vi_valuation_text_display')}*")
    
    with tabs[3]:
        st.subheader("News Analysis & Filings")
        if res_detail.get('news_summary'):
            with st.container(border=True):
                st.markdown("**AI-Generated News Summary**"); st.write(res_detail.get('news_summary'))
                if res_detail.get('sentiment_error'): st.warning(f"Sentiment Analysis Note: {res_detail.get('sentiment_error')}")
        file_col1, file_col2 = st.columns(2)
        with file_col1:
            st.markdown("**SEC Filings**"); st.metric("Insider Signal", res_detail.get('sec_filings_signal', 'hold').upper())
            with st.expander("View Recent Filings"):
                filings = res_detail.get('sec_other_recent_filings', [])
                if filings:
                    for f in filings: st.write(f"**{f.get('filing_date')}**: Form {f.get('form_type')} - [Link]({f.get('summary_link')})")
                else: st.info("No recent SEC filings found.")
        with file_col2:
            st.markdown("**Institutional Holdings**"); st.metric("Institutional Signal", res_detail.get('inst_holdings_signal', 'hold').upper())
            with st.expander("View Top 10 Institutional Holders"):
                holders = res_detail.get('inst_top_holders', [])
                if holders:
                    df_holders = pd.DataFrame(holders)
                    st.dataframe(df_holders[["Holder", "Shares", "% Out"]].rename(columns={"% Out":"% of Outstanding"}), column_config={"% of Outstanding": st.column_config.ProgressColumn(format="%.2f%%", min_value=0, max_value=0.10)}, hide_index=True, use_container_width=True)
                else: st.info("No institutional holder data available.")

    with tabs[4]:
        st.subheader("All Agent Signals at a Glance")
        signals_data = {"Price Signal (SMA/RSI)": res_detail.get("price_signal","N/A").upper(), "Momentum Signal": res_detail.get("momentum_signal","N/A").upper(), "Volatility Signal": res_detail.get("volatility_signal","N/A").upper(), "Fundamental Signal": res_detail.get("fund_signal","N/A").upper(), "Analyst Signal": res_detail.get("analyst_signal","N/A").upper(), "ValueInvesting.io Signal": res_detail.get("vi_signal","N/A").upper(), "News Sentiment Signal": res_detail.get("sentiment_signal","N/A").upper(), "SEC Filings Signal": res_detail.get("sec_filings_signal","N/A").upper(), "Institutional Signal": res_detail.get("inst_holdings_signal","N/A").upper()}
        df_signals = pd.DataFrame(signals_data.items(), columns=["Agent", "Signal"])
        st.dataframe(df_signals.style.applymap(lambda x: f'color: {get_signal_color(x)}', subset=['Signal']), hide_index=True, use_container_width=True)
        st.markdown("---")
        final_decision = res_detail.get('final_decision', 'hold').upper(); final_color = get_signal_color(final_decision)
        st.markdown(f"""<div style="border:2px solid {final_color}; border-radius:8px; padding:15px; text-align:center;"><p style="font-size:1.2em; margin-bottom:5px;">Final AI Decision</p><h2 style="color:{final_color}; margin-bottom:5px;">{final_decision}</h2><p style="font-size:1em;">Composite Score: <strong>{res_detail.get('composite_score', 0):.2f}</strong></p></div>""", unsafe_allow_html=True)


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

app_mode_options = ["Live Analysis", "Backtesting", "💼 Portfolio Management"] 
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = app_mode_options[0] 

with config_cont:
    st.session_state.app_mode = st.radio("Select Mode:", app_mode_options, key="app_mode_sel_main_key", horizontal=True, index=app_mode_options.index(st.session_state.app_mode))
    st.markdown("---")

    if st.session_state.app_mode == "Live Analysis":
        st.subheader("Live Analysis Settings")
        tickers_in_live = st.text_input("Tickers (comma-separated):", "AAPL,MSFT,GOOG,CRWV", key="live_tickers_input")
        st.caption("ℹ️ Live analysis uses all available historical data.")
        st.subheader("Feature Toggles"); feat_cols = st.columns(3)
        with feat_cols[0]:
            use_sent_live = st.checkbox("News Sentiment & Summary (LLM)", value=bool(llm_client), disabled=not llm_client, key="live_sent_cb_main", help="Uses LLM. Requires NewsAPI key.")
            use_filings_live = st.checkbox("SEC & Inst. Filings", value=True, key="live_sec_cb_main")
        with feat_cols[1]:
            use_poli_live = st.checkbox("Politician Filings (Exp.)", value=False, key="live_poli_cb_main", help="Scrapes CapitolTrades. May be slow/unreliable.")
            use_valtrades_live = st.checkbox("ValueInvesting.io (Exp.)", value=False, key="live_vt_cb_main", help="Scrapes ValueInvesting.io. May be slow/unreliable.")
        st.markdown(""); run_live_btn = st.button("🚀 Run Live Analysis", use_container_width=True, type="primary", key="run_live_analysis_button")

    elif st.session_state.app_mode == "Backtesting":
        st.subheader("Backtesting Settings"); bt_ticker = st.text_input("Ticker:", "AAPL", key="bt_ticker_in_bt").upper()
        bt_capital_source = st.radio("Capital Source:", ("Manual Input", "From Saved Portfolio"), horizontal=True, key="bt_capital_source_radio")
        bt_capital = 10000 
        if bt_capital_source == "Manual Input":
             bt_capital = st.number_input("Initial Capital:", 1000, 1000000, 10000, 1000, key="bt_cap_in_bt", format="%d")
        else:
            portfolio_names_bt = list(st.session_state.portfolios_data.keys())
            if not portfolio_names_bt: st.warning("No portfolios found. Create one in the Portfolio Management tab to use this feature.")
            else:
                sel_pf_bt = st.selectbox("Select Portfolio to use its total value:", portfolio_names_bt, key="bt_pf_select")
                holdings_bt = st.session_state.portfolios_data.get(sel_pf_bt, [])
                total_value = 0
                if holdings_bt:
                    for holding in holdings_bt:
                        info_bt_cap = fetch_ticker_info(holding['ticker'])
                        price_bt_cap = info_bt_cap.get('currentPrice')
                        if isinstance(price_bt_cap, (int,float)) and isinstance(holding.get('quantity'), (int,float)):
                            total_value += price_bt_cap * holding['quantity']
                    bt_capital = int(total_value) if total_value > 0 else 10000
                    st.info(f"Using **${bt_capital:,.2f}** as initial capital from portfolio '{sel_pf_bt}'.")
                else: st.warning(f"Portfolio '{sel_pf_bt}' is empty. Using default capital.")
        
        bt_c1, bt_c2 = st.columns(2)
        with bt_c1:
            def_end_dt = datetime.now()-timedelta(days=1); def_start_dt = def_end_dt-pd.DateOffset(years=3)
            start_dt_in = st.date_input("Start Date:", def_start_dt, max_value=def_end_dt-timedelta(days=30), key="bt_start_dt_bt")
            bt_start_str = start_dt_in.strftime("%Y-%m-%d")
        with bt_c2:
            min_end_dt_bt = start_dt_in+timedelta(days=30)
            bt_end_str = st.date_input("End Date:", def_end_dt, min_value=min_end_dt_bt, max_value=datetime.now()-timedelta(days=1), key="bt_end_dt_bt").strftime("%Y-%m-%d")
        
        with st.expander("Adjust Backtest Agent Weights",expanded=False):
            st.caption("Backtesting uses a simplified strategy. Adjust weights:")
            w_p, w_m, w_v = st.slider("Price W:",0.,2.,1.,.1,key="bt_w_p_bt"), st.slider("Mom W:",0.,2.,.8,.1,key="bt_w_m_bt"), st.slider("Vol W:",0.,2.,.2,.1,key="bt_w_v_bt")
            st.info("Other signals disabled in backtest.")
        bt_weights = {"price":w_p, "momentum":w_m, "volatility":w_v, "sentiment":0.,"fund":0.,"valuation_dcf":0.,"valuation_pe":0.,"sec_filings":0.,"inst_holdings":0.,"analyst":0.,"politician_filings":0.,"vi_signal":0.}
        st.markdown(""); run_bt_btn = st.button("📈 Run Backtest",use_container_width=True,type="primary",key="run_bt_btn_main")

    elif st.session_state.app_mode == "💼 Portfolio Management":
        st.subheader("💼 Portfolio Management")
        st.sidebar.subheader("Portfolio Actions")
        portfolio_names_list = list(st.session_state.portfolios_data.keys())
        if not portfolio_names_list: 
             st.session_state.portfolios_data["My First Portfolio"] = []
             st.session_state.selected_portfolio_name = "My First Portfolio"
             save_portfolios(st.session_state.portfolios_data); st.rerun()
        if st.session_state.selected_portfolio_name not in portfolio_names_list and portfolio_names_list:
             st.session_state.selected_portfolio_name = portfolio_names_list[0]
        elif not portfolio_names_list: st.session_state.selected_portfolio_name = None
        selected_portfolio_sidebar = st.sidebar.selectbox("Select Portfolio", options=portfolio_names_list, 
            index=portfolio_names_list.index(st.session_state.selected_portfolio_name) if st.session_state.selected_portfolio_name in portfolio_names_list else 0, 
            key="portfolio_selector_sidebar")
        if selected_portfolio_sidebar != st.session_state.selected_portfolio_name : 
            st.session_state.selected_portfolio_name = selected_portfolio_sidebar
            st.session_state.portfolio_stock_analysis = {}; st.rerun()

        new_portfolio_name_sidebar = st.sidebar.text_input("Create New Portfolio Name", key="new_portfolio_name_sidebar_input")
        if st.sidebar.button("Create Portfolio", key="create_portfolio_sidebar_btn"):
            if new_portfolio_name_sidebar and new_portfolio_name_sidebar not in st.session_state.portfolios_data:
                st.session_state.portfolios_data[new_portfolio_name_sidebar] = []
                st.session_state.selected_portfolio_name = new_portfolio_name_sidebar
                save_portfolios(st.session_state.portfolios_data)
                st.sidebar.success(f"Portfolio '{new_portfolio_name_sidebar}' created."); st.rerun()
            elif not new_portfolio_name_sidebar: st.sidebar.warning("Portfolio name cannot be empty.")
            else: st.sidebar.warning(f"Portfolio '{new_portfolio_name_sidebar}' already exists.")
        if st.session_state.selected_portfolio_name:
            if st.sidebar.button(f"Delete '{st.session_state.selected_portfolio_name}'", type="secondary", key=f"delete_pf_sidebar_{st.session_state.selected_portfolio_name}"):
                if st.session_state.selected_portfolio_name in st.session_state.portfolios_data:
                    deleted_name = st.session_state.selected_portfolio_name
                    del st.session_state.portfolios_data[st.session_state.selected_portfolio_name]
                    save_portfolios(st.session_state.portfolios_data)
                    st.sidebar.success(f"Portfolio '{deleted_name}' deleted.")
                    st.session_state.selected_portfolio_name = None; st.session_state.portfolio_stock_analysis = {}; st.rerun()
        
        if st.session_state.selected_portfolio_name:
            st.subheader(f"Managing: {st.session_state.selected_portfolio_name}")
            current_holdings_list = st.session_state.portfolios_data.get(st.session_state.selected_portfolio_name, [])
            with st.form(key="portfolio_add_stock_form"):
                st.markdown("##### Add or Update Stock in Portfolio")
                pf_cols_form = st.columns([2,1,1])
                pf_new_ticker = pf_cols_form[0].text_input("Ticker Symbol", key="pf_form_ticker").upper()
                pf_new_quantity = pf_cols_form[1].number_input("Quantity", min_value=0.0, value=1.0, step=0.01, format="%.2f", key="pf_form_qty")
                pf_new_avg_price = pf_cols_form[2].number_input("Average Purchase Price", min_value=0.0, value=0.0, step=0.01, format="%.2f", key="pf_form_avg_price")
                pf_add_stock_submitted = st.form_submit_button("💾 Save to Portfolio")
                if pf_add_stock_submitted:
                    if pf_new_ticker and pf_new_quantity > 0:
                        ticker_exists_idx = -1
                        for idx, holding in enumerate(current_holdings_list):
                            if holding['ticker'] == pf_new_ticker: ticker_exists_idx = idx; break
                        new_holding_data = {'ticker': pf_new_ticker, 'quantity': pf_new_quantity, 'avg_price': pf_new_avg_price}
                        if ticker_exists_idx != -1: current_holdings_list[ticker_exists_idx] = new_holding_data
                        else: current_holdings_list.append(new_holding_data)
                        st.session_state.portfolios_data[st.session_state.selected_portfolio_name] = current_holdings_list
                        save_portfolios(st.session_state.portfolios_data)
                        st.success(f"{pf_new_ticker} saved to {st.session_state.selected_portfolio_name}.")
                        if pf_new_ticker in st.session_state.portfolio_stock_analysis: del st.session_state.portfolio_stock_analysis[pf_new_ticker]
                        st.rerun()
                    else: st.error("Ticker and Quantity (>0) are required.")
            
            st.markdown("---")
            st.markdown("##### Configure Analysis for this Portfolio") 
            pf_analysis_feat_cols = st.columns(2)
            with pf_analysis_feat_cols[0]:
                pf_use_sentiment_config = st.checkbox("News Sentiment & Summary (LLM)", value=bool(llm_client), disabled=not llm_client, key="pf_config_sent_cb")
                pf_use_filings_config = st.checkbox("SEC & Inst. Filings", value=True, key="pf_config_sec_cb")
            with pf_analysis_feat_cols[1]:
                pf_use_politician_filings_config = st.checkbox("Politician Filings (Exp.)", value=False, key="pf_config_poli_cb")
                pf_use_value_trades_config = st.checkbox("ValueInvesting.io (Exp.)", value=False, key="pf_config_vt_cb")
            
            st.markdown("---")
            st.markdown("##### Current Holdings & Analysis")
            if not current_holdings_list: st.info("This portfolio is empty. Add stocks using the form above.")
            else:
                tickers_to_fetch_price_pf = [h['ticker'] for h in current_holdings_list if h['ticker'] not in st.session_state.portfolio_stock_analysis or st.session_state.portfolio_stock_analysis[h['ticker']].get("current_price_display") is None]
                if tickers_to_fetch_price_pf:
                    with st.spinner("Fetching current prices for portfolio display..."):
                        for ticker_sym_pf in tickers_to_fetch_price_pf:
                            if ticker_sym_pf not in st.session_state.portfolio_stock_analysis: st.session_state.portfolio_stock_analysis[ticker_sym_pf] = {}
                            info_price_pf = fetch_ticker_info(ticker_sym_pf)
                            st.session_state.portfolio_stock_analysis[ticker_sym_pf]["current_price_display"] = info_price_pf.get("currentPrice") if info_price_pf else None
                
                header_cols_pf_disp = st.columns([2,1,1,1,1,1,1,2,0.5]) 
                header_cols_pf_disp[0].markdown("**Ticker**"); header_cols_pf_disp[1].markdown("**Qty**"); header_cols_pf_disp[2].markdown("**Avg Price**")
                header_cols_pf_disp[3].markdown("**Curr Price**"); header_cols_pf_disp[4].markdown("**Curr Value**"); header_cols_pf_disp[5].markdown("**P&L**")
                header_cols_pf_disp[6].markdown("**Signal**"); header_cols_pf_disp[7].markdown("**Advice**"); header_cols_pf_disp[8].markdown("**Act**")
                for i, holding in enumerate(current_holdings_list):
                    row_data_pf_item = holding.copy()
                    analysis_res_pf_item = st.session_state.portfolio_stock_analysis.get(holding['ticker'], {})
                    current_mkt_price_pf_item = analysis_res_pf_item.get("current_price_display")
                    row_data_pf_item["Current Price"] = f"${current_mkt_price_pf_item:.2f}" if isinstance(current_mkt_price_pf_item, (int, float)) else "N/A"
                    if isinstance(current_mkt_price_pf_item, (int, float)) and isinstance(holding.get('quantity'), (int,float)):
                        row_data_pf_item["Current Value"] = f"${current_mkt_price_pf_item * holding['quantity']:,.2f}"
                        if isinstance(holding.get('avg_price'), (int,float)) and holding.get('avg_price') > 0: row_data_pf_item["Unrealized P&L"] = f"${(current_mkt_price_pf_item - holding['avg_price']) * holding['quantity']:,.2f}"
                        else: row_data_pf_item["Unrealized P&L"] = "N/A"
                    else: row_data_pf_item["Current Value"], row_data_pf_item["Unrealized P&L"] = "N/A", "N/A"
                    row_data_pf_item["Signal"] = analysis_res_pf_item.get("final_decision", "N/A").upper()
                    advice_pf_text = "Analyze for advice"
                    if row_data_pf_item["Signal"] == "BUY": advice_pf_text = "📈 Consider Buying More"
                    elif row_data_pf_item["Signal"] == "SELL": advice_pf_text = "📉 Consider Selling/Reducing"
                    elif row_data_pf_item["Signal"] == "HOLD": advice_pf_text = "HOLD"
                    row_data_pf_item["Advice"] = advice_pf_text
                    
                    data_cols_pf_disp = st.columns([2,1,1,1,1,1,1,2,0.5])
                    data_cols_pf_disp[0].write(row_data_pf_item['ticker']); data_cols_pf_disp[1].write(f"{row_data_pf_item['quantity']:.2f}"); data_cols_pf_disp[2].write(f"${row_data_pf_item['avg_price']:.2f}")
                    data_cols_pf_disp[3].write(row_data_pf_item['Current Price']); data_cols_pf_disp[4].write(row_data_pf_item['Current Value']); data_cols_pf_disp[5].write(row_data_pf_item['Unrealized P&L'])
                    data_cols_pf_disp[6].markdown(f"**{row_data_pf_item['Signal']}**"); data_cols_pf_disp[7].write(row_data_pf_item['Advice'])
                    if data_cols_pf_disp[8].button("🗑️", key=f"del_holding_{st.session_state.selected_portfolio_name}_{row_data_pf_item['ticker']}_{i}", help="Remove this holding"):
                        current_holdings_list.pop(i) 
                        if row_data_pf_item['ticker'] in st.session_state.portfolio_stock_analysis: del st.session_state.portfolio_stock_analysis[row_data_pf_item['ticker']]
                        st.session_state.portfolios_data[st.session_state.selected_portfolio_name] = current_holdings_list
                        save_portfolios(st.session_state.portfolios_data); st.rerun()
                    st.markdown("<hr style='margin:0.1rem'>", unsafe_allow_html=True)

                if st.button("📊 Analyze Entire Portfolio Holdings", key="analyze_portfolio_holdings_btn", type="primary", use_container_width=True):
                    if current_holdings_list:
                        with st.spinner("Analyzing portfolio stocks... This may take time."):
                            portfolio_analysis_configs = {"use_sentiment": pf_use_sentiment_config, "use_filings": pf_use_filings_config, "use_politician_filings": pf_use_politician_filings_config, "use_value_trades": pf_use_value_trades_config }
                            tickers_to_analyze_pf = [h['ticker'] for h in current_holdings_list]
                            analysis_batch_results_pf = run_live_analysis(tickers_to_analyze_pf, llm_client, portfolio_analysis_configs)
                            st.session_state.portfolio_stock_analysis.update(analysis_batch_results_pf)
                            st.success("Portfolio analysis complete!"); st.rerun()
                    else: st.warning("Portfolio is empty. Add stocks to analyze.")
        elif not st.session_state.selected_portfolio_name: st.info("Please create or select a portfolio using the sidebar.")

st.markdown("---") 

if st.session_state.app_mode == "Live Analysis":
    if 'run_live_btn' in locals() and run_live_btn and 'tickers_in_live' in locals() and tickers_in_live:
        live_tickers = [t.strip().upper() for t in tickers_in_live.split(",") if t.strip()]
        if not live_tickers: st.error("Please enter at least one ticker.")
        else:
            live_configs = {"use_sentiment":use_sent_live, "use_filings":use_filings_live, "use_politician_filings":use_poli_live, "use_value_trades":use_valtrades_live}
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
            for sym_detail in live_tickers: 
                res_detail = st.session_state.live_output.get(sym_detail)
                if not res_detail or res_detail.get("error"): continue
                with st.expander(f"🔍 Detailed Analysis for {sym_detail} ({res_detail.get('ticker_info',{}).get('longName','N/A')})"):
                    # This now calls the full display function, replacing the st.json placeholder
                    display_detailed_analysis(res_detail)

elif st.session_state.app_mode == "Backtesting":
    if 'run_bt_btn' in locals() and run_bt_btn and 'bt_ticker' in locals() and bt_ticker: 
        if 'backtest_results' not in st.session_state: st.session_state.backtest_results = {}
        with st.spinner(f"⏳ Running backtest for {bt_ticker} from {bt_start_str} to {bt_end_str}..."): 
            st.session_state.backtest_results[bt_ticker] = run_backtest(bt_ticker, bt_start_str, bt_end_str, bt_capital, llm_client, bt_weights)
    
    if 'bt_ticker' in locals() and bt_ticker and bt_ticker in st.session_state.backtest_results:
        bt_res_for_ticker = st.session_state.backtest_results[bt_ticker]
        metrics, log_df = bt_res_for_ticker.get("metrics"), bt_res_for_ticker.get("log_df")
        if metrics and not (metrics.get("message") or metrics.get("error")):
            st.header(f"📈 Backtest Results for {bt_ticker}") 
            metrics_df_bt = pd.DataFrame.from_dict(metrics,orient='index',columns=['Value']); st.table(metrics_df_bt)
            if log_df is not None and not log_df.empty:
                st.subheader("Portfolio Value Over Time"); st.line_chart(log_df["portfolio_value"])
                st.subheader("Drawdown Over Time"); st.area_chart(log_df["drawdown"].fillna(0))
                with st.expander("View Raw Backtest Log (Last 1000)"): st.dataframe(log_df.tail(1000))
            else: st.warning("Backtest log empty.")
        elif metrics:
            st.error(f"Backtest failed: {metrics.get('message','') or metrics.get('error','Unknown error')}")

st.sidebar.markdown("---")
st.sidebar.info("Educational purposes only. Not financial advice.")
st.sidebar.markdown("Experimental scraping features may be unreliable.")

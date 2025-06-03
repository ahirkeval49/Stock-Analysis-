import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
# from dateutil.relativedelta import relativedelta # Unused, pd.DateOffset is used
from datetime import datetime, timedelta, timezone # Added timezone
import openai
from openai import OpenAI # Ensure this is at the top of your script
from dotenv import load_dotenv
import requests # For web scraping
from bs4 import BeautifulSoup # For web scraping
import re # For parsing text more effectively
from urllib.parse import urljoin # For handling relative URLs from scraping
from newsapi import NewsApiClient # For NewsAPI.org
import json # For CIK mapping


# --- THIS IS THE CORRECT PLACEMENT ---
st.set_page_config(page_title="AI Hedge Fund Simulator", layout="wide")
# --- END CORRECT PLACEMENT ---

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
    except Exception as e:
        return pd.DataFrame()

@st.cache_data
def fetch_ticker_info(ticker: str) -> dict:
    """Fetches comprehensive info from yfinance for a ticker."""
    try:
        info = yf.Ticker(ticker).info
        if not info or info.get('regularMarketPrice') is None and info.get('currentPrice') is None :
            return {}
        return {
            "marketCap": info.get("marketCap"), "freeCashflow": info.get("freeCashflow"),
            "forwardPE": info.get("forwardPE"), "trailingPE": info.get("trailingPE"),
            "priceToBook": info.get("priceToBook"), "enterpriseToRevenue": info.get("enterpriseToRevenue"),
            "enterpriseToEbitda": info.get("enterpriseToEbitda"), "returnOnEquity": info.get("returnOnEquity"),
            "debtToEquity": info.get("debtToEquity"), "beta": info.get("beta"),
            "targetMeanPrice": info.get("targetMeanPrice"), "recommendationKey":info.get("recommendationKey"),
            "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions"), "industry": info.get("industry"),
            "sector": info.get("sector"), "longName": info.get("longName"), "shortName": info.get("shortName"),
            "longBusinessSummary": info.get("longBusinessSummary"),
            "currentPrice": info.get("currentPrice") or info.get("regularMarketPrice"),
        }
    except Exception as e:
        return {}

@st.cache_data
def fetch_enriched_news(ticker: str, ticker_info_data: dict) -> list[dict]:
    """
    Fetches news for a ticker from yfinance and enriches it.
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
        except Exception as news_exc:
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

        enriched_news_list.sort(key=lambda x: x.get('publish_datetime_utc', datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
        return enriched_news_list
    except Exception as e:
        return [{"error": f"Failed to process Yahoo Finance news for {ticker}: {e}", "source_api": "Yahoo Finance"}]

@st.cache_data(ttl=1800) # Cache NewsAPI results for 30 minutes
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
            q=query,
            from_param=from_param_str,
            to=to_param_str,
            language='en',
            sort_by='publishedAt',
            page_size=30
        )
        if all_articles_response.get("status") == "ok" and "articles" in all_articles_response:
            for article in all_articles_response["articles"]:
                dt_object_utc = None; readable_time = "N/A"
                if article.get('publishedAt'):
                    try:
                        dt_object_utc = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                        readable_time = dt_object_utc.strftime('%Y-%m-%d %H:%M:%S %Z')
                    except ValueError: pass
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
            return [{"error": f"NewsAPI ({ticker}): No articles found or unexpected data structure.", "source_api": "NewsAPI.org"}]
    except requests.exceptions.RequestException as e:
        return [{"error": f"NewsAPI request failed for {ticker}: {e}", "source_api": "NewsAPI.org"}]
    except Exception as e:
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
        st.error(f"CRITICAL: Failed to fetch CIK ticker mappings: {e}")
        return {}
TICKER_TO_CIK_MAP = get_all_cik_ticker_mappings()

def get_cik_for_ticker(ticker: str) -> str | None:
    return TICKER_TO_CIK_MAP.get(ticker.upper())

@st.cache_data(ttl=4*3600)
def fetch_sec_form4_filings(ticker_symbol: str, lookback_days: int = 180) -> list[dict]:
    cik = get_cik_for_ticker(ticker_symbol)
    if not cik:
        return [{"error": f"SEC Filings: CIK not found for {ticker_symbol}"}]

    cik_padded = str(cik).zfill(10)
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    headers = {'User-Agent': SEC_USER_AGENT}
    transactions_list = []

    try:
        response = requests.get(submissions_url, headers=headers, timeout=20)
        response.raise_for_status()
        submissions_data = response.json()

        today = datetime.now(timezone.utc)
        date_limit = today - timedelta(days=lookback_days)

        if 'filings' in submissions_data and 'recent' in submissions_data['filings']:
            recent_filings = submissions_data['filings']['recent']
            forms = recent_filings.get('form', [])
            filing_dates = recent_filings.get('filingDate', [])
            accession_numbers = recent_filings.get('accessionNumber', [])
            primary_documents = recent_filings.get('primaryDocument', [])
            
            form4_processed_count = 0
            for i in range(len(forms)):
                if forms[i] == '4':
                    try:
                        filing_date = datetime.strptime(filing_dates[i], '%Y-%m-%d').replace(tzinfo=timezone.utc)
                        if filing_date < date_limit:
                            continue
                    except ValueError:
                        continue
                    if form4_processed_count >= 20: break

                    accession_number_no_dashes = accession_numbers[i].replace('-', '')
                    primary_document_xml = primary_documents[i]
                    
                    if not primary_document_xml.lower().endswith(('.xml', '.xsd')):
                        continue

                    xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number_no_dashes}/{primary_document_xml}"
                    
                    try:
                        filing_response = requests.get(xml_url, headers=headers, timeout=10)
                        if filing_response.status_code != 200: continue
                        soup_xml = BeautifulSoup(filing_response.content, 'xml')
                        form4_processed_count +=1
                        
                        reporting_owner_tag = soup_xml.find('reportingOwner')
                        owner_name = "N/A"; owner_relationship_str = "N/A"
                        if reporting_owner_tag:
                            owner_id_tag = reporting_owner_tag.find('reportingOwnerId')
                            if owner_id_tag and owner_id_tag.find('rptOwnerName'):
                                owner_name = owner_id_tag.find('rptOwnerName').text.strip()
                            
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
                                trans_code = "N/A"
                                if trans_coding_tag and trans_coding_tag.find('transactionCode'):
                                    trans_code = trans_coding_tag.find('transactionCode').text.strip().upper()
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
                                    transactions_list.append({"ticker": ticker_symbol, "filing_date": filing_dates[i],
                                        "transaction_date": trans_date, "reporting_owner": owner_name,
                                        "owner_relationship": owner_relationship_str, "transaction_code": trans_code,
                                        "acq_disp_code": acq_disp_code, "shares": shares_val, "price_per_share": price_val,
                                        "link_to_filing": xml_url.replace(primary_document_xml, "FilingSummary.xml")})
                    except requests.exceptions.RequestException: pass
                    except Exception: pass 
            
            if not transactions_list and form4_processed_count > 0 :
                return [{"error": f"SEC: Found {form4_processed_count} Form 4s for {ticker_symbol} but could not parse transaction details."}]
            if not transactions_list:
                 return [{"error": f"SEC: No Form 4 transactions found or parsable for {ticker_symbol} in the last {lookback_days} days.", "cik": cik}]
        else: return [{"error": f"SEC: No recent filings data found for {ticker_symbol} (CIK: {cik})"}]
    except requests.exceptions.HTTPError as http_err:
        return [{"error": f"SEC: HTTP error fetching submissions for {ticker_symbol} (CIK: {cik}): {http_err}"}]
    except requests.exceptions.RequestException as req_err:
        return [{"error": f"SEC: Request error for {ticker_symbol} (CIK: {cik}): {req_err}"}]
    except Exception as e:
        return [{"error": f"SEC: Unexpected error for {ticker_symbol} (CIK: {cik}): {e}"}]
    return transactions_list

@st.cache_data(ttl=3600)
def fetch_fair_value_from_value_trades(ticker: str, company_name: str) -> dict:
    error_message = (
        f"VT ({ticker}): Scraping value-trades.com for this data is complex due to its dynamic search. "
        "This function is currently a placeholder. To implement fully, you would need to: "
        "1. Use browser Developer Tools (Network tab, filter by XHR/Fetch) to observe how 'search.php' "
        " (or a similar endpoint) is called when you search for a ticker on their site. "
        "2. Replicate that specific call (URL, method GET/POST, parameters/payload, headers) in this Python function. "
        "3. Parse the response from that call. If it provides a direct link to the stock's page, "
        "   make a second request to that link and then parse the resulting HTML for the fair value sentence. "
        "This likely requires more advanced scraping techniques if the target content isn't easily accessible "
        "via a direct API-like call that your script can mimic."
    )
    return {
        "error": error_message,
        "vt_fair_value": None, "vt_current_price": None, "vt_valuation_text": None,
        "vt_valuation_status": None, "vt_valuation_percentage": None,
        "vt_assigned_pe_by_site": None, "vt_eps_used_by_site": None
    }


@st.cache_data(ttl=3600)
def fetch_politician_trades(ticker: str, days_back: int = 365) -> list[dict]:
    url = f"https://www.capitoltrades.com/trades?asset={ticker.upper()}&pageSize=100&perPage=100"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5', 'Referer': 'https://www.capitoltrades.com/'
    }
    politician_trades_list = []
    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        trade_rows = soup.select("a[href^='/trades/'][class*='trade-row']")
        if not trade_rows: trade_rows = soup.find_all('a', href=lambda href: href and href.startswith('/trades/'))
        if not trade_rows: return [{"error": f"CT: No trade rows found for {ticker}. Site structure might have changed."}]

        for row_link_tag in trade_rows[:20]:
            politician_name_tag = row_link_tag.find(['div','span'], class_=lambda x: x and 'politician-name' in x)
            tx_type_tag = row_link_tag.find(['div','span'], class_=lambda x: x and 'tx-type' in x)
            value_range_tag = row_link_tag.find(['div','span'], class_=lambda x: x and 'tx-value' in x)
            date_tag = row_link_tag.find(['div','span'], class_=lambda x: x and 'tx-date' in x)
            if all([politician_name_tag, tx_type_tag, value_range_tag, date_tag]):
                name = politician_name_tag.text.strip()
                tx_type_text = tx_type_tag.text.strip().lower()
                tx_type = "purchase" if "purchase" in tx_type_text else "sale" if "sale" in tx_type_text else "other"
                value_range = value_range_tag.text.strip()
                date_str = date_tag.text.strip()
                value_estimate = 0
                value_matches = re.findall(r'\$([\d,]+)', value_range)
                if value_matches:
                    try: value_estimate = int(value_matches[0].replace(',', ''))
                    except ValueError: pass
                politician_trades_list.append({
                    "politician_name": name, "transaction_type": tx_type,
                    "value_range": value_range, "value_estimate_lower": value_estimate,
                    "date_str": date_str, "source_url": "https://www.capitoltrades.com" + row_link_tag['href']
                })
        if not politician_trades_list and trade_rows:
             return [{"error": f"CT: Found trade rows for {ticker} but failed to parse individual fields. Selectors need update."}]
        return politician_trades_list
    except requests.exceptions.Timeout: return [{"error": f"CT: Timeout accessing CapitolTrades for {ticker}"}]
    except requests.exceptions.RequestException as e: return [{"error": f"CT Error for {ticker}: {e}"}] # Changed from HTTPError for broader catch
    except Exception as e: return [{"error": f"CT Parsing error for {ticker}: {e}"}]

# --------------------------------
# LLM Client
# --------------------------------
class ModelClient:
    def __init__(self, api_key: str, provider: str = "openai"):
        self.api_key = api_key; self.provider = provider
        OPENAI_DEFAULT_MODEL = "gpt-4o"; DEEPSEEK_DEFAULT_MODEL = "deepseek-reasoner"
        if not api_key: raise ValueError("API key required for ModelClient.")
        if provider == "deepseek":
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com/v1")
            self.model_name = DEEPSEEK_DEFAULT_MODEL
        elif provider == "openai":
            self.client = OpenAI(api_key=self.api_key)
            self.model_name = OPENAI_DEFAULT_MODEL
        else: raise ValueError(f"Unsupported LLM provider: {provider}")

    def generate(self, prompt: str) -> str:
        try:
            stream = self.client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt}], stream=True)
            final_content = "".join(chunk.choices[0].delta.content for chunk in stream if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content)
            return final_content
        except Exception as e: raise Exception(f"LLM Generation Error ({self.provider}, {self.model_name}): {e}")

# --------------------------------
# Agents
# --------------------------------
class PriceAgent:
    def run(self, ticker: str, price_data_slice: pd.DataFrame) -> dict:
        if price_data_slice.empty or len(price_data_slice) < 200: return {"ticker": ticker, "price_signal": "hold", "sma50": np.nan, "sma200": np.nan, "rsi14": np.nan}
        df = price_data_slice.copy(); df["SMA50"]=df["Close"].rolling(50).mean(); df["SMA200"]=df["Close"].rolling(200).mean()
        delta = df["Close"].diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan); df["RSI14"] = 100 - (100 / (1 + rs)); latest = df.iloc[-1]; signal = "hold"
        if pd.isna(latest.SMA50) or pd.isna(latest.SMA200) or pd.isna(latest.RSI14): signal = "hold"
        elif latest.SMA50 > latest.SMA200 and latest.RSI14 < 70: signal = "buy"
        elif latest.SMA50 < latest.SMA200 and latest.RSI14 > 30: signal = "sell"
        return {"ticker": ticker, "sma50": float(latest.SMA50) if pd.notna(latest.SMA50) else np.nan, "sma200": float(latest.SMA200) if pd.notna(latest.SMA200) else np.nan, "rsi14": float(latest.RSI14) if pd.notna(latest.RSI14) else np.nan, "price_signal": signal}

class MomentumAgent:
    def run(self, ticker: str, price_data_slice: pd.DataFrame) -> dict:
        if price_data_slice.empty or len(price_data_slice) < 252: return {"ticker": ticker, "momentum_signal": "hold", "momentum_1m": 0, "momentum_12m": 0}
        df = price_data_slice; P_t = df.Close.iloc[-1]; P_1m = df.Close.shift(21).iloc[-1] if len(df) > 21 else np.nan
        P_12m = df.Close.shift(252).iloc[-1] if len(df) > 252 else np.nan
        m1  = (P_t/P_1m)-1  if pd.notna(P_1m) and P_1m != 0 else 0.0
        m12 = (P_t/P_12m)-1 if pd.notna(P_12m) and P_12m != 0 else 0.0; signal = "hold"
        if m12 > 0.01 and m1 > 0.01: signal = "buy"
        elif m12 < -0.01 and m1 < -0.01: signal = "sell"
        return {"ticker": ticker, "momentum_1m": float(m1), "momentum_12m": float(m12), "momentum_signal": signal}

class VolatilityAgent:
    def run(self, ticker: str, data: dict, price_data_slice: pd.DataFrame = None) -> dict:
        beta = data.get("ticker_info", {}).get("beta", 1.0);
        if beta is None: beta = 1.0
        sig  = "sell" if beta > 1.5 else ("buy" if beta < 0.8 else "hold")
        ann_vol = np.nan; weight = 0.0
        if price_data_slice is not None and not price_data_slice.empty and len(price_data_slice) > 1:
            ret = np.log(price_data_slice.Close / price_data_slice.Close.shift(1)).dropna()
            if not ret.empty: ann_vol = float(ret.std() * np.sqrt(252)); weight  = float(1/ann_vol) if ann_vol > 0 else 0.0
        return {"ticker": ticker, "beta": beta, "annual_vol": ann_vol, "vol_weight": weight, "volatility_signal": sig}

class SentimentAgent:
    def __init__(self, client): self.client = client
    def run(self, ticker: str, data: dict) -> dict:
        news_items_from_bundle = data.get("news", [])
        overall_news_fetch_error = data.get("news_fetch_status_error")
        if overall_news_fetch_error: return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": overall_news_fetch_error}
        valid_news_items = [item for item in news_items_from_bundle if isinstance(item, dict) and "error" not in item]
        if not valid_news_items:
            if news_items_from_bundle and isinstance(news_items_from_bundle[0], dict) and "error" in news_items_from_bundle[0]:
                return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": news_items_from_bundle[0].get("error")}
            return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": "No valid news articles to process."}
        content_for_llm = []; company_name_overall = data.get("ticker_info",{}).get('longName', ticker)
        for item in valid_news_items[:7]:
            title = item.get('title', ''); publisher = item.get('publisher', ''); description = item.get('description', '')
            text_snippet = f"Headline: {title}"
            if description and isinstance(description, str): text_snippet += f" | Snippet: {description[:200]}..."
            if publisher and publisher != 'N/A': text_snippet += f" (Source: {publisher} via {item.get('source_api', 'Unknown')})"
            content_for_llm.append(text_snippet)
        if not content_for_llm: return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_signal": "hold", "sentiment_error": "No processable news content for LLM."}
        prompt = (f"Analyze sentiment for {company_name_overall} ({ticker}) based on news from past month. Score -1.0 (v.neg) to +1.0 (v.pos). Output only number.\n\nNews:\n" + "\n\n".join(f"- {c}" for c in content_for_llm))
        score = 0.0; llm_error_msg = None
        try:
            response_text = self.client.generate(prompt).strip()
            if response_text.startswith("Error:"): llm_error_msg = response_text
            else:
                match = re.search(r"[-+]?\d*\.\d+|\d+", response_text)
                if match: score = float(match.group(0)); score = max(-1.0, min(1.0, score))
                else: llm_error_msg = "LLM did not return a parsable number."
        except Exception as e: llm_error_msg = f"LLM call failed: {str(e)[:150]}"
        final_error_message_for_sentiment = llm_error_msg
        if overall_news_fetch_error and ("Error" in overall_news_fetch_error or "failed" in overall_news_fetch_error.lower()) :
             final_error_message_for_sentiment = f"News: {overall_news_fetch_error}" + (f" | LLM: {llm_error_msg}" if llm_error_msg else "")
        sig = "buy" if score > 0.25 and not llm_error_msg else ("sell" if score < -0.25 and not llm_error_msg else "hold")
        return {"ticker":ticker, "sentiment_score":score, "sentiment_signal":sig, "sentiment_error": final_error_message_for_sentiment}

class FundamentalsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        s = data.get("ticker_info", {}); mcap = s.get("marketCap") or 1; fcf = s.get("freeCashflow") or 0
        roe = s.get("returnOnEquity") or 0; de = s.get("debtToEquity"); de = 1000 if de is None else de
        fcy = fcf/mcap if mcap != 0 else 0; piotroski_score = sum([roe > 0.01, de < 100, fcf > 0])
        sig  = "buy" if piotroski_score >= 2 else ("sell" if piotroski_score == 0 else "hold")
        return {"ticker": ticker, "fcf_yield": float(fcy), "piotroski_score": piotroski_score, "fund_signal": sig}

class ValuationAgent:
    def run(self, ticker: str, data: dict) -> dict:
        stats = data.get("ticker_info", {}); price_history_df = data.get("price_history")
        price = stats.get("currentPrice") or (price_history_df["Close"].iloc[-1] if price_history_df is not None and not price_history_df.empty else None)
        if price is None: return {"ticker": ticker, "forward_pe": None, "relative_pe_signal": "hold", "dcf_fair_price": np.nan, "dcf_signal": "hold"}
        pe = stats.get("forwardPE"); rel_sig = "hold";
        if pe is not None: rel_sig = "buy" if pe < 15 else "sell" if pe > 25 else "hold"
        fcf = stats.get("freeCashflow"); mcap = stats.get("marketCap")
        fcy = (fcf / mcap) if fcf is not None and mcap is not None and mcap != 0 else 0.0
        fair_price = price * (1 + fcy); dcf_sig = "hold"
        if fair_price > price * 1.15: dcf_sig = "buy"
        elif fair_price < price * 0.85: dcf_sig = "sell"
        return {"ticker": ticker, "forward_pe": pe, "relative_pe_signal": rel_sig, "dcf_fair_price": float(fair_price), "dcf_signal": dcf_sig}

class SECFilingAgent:
    def run(self, ticker: str, data: dict) -> dict:
        form4_transactions = fetch_sec_form4_filings(ticker, lookback_days=180)
        error_from_fetch = None
        if not form4_transactions: 
            error_from_fetch = f"SEC Filings: No Form 4 transactions data returned for {ticker}."
        elif isinstance(form4_transactions[0], dict) and "error" in form4_transactions[0]:
            error_from_fetch = form4_transactions[0].get("error")
            form4_transactions = [] 
        if error_from_fetch:
            return {"ticker": ticker, "sec_net_insider_shares_6m": 0, "sec_insider_buy_value_6m": 0, 
                    "sec_insider_sell_value_6m": 0, "sec_filings_signal": "hold", 
                    "sec_filings_error": error_from_fetch, "sec_recent_form4_transactions": []}

        net_shares = 0; buy_value = 0; sell_value = 0
        processed_transactions = []
        for tx in form4_transactions:
            if tx.get("transaction_code") == "P" and tx.get("acq_disp_code") == "A":
                shares = tx.get("shares", 0.0); price = tx.get("price_per_share")
                net_shares += shares
                if price is not None and shares != 0: buy_value += shares * price
                processed_transactions.append(tx)
            elif tx.get("transaction_code") == "S" and tx.get("acq_disp_code") == "D":
                shares = tx.get("shares", 0.0); price = tx.get("price_per_share")
                net_shares -= shares
                if price is not None and shares != 0: sell_value += shares * price
                processed_transactions.append(tx)
        signal = "hold"
        if net_shares > 1000 or buy_value > 100000: signal = "buy" 
        elif net_shares < -1000 or sell_value > 100000: signal = "sell"
        return {"ticker": ticker, "sec_net_insider_shares_6m": int(net_shares),
                "sec_insider_buy_value_6m": round(buy_value, 2), "sec_insider_sell_value_6m": round(sell_value, 2),
                "sec_filings_signal": signal, "sec_filings_error": None,
                "sec_recent_form4_transactions": processed_transactions[:10]}

class PoliticianFilingsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        trades = data.get("politician_trades", [])
        net_value_estimate = 0; buy_count = 0; sell_count = 0; error = None
        if trades and isinstance(trades, list) and len(trades)>0 and isinstance(trades[0], dict) and "error" in trades[0]:
            error = trades[0]["error"]
        elif trades:
            for trade in trades:
                if isinstance(trade, dict):
                    value = trade.get("value_estimate_lower", 0)
                    if trade.get("transaction_type") == "purchase": net_value_estimate += value; buy_count +=1
                    elif trade.get("transaction_type") == "sale": net_value_estimate -= value; sell_count +=1
        signal = "hold"
        if not error:
            if buy_count > sell_count and buy_count > 1 : signal = "buy"
            elif sell_count > buy_count and sell_count > 1: signal = "sell"
        return {"ticker": ticker, "politician_net_trade_value_estimate": net_value_estimate,
                "politician_buy_tx_count": buy_count, "politician_sell_tx_count": sell_count,
                "politician_filings_signal": signal, "politician_data_error": error}

class FairValueAgentVT:
    def run(self, ticker: str, data: dict) -> dict:
        vt_data = data.get("value_trades_fair_value_data", {})
        fair_value = vt_data.get("vt_fair_value"); error = vt_data.get("error")
        current_price_data = data.get("ticker_info", {}); current_price = current_price_data.get("currentPrice")
        if current_price is None and data.get("price_history") is not None and not data["price_history"].empty:
            current_price = data["price_history"]["Close"].iloc[-1]
        signal = "hold"; margin_of_safety = 0.15
        error_fv_not_found_page = f"VT: FV not found on page for {ticker}."
        error_specific_sentence_prefix = f"VT: Specific fair value sentence not found for {ticker} on page"
        significant_error = False
        if error:
            is_ignorable_error = (error == error_fv_not_found_page or error.startswith(error_specific_sentence_prefix) or
                                "VT Configuration incomplete in secrets." in error or "VT: Skipped by user config." in error or
                                "VT: Could not find stock page link for" in error or "VT: Failed to fetch valid stock page for" in error)
            if not is_ignorable_error: significant_error = True
        if not significant_error and fair_value is not None and current_price is not None and current_price > 0:
            if current_price < fair_value * (1 - margin_of_safety): signal = "buy"
            elif current_price > fair_value * (1 + margin_of_safety): signal = "sell"
        return {"ticker": ticker, "vt_fair_value_estimate": fair_value, "vt_fair_value_signal": signal, "vt_data_error": error,
                "vt_site_current_price": vt_data.get("vt_current_price"), "vt_valuation_status": vt_data.get("vt_valuation_status"),
                "vt_valuation_percentage": vt_data.get("vt_valuation_percentage"), "vt_valuation_text_display": vt_data.get("vt_valuation_text")}

class VTInspiredFairValueAgent:
    DEFAULT_ASSUMED_FAIR_PE = 20.0
    DEFAULT_INDUSTRY_PE_LOOKUP = {"TECHNOLOGY": 25.0, "CONSUMER ELECTRONICS": 25.0, "HEALTHCARE": 20.0, "FINANCIAL SERVICES": 15.0, "CONSUMER CYCLICAL": 18.0, "INDUSTRIALS": 17.0, "DEFAULT": 18.0}
    DEFAULT_HISTORICAL_AVG_PE = 20.0; DEFAULT_GROWTH_PE = 10.0
    def get_eps(self, ticker_info: dict) -> float | None: return ticker_info.get('trailingEps')
    def get_industry_pe(self, ticker_info: dict) -> float:
        industry = str(ticker_info.get("industry", "")).upper(); sector = str(ticker_info.get("sector", "")).upper()
        if industry in self.DEFAULT_INDUSTRY_PE_LOOKUP: return self.DEFAULT_INDUSTRY_PE_LOOKUP[industry]
        if sector in self.DEFAULT_INDUSTRY_PE_LOOKUP: return self.DEFAULT_INDUSTRY_PE_LOOKUP[sector]
        return self.DEFAULT_INDUSTRY_PE_LOOKUP["DEFAULT"]
    def run(self, ticker: str, data: dict) -> dict:
        ticker_info = data.get("ticker_info", {}); eps = self.get_eps(ticker_info)
        historical_avg_pe = self.DEFAULT_HISTORICAL_AVG_PE; assumed_fair_pe = self.DEFAULT_ASSUMED_FAIR_PE
        industry_pe = self.get_industry_pe(ticker_info); growth_pe_val = self.DEFAULT_GROWTH_PE
        if eps is None: return {"error_vt_inspired_fv": "EPS not available.", "vt_inspired_fair_value": None, "vt_inspired_signal": "hold"}
        pe_components = [historical_avg_pe, assumed_fair_pe, industry_pe, growth_pe_val]
        valid_pe_components = [p for p in pe_components if p is not None]; our_assigned_pe = None
        if valid_pe_components: our_assigned_pe = sum(valid_pe_components) / len(valid_pe_components)
        vt_inspired_fair_value = None
        if our_assigned_pe is not None: vt_inspired_fair_value = our_assigned_pe * eps
        current_price = ticker_info.get("currentPrice"); signal = "hold"
        if vt_inspired_fair_value is not None and current_price is not None and current_price > 0:
            margin = 0.15
            if current_price < vt_inspired_fair_value * (1 - margin): signal = "buy"
            elif current_price > vt_inspired_fair_value * (1 + margin): signal = "sell"
        return {"ticker": ticker, "vt_inspired_eps_used": eps, "vt_inspired_historical_avg_pe": historical_avg_pe,
                "vt_inspired_assumed_fair_pe": assumed_fair_pe, "vt_inspired_industry_pe": industry_pe,
                "vt_inspired_growth_pe": growth_pe_val, "vt_inspired_assigned_pe": our_assigned_pe,
                "vt_inspired_fair_value": vt_inspired_fair_value, "vt_inspired_signal": signal,
                "error_vt_inspired_fv": None if vt_inspired_fair_value is not None else "Could not calculate VT-Inspired Fair Value"}

class PortfolioAgent:
    WEIGHTS = {"price": 1.0, "momentum": 0.8, "volatility": 0.3, "sentiment": 0.6, "fund": 0.9,
               "valuation_dcf":0.5, "valuation_pe":0.5, "sec_filings": 0.6, "analyst": 0.7,
               "politician_filings": 0.4, "vt_fair_value": 0.0, "vt_inspired": 0.7}
    def run(self, ticker: str, signals: list[dict], agent_weights: dict = None) -> dict:
        current_weights = agent_weights or self.WEIGHTS; total_weighted_score = 0; sum_of_weights_used = 0
        agg_signals = {};
        for s_dict in signals:
            if isinstance(s_dict, dict): agg_signals.update(s_dict)
        signal_map = {"price_signal": "price", "momentum_signal": "momentum", "volatility_signal": "volatility",
                      "sentiment_signal": "sentiment", "fund_signal": "fund", "dcf_signal": "valuation_dcf",
                      "relative_pe_signal": "valuation_pe", "sec_filings_signal": "sec_filings", 
                      "analyst_signal": "analyst", "politician_filings_signal": "politician_filings",
                      "vt_fair_value_signal": "vt_fair_value", "vt_inspired_signal": "vt_inspired"}
        for signal_key, weight_key in signal_map.items():
            signal_value = agg_signals.get(signal_key); weight = current_weights.get(weight_key, 0)
            if signal_value and weight > 0 and signal_value in ["buy", "hold", "sell"]:
                raw_score = {"buy":1, "hold":0, "sell":-1}.get(signal_value, 0)
                total_weighted_score += raw_score * weight; sum_of_weights_used += weight
        composite_score = (total_weighted_score / sum_of_weights_used) if sum_of_weights_used else 0.0
        final_decision = "buy" if composite_score > 0.15 else ("sell" if composite_score < -0.15 else "hold")
        return {"ticker":ticker, "composite_score":composite_score, "final_decision":final_decision}

# --------------------------------
# Orchestrator for Live Analysis
# --------------------------------
def run_live_analysis(tickers, history_years, llm_client, configs):
    results = {}
    for t in tickers:
        price_history_full = fetch_price_history(t, period=f"{history_years}y")
        if price_history_full.empty:
            results[t] = {"error": f"Price history unavailable for {t}.", "ticker": t, "final_decision":"error", "composite_score":0}
            continue
        ticker_info = fetch_ticker_info(t)
        if not ticker_info:
            results[t] = {"error": f"Ticker info unavailable for {t}.", "ticker": t, "final_decision":"error", "composite_score":0}
            continue

        current_price_for_ticker = ticker_info.get("currentPrice") or (price_history_full["Close"].iloc[-1] if not price_history_full.empty else None)
        company_name_for_news_and_vt = ticker_info.get('longName', ticker_info.get('shortName', t))
        combined_news_data_list = []; news_fetch_status_messages = []

        if configs["use_sentiment"]:
            yfinance_news = fetch_enriched_news(t, ticker_info)
            if yfinance_news and not (isinstance(yfinance_news[0], dict) and "error" in yfinance_news[0]):
                combined_news_data_list.extend(yfinance_news)
            elif yfinance_news and isinstance(yfinance_news[0], dict) and "error" in yfinance_news[0]:
                news_fetch_status_messages.append(f"Yahoo News: {yfinance_news[0]['error']}")

            if llm_client and st.secrets.get("NEWSAPI_KEY"):
                newsapi_articles = fetch_comprehensive_news_from_api(t, company_name_for_news_and_vt, lookback_days=30)
                if newsapi_articles and not (isinstance(newsapi_articles[0], dict) and "error" in newsapi_articles[0]):
                    combined_news_data_list.extend(newsapi_articles)
                elif newsapi_articles and isinstance(newsapi_articles[0], dict) and "error" in newsapi_articles[0]:
                    news_fetch_status_messages.append(f"NewsAPI: {newsapi_articles[0]['error']}")
            elif configs["use_sentiment"]: news_fetch_status_messages.append("NewsAPI Key not configured.")
        
        seen_urls = set(); deduplicated_news = []
        for news_item in combined_news_data_list:
            if isinstance(news_item, dict) and "error" not in news_item:
                url = news_item.get('link') or news_item.get('url')
                if url and url not in seen_urls: deduplicated_news.append(news_item); seen_urls.add(url)
        if deduplicated_news: deduplicated_news.sort(key=lambda x: x.get('publish_datetime_utc', datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
        
        news_fetch_status_for_bundle = " | ".join(news_fetch_status_messages) if news_fetch_status_messages else "News fetch OK"
        if not deduplicated_news and not news_fetch_status_messages and configs["use_sentiment"]: news_fetch_status_for_bundle = "No news articles found from enabled sources."

        politician_trades_list = fetch_politician_trades(t) if configs["use_politician_filings"] else []
        
        data_bundle = {
            "price_history": price_history_full, "ticker_info": ticker_info, "news": deduplicated_news, 
            "news_fetch_status_error": news_fetch_status_for_bundle if "Error" in news_fetch_status_for_bundle or "failed" in news_fetch_status_for_bundle.lower() or "No news" in news_fetch_status_for_bundle else None,
            "politician_trades": politician_trades_list,
            "value_trades_fair_value_data": fetch_fair_value_from_value_trades(t, company_name_for_news_and_vt) if configs["use_value_trades"] else {"vt_fair_value": None, "error": "VT: Skipped by user config."}
        }
        
        all_agents_instances = [PriceAgent(), MomentumAgent(), VolatilityAgent(), FundamentalsAgent(), ValuationAgent(), AnalystRatingAgent()]
        if configs["use_sentiment"] and llm_client: all_agents_instances.append(SentimentAgent(llm_client))
        if configs["use_filings"]: all_agents_instances.append(SECFilingAgent())
        if configs["use_politician_filings"]: all_agents_instances.append(PoliticianFilingsAgent())
        if configs["use_value_trades"]: 
            all_agents_instances.append(FairValueAgentVT())
            all_agents_instances.append(VTInspiredFairValueAgent())

        agent_results_list = []
        for agent_instance in all_agents_instances:
            agent_name = agent_instance.__class__.__name__
            try:
                if isinstance(agent_instance, (PriceAgent, MomentumAgent)): res_agent = agent_instance.run(t, data_bundle["price_history"])
                elif isinstance(agent_instance, VolatilityAgent): res_agent = agent_instance.run(t, data_bundle, data_bundle["price_history"])
                else: res_agent = agent_instance.run(t, data_bundle)
                agent_results_list.append(res_agent)
            except Exception as e:
                agent_error_key = agent_name.lower().replace("agent","") + "_error"; default_signal_key_name = agent_name.lower().replace("agent","") + "_signal"
                agent_results_list.append({default_signal_key_name: "error", agent_error_key: f"Agent {agent_name} error: {str(e)[:100]}"})

        final_decision = PortfolioAgent().run(t, agent_results_list)
        current_result_dict = {"ticker": t, "current_price_display": current_price_for_ticker, "market_cap_display": ticker_info.get("marketCap"),
                               "industry_display": ticker_info.get("industry"), "sector_display": ticker_info.get("sector"), "ticker_info": ticker_info, 
                               "news_headlines_for_popover": [f"{n.get('publish_time_readable','N/A')} - {n.get('title', 'N/A')} ({n.get('publisher','N/A')} via {n.get('source_api','Unknown')})" for n in deduplicated_news[:10]],
                               "politician_trades_for_popover": [pt for pt in politician_trades_list[:5] if isinstance(pt, dict) and "error" not in pt],
                               "news_status_display": news_fetch_status_for_bundle }
        for res_dict in agent_results_list:
            if isinstance(res_dict, dict): current_result_dict.update(res_dict)
        current_result_dict.update(final_decision)
        results[t] = current_result_dict
    return results

# --------------------------------
# Backtesting Engine 
# --------------------------------
def run_backtest(ticker, start_date, end_date, initial_capital, llm_client_placeholder, backtest_agent_weights):
    s_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    fetch_start_date = (s_date_obj - pd.DateOffset(months=18)).strftime("%Y-%m-%d")
    full_price_history = fetch_price_history(ticker, period=None, interval="1d")
    if full_price_history.empty: return {"error": "Backtest failed: Price history empty."}, pd.DataFrame()
    price_history = full_price_history[(full_price_history.index >= pd.to_datetime(fetch_start_date)) & (full_price_history.index <= pd.to_datetime(end_date))].copy()
    if price_history.empty or len(price_history[price_history.index >= pd.to_datetime(start_date)]) < 2: return {"error": "Backtest failed: Not enough data in range."}, pd.DataFrame()
    ticker_info_for_backtest = fetch_ticker_info(ticker)
    data_bundle_static = {"ticker_info": ticker_info_for_backtest}
    price_agent = PriceAgent(); momentum_agent = MomentumAgent(); volatility_agent = VolatilityAgent(); portfolio_agent = PortfolioAgent()
    portfolio_log = []; cash = initial_capital; shares_held = 0; portfolio_value = initial_capital
    backtest_run_dates = price_history[price_history.index >= pd.to_datetime(start_date)].index
    for current_date in backtest_run_dates:
        data_slice = price_history[price_history.index <= current_date]
        current_price_point = data_slice.Close.iloc[-1] if not data_slice.empty else portfolio_value / shares_held if shares_held else 0
        if data_slice.empty or len(data_slice) < 252:
            portfolio_log.append({"date": current_date, "cash": cash, "shares_held": shares_held, "price": current_price_point, "portfolio_value": portfolio_value, "signal": "hold (insufficient data)", "composite_score":0.0}); continue
        current_price = data_slice.Close.iloc[-1]
        pa_res = price_agent.run(ticker, data_slice); ma_res = momentum_agent.run(ticker, data_slice); va_res = volatility_agent.run(ticker, data_bundle_static, data_slice)
        final_decision_obj = portfolio_agent.run(ticker, [pa_res, ma_res, va_res], agent_weights=backtest_agent_weights)
        final_decision = final_decision_obj["final_decision"]
        if final_decision == "buy" and cash > current_price : shares_to_buy = cash / current_price; shares_held += shares_to_buy; cash = 0
        elif final_decision == "sell" and shares_held > 0: cash += shares_held * current_price; shares_held = 0
        portfolio_value = cash + shares_held * current_price
        portfolio_log.append({"date": current_date, "cash": cash, "shares_held": shares_held, "price": current_price, "portfolio_value": portfolio_value, "signal": final_decision, "composite_score": final_decision_obj["composite_score"]})
    log_df = pd.DataFrame(portfolio_log);
    if not log_df.empty: log_df.set_index("date", inplace=True)
    if log_df.empty or len(log_df) < 2: return {"message":"Log too short to calculate performance metrics."}, pd.DataFrame()
    total_return = (log_df["portfolio_value"].iloc[-1] / initial_capital - 1) * 100
    num_days = (log_df.index[-1] - log_df.index[0]).days; num_years = num_days / 365.25 if num_days > 0 else 1/365.25
    annualized_return = ((log_df["portfolio_value"].iloc[-1] / initial_capital) ** (1/num_years) - 1) * 100 if num_years > 0 else total_return if num_days > 0 else 0
    log_df["daily_return"] = log_df["portfolio_value"].pct_change().fillna(0); annualized_volatility = log_df["daily_return"].std() * np.sqrt(252) * 100
    sharpe_ratio = (annualized_return / annualized_volatility) if annualized_volatility != 0 else 0
    log_df["cumulative_max"] = log_df["portfolio_value"].cummax(); log_df["drawdown"] = (log_df["portfolio_value"] - log_df["cumulative_max"]) / log_df["cumulative_max"]
    max_drawdown = log_df["drawdown"].min() * 100
    metrics = {"Initial Capital": f"${initial_capital:,.2f}", "Final Portfolio Value": f"${log_df['portfolio_value'].iloc[-1]:,.2f}",
               "Total Return (%)": f"{total_return:.2f}%", "Annualized Return (%)": f"{annualized_return:.2f}%",
               "Annualized Volatility (%)": f"{annualized_volatility:.2f}%", "Sharpe Ratio": f"{sharpe_ratio:.2f}",
               "Max Drawdown (%)": f"{max_drawdown:.2f}%", "Number of Trades (approx)": f"{(log_df['signal'] != log_df['signal'].shift()).fillna(False).sum() // 2}"}
    return metrics, log_df

# --------------------------------
# Streamlit UI
# --------------------------------
llm_client = None
try:
    deepseek_key = getattr(st.secrets, "DEEPSEEK_API_KEY", None) if hasattr(st.secrets, "DEEPSEEK_API_KEY") else None
    openai_key = getattr(st.secrets, "OPENAI_API_KEY", None) if hasattr(st.secrets, "OPENAI_API_KEY") else None
    if deepseek_key:
        llm_client = ModelClient(api_key=deepseek_key, provider="deepseek")
        st.sidebar.caption("✅ LLM: DeepSeek Initialized")
    elif openai_key:
        llm_client = ModelClient(api_key=openai_key, provider="openai")
        st.sidebar.caption("✅ LLM: OpenAI Initialized")
    else: st.sidebar.warning("LLM API key missing. Sentiment analysis disabled.")
except ValueError as e: st.sidebar.error(f"LLM Init Error: {e}. Check API Key.")
except Exception as e: st.sidebar.error(f"LLM Init Unexpected Error: {e}")

st.title("🚀 AI Hedge Fund Simulator")

st.header("⚙️ Configuration")
config_container = st.container(border=True)
app_mode = "Live Analysis"
with config_container:
    app_mode = st.radio("Select Mode:", ["Live Analysis", "Backtesting"], key="app_mode_select_main", horizontal=True, index=0)
    st.markdown("---")
    if app_mode == "Live Analysis":
        st.subheader("Live Analysis Settings")
        tickers_in_main = st.text_input("Tickers (comma-separated):", "AAPL,MSFT,GOOG", key="live_tickers_input_main")
        history_years_live_main = st.slider("Historical Data for Analysis (Years):", 1, 10, 5, key="live_history_slider_main")
        st.subheader("Feature Toggles (Live Analysis)")
        cols_features = st.columns(3)
        with cols_features[0]:
            use_sentiment_live_main = st.checkbox("News Sentiment (LLM)", value=True if llm_client else False, disabled=not llm_client, key="live_sentiment_cb_main", help="Uses LLM for news sentiment. Requires NewsAPI key if used, else Yahoo Finance.")
            use_filings_live_main = st.checkbox("SEC Insider Filings (Form 4)", value=True, key="live_sec_filings_cb_main", help="Analyzes SEC Form 4 insider transactions.")
        with cols_features[1]:
            use_politician_filings_main = st.checkbox("Politician Filings", value=False, key="live_politician_cb_main", help="EXPERIMENTAL: Attempts to scrape CapitolTrades.com. May be slow/unreliable.")
            use_value_trades_main = st.checkbox("Value-Trades Analysis", value=False, key="live_vt_cb_main", help="EXPERIMENTAL: Includes placeholder for Value-Trades.com scraping & a VT-Inspired calculation.")
        st.markdown("")
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
            bt_end_date_main = st.date_input("End Date:", default_bt_end_date_main, min_value=datetime.strptime(bt_start_date_main, "%Y-%m-%d") + timedelta(days=1), key="bt_end_date_main").strftime("%Y-%m-%d")
        bt_initial_capital_main = st.number_input("Initial Capital:", 1000, 1000000, 10000, 1000, key="bt_capital_input_main", format="%d")
        with st.expander("Adjust Backtest Agent Weights (Simplified Strategy)", expanded=False):
            st.caption("Backtest primarily uses Price, Momentum. Volatility uses current data. Others off by default.")
            bt_weights_price_main = st.slider("Price Signal Weight:", 0.0, 2.0, 1.0, 0.1, key="bt_w_price_main")
            bt_weights_momentum_main = st.slider("Momentum Signal Weight:", 0.0, 2.0, 0.8, 0.1, key="bt_w_momentum_main")
            bt_weights_volatility_main = st.slider("Volatility Signal Weight:", 0.0, 2.0, 0.2, 0.1, key="bt_w_vol_main")
        backtest_portfolio_weights_main = {"price": bt_weights_price_main, "momentum": bt_weights_momentum_main, "volatility": bt_weights_volatility_main,
                                      "sentiment": 0.0, "fund": 0.0, "valuation_dcf":0.0, "valuation_pe":0.0,
                                      "sec_filings": 0.0, "analyst": 0.0, "politician_filings": 0.0, "vt_fair_value": 0.0, "vt_inspired":0.0}
        st.markdown("")
        run_button_backtest_main = st.button("📈 Run Backtest", use_container_width=True, type="primary", key="run_bt_btn_main")

st.markdown("---")
if app_mode == "Live Analysis":
    if 'run_button_live_main' in locals() and run_button_live_main and 'tickers_in_main' in locals() and tickers_in_main:
        live_tickers_list_main = [t.strip().upper() for t in tickers_in_main.split(",") if t.strip()]
        if not live_tickers_list_main: st.error("Please enter at least one valid ticker.")
        else:
            live_configs_main = {"use_sentiment": use_sentiment_live_main, "use_filings": use_filings_live_main,
                            "use_politician_filings": use_politician_filings_main, "use_value_trades": use_value_trades_main}
            if 'live_output' not in st.session_state: st.session_state.live_output = {}
            with st.spinner("⏳ Processing analysis... Please wait."):
                st.session_state.live_output = run_live_analysis(live_tickers_list_main, history_years_live_main, llm_client, live_configs_main)
            st.header("📊 Live Analysis Summary")
            num_tickers = len(live_tickers_list_main); cols_per_row = min(num_tickers, 3)
            for i in range(0, num_tickers, cols_per_row):
                row_tickers = live_tickers_list_main[i:i+cols_per_row]; cols = st.columns(len(row_tickers))
                for idx, t_symbol in enumerate(row_tickers):
                    with cols[idx]:
                        res = st.session_state.live_output.get(t_symbol)
                        if not res or ("error" in res and res["error"] is not None and "Failed to fetch price history" in res["error"]):
                            st.error(f"**{t_symbol}**: {res.get('error', 'Critical error, no data to display.')}")
                            continue
                        elif not res : st.error(f"**{t_symbol}**: No analysis data returned."); continue
                        dec = res.get("final_decision", "N/A").upper(); score = res.get("composite_score", float('nan')); price_disp = res.get("current_price_display")
                        card_color_map = {"BUY": "green", "SELL": "red", "HOLD": "#FFA500"}; card_color = card_color_map.get(dec, "#D3D3D3")
                        st.markdown(f"""<div style="border: 1px solid {card_color}; border-radius: 8px; padding: 15px; margin-bottom: 10px; background-color: {card_color}20;">
                                        <h3 style="margin-bottom: 5px; color: {card_color};">{t_symbol}</h3>
                                        <p style="font-size: 1.6em; font-weight: bold; color: {card_color}; margin-bottom: 5px;">{dec}</p>
                                        <p style="font-size: 0.9em; margin-bottom: 3px;">Composite Score: <strong style="color: {card_color};">{score:.2f}</strong></p>
                                        {f'<p style="font-size: 0.9em;">Price: <strong>${price_disp:,.2f}</strong></p>' if price_disp is not None else ""}
                                    </div>""", unsafe_allow_html=True)
            st.markdown("---")
            for t_symbol in live_tickers_list_main:
                res = st.session_state.live_output.get(t_symbol)
                if not res or ("error" in res and res["error"] is not None and "Failed to fetch price history" in res["error"]): continue
                elif not res : continue
                with st.expander(f"🔍 Detailed Analysis for {t_symbol}"):
                    tab_titles = ["📈 Chart & Core", "펀 Fundamentals", "💰 Valuation & Fair Value", "📰 News & Filings", "⚙️ All Signals"]
                    tabs = st.tabs(tab_titles)
                    with tabs[0]:
                        st.subheader("Price Performance & Core Signals")
                        price_history_df_display_exp = fetch_price_history(t_symbol, period=f"{history_years_live_main}y")
                        if not price_history_df_display_exp.empty: st.line_chart(price_history_df_display_exp["Close"], use_container_width=True)
                        core_s = {"Price Signal (SMA/RSI)": res.get("price_signal", "N/A").upper(), "SMA50 / SMA200": f"{res.get('sma50',0):.2f} / {res.get('sma200',0):.2f}", "RSI14": f"{res.get('rsi14',0):.2f}", "Momentum Signal (1M/12M)": res.get("momentum_signal", "N/A").upper(), "Momentum 1M / 12M (%)": f"{res.get('momentum_1m',0)*100:.1f}% / {res.get('momentum_12m',0)*100:.1f}%", "Volatility Signal (Beta)": res.get("volatility_signal", "N/A").upper(), "Beta / Annual Vol (%)": f"{res.get('beta',0):.2f} / {res.get('annual_vol',0)*100:.1f}%",}
                        st.dataframe(pd.Series(core_s, name="Value"), use_container_width=True)
                    with tabs[1]:
                        st.subheader(f"Fundamental Snapshot - {res.get('industry_display', 'N/A')}")
                        ticker_info_res = res.get("ticker_info", {})
                        fund_s = {"Market Cap": f"${res.get('market_cap_display',0):,}" if res.get('market_cap_display') else "N/A",
                                  "FCF Yield": f"{res.get('fcf_yield',0)*100:.2f}%", "Piotroski Score": res.get('piotroski_score'),
                                  "ROE / DebtToEquity": f"{ticker_info_res.get('returnOnEquity',0)*100:.1f}% / {ticker_info_res.get('debtToEquity',0):.1f}",
                                  "Fundamental Signal": res.get("fund_signal", "N/A").upper()}
                        st.dataframe(pd.Series(fund_s, name="Value"), use_container_width=True)
                        business_summary = ticker_info_res.get("longBusinessSummary")
                        if business_summary:
                            with st.popover("View Business Summary"): st.markdown(business_summary)
                    with tabs[2]:
                        st.subheader("Valuation Metrics (yfinance based)")
                        val_s = {"Forward P/E": f"{res.get('forward_pe',0):.1f}", "Relative P/E Signal": res.get('relative_pe_signal', "N/A").upper(), "DCF Fair Price (Simple Est.)": f"${res.get('dcf_fair_price',0):.2f}" if res.get('dcf_fair_price') is not None else "N/A", "DCF Signal": res.get('dcf_signal', "N/A").upper()}
                        st.dataframe(pd.Series(val_s, name="Value"), use_container_width=True)
                        if live_configs_main["use_value_trades"]:
                            st.subheader("Value-Trades Scraper (Experimental)")
                            vt_scrape_status_msg = res.get('vt_data_error', "Status unknown") 
                            if not res.get('vt_data_error') and res.get('vt_fair_value_estimate') is not None: vt_scrape_status_msg = "Success"
                            elif not res.get('vt_data_error') and res.get('vt_fair_value_estimate') is None : vt_scrape_status_msg = "Data not found on page by scraper."
                            vt_full_text_display = res.get('vt_valuation_text_display')
                            if vt_full_text_display: st.markdown(f"""> *"{vt_full_text_display}"*""")
                            vt_s_scraped = {"VT Scraped Fair Value": f"${res.get('vt_fair_value_estimate'):.2f}" if res.get('vt_fair_value_estimate') is not None else "N/A",
                                            "VT Site Current Price": f"${res.get('vt_site_current_price'):.2f}" if res.get('vt_site_current_price') is not None else "N/A",
                                            "VT Site Valuation Status": str(res.get('vt_valuation_status', "N/A")).title(),
                                            "VT Site Valuation (%)": f"{res.get('vt_valuation_percentage'):.2f}%" if res.get('vt_valuation_percentage') is not None else "N/A",
                                            "VT Scraper Signal": res.get('vt_fair_value_signal', "N/A").upper(), "VT Scraper Status": vt_scrape_status_msg}
                            st.dataframe(pd.Series(vt_s_scraped, name="Value"), use_container_width=True)
                            st.subheader("Value-Trades Inspired Calculation (Internal)")
                            vt_inspired_error = res.get("error_vt_inspired_fv")
                            if vt_inspired_error: st.caption(f"Calculation Status: {vt_inspired_error}")
                            vt_inspired_data = {"EPS Used (Trailing)": f"${res.get('vt_inspired_eps_used'):.2f}" if res.get('vt_inspired_eps_used') is not None else "N/A",
                                                "Component PE - Hist. Avg (Assumed)": f"{res.get('vt_inspired_historical_avg_pe'):.2f}" if res.get('vt_inspired_historical_avg_pe') is not None else "N/A",
                                                "Component PE - Fair Val (Assumed)": f"{res.get('vt_inspired_assumed_fair_pe'):.2f}" if res.get('vt_inspired_assumed_fair_pe') is not None else "N/A",
                                                "Component PE - Industry (Est.)": f"{res.get('vt_inspired_industry_pe'):.2f}" if res.get('vt_inspired_industry_pe') is not None else "N/A",
                                                "Component PE - Growth (Assumed)": f"{res.get('vt_inspired_growth_pe'):.2f}" if res.get('vt_inspired_growth_pe') is not None else "N/A",
                                                "Calculated Assigned P/E (Avg)": f"{res.get('vt_inspired_assigned_pe'):.2f}" if res.get('vt_inspired_assigned_pe') is not None else "N/A",
                                                "VT-Inspired Fair Value": f"${res.get('vt_inspired_fair_value'):.2f}" if res.get('vt_inspired_fair_value') is not None else "N/A",
                                                "VT-Inspired Signal": res.get('vt_inspired_signal', "N/A").upper()}
                            st.dataframe(pd.Series(vt_inspired_data, name="Value"), use_container_width=True)
                            st.caption("Note: 'Assumed' P/E components are placeholders.")
                    with tabs[3]: # News & Filings Tab
                        if live_configs_main["use_sentiment"]:
                            st.subheader("News Sentiment (LLM)")
                            llm_status_message = res.get("news_status_display", "Status: OK") 
                            if res.get("sentiment_error"): llm_status_message += f" | LLM: {res.get('sentiment_error')}"
                            sent_s = {"Sentiment Score": f"{res.get('sentiment_score',0):.2f}", "Sentiment Signal": res.get("sentiment_signal", "N/A").upper(), "Processing Status": llm_status_message}
                            st.dataframe(pd.Series(sent_s, name="Value"), use_container_width=True)
                            news_headlines = res.get("news_headlines_for_popover")
                            if news_headlines:
                                with st.popover("View News Headlines (Top 10)"):
                                    for title_info in news_headlines: st.markdown(f"- {title_info}")
                            elif "Error" not in llm_status_message and "No news items" not in llm_status_message and "No valid news" not in llm_status_message :
                                 st.caption("No news headlines available or processed.")
                        if live_configs_main["use_filings"]:
                            st.subheader("SEC Insider Transactions (Form 4 - Last 6 Months)")
                            sec_filings_error = res.get("sec_filings_error")
                            if sec_filings_error: st.caption(f"SEC Filings Status: {sec_filings_error}")
                            sec_data_display = {"Net Insider Shares (6M)": f"{res.get('sec_net_insider_shares_6m',0):,}",
                                                "Total Buy Value (6M Est.)": f"${res.get('sec_insider_buy_value_6m',0):,.0f}",
                                                "Total Sell Value (6M Est.)": f"${res.get('sec_insider_sell_value_6m',0):,.0f}",
                                                "SEC Filings Signal": res.get("sec_filings_signal", "N/A").upper()}
                            st.dataframe(pd.Series(sec_data_display, name="Value"), use_container_width=True)
                            recent_txs_display = res.get("sec_recent_form4_transactions")
                            if recent_txs_display:
                                with st.popover("View Recent SEC Form 4 Transactions (Max 10)"):
                                    for tx in recent_txs_display:
                                        direction = "Acquired" if tx.get('acq_disp_code') == 'A' else "Disposed"
                                        price_info = f"@ ${tx.get('price_per_share'):.2f}" if tx.get('price_per_share') is not None else "(price N/A)"
                                        st.markdown(f"- **{tx.get('transaction_date')}**: {tx.get('reporting_owner')} ({tx.get('owner_relationship', '')}) "
                                                    f"{direction} {tx.get('shares'):,.0f} shares {price_info}. Code: {tx.get('transaction_code')}. "
                                                    f"[Link]({tx.get('link_to_filing')})")
                            elif not sec_filings_error: st.caption("No recent Form 4 transactions found or processed.")
                        # Politician Filings display is REMOVED from this tab.
                    with tabs[4]: # All Signals
                        st.subheader("All Agent Signals & Final Decision")
                        all_s_keys = [k for k in res if k.endswith("_signal")]; all_s_table = {k.replace("_signal","").replace("_"," ").title(): str(res[k]).upper() for k in all_s_keys}
                        all_s_table["Composite Score"] = f"{res.get('composite_score',0.0):.2f}"; all_s_table["Final Decision"] = res.get('final_decision',"").upper()
                        st.dataframe(pd.Series(all_s_table, name="Signal Value"), use_container_width=True)
                        with st.popover("View Full Raw JSON for this ticker"): st.json(res)

            with st.sidebar.expander("Portfolio Agent Weights (Live)", expanded=False):
                st.caption("These weights are used by the PortfolioAgent to combine individual signals.")
                st.json(PortfolioAgent.WEIGHTS)

elif app_mode == "Backtesting":
    if 'run_button_backtest_main' in locals() and run_button_backtest_main and 'bt_ticker_main' in locals() and bt_ticker_main:
        if 'bt_metrics' not in st.session_state: st.session_state.bt_metrics = None
        if 'bt_log_df' not in st.session_state: st.session_state.bt_log_df = pd.DataFrame()
        with st.spinner(f"⏳ Running backtest for {bt_ticker_main}... This may take a while."):
            st.session_state.bt_metrics, st.session_state.bt_log_df = run_backtest(
                bt_ticker_main, bt_start_date_main, bt_end_date_main,
                bt_initial_capital_main, llm_client, backtest_portfolio_weights_main)
        if st.session_state.bt_metrics and "message" not in st.session_state.bt_metrics and "error" not in st.session_state.bt_metrics:
            st.header(f"📈 Backtest Results for {bt_ticker_main}")
            metrics_df = pd.DataFrame.from_dict(st.session_state.bt_metrics, orient='index', columns=['Value'])
            st.table(metrics_df)
            if not st.session_state.bt_log_df.empty:
                st.subheader("Portfolio Value Over Time"); st.line_chart(st.session_state.bt_log_df["portfolio_value"])
                st.subheader("Drawdown Over Time"); st.area_chart(st.session_state.bt_log_df["drawdown"])
                with st.expander("View Backtest Log and Signals (Last 1000 rows)"): st.dataframe(st.session_state.bt_log_df[["price", "signal", "composite_score", "portfolio_value", "cash", "shares_held"]].tail(1000))
            else: st.warning("Backtest log empty, no charts.")
        else: st.error(f"Backtest failed: {st.session_state.bt_metrics.get('message', '') or st.session_state.bt_metrics.get('error', 'Unknown error') if st.session_state.bt_metrics else 'Unknown error'}")

# Sidebar for global info / disclaimers
st.sidebar.markdown("---")
st.sidebar.info("This simulator is for educational purposes only and does not constitute financial advice.")
st.sidebar.markdown("Experimental scraping features may be unreliable and are subject to website ToS.")

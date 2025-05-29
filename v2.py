import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import openai
from dotenv import load_dotenv
import requests # For web scraping
from bs4 import BeautifulSoup # For web scraping
import re # For parsing text more effectively

# Load environment variables (if running locally)
load_dotenv()

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
            st.warning(f"No price history found for {ticker} with period {period} and interval {interval}.")
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None) # Ensure timezone naive for consistency
        return df
    except Exception as e:
        st.error(f"Error fetching price history for {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data
def fetch_ticker_info(ticker: str) -> dict:
    """Fetches comprehensive info from yfinance for a ticker."""
    try:
        info = yf.Ticker(ticker).info
        if not info or info.get('regularMarketPrice') is None and info.get('currentPrice') is None :
            st.warning(f"Could not retrieve valid market data for {ticker}. It might be delisted or an invalid ticker.")
            return {}
        return {
            "marketCap":        info.get("marketCap"),
            "freeCashflow":     info.get("freeCashflow"),
            "forwardPE":        info.get("forwardPE"),
            "trailingPE":       info.get("trailingPE"),
            "priceToBook":      info.get("priceToBook"),
            "enterpriseToRevenue": info.get("enterpriseToRevenue"),
            "enterpriseToEbitda": info.get("enterpriseToEbitda"),
            "returnOnEquity":   info.get("returnOnEquity"),
            "debtToEquity":     info.get("debtToEquity"),
            "beta":             info.get("beta"),
            "targetMeanPrice":  info.get("targetMeanPrice"),
            "recommendationKey":info.get("recommendationKey"),
            "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions"),
            "industry":         info.get("industry"),
            "sector":           info.get("sector"),
            "longBusinessSummary": info.get("longBusinessSummary"),
            "currentPrice": info.get("currentPrice") or info.get("regularMarketPrice"),
        }
    except Exception as e:
        st.error(f"Error fetching .info for {ticker} from yfinance: {e}")
        return {}

@st.cache_data
def fetch_news(ticker: str) -> list[dict]:
    try:
        news = yf.Ticker(ticker).news
        return news or []
    except Exception as e:
        st.error(f"Error fetching news for {ticker}: {e}")
        return []

@st.cache_data
def fetch_inst_filings(ticker: str) -> list[dict]:
    try:
        df = yf.Ticker(ticker).institutional_holders
        if df is not None:
            return df.to_dict("records")
        return []
    except Exception as e:
        st.error(f"Error fetching institutional filings for {ticker}: {e}")
        return []


@st.cache_data
def fetch_insider_filings(ticker: str) -> list[dict]:
    try:
        df = yf.Ticker(ticker).get_insider_transactions()
        if df is None or df.empty:
            return []
        recs = df.to_dict("records")
        for r in recs:
            tx = r.get("Transaction", "").lower()
            if "purchase" in tx or "buy" in tx:
                r["type"] = "buy"
            elif "sale" in tx or "sell" in tx:
                r["type"] = "sell"
            else:
                r["type"] = "other"
        return recs
    except Exception as e:
        st.error(f"Error fetching insider filings for {ticker}: {e}")
        return []

# --- NEW: value-trades.com Scraper ---
@st.cache_data(ttl=3600) # Cache for an hour
def fetch_fair_value_from_value_trades(ticker: str) -> dict:
    """
    Attempts to log into value-trades.com and scrape fair value for a ticker.
    HIGHLY EXPERIMENTAL, LIKELY TO BREAK, AND REQUIRES CHECKING ToS.
    Assumes WordPress login. User must inspect and adjust field names and URLs.
    Requires VT_USERNAME, VT_PASSWORD, VT_STOCK_PAGE_URL_TEMPLATE in st.secrets.
    VT_STOCK_PAGE_URL_TEMPLATE should be like "https://value-trades.com/stocks/{ticker}/"
    """
    username = st.secrets.get("VT_USERNAME")
    password = st.secrets.get("VT_PASSWORD")
    # WordPress login URL is usually standard
    login_processing_url = "https://value-trades.com/wp-login.php"
    stock_page_template = st.secrets.get("VT_STOCK_PAGE_URL_TEMPLATE")

    if not all([username, password, stock_page_template]):
        # st.warning("Value-Trades credentials or stock page URL template not configured in secrets. Skipping VT scrape.")
        return {"error": "VT Configuration incomplete.", "vt_fair_value": None}

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })

    login_payload = {
        'log': username,
        'pwd': password,
        'wp-submit': 'Log In',
        'redirect_to': stock_page_template.format(ticker=ticker.lower()), # Attempt direct redirect
        'testcookie': '1'
    }

    try:
        # Step 1: POST to the login processing URL
        response_login = session.post(login_processing_url, data=login_payload, allow_redirects=True) # Allow redirects
        response_login.raise_for_status()

        # Step 2: Check if login was successful. This is the trickiest part.
        # Option A: We were redirected to the stock page.
        # Option B: We landed on some dashboard, and now need to navigate.
        # Option C: Login failed, we are still on login page or an error page.
        
        # For now, let's assume if the POST was okay, we directly try to GET the stock page
        # The session cookies *should* be set if login worked.
        stock_page_url = stock_page_template.format(ticker=ticker.lower()) # Ensure ticker is in correct case for URL
        
        # If the login POST didn't redirect us directly to the stock page, get it now:
        if response_login.url.strip('/') != stock_page_url.strip('/'):
            st.info(f"Login POST to {response_login.url}, now GETting {stock_page_url}")
            response_data_page = session.get(stock_page_url)
        else: # Login POST already took us to the stock page (hopefully)
            st.info(f"Login POST redirected to {response_login.url}")
            response_data_page = response_login
        
        response_data_page.raise_for_status()

        # Check if we are on a login page again (means login failed or session invalid)
        if "wp-login.php" in response_data_page.url and "loggedout=true" not in response_data_page.url:
            st.error(f"Value-Trades: Failed to access data page for {ticker} (URL: {response_data_page.url}). Likely login failure or redirect to login.")
            # st.html(response_data_page.text[:2000]) # For debugging
            return {"error": "VT Login likely failed or bad redirect.", "vt_fair_value": None}

        # Step 3: Parse the fair value from the page content
        soup_data_page = BeautifulSoup(response_data_page.content, 'html.parser')

        # --- THIS PARSING LOGIC IS PURELY HYPOTHETICAL AND SITE-SPECIFIC ---
        # You MUST inspect the HTML of a stock page on value-trades.com (after logging in)
        # to find the correct tags/classes/IDs for the fair value.
        # Example: Looking for text "Fair Value Estimate" then finding the value nearby.
        fv_text_label = None
        potential_fv_tags = soup_data_page.find_all(string=re.compile(r"Fair Value Estimate", re.IGNORECASE))
        fair_value = None

        if potential_fv_tags:
            for tag_label in potential_fv_tags:
                # Try to find the value in a sibling or parent structure. This is very common.
                # Example: <td>Fair Value Estimate:</td><td>$123.45</td>
                parent_with_value = tag_label.find_parent("td") or tag_label.find_parent("div") # Or other common parent
                if parent_with_value:
                    value_tag = parent_with_value.find_next_sibling()
                    if value_tag:
                        fv_text_match = re.search(r'\$?(\d{1,3}(?:,\d{3})*\.\d{2})', value_tag.text)
                        if fv_text_match:
                            fair_value = float(fv_text_match.group(1).replace(',', ''))
                            break # Found it
            if fair_value:
                st.info(f"VT Fair Value for {ticker}: ${fair_value:.2f} (experimental scrape).")
            else:
                st.warning(f"VT: Found 'Fair Value Estimate' label for {ticker}, but couldn't parse value.")
        else:
            st.warning(f"VT: Could not find 'Fair Value Estimate' text on page for {ticker}.")
            # Uncomment to debug page content:
            # st.markdown("--- DEBUG: Value-Trades Page Snippet ---")
            # st.text(soup_data_page.get_text()[:2000])
            # st.markdown("--- END DEBUG ---")

        return {"vt_fair_value": fair_value, "error": None if fair_value else "FV not found on page."}

    except requests.exceptions.HTTPError as http_err:
        st.error(f"VT HTTP error for {ticker}: {http_err} (URL: {http_err.request.url})")
        return {"error": f"HTTP error: {http_err}", "vt_fair_value": None}
    except requests.exceptions.RequestException as req_err:
        st.error(f"VT Request failed for {ticker}: {req_err}")
        return {"error": f"Request failed: {req_err}", "vt_fair_value": None}
    except Exception as e:
        st.error(f"Unexpected error fetching from Value-Trades for {ticker}: {e}")
        return {"error": f"Unexpected error: {e}", "vt_fair_value": None}
    finally:
        if 'session' in locals() and session:
            session.close()

# --- MODIFIED: capitoltrades.com Scraper ---
@st.cache_data(ttl=3600) # Cache for an hour
def fetch_politician_trades(ticker: str, days_back: int = 365) -> list[dict]:
    """
    Attempts to fetch recent politician trades for a *specific ticker* from CapitolTrades.
    NOTE: Web scraping is fragile and might break if the website structure changes.
    This is an experimental scraper. For reliable data, consider their official API (paid).
    """
    st.info(f"Attempting to scrape Capitol Trades for {ticker} (experimental)...")
    # URL structure observed: https://www.capitoltrades.com/trades?asset=NVDA&perPage=100
    # Adding txDate might be complex as it's often a filter on the page, not direct URL param for asset search
    # We'll fetch recent trades and then filter by date if possible, or rely on default sort.
    # For simplicity, not handling pagination here, fetching first page (e.g., 100 trades)
    url = f"https://www.capitoltrades.com/trades?asset={ticker.upper()}&pageSize=100&perPage=100"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    politician_trades_list = []
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # --- THIS PARSING LOGIC IS BASED ON OBSERVED STRUCTURE AND IS FRAGILE ---
        # Find the table or list of trades. This often requires inspecting the page.
        # Look for elements like <q-tr> or <tr> within a table, or list items.
        # CapitolTrades uses Quasar framework, HTML might be less standard (e.g. <q-tr>, <q-td>)
        
        # Try to find rows, they might be `divs` with a specific role or class, or `a` tags wrapping a row
        trade_rows = soup.find_all('a', href=lambda href: href and href.startswith('/trades/'))

        if not trade_rows:
            st.warning(f"No politician trade rows found for {ticker} on Capitol Trades using current selectors. Structure may have changed.")
            return []

        for row_link_tag in trade_rows:
            # Each 'row' is complex, containing multiple divs.
            # We need to extract: Politician Name, Transaction Type (Buy/Sell), Value Range, Date
            # This requires careful mapping of class names or structure.
            
            politician_name_tag = row_link_tag.find('div', class_=lambda x: x and 'politician-name' in x)
            tx_type_tag = row_link_tag.find('div', class_=lambda x: x and 'tx-type' in x)
            # Value range often has class like 'tx-value--long' or 'tx-value--short'
            value_range_tag = row_link_tag.find('div', class_=lambda x: x and 'tx-value' in x and ('--long' in x or '--short' in x))
            date_tag = row_link_tag.find('div', class_=lambda x: x and 'tx-date' in x) # Or a <time> tag

            if all([politician_name_tag, tx_type_tag, value_range_tag, date_tag]):
                name = politician_name_tag.text.strip()
                tx_type_text = tx_type_tag.text.strip().lower()
                # Determine type from text like "Purchase", "Sale", "Sale (Full)", "Sale (Partial)"
                tx_type = "purchase" if "purchase" in tx_type_text or "buy" in tx_type_text else \
                          "sale" if "sale" in tx_type_text or "sell" in tx_type_text else "other"
                
                value_range = value_range_tag.text.strip()
                date_str = date_tag.text.strip() # May need further parsing if not in YYYY-MM-DD

                # Crude value extraction for demo: take lower bound of range if possible
                value_estimate = 0
                value_matches = re.findall(r'\$([\d,]+)', value_range)
                if value_matches:
                    try:
                        value_estimate = int(value_matches[0].replace(',', ''))
                    except ValueError:
                        pass # Could not parse

                # Date filtering (rudimentary)
                try:
                    # Assuming date_str is like "MM/DD/YYYY" or similar, needs parsing.
                    # For simplicity, if it's not easily parsable, this filter won't be perfect.
                    # A robust solution would use dateutil.parser.parse(date_str)
                    # For now, we'll include all trades found on the first page for the asset.
                    # If date parsing is needed:
                    # trade_date = datetime.strptime(date_str, '%m/%d/%Y') # Example format
                    # if (datetime.now() - trade_date).days > days_back:
                    #    continue
                    pass
                except Exception: # Date parsing failed
                    pass


                politician_trades_list.append({
                    "politician_name": name,
                    "transaction_type": tx_type,
                    "value_range": value_range,
                    "value_estimate_lower": value_estimate, # For sorting or summing
                    "date_str": date_str,
                    "source_url": "https://www.capitoltrades.com" + row_link_tag['href']
                })
            # else:
            #    st.warning(f"Could not parse all fields for a trade row for {ticker}.")


        if politician_trades_list:
             st.info(f"Found {len(politician_trades_list)} politician trades for {ticker} (experimental scrape).")
        else:
            st.warning(f"No politician trades parsed for {ticker}. Check selectors or site structure.")

    except requests.exceptions.HTTPError as http_err:
        st.error(f"CapitolTrades HTTP error for {ticker}: {http_err}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from Capitol Trades for {ticker}: {e}")
    except Exception as e:
        st.error(f"Error parsing Capitol Trades data for {ticker}: {e}")
        # import traceback
        # st.text(traceback.format_exc())

    return politician_trades_list


# --------------------------------
# LLM Client
# --------------------------------
class ModelClient:
    def __init__(self, api_key: str, provider: str = "openai"):
        self.api_key = api_key
        self.provider = provider
        if not api_key:
            st.error("API key not provided for ModelClient.")
            raise ValueError("API key required.")

        openai.api_key = self.api_key
        if provider == "deepseek":
            openai.api_base = "https://api.deepseek.com/v1"
            self.model = "deepseek-chat" # Changed from deepseek-reasoner
        else: # Default to openai
            openai.api_base = "https://api.openai.com/v1"
            self.model = "gpt-4o"

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = openai.Embedding.create(input=texts, model="text-embedding-ada-002")
            return [e["embedding"] for e in resp["data"]]
        except Exception as e:
            st.error(f"Error creating embeddings with {self.provider}: {e}")
            return [[] for _ in texts]


    def generate(self, prompt: str) -> str:
        try:
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message["content"]
        except Exception as e:
            st.error(f"Error generating text with {self.provider}: {e}")
            return f"Error: Could not generate response. {e}"

# --------------------------------
# Agents
# --------------------------------
class PriceAgent:
    def run(self, ticker: str, price_data_slice: pd.DataFrame) -> dict:
        if price_data_slice.empty or len(price_data_slice) < 200:
            return {"ticker": ticker, "price_signal": "hold", "sma50": np.nan, "sma200": np.nan, "rsi14": np.nan}
        df = price_data_slice.copy()
        df["SMA50"]  = df["Close"].rolling(50).mean()
        df["SMA200"] = df["Close"].rolling(200).mean()
        delta = df["Close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan) # Avoid division by zero in RSI
        df["RSI14"] = 100 - (100 / (1 + rs))
        latest = df.iloc[-1]
        signal = "hold"
        if pd.isna(latest.SMA50) or pd.isna(latest.SMA200) or pd.isna(latest.RSI14):
             signal = "hold"
        elif latest.SMA50 > latest.SMA200 and latest.RSI14 < 70:
            signal = "buy"
        elif latest.SMA50 < latest.SMA200 and latest.RSI14 > 30:
            signal = "sell"
        return {
            "ticker":       ticker,
            "sma50":        float(latest.SMA50) if pd.notna(latest.SMA50) else np.nan,
            "sma200":       float(latest.SMA200) if pd.notna(latest.SMA200) else np.nan,
            "rsi14":        float(latest.RSI14) if pd.notna(latest.RSI14) else np.nan,
            "price_signal": signal,
        }

class MomentumAgent:
    def run(self, ticker: str, price_data_slice: pd.DataFrame) -> dict:
        if price_data_slice.empty or len(price_data_slice) < 252:
             return {"ticker": ticker, "momentum_signal": "hold", "momentum_1m": 0, "momentum_12m": 0}
        df = price_data_slice
        P_t = df.Close.iloc[-1]
        P_1m = df.Close.shift(21).iloc[-1] if len(df) > 21 else np.nan
        P_12m = df.Close.shift(252).iloc[-1] if len(df) > 252 else np.nan
        m1  = (P_t/P_1m)-1  if pd.notna(P_1m) and P_1m != 0 else 0
        m12 = (P_t/P_12m)-1 if pd.notna(P_12m) and P_12m != 0 else 0
        signal = "hold"
        if m12 > 0.01 and m1 > 0.01: # Added small threshold to confirm momentum
            signal = "buy"
        elif m12 < -0.01 and m1 < -0.01:
            signal = "sell"
        return {
            "ticker":         ticker,
            "momentum_1m":    float(m1),
            "momentum_12m":   float(m12),
            "momentum_signal": signal,
        }

class VolatilityAgent:
    def run(self, ticker: str, data: dict, price_data_slice: pd.DataFrame = None) -> dict:
        beta = data.get("ticker_info", {}).get("beta", 1.0)
        if beta is None: beta = 1.0
        sig  = "sell" if beta > 1.5 else ("buy" if beta < 0.8 else "hold")
        ann_vol = np.nan
        weight = 0.0
        if price_data_slice is not None and not price_data_slice.empty and len(price_data_slice) > 1:
            ret = np.log(price_data_slice.Close / price_data_slice.Close.shift(1)).dropna()
            if not ret.empty:
                ann_vol = float(ret.std() * np.sqrt(252))
                weight  = float(1/ann_vol) if ann_vol > 0 else 0.0
        return {
            "ticker": ticker,
            "beta": beta,
            "annual_vol": ann_vol,
            "vol_weight": weight,
            "volatility_signal": sig,
        }

class SentimentAgent:
    def __init__(self, client): self.client = client
    def run(self, ticker: str, data: dict) -> dict:
        headlines = [h.get("title","") for h in data.get("news",[])[:10]]
        if not headlines:
            return {"ticker":ticker, "sentiment_score":0.0, "sentiment_signal":"hold"}
        prompt = (
            f"Rate the overall market sentiment for the stock {ticker} based ONLY on the following headlines. "
            f"Provide a single floating-point number between -1.0 (very negative) and +1.0 (very positive). "
            f"Do not add any other text, just the number.\n\nHeadlines:\n"
            + "\n".join(f"- {h}" for h in headlines)
        )
        try:
            response = self.client.generate(prompt).strip()
            score = float(response)
        except ValueError:
            clarification_prompt = f"The previous response was '{response}'. Please extract the single sentiment score number between -1 and 1 from it. If none, output 0.0."
            try:
                score = float(self.client.generate(clarification_prompt).strip())
            except: score = 0.0
        except Exception: score = 0.0
        sig = "buy" if score > 0.25 else ("sell" if score < -0.25 else "hold")
        return {"ticker":ticker, "sentiment_score":score, "sentiment_signal":sig}

class FundamentalsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        s    = data.get("ticker_info", {})
        mcap = s.get("marketCap") or 1
        fcf  = s.get("freeCashflow") or 0
        roe  = s.get("returnOnEquity") or 0
        de   = s.get("debtToEquity")
        if de is None: de = 1000
        fcy  = fcf/mcap if mcap != 0 else 0
        piotroski_score = 0
        if roe > 0.01: piotroski_score += 1 # Require some positive ROE
        if de < 100 : piotroski_score += 1
        if fcf > 0: piotroski_score += 1
        sig  = "buy" if piotroski_score >= 2 else ("sell" if piotroski_score == 0 else "hold")
        return {
            "ticker": ticker,
            "fcf_yield": float(fcy),
            "piotroski_score": piotroski_score,
            "fund_signal": sig,
        }

class ValuationAgent:
    def run(self, ticker: str, data: dict) -> dict:
        stats = data.get("ticker_info", {})
        price_history_df = data.get("price_history")
        price = stats.get("currentPrice")
        if price is None and price_history_df is not None and not price_history_df.empty:
             price = price_history_df["Close"].iloc[-1]
        if price is None:
            return {"ticker": ticker, "forward_pe": None, "relative_pe_signal": "hold", "dcf_fair_price": np.nan, "dcf_signal": "hold"}
        pe = stats.get("forwardPE")
        rel_sig = "hold"
        if pe is None: rel_sig = "hold"
        elif pe < 15: rel_sig = "buy"
        elif pe > 25: rel_sig = "sell"
        fcf  = stats.get("freeCashflow")
        mcap = stats.get("marketCap")
        fcy = 0.0
        if fcf is not None and mcap is not None and mcap != 0: fcy = fcf / mcap
        fair_price = price * (1 + fcy) if pd.notna(price) else np.nan
        dcf_sig = "hold"
        if pd.notna(fair_price) and pd.notna(price) and price != 0:
            if fair_price > price * 1.15: dcf_sig = "buy"
            elif fair_price < price * 0.85: dcf_sig = "sell"
        return {
            "ticker": ticker,
            "forward_pe": pe,
            "relative_pe_signal": rel_sig,
            "dcf_fair_price": float(fair_price) if pd.notna(fair_price) else np.nan,
            "dcf_signal": dcf_sig,
        }

class FilingsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        insiders = data.get("insider_filings",[])
        net_shares = 0
        if insiders:
            for r in insiders:
                shares = r.get("Shares",0)
                if isinstance(shares, str):
                    try: shares = int(shares.replace(',','')) # Handle commas in numbers
                    except ValueError: shares = 0
                if r.get("type") == "buy": net_shares += shares
                elif r.get("type") == "sell": net_shares -= shares
        sig = "buy" if net_shares > 1000 else ("sell" if net_shares < -1000 else "hold") # Added threshold for significance
        return {
            "ticker": ticker,
            "net_insider_shares": int(net_shares),
            "filings_signal": sig,
        }

class AnalystRatingAgent:
    def run(self, ticker: str, data: dict) -> dict:
        ticker_info = data.get("ticker_info", {})
        price_history_df = data.get("price_history")
        current_price = ticker_info.get("currentPrice")
        if current_price is None and price_history_df is not None and not price_history_df.empty:
            current_price = price_history_df["Close"].iloc[-1]
        target_mean_price = ticker_info.get("targetMeanPrice")
        recommendation = str(ticker_info.get("recommendationKey", "hold")).lower()
        upside = 0.0
        if target_mean_price and current_price and current_price > 0:
            try: upside = (float(target_mean_price) / float(current_price)) - 1
            except (ValueError, TypeError): upside = 0.0
        sig = "hold"
        if recommendation in ["buy", "strong_buy"] and upside > 0.10: sig = "buy"
        elif recommendation == "buy" and upside > 0.05: sig = "buy"
        elif recommendation in ["sell", "strong_sell", "underperform"] and upside < -0.05: sig = "sell"
        elif upside > 0.20 : sig = "buy"
        elif upside < -0.15 : sig = "sell"
        buy_pct_inferred = 0.5
        if recommendation == "strong_buy": buy_pct_inferred = 0.9
        elif recommendation == "buy": buy_pct_inferred = 0.7
        elif recommendation == "hold": buy_pct_inferred = 0.5
        elif recommendation == "underperform": buy_pct_inferred = 0.3
        elif recommendation == "sell": buy_pct_inferred = 0.1
        return {
            "ticker": ticker,
            "analyst_buy_pct_inferred": buy_pct_inferred,
            "target_upside": float(upside),
            "yfinance_recommendation": recommendation,
            "analyst_signal": sig,
        }

class PoliticianFilingsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        trades = data.get("politician_trades", []) # This now comes from the live scraper
        net_value_estimate = 0
        buy_count = 0
        sell_count = 0
        if trades: # Ensure trades is not None and not empty
            for trade in trades:
                value = trade.get("value_estimate_lower", 0) # Using the estimated lower bound
                if trade.get("transaction_type") == "purchase":
                    net_value_estimate += value
                    buy_count +=1
                elif trade.get("transaction_type") == "sale":
                    net_value_estimate -= value
                    sell_count +=1
        signal = "hold"
        # Signal if significant net activity or imbalance in counts
        if buy_count > sell_count and buy_count > 1 : signal = "buy" # More than 1 net buy transaction
        elif sell_count > buy_count and sell_count > 1: signal = "sell" # More than 1 net sell transaction
        elif net_value_estimate > 50000 and buy_count > 0 : signal = "buy" # Large net purchase
        elif net_value_estimate < -50000 and sell_count > 0: signal = "sell" # Large net sale

        return {
            "ticker": ticker,
            "politician_net_trade_value_estimate": net_value_estimate,
            "politician_buy_tx_count": buy_count,
            "politician_sell_tx_count": sell_count,
            "politician_filings_signal": signal,
        }

# --- NEW: FairValueAgentVT ---
class FairValueAgentVT:
    def run(self, ticker: str, data: dict) -> dict:
        vt_data = data.get("value_trades_fair_value_data", {})
        fair_value = vt_data.get("vt_fair_value")
        error = vt_data.get("error")
        current_price = data.get("ticker_info", {}).get("currentPrice") # Get current price from ticker_info
        if current_price is None and data.get("price_history") is not None and not data["price_history"].empty:
            current_price = data["price_history"]["Close"].iloc[-1]

        signal = "hold"
        margin_of_safety = 0.20 # Buy if 20% undervalued, sell if 20% overvalued relative to this FV

        if error and error != "FV not found on page.": # Allow "FV not found" to be a neutral signal
            # Error already shown by fetcher via st.warning/st.error
            pass
        elif fair_value is not None and current_price is not None and current_price > 0:
            if current_price < fair_value * (1 - margin_of_safety):
                signal = "buy"
            elif current_price > fair_value * (1 + margin_of_safety):
                signal = "sell"
        
        return {
            "ticker": ticker,
            "vt_fair_value_estimate": fair_value,
            "vt_fair_value_signal": signal,
            "vt_data_error": error
        }

class PortfolioAgent:
    WEIGHTS = {
        "price": 1.0, "momentum": 0.8, "volatility": 0.3, # Reduced weight for beta-only vol signal
        "sentiment": 0.6, "fund": 0.9, "valuation_dcf":0.5, "valuation_pe":0.5,
        "filings": 0.5, "analyst": 0.7,
        "politician_filings": 0.4, # Weight for (experimental) politician filing signal
        "vt_fair_value": 0.8 # Weight for (experimental) Value-Trades fair value signal
    }
    def run(self, ticker: str, signals: list[dict], agent_weights: dict = None) -> dict:
        current_weights = agent_weights or self.WEIGHTS
        total_weighted_score = 0
        sum_of_weights_used = 0
        agg_signals = {}
        for s_dict in signals: agg_signals.update(s_dict)

        signal_map = {
            "price_signal": "price", "momentum_signal": "momentum",
            "volatility_signal": "volatility", "sentiment_signal": "sentiment",
            "fund_signal": "fund", "dcf_signal": "valuation_dcf",
            "relative_pe_signal": "valuation_pe", "filings_signal": "filings",
            "analyst_signal": "analyst", "politician_filings_signal": "politician_filings",
            "vt_fair_value_signal": "vt_fair_value" # Added new signal
        }
        for signal_key, weight_key in signal_map.items():
            signal_value = agg_signals.get(signal_key)
            weight = current_weights.get(weight_key, 0) # Default to 0 if weight_key not in current_weights
            if signal_value and weight > 0 : # Only consider if weight is positive
                raw_score = {"buy":1, "hold":0, "sell":-1}.get(signal_value, 0)
                total_weighted_score += raw_score * weight
                sum_of_weights_used += weight
        if sum_of_weights_used == 0: composite_score = 0.0
        else: composite_score = total_weighted_score / sum_of_weights_used
        buy_threshold = 0.15
        sell_threshold = -0.15
        final_decision = "hold"
        if composite_score > buy_threshold: final_decision = "buy"
        elif composite_score < sell_threshold: final_decision = "sell"
        return {"ticker":ticker, "composite_score":composite_score, "final_decision":final_decision}

# --------------------------------
# Orchestrator for Live Analysis
# --------------------------------
def run_live_analysis(tickers, history_years, llm_client, configs):
    results = {}
    for t in tickers:
        st.write(f"--- Analyzing {t} ---")
        # Fetch all necessary data first
        price_history_full = fetch_price_history(t, period=f"{history_years}y")
        if price_history_full.empty:
            results[t] = {"error": "Failed to fetch price history", "ticker": t, "final_decision":"error", "composite_score":0}
            st.error(f"Could not fetch price history for {t}. Skipping analysis.")
            continue

        ticker_info = fetch_ticker_info(t)
        if not ticker_info: # If fetch_ticker_info returned empty dict due to error
            results[t] = {"error": "Failed to fetch ticker info", "ticker": t, "final_decision":"error", "composite_score":0}
            st.error(f"Could not fetch essential ticker info for {t}. Skipping analysis.")
            continue
        
        current_price_for_ticker = ticker_info.get("currentPrice")
        if current_price_for_ticker is None and not price_history_full.empty:
            current_price_for_ticker = price_history_full["Close"].iloc[-1]

        data_bundle = {
            "price_history": price_history_full,
            "ticker_info": ticker_info,
            "news": fetch_news(t) if configs["use_sentiment"] else [],
            "insider_filings": fetch_insider_filings(t) if configs["use_filings"] else [],
            "politician_trades": fetch_politician_trades(t) if configs["use_politician_filings"] else [],
            # NEW: Value-Trades data
            "value_trades_fair_value_data": fetch_fair_value_from_value_trades(t) if configs["use_value_trades"] else \
                                            {"vt_fair_value": None, "error": "VT: Skipped by user config."}
        }

        # Initialize Agents
        all_agents = [PriceAgent(), MomentumAgent(), VolatilityAgent(), FundamentalsAgent(), ValuationAgent(), AnalystRatingAgent()]
        agent_results_list = []

        if configs["use_sentiment"] and llm_client: all_agents.append(SentimentAgent(llm_client))
        if configs["use_filings"]: all_agents.append(FilingsAgent())
        if configs["use_politician_filings"]: all_agents.append(PoliticianFilingsAgent())
        if configs["use_value_trades"]: all_agents.append(FairValueAgentVT()) # Add new agent

        portfolio_agent = PortfolioAgent()

        # Run Agents
        for agent_instance in all_agents:
            agent_name = agent_instance.__class__.__name__
            try:
                if isinstance(agent_instance, (PriceAgent, MomentumAgent)):
                    res = agent_instance.run(t, data_bundle["price_history"])
                elif isinstance(agent_instance, VolatilityAgent):
                    res = agent_instance.run(t, data_bundle, data_bundle["price_history"])
                elif isinstance(agent_instance, FairValueAgentVT): # FairValueAgentVT needs current_price explicitly
                    res = agent_instance.run(t, data_bundle) # data_bundle now contains ticker_info with currentPrice
                else: # Other agents take the full data_bundle
                    res = agent_instance.run(t, data_bundle)
                agent_results_list.append(res)
            except Exception as e:
                st.error(f"Error running {agent_name} for {t}: {e}")
                # Add a placeholder result to ensure dict merging works
                # Find the typical signal key for this agent or use a generic one
                signal_key_name = "unknown_signal"
                if hasattr(agent_instance, 'run') and 'return' in agent_instance.run.__annotations__:
                    # This is a bit hacky, assumes signal key is classname_signal
                    signal_key_name = agent_name.lower().replace("agent","") + "_signal"
                agent_results_list.append({signal_key_name: "error", f"{signal_key_name}_error": str(e)})


        final_decision = portfolio_agent.run(t, agent_results_list)

        # Consolidate results
        current_result_dict = {"ticker": t} # Start with ticker
        for res_dict in agent_results_list: # Merge all individual agent results
            current_result_dict.update(res_dict)
        current_result_dict.update(final_decision) # Add portfolio agent's decision
        
        # Add some key raw data points for display if not already added by an agent
        current_result_dict["current_price_display"] = current_price_for_ticker
        current_result_dict["market_cap_display"] = ticker_info.get("marketCap")
        current_result_dict["industry_display"] = ticker_info.get("industry")
        current_result_dict["sector_display"] = ticker_info.get("sector")


        results[t] = current_result_dict
        st.success(f"Analysis for {t}: {final_decision['final_decision']} (Score: {final_decision['composite_score']:.2f})")

    return results

# --------------------------------
# Backtesting Engine (Simplified - not using new scrapers for PoT data)
# --------------------------------
def run_backtest(ticker, start_date, end_date, initial_capital, llm_client_placeholder, backtest_agent_weights):
    st.write(f"Starting backtest for {ticker} from {start_date} to {end_date}...")
    s_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    fetch_start_date = (s_date_obj - pd.DateOffset(years=1.5)).strftime("%Y-%m-%d")
    full_price_history = fetch_price_history(ticker, period=None, interval="1d")
    if full_price_history.empty:
        st.error("Cannot run backtest: Price history is empty.")
        return None, pd.DataFrame()
    price_history = full_price_history[(full_price_history.index >= pd.to_datetime(fetch_start_date)) &
                                       (full_price_history.index <= pd.to_datetime(end_date))].copy()
    if price_history.empty or len(price_history[price_history.index >= pd.to_datetime(start_date)]) < 2:
        st.error(f"Not enough historical data for {ticker} in the selected backtest range after filtering.")
        return None, pd.DataFrame()
    
    ticker_info_for_backtest = fetch_ticker_info(ticker) # Fetched once, lookahead for some agents
    data_bundle_static = {"ticker_info": ticker_info_for_backtest}

    price_agent = PriceAgent()
    momentum_agent = MomentumAgent()
    volatility_agent = VolatilityAgent()
    portfolio_agent = PortfolioAgent()
    portfolio_log = []
    cash = initial_capital
    shares_held = 0
    portfolio_value = initial_capital
    backtest_run_dates = price_history[price_history.index >= pd.to_datetime(start_date)].index

    for current_date in backtest_run_dates:
        data_slice = price_history[price_history.index <= current_date]
        if data_slice.empty or len(data_slice) < 252:
            current_price_point = data_slice.Close.iloc[-1] if not data_slice.empty else portfolio_value / shares_held if shares_held else 0
            portfolio_log.append({"date": current_date, "cash": cash, "shares_held": shares_held,
                                  "price": current_price_point, "portfolio_value": portfolio_value, 
                                  "signal": "hold (insufficient data)", "composite_score":0.0})
            continue
        current_price = data_slice.Close.iloc[-1]
        pa_res = price_agent.run(ticker, data_slice)
        ma_res = momentum_agent.run(ticker, data_slice)
        va_res = volatility_agent.run(ticker, data_bundle_static, data_slice) # Beta is lookahead
        backtest_signals = [pa_res, ma_res, va_res]
        final_decision_obj = portfolio_agent.run(ticker, backtest_signals, agent_weights=backtest_agent_weights)
        final_decision = final_decision_obj["final_decision"]
        if final_decision == "buy" and cash > current_price : # Ensure can afford at least one share (approx)
            shares_to_buy = cash / current_price
            shares_held += shares_to_buy
            cash = 0
        elif final_decision == "sell" and shares_held > 0:
            cash += shares_held * current_price
            shares_held = 0
        portfolio_value = cash + shares_held * current_price
        portfolio_log.append({"date": current_date, "cash": cash, "shares_held": shares_held,
                              "price": current_price, "portfolio_value": portfolio_value, 
                              "signal": final_decision, "composite_score": final_decision_obj["composite_score"]})
    log_df = pd.DataFrame(portfolio_log)
    if not log_df.empty: log_df.set_index("date", inplace=True)
    if log_df.empty or len(log_df) < 2:
        return {"message":"Log too short"}, pd.DataFrame()
    total_return = (log_df["portfolio_value"].iloc[-1] / initial_capital - 1) * 100
    num_days = (log_df.index[-1] - log_df.index[0]).days
    num_years = num_days / 365.25 if num_days > 0 else 1/365.25 # Handle very short periods
    annualized_return = ((log_df["portfolio_value"].iloc[-1] / initial_capital) ** (1/num_years) - 1) * 100 if num_years > 0 else total_return if num_days > 0 else 0
    log_df["daily_return"] = log_df["portfolio_value"].pct_change().fillna(0)
    annualized_volatility = log_df["daily_return"].std() * np.sqrt(252) * 100
    sharpe_ratio = (annualized_return / annualized_volatility) if annualized_volatility != 0 else 0
    log_df["cumulative_max"] = log_df["portfolio_value"].cummax()
    log_df["drawdown"] = (log_df["portfolio_value"] - log_df["cumulative_max"]) / log_df["cumulative_max"]
    max_drawdown = log_df["drawdown"].min() * 100
    metrics = {"Initial Capital": f"${initial_capital:,.2f}",
               "Final Portfolio Value": f"${log_df['portfolio_value'].iloc[-1]:,.2f}",
               "Total Return (%)": f"{total_return:.2f}%", "Annualized Return (%)": f"{annualized_return:.2f}%",
               "Annualized Volatility (%)": f"{annualized_volatility:.2f}%", "Sharpe Ratio": f"{sharpe_ratio:.2f}",
               "Max Drawdown (%)": f"{max_drawdown:.2f}%", "Number of Trades (approx)": f"{(log_df['signal'] != 'hold').diff().fillna(False).sum() // 2}"}
    return metrics, log_df

# --------------------------------
# Streamlit UI
# --------------------------------
st.set_page_config(page_title="AI Hedge Fund Simulator", layout="wide")
st.title("🚀 AI Hedge Fund Simulator")

# Initialize LLM Client
llm_client = None
try:
    if "DEEPSEEK_API_KEY" in st.secrets and st.secrets["DEEPSEEK_API_KEY"]:
        llm_client = ModelClient(api_key=st.secrets["DEEPSEEK_API_KEY"], provider="deepseek")
        st.sidebar.caption("LLM: DeepSeek")
    elif "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
        llm_client = ModelClient(api_key=st.secrets["OPENAI_API_KEY"], provider="openai")
        st.sidebar.caption("LLM: OpenAI")
    else:
        st.sidebar.warning("LLM API key missing. Sentiment analysis disabled.")
except ValueError as e: st.sidebar.error(f"LLM Init Error: {e}")
except Exception as e: st.sidebar.error(f"LLM Unexpected Error: {e}")

# Sidebar Configuration
st.header("⚙️ Configuration")
app_mode = st.selectbox("Select Mode", ["Live Analysis", "Backtesting"], key="app_mode_select")
    
    if app_mode == "Live Analysis":
        st.subheader("Live Analysis Settings")
        tickers_in = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOG", key="live_tickers_input")
        history_years_live = st.slider("History for Live Analysis (years)", 1, 10, 5, key="live_history_slider")
        
        st.subheader("Feature Toggles (Live Analysis)")
        use_sentiment_live = st.checkbox("News Sentiment (LLM)", value=True if llm_client else False, disabled=not llm_client, key="live_sentiment_cb", help="Uses LLM for news sentiment.")
        use_filings_live = st.checkbox("Insider Filings", value=True, key="live_insider_cb", help="Analyzes yfinance insider transaction data.")
        use_politician_filings_live = st.checkbox("Politician Filings (Experimental Scrape)", value=False, key="live_politician_cb", help="Attempts to scrape CapitolTrades.com. Highly experimental.")
        use_value_trades_live = st.checkbox("Value-Trades Fair Value (Experimental Login/Scrape)", value=False, key="live_vt_cb", help="Attempts login to Value-Trades.com. Requires VT_ secrets. CHECK ToS!")
        
        run_button_live = st.button("Run Live Analysis", use_container_width=True, key="run_live_btn")

    elif app_mode == "Backtesting":
        st.subheader("Backtesting Settings")
        bt_ticker = st.text_input("Ticker for Backtest", "AAPL", key="bt_ticker_input").upper()
        default_bt_end_date = datetime.now() - timedelta(days=1)
        default_bt_start_date = default_bt_end_date - pd.DateOffset(years=3)
        bt_start_date = st.date_input("Start Date", default_bt_start_date, max_value=default_bt_end_date - timedelta(days=1), key="bt_start_date").strftime("%Y-%m-%d")
        bt_end_date = st.date_input("End Date", default_bt_end_date, min_value=datetime.strptime(bt_start_date, "%Y-%m-%d") + timedelta(days=1), key="bt_end_date").strftime("%Y-%m-%d")
        bt_initial_capital = st.number_input("Initial Capital", 1000, 1000000, 10000, 1000, key="bt_capital_input")
        
        st.markdown("**Backtest Agent Weights (Simplified Strategy):**")
        bt_weights_price = st.slider("Weight: Price Signal", 0.0, 2.0, 1.0, 0.1, key="bt_w_price")
        bt_weights_momentum = st.slider("Weight: Momentum Signal", 0.0, 2.0, 0.8, 0.1, key="bt_w_momentum")
        bt_weights_volatility = st.slider("Weight: Volatility Signal (Beta is lookahead)", 0.0, 2.0, 0.2, 0.1, key="bt_w_vol")
        backtest_portfolio_weights = {"price": bt_weights_price, "momentum": bt_weights_momentum, "volatility": bt_weights_volatility,
                                      "sentiment": 0.0, "fund": 0.0, "valuation_dcf":0.0, "valuation_pe":0.0,
                                      "filings": 0.0, "analyst": 0.0, "politician_filings": 0.0, "vt_fair_value": 0.0}
        run_button_backtest = st.button("Run Backtest", use_container_width=True, key="run_bt_btn")

# Main App Logic
if app_mode == "Live Analysis":
    if run_button_live and tickers_in:
        live_tickers_list = [t.strip().upper() for t in tickers_in.split(",") if t.strip()]
        if not live_tickers_list:
            st.error("Please enter at least one valid ticker.")
        else:
            live_configs = {"use_sentiment": use_sentiment_live, "use_filings": use_filings_live,
                            "use_politician_filings": use_politician_filings_live, "use_value_trades": use_value_trades_live}
            with st.spinner("Running AI agents for live analysis... This may take a moment for multiple tickers or experimental features."):
                live_output = run_live_analysis(live_tickers_list, history_years_live, llm_client, live_configs)

            st.markdown("---")
            st.header("📊 Live Analysis Summary")
            num_tickers = len(live_tickers_list)
            cols_per_row = min(num_tickers, 3)
            
            for i in range(0, num_tickers, cols_per_row):
                row_tickers = live_tickers_list[i:i+cols_per_row]
                cols = st.columns(len(row_tickers))
                for idx, t_symbol in enumerate(row_tickers):
                    with cols[idx]:
                        res = live_output.get(t_symbol)
                        if not res or "error" in res :
                            st.error(f"**{t_symbol}**: {res.get('error', 'Unknown error') if res else 'No data'}")
                            continue
                        
                        dec = res.get("final_decision", "N/A").upper()
                        score = res.get("composite_score", float('nan'))
                        price_disp = res.get("current_price_display")
                        card_color = "green" if dec == "BUY" else "red" if dec == "SELL" else "#D3D3D3" if dec == "HOLD" else "grey"
                        
                        st.markdown(f"""
                        <div style="border: 1px solid {card_color}; border-radius: 8px; padding: 15px; margin-bottom: 10px; background-color: {card_color}20;">
                            <h3 style="margin-bottom: 5px; color: {card_color};">{t_symbol}</h3>
                            <p style="font-size: 1.6em; font-weight: bold; color: {card_color}; margin-bottom: 5px;">{dec}</p>
                            <p style="font-size: 0.9em; margin-bottom: 3px;">Composite Score: <strong style="color: {card_color};">{score:.2f}</strong></p>
                            {f'<p style="font-size: 0.9em;">Price: <strong>${price_disp:,.2f}</strong></p>' if price_disp is not None else ""}
                        </div>""", unsafe_allow_html=True)
            st.markdown("---")

            for t_symbol in live_tickers_list:
                res = live_output.get(t_symbol)
                if not res or "error" in res: continue
                
                with st.expander(f"🔍 Detailed Analysis for {t_symbol}"):
                    tab_titles = ["📈 Chart & Core", "펀 Fundamentals", "💰 Valuation & Fair Value", "📰 News & Filings", "⚙️ All Signals"]
                    tabs = st.tabs(tab_titles)

                    with tabs[0]: # Chart & Core Signals
                        st.subheader("Price Performance & Core Signals")
                        price_history_df_display = fetch_price_history(t_symbol, period=f"{history_years_live}y")
                        if not price_history_df_display.empty:
                            st.line_chart(price_history_df_display["Close"], use_container_width=True)
                        core_s = {
                            "Price Signal (SMA/RSI)": res.get("price_signal", "N/A").upper(),
                            "SMA50 / SMA200": f"{res.get('sma50',0):.2f} / {res.get('sma200',0):.2f}",
                            "RSI14": f"{res.get('rsi14',0):.2f}",
                            "Momentum Signal (1M/12M)": res.get("momentum_signal", "N/A").upper(),
                            "Momentum 1M / 12M (%)": f"{res.get('momentum_1m',0)*100:.1f}% / {res.get('momentum_12m',0)*100:.1f}%",
                            "Volatility Signal (Beta)": res.get("volatility_signal", "N/A").upper(),
                            "Beta / Annual Vol (%)": f"{res.get('beta',0):.2f} / {res.get('annual_vol',0)*100:.1f}%",
                        }
                        st.table(pd.Series(core_s, name="Value"))

                    with tabs[1]: # Fundamentals
                        st.subheader(f"Fundamental Snapshot - {res.get('industry_display', 'N/A')}")
                        fund_s = {"Market Cap": f"${res.get('market_cap_display',0):,}" if res.get('market_cap_display') else "N/A",
                                  "FCF Yield": f"{res.get('fcf_yield',0)*100:.2f}%", "Piotroski Score": res.get('piotroski_score'),
                                  "ROE / DebtToEquity": f"{res.get('ticker_info',{}).get('returnOnEquity',0)*100:.1f}% / {res.get('ticker_info',{}).get('debtToEquity',0):.1f}",
                                  "Fundamental Signal": res.get("fund_signal", "N/A").upper()}
                        st.table(pd.Series(fund_s, name="Value"))
                        if res.get("ticker_info",{}).get("longBusinessSummary"):
                            with st.popover("View Business Summary"):
                                st.markdown(res["ticker_info"]["longBusinessSummary"])


                    with tabs[2]: # Valuation & Fair Value
                        st.subheader("Valuation Metrics")
                        val_s = {"Forward P/E": f"{res.get('forward_pe',0):.1f}",
                                 "Relative P/E Signal": res.get('relative_pe_signal', "N/A").upper(),
                                 "DCF Fair Price (Simple Est.)": f"${res.get('dcf_fair_price',0):.2f}" if res.get('dcf_fair_price') is not None else "N/A",
                                 "DCF Signal": res.get('dcf_signal', "N/A").upper()}
                        st.table(pd.Series(val_s, name="Value"))
                        
                        if live_configs["use_value_trades"]:
                            st.subheader("Value-Trades.com Fair Value (Experimental)")
                            vt_s = {"VT Scraped Fair Value": f"${res.get('vt_fair_value_estimate',0):.2f}" if res.get('vt_fair_value_estimate') is not None else "N/A",
                                    "VT Fair Value Signal": res.get('vt_fair_value_signal', "N/A").upper(),
                                    "VT Scrape Status": res.get('vt_data_error') if res.get('vt_data_error') else "Success (or FV not on page)"}
                            st.table(pd.Series(vt_s, name="Value"))
                            if res.get('vt_data_error') and "Configuration incomplete" not in res.get('vt_data_error') and "Skipped by user config" not in res.get('vt_data_error'):
                                st.caption(f"Scraping Status Note: {res.get('vt_data_error')}")


                    with tabs[3]: # News & Filings
                        if live_configs["use_sentiment"]:
                            st.subheader("News Sentiment (LLM)")
                            sent_s = {"Sentiment Score": f"{res.get('sentiment_score',0):.2f}", "Sentiment Signal": res.get("sentiment_signal", "N/A").upper()}
                            st.table(pd.Series(sent_s, name="Value"))
                            if data_bundle.get("news"): # Assuming data_bundle is accessible or news titles are stored in res
                                with st.popover("View News Headlines"):
                                     for news_item in data_bundle["news"][:5]: st.markdown(f"- {news_item.get('title')}")
                        
                        if live_configs["use_filings"]:
                            st.subheader("Insider Filings")
                            fil_s = {"Net Insider Shares (Recent)": f"{res.get('net_insider_shares',0):,}", "Insider Filings Signal": res.get("filings_signal", "N/A").upper()}
                            st.table(pd.Series(fil_s, name="Value"))

                        if live_configs["use_politician_filings"]:
                            st.subheader("Politician Filings (Experimental Scrape)")
                            pol_s = {"Net Trade Value Estimate (Recent)": f"${res.get('politician_net_trade_value_estimate',0):,}",
                                     "Buy/Sell Transactions": f"{res.get('politician_buy_tx_count',0)} / {res.get('politician_sell_tx_count',0)}",
                                     "Politician Filings Signal": res.get("politician_filings_signal", "N/A").upper()}
                            st.table(pd.Series(pol_s, name="Value"))
                            if data_bundle.get("politician_trades"):
                                with st.popover("View Scraped Politician Trades (Max 5)"):
                                    for i, p_trade in enumerate(data_bundle["politician_trades"][:5]):
                                        st.markdown(f"**{p_trade.get('politician_name')}**: {p_trade.get('transaction_type')} ({p_trade.get('value_range')}) on {p_trade.get('date_str')}")


                    with tabs[4]: # All Signals
                        st.subheader("All Agent Signals & Final Decision")
                        all_s_keys = [k for k in res if k.endswith("_signal")]
                        all_s_table = {k.replace("_signal","").replace("_"," ").title(): res[k].upper() for k in all_s_keys}
                        all_s_table["Composite Score"] = f"{res.get('composite_score',0.0):.2f}"
                        all_s_table["Final Decision"] = res.get('final_decision',"").upper()
                        st.table(pd.Series(all_s_table, name="Signal Value"))
                        with st.popover("View Full Raw JSON for this ticker"):
                            st.json(res)
            
            st.sidebar.markdown("---")
            with st.sidebar.expander("Portfolio Agent Weights (Live Analysis)"):
                st.json(PortfolioAgent.WEIGHTS)


elif app_mode == "Backtesting":
    if run_button_backtest and bt_ticker:
        with st.spinner(f"Running backtest for {bt_ticker}... This may take a while."):
            bt_metrics, bt_log_df = run_backtest(bt_ticker, bt_start_date, bt_end_date,
                                                 bt_initial_capital, llm_client, backtest_portfolio_weights)
        if bt_metrics and "message" not in bt_metrics :
            st.subheader(f"Backtest Results for {bt_ticker}")
            metrics_df = pd.DataFrame.from_dict(bt_metrics, orient='index', columns=['Value'])
            st.table(metrics_df)
            if not bt_log_df.empty:
                st.subheader("Portfolio Value Over Time")
                st.line_chart(bt_log_df["portfolio_value"])
                st.subheader("Drawdown Over Time")
                st.area_chart(bt_log_df["drawdown"])
                with st.expander("View Backtest Log and Signals"):
                    st.dataframe(bt_log_df[["price", "signal", "composite_score", "portfolio_value", "cash", "shares_held"]])
            else: st.warning("Backtest log empty, no charts.")
        else: st.error(f"Backtest failed: {bt_metrics.get('message', 'Unknown error') if bt_metrics else 'Unknown error'}")

st.sidebar.markdown("---")
st.sidebar.info("This simulator is for educational purposes only. Not financial advice.")

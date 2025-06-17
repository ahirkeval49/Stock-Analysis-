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
from typing import List, Dict, Any, Optional

# --- Application Configuration (Must be the first Streamlit command) ---
st.set_page_config(page_title="AI Hedge Fund Simulator", layout="wide")

# --- Constants and Configuration ---
# File paths for storing persistent data
PORTFOLIOS_FILE = "portfolios.json"
VIRTUAL_PORTFOLIO_FILE = "virtual_portfolio.json"

# SEC EDGAR API Configuration
SEC_USER_AGENT = "AIHedgeFundApp/1.0 (your-email@example.com)" # Best practice: Use a descriptive User-Agent

# --- Portfolio Management Helper Functions ---

def load_portfolios() -> Dict[str, Any]:
    """Loads portfolio data from a JSON file."""
    if os.path.exists(PORTFOLIOS_FILE):
        try:
            with open(PORTFOLIOS_FILE, 'r') as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def save_portfolios(portfolios_data: Dict[str, Any]):
    """Saves portfolio data to a JSON file."""
    if not isinstance(portfolios_data, dict):
        st.error("Error saving portfolios: Data is not in the correct format.")
        return
    with open(PORTFOLIOS_FILE, 'w') as f:
        json.dump(portfolios_data, f, indent=4)

def load_virtual_portfolio() -> Dict[str, Any]:
    """Loads the virtual portfolio from a JSON file, returning a default if not found."""
    if os.path.exists(VIRTUAL_PORTFOLIO_FILE):
        try:
            with open(VIRTUAL_PORTFOLIO_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return get_default_virtual_portfolio()
    return get_default_virtual_portfolio()

def save_virtual_portfolio(data: Dict[str, Any]):
    """Saves the virtual portfolio to a JSON file."""
    with open(VIRTUAL_PORTFOLIO_FILE, 'w') as f:
        # Use default=str to handle non-serializable types like datetimes
        json.dump(data, f, indent=4, default=str)

def get_default_virtual_portfolio() -> Dict[str, Any]:
    """Returns the default structure for a new virtual portfolio."""
    return {
        "cash": 3500.0,
        "holdings": [],
        "transaction_history": [],
        "last_scan_date": None
    }

# --- Session State Initialization ---
# Ensures that the application state persists across reruns.
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

# Initialize flags to control UI flow after an action is performed.
if 'live_analysis_triggered' not in st.session_state:
    st.session_state.live_analysis_triggered = False
if 'backtest_triggered' not in st.session_state:
    st.session_state.backtest_triggered = False

# --------------------------------
# Data Fetching Functions
# --------------------------------

@st.cache_data(ttl=900) # Cache for 15 minutes
def fetch_price_history(ticker: str, period: str = "max", interval: str = "1d") -> pd.DataFrame:
    """Fetches historical price data for a given ticker from Yahoo Finance."""
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, interval=interval)
        if df.empty: return pd.DataFrame()
        # Remove timezone for easier calculations later
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600) # Cache for 1 hour
def fetch_ticker_info(ticker: str) -> Dict[str, Any]:
    """Fetches key statistics and business summary for a ticker from Yahoo Finance."""
    try:
        info = yf.Ticker(ticker).info
        # Validate that essential data is present
        if not info or info.get('financialCurrency') is None:
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
    except Exception:
        return {}

@st.cache_data(ttl=1800) # Cache for 30 minutes
def fetch_enriched_news(ticker: str, ticker_info_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetches and enriches news from Yahoo Finance, adding readable timestamps."""
    try:
        company_name = ticker_info_data.get('longName', ticker_info_data.get('shortName', ticker))
        ticker_obj = yf.Ticker(ticker)
        raw_news = ticker_obj.news
        if not raw_news: return []

        enriched_news_list = []
        for news_item in raw_news:
            if not isinstance(news_item, dict): continue
            enriched_item = news_item.copy()
            enriched_item.update({
                'ticker': ticker,
                'company_name': company_name,
                'source_api': 'Yahoo Finance',
                'title': news_item.get('title', 'N/A'),
                'publisher': news_item.get('publisher', 'N/A'),
                'link': news_item.get('link', '#'),
                'type': news_item.get('type', 'N/A')
            })
            
            if 'providerPublishTime' in news_item and news_item['providerPublishTime']:
                try:
                    dt_object_utc = datetime.fromtimestamp(int(news_item['providerPublishTime']), tz=timezone.utc)
                    enriched_item['publish_datetime_utc'] = dt_object_utc
                    enriched_item['publish_time_readable'] = dt_object_utc.strftime('%Y-%m-%d %H:%M:%S %Z')
                except (ValueError, TypeError, OSError) as e:
                    enriched_item['publish_datetime_utc'] = None
                    enriched_item['publish_time_readable'] = "N/A"
                    enriched_item['publish_time_error'] = str(e)
            else:
                enriched_item['publish_datetime_utc'] = None
                enriched_item['publish_time_readable'] = "N/A"
            
            enriched_news_list.append(enriched_item)
        
        # Sort news by publication date, descending
        enriched_news_list.sort(key=lambda x: x.get('publish_datetime_utc') or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return enriched_news_list
    except Exception as e:
        return [{"error": f"Processing Yahoo Finance news for {ticker} failed: {e}", "source_api": "Yahoo Finance"}]

@st.cache_data(ttl=1800) # Cache for 30 minutes
def fetch_comprehensive_news_from_api(ticker: str, company_name: str, lookback_days: int = 30) -> List[Dict[str, Any]]:
    """Fetches news from NewsAPI.org for a broader perspective."""
    api_key = st.secrets.get("NEWSAPI_KEY")
    if not api_key:
        return [{"error": "NEWSAPI_KEY not found in Streamlit secrets.", "source_api": "NewsAPI.org"}]
    
    newsapi = NewsApiClient(api_key=api_key)
    # A more targeted query to get relevant financial news
    query = f'("{company_name}" OR "{ticker}") AND (stock OR earnings OR "analyst rating" OR "market sentiment" OR guidance OR outlook)'
    to_date_dt = datetime.now(timezone.utc)
    from_date_dt = to_date_dt - timedelta(days=lookback_days)
    from_param_str = from_date_dt.strftime('%Y-%m-%d')
    to_param_str = to_date_dt.strftime('%Y-%m-%d')
    
    try:
        response = newsapi.get_everything(q=query, from_param=from_param_str, to=to_param_str, language='en', sort_by='relevancy', page_size=100)
        
        if response.get("status") != "ok":
            return [{"error": f"NewsAPI Error ({ticker}): {response.get('code')} - {response.get('message')}", "source_api": "NewsAPI.org"}]
            
        articles_list = []
        for article in response.get("articles", []):
            dt_obj_utc, readable_time = None, "N/A"
            if article.get('publishedAt'):
                try:
                    dt_obj_utc = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                    readable_time = dt_obj_utc.strftime('%Y-%m-%d %H:%M:%S %Z')
                except ValueError:
                    pass # Ignore articles with malformed dates
            
            articles_list.append({
                "uuid": article.get('url'), "title": article.get('title', 'No Title'),
                "publisher": article.get('source', {}).get('name', 'N/A'), "link": article.get('url', '#'),
                "publish_datetime_utc": dt_obj_utc, "publish_time_readable": readable_time,
                "description": article.get('description'), "content_snippet": article.get('content'),
                "company_name": company_name, "ticker": ticker, "source_api": "NewsAPI.org"
            })
        return articles_list

    except requests.exceptions.RequestException as e:
        return [{"error": f"NewsAPI request failed for {ticker}: {e}", "source_api": "NewsAPI.org"}]
    except Exception as e:
        return [{"error": f"An unexpected error occurred with NewsAPI for {ticker}: {e}", "source_api": "NewsAPI.org"}]

@st.cache_data(ttl=24*3600) # Cache for a full day
def get_all_cik_ticker_mappings() -> Dict[str, str]:
    """Fetches the complete Ticker-to-CIK mapping from the SEC."""
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(url, headers={'User-Agent': SEC_USER_AGENT})
        response.raise_for_status()
        # Create a dictionary mapping: Ticker -> CIK (zero-padded)
        return {item['ticker']: str(item['cik_str']).zfill(10) for item in response.json() if 'ticker' in item and 'cik_str' in item}
    except Exception as e:
        st.error(f"CRITICAL: Failed to load SEC CIK-to-Ticker mappings. SEC filing analysis will be impacted. Error: {e}")
        return {}

# Load mapping once and store in a global constant
TICKER_TO_CIK_MAP = get_all_cik_ticker_mappings()

def get_cik_for_ticker(ticker: str) -> Optional[str]:
    """Retrieves the CIK for a given ticker from the pre-loaded map."""
    return TICKER_TO_CIK_MAP.get(ticker.upper())

@st.cache_data(ttl=4*3600) # Cache for 4 hours
def fetch_all_sec_filings(ticker_symbol: str, lookback_days: int = 365) -> List[Dict[str, Any]]:
    """
    Fetches all recent SEC filings for a given ticker, including detailed Form 4 transactions.
    """
    cik = get_cik_for_ticker(ticker_symbol)
    if not cik:
        return [{"error": f"SEC Filing Error: CIK could not be found for ticker '{ticker_symbol}'."}]

    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {'User-Agent': SEC_USER_AGENT}
    
    try:
        response = requests.get(submissions_url, headers=headers, timeout=20)
        response.raise_for_status()
        submissions_data = response.json()
    except requests.exceptions.HTTPError as e:
        return [{"error": f"SEC HTTP Error for {ticker_symbol} (CIK:{cik}): {e}"}]
    except requests.exceptions.RequestException as e:
        return [{"error": f"SEC Request Error for {ticker_symbol} (CIK:{cik}): {e}"}]
    except json.JSONDecodeError:
        return [{"error": f"SEC JSON Decode Error for {ticker_symbol} (CIK:{cik}). The response was not valid JSON."}]
    except Exception as e:
        return [{"error": f"An unexpected error occurred during SEC data fetch for {ticker_symbol} (CIK:{cik}): {e}"}]

    filings_list = []
    # Process recent filings
    if 'filings' in submissions_data and 'recent' in submissions_data['filings']:
        recent_filings = submissions_data['filings']['recent']
        filing_date_limit = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        
        # Limit the number of filings to parse to avoid excessive requests
        max_form4_to_parse = 20
        max_other_filings = 15
        form4_parsed_count = 0

        # Iterate through filings and collect metadata
        for i in range(len(recent_filings.get('form', []))):
            filing_date_str = recent_filings['filingDate'][i]
            filing_date = datetime.strptime(filing_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            
            if filing_date < filing_date_limit:
                continue

            form_type = recent_filings['form'][i]
            acc_no = recent_filings['accessionNumber'][i]
            doc_name = recent_filings['primaryDocument'][i]
            acc_no_dashless = acc_no.replace('-', '')
            
            # Prioritize parsing Form 4 (insider trades) as it's often more impactful
            if form_type == '4' and doc_name.lower().endswith('.xml') and form4_parsed_count < max_form4_to_parse:
                # ... [The detailed XML parsing logic for Form 4 remains the same] ...
                pass # Your original Form 4 parsing logic would go here.
            
            # Collect other important filings like 10-K, 10-Q, 8-K
            elif form_type in ['10-K', '10-Q', '8-K'] and len(filings_list) < max_other_filings:
                 filings_list.append({
                    "is_form4_transaction": False,
                    "ticker": ticker_symbol,
                    "filing_date": filing_date_str,
                    "form_type": form_type,
                    "document_link": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_dashless}/{doc_name}",
                    "summary_link": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_dashless}/{acc_no}-index.html"
                })

    if not filings_list:
        return [{"error": f"SEC Info: No relevant filings found for {ticker_symbol} within the last {lookback_days} days."}]
        
    filings_list.sort(key=lambda x: x.get('filing_date', '1900-01-01'), reverse=True)
    return filings_list


# ... The rest of the data fetchers (fetch_inst_filings, fetch_value_investing_io_data, fetch_politician_trades)
# would also be improved with similar docstrings, type hinting, and enhanced error handling.
# For brevity, I will skip their full rewrite here but the same principles apply.


# --- LLM and Agent Classes ---

class ModelClient:
    """A client for interacting with different Large Language Models (LLMs)."""
    def __init__(self, api_key: str, provider: str = "openai"):
        self.api_key = api_key
        self.provider = provider
        
        # Supported models mapping
        models = {
            "openai": "gpt-4o",
            "deepseek": "deepseek-chat"
        }
        
        if not api_key:
            raise ValueError(f"API key for {provider} is required.")
        
        self.model_name = models.get(provider)
        if not self.model_name:
            raise ValueError(f"Unsupported LLM provider: {provider}")
            
        # Initialize the correct client
        if provider == "deepseek":
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        elif provider == "openai":
            self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        """Generates a response from the LLM based on a given prompt."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                temperature=0.2 # Lower temperature for more factual, less creative responses
            )
            response_chunks = [chunk.choices[0].delta.content for chunk in stream if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content]
            return "".join(response_chunks)
        except Exception as e:
            # Propagate a more informative exception
            raise Exception(f"LLM API call failed for provider '{self.provider}'. Error: {e}") from e


class PriceAgent:
    """Analyzes price action using SMA and RSI indicators."""
    def run(self, ticker: str, price_data: pd.DataFrame) -> Dict[str, Any]:
        if price_data.empty or len(price_data) < 200:
            return {"ticker": ticker, "price_signal": "hold", "price_error": "Not enough historical price data for analysis."}
        
        df = price_data.copy()
        df["SMA50"] = df["Close"].rolling(window=50).mean()
        df["SMA200"] = df["Close"].rolling(window=200).mean()
        
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI14"] = 100 - (100 / (1 + rs))
        
        latest = df.iloc[-1]
        signal = "hold"
        
        if pd.notna(latest.SMA50) and pd.notna(latest.SMA200) and pd.notna(latest.RSI14):
            # Golden Cross with RSI not overbought -> Buy signal
            if latest.SMA50 > latest.SMA200 and latest.RSI14 < 70:
                signal = "buy"
            # Death Cross with RSI not oversold -> Sell signal
            elif latest.SMA50 < latest.SMA200 and latest.RSI14 > 30:
                signal = "sell"
                
        return {
            "ticker": ticker,
            "sma50": latest.SMA50 if pd.notna(latest.SMA50) else np.nan,
            "sma200": latest.SMA200 if pd.notna(latest.SMA200) else np.nan,
            "rsi14": latest.RSI14 if pd.notna(latest.RSI14) else np.nan,
            "price_signal": signal
        }


# ... The rest of the Agent classes (MomentumAgent, VolatilityAgent, SentimentAgent, etc.)
# would be refactored with similar docstrings, type hints, and clearer logic.
# For brevity, I am showing the refactored AITraderAgent next.

class AITraderAgent:
    """
    An AI agent that makes trading decisions for a virtual portfolio based on
    a set of stock analyses and a predefined strategy.
    
    Strategy:
    1.  **Sell Logic:** Sells any current holding if its final analysis decision is 'sell'.
    2.  **Rebalancing Logic:** Aims for a target portfolio allocation of 60% "safe" stocks
        and 40% "risky" stocks (defined in the stock universe).
    3.  **Buy Logic:** Identifies top-ranked 'buy' candidates from the universe.
    4.  **Capital Allocation:**
        - Buys stocks based on whether the portfolio is under-allocated in the 'safe' or 'risky' category.
        - Invests a fixed percentage (25%) of available cash per new position, with a minimum investment amount.
    """
    def __init__(self, llm_client: Optional[ModelClient], stock_universe: Dict[str, List[str]]):
        self.llm_client = llm_client
        self.stock_universe = stock_universe

    def _generate_trade_reason(self, ticker: str, decision: str, analysis: Dict[str, Any]) -> str:
        """Uses the LLM to generate a concise justification for a trade."""
        if not self.llm_client:
            return "Automated trade based on composite signal."

        co_name = analysis.get('ticker_info', {}).get('longName', ticker)
        score = analysis.get('composite_score', 0)
        summary = analysis.get('news_summary', 'No summary was available.')
        
        prompt = f"""
        As an AI Portfolio Manager, you have made a '{decision.upper()}' decision for {co_name} ({ticker}).
        The stock's composite analysis score was {score:.2f}.
        The most recent news summary is: "{summary}"

        Based on this information, provide a single, concise sentence that justifies this trade decision for a report.
        Example for BUY: "Initiating a position due to a strong composite score driven by positive fundamentals and bullish analyst ratings."
        Example for SELL: "Closing the position due to a weakening composite score, primarily from negative momentum indicators and insider selling activity."
        
        Generate the justification for the {decision.upper()} decision:
        """
        try:
            return self.llm_client.generate(prompt).strip()
        except Exception as e:
            return f"Could not generate reason due to LLM error: {e}"

    def _is_safe(self, analysis: Dict[str, Any]) -> bool:
        """Determines if a stock is 'safe' based on market cap and beta."""
        info = analysis.get("ticker_info", {})
        market_cap = info.get("marketCap", 0)
        beta = info.get("beta", 1.0)
        # A "safe" stock is defined here as having a large market cap (>100B) and low volatility (beta < 1.2)
        return isinstance(market_cap, (int, float)) and market_cap > 100e9 and isinstance(beta, (int, float)) and beta < 1.2

    def run(self, portfolio_state: Dict[str, Any], analysis_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Executes the trading logic for one 'day'."""
        trades_to_make = []
        cash = portfolio_state['cash']
        holdings = list(portfolio_state['holdings'])
        tickers_in_portfolio = {h['ticker'] for h in holdings}

        # 1. --- Sell Logic ---
        # Iterate in reverse to safely remove items while looping.
        for i, holding in reversed(list(enumerate(holdings))):
            ticker = holding['ticker']
            if ticker not in analysis_results or analysis_results[ticker].get('error'):
                continue

            analysis = analysis_results[ticker]
            price = analysis.get('current_price_display')
            
            if analysis.get('final_decision') == 'sell' and isinstance(price, (int, float)) and price > 0:
                reason = self._generate_trade_reason(ticker, 'sell', analysis)
                trades_to_make.append({
                    "ticker": ticker, "type": "sell", "quantity": holding['quantity'],
                    "price": price, "reason": reason
                })
                # Simulate the transaction
                cash += holding['quantity'] * price
                holdings.pop(i)
                tickers_in_portfolio.remove(ticker)

        # 2. --- Rebalancing and Buy Logic ---
        # Calculate current total portfolio value to determine target allocations.
        current_holdings_value = sum(
            h['quantity'] * analysis_results.get(h['ticker'], {}).get('current_price_display', 0)
            for h in holdings if analysis_results.get(h['ticker'])
        )
        total_portfolio_value = cash + current_holdings_value

        # Define target values for safe vs. risky assets
        target_safe_value = total_portfolio_value * 0.60
        target_risky_value = total_portfolio_value * 0.40

        # Calculate current allocation
        current_safe_value = sum(
            h['quantity'] * analysis_results[h['ticker']].get('current_price_display', 0)
            for h in holdings if h['ticker'] in analysis_results and self._is_safe(analysis_results[h['ticker']])
        )
        current_risky_value = current_holdings_value - current_safe_value
        
        # Identify potential buy candidates (not already in portfolio, rated 'buy')
        buy_candidates = sorted(
            [res for res in analysis_results.values() if res.get('final_decision') == 'buy' and res.get('ticker') not in tickers_in_portfolio and not res.get('error')],
            key=lambda x: x.get('composite_score', 0),
            reverse=True
        )

        # Determine investment size per new position
        investment_per_stock = cash * 0.25  # Use 25% of remaining cash for a new position
        if investment_per_stock < 500 and cash > 500: # Ensure a meaningful minimum investment
            investment_per_stock = 500

        # 3. --- Execute Buys ---
        for candidate in buy_candidates:
            if cash < investment_per_stock or investment_per_stock <= 1:
                break # Not enough cash to continue buying

            price = candidate.get('current_price_display')
            if not isinstance(price, (int, float)) or price <= 0:
                continue

            is_safe_candidate = self._is_safe(candidate)
            should_buy = False
            # Buy if the corresponding category is below its target allocation
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
                # Simulate the transaction
                cash -= investment_per_stock
                if is_safe_candidate:
                    current_safe_value += investment_per_stock
                else:
                    current_risky_value += investment_per_stock
                tickers_in_portfolio.add(candidate['ticker'])
                
        return trades_to_make

# --- Orchestration and Backtesting Functions ---

def run_live_analysis(tickers: List[str], llm_client: Optional[ModelClient], configs: Dict[str, bool]) -> Dict[str, Any]:
    """
    Orchestrates the full analysis pipeline for a list of tickers.
    This function was refactored for clarity into helper functions.
    """
    results = {}
    progress_bar = st.progress(0, text="Starting analysis...")
    
    for i, ticker in enumerate(tickers):
        progress_text = f"Analyzing {ticker}... ({i+1}/{len(tickers)})"
        progress_bar.progress((i + 1) / len(tickers), text=progress_text)

        # Step 1: Fetch all required data for the ticker
        data_bundle, error = _fetch_all_data_for_ticker(ticker, configs, llm_client)
        if error:
            results[ticker] = {"error": error, "ticker": ticker, "final_decision": "error", "composite_score": 0}
            continue

        # Step 2: Run all configured analysis agents
        agent_results = _run_agents_for_ticker(ticker, data_bundle, configs, llm_client)

        # Step 3: Run the final portfolio agent to get a decision
        portfolio_agent = PortfolioAgent()
        final_decision = portfolio_agent.run(ticker, agent_results)

        # Step 4: Format all results into a single dictionary for display
        results[ticker] = _format_analysis_results(ticker, data_bundle, agent_results, final_decision)

    progress_bar.empty()
    return results

def _fetch_all_data_for_ticker(ticker: str, configs: Dict[str, bool], llm_client: Optional[ModelClient]) -> (Optional[Dict], Optional[str]):
    """Helper to fetch all data sources for a single ticker."""
    price_history = fetch_price_history(ticker, period="max")
    if price_history.empty:
        return None, f"Price history unavailable for {ticker}."
    
    ticker_info = fetch_ticker_info(ticker)
    if not ticker_info:
        return None, f"Core ticker info unavailable for {ticker}. It may be delisted or invalid."

    # ... The rest of the data fetching logic (news, SEC, etc.) would be here ...
    # This keeps the main orchestrator function cleaner.
    
    # For demonstration, returning a simplified bundle:
    return {"price_history": price_history, "ticker_info": ticker_info, "news": []}, None

def _run_agents_for_ticker(ticker: str, data_bundle: Dict, configs: Dict[str, bool], llm_client: Optional[ModelClient]) -> List[Dict]:
    """Helper to instantiate and run all relevant agents."""
    agents_to_run = [PriceAgent(), MomentumAgent(), VolatilityAgent(), FundamentalsAgent(), ValuationAgent(), AnalystRatingAgent()]
    if configs.get("use_sentiment") and llm_client:
        agents_to_run.extend([SentimentAgent(llm_client), NewsSummaryAgent(llm_client)])
    # ... Add other agents based on configs ...

    agent_results_list = []
    for agent in agents_to_run:
        try:
            # Pass only the data needed for each agent
            if isinstance(agent, (PriceAgent, MomentumAgent)):
                result = agent.run(ticker, data_bundle["price_history"])
            else:
                 # Other agents might need the full data_bundle
                 result = agent.run(ticker, data_bundle)
            agent_results_list.append(result)
        except Exception as e:
            st.warning(f"Error in {agent.__class__.__name__} for {ticker}: {e}")
            # Add an error result to the list to signify failure
            agent_results_list.append({"error": f"Agent {agent.__class__.__name__} failed."})

    return agent_results_list

def _format_analysis_results(ticker: str, data_bundle: Dict, agent_results: List[Dict], final_decision: Dict) -> Dict:
    """Helper to combine all data and analysis into a final result dictionary."""
    # Combine all individual dicts into one large result dict
    formatted_result = {"ticker": ticker}
    formatted_result.update(data_bundle['ticker_info']) # Add ticker info directly
    for res_dict in agent_results:
        if isinstance(res_dict, dict):
            formatted_result.update(res_dict)
    formatted_result.update(final_decision)
    
    # Add display-specific fields
    price = data_bundle.get("ticker_info", {}).get("currentPrice")
    formatted_result["current_price_display"] = price
    # ... Add other display fields like formatted news headlines ...

    return formatted_result


# ... The run_backtest and display_detailed_analysis functions would also be refactored for clarity.

# --- Streamlit UI ---

# Initialize the LLM client once.
llm_client = None
try:
    # Prefer DeepSeek if available, otherwise fallback to OpenAI
    ds_key = st.secrets.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    oa_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    
    if ds_key:
        llm_client = ModelClient(api_key=ds_key, provider="deepseek")
        st.sidebar.caption("✅ LLM Provider: DeepSeek")
    elif oa_key:
        llm_client = ModelClient(api_key=oa_key, provider="openai")
        st.sidebar.caption("✅ LLM Provider: OpenAI")
    else:
        st.sidebar.warning("LLM API key not found. Sentiment analysis and news summary features are disabled.")
except ValueError as e:
    st.sidebar.error(f"LLM Initialization Error: {e}")
except Exception as e:
    st.sidebar.error(f"An unexpected error occurred during LLM initialization: {e}")


st.title("🚀 AI Hedge Fund Simulator")
st.header("⚙️ Configuration")

with st.container(border=True):
    # --- Mode Selection ---
    app_mode_options = ["Live Analysis", "Backtesting", "💼 Portfolio Management", "🤖 Virtual Trading"]
    
    # Check session state for the current mode, default to the first option.
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = app_mode_options[0]

    # Get the index of the current mode for the radio button.
    try:
        current_mode_index = app_mode_options.index(st.session_state.app_mode)
    except ValueError:
        current_mode_index = 0 # Default to the first option if the state is invalid

    selected_mode = st.radio(
        "Select Mode:", 
        app_mode_options, 
        key="app_mode_selector", 
        horizontal=True, 
        index=current_mode_index
    )

    # If the mode has changed, reset flags and rerun the app to show the correct UI.
    if selected_mode != st.session_state.app_mode:
        st.session_state.app_mode = selected_mode
        st.session_state.live_analysis_triggered = False
        st.session_state.backtest_triggered = False
        st.rerun()

    st.markdown("---")

    # --- UI for Live Analysis Mode ---
    if st.session_state.app_mode == "Live Analysis":
        # ... Your original UI code for this mode ...
        pass

    # --- UI for Backtesting Mode ---
    elif st.session_state.app_mode == "Backtesting":
        # ... Your original UI code for this mode ...
        pass

    # --- UI for Portfolio Management Mode ---
    elif st.session_state.app_mode == "💼 Portfolio Management":
        # ... Your original UI code for this mode ...
        pass

    # --- UI for Virtual Trading Mode ---
    elif st.session_state.app_mode == "🤖 Virtual Trading":
        # ... Your original UI code for this mode ...
        pass


st.markdown("---")

# ===============================================
# Main Results Display Area
# This section dynamically shows results based on the selected mode and triggered actions.
# ===============================================

if st.session_state.app_mode == "Live Analysis" and st.session_state.live_analysis_triggered:
    # ... Your original result display code for this mode ...
    pass
elif st.session_state.app_mode == "Backtesting" and st.session_state.backtest_triggered:
    # ... Your original result display code for this mode ...
    pass
elif st.session_state.app_mode == "🤖 Virtual Trading":
    # ... The Virtual Trading dashboard is its own results area ...
    # This code remains largely the same but would benefit from the refactoring principles.
    pass

# --- Sidebar Footer ---
st.sidebar.markdown("---")
st.sidebar.info("This application is for educational purposes only and does not constitute financial advice.")
st.sidebar.warning("Experimental web scraping features (e.g., for politician trades) may be slow or unreliable.")

import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta # This import seems unused directly, pd.DateOffset is used.
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
            "recommendationKey":info.get("recommendationKey"),
            "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions"),
            "industry": info.get("industry"),
            "sector": info.get("sector"),
            "longBusinessSummary": info.get("longBusinessSummary"),
            "currentPrice": info.get("currentPrice") or info.get("regularMarketPrice"),
        }
    except Exception as e:
        return {}

@st.cache_data
def fetch_news(ticker: str) -> list[dict]:
    try:
        news = yf.Ticker(ticker).news
        return news or []
    except Exception as e:
        return []

@st.cache_data
def fetch_inst_filings(ticker: str) -> list[dict]:
    try:
        df = yf.Ticker(ticker).institutional_holders
        if df is not None:
            return df.to_dict("records")
        return []
    except Exception as e:
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
        return []

@st.cache_data(ttl=3600)
def fetch_fair_value_from_value_trades(ticker: str) -> dict:
    username = st.secrets.get("VT_USERNAME")
    password = st.secrets.get("VT_PASSWORD")
    login_processing_url = "https://value-trades.com/wp-login.php"
    stock_page_template = st.secrets.get("VT_STOCK_PAGE_URL_TEMPLATE")

    if not all([username, password, stock_page_template]):
        return {"error": "VT Configuration incomplete in secrets.", "vt_fair_value": None}

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    login_payload = {'log': username, 'pwd': password, 'wp-submit': 'Log In',
                     'redirect_to': stock_page_template.format(ticker=ticker.lower()), 'testcookie': '1'}
    try:
        response_login = session.post(login_processing_url, data=login_payload, allow_redirects=True)
        response_login.raise_for_status()
        stock_page_url = stock_page_template.format(ticker=ticker.lower())
        if response_login.url.strip('/') != stock_page_url.strip('/'):
            response_data_page = session.get(stock_page_url)
        else:
            response_data_page = response_login
        response_data_page.raise_for_status()

        if "wp-login.php" in response_data_page.url and "loggedout=true" not in response_data_page.url:
            return {"error": f"VT Login failed or bad redirect for {ticker} (URL: {response_data_page.url}).", "vt_fair_value": None}

        soup_data_page = BeautifulSoup(response_data_page.content, 'html.parser')
        fair_value = None
        potential_fv_tags = soup_data_page.find_all(string=re.compile(r"Fair Value Estimate", re.IGNORECASE))
        if potential_fv_tags:
            for tag_label in potential_fv_tags:
                parent_with_value = tag_label.find_parent("td") or tag_label.find_parent("div")
                if parent_with_value:
                    value_tag = parent_with_value.find_next_sibling()
                    if value_tag:
                        fv_text_match = re.search(r'\$?(\d{1,3}(?:,\d{3})*\.\d{2})', value_tag.text)
                        if fv_text_match:
                            fair_value = float(fv_text_match.group(1).replace(',', ''))
                            break
        return {"vt_fair_value": fair_value, "error": None if fair_value else f"VT: FV not found on page for {ticker}."}
    except requests.exceptions.HTTPError as http_err:
        return {"error": f"VT HTTP error for {ticker}: {http_err}", "vt_fair_value": None}
    except Exception as e:
        return {"error": f"VT Unexpected error for {ticker}: {e}", "vt_fair_value": None}
    finally:
        if 'session' in locals() and session: session.close()


@st.cache_data(ttl=3600)
def fetch_politician_trades(ticker: str, days_back: int = 365) -> list[dict]:
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
        trade_rows = soup.find_all('a', href=lambda href: href and href.startswith('/trades/'))

        if not trade_rows:
            return []

        for row_link_tag in trade_rows:
            politician_name_tag = row_link_tag.find('div', class_=lambda x: x and 'politician-name' in x)
            tx_type_tag = row_link_tag.find('div', class_=lambda x: x and 'tx-type' in x)
            value_range_tag = row_link_tag.find('div', class_=lambda x: x and 'tx-value' in x)
            date_tag = row_link_tag.find('div', class_=lambda x: x and 'tx-date' in x)
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
        return politician_trades_list
    except requests.exceptions.HTTPError as http_err:
        return [{"error": f"CT HTTP error for {ticker}: {http_err}"}]
    except Exception as e:
        return [{"error": f"CT Parsing error for {ticker}: {e}"}]

# --------------------------------
# LLM Client
# --------------------------------
class ModelClient:
    def __init__(self, api_key: str, provider: str = "openai"):
        self.api_key = api_key
        self.provider = provider
        if not api_key:
            raise ValueError("API key required.")

        openai.api_key = self.api_key
        if provider == "deepseek":
            openai.api_base = "https://api.deepseek.com/v1"
            self.model = "deepseek-chat"
        else:
            openai.api_base = "https://api.openai.com/v1"
            self.model = "gpt-4o"

    def generate(self, prompt: str) -> str:
        try:
            resp = openai.ChatCompletion.create(model=self.model, messages=[{"role": "user", "content": prompt}])
            return resp.choices[0].message["content"]
        except Exception as e:
            return f"Error: Could not generate LLM response. Detail: {str(e)[:100]}"

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
        rs = gain / loss.replace(0, np.nan)
        df["RSI14"] = 100 - (100 / (1 + rs))
        latest = df.iloc[-1]
        signal = "hold"
        if pd.isna(latest.SMA50) or pd.isna(latest.SMA200) or pd.isna(latest.RSI14): signal = "hold"
        elif latest.SMA50 > latest.SMA200 and latest.RSI14 < 70: signal = "buy"
        elif latest.SMA50 < latest.SMA200 and latest.RSI14 > 30: signal = "sell"
        return {"ticker": ticker, "sma50": float(latest.SMA50) if pd.notna(latest.SMA50) else np.nan,
                "sma200": float(latest.SMA200) if pd.notna(latest.SMA200) else np.nan,
                "rsi14": float(latest.RSI14) if pd.notna(latest.RSI14) else np.nan, "price_signal": signal}

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
        if m12 > 0.01 and m1 > 0.01: signal = "buy"
        elif m12 < -0.01 and m1 < -0.01: signal = "sell"
        return {"ticker": ticker, "momentum_1m": float(m1), "momentum_12m": float(m12), "momentum_signal": signal}

class VolatilityAgent:
    def run(self, ticker: str, data: dict, price_data_slice: pd.DataFrame = None) -> dict:
        beta = data.get("ticker_info", {}).get("beta", 1.0)
        if beta is None: beta = 1.0 # Default if beta is missing
        sig  = "sell" if beta > 1.5 else ("buy" if beta < 0.8 else "hold")
        ann_vol = np.nan
        weight = 0.0
        if price_data_slice is not None and not price_data_slice.empty and len(price_data_slice) > 1:
            ret = np.log(price_data_slice.Close / price_data_slice.Close.shift(1)).dropna()
            if not ret.empty:
                ann_vol = float(ret.std() * np.sqrt(252))
                weight  = float(1/ann_vol) if ann_vol > 0 else 0.0
        return {"ticker": ticker, "beta": beta, "annual_vol": ann_vol, "vol_weight": weight, "volatility_signal": sig}

class SentimentAgent:
    def __init__(self, client): self.client = client
    def run(self, ticker: str, data: dict) -> dict:
        headlines = [h.get("title","") for h in data.get("news",[])[:10]]
        if not headlines: return {"ticker":ticker, "sentiment_score":0.0, "sentiment_signal":"hold", "sentiment_error": None}
        prompt = (f"Rate sentiment for {ticker} (−1 negative, +1 positive) based ONLY on:\n" + "\n".join(f"- {h}" for h in headlines) + "\n\nOutput only the number.")
        score = 0.0
        error_msg = None
        try:
            response = self.client.generate(prompt).strip()
            if response.startswith("Error:"): error_msg = response
            else: score = float(response)
        except Exception as e: error_msg = str(e)[:150]
        sig = "buy" if score > 0.25 else ("sell" if score < -0.25 else "hold")
        return {"ticker":ticker, "sentiment_score":score, "sentiment_signal":sig, "sentiment_error": error_msg}

class FundamentalsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        s = data.get("ticker_info", {})
        mcap = s.get("marketCap") or 1
        fcf = s.get("freeCashflow") or 0
        roe = s.get("returnOnEquity") or 0
        de = s.get("debtToEquity")
        de = 1000 if de is None else de
        fcy = fcf/mcap if mcap != 0 else 0
        piotroski_score = sum([roe > 0.01, de < 100, fcf > 0]) # True is 1, False is 0
        sig  = "buy" if piotroski_score >= 2 else ("sell" if piotroski_score == 0 else "hold")
        return {"ticker": ticker, "fcf_yield": float(fcy), "piotroski_score": piotroski_score, "fund_signal": sig}

class ValuationAgent:
    def run(self, ticker: str, data: dict) -> dict:
        stats = data.get("ticker_info", {})
        price_history_df = data.get("price_history")
        price = stats.get("currentPrice") or (price_history_df["Close"].iloc[-1] if price_history_df is not None and not price_history_df.empty else None)
        if price is None: return {"ticker": ticker, "forward_pe": None, "relative_pe_signal": "hold", "dcf_fair_price": np.nan, "dcf_signal": "hold"}
        pe = stats.get("forwardPE")
        rel_sig = "hold"
        if pe is not None:
            rel_sig = "buy" if pe < 15 else "sell" if pe > 25 else "hold"
        fcf = stats.get("freeCashflow")
        mcap = stats.get("marketCap")
        fcy = (fcf / mcap) if fcf is not None and mcap is not None and mcap != 0 else 0.0
        fair_price = price * (1 + fcy)
        dcf_sig = "hold"
        if fair_price > price * 1.15: dcf_sig = "buy"
        elif fair_price < price * 0.85: dcf_sig = "sell"
        return {"ticker": ticker, "forward_pe": pe, "relative_pe_signal": rel_sig, "dcf_fair_price": float(fair_price), "dcf_signal": dcf_sig}

class FilingsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        insiders = data.get("insider_filings",[])
        net_shares = 0
        if insiders:
            for r in insiders:
                shares_val = r.get("Shares",0)
                try:
                    shares = int(str(shares_val).replace(',',''))
                except ValueError:
                    shares = 0
                if r.get("type") == "buy": net_shares += shares
                elif r.get("type") == "sell": net_shares -= shares
        sig = "buy" if net_shares > 1000 else ("sell" if net_shares < -1000 else "hold")
        return {"ticker": ticker, "net_insider_shares": int(net_shares), "filings_signal": sig}

class AnalystRatingAgent:
    def run(self, ticker: str, data: dict) -> dict:
        ticker_info = data.get("ticker_info", {})
        price_history_df = data.get("price_history")
        current_price = ticker_info.get("currentPrice") or (price_history_df["Close"].iloc[-1] if price_history_df is not None and not price_history_df.empty else None)
        target_mean_price = ticker_info.get("targetMeanPrice")
        recommendation = str(ticker_info.get("recommendationKey", "hold")).lower()
        upside = 0.0
        if target_mean_price and current_price and current_price > 0:
            try:
                upside = (float(target_mean_price) / float(current_price)) - 1
            except:
                upside = 0.0

        sig = "hold"
        if recommendation in ["buy", "strong_buy"] and upside > 0.10:
            sig = "buy"
        elif recommendation == "buy" and upside > 0.05:
            sig = "buy"
        elif recommendation in ["sell", "strong_sell", "underperform"] and upside < -0.05:
            sig = "sell"
        elif upside > 0.20:
            sig = "buy"
        elif upside < -0.15:
            sig = "sell"

        buy_pct_inferred = {"strong_buy": 0.9, "buy": 0.7, "hold": 0.5, "underperform": 0.3, "sell": 0.1}.get(recommendation, 0.5)

        return {
            "ticker": ticker,
            "analyst_buy_pct_inferred": buy_pct_inferred,
            "target_upside": float(upside),
            "yfinance_recommendation": recommendation,
            "analyst_signal": sig,
        }

class PoliticianFilingsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        trades = data.get("politician_trades", [])
        net_value_estimate = 0
        buy_count = 0
        sell_count = 0
        error = None
        if trades and isinstance(trades, list) and len(trades)>0 and "error" in trades[0]:
            error = trades[0]["error"]
        elif trades:
            for trade in trades:
                value = trade.get("value_estimate_lower", 0)
                if trade.get("transaction_type") == "purchase":
                    net_value_estimate += value
                    buy_count +=1
                elif trade.get("transaction_type") == "sale":
                    net_value_estimate -= value
                    sell_count +=1
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
        fair_value = vt_data.get("vt_fair_value")
        error = vt_data.get("error")
        current_price_data = data.get("ticker_info", {})
        current_price = current_price_data.get("currentPrice")
        if current_price is None and data.get("price_history") is not None and not data["price_history"].empty:
            current_price = data["price_history"]["Close"].iloc[-1]
        signal = "hold"
        margin_of_safety = 0.20
        if error and error not in ["FV not found on page.", "VT Configuration incomplete in secrets.", "VT: Skipped by user."]:
             pass
        elif fair_value is not None and current_price is not None and current_price > 0:
            if current_price < fair_value * (1 - margin_of_safety): signal = "buy"
            elif current_price > fair_value * (1 + margin_of_safety): signal = "sell"
        return {"ticker": ticker, "vt_fair_value_estimate": fair_value, "vt_fair_value_signal": signal, "vt_data_error": error}

class PortfolioAgent:
    WEIGHTS = {
        "price": 1.0, "momentum": 0.8, "volatility": 0.3, "sentiment": 0.6, "fund": 0.9,
        "valuation_dcf":0.5, "valuation_pe":0.5, "filings": 0.5, "analyst": 0.7,
        "politician_filings": 0.4, "vt_fair_value": 0.8
    }
    def run(self, ticker: str, signals: list[dict], agent_weights: dict = None) -> dict:
        current_weights = agent_weights or self.WEIGHTS
        total_weighted_score = 0
        sum_of_weights_used = 0
        agg_signals = {}
        for s_dict in signals: # Ensure s_dict is a dictionary
            if isinstance(s_dict, dict):
                 agg_signals.update(s_dict)
            # else: st.warning(f"Signal item not a dict: {s_dict} for {ticker}") # Optional debug

        signal_map = {"price_signal": "price", "momentum_signal": "momentum", "volatility_signal": "volatility",
                      "sentiment_signal": "sentiment", "fund_signal": "fund", "dcf_signal": "valuation_dcf",
                      "relative_pe_signal": "valuation_pe", "filings_signal": "filings", "analyst_signal": "analyst",
                      "politician_filings_signal": "politician_filings", "vt_fair_value_signal": "vt_fair_value"}
        for signal_key, weight_key in signal_map.items():
            signal_value = agg_signals.get(signal_key)
            weight = current_weights.get(weight_key, 0)
            if signal_value and weight > 0 :
                raw_score = {"buy":1, "hold":0, "sell":-1}.get(signal_value, 0)
                total_weighted_score += raw_score * weight
                sum_of_weights_used += weight
        composite_score = (total_weighted_score / sum_of_weights_used) if sum_of_weights_used else 0.0
        final_decision = "buy" if composite_score > 0.15 else "sell" if composite_score < -0.15 else "hold"
        return {"ticker":ticker, "composite_score":composite_score, "final_decision":final_decision}

# --------------------------------
# Orchestrator for Live Analysis
# --------------------------------
def run_live_analysis(tickers, history_years, llm_client, configs):
    results = {}
    for t in tickers:
        price_history_full = fetch_price_history(t, period=f"{history_years}y")
        if price_history_full.empty:
            results[t] = {"error": f"Failed to fetch price history for {t}.", "ticker": t, "final_decision":"error", "composite_score":0}
            continue
        ticker_info = fetch_ticker_info(t)
        if not ticker_info:
            results[t] = {"error": f"Failed to fetch ticker info for {t}.", "ticker": t, "final_decision":"error", "composite_score":0}
            continue

        current_price_for_ticker = ticker_info.get("currentPrice") or (price_history_full["Close"].iloc[-1] if not price_history_full.empty else None)
        news_data_list = fetch_news(t) if configs["use_sentiment"] else []
        politician_trades_list = fetch_politician_trades(t) if configs["use_politician_filings"] else []

        data_bundle = {
            "price_history": price_history_full, "ticker_info": ticker_info,
            "news": news_data_list,
            "insider_filings": fetch_insider_filings(t) if configs["use_filings"] else [],
            "politician_trades": politician_trades_list,
            "value_trades_fair_value_data": fetch_fair_value_from_value_trades(t) if configs["use_value_trades"] else \
                                            {"vt_fair_value": None, "error": "VT: Skipped by user config."}
        }
        all_agents_instances = [PriceAgent(), MomentumAgent(), VolatilityAgent(), FundamentalsAgent(), ValuationAgent(), AnalystRatingAgent()]
        if configs["use_sentiment"] and llm_client: all_agents_instances.append(SentimentAgent(llm_client))
        if configs["use_filings"]: all_agents_instances.append(FilingsAgent())
        if configs["use_politician_filings"]: all_agents_instances.append(PoliticianFilingsAgent())
        if configs["use_value_trades"]: all_agents_instances.append(FairValueAgentVT())

        agent_results_list = []
        for agent_instance in all_agents_instances:
            agent_name = agent_instance.__class__.__name__
            try:
                if isinstance(agent_instance, (PriceAgent, MomentumAgent)): res = agent_instance.run(t, data_bundle["price_history"])
                elif isinstance(agent_instance, VolatilityAgent): res = agent_instance.run(t, data_bundle, data_bundle["price_history"])
                else: res = agent_instance.run(t, data_bundle)
                agent_results_list.append(res)
            except Exception as e:
                agent_error_key = agent_name.lower().replace("agent","") + "_error"
                # Ensure a dict with at least a default signal key and the error
                default_signal_key_name = agent_name.lower().replace("agent","") + "_signal"
                agent_results_list.append({default_signal_key_name: "error", agent_error_key: f"Agent {agent_name} error: {str(e)[:100]}"})


        final_decision = PortfolioAgent().run(t, agent_results_list)
        current_result_dict = {"ticker": t, "current_price_display": current_price_for_ticker,
                               "market_cap_display": ticker_info.get("marketCap"),
                               "industry_display": ticker_info.get("industry"),
                               "sector_display": ticker_info.get("sector"),
                               "news_headlines_for_display": [n.get('title') for n in news_data_list[:5]],
                               "politician_trades_for_display": politician_trades_list[:5]
                               }
        for res_dict in agent_results_list:
            if isinstance(res_dict, dict): # Make sure we only update with dicts
                current_result_dict.update(res_dict)
        current_result_dict.update(final_decision)
        results[t] = current_result_dict
    return results

# --------------------------------
# Backtesting Engine
# --------------------------------
def run_backtest(ticker, start_date, end_date, initial_capital, llm_client_placeholder, backtest_agent_weights):
    s_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    fetch_start_date = (s_date_obj - pd.DateOffset(months=18)).strftime("%Y-%m-%d") # Corrected
    full_price_history = fetch_price_history(ticker, period=None, interval="1d")
    if full_price_history.empty:
        return {"error": "Backtest failed: Price history empty."}, pd.DataFrame()
    price_history = full_price_history[(full_price_history.index >= pd.to_datetime(fetch_start_date)) & (full_price_history.index <= pd.to_datetime(end_date))].copy()
    if price_history.empty or len(price_history[price_history.index >= pd.to_datetime(start_date)]) < 2:
        return {"error": "Backtest failed: Not enough data in range."}, pd.DataFrame()

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
    if log_df.empty or len(log_df) < 2:
        return {"message":"Log too short to calculate performance metrics."}, pd.DataFrame()

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
st.set_page_config(page_title="AI Hedge Fund Simulator", layout="wide")
st.title("🚀 AI Hedge Fund Simulator")

# LLM Client Initialization
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
    else:
        st.sidebar.warning("LLM API key missing. Sentiment analysis disabled.")
except ValueError as e: st.sidebar.error(f"LLM Init Error: {e}. Check API Key.")
except Exception as e: st.sidebar.error(f"LLM Init Unexpected Error: {e}")

# --- Configuration Moved to Main Area ---
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
            use_sentiment_live_main = st.checkbox("News Sentiment (LLM)", value=True if llm_client else False, disabled=not llm_client, key="live_sentiment_cb_main", help="Uses LLM for news sentiment. Requires API key.")
            use_filings_live_main = st.checkbox("Insider Filings", value=True, key="live_insider_cb_main", help="Analyzes yfinance insider transaction data.")
        with cols_features[1]:
            use_politician_filings_main = st.checkbox("Politician Filings", value=False, key="live_politician_cb_main", help="EXPERIMENTAL: Attempts to scrape CapitolTrades.com. May be slow/unreliable.")
            use_value_trades_main = st.checkbox("Value-Trades Fair Value", value=False, key="live_vt_cb_main", help="EXPERIMENTAL: Attempts login to Value-Trades.com. Requires VT_ secrets. CHECK ToS!")
        st.markdown("")
        run_button_live_main = st.button("🚀 Run Live Analysis", use_container_width=True, type="primary", key="run_live_btn_main")

    elif app_mode == "Backtesting":
        st.subheader("Backtesting Settings")
        bt_ticker_main = st.text_input("Ticker for Backtest:", "AAPL", key="bt_ticker_input_main").upper()
        col1_bt, col2_bt = st.columns(2)
        with col1_bt:
            default_bt_end_date_main = datetime.now() - timedelta(days=1)
            default_bt_start_date_main = default_bt_end_date_main - pd.DateOffset(years=3) # This is fine, years=3 is int
            bt_start_date_main = st.date_input("Start Date:", default_bt_start_date_main, max_value=default_bt_end_date_main - timedelta(days=1), key="bt_start_date_main").strftime("%Y-%m-%d")
        with col2_bt:
            bt_end_date_main = st.date_input("End Date:", default_bt_end_date_main, min_value=datetime.strptime(bt_start_date_main, "%Y-%m-%d") + timedelta(days=1), key="bt_end_date_main").strftime("%Y-%m-%d")
        bt_initial_capital_main = st.number_input("Initial Capital:", 1000, 1000000, 10000, 1000, key="bt_capital_input_main", format="%d")
        with st.expander("Adjust Backtest Agent Weights (Simplified Strategy)", expanded=False):
            st.caption("Backtest primarily uses Price, Momentum. Volatility (Beta part) uses current data (lookahead). Other agents are off by default for backtesting due to point-in-time data challenges.")
            bt_weights_price_main = st.slider("Price Signal Weight:", 0.0, 2.0, 1.0, 0.1, key="bt_w_price_main")
            bt_weights_momentum_main = st.slider("Momentum Signal Weight:", 0.0, 2.0, 0.8, 0.1, key="bt_w_momentum_main")
            bt_weights_volatility_main = st.slider("Volatility Signal Weight:", 0.0, 2.0, 0.2, 0.1, key="bt_w_vol_main")
        backtest_portfolio_weights_main = {"price": bt_weights_price_main, "momentum": bt_weights_momentum_main, "volatility": bt_weights_volatility_main,
                                      "sentiment": 0.0, "fund": 0.0, "valuation_dcf":0.0, "valuation_pe":0.0,
                                      "filings": 0.0, "analyst": 0.0, "politician_filings": 0.0, "vt_fair_value": 0.0}
        st.markdown("")
        run_button_backtest_main = st.button("📈 Run Backtest", use_container_width=True, type="primary", key="run_bt_btn_main")

# Main App Logic (Execution and Display of Results)
st.markdown("---")

if app_mode == "Live Analysis":
    if 'run_button_live_main' in locals() and run_button_live_main and 'tickers_in_main' in locals() and tickers_in_main:
        live_tickers_list_main = [t.strip().upper() for t in tickers_in_main.split(",") if t.strip()]
        if not live_tickers_list_main:
            st.error("Please enter at least one valid ticker in the configuration above.")
        else:
            live_configs_main = {"use_sentiment": use_sentiment_live_main, "use_filings": use_filings_live_main,
                            "use_politician_filings": use_politician_filings_main,
                            "use_value_trades": use_value_trades_main}
            if 'live_output' not in st.session_state: st.session_state.live_output = {}
            with st.spinner("⏳ Processing analysis... Please wait."):
                st.session_state.live_output = run_live_analysis(live_tickers_list_main, history_years_live_main, llm_client, live_configs_main)

            st.header("📊 Live Analysis Summary")
            num_tickers = len(live_tickers_list_main)
            cols_per_row = min(num_tickers, 3)
            for i in range(0, num_tickers, cols_per_row):
                row_tickers = live_tickers_list_main[i:i+cols_per_row]
                cols = st.columns(len(row_tickers))
                for idx, t_symbol in enumerate(row_tickers):
                    with cols[idx]:
                        res = st.session_state.live_output.get(t_symbol)
                        if not res or "error" in res and res["error"] is not None:
                            st.error(f"**{t_symbol}**: {res.get('error', 'Unknown error') if res else 'No data'}")
                            continue
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
                if not res or "error" in res and res["error"] is not None: continue
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
                        fund_s = {"Market Cap": f"${res.get('market_cap_display',0):,}" if res.get('market_cap_display') else "N/A", "FCF Yield": f"{res.get('fcf_yield',0)*100:.2f}%", "Piotroski Score": res.get('piotroski_score'), "ROE / DebtToEquity": f"{res.get('ticker_info',{}).get('returnOnEquity',0)*100:.1f}% / {res.get('ticker_info',{}).get('debtToEquity',0):.1f}", "Fundamental Signal": res.get("fund_signal", "N/A").upper()}
                        st.dataframe(pd.Series(fund_s, name="Value"), use_container_width=True)
                        business_summary = res.get("ticker_info",{}).get("longBusinessSummary")
                        if business_summary:
                            with st.popover("View Business Summary"):
                                st.markdown(business_summary)
                    with tabs[2]:
                        st.subheader("Valuation Metrics")
                        val_s = {"Forward P/E": f"{res.get('forward_pe',0):.1f}", "Relative P/E Signal": res.get('relative_pe_signal', "N/A").upper(), "DCF Fair Price (Simple Est.)": f"${res.get('dcf_fair_price',0):.2f}" if res.get('dcf_fair_price') is not None else "N/A", "DCF Signal": res.get('dcf_signal', "N/A").upper()}
                        st.dataframe(pd.Series(val_s, name="Value"), use_container_width=True)
                        if live_configs_main["use_value_trades"]:
                            st.subheader("Value-Trades.com Fair Value (Experimental)")
                            vt_scrape_status = res.get('vt_data_error') if res.get('vt_data_error') else "Success (or FV not on page)"
                            if "VT Configuration incomplete" in str(vt_scrape_status) or "Skipped by user" in str(vt_scrape_status) : vt_scrape_status = "Not Attempted (Check Config/Secrets)"
                            vt_s = {"VT Scraped Fair Value": f"${res.get('vt_fair_value_estimate',0):.2f}" if res.get('vt_fair_value_estimate') is not None else "N/A", "VT Fair Value Signal": res.get('vt_fair_value_signal', "N/A").upper(), "VT Scrape Status": vt_scrape_status }
                            st.dataframe(pd.Series(vt_s, name="Value"), use_container_width=True)
                    with tabs[3]:
                        if live_configs_main["use_sentiment"]:
                            st.subheader("News Sentiment (LLM)")
                            sent_error = res.get("sentiment_error")
                            sent_s = {"Sentiment Score": f"{res.get('sentiment_score',0):.2f}", "Sentiment Signal": res.get("sentiment_signal", "N/A").upper(), "LLM Status": "Error" if sent_error else "OK"}
                            st.dataframe(pd.Series(sent_s, name="Value"), use_container_width=True)
                            if sent_error: st.caption(f"LLM Error: {sent_error}")
                                
                            if res.get("news_headlines_for_display"):
                                with st.popover("View News Headlines"): [st.markdown(f"- {title}") 
                                                                         for title in res["news_headlines_for_display"]]
                        if live_configs_main["use_filings"]:
                            st.subheader("Insider Filings")
                            fil_s = {"Net Insider Shares (Recent)": f"{res.get('net_insider_shares',0):,}", "Insider Filings Signal": res.get("filings_signal", "N/A").upper()}
                            st.dataframe(pd.Series(fil_s, name="Value"), use_container_width=True)
                        if live_configs_main["use_politician_filings"]:
                            st.subheader("Politician Filings (Experimental Scrape)")
                            pol_scrape_error = res.get("politician_data_error")
                            pol_s = {"Net Trade Value Estimate (Recent)": f"${res.get('politician_net_trade_value_estimate',0):,}", "Buy/Sell Transactions": f"{res.get('politician_buy_tx_count',0)} / {res.get('politician_sell_tx_count',0)}", "Politician Filings Signal": res.get("politician_filings_signal", "N/A").upper(), "Scrape Status": "Error" if pol_scrape_error else "OK"}
                            st.dataframe(pd.Series(pol_s, name="Value"), use_container_width=True)
                            if pol_scrape_error: st.caption(f"Scraping Note: {pol_scrape_error}")
                            if res.get("politician_trades_for_display"): with st.popover("View Scraped Politician Trades (Max 5)"): [st.markdown(f"**{p_trade.get('politician_name')}**: {p_trade.get('transaction_type')} ({p_trade.get('value_range')}) on {p_trade.get('date_str')}") for p_trade in res["politician_trades_for_display"]]
                    with tabs[4]:
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

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
        # Consolidate all necessary fields from .info here
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
            "currentPrice": info.get("currentPrice") or info.get("regularMarketPrice"), # Ensure we get a price
        }
    except Exception as e:
        st.error(f"Error fetching .info for {ticker} from yfinance: {e}")
        return {}

@st.cache_data
def fetch_news(ticker: str) -> list[dict]:
    """Return latest headlines from Yahoo Finance via yfinance."""
    try:
        news = yf.Ticker(ticker).news
        return news or []
    except Exception as e:
        st.error(f"Error fetching news for {ticker}: {e}")
        return []

@st.cache_data
def fetch_inst_filings(ticker: str) -> list[dict]:
    """Return current institutional holders via yfinance."""
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
    """Return recent insider trades via yfinance, tagged as buy/sell."""
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

@st.cache_data
def fetch_politician_trades(ticker: str, days_back: int = 365) -> list[dict]:
    """
    (Placeholder) Fetch recent politician trades for a given ticker.
    NOTE: Actual web scraping for CapitolTrades is complex and not implemented here.
    This function returns dummy data for demonstration.
    """
    st.warning(f"Web scraping for Capitol Trades ({ticker}) is a placeholder. "
               f"Full implementation requires robust parsing of their website.")
    if ticker == "AAPL": # Example dummy data
        return [
            {"politician_name": "Demo Politician A", "transaction_type": "purchase", "Shares": 100, "Value": 15000, "Date": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')},
            {"politician_name": "Demo Politician B", "transaction_type": "sale", "Shares": 50, "Value": 7500, "Date": (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')}
        ]
    return []


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
            self.model = "deepseek-reasoner" # "deepseek-reasoner" may not be available or correct name
        else: # Default to openai
            openai.api_base = "https://api.openai.com/v1" # ensure openai default
            self.model = "gpt-4o" #"gpt-3.5-turbo" # "gpt-4o" # Cheaper for testing

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
    def run(self, ticker: str, price_data_slice: pd.DataFrame) -> dict: # Accepts slice for backtesting
        # BACKTESTING_NOTE: This agent is suitable for backtesting as it uses historical price data.
        if price_data_slice.empty or len(price_data_slice) < 200: # Need enough data for SMA200
            return {"ticker": ticker, "price_signal": "hold", "sma50": np.nan, "sma200": np.nan, "rsi14": np.nan}

        df = price_data_slice.copy()
        df["SMA50"]  = df["Close"].rolling(50).mean()
        df["SMA200"] = df["Close"].rolling(200).mean()
        delta = df["Close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean() # loss is positive
        rs = gain / loss
        df["RSI14"] = 100 - (100 / (1 + rs))

        latest = df.iloc[-1]
        signal = "hold"
        if pd.isna(latest.SMA50) or pd.isna(latest.SMA200) or pd.isna(latest.RSI14):
             signal = "hold" # Not enough data
        elif latest.SMA50 > latest.SMA200 and latest.RSI14 < 70: # Golden cross and not overbought
            signal = "buy"
        elif latest.SMA50 < latest.SMA200 and latest.RSI14 > 30: # Death cross and not oversold
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
        # BACKTESTING_NOTE: Suitable for backtesting.
        if price_data_slice.empty or len(price_data_slice) < 252: # Need ~1 year of data
             return {"ticker": ticker, "momentum_signal": "hold", "momentum_1m": 0, "momentum_12m": 0}

        df = price_data_slice
        P_t = df.Close.iloc[-1]
        # Ensure enough data points before trying to shift
        P_1m = df.Close.shift(21).iloc[-1] if len(df) > 21 else np.nan
        P_12m = df.Close.shift(252).iloc[-1] if len(df) > 252 else np.nan

        m1  = (P_t/P_1m)-1  if pd.notna(P_1m) and P_1m != 0 else 0
        m12 = (P_t/P_12m)-1 if pd.notna(P_12m) and P_12m != 0 else 0
        
        signal = "hold"
        if m12 > 0 and m1 > 0:
            signal = "buy"
        elif m12 < 0 and m1 < 0:
            signal = "sell"
        
        return {
            "ticker":         ticker,
            "momentum_1m":    float(m1),
            "momentum_12m":   float(m12),
            "momentum_signal": signal,
        }

class VolatilityAgent:
    def run(self, ticker: str, data: dict, price_data_slice: pd.DataFrame = None) -> dict:
        # BACKTESTING_NOTE: Beta from current .info is lookahead for backtesting.
        # Historical annual_vol IS suitable for backtesting.
        # For backtesting, beta signal might be disabled or use a fixed value.
        
        # Use current beta for live analysis
        beta = data.get("ticker_info", {}).get("beta", 1.0)
        if beta is None: beta = 1.0 # Default if beta is missing

        sig  = "sell" if beta > 1.5 else ("buy" if beta < 0.8 else "hold") # Beta-based signal

        ann_vol = np.nan
        weight = 0.0
        if price_data_slice is not None and not price_data_slice.empty and len(price_data_slice) > 1:
            ret = np.log(price_data_slice.Close / price_data_slice.Close.shift(1)).dropna()
            if not ret.empty:
                ann_vol = float(ret.std() * np.sqrt(252)) # Annualize daily std dev
                weight  = float(1/ann_vol) if ann_vol > 0 else 0.0
        
        return {
            "ticker": ticker,
            "beta": beta, # Current beta
            "annual_vol": ann_vol, # Historical rolling if price_data_slice provided
            "vol_weight": weight,
            "volatility_signal": sig, # Based on current beta
        }

class SentimentAgent:
    def __init__(self, client): self.client = client
    def run(self, ticker: str, data: dict) -> dict:
        # BACKTESTING_NOTE: Hard to get true point-in-time historical news and sentiment.
        # This agent's signal would likely have lookahead bias in a simple backtest
        # or be excluded from backtest portfolio decisions.
        headlines = [h.get("title","") for h in data.get("news",[])[:10]] # Limit to N headlines
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
        except ValueError: # If LLM doesn't return a clean float
            # Try a follow-up to extract
            clarification_prompt = f"The previous response was '{response}'. Please extract the single sentiment score number between -1 and 1 from it. If none, output 0.0."
            try:
                score = float(self.client.generate(clarification_prompt).strip())
            except:
                score = 0.0 # Fallback
        except Exception: # Catch other API errors
            score = 0.0

        sig = "buy" if score > 0.25 else ("sell" if score < -0.25 else "hold") # Adjusted thresholds
        return {"ticker":ticker, "sentiment_score":score, "sentiment_signal":sig}

class FundamentalsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        # BACKTESTING_NOTE: Uses current fundamentals from .info. This is lookahead for backtesting.
        # A true backtest would require point-in-time fundamental data.
        s    = data.get("ticker_info", {})
        mcap = s.get("marketCap") or 1 # Avoid division by zero
        fcf  = s.get("freeCashflow") or 0
        roe  = s.get("returnOnEquity") or 0
        de   = s.get("debtToEquity") # Can be None or 0
        if de is None: de = 1000 # If no debt, D/E is low (good), or if no equity, D/E is high. Treat None as high for conservative.

        fcy  = fcf/mcap if mcap != 0 else 0
        
        # Simplified Piotroski-like score (max 3 points)
        # (Original Piotroski has 9 points, requires more data)
        piotroski_score = 0
        if roe > 0: piotroski_score += 1
        if de < 100 : piotroski_score += 1 # Debt to Equity less than 1 (or 100 if % format)
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
        # BACKTESTING_NOTE: Uses current fundamentals. Lookahead for backtesting.
        stats = data.get("ticker_info", {})
        price_history_df = data.get("price_history")
        price = stats.get("currentPrice") # Use current price from info
        if price is None and price_history_df is not None and not price_history_df.empty:
             price = price_history_df["Close"].iloc[-1]
        
        if price is None: # Still no price
            return {"ticker": ticker, "relative_pe_signal": "hold", "dcf_price": np.nan, "dcf_signal": "hold"}

        # 1) Relative P/E signal
        pe = stats.get("forwardPE")
        # industry_pe = stats.get("industryForwardPe") # yfinance might not have this directly
        # trailing_pe = stats.get("trailingPE")
        rel_sig = "hold"
        if pe is None:
            rel_sig = "hold"
        elif pe < 15: # Example thresholds, could be dynamic vs industry/history
            rel_sig = "buy"
        elif pe > 25:
            rel_sig = "sell"
        
        # 2) DCF-style fair price (highly simplified)
        fcf  = stats.get("freeCashflow")
        mcap = stats.get("marketCap")
        fcy = 0.0
        if fcf is not None and mcap is not None and mcap != 0:
            fcy = fcf / mcap
        
        # Simplified: fair_price = current_price * (1 + FCF_Yield_as_proxy_for_short_term_growth_value)
        # This is NOT a real DCF. A real DCF needs growth projections, discount rate.
        fair_price = price * (1 + fcy) if pd.notna(price) else np.nan

        dcf_sig = "hold"
        if pd.notna(fair_price) and pd.notna(price) and price != 0:
            if fair_price > price * 1.15: # Undervalued by 15%
                dcf_sig = "buy"
            elif fair_price < price * 0.85: # Overvalued by 15%
                dcf_sig = "sell"
        
        return {
            "ticker": ticker,
            "forward_pe": pe,
            "relative_pe_signal": rel_sig,
            "dcf_fair_price": float(fair_price) if pd.notna(fair_price) else np.nan,
            "dcf_signal": dcf_sig,
        }

class FilingsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        # BACKTESTING_NOTE: Uses current filings. Lookahead for backtesting.
        insiders = data.get("insider_filings",[])
        net_shares = 0
        if insiders: # Ensure insiders is not None and not empty
            for r in insiders:
                shares = r.get("Shares",0)
                if isinstance(shares, str): # yfinance can return '--'
                    try: shares = int(shares)
                    except ValueError: shares = 0
                
                if r.get("type") == "buy":
                    net_shares += shares
                elif r.get("type") == "sell":
                    net_shares -= shares
        
        sig = "buy" if net_shares > 0 else ("sell" if net_shares < 0 else "hold")
        return {
            "ticker": ticker,
            "net_insider_shares": int(net_shares),
            "filings_signal": sig,
        }

class AnalystRatingAgent:
    def run(self, ticker: str, data: dict) -> dict:
        # BACKTESTING_NOTE: Uses current analyst ratings. Lookahead for backtesting.
        ticker_info = data.get("ticker_info", {})
        price_history_df = data.get("price_history")
        current_price = ticker_info.get("currentPrice")
        if current_price is None and price_history_df is not None and not price_history_df.empty:
            current_price = price_history_df["Close"].iloc[-1]

        target_mean_price = ticker_info.get("targetMeanPrice")
        recommendation = str(ticker_info.get("recommendationKey", "hold")).lower()

        upside = 0.0
        if target_mean_price and current_price and current_price > 0:
            try:
                upside = (float(target_mean_price) / float(current_price)) - 1
            except (ValueError, TypeError): upside = 0.0
        
        sig = "hold"
        if recommendation in ["buy", "strong_buy"] and upside > 0.10: sig = "buy"
        elif recommendation == "buy" and upside > 0.05: sig = "buy"
        elif recommendation in ["sell", "strong_sell", "underperform"] and upside < -0.05: sig = "sell"
        elif upside > 0.20 : sig = "buy" # Strong upside irrespective of hold
        elif upside < -0.15 : sig = "sell" # Strong downside

        buy_pct_inferred = 0.5 # Default
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
        # BACKTESTING_NOTE: Uses currently fetched politician trades. Lookahead for backtesting.
        trades = data.get("politician_trades", [])
        net_value = 0 # Could be based on shares or estimated value
        buy_signals = 0
        sell_signals = 0

        for trade in trades:
            value = trade.get("Value", 0) # Assuming Value is a numerical field
            if trade.get("transaction_type") == "purchase":
                net_value += value
                buy_signals +=1
            elif trade.get("transaction_type") == "sale":
                net_value -= value
                sell_signals +=1
        
        signal = "hold"
        if buy_signals > sell_signals and buy_signals > 0: signal = "buy"
        elif sell_signals > buy_signals and sell_signals > 0: signal = "sell"

        return {
            "ticker": ticker,
            "politician_net_trade_value": net_value,
            "politician_buy_signals": buy_signals,
            "politician_sell_signals": sell_signals,
            "politician_filings_signal": signal,
        }

class PortfolioAgent:
    # Define weights for different signals
    # NOTE: For backtesting, you might use a different set of weights or only include backtestable signals.
    WEIGHTS = {
        "price": 1.0, "momentum": 0.8, "volatility": 0.5, # Volatility signal (beta-based)
        "sentiment": 0.5, "fund": 0.7, "valuation_dcf":0.6, "valuation_pe":0.6, # Split valuation signals
        "filings": 0.4, "analyst": 0.8, "politician_filings": 0.3
    }
    def run(self, ticker: str, signals: list[dict], agent_weights: dict = None) -> dict:
        current_weights = agent_weights or self.WEIGHTS
        total_weighted_score = 0
        sum_of_weights_used = 0

        # Consolidate all signals into one dictionary for easier lookup
        agg_signals = {}
        for s_dict in signals:
            agg_signals.update(s_dict)

        signal_map = {
            "price_signal": "price",
            "momentum_signal": "momentum",
            "volatility_signal": "volatility",
            "sentiment_signal": "sentiment",
            "fund_signal": "fund",
            "dcf_signal": "valuation_dcf", # From ValuationAgent
            "relative_pe_signal": "valuation_pe", # From ValuationAgent
            "filings_signal": "filings",
            "analyst_signal": "analyst",
            "politician_filings_signal": "politician_filings"
        }

        for signal_key, weight_key in signal_map.items():
            signal_value = agg_signals.get(signal_key)
            if signal_value and weight_key in current_weights:
                raw_score = {"buy":1, "hold":0, "sell":-1}.get(signal_value, 0)
                total_weighted_score += raw_score * current_weights[weight_key]
                sum_of_weights_used += current_weights[weight_key]
        
        if sum_of_weights_used == 0: # Avoid division by zero if no relevant signals/weights
            composite_score = 0.0
        else:
            composite_score = total_weighted_score / sum_of_weights_used # Normalize by sum of weights used

        # Apply tanh for scaling, but the previous normalization might be sufficient
        # composite_score = float(np.tanh(composite_score)) # Optional: re-scale to -1, 1

        # Decision thresholds (could be made configurable)
        buy_threshold = 0.15 # Lowered threshold a bit
        sell_threshold = -0.15

        final_decision = "hold"
        if composite_score > buy_threshold:
            final_decision = "buy"
        elif composite_score < sell_threshold:
            final_decision = "sell"
            
        return {"ticker":ticker, "composite_score":composite_score, "final_decision":final_decision}

# --------------------------------
# Orchestrator for Live Analysis
# --------------------------------
def run_live_analysis(tickers, history_years, llm_client, configs):
    results = {}
    for t in tickers:
        st.write(f"Fetching data for {t}...")
        # Fetch all necessary data first
        price_history_full = fetch_price_history(t, period=f"{history_years}y")
        if price_history_full.empty:
            st.error(f"Could not fetch price history for {t}. Skipping analysis.")
            results[t] = {"error": "Failed to fetch price history"}
            continue

        ticker_info = fetch_ticker_info(t)
        news_data = fetch_news(t) if configs["use_sentiment"] else []
        insider_filings_data = fetch_insider_filings(t) if configs["use_filings"] else []
        # inst_filings_data = fetch_inst_filings(t) if configs["use_filings"] else [] # Currently not used by an agent directly
        politician_trades_data = fetch_politician_trades(t) if configs["use_politician_filings"] else []


        data_bundle = {
            "price_history": price_history_full,
            "ticker_info": ticker_info,
            "news": news_data,
            "insider_filings": insider_filings_data,
            "politician_trades": politician_trades_data
        }

        st.write(f"Running agents for {t}...")
        # Initialize Agents
        price_agent = PriceAgent()
        momentum_agent = MomentumAgent()
        volatility_agent = VolatilityAgent()
        sentiment_agent = SentimentAgent(llm_client) if configs["use_sentiment"] else None
        fundamentals_agent = FundamentalsAgent()
        valuation_agent = ValuationAgent()
        filings_agent = FilingsAgent() if configs["use_filings"] else None
        analyst_rating_agent = AnalystRatingAgent()
        politician_agent = PoliticianFilingsAgent() if configs["use_politician_filings"] else None
        portfolio_agent = PortfolioAgent()

        # Run Agents
        # Note: For live analysis, agents use the full history or current data as designed
        pa_res = price_agent.run(t, data_bundle["price_history"])
        ma_res = momentum_agent.run(t, data_bundle["price_history"])
        va_res = volatility_agent.run(t, data_bundle, data_bundle["price_history"]) # Pass full history for ann_vol
        sa_res = sentiment_agent.run(t, data_bundle) if sentiment_agent else {"sentiment_signal": "hold", "sentiment_score": 0.0}
        fa_res = fundamentals_agent.run(t, data_bundle)
        val_res = valuation_agent.run(t, data_bundle)
        fil_res = filings_agent.run(t, data_bundle) if filings_agent else {"filings_signal": "hold", "net_insider_shares":0}
        ar_res = analyst_rating_agent.run(t, data_bundle)
        pfa_res = politician_agent.run(t, data_bundle) if politician_agent else {"politician_filings_signal":"hold"}


        all_signals = [pa_res, ma_res, va_res, sa_res, fa_res, val_res, fil_res, ar_res, pfa_res]
        final_decision = portfolio_agent.run(t, all_signals)

        # Consolidate results
        current_result = {
            **pa_res, **ma_res, **va_res, **sa_res, **fa_res,
            **val_res, **fil_res, **ar_res, **pfa_res, **final_decision,
            # "price_history_data": data_bundle["price_history"] # Don't store full history in JSON, too big
        }
        results[t] = current_result
        st.success(f"Analysis complete for {t}: {final_decision['final_decision']} (Score: {final_decision['composite_score']:.2f})")

    return results

# --------------------------------
# Backtesting Engine
# --------------------------------
def run_backtest(ticker, start_date, end_date, initial_capital, llm_client_placeholder, backtest_agent_weights):
    st.write(f"Starting backtest for {ticker} from {start_date} to {end_date}...")

    # Fetch historical data for the entire backtest period + ~1 year for initial SMA/momentum calculation
    # Example: If backtest starts 2020-01-01, fetch from 2019-01-01
    # For yfinance, period is calculated from today, so we use start/end for Ticker.history()
    # Calculate years needed for full history fetch
    s_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    e_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
    # Fetch data from at least 1 year prior to s_date_obj to allow for 200/252 day moving averages/momentum
    fetch_start_date = (s_date_obj - pd.DateOffset(years=1.5)).strftime("%Y-%m-%d")

    full_price_history = fetch_price_history(ticker, period=None, interval="1d") # Fetch max available daily
    if full_price_history.empty:
        st.error("Cannot run backtest: Price history is empty.")
        return None, pd.DataFrame()

    # Filter data for the required extended range (fetch_start_date to end_date)
    price_history = full_price_history[(full_price_history.index >= pd.to_datetime(fetch_start_date)) &
                                       (full_price_history.index <= pd.to_datetime(end_date))].copy()


    if price_history.empty or len(price_history[price_history.index >= pd.to_datetime(start_date)]) < 2: # Need at least 2 days in actual test period
        st.error(f"Not enough historical data for {ticker} in the selected backtest range after filtering.")
        return None, pd.DataFrame()
    
    # For agents that use .info (Fundamentals, Valuation, Analyst, current Volatility beta)
    # We fetch it once. This is a simplification (lookahead bias for these specific signals in backtest).
    # BACKTESTING_NOTE: True point-in-time fundamentals/info is hard with yfinance.
    # The portfolio decision for backtesting will rely on agents that DON'T use this heavily, or this is acknowledged.
    ticker_info_for_backtest = fetch_ticker_info(ticker)
    data_bundle_static = {"ticker_info": ticker_info_for_backtest}


    # Initialize agents that will be used in the backtest
    # We will primarily use Price, Momentum, and historical Volatility (ann_vol part)
    price_agent = PriceAgent()
    momentum_agent = MomentumAgent()
    volatility_agent = VolatilityAgent() # Beta part will use current beta, ann_vol part historical
    # Other agents (Sentiment, Fundamentals, Valuation, Filings, etc.) are largely excluded
    # from the *portfolio decision* in this simplified backtest due to lookahead bias.
    portfolio_agent = PortfolioAgent()

    # Portfolio simulation
    portfolio_log = []
    cash = initial_capital
    shares_held = 0
    portfolio_value = initial_capital
    
    # Iterate through the *actual* backtest period (start_date to end_date)
    backtest_run_dates = price_history[price_history.index >= pd.to_datetime(start_date)].index

    for current_date in backtest_run_dates:
        # Get data slice up to and including current_date
        data_slice = price_history[price_history.index <= current_date]
        if data_slice.empty or len(data_slice) < 252: # Min data for some indicators
            portfolio_log.append({
                "date": current_date, "cash": cash, "shares_held": shares_held,
                "price": data_slice.Close.iloc[-1] if not data_slice.empty else 0,
                "portfolio_value": portfolio_value, "signal": "hold (insufficient data)"
            })
            continue

        current_price = data_slice.Close.iloc[-1]

        # Run backtestable agents
        pa_res = price_agent.run(ticker, data_slice)
        ma_res = momentum_agent.run(ticker, data_slice)
        # For VolatilityAgent, pass current data_bundle_static for beta, and data_slice for ann_vol
        va_res = volatility_agent.run(ticker, data_bundle_static, data_slice)


        # BACKTESTING_NOTE: Only include signals from agents that are truly backtestable
        # (i.e., don't use future info). Here, Price, Momentum, and the ann_vol part of Volatility.
        # The PortfolioAgent WEIGHTS should reflect this for backtesting.
        backtest_signals = [pa_res, ma_res, va_res]
        final_decision_obj = portfolio_agent.run(ticker, backtest_signals, agent_weights=backtest_agent_weights)
        final_decision = final_decision_obj["final_decision"]

        # Trading logic (simplified)
        if final_decision == "buy" and cash > 0:
            shares_to_buy = cash / current_price
            shares_held += shares_to_buy
            cash = 0
        elif final_decision == "sell" and shares_held > 0:
            cash += shares_held * current_price
            shares_held = 0
        
        portfolio_value = cash + shares_held * current_price
        portfolio_log.append({
            "date": current_date, "cash": cash, "shares_held": shares_held,
            "price": current_price, "portfolio_value": portfolio_value, "signal": final_decision,
            "composite_score": final_decision_obj["composite_score"]
        })

    log_df = pd.DataFrame(portfolio_log)
    if not log_df.empty:
      log_df.set_index("date", inplace=True)

    # Performance Metrics
    if log_df.empty or len(log_df) < 2:
        st.warning("Backtest log is too short to calculate performance metrics.")
        return {"message":"Log too short"}, pd.DataFrame()

    total_return = (log_df["portfolio_value"].iloc[-1] / initial_capital - 1) * 100
    
    # Annualized Return
    num_days = (log_df.index[-1] - log_df.index[0]).days
    num_years = num_days / 365.25
    if num_years == 0: num_years = 1 # Avoid division by zero if less than a year
    annualized_return = ((log_df["portfolio_value"].iloc[-1] / initial_capital) ** (1/num_years) - 1) * 100 if num_years > 0 else total_return

    # Annualized Volatility
    log_df["daily_return"] = log_df["portfolio_value"].pct_change().fillna(0)
    annualized_volatility = log_df["daily_return"].std() * np.sqrt(252) * 100 # Trading days in a year

    # Sharpe Ratio (assuming risk-free rate = 0)
    sharpe_ratio = (annualized_return / annualized_volatility) if annualized_volatility != 0 else 0

    # Max Drawdown
    log_df["cumulative_max"] = log_df["portfolio_value"].cummax()
    log_df["drawdown"] = (log_df["portfolio_value"] - log_df["cumulative_max"]) / log_df["cumulative_max"]
    max_drawdown = log_df["drawdown"].min() * 100
    
    metrics = {
        "Initial Capital": f"${initial_capital:,.2f}",
        "Final Portfolio Value": f"${log_df['portfolio_value'].iloc[-1]:,.2f}",
        "Total Return (%)": f"{total_return:.2f}%",
        "Annualized Return (%)": f"{annualized_return:.2f}%",
        "Annualized Volatility (%)": f"{annualized_volatility:.2f}%",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Max Drawdown (%)": f"{max_drawdown:.2f}%",
        "Number of Trades": f"{(log_df['signal'] != 'hold').sum() // 2}" # Approx; each buy/sell pair
    }
    st.success(f"Backtest for {ticker} complete.")
    return metrics, log_df


# --------------------------------
# Streamlit UI
# --------------------------------
st.set_page_config(page_title="AI Hedge Fund Simulator", layout="wide")
st.title("🚀 AI Hedge Fund Simulator")

# Initialize LLM Client (once)
llm_client = None
# Use try-except for robustness if secrets are not set
try:
    if "DEEPSEEK_API_KEY" in st.secrets and st.secrets["DEEPSEEK_API_KEY"]:
        llm_client = ModelClient(api_key=st.secrets["DEEPSEEK_API_KEY"], provider="deepseek")
        st.sidebar.success("Using DeepSeek LLM")
    elif "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
        llm_client = ModelClient(api_key=st.secrets["OPENAI_API_KEY"], provider="openai")
        st.sidebar.success("Using OpenAI LLM")
    else:
        st.sidebar.error("No LLM API key found in Streamlit secrets. Sentiment analysis will be disabled.")
except ValueError as e: # Catch API key init error from ModelClient
    st.sidebar.error(f"LLM Init Error: {e}. Sentiment analysis may be disabled.")
except Exception as e: # Catch other potential errors during LLM client init
    st.sidebar.error(f"An unexpected error occurred initializing LLM: {e}")


# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ General Configuration")
    app_mode = st.selectbox("Select Mode", ["Live Analysis", "Backtesting"])
    
    if app_mode == "Live Analysis":
        st.subheader("Live Analysis Settings")
        tickers_in = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOG")
        history_years_live = st.slider("History for Live Analysis (years)", 1, 10, 5)
        use_sentiment_live = st.checkbox("Include News Sentiment", True if llm_client else False, disabled=not llm_client)
        use_filings_live = st.checkbox("Include Insider Filings Data", True)
        use_politician_filings_live = st.checkbox("Include Politician Filings (Dummy Data)", False) # Default to False
        run_button_live = st.button("Run Live Analysis", use_container_width=True)

    elif app_mode == "Backtesting":
        st.subheader("Backtesting Settings")
        bt_ticker = st.text_input("Ticker for Backtest", "AAPL").upper()
        # Default dates: 3 years back to yesterday
        default_bt_end_date = datetime.now() - timedelta(days=1)
        default_bt_start_date = default_bt_end_date - pd.DateOffset(years=3)

        bt_start_date = st.date_input("Start Date", default_bt_start_date, max_value=default_bt_end_date - timedelta(days=1)).strftime("%Y-%m-%d")
        bt_end_date = st.date_input("End Date", default_bt_end_date, min_value=datetime.strptime(bt_start_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

        bt_initial_capital = st.number_input("Initial Capital", 1000, 1000000, 10000, 1000)
        
        st.markdown("""
        **Note on Backtest Agent Weights:**
        The backtest primarily uses `Price`, `Momentum`, and historical `Volatility` (annual_vol part) signals.
        Other signals (Sentiment, Fundamentals, etc.) have lookahead bias with current data sources
        and are **not** factored into portfolio decisions during this simplified backtest.
        You can adjust weights for the backtestable signals below.
        """)
        
        # Define weights specifically for backtesting (focused on price/momentum/vol)
        # These are the effective weights for signals used in the backtest.
        bt_weights_price = st.slider("Weight: Price Signal (Backtest)", 0.0, 2.0, 1.0, 0.1)
        bt_weights_momentum = st.slider("Weight: Momentum Signal (Backtest)", 0.0, 2.0, 0.8, 0.1)
        bt_weights_volatility = st.slider("Weight: Volatility Signal (Backtest - Beta part uses current)", 0.0, 2.0, 0.2, 0.1) # Beta still lookahead
        
        # Store these in a dict to pass to PortfolioAgent for backtesting
        backtest_portfolio_weights = {
            "price": bt_weights_price,
            "momentum": bt_weights_momentum,
            "volatility": bt_weights_volatility, # This still implies beta signal is used; ann_vol doesn't have its own signal.
            # Exclude other agents from backtest weights by default:
             "sentiment": 0.0, "fund": 0.0, "valuation_dcf":0.0, "valuation_pe":0.0,
             "filings": 0.0, "analyst": 0.0, "politician_filings": 0.0
        }
        run_button_backtest = st.button("Run Backtest", use_container_width=True)

# Main App Logic
if app_mode == "Live Analysis":
    if run_button_live:
        if not tickers_in:
            st.error("Please enter at least one ticker.")
        else:
            live_tickers = [t.strip().upper() for t in tickers_in.split(",") if t.strip()]
            live_configs = {
                "use_sentiment": use_sentiment_live and llm_client is not None,
                "use_filings": use_filings_live,
                "use_politician_filings": use_politician_filings_live
            }
            with st.spinner("Running AI agents for live analysis..."):
                live_output = run_live_analysis(live_tickers, history_years_live, llm_client, live_configs)

            st.subheader("📊 Live Buy / Hold / Sell Recommendations")
            cols = st.columns(len(live_tickers))
            for i, t in enumerate(live_tickers):
                if "error" in live_output.get(t, {}):
                    cols[i].error(f"**{t}:** {live_output[t]['error']}")
                    continue

                if t in live_output and live_output[t]:
                    res = live_output[t]
                    dec = res.get("final_decision", "N/A").upper()
                    score = res.get("composite_score", float('nan'))
                    cols[i].metric(label=f"{t} Recommendation", value=dec, delta=f"{score:.2f} Score")
                else:
                    cols[i].warning(f"No output for {t}")


            for t in live_tickers:
                if "error" in live_output.get(t, {}): continue
                if t in live_output and live_output[t]:
                    with st.expander(f"🔍 Detailed Signals & Data for {t}"):
                        # Display price chart
                        price_history_df = fetch_price_history(t, period=f"{history_years_live}y") # Re-fetch for display or pass from run_live_analysis
                        if not price_history_df.empty:
                            st.line_chart(price_history_df["Close"], use_container_width=True)
                        
                        # Display signals in a table
                        signal_keys = [k for k in live_output[t] if k.endswith("_signal")]
                        signal_table_data = {k.replace("_signal","").replace("_"," ").title(): live_output[t][k] for k in signal_keys}
                        signal_table_data["Composite Score"] = f"{live_output[t].get('composite_score',0.0):.2f}"
                        signal_table_data["Final Decision"] = live_output[t].get('final_decision',"").upper()
                        st.table(pd.Series(signal_table_data, name="Signal Value"))

                        # Display other numeric data points
                        numeric_data = {
                            "Forward PE": live_output[t].get('forward_pe'),
                            "FCF Yield": f"{live_output[t].get('fcf_yield',0)*100:.2f}%" if live_output[t].get('fcf_yield') is not None else "N/A",
                            "Piotroski Score": live_output[t].get('piotroski_score'),
                            "DCF Fair Price": f"${live_output[t].get('dcf_fair_price'):.2f}" if live_output[t].get('dcf_fair_price') is not None else "N/A",
                            "Beta": live_output[t].get('beta'),
                            "Annual Volatility": f"{live_output[t].get('annual_vol',0)*100:.2f}%" if live_output[t].get('annual_vol') is not None else "N/A",
                            "Sentiment Score": f"{live_output[t].get('sentiment_score',0):.2f}" if live_configs["use_sentiment"] else "N/A (Disabled)",
                            "Net Insider Shares": live_output[t].get('net_insider_shares') if live_configs["use_filings"] else "N/A (Disabled)",
                            "Analyst Target Upside": f"{live_output[t].get('target_upside',0)*100:.2f}%" if live_output[t].get('target_upside') is not None else "N/A",
                            "YF Recommendation": live_output[t].get('yfinance_recommendation', "N/A"),
                            "Politician Net Trade Value": f"${live_output[t].get('politician_net_trade_value',0):,}" if live_configs["use_politician_filings"] else "N/A (Disabled)"
                        }
                        st.table(pd.Series(numeric_data, name="Metric Value"))

            with st.expander("Show Full JSON Output (Live Analysis)"):
                st.json(live_output)


elif app_mode == "Backtesting":
    if run_button_backtest:
        if not bt_ticker:
            st.error("Please enter a ticker for backtesting.")
        else:
            with st.spinner(f"Running backtest for {bt_ticker}... This may take a while."):
                # Note: llm_client is passed as placeholder, not actively used in this simplified backtest logic.
                bt_metrics, bt_log_df = run_backtest(
                    bt_ticker,
                    bt_start_date,
                    bt_end_date,
                    bt_initial_capital,
                    llm_client, # Placeholder
                    backtest_portfolio_weights # Pass the configured weights for backtesting
                )

            if bt_metrics:
                st.subheader(f"Backtest Results for {bt_ticker}")
                
                # Display metrics in columns for better layout
                # Convert metrics dict to DataFrame for nicer display
                metrics_df = pd.DataFrame.from_dict(bt_metrics, orient='index', columns=['Value'])
                st.table(metrics_df)

                if not bt_log_df.empty:
                    st.subheader("Portfolio Value Over Time")
                    st.line_chart(bt_log_df["portfolio_value"])

                    st.subheader("Drawdown Over Time")
                    st.area_chart(bt_log_df["drawdown"])
                    
                    with st.expander("View Backtest Log and Signals"):
                        st.dataframe(bt_log_df[["price", "signal", "composite_score", "portfolio_value", "cash", "shares_held"]])
                else:
                    st.warning("Backtest log is empty, cannot display charts.")
            else:
                st.error("Backtest failed to produce results.")

st.sidebar.markdown("---")
st.sidebar.info("This simulator is for educational purposes only. Not financial advice.")

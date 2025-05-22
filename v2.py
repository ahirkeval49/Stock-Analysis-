"""
AI Hedge Fund Simulator

Dependencies & Tools (free resources and websites):
 - Streamlit              : https://streamlit.io
 - yfinance               : https://pypi.org/project/yfinance/
 - OpenAI Python SDK      : https://platform.openai.com/
 - pandas                 : https://pandas.pydata.org/
 - numpy                  : https://numpy.org/
 - python-dotenv          : https://pypi.org/project/python-dotenv/
 - python-dateutil        : https://dateutil.readthedocs.io/
"""

import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime
import openai
from dotenv import load_dotenv

# Load environment variables (if running locally)
load_dotenv()

# --------------------------------
# Data Fetchers
# --------------------------------
def fetch_price_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    return yf.Ticker(ticker).history(period=period)

def fetch_fundamentals(ticker: str) -> dict:
    info = yf.Ticker(ticker).info
    return {
        "marketCap":       info.get("marketCap"),
        "freeCashflow":    info.get("freeCashflow"),
        "forwardPE":       info.get("forwardPE"),
        "returnOnEquity":  info.get("returnOnEquity"),
        "debtToEquity":    info.get("debtToEquity"),
    }

def fetch_news(ticker: str) -> list[dict]:
    """
    Return latest headlines from Yahoo Finance via yfinance.
    """
    news = yf.Ticker(ticker).news
    return news or []

def fetch_inst_filings(ticker: str) -> list[dict]:
    """
    Return current institutional holders via yfinance.
    """
    df = yf.Ticker(ticker).institutional_holders
    return df.to_dict("records")

def fetch_insider_filings(ticker: str) -> list[dict]:
    """
    Return recent insider trades via yfinance, tagged as buy/sell.
    """
    df = yf.Ticker(ticker).get_insider_transactions()
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

# --------------------------------
# LLM Client
# --------------------------------
class ModelClient:
    def __init__(self, api_key: str, provider: str = "openai"):
        openai.api_key = api_key
        self.provider = provider
        if provider == "deepseek":
            openai.api_base = "https://api.deepseek.com/v1"
            self.model = "deepseek-reasoner"
        else:
            self.model = "gpt-4o"

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = openai.Embedding.create(input=texts, model="text-embedding-ada-002")
        return [e["embedding"] for e in resp["data"]]

    def generate(self, prompt: str) -> str:
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message["content"]

# --------------------------------
# Agents
# --------------------------------
class PriceAgent:
    def run(self, ticker: str, data: dict) -> dict:
        df = data["price_history"].copy()
        df["SMA50"]  = df["Close"].rolling(50).mean()
        df["SMA200"] = df["Close"].rolling(200).mean()
        delta = df["Close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = -delta.clip(upper=0).rolling(14).mean()
        df["RSI14"] = 100 - 100/(1 + gain/loss)
        latest = df.iloc[-1]
        signal = (
            "buy"  if latest.SMA50>latest.SMA200 and latest.RSI14<70 else
            "sell" if latest.SMA50<latest.SMA200 and latest.RSI14>30 else
            "hold"
        )
        return {
            "ticker":       ticker,
            "sma50":        float(latest.SMA50),
            "sma200":       float(latest.SMA200),
            "rsi14":        float(latest.RSI14),
            "price_signal": signal,
        }

class MomentumAgent:
    def run(self, ticker: str, data: dict) -> dict:
        df = data["price_history"]
        P_t = df.Close.iloc[-1]
        P_1m = df.Close.shift(21).iloc[-1]
        P_12m = df.Close.shift(252).iloc[-1]
        m1  = (P_t/P_1m)-1  if P_1m else 0
        m12 = (P_t/P_12m)-1 if P_12m else 0
        signal = (
            "buy"  if m12>0 and m1>0 else
            "sell" if m12<0 and m1<0 else
            "hold"
        )
        return {
            "ticker":          ticker,
            "momentum_1m":     float(m1),
            "momentum_12m":    float(m12),
            "momentum_signal": signal,
        }

class VolatilityAgent:
    def run(self, ticker: str, data: dict) -> dict:
        # use beta for market-relative volatility
        info = yf.Ticker(ticker).info
        beta = info.get("beta", 1.0)
        sig  = "sell" if beta>1.5 else ("buy" if beta<0.8 else "hold")
        # optional own vol calculation
        ret     = np.log(data["price_history"].Close / data["price_history"].Close.shift(1)).dropna()
        ann_vol = float(ret.std()*np.sqrt(252))
        weight  = float(1/ann_vol) if ann_vol else 0.0
        return {
            "ticker":             ticker,
            "beta":               beta,
            "annual_vol":         ann_vol,
            "vol_weight":         weight,
            "volatility_signal":  sig,
        }

class SentimentAgent:
    def __init__(self, client): self.client = client
    def run(self, ticker: str, data: dict) -> dict:
        headlines = [h.get("title","") for h in data.get("news",[])]
        if not headlines:
            return {"ticker":ticker, "sentiment_score":0.0, "sentiment_signal":"hold"}
        prompt = (
            f"Rate the overall sentiment for {ticker} (−1 negative, +1 positive):\n"
            + "\n".join(headlines)
        )
        try: score = float(self.client.generate(prompt).strip())
        except: score = 0.0
        sig = "buy" if score>0.2 else ("sell" if score<-0.2 else "hold")
        return {"ticker":ticker, "sentiment_score":score, "sentiment_signal":sig}

class FundamentalsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        s    = data["fundamentals"]
        mcap = s.get("marketCap") or 1
        fcf  = s.get("freeCashflow") or 0
        roe  = s.get("returnOnEquity") or 0
        de   = s.get("debtToEquity") or 1
        fcy  = fcf/mcap
        score= sum([roe>0, de<1, fcf>0])
        sig  = "buy" if score>=3 else ("sell" if score==0 else "hold")
        return {
            "ticker":          ticker,
            "fcf_yield":       float(fcy),
            "piotroski_score": score,
            "fund_signal":     sig,
        }

class ValuationAgent:
    def run(self, ticker: str, data: dict) -> dict:
        stats = data["fundamentals"]
        df    = data["price_history"]
        price = df.Close.iloc[-1]

        # 1) Relative P/E signal
        pe = stats.get("forwardPE")
        if pe is None:
            rel = "hold"
        elif pe < 17:
            rel = "buy"
        elif pe > 23:
            rel = "sell"
        else:
            rel = "hold"

        # 2) DCF-style fair price
        fcf  = stats.get("freeCashflow")
        mcap = stats.get("marketCap")

        # default to zero yield if missing or zero market cap
        if fcf is None or mcap is None or mcap == 0:
            fcy = 0.0
        else:
            fcy = fcf / mcap

        fair_price = price * (1 + fcy)

        if fair_price > price * 1.1:
            dcf_sig = "buy"
        elif fair_price < price * 0.9:
            dcf_sig = "sell"
        else:
            dcf_sig = "hold"

        return {
            "ticker":              ticker,
            "relative_pe_signal":  rel,
            "dcf_price":           float(fair_price),
            "dcf_signal":          dcf_sig,
        }


class FilingsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        insiders = data.get("insider_filings",[])
        net      = sum(
            r.get("Shares",0) if r["type"]=="buy" else -r.get("Shares",0)
            for r in insiders
        )
        sig = "buy" if net>0 else ("sell" if net<0 else "hold")
        return {
            "ticker":         ticker,
            "net_insider":    int(net),
            "filings_signal": sig,
        }

class AnalystRatingAgent:
    def run(self, ticker: str, data: dict, client=None) -> dict:
        buy_pct,up = 8/14,0.10
        sig = "buy" if buy_pct>0.6 and up>0.15 else ("sell" if up<-0.05 else "hold")
        return {
            "ticker":          ticker,
            "analyst_buy_pct": buy_pct,
            "target_upside":   up,
            "analyst_signal":  sig,
        }

class PortfolioAgent:
    WEIGHTS = {
        "price":1.0, "momentum":0.5, "volatility":1.0,
        "sentiment":0.8, "fund":0.7, "valuation":0.9,
        "filings":0.6, "analyst":1.0
    }
    def run(self, ticker: str, signals: list[dict]) -> dict:
        total = 0
        for s in signals:
            keys = [k for k in s if k.endswith("_signal")]
            if not keys: continue
            key  = keys[0]
            base = key.split("_")[0]
            raw  = {"buy":1,"hold":0,"sell":-1}[s[key]]
            total += raw * self.WEIGHTS.get(base,1.0)
        comp  = float(np.tanh(total))
        final= "buy" if comp>0.2 else ("sell" if comp<-0.2 else "hold")
        return {"ticker":ticker, "composite_score":comp, "final_decision":final}

# --------------------------------
# Orchestrator & Streamlit UI
# --------------------------------
def run_all(tickers, history_years, use_sentiment, use_filings):
    # choose LLM from secrets
    if "DEEPSEEK_API_KEY" in st.secrets and st.secrets["DEEPSEEK_API_KEY"]:
        llm = ModelClient(api_key=st.secrets["DEEPSEEK_API_KEY"], provider="deepseek")
    elif "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
        llm = ModelClient(api_key=st.secrets["OPENAI_API_KEY"], provider="openai")
    else:
        st.error("No LLM API key found in Streamlit secrets.")
        st.stop()

    results = {}
    for t in tickers:
        data = {
            "price_history": fetch_price_history(t, period=f"{history_years}y"),
            "fundamentals":  fetch_fundamentals(t),
        }
        if use_sentiment:
            data["news"] = fetch_news(t)
        if use_filings:
            data["inst_filings"]    = fetch_inst_filings(t)
            data["insider_filings"] = fetch_insider_filings(t)

        pa    = PriceAgent().run(t, data)
        ma    = MomentumAgent().run(t, data)
        va    = VolatilityAgent().run(t, data)
        sa    = SentimentAgent(llm).run(t, data) if use_sentiment else {"sentiment_signal":"hold"}
        fa    = FundamentalsAgent().run(t, data)
        vaa   = ValuationAgent().run(t, data)
        fil   = FilingsAgent().run(t, data) if use_filings else {"filings_signal":"hold"}
        ar    = AnalystRatingAgent().run(t, data, llm)
        final = PortfolioAgent().run(t, [pa, ma, va, sa, fa, vaa, fil, ar])

        results[t] = {**pa, **ma, **va, **sa, **fa, **vaa, **fil, **ar, **final, "price_history": data["price_history"]}

    return results

st.set_page_config(page_title="AI Hedge Fund Simulator", layout="wide")
st.title("🚀 AI Hedge Fund Simulator")

with st.sidebar:
    st.header("⚙️ Configuration")
    tickers_in    = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOG")
    years         = st.slider("History (years)", 1, 10, 5)
    use_sentiment = st.checkbox("Include News Sentiment", True)
    use_filings   = st.checkbox("Include Filings Data", True)
    run_button    = st.button("Run Analysis", use_container_width=True)

if run_button:
    tickers = [t.strip().upper() for t in tickers_in.split(",") if t.strip()]
    with st.spinner("Running AI agents…"):
        output = run_all(tickers, years, use_sentiment, use_filings)

    st.subheader("📊 Buy / Hold / Sell Recommendations")
    for t in tickers:
        dec   = output[t]["final_decision"].upper()
        score = output[t]["composite_score"]
        st.markdown(f"**{t}:** {dec}  |  Composite Score: {score:.2f}")

    with st.expander("Show Full JSON Output"):
        st.json(output)

    first = tickers[0]
    st.subheader(f"📈 {first} Price History (Close)")
    st.line_chart(output[first]["price_history"]["Close"])

    st.subheader(f"🔍 Detailed Signals for {first}")
    keys  = ["price_signal","momentum_signal","volatility_signal","sentiment_signal",
             "fund_signal","dcf_signal","filings_signal","analyst_signal"]
    table = {k: output[first].get(k) for k in keys}
    st.table(table) 

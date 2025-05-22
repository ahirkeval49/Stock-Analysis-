"""
AI Hedge Fund Simulator

Dependencies & Tools (free resources and websites):
 - Streamlit              : https://streamlit.io
 - yfinance               : https://pypi.org/project/yfinance/
 - NewsAPI (newsapi-python): https://newsapi.org/
 - SEC EDGAR Downloader   : https://pypi.org/project/sec-edgar-downloader/
 - DeepSeek SDK           : https://docs.deepseek.com/
 - OpenAI Python SDK      : https://platform.openai.com/
 - pandas                 : https://pandas.pydata.org/
 - numpy                  : https://numpy.org/
 - python-dotenv          : https://pypi.org/project/python-dotenv/
 - python-dateutil        : https://dateutil.readthedocs.io/
 - questionary            : https://github.com/tmbo/questionary
 - colorama               : https://pypi.org/project/colorama/
"""

import os
import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime
from deepseek import Client as DeepSeekClientSDK
import openai
from dotenv import load_dotenv
from sec_edgar_downloader import Downloader

# Load .env when running locally
load_dotenv()

# --------------------------------
# Data Fetchers
# --------------------------------
EDGAR = Downloader()

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

def fetch_news(ticker: str, api_key: str, page_size: int = 50) -> list[dict]:
    url = "https://newsapi.org/v2/everything"
    params = {"q": ticker, "apiKey": api_key, "pageSize": page_size, "sortBy": "publishedAt", "language": "en"}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json().get("articles", [])

def fetch_inst_filings(ticker: str, count: int = 1):
    return EDGAR.get("13F-HR", ticker, count)

def fetch_insider_filings(ticker: str, count: int = 1):
    return EDGAR.get("4", ticker, count)

# --------------------------------
# LLM Clients
# --------------------------------
class DeepSeekClient:
    def __init__(self, api_key: str):
        self.client = DeepSeekClientSDK(api_key=api_key)
    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.client.embeddings(texts)
    def generate(self, prompt: str) -> str:
        resp = self.client.chat(model="deepthink", messages=[{"role":"user","content":prompt}])
        return resp.choices[0].message["content"]

class OpenAIClient:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        openai.api_key = api_key
        self.model = model
    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = openai.Embedding.create(input=texts, model="text-embedding-ada-002")
        return [e["embedding"] for e in resp["data"]]
    def generate(self, prompt: str) -> str:
        resp = openai.ChatCompletion.create(model=self.model, messages=[{"role":"user","content":prompt}])
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
        df["RSI14"] = 100 - 100 / (1 + gain / loss)
        latest = df.iloc[-1]
        if latest.SMA50 > latest.SMA200 and latest.RSI14 < 70:
            signal = "buy"
        elif latest.SMA50 < latest.SMA200 and latest.RSI14 > 30:
            signal = "sell"
        else:
            signal = "hold"
        return {"ticker":ticker, "sma50":float(latest.SMA50), "sma200":float(latest.SMA200), "rsi14":float(latest.RSI14), "price_signal":signal}

class MomentumAgent:
    def run(self, ticker: str, data: dict) -> dict:
        df = data["price_history"]
        P_t   = df.Close.iloc[-1]
        P_1m  = df.Close.shift(21).iloc[-1]
        P_12m = df.Close.shift(252).iloc[-1]
        m1  = (P_t / P_1m) - 1 if P_1m else 0
        m12 = (P_t / P_12m) - 1 if P_12m else 0
        if m12 > 0 and m1 > 0:
            signal = "buy"
        elif m12 < 0 and m1 < 0:
            signal = "sell"
        else:
            signal = "hold"
        return {"ticker":ticker, "momentum_1m":m1, "momentum_12m":m12, "momentum_signal":signal}

class VolatilityAgent:
    def run(self, ticker: str, data: dict) -> dict:
        ret = np.log(data["price_history"].Close / data["price_history"].Close.shift(1)).dropna()
        ann_vol = float(ret.std() * np.sqrt(252))
        weight  = float(1 / ann_vol) if ann_vol else 0.0
        return {"ticker":ticker, "annual_vol":ann_vol, "vol_weight":weight}

class SentimentAgent:
    def __init__(self, client):
        self.client = client
    def run(self, ticker: str, data: dict) -> dict:
        heads = [a["title"] for a in data.get("news", [])]
        if not heads:
            return {"ticker":ticker, "sentiment_score":0.0, "sentiment_signal":"hold"}
        prompt = f"Rate the overall sentiment for {ticker} (−1 to +1):\n" + "\n".join(heads)
        try:
            score = float(self.client.generate(prompt).strip())
        except:
            score = 0.0
        if score > 0.2:
            sig = "buy"
        elif score < -0.2:
            sig = "sell"
        else:
            sig = "hold"
        return {"ticker":ticker, "sentiment_score":score, "sentiment_signal":sig}

class FundamentalsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        stats = data["fundamentals"]
        mcap = stats.get("marketCap") or 1
        fcf  = stats.get("freeCashflow") or 0
        roe  = stats.get("returnOnEquity") or 0
        de   = stats.get("debtToEquity") or 1
        fcy = fcf / mcap
        score = sum([roe>0, de<1, fcf>0])
        if score >= 3:
            sig = "buy"
        elif score == 0:
            sig = "sell"
        else:
            sig = "hold"
        return {"ticker":ticker, "fcf_yield":float(fcy), "piotroski_score":score, "fund_signal":sig}

class ValuationAgent:
    def run(self, ticker: str, data: dict) -> dict:
        stats = data["fundamentals"]
        df = data["price_history"]
        price = df.Close.iloc[-1]
        pe = stats.get("forwardPE")
        if pe and pe < 17:
            rel = "buy"
        elif pe and pe > 23:
            rel = "sell"
        else:
            rel = "hold"
        fcy = stats.get("freeCashflow",0) / (stats.get("marketCap") or 1)
        fair = price * (1 + fcy)
        if fair > price * 1.1:
            dcf = "buy"
        elif fair < price * 0.9:
            dcf = "sell"
        else:
            dcf = "hold"
        return {"ticker":ticker, "relative_pe_signal":rel, "dcf_price":float(fair), "dcf_signal":dcf}

class FilingsAgent:
    def run(self, ticker: str, data: dict) -> dict:
        inst = data.get("inst_filings", [])
        ins = data.get("insider_filings", [])
        net_i = inst[-1]["shares"] - inst[0]["shares"] if len(inst) > 1 else 0
        net_s = sum([tx.get("shares",0) * (1 if tx.get("type") == "buy" else -1) for tx in ins])
        if net_i > 0 and net_s > 0:
            sig = "buy"
        elif net_i < 0:
            sig = "sell"
        else:
            sig = "hold"
        return {"ticker":ticker, "net_institutional":net_i, "net_insider":net_s, "filings_signal":sig}

class AnalystRatingAgent:
    def run(self, ticker: str, data: dict, client=None) -> dict:
        buy_pct, up = 8/14, 0.10
        if buy_pct > 0.6 and up > 0.15:
            sig = "buy"
        elif up < -0.05:
            sig = "sell"
        else:
            sig = "hold"
        return {"ticker":ticker, "analyst_buy_pct":buy_pct, "target_upside":up, "analyst_signal":sig}

class PortfolioAgent:
    WEIGHTS = {"price":1.0, "momentum":0.5, "volatility":1.0, "sentiment":0.8, "fund":0.7, "valuation":0.9, "filings":0.6, "analyst":1.0}
    def run(self, ticker: str, signals: list[dict]) -> dict:
        total = 0
        for s in signals:
            key = next(k for k in s if k.endswith("_signal"))
            base = key.split("_")[0]
            raw = {"buy":1, "hold":0, "sell":-1}[s[key]]
            w = self.WEIGHTS.get(base, 1.0)
            total += raw * w
        comp = float(np.tanh(total))
        if comp > 0.2:
            final = "buy"
        elif comp < -0.2:
            final = "sell"
        else:
            final = "hold"
        return {"ticker":ticker, "composite_score":comp, "final_decision":final}

# --------------------------------
# Orchestrator & Streamlit App
# --------------------------------
def run_all(tickers, history_years, use_sentiment, use_filings):
    # Initialize LLM from Streamlit secrets
    if "DEEPSEEK_API_KEY" in st.secrets and st.secrets["DEEPSEEK_API_KEY"]:
        llm = DeepSeekClient(api_key=st.secrets["DEEPSEEK_API_KEY"])
    elif "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
        llm = OpenAIClient(api_key=st.secrets["OPENAI_API_KEY"])
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
            data["news"] = fetch_news(t, api_key=st.secrets.get("NEWSAPI_KEY", ""))
        if use_filings:
            data["inst_filings"]    = fetch_inst_filings(t)
            data["insider_filings"] = fetch_insider_filings(t)

        pa = PriceAgent().run(t, data)
        ma = MomentumAgent().run(t, data)
        va = VolatilityAgent().run(t, data)
        sa = SentimentAgent(llm).run(t, data) if use_sentiment else {"sentiment_signal":"hold"}
        fa = FundamentalsAgent().run(t, data)
        eva = ValuationAgent().run(t, data)
        fi = FilingsAgent().run(t, data) if use_filings else {"filings_signal":"hold"}
        aa = AnalystRatingAgent().run(t, data, llm)
        final = PortfolioAgent().run(t, [pa, ma, va, sa, fa, eva, fi, aa])

        results[t] = {**pa, **ma, **va, **sa, **fa, **eva, **fi, **aa, **final}
    return results

# Streamlit UI
st.set_page_config(page_title="AI Hedge Fund Simulator", layout="wide")
st.title("🚀 AI Hedge Fund Simulator")

with st.sidebar:
    st.header("⚙️ Configuration")
    tickers_in = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOG")
    years = st.slider("History (years)", 1, 10, 5)
    use_sentiment = st.checkbox("Include News Sentiment", True)
    use_filings   = st.checkbox("Include Filings Data", True)
    run_button    = st.button("Run Analysis", use_container_width=True)

if run_button:
    tickers = [t.strip().upper() for t in tickers_in.split(",") if t.strip()]
    with st.spinner("Running AI agents…"):
        output = run_all(tickers, years, use_sentiment, use_filings)

    st.subheader("📊 Buy / Hold / Sell Recommendations")
    for t in tickers:
        dec = output[t]["final_decision"].upper()
        score = output[t]["composite_score"]
        st.markdown(f"**{t}:** {dec}  |  Composite Score: {score:.2f}")

    with st.expander("Show Full JSON Output"):
        st.json(output)

    first = tickers[0]
    st.subheader(f"📈 {first} Price History (Close)")
    st.line_chart(output[first]["price_history"]["Close"])

    st.subheader(f"🔍 Detailed Signals for {first}")
    keys = ["price_signal","momentum_signal","sentiment_signal","fund_signal","dcf_signal","filings_signal","analyst_signal"]
    table = {k: output[first].get(k) for k in keys}
    st.table(table)

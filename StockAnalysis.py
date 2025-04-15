# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import random
from pydantic import BaseModel
import datetime
import math

#########################################
# 1. LLM Module (DeepSeek stub)
#########################################
def call_deepseek(prompt: str) -> dict:
    """
    Stub for calling the DeepSeek reasoner model.
    In production, this function would call the DeepSeek API.
    Here we simulate an LLM response with random but plausible outputs.
    """
    signals = ["bullish", "bearish", "neutral"]
    signal = random.choice(signals)
    confidence = round(random.uniform(60, 100), 1)
    reasoning = f"This decision is based on the input: {prompt[:50]}..."
    return {"signal": signal, "confidence": confidence, "reasoning": reasoning}

#########################################
# 2. Data Module (using yfinance)
#########################################
def get_current_price(ticker: str) -> float:
    """
    Fetches the latest closing price for the given ticker.
    """
    try:
        df = yf.download(ticker, period="1d", interval="1d", progress=False)
        if df.empty:
            raise Exception("No data")
        return float(df["Close"].iloc[-1])
    except Exception as e:
        st.error(f"Error fetching price for {ticker}: {e}")
        return 0.0

#########################################
# 3. Agents Module
#########################################
# Define Pydantic models for agent outputs
class FundamentalSignal(BaseModel):
    signal: str
    confidence: float
    reasoning: str

class TechnicalSignal(BaseModel):
    signal: str
    confidence: float
    reasoning: str

class SentimentSignal(BaseModel):
    signal: str
    confidence: float
    reasoning: str

class RiskSignal(BaseModel):
    max_position: int
    reasoning: str

class PortfolioDecision(BaseModel):
    action: str  # buy, sell, or hold
    quantity: int
    confidence: float
    reasoning: str

# Fundamental Agent: Uses yfinance.info to get fundamentals.
def fundamental_agent(ticker: str) -> FundamentalSignal:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
    except Exception as e:
        return FundamentalSignal(signal="neutral", confidence=50.0, reasoning="Failed to fetch fundamental data.")
    
    pe_ratio = info.get("trailingPE")
    if pe_ratio is None:
        reasoning = "No PE ratio found; defaulting to neutral."
        signal = "neutral"
        confidence = 50.0
    else:
        if pe_ratio < 15:
            signal = "bullish"
            confidence = 90.0
            reasoning = f"Low PE ratio of {pe_ratio:.2f} indicates undervaluation."
        elif pe_ratio > 25:
            signal = "bearish"
            confidence = 90.0
            reasoning = f"High PE ratio of {pe_ratio:.2f} indicates overvaluation."
        else:
            signal = "neutral"
            confidence = 60.0
            reasoning = f"PE ratio of {pe_ratio:.2f} lies in a moderate range."
    return FundamentalSignal(signal=signal, confidence=confidence, reasoning=reasoning)

# Technical Agent: Uses moving averages as a simple technical indicator.
def technical_agent(ticker: str, period: str = "1mo") -> TechnicalSignal:
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
    except Exception as e:
        return TechnicalSignal(signal="neutral", confidence=50.0, reasoning="Failed to fetch technical data.")
    
    if df.empty or len(df) < 20:
        return TechnicalSignal(signal="neutral", confidence=50.0, reasoning="Not enough data for technical analysis.")
    
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    latest_sma10 = df['SMA_10'].iloc[-1]
    latest_sma50 = df['SMA_50'].iloc[-1]
    
    if latest_sma10 > latest_sma50:
        signal = "bullish"
        confidence = 80.0
        reasoning = f"10-day SMA ({latest_sma10:.2f}) is above 50-day SMA ({latest_sma50:.2f})."
    elif latest_sma10 < latest_sma50:
        signal = "bearish"
        confidence = 80.0
        reasoning = f"10-day SMA ({latest_sma10:.2f}) is below 50-day SMA ({latest_sma50:.2f})."
    else:
        signal = "neutral"
        confidence = 50.0
        reasoning = f"SMA values are similar at {latest_sma10:.2f}."
    return TechnicalSignal(signal=signal, confidence=confidence, reasoning=reasoning)

# Sentiment Agent: Simulated sentiment based on recent price change.
def sentiment_agent(ticker: str) -> SentimentSignal:
    try:
        df = yf.download(ticker, period="1mo", interval="1d", progress=False)
    except Exception as e:
        return SentimentSignal(signal="neutral", confidence=50.0, reasoning="Failed to fetch sentiment data.")
    
    if df.empty or len(df) < 5:
        return SentimentSignal(signal="neutral", confidence=50.0, reasoning="Not enough data for sentiment analysis.")
    
    price_change = df['Close'].iloc[-1] - df['Close'].iloc[0]
    if price_change > 0:
        signal = "bullish"
        confidence = 75.0
        reasoning = "Price has risen over the past month."
    elif price_change < 0:
        signal = "bearish"
        confidence = 75.0
        reasoning = "Price has fallen over the past month."
    else:
        signal = "neutral"
        confidence = 50.0
        reasoning = "No significant price change observed."
    return SentimentSignal(signal=signal, confidence=confidence, reasoning=reasoning)

# Risk Management Agent: Sets a maximum allowed position size for the stock.
def risk_management_agent(portfolio_cash: float, ticker: str, current_price: float) -> RiskSignal:
    # Allow at most 20% of available cash for one stock.
    max_allowed = int((portfolio_cash * 0.20) / current_price)
    reasoning = f"Based on 20% allocation, max {max_allowed} shares can be held."
    return RiskSignal(max_position=max_allowed, reasoning=reasoning)

# Portfolio Manager Agent: Aggregates agent signals and decides final action.
def portfolio_manager_agent(
    fundamental: FundamentalSignal,
    technical: TechnicalSignal,
    sentiment: SentimentSignal,
    risk: RiskSignal
) -> PortfolioDecision:
    signals = [fundamental.signal, technical.signal, sentiment.signal]
    bullish_count = signals.count("bullish")
    bearish_count = signals.count("bearish")
    avg_conf = np.mean([fundamental.confidence, technical.confidence, sentiment.confidence])
    
    if bullish_count > bearish_count:
        action = "buy"
        quantity = risk.max_position
        reasoning = (
            f"Signals: Fundamental ({fundamental.signal}), Technical ({technical.signal}), Sentiment ({sentiment.signal}). "
            f"Decision: BUY up to {quantity} shares."
        )
    elif bearish_count > bullish_count:
        action = "sell"
        quantity = risk.max_position  # In a real system, would compare against current holdings.
        reasoning = (
            f"Signals: Fundamental ({fundamental.signal}), Technical ({technical.signal}), Sentiment ({sentiment.signal}). "
            f"Decision: SELL the position."
        )
    else:
        action = "hold"
        quantity = 0
        reasoning = (
            f"Mixed signals: Fundamental ({fundamental.signal}), Technical ({technical.signal}), Sentiment ({sentiment.signal}). "
            f"Decision: HOLD."
        )
    
    return PortfolioDecision(action=action, quantity=quantity, confidence=avg_conf, reasoning=reasoning)

#########################################
# 4. Workflow Module
#########################################
def run_workflow(ticker: str, portfolio_cash: float) -> dict:
    """
    For a given ticker and portfolio cash, run the full workflow through all agents.
    Returns a dictionary with all agent outputs and the final decision.
    """
    current_price = get_current_price(ticker)
    if current_price <= 0:
        return {"error": f"Could not retrieve price for {ticker}."}
    
    # Run the individual agents
    fund_signal = fundamental_agent(ticker)
    tech_signal = technical_agent(ticker, period="1mo")
    sent_signal = sentiment_agent(ticker)
    risk_signal = risk_management_agent(portfolio_cash, ticker, current_price)
    
    # Optionally you might call deepseek here by constructing a prompt and calling call_deepseek(...)
    # For this simplified version, we use our own logic.
    
    decision = portfolio_manager_agent(fund_signal, tech_signal, sent_signal, risk_signal)
    
    return {
        "ticker": ticker,
        "current_price": current_price,
        "fundamental": fund_signal.dict(),
        "technical": tech_signal.dict(),
        "sentiment": sent_signal.dict(),
        "risk": risk_signal.dict(),
        "decision": decision.dict()
    }

#########################################
# 5. Web Application (Streamlit)
#########################################
def main():
    st.title("AI-Based Stock Optimizer")
    st.markdown("This web app uses several AI agents to analyze your stocks and optimize your portfolio.")
    
    # User inputs
    ticker_input = st.text_input("Enter stock tickers (comma-separated)", value="AAPL, MSFT, GOOGL")
    cash_input = st.number_input("Enter available portfolio cash ($)", value=100000.0, step=1000.0)
    
    if st.button("Run Analysis"):
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        st.write("### Running analysis on selected tickers...")
        results = []
        for ticker in tickers:
            with st.spinner(f"Analyzing {ticker}..."):
                result = run_workflow(ticker, cash_input)
                results.append(result)
        st.write("### Analysis Results")
        for res in results:
            st.subheader(f"Ticker: {res.get('ticker', 'Unknown')}")
            if "error" in res:
                st.error(res["error"])
            else:
                st.write(f"**Current Price:** ${res['current_price']:.2f}")
                st.write("**Fundamental Analysis:**")
                st.json(res["fundamental"])
                st.write("**Technical Analysis:**")
                st.json(res["technical"])
                st.write("**Sentiment Analysis:**")
                st.json(res["sentiment"])
                st.write("**Risk Assessment:**")
                st.json(res["risk"])
                st.write("**Final Portfolio Decision:**")
                st.json(res["decision"])
                st.markdown("---")
                
if __name__ == "__main__":
    main()

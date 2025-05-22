# Add this at the very top of the code
import streamlit as st

# Then keep the rest of the imports
import os
import yfinance as yf
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime
import openai
from dotenv import load_dotenv# ... (keep previous imports and environment setup)

# --------------------------------
# Enhanced Data Fetchers
# --------------------------------
@st.cache_data(ttl=3600)
def fetch_price_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    try:
        data = yf.Ticker(ticker).history(period=period)
        if data.empty:
            st.warning(f"No price data for {ticker}")
            return pd.DataFrame()
        return data
    except Exception as e:
        st.error(f"Price data error for {ticker}: {str(e)}")
        return pd.DataFrame()

def fetch_fundamentals(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {
            "marketCap": info.get("marketCap"),
            "freeCashflow": info.get("freeCashflow"),
            "sharesOutstanding": info.get("sharesOutstanding"),
            "forwardPE": info.get("forwardPE"),
            "returnOnEquity": info.get("returnOnEquity"),
            "debtToEquity": info.get("debtToEquity"),
            "industry": info.get("industry"),
        }
    except Exception as e:
        st.error(f"Fundamentals error for {ticker}: {str(e)}")
        return {}

# --------------------------------
# Enhanced Valuation Agent
# --------------------------------
class ValuationAgent:
    def run(self, ticker: str, data: dict) -> dict:
        stats = data["fundamentals"]
        df = data["price_history"]
        price = df.Close.iloc[-1] if not df.empty else 0
        
        # Improved DCF Calculation
        fcf = stats.get("freeCashflow", 0)
        shares = stats.get("sharesOutstanding", 1)
        fair_price = 0
        dcf_sig = "hold"
        
        if fcf > 0 and shares > 0:
            try:
                growth = 0.05  # Conservative growth assumption
                terminal_growth = 0.03
                discount_rate = 0.10  # Market risk premium
                years = 5
                
                future_fcf = fcf * (1 + growth) ** years
                terminal_value = future_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
                present_value = terminal_value / (1 + discount_rate) ** years
                fair_value = present_value / shares
                
                margin_of_safety = 0.25  # 25% discount required
                fair_price = fair_value * (1 - margin_of_safety)
                
                if fair_price > price * 1.15:
                    dcf_sig = "buy"
                elif fair_price < price * 0.85:
                    dcf_sig = "sell"
            except:
                pass

        # Industry-adjusted PE ratio
        pe = stats.get("forwardPE", 0)
        industry_pe = {
            "Technology": 25.0,
            "Healthcare": 20.0,
            "Financial": 15.0,
        }.get(stats.get("industry", ""), 18.0)
        
        pe_signal = "hold"
        if pe < industry_pe * 0.7:
            pe_signal = "buy"
        elif pe > industry_pe * 1.3:
            pe_signal = "sell"

        return {
            "ticker": ticker,
            "dcf_price": float(fair_price),
            "dcf_signal": dcf_sig,
            "pe_ratio": float(pe),
            "pe_signal": pe_signal,
        }

# --------------------------------
# Enhanced Sentiment Agent
# --------------------------------
class SentimentAgent:
    def __init__(self, client): self.client = client
    
    def run(self, ticker: str, data: dict) -> dict:
        headlines = [h.get("title","") for h in data.get("news",[])]
        if not headlines:
            return {"ticker":ticker, "sentiment_score":0.0, "sentiment_signal":"hold"}
        
        scores = []
        for headline in headlines[:10]:  # Limit to 10 headlines
            prompt = (f"Analyze this stock-related headline for {ticker}. "
                      f"Respond ONLY with a number between -1 (very negative) "
                      f"and +1 (very positive):\n{headline}")
            try:
                response = self.client.generate(prompt).strip()
                score = float(response)
                scores.append(score)
            except:
                continue
        
        avg_score = np.mean(scores) if scores else 0.0
        sig = "buy" if avg_score > 0.3 else ("sell" if avg_score < -0.3 else "hold")
        
        return {
            "ticker": ticker,
            "sentiment_score": float(avg_score),
            "sentiment_signal": sig,
            "headlines_analyzed": len(scores),
        }

# --------------------------------
# Dynamic Portfolio Agent
# --------------------------------
class PortfolioAgent:
    def get_weights(self, history_years: int) -> dict:
        """Dynamic weights based on investment horizon"""
        if history_years >= 5:  # Long-term focus
            return {
                "valuation": 1.5,
                "fund": 1.2,
                "price": 0.8,
                "momentum": 0.5,
                "sentiment": 0.7,
                "volatility": 1.0,
                "filings": 0.6,
                "analyst": 0.9
            }
        else:  # Short-term focus
            return {
                "momentum": 1.2,
                "price": 1.0,
                "sentiment": 1.0,
                "volatility": 0.8,
                "valuation": 0.7,
                "fund": 0.6,
                "filings": 0.9,
                "analyst": 0.8
            }
    
    def run(self, ticker: str, signals: list[dict], history_years: int) -> dict:
        weights = self.get_weights(history_years)
        total = 0
        max_possible = sum(weights.values())
        
        for s in signals:
            keys = [k for k in s if k.endswith("_signal")]
            if not keys: continue
            key = keys[0]
            base = key.split("_")[0]
            raw = {"buy":1, "hold":0, "sell":-1}.get(s[key], 0)
            total += raw * weights.get(base, 0)
        
        # Normalize score between -1 and 1
        normalized = total / max_possible if max_possible != 0 else 0
        confidence = abs(normalized)
        
        if normalized > 0.25:
            final = ("buy", confidence)
        elif normalized < -0.25:
            final = ("sell", confidence)
        else:
            final = ("hold", confidence)
        
        return {
            "ticker": ticker,
            "composite_score": normalized,
            "final_decision": final[0],
            "confidence": final[1],
        }

# --------------------------------
# Enhanced UI & Visualization
# --------------------------------
def color_decision(val):
    color = 'green' if val == 'buy' else 'red' if val == 'sell' else 'orange'
    return f'background-color: {color}'

def display_results(output):
    st.subheader("📈 Investment Recommendations")
    df = pd.DataFrame([{
        'Ticker': v['ticker'],
        'Decision': v['final_decision'].upper(),
        'Confidence': f"{v['confidence']:.0%}",
        'Price': v['price_history']['Close'].iloc[-1],
        'DCF Fair Value': v.get('dcf_price', 0),
        'P/E Ratio': v.get('pe_ratio', 0),
        'Sentiment': v.get('sentiment_score', 0),
    } for v in output.values()])
    
    # Apply styling
    styled_df = df.style.applymap(color_decision, subset=['Decision'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Detailed analysis expander
    with st.expander("🔍 Detailed Analysis Breakdown"):
        for t in output:
            st.markdown(f"### {t} Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(output[t]["price_history"]["Close"], 
                             use_container_width=True)
            with col2:
                signals = pd.DataFrame({
                    'Signal': [k.replace('_signal', '') for k in output[t] 
                              if '_signal' in k],
                    'Value': [output[t][k] for k in output[t] 
                              if '_signal' in k]
                })
                st.dataframe(signals, hide_index=True)
            
            st.markdown(f"**Valuation:** DCF Fair Value ${output[t].get('dcf_price', 0):.2f} "
                       f"(Current: ${output[t]['price_history']['Close'].iloc[-1]:.2f})")
            st.progress(output[t]['confidence'])

# --------------------------------
# Updated Orchestrator
# --------------------------------
def run_all(tickers, history_years, use_sentiment, use_filings):
    # ... (previous setup code)
    
    for t in tickers:
        # ... (data collection)
        
        # Run agents with error handling
        try:
            final = PortfolioAgent().run(t, [pa, ma, va, sa, fa, vaa, fil, ar], history_years)
        except Exception as e:
            st.error(f"Error processing {t}: {str(e)}")
            continue
        
        results[t] = {**pa, **ma, **va, **sa, **fa, **vaa, **fil, **ar, **final, 
                      "price_history": data["price_history"]}
    
    return results

# Update the main UI section
if run_button:
    # ... (previous validation code)
    
    with st.spinner("Running Enhanced Analysis..."):
        output = run_all(tickers, years, use_sentiment, use_filings)
    
    if output:
        display_results(output)
    else:
        st.error("No valid analysis results returned. Check input parameters.")

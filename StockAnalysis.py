# app.py
import streamlit as st
import robin_stocks as rh
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from openai import OpenAI
from bs4 import BeautifulSoup
import praw
from transformers import pipeline
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns

# Configuration
st.set_page_config(page_title="AI Portfolio Optimizer", layout="wide")

# Initialize components
sentiment_analyzer = pipeline('sentiment-analysis')
reddit = praw.Reddit(client_id=st.secrets["REDDIT"]["CLIENT_ID"],
                     client_secret=st.secrets["REDDIT"]["CLIENT_SECRET"],
                     user_agent='portfolio-analyzer')

# Initialize DeepSeek client
deepseek_client = OpenAI(
    api_key=st.secrets["DEEPSEEK"]["API_KEY"],
    base_url="https://api.deepseek.com"
)

# SEC API Configuration
SEC_API = "https://data.sec.gov/submissions/"

def analyze_with_deepseek(context, question):
    """Use DeepSeek's official API for advanced analysis"""
    system_prompt = """You are a senior financial analyst with 20+ years experience. 
    Analyze the provided portfolio data considering:
    1. Fundamental analysis (P/E ratios, market caps, valuations)
    2. Technical indicators
    3. Market sentiment analysis
    4. Modern portfolio theory principles
    5. Risk/reward ratios
    6. Institutional trading patterns"""
    
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0.3,
            max_tokens=500,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"DeepSeek analysis failed: {str(e)}")
        return "Analysis unavailable"

def get_analysis_context(portfolio, financial_data, sentiment_scores):
    """Prepare structured context for AI analysis"""
    context = f"""
    ## Portfolio Summary
    - Total Value: ${portfolio['Equity'].sum():,.2f}
    - Holdings Count: {len(portfolio)}
    - Top 3 Holdings:
    {portfolio.nlargest(3, 'Weight').to_markdown(index=False)}
    
    ## Fundamental Analysis
    {pd.DataFrame(financial_data).T.to_markdown()}
    
    ## Market Sentiment
    {pd.DataFrame(sentiment_scores.items(), columns=['Ticker', 'Sentiment']).to_markdown(index=False)}
    
    ## Risk Metrics
    - Volatility (1Y):
    {pd.Series({t: prices[t].pct_change().std() for t in portfolio['Ticker']}).to_markdown()}
    """
    return context

# [Previous robinhood_login, get_robinhood_portfolio, 
#  get_13f_data, analyze_sentiment, get_financial_data, 
#  optimize_portfolio functions remain unchanged]

# Streamlit UI
st.title("AI Portfolio Optimizer with DeepSeek Reasoner")

# [Previous Robinhood connection and portfolio display code remains unchanged]

# Main Analysis with DeepSeek Integration
if 'portfolio' in st.session_state:
    portfolio = st.session_state.portfolio
    tickers = portfolio['Ticker'].tolist()
    
    # Data Collection
    with st.spinner("Analyzing portfolio with DeepSeek AI..."):
        financial_data = {t: get_financial_data(t) for t in tickers}
        sentiment_scores = {t: analyze_sentiment(t) for t in tickers}
        prices = pd.DataFrame({t: yf.download(t, period="1y")['Close'] for t in tickers})
        optimal_weights = optimize_portfolio(prices)
        analysis_context = get_analysis_context(portfolio, financial_data, sentiment_scores)
    
    # DeepSeek Analysis Section
    st.subheader("DeepSeek AI Insights")
    
    analysis_tabs = st.tabs(["Portfolio Diagnosis", "Stock Recommendations", "Risk Analysis"])
    
    with analysis_tabs[0]:
        analysis = analyze_with_deepseek(analysis_context, 
            "Analyze portfolio strengths/weaknesses and suggest optimization strategies")
        st.markdown(f"**Portfolio Analysis**\n\n{analysis}")
        
    with analysis_tabs[1]:
        selected_ticker = st.selectbox("Select stock for DeepSeek analysis", tickers)
        analysis = analyze_with_deepseek(analysis_context,
            f"Should we increase or decrease exposure to {selected_ticker}? Consider: "
            f"1. Current valuation 2. Market sentiment 3. Technical indicators 4. Institutional activity")
        st.markdown(f"**{selected_ticker} Analysis**\n\n{analysis}")
        
    with analysis_tabs[2]:
        analysis = analyze_with_deepseek(analysis_context,
            "Identify top 3 portfolio risks and suggest mitigation strategies")
        st.markdown(f"**Risk Assessment**\n\n{analysis}")

    # [Previous visualization and SEC search code remains unchanged]

# Security Disclaimer
st.sidebar.warning("""
**Security Notice:**  
- All API keys stored securely in Streamlit secrets
- Robinhood credentials never stored
- DeepSeek API usage monitored
- Enable 2FA on all connected services
""")

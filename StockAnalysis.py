
import streamlit as st
import robin_stocks as rh
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import openai
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

def robinhood_login():
    try:
        rh.login(
            username=st.secrets["ROBINHOOD"]["USERNAME"],
            password=st.secrets["ROBINHOOD"]["PASSWORD"],
            mfa_code=st.secrets["ROBINHOOD"]["MFA_CODE"]
        )
        return True
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return False

def get_robinhood_portfolio():
    """Fetch and format Robinhood portfolio"""
    holdings = rh.build_holdings()
    portfolio = []
    
    for symbol, data in holdings.items():
        portfolio.append({
            'Ticker': symbol,
            'Quantity': float(data['quantity']),
            'Avg Cost': float(data['average_buy_price']),
            'Price': float(data['price']),
            'Equity': float(data['equity'])
        })
    
    df = pd.DataFrame(portfolio)
    df['Weight'] = df['Equity'] / df['Equity'].sum()
    return df

def get_13f_data(company_name):
    """Retrieve SEC 13F filings data"""
    try:
        response = requests.get(SEC_API + company_name.lower().replace(" ", "-") + ".json")
        return pd.DataFrame(response.json()['filings']['recent'])
    except Exception as e:
        st.error(f"Failed to fetch 13F data: {str(e)}")
        return pd.DataFrame()

def analyze_sentiment(ticker):
    """Analyze Reddit sentiment for a ticker"""
    try:
        posts = reddit.subreddit('stocks').search(ticker, limit=50)
        sentiments = []
        for post in posts:
            sentiment = sentiment_analyzer(post.title)[0]
            sentiments.append(sentiment['score'] if sentiment['label'] == 'POSITIVE' else -sentiment['score'])
        return np.mean(sentiments)
    except Exception as e:
        st.error(f"Sentiment analysis failed: {str(e)}")
        return 0.0

def get_financial_data(ticker):
    """Fetch financial data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")
        return {
            'price': hist['Close'][-1],
            'fair_value': info.get('fairValue', None),
            'pe_ratio': info.get('trailingPE', None),
            'market_cap': info.get('marketCap', None)
        }
    except Exception as e:
        st.error(f"Failed to get data for {ticker}: {str(e)}")
        return {}

def optimize_portfolio(prices):
    """Portfolio optimization using Modern Portfolio Theory"""
    try:
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        return ef.clean_weights()
    except Exception as e:
        st.error(f"Optimization failed: {str(e)}")
        return {}

def analyze_with_deepseek(context, question):
    """Use DeepSeek for advanced financial analysis"""
    system_prompt = """You are a senior financial analyst with expertise in:
    - Portfolio optimization
    - Fundamental analysis
    - Technical analysis
    - Market sentiment evaluation
    - Risk management
    Provide detailed, professional recommendations."""
    
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"DeepSeek analysis failed: {str(e)}")
        return "Analysis unavailable"

def get_analysis_context(portfolio, financial_data, sentiment_scores):
    """Prepare context for AI analysis"""
    return f"""
    Portfolio Summary:
    - Total Value: ${portfolio['Equity'].sum():,.2f}
    - Holdings: {len(portfolio)} stocks
    - Top Holdings: {portfolio.nlargest(3, 'Weight').to_dict()}

    Fundamental Analysis:
    {pd.DataFrame(financial_data).T.to_markdown()}

    Market Sentiment:
    {pd.Series(sentiment_scores).to_markdown()}

    Risk Metrics:
    - Volatility: {portfolio['Equity'].std():.2f}
    - Concentration Risk: {portfolio['Weight'].max()*100:.1f}% in top holding
    """

# Streamlit UI
st.title("AI Portfolio Optimizer with DeepSeek Integration")

# Sidebar Connection
with st.sidebar:
    st.header("Account Setup")
    if st.button("Connect Robinhood"):
        if robinhood_login():
            st.session_state.portfolio = get_robinhood_portfolio()
            st.success("Portfolio loaded!")
        else:
            st.error("Connection failed")

    if 'portfolio' in st.session_state:
        st.subheader("Current Snapshot")
        st.write(f"Total Value: ${st.session_state.portfolio['Equity'].sum():,.2f}")
        st.write(f"Number of Holdings: {len(st.session_state.portfolio)}")

# Main Analysis
if 'portfolio' in st.session_state:
    portfolio = st.session_state.portfolio
    tickers = portfolio['Ticker'].tolist()
    
    # Data Collection
    with st.spinner("Analyzing portfolio..."):
        financial_data = {t: get_financial_data(t) for t in tickers}
        sentiment_scores = {t: analyze_sentiment(t) for t in tickers}
        prices = pd.DataFrame({t: yf.download(t, period="1y")['Close'] for t in tickers})
        optimal_weights = optimize_portfolio(prices)
        analysis_context = get_analysis_context(portfolio, financial_data, sentiment_scores)

    # Display Sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Financial Overview")
        for ticker in tickers:
            data = financial_data[ticker]
            with st.expander(f"{ticker} Analysis"):
                st.metric("Current Price", f"${data['price']:.2f}")
                st.metric("Fair Value", f"${data['fair_value']:.2f}" if data['fair_value'] else "N/A")
                st.write(f"P/E Ratio: {data['pe_ratio']:.2f}" if data['pe_ratio'] else "P/E: N/A")
                st.write(f"Market Cap: ${data['market_cap']/1e9:.2f}B" if data['market_cap'] else "")
                st.write(f"Sentiment Score: {sentiment_scores[ticker]:.2f}")

    with col2:
        st.subheader("Optimization Recommendations")
        for ticker, weight in optimal_weights.items():
            current = portfolio[portfolio['Ticker'] == ticker]['Weight'].values[0]
            delta = weight - current
            
            st.markdown(f"**{ticker}**")
            st.metric("Recommended Allocation", 
                     f"{weight*100:.2f}%", 
                     f"{delta*100:.2f}% {'↑' if delta > 0 else '↓'}")
            st.progress(weight)
            st.markdown("---")

    # DeepSeek Analysis
    st.subheader("AI-Powered Insights")
    analysis_type = st.selectbox("Analysis Type", [
        "Portfolio Health Check",
        "Stock-Specific Recommendation",
        "Risk Assessment"
    ])
    
    if analysis_type == "Portfolio Health Check":
        analysis = analyze_with_deepseek(analysis_context, 
            "Analyze portfolio strengths/weaknesses and provide optimization recommendations")
        st.write(analysis)
    elif analysis_type == "Stock-Specific Recommendation":
        selected = st.selectbox("Select Stock", tickers)
        analysis = analyze_with_deepseek(analysis_context,
            f"Should we increase or decrease exposure to {selected}? Consider valuation and market conditions")
        st.write(analysis)
    else:
        analysis = analyze_with_deepseek(analysis_context,
            "Identify key risks and suggest mitigation strategies")
        st.write(analysis)

    # SEC 13F Search
    st.subheader("Institutional Holdings Research")
    company = st.text_input("Enter company name for 13F filings:")
    if company:
        filings = get_13f_data(company)
        st.dataframe(filings)

    # Visualizations
    st.subheader("Portfolio Analytics")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pie Chart
    ax1.pie(portfolio['Weight'], labels=portfolio['Ticker'], autopct='%1.1f%%')
    ax1.set_title('Current Allocation')
    
    # Price Trends
    for ticker in tickers[:3]:
        ax2.plot(prices[ticker]/prices[ticker].iloc[0], label=ticker)
    ax2.set_title('Normalized Price Performance (1Y)')
    ax2.legend()
    
    st.pyplot(fig)

# Security Notes
st.sidebar.warning("""
**Security Best Practices:**
1. Enable 2FA on all accounts
2. Never share API credentials
3. Use dedicated API keys
4. Monitor account activity
5. Revoke access if compromised
""")

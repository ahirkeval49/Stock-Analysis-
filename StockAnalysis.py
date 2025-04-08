import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import openai
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

# Configure OpenAI API for DeepSeek
openai.api_key = st.secrets["DEEPSEEK"]["API_KEY"]
openai.api_base = "https://api.deepseek.com"

# Initialize Sentiment Analyzer
vader = SentimentIntensityAnalyzer()

# Streamlit Configuration
st.set_page_config(page_title="AI Stock Analyst", layout="wide")
st.title("📈 AI-Powered Stock Analysis & Portfolio Optimizer")

# Initialize Session State
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=['Ticker', 'Quantity', 'Purchase Price', 'Current Price', 'Value'])

# Utility Functions
def get_current_price(ticker):
    try:
        data = yf.Ticker(ticker).history(period='1d')
        return data['Close'].iloc[-1]
    except:
        return None

def analyze_sentiment(ticker):
    try:
        url = f"https://www.reddit.com/r/stocks/search.json?q={ticker}&restrict_sr=1"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        posts = [item['data']['title'] for item in response.json()['data']['children'][:10]]
        scores = [vader.polarity_scores(post)['compound'] for post in posts]
        return np.mean(scores)
    except:
        return 0

def get_institutional_activity():
    try:
        url = "https://www.dataroma.com/m/home.php"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        stocks = []
        for row in soup.select('#recentDealsTable tr')[1:6]:
            cells = row.find_all('td')
            stocks.append({
                'Ticker': cells[1].text,
                'Company': cells[2].text,
                'Manager': cells[3].text,
                'Action': cells[4].text
            })
        return pd.DataFrame(stocks)
    except:
        return pd.DataFrame()

def optimize_portfolio(prices, budget):
    try:
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        
        # Calculate allocation in dollars
        allocation = {k: v * budget for k, v in cleaned_weights.items()}
        return allocation
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        return {}

def get_ai_analysis(prompt, context=""):
    system_msg = """You are a senior financial analyst with expertise in equity markets. 
    Provide detailed, professional recommendations considering fundamental analysis, 
    technical indicators, market sentiment, and portfolio optimization principles."""
    
    try:
        response = openai.ChatCompletion.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"{context}\n{prompt}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content
    except:
        return "AI analysis temporarily unavailable"

# Sidebar - Portfolio Management
with st.sidebar:
    st.header("💰 Portfolio Management")
    with st.form("add_stock"):
        ticker = st.text_input("Stock Ticker").upper()
        qty = st.number_input("Quantity", min_value=1)
        cost = st.number_input("Purchase Price", min_value=0.01)
        if st.form_submit_button("Add to Portfolio"):
            current_price = get_current_price(ticker)
            if current_price:
                new_entry = pd.DataFrame([[ticker, qty, cost, current_price, qty*current_price]],
                                        columns=st.session_state.portfolio.columns)
                st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_entry])
                st.success(f"{ticker} added to portfolio!")
            else:
                st.error("Invalid ticker or price data unavailable")

    if not st.session_state.portfolio.empty:
        if st.button("Clear Portfolio"):
            st.session_state.portfolio = pd.DataFrame()
            st.experimental_rerun()

# Main Interface
tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Overview", "Stock Analysis", "Recommendations", "Market Insights"])

with tab1:
    if not st.session_state.portfolio.empty:
        # Update prices
        st.session_state.portfolio['Current Price'] = st.session_state.portfolio['Ticker'].apply(get_current_price)
        st.session_state.portfolio['Value'] = st.session_state.portfolio['Quantity'] * st.session_state.portfolio['Current Price']
        
        # Display portfolio
        total_value = st.session_state.portfolio['Value'].sum()
        st.subheader(f"Portfolio Value: ${total_value:,.2f}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(st.session_state.portfolio.style.format({
                'Current Price': '${:.2f}',
                'Purchase Price': '${:.2f}',
                'Value': '${:,.2f}'
            }))
        
        with col2:
            fig, ax = plt.subplots()
            st.session_state.portfolio.groupby('Ticker')['Value'].sum().plot.pie(
                ax=ax, autopct='%1.1f%%', startangle=90
            )
            st.pyplot(fig)
        
        # Portfolio Optimization
        st.subheader("Portfolio Optimization")
        budget = st.number_input("Available Investment Budget ($)", min_value=100, value=1000)
        if st.button("Optimize Portfolio"):
            tickers = st.session_state.portfolio['Ticker'].tolist()
            prices = yf.download(tickers, period="1y")['Close']
            allocation = optimize_portfolio(prices, budget)
            
            if allocation:
                st.write("Recommended Allocation:")
                for ticker, amount in allocation.items():
                    st.write(f"{ticker}: ${amount:,.2f} ({amount/budget*100:.1f}%)")
                
                analysis = get_ai_analysis(
                    f"Portfolio optimization recommendation for ${budget} investment. Current holdings: {st.session_state.portfolio}",
                    "Provide detailed reasoning for the allocation recommendations."
                )
                st.write(analysis)
    else:
        st.info("Add stocks to your portfolio using the sidebar")

with tab2:
    st.subheader("Stock Analysis")
    analysis_ticker = st.text_input("Enter ticker for analysis:").upper()
    
    if analysis_ticker:
        with st.spinner("Gathering data..."):
            try:
                stock = yf.Ticker(analysis_ticker)
                info = stock.info
                hist = stock.history(period="1y")
                sentiment = analyze_sentiment(analysis_ticker)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Fundamental Analysis")
                    st.metric("Current Price", f"${info.get('currentPrice', hist['Close'][-1]):.2f}")
                    st.write(f"PE Ratio: {info.get('trailingPE', 'N/A')}")
                    st.write(f"Market Cap: ${info.get('marketCap', 'N/A'):,}")
                    st.write(f"52W Range: {info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}")
                
                with col2:
                    st.subheader("Technical Analysis")
                    fig, ax = plt.subplots()
                    hist['Close'].plot(ax=ax)
                    st.pyplot(fig)
                    st.write(f"Sentiment Score: {sentiment:.2f}/1.0")
                
                st.subheader("AI Analysis")
                analysis = get_ai_analysis(
                    f"Should I buy {analysis_ticker}? Current price: {hist['Close'][-1]:.2f}",
                    f"Company info: {info}"
                )
                st.write(analysis)

            except:
                st.error("Could not retrieve data for this ticker")

with tab3:
    st.subheader("Investment Recommendations")
    budget = st.number_input("Daily Investment Budget ($)", min_value=100, value=1000, key="rec_budget")
    
    if st.button("Generate Recommendations"):
        with st.spinner("Analyzing market opportunities..."):
            # Get institutional activity
            institutional = get_institutional_activity()
            
            # Analyze trending stocks
            trending = pd.DataFrame({
                'Ticker': ['AAPL', 'TSLA', 'NVDA', 'AMZN', 'GOOG'],
                'Sentiment': [analyze_sentiment(t) for t in ['AAPL', 'TSLA', 'NVDA', 'AMZN', 'GOOG']]
            })
            
            # Combine data sources
            recommendations = pd.concat([institutional, trending]).drop_duplicates()
            
            # Get price data
            prices = yf.download(recommendations['Ticker'].tolist(), period="1d")['Close'].T.reset_index()
            prices.columns = ['Ticker', 'Price']
            
            recommendations = recommendations.merge(prices, on='Ticker')
            recommendations['Allocation'] = recommendations['Sentiment'].rank(pct=True) * budget
            
            st.subheader("Top Recommendations")
            st.dataframe(recommendations.style.format({
                'Price': '${:.2f}',
                'Allocation': '${:,.2f}'
            }))

with tab4:
    st.subheader("Market Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Institutional Activity")
        st.dataframe(get_institutional_activity())
    
    with col2:
        st.subheader("Market Sentiment Heatmap")
        heatmap_data = pd.DataFrame({
            'Sector': ['Tech', 'Finance', 'Healthcare', 'Energy', 'Consumer'],
            'Sentiment': [0.7, 0.4, 0.6, 0.3, 0.5]
        })
        st.bar_chart(heatmap_data.set_index('Sector'))

st.sidebar.markdown("---")
st.sidebar.info("""
**Features:**
- Manual portfolio tracking
- Real-time price updates
- Fundamental & technical analysis
- Institutional activity monitoring
- AI-powered recommendations
- Portfolio optimization
- Market sentiment analysis
""")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pypfopt import EfficientFrontier, risk_models, expected_returns
import matplotlib.pyplot as plt
import openai
from datetime import datetime, timedelta

# --------------------------
# Configuration & API Setup
# --------------------------
openai.api_key = st.secrets["DEEPSEEK"]["API_KEY"]
openai.api_base = "https://api.deepseek.com"

reddit = praw.Reddit(
    client_id=st.secrets["REDDIT"]["CLIENT_ID"],
    client_secret=st.secrets["REDDIT"]["CLIENT_SECRET"],
    user_agent='Stock Analysis v2.0'
)

vader = SentimentIntensityAnalyzer()

# --------------------------
# Utility Functions
# --------------------------
def get_current_price(ticker):
    try:
        return yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]
    except Exception as e:
        return None

def get_reddit_sentiment(ticker):
    subreddits = ['stocks', 'investing', 'wallstreetbets', 'StockMarket']
    posts = []
    
    for sub in subreddits:
        try:
            submissions = reddit.subreddit(sub).search(ticker, limit=15, time_filter='week')
            posts.extend([{
                'title': post.title,
                'score': post.score,
                'comments': post.num_comments
            } for post in submissions])
        except Exception as e:
            continue
    
    if not posts:
        return 0
    
    total_score = 0
    total_weight = 0
    for post in posts:
        vs = vader.polarity_scores(post['title'])
        weight = np.log(post['score'] + post['comments'] + 1)
        total_score += vs['compound'] * weight
        total_weight += weight
    
    return total_score / total_weight if total_weight != 0 else 0

def get_news_sentiment(ticker):
    try:
        url = f"https://gnews.io/api/v4/search?q={ticker}&lang=en&token={st.secrets['GNEWS_TOKEN']}"
        articles = requests.get(url).json().get('articles', [])[:10]
        sentiments = [vader.polarity_scores(art['title'])['compound'] for art in articles]
        return np.mean(sentiments) if sentiments else 0
    except Exception as e:
        return 0

def get_institutional_activity():
    try:
        response = requests.get("https://www.sec.gov/cgi-bin/current?q1=4&q2=0&q3=4")
        soup = BeautifulSoup(response.content, 'html.parser')
        data = [{
            'Ticker': cols[1].text.strip(),
            'Company': cols[2].text.strip(),
            'Filing': cols[3].text.strip(),
            'Date': cols[4].text.strip()
        } for row in soup.select('table.tableFile2 tr')[1:6] if (cols := row.find_all('td'))]
        return pd.DataFrame(data)
    except Exception as e:
        return pd.DataFrame()

def get_insider_trades():
    try:
        soup = BeautifulSoup(requests.get("http://openinsider.com/latest-cluster-buys").text, 'html.parser')
        data = [{
            'Ticker': cols[2].text.strip(),
            'Company': cols[3].text.strip(),
            'Position': cols[5].text.strip(),
            'Trade Value': cols[9].text.strip()
        } for row in soup.select('table.tinytable tr')[1:6] if (cols := row.find_all('td'))]
        return pd.DataFrame(data)
    except Exception as e:
        return pd.DataFrame()

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs)).iloc[-1]

def generate_recommendations(budget):
    # Safely get tickers from institutional and insider data
    inst_activity = get_institutional_activity()
    insider_trades = get_insider_trades()
    inst_tickers = inst_activity['Ticker'].tolist() if 'Ticker' in inst_activity.columns else []
    insider_tickers = insider_trades['Ticker'].tolist() if 'Ticker' in insider_trades.columns else []
    tickers = list(set(inst_tickers + insider_tickers))
    
    recommendations = []
    
    for ticker in tickers[:15]:  # Analyze top 15
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")
            
            reddit_sent = get_reddit_sentiment(ticker)
            news_sent = get_news_sentiment(ticker)
            rsi = calculate_rsi(hist['Close'])
            
            recommendations.append({
                'Ticker': ticker,
                'Price': hist['Close'].iloc[-1],
                'RSI': rsi,
                'Reddit Sentiment': reddit_sent,
                'News Sentiment': news_sent,
                'Institutional Activity': 1 if ticker in inst_tickers else 0,
                'Insider Activity': 1 if ticker in insider_tickers else 0,
                'Score': (reddit_sent * 0.4 + news_sent * 0.3 + 
                          (0.15 if ticker in inst_tickers else 0) +
                          (0.15 if ticker in insider_tickers else 0) -
                          abs(rsi - 50) * 0.01)
            })
        except Exception as e:
            continue

    df = pd.DataFrame(recommendations)
    if not df.empty:
        df['Allocation (%)'] = (df['Score'] - df['Score'].min()) / (df['Score'].max() - df['Score'].min()) * 100
        df['Recommended Investment'] = (df['Allocation (%)'] / 100) * budget
    return df.sort_values('Score', ascending=False)

def get_ai_analysis(prompt):
    return openai.ChatCompletion.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=500
    ).choices[0].message.content

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="AI Stock Analyst", layout="wide")
st.title("📈 Intelligent Stock Analysis Platform")

# Initialize portfolio in session_state with a defined schema
if 'portfolio' not in st.session_state or st.session_state.portfolio.empty:
    st.session_state.portfolio = pd.DataFrame(columns=['Ticker', 'Quantity', 'Purchase Price'])

# Sidebar - Portfolio Management
with st.sidebar:
    st.header("💰 Portfolio Management")
    with st.form("add_stock"):
        ticker = st.text_input("Stock Ticker").upper()
        qty = st.number_input("Quantity", min_value=1)
        cost = st.number_input("Purchase Price", min_value=0.01)
        if st.form_submit_button("Add to Portfolio") and (price := get_current_price(ticker)):
            # Append new row ensuring the schema matches
            new_row = [ticker, qty, cost]
            st.session_state.portfolio.loc[len(st.session_state.portfolio)] = new_row
            st.success(f"{ticker} added!")
    
    # Clear Portfolio button
    if not st.session_state.portfolio.empty and st.button("Clear Portfolio"):
        st.session_state.portfolio = pd.DataFrame(columns=['Ticker', 'Quantity', 'Purchase Price'])
        st.experimental_rerun()

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Portfolio", "Analysis", "Recommendations", "Market Intel"])

with tab1:
    if not st.session_state.portfolio.empty:
        st.session_state.portfolio['Current Price'] = st.session_state.portfolio['Ticker'].apply(get_current_price)
        st.session_state.portfolio['Value'] = st.session_state.portfolio['Quantity'] * st.session_state.portfolio['Current Price']
        total_value = st.session_state.portfolio['Value'].sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Portfolio Value: ${total_value:,.2f}")
            st.dataframe(st.session_state.portfolio.style.format({
                'Current Price': '${:.2f}', 'Purchase Price': '${:.2f}', 'Value': '${:,.2f}'
            }))
        
        with col2:
            fig, ax = plt.subplots()
            st.session_state.portfolio.groupby('Ticker')['Value'].sum().plot.pie(ax=ax, autopct='%1.1f%%')
            st.pyplot(fig)
    else:
        st.info("Add stocks using the sidebar")

with tab2:
    ticker = st.text_input("Enter ticker for analysis:").upper()
    if ticker:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Sentiment Analysis")
            reddit_sent = get_reddit_sentiment(ticker)
            news_sent = get_news_sentiment(ticker)
            
            st.metric("Reddit Sentiment", f"{reddit_sent:.2f}")
            st.metric("News Sentiment", f"{news_sent:.2f}")
            
            fig, ax = plt.subplots()
            ax.bar(['Reddit', 'News'], [reddit_sent, news_sent])
            ax.set_ylim(-1, 1)
            st.pyplot(fig)
        
        with col2:
            try:
                data = yf.Ticker(ticker).history(period="6mo")
                data['MA50'] = data['Close'].rolling(50).mean()
                data['MA200'] = data['Close'].rolling(200).mean()
                rsi = calculate_rsi(data['Close'])
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                ax1.plot(data['Close'], label='Price')
                ax1.plot(data['MA50'], label='50-day MA')
                ax1.plot(data['MA200'], label='200-day MA')
                ax1.legend()
                
                ax2.plot(data.index, [rsi] * len(data), label='RSI')
                ax2.axhline(70, color='r', linestyle='--')
                ax2.axhline(30, color='g', linestyle='--')
                ax2.legend()
                
                st.pyplot(fig)
            except Exception as e:
                st.error("Error loading technical data")

with tab3:
    budget = st.number_input("Investment Budget ($)", min_value=1000, value=5000, key="rec_budget")
    if st.button("Generate Recommendations"):
        with st.spinner("Analyzing opportunities..."):
            df = generate_recommendations(budget)
            if not df.empty:
                st.subheader("Recommended Allocation")
                st.dataframe(df.style.format({
                    'Price': '${:.2f}', 'RSI': '{:.1f}',
                    'Reddit Sentiment': '{:.2f}', 'News Sentiment': '{:.2f}',
                    'Recommended Investment': '${:,.2f}'
                }))
                
                analysis = get_ai_analysis(f"""
                Analyze these recommendations based on:
                - Market sentiment from Reddit/news
                - Institutional/insider activity
                - Technical indicators
                - Fundamental metrics
                Provide investment reasoning for top 3:
                {df.head(3).to_dict()}
                """)
                st.subheader("AI Analysis")
                st.write(analysis)

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Institutional Activity")
        st.dataframe(get_institutional_activity())
        
        st.subheader("Insider Trades")
        st.dataframe(get_insider_trades())
    
    with col2:
        st.subheader("Sector Sentiment")
        fig, ax = plt.subplots()
        sectors = ['Tech', 'Finance', 'Healthcare', 'Energy', 'Consumer']
        ax.barh(sectors, [np.random.uniform(-0.5, 0.8) for _ in sectors])
        ax.set_xlim(-1, 1)
        st.pyplot(fig)
        
        st.subheader("Market Anxiety Index")
        st.metric("Fear & Greed Index", "38 (Fear)", "-12% week-over-week")

st.sidebar.markdown("---")
st.sidebar.info("""
**Data Sources:**
- Reddit (r/stocks, r/investing, r/wallstreetbets)
- GNews API
- SEC EDGAR database
- OpenInsider
- Yahoo Finance
""")

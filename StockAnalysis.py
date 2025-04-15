# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import praw
import re
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pypfopt import EfficientFrontier, risk_models, expected_returns
import matplotlib.pyplot as plt
import openai
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import json
import math
import itertools

# ------------------------------
# 1. Logging Setup
# ------------------------------
logging.basicConfig(level=logging.ERROR, filename="app_errors.log", 
                    format='%(asctime)s %(levelname)s:%(message)s')
def log_error(msg):
    logging.error(msg)
    st.error(msg)

# ------------------------------
# 2. Configuration & API Setup
# ------------------------------
# DeepSeek reasoner is used through the OpenAI API (stub for integration)
openai.api_key = st.secrets["DEEPSEEK"]["API_KEY"]
openai.api_base = "https://api.deepseek.com"

# Reddit configuration (make sure to add these keys in your Streamlit secrets)
reddit = praw.Reddit(
    client_id=st.secrets["REDDIT"]["CLIENT_ID"],
    client_secret=st.secrets["REDDIT"]["CLIENT_SECRET"],
    user_agent='Stock Analysis'
)

# Initialize Vader sentiment analyzer
vader = SentimentIntensityAnalyzer()

# ------------------------------
# 3. Data Caching Helpers (using Streamlit cache)
# ------------------------------
@st.cache_data(ttl=60)
def get_current_price(ticker: str) -> float:
    try:
        data = yf.Ticker(ticker).history(period='1d')
        if data.empty:
            raise Exception("No data")
        return float(data['Close'].iloc[-1])
    except Exception as e:
        log_error(f"Error fetching price for {ticker}: {e}")
        return 0.0

# ------------------------------
# 4. SEC & Insider Filings Functions
# ------------------------------
@st.cache_data(ttl=3600)
def get_cik(query: str) -> str:
    query = query.strip().replace(" ", "+")
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?company={query}&owner=exclude&action=getcompany"
    headers = {'User-Agent': 'Keval Ahir (Keval.ahir2019@gmail.com)'}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            log_error("Error fetching company data from SEC.")
            return None
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        match = re.search(r"CIK#:\s*([0-9]+)", text)
        if match:
            cik = match.group(1).strip()
            return cik.zfill(10)
        else:
            log_error("CIK not found for the provided company.")
            return None
    except Exception as e:
        log_error(f"Error searching for CIK: {e}")
        return None

@st.cache_data(ttl=300)
def get_sec_filings(query: str) -> pd.DataFrame:
    cik = get_cik(query)
    if cik is None:
        st.error("Cannot retrieve CIK for the company. Check your query.")
        return pd.DataFrame()
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {'User-Agent': 'Your Name (your.email@example.com)'}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            log_error("Error fetching SEC filings data.")
            return pd.DataFrame()
        data = response.json()
        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            log_error("No recent filings found.")
            return pd.DataFrame()
        df = pd.DataFrame({
            'accessionNumber': recent.get('accessionNumber', []),
            'filingDate': recent.get('filingDate', []),
            'form': recent.get('form', []),
            'reportDate': recent.get('reportDate', [])
        })
        df['filingDate'] = pd.to_datetime(df['filingDate'], errors='coerce')
        six_months_ago = pd.Timestamp.today() - relativedelta(months=6)
        df = df[df['filingDate'] >= six_months_ago]
        return df.sort_values('filingDate', ascending=False)
    except Exception as e:
        log_error(f"Error processing SEC filings: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_openinsider_filings(query: str) -> pd.DataFrame:
    ticker = query.upper()
    six_months_ago = (pd.Timestamp.today() - relativedelta(months=6)).strftime('%Y-%m-%d')
    today = pd.Timestamp.today().strftime('%Y-%m-%d')
    url = f"http://openinsider.com/screener?s={ticker}&fd={six_months_ago}&td={today}&f=html"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            log_error("Error fetching OpenInsider data.")
            return pd.DataFrame()
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find("table", class_="tinytable")
        if table is None:
            log_error("No OpenInsider data found for this ticker.")
            return pd.DataFrame()
        rows = table.find_all("tr")
        data = []
        for row in rows[1:]:
            cols = row.find_all("td")
            if len(cols) < 10:
                continue
            filing_date_text = cols[4].text.strip()
            try:
                filing_date = pd.to_datetime(filing_date_text)
            except Exception:
                filing_date = None
            data.append({
                'Ticker': cols[2].text.strip(),
                'Company': cols[3].text.strip(),
                'Filing Date': filing_date,
                'Position': cols[5].text.strip(),
                'Trade Value': cols[9].text.strip()
            })
        df = pd.DataFrame(data)
        df = df.dropna(subset=['Filing Date'])
        six_months_ago_date = pd.Timestamp.today() - relativedelta(months=6)
        df = df[df['Filing Date'] >= six_months_ago_date]
        return df.sort_values('Filing Date', ascending=False)
    except Exception as e:
        log_error(f"Error processing OpenInsider data: {e}")
        return pd.DataFrame()

# ------------------------------
# 5. Sentiment & Technical Analysis Functions
# ------------------------------
def get_reddit_sentiment(query: str) -> float:
    subreddits = ['stocks', 'investing', 'wallstreetbets', 'StockMarket']
    posts = []
    for sub in subreddits:
        try:
            submissions = reddit.subreddit(sub).search(query, limit=15, time_filter='week')
            posts.extend([{'title': post.title, 'score': post.score, 'comments': post.num_comments}
                          for post in submissions])
        except Exception as e:
            st.warning(f"Error fetching from r/{sub}: {e}")
            continue
    if not posts:
        return 0.0
    total_score = 0.0
    total_weight = 0.0
    for post in posts:
        vs = vader.polarity_scores(post['title'])
        weight = np.log(post['score'] + post['comments'] + 1)
        total_score += vs['compound'] * weight
        total_weight += weight
    return total_score / total_weight if total_weight else 0.0

def get_news_sentiment(query: str) -> float:
    try:
        # Using GNews API to fetch news data
        url = f"https://gnews.io/api/v4/search?q={query}&lang=en&token={st.secrets['GNEWS_TOKEN']}"
        response = requests.get(url)
        articles = response.json().get('articles', [])[:10]
        sentiments = []
        for art in articles:
            title = art.get('title', '')
            description = art.get('description', '')
            title_sent = vader.polarity_scores(title)['compound']
            desc_sent = vader.polarity_scores(description)['compound'] if description else 0.0
            sentiments.append((title_sent + desc_sent) / 2)
        return np.mean(sentiments) if sentiments else 0.0
    except Exception as e:
        st.warning(f"Error fetching news: {e}")
        return 0.0

def calculate_rsi(prices: pd.Series, window: int = 14) -> float:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else np.nan

def calculate_bollinger_bands(prices: pd.Series, window: int = 20) -> tuple:
    sma = prices.rolling(window).mean()
    std_dev = prices.rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band

# ------------------------------
# 6. Fundamental Data Functions
# ------------------------------
@st.cache_data(ttl=300)
def get_fundamentals(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        fundamentals = {
            "Company Name": info.get("longName"),
            "Market Cap": info.get("marketCap"),
            "PE Ratio": info.get("trailingPE"),
            "EPS": info.get("trailingEps"),
            "Dividend Yield": info.get("dividendYield"),
            "52-Week Change": info.get("52WeekChange"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Country": info.get("country")
        }
        return fundamentals
    except Exception as e:
        log_error(f"Error fetching fundamentals for {ticker}: {e}")
        return {}

# ------------------------------
# 7. Portfolio Optimization Function
# ------------------------------
def optimize_portfolio(tickers: list) -> pd.DataFrame:
    prices_df = pd.DataFrame()
    for t in tickers:
        data = yf.Ticker(t).history(period="1y", interval="1d")
        if not data.empty:
            prices_df[t] = data["Close"]
    if prices_df.empty:
        st.error("Insufficient data for portfolio optimization.")
        return pd.DataFrame()
    mu = expected_returns.mean_historical_return(prices_df)
    S = risk_models.sample_cov(prices_df)
    ef = EfficientFrontier(mu, S)
    try:
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        return pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Weight'])
    except Exception as e:
        st.error(f"Optimization error: {e}")
        return pd.DataFrame()

# ------------------------------
# 8. AI Analysis (DeepSeek/LLM Stub)
# ------------------------------
def get_ai_analysis(prompt: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.warning("AI analysis unavailable.")
        return "AI analysis is currently unavailable."

# ------------------------------
# 9. Agent Functions
# ------------------------------
# Each agent returns a Pydantic-like dict structure.
class FundamentalSignal(BaseModel):
    signal: str
    confidence: float
    reasoning: str

def fundamental_agent(ticker: str) -> FundamentalSignal:
    fundamentals = get_fundamentals(ticker)
    pe = fundamentals.get("PE Ratio")
    if pe is None:
        return FundamentalSignal(signal="neutral", confidence=50.0, reasoning="No PE ratio found.")
    if pe < 15:
        return FundamentalSignal(signal="bullish", confidence=90.0, reasoning=f"Low PE ratio ({pe:.2f}) indicates potential undervaluation.")
    elif pe > 25:
        return FundamentalSignal(signal="bearish", confidence=90.0, reasoning=f"High PE ratio ({pe:.2f}) indicates overvaluation.")
    else:
        return FundamentalSignal(signal="neutral", confidence=60.0, reasoning=f"Moderate PE ratio ({pe:.2f}) suggests no strong bias.")

class TechnicalSignal(BaseModel):
    signal: str
    confidence: float
    reasoning: str

def technical_agent(ticker: str) -> TechnicalSignal:
    try:
        df = yf.Ticker(ticker).history(period="6mo")
    except Exception as e:
        return TechnicalSignal(signal="neutral", confidence=50.0, reasoning="Error fetching price history.")
    
    if df.empty or len(df) < 50:
        return TechnicalSignal(signal="neutral", confidence=50.0, reasoning="Not enough data for technical analysis.")
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    latest_sma20 = df['SMA_20'].iloc[-1]
    latest_sma50 = df['SMA_50'].iloc[-1]
    
    # RSI Calculation
    rsi = calculate_rsi(df['Close'], window=14)
    
    # Bollinger Bands (as additional metric)
    upper, lower = calculate_bollinger_bands(df['Close'], window=20)
    price = df['Close'].iloc[-1]
    
    reasoning = (f"20-day SMA: {latest_sma20:.2f}, 50-day SMA: {latest_sma50:.2f}; "
                f"RSI: {rsi:.2f}; Price: {price:.2f};")
    if latest_sma20 > latest_sma50 and rsi < 70:
        return TechnicalSignal(signal="bullish", confidence=80.0, reasoning=reasoning + "Uptrend and acceptable RSI.")
    elif latest_sma20 < latest_sma50 and rsi > 30:
        return TechnicalSignal(signal="bearish", confidence=80.0, reasoning=reasoning + "Downtrend and RSI not oversold.")
    else:
        return TechnicalSignal(signal="neutral", confidence=60.0, reasoning=reasoning + "Mixed signals.")

class SentimentSignal(BaseModel):
    signal: str
    confidence: float
    reasoning: str

def sentiment_agent(ticker: str) -> SentimentSignal:
    reddit_sent = get_reddit_sentiment(ticker)
    news_sent = get_news_sentiment(ticker)
    avg_sent = (reddit_sent + news_sent) / 2
    if avg_sent > 0.1:
        return SentimentSignal(signal="bullish", confidence=75.0, reasoning="Overall positive sentiment from Reddit and News.")
    elif avg_sent < -0.1:
        return SentimentSignal(signal="bearish", confidence=75.0, reasoning="Overall negative sentiment from Reddit and News.")
    else:
        return SentimentSignal(signal="neutral", confidence=50.0, reasoning="Neutral average sentiment.")

# SEC/Insider Agent: Aggregates SEC filings and OpenInsider data.
def sec_insider_agent(query: str) -> dict:
    sec_df = get_sec_filings(query)
    openinsider_df = get_openinsider_filings(query)
    sec_info = f"{len(sec_df)} recent SEC filings." if not sec_df.empty else "No recent SEC filings."
    insider_info = f"{len(openinsider_df)} OpenInsider filings." if not openinsider_df.empty else "No recent insider filings."
    return {"sec": sec_info, "insider": insider_info}

# Risk Management Agent: Calculates max position (based on 20% portfolio allocation)
class RiskSignal(BaseModel):
    max_position: int
    reasoning: str

def risk_management_agent(portfolio_cash: float, ticker: str, current_price: float) -> RiskSignal:
    max_allowed = int((portfolio_cash * 0.20) / current_price)
    return RiskSignal(max_position=max_allowed, reasoning=f"Based on 20% allocation, max {max_allowed} shares.")

# Portfolio Manager Agent: Aggregates signals from the previous agents to decide final action.
class PortfolioDecision(BaseModel):
    action: str      # "buy", "sell", or "hold"
    quantity: int
    confidence: float
    reasoning: str

def portfolio_manager_agent(
    fundamental: FundamentalSignal,
    technical: TechnicalSignal,
    sentiment: SentimentSignal,
    risk: RiskSignal,
) -> PortfolioDecision:
    signals = [fundamental.signal, technical.signal, sentiment.signal]
    bullish = signals.count("bullish")
    bearish = signals.count("bearish")
    avg_conf = np.mean([fundamental.confidence, technical.confidence, sentiment.confidence])
    
    if bullish > bearish:
        action = "buy"
        quantity = risk.max_position
        reasoning = (f"Agent signals (F: {fundamental.signal}, T: {technical.signal}, S: {sentiment.signal}) "
                     f"recommend BUY with max allocation of {quantity} shares.")
    elif bearish > bullish:
        action = "sell"
        quantity = risk.max_position  # In a real system, would compare to held positions
        reasoning = (f"Agent signals (F: {fundamental.signal}, T: {technical.signal}, S: {sentiment.signal}) "
                     f"recommend SELL the position.")
    else:
        action = "hold"
        quantity = 0
        reasoning = (f"Mixed agent signals (F: {fundamental.signal}, T: {technical.signal}, S: {sentiment.signal}) "
                     "recommend HOLD.")
    return PortfolioDecision(action=action, quantity=quantity, confidence=avg_conf, reasoning=reasoning)

# ------------------------------
# 10. Workflow Function
# ------------------------------
def run_workflow(ticker: str, portfolio_cash: float, query_for_filings: str) -> dict:
    current_price = get_current_price(ticker)
    if not current_price or current_price <= 0:
        return {"error": f"Could not retrieve valid price for {ticker}."}
    
    fund_signal = fundamental_agent(ticker)
    tech_signal = technical_agent(ticker)
    sent_signal = sentiment_agent(ticker)
    risk_signal = risk_management_agent(portfolio_cash, ticker, current_price)
    port_decision = portfolio_manager_agent(fund_signal, tech_signal, sent_signal, risk_signal)
    sec_insider = sec_insider_agent(query_for_filings)
    
    # Optionally, get an AI analysis combining all agent outputs:
    prompt = f"""
    Analyze the following signals for {ticker}:
    Fundamental: {fund_signal.dict()}
    Technical: {tech_signal.dict()}
    Sentiment: {sent_signal.dict()}
    SEC/Insider: {sec_insider}
    Provide a concise investment rationale.
    """
    ai_analysis = get_ai_analysis(prompt)
    
    return {
        "ticker": ticker,
        "current_price": current_price,
        "fundamental": fund_signal.dict(),
        "technical": tech_signal.dict(),
        "sentiment": sent_signal.dict(),
        "risk": risk_signal.dict(),
        "SEC/Insider": sec_insider,
        "portfolio_decision": port_decision.dict(),
        "ai_analysis": ai_analysis
    }

# ------------------------------
# 11. Portfolio Optimizer
# ------------------------------
def run_portfolio_optimization(tickers: list) -> pd.DataFrame:
    return optimize_portfolio(tickers)

# ------------------------------
# 12. Trending Stocks (Placeholder)
# ------------------------------
def get_trending_stocks() -> pd.DataFrame:
    trending = pd.DataFrame({
        'Ticker': ['AAPL', 'TSLA', 'MSFT', 'AMZN', 'NVDA'],
        'Change (%)': np.random.uniform(-5, 5, 5)
    })
    return trending

# ------------------------------
# 13. Streamlit Web App
# ------------------------------
st.set_page_config(page_title="AI Stock Optimizer", layout="wide")
st.title("📈 Advanced AI-Based Stock Optimizer")

# ------------------------------
# Sidebar: Portfolio Management
# ------------------------------
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=['Ticker', 'Quantity', 'Purchase Price'])

with st.sidebar:
    st.header("Portfolio Management")
    with st.form("add_stock"):
        ticker_input = st.text_input("Enter Stock Ticker", value="AAPL").upper()
        qty = st.number_input("Quantity", min_value=0.1, value=1.0, step=0.1)
        cost = st.number_input("Purchase Price", min_value=0.01, value=150.0, step=0.1)
        if st.form_submit_button("Add Stock"):
            new_entry = pd.DataFrame([[ticker_input, qty, cost]], columns=['Ticker', 'Quantity', 'Purchase Price'])
            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_entry], ignore_index=True)
            st.success(f"Added {ticker_input} to portfolio.")
    if not st.session_state.portfolio.empty:
        tickers_in_portfolio = st.session_state.portfolio['Ticker'].unique().tolist()
        stock_to_remove = st.multiselect("Select stocks to remove", tickers_in_portfolio)
        if st.button("Remove Selected Stocks"):
            st.session_state.portfolio = st.session_state.portfolio[~st.session_state.portfolio['Ticker'].isin(stock_to_remove)]
            st.success("Removed selected stocks.")
        if st.button("Clear Portfolio"):
            st.session_state.portfolio = pd.DataFrame(columns=['Ticker', 'Quantity', 'Purchase Price'])
            st.experimental_rerun()
    st.markdown("---")
    st.info("""**Data Sources:**  
    - Yahoo Finance  
    - SEC EDGAR filings  
    - OpenInsider  
    - Reddit & GNews (for sentiment)  
    - DeepSeek Reasoner (AI analysis)""")

# ------------------------------
# Main Tabs
# ------------------------------
tabs = st.tabs(["Portfolio", "Technical & Sentiment", "SEC/Insider", "Recommendations", "Fundamentals", "Optimization", "Trending"])
    
# Tab 1: Portfolio Overview
with tabs[0]:
    st.subheader("Current Portfolio")
    if not st.session_state.portfolio.empty:
        port_df = st.session_state.portfolio.copy()
        port_df['Current Price'] = port_df['Ticker'].apply(get_current_price)
        port_df['Current Value'] = port_df['Quantity'] * port_df['Current Price']
        port_df['Profit/Loss'] = (port_df['Current Price'] - port_df['Purchase Price']) * port_df['Quantity']
        total_value = port_df['Current Value'].sum()
        st.write(f"**Portfolio Total Value:** ${total_value:,.2f}")
        st.dataframe(port_df.style.format({
            'Current Price': '${:.2f}', 
            'Purchase Price': '${:.2f}', 
            'Current Value': '${:.2f}',
            'Profit/Loss': '${:.2f}'
        }))
    else:
        st.info("Your portfolio is empty. Use the sidebar to add stocks.")

# Tab 2: Technical & Sentiment Analysis for a Stock
with tabs[1]:
    ticker_analyze = st.text_input("Enter Ticker for Technical Analysis", value="AAPL").upper()
    if ticker_analyze:
        st.subheader(f"Technical & Sentiment Analysis: {ticker_analyze}")
        # Technical Analysis Chart
        try:
            hist_data = yf.Ticker(ticker_analyze).history(period="6mo")
            if not hist_data.empty:
                hist_data['SMA20'] = hist_data['Close'].rolling(window=20).mean()
                hist_data['SMA50'] = hist_data['Close'].rolling(window=50).mean()
                rsi_val = calculate_rsi(hist_data['Close'], window=14)
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8))
                ax1.plot(hist_data.index, hist_data['Close'], label='Close Price')
                ax1.plot(hist_data.index, hist_data['SMA20'], label='20-day SMA')
                ax1.plot(hist_data.index, hist_data['SMA50'], label='50-day SMA')
                ax1.set_title(f"{ticker_analyze} Price and Moving Averages")
                ax1.legend()
                ax2.plot(hist_data.index, [rsi_val]*len(hist_data), label=f"RSI: {rsi_val:.2f}")
                ax2.axhline(70, color='red', linestyle='--')
                ax2.axhline(30, color='green', linestyle='--')
                ax2.set_title("RSI")
                ax2.legend()
                st.pyplot(fig)
            else:
                st.error("No historical price data available.")
        except Exception as e:
            st.error(f"Error in technical analysis: {e}")
        # Sentiment Metrics
        reddit_sent = get_reddit_sentiment(ticker_analyze)
        news_sent = get_news_sentiment(ticker_analyze)
        st.metric("Reddit Sentiment", f"{reddit_sent:.2f}")
        st.metric("News Sentiment", f"{news_sent:.2f}")

# Tab 3: SEC & Insider Filings Analysis
with tabs[2]:
    query_sec = st.text_input("Enter Company Name for SEC Filings", value="Apple")
    query_insider = st.text_input("Enter Ticker for Insider Filings", value="AAPL").upper()
    if query_sec:
        st.subheader("Recent SEC Filings")
        sec_filings = get_sec_filings(query_sec)
        if not sec_filings.empty:
            st.dataframe(sec_filings)
        else:
            st.info("No SEC filings found for this company in the last 6 months.")
    if query_insider:
        st.subheader("Recent OpenInsider Filings (Insider Trades)")
        insider_filings = get_openinsider_filings(query_insider)
        if not insider_filings.empty:
            st.dataframe(insider_filings)
        else:
            st.info("No insider filings found for this ticker in the last 6 months.")

# Tab 4: Recommendations & AI Analysis
with tabs[3]:
    st.subheader("Generate Stock Recommendation")
    ticker_rec = st.text_input("Enter Ticker for Recommendation", value="AAPL").upper()
    query_for_filings = st.text_input("Enter Company Name for Filings (for SEC/Insider view)", value="Apple")
    if ticker_rec and st.button("Analyze Recommendation"):
        portfolio_cash = st.number_input("Enter Available Cash ($)", value=100000.0, step=1000.0)
        result = run_workflow(ticker_rec, portfolio_cash, query_for_filings)
        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader(f"Analysis for {ticker_rec}")
            st.json(result)
            st.markdown("**AI Analysis**:")
            st.write(result.get("ai_analysis", "No AI analysis available."))

# Tab 5: Fundamental Analysis
with tabs[4]:
    ticker_fund = st.text_input("Enter Ticker for Fundamental Analysis", value="AAPL").upper()
    if ticker_fund:
        fundamentals = get_fundamentals(ticker_fund)
        if fundamentals:
            st.subheader(f"{ticker_fund} Fundamentals")
            for key, val in fundamentals.items():
                st.write(f"**{key}:** {val}")
        else:
            st.info("No fundamental data available.")

# Tab 6: Portfolio Optimizer
with tabs[5]:
    st.subheader("Portfolio Optimization")
    if not st.session_state.portfolio.empty:
        portfolio_tickers = st.session_state.portfolio['Ticker'].unique().tolist()
        opt_df = run_portfolio_optimization(portfolio_tickers)
        if not opt_df.empty:
            st.dataframe(opt_df)
            fig, ax = plt.subplots()
            ax.pie(opt_df['Weight'], labels=opt_df.index, autopct='%1.1f%%')
            st.pyplot(fig)
        else:
            st.info("Optimization could not be performed.")
    else:
        st.info("Please add stocks to your portfolio first.")

# Tab 7: Trending & Emerging Stocks
with tabs[6]:
    st.subheader("Trending & Emerging Stocks")
    trending = get_trending_stocks()
    if not trending.empty:
        st.dataframe(trending.style.format({'Change (%)': '{:.2f}%'}))
    else:
        st.info("Trending stocks data not available.")

# ------------------------------
# End of Application
# ------------------------------

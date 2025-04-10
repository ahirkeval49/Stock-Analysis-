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

# ------------------------------
# Logging Setup
# ------------------------------
logging.basicConfig(level=logging.ERROR, filename="app_errors.log", 
                    format='%(asctime)s %(levelname)s:%(message)s')
def log_error(msg):
    logging.error(msg)
    # Optionally: st.error(msg)

# ------------------------------
# 1. Configuration & API Setup
# ------------------------------
openai.api_key = st.secrets["DEEPSEEK"]["API_KEY"]
openai.api_base = "https://api.deepseek.com"

reddit = praw.Reddit(
    client_id=st.secrets["REDDIT"]["CLIENT_ID"],
    client_secret=st.secrets["REDDIT"]["CLIENT_SECRET"],
    user_agent='Stock Analysis v2.0'
)

vader = SentimentIntensityAnalyzer()

# ------------------------------
# 2. Caching Helpers
# ------------------------------
@st.cache_data(ttl=60)
def get_current_price(ticker):
    try:
        data = yf.Ticker(ticker).history(period='1d')
        return data['Close'].iloc[-1]
    except Exception as e:
        log_error(f"Error fetching price for {ticker}: {e}")
        st.error(f"Error fetching price for {ticker}.")
        return None

@st.cache_data(ttl=3600)
def get_cik(query):
    query = query.strip().replace(" ", "+")
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?company={query}&owner=exclude&action=getcompany"
    headers = {'User-Agent': 'Keval Ahir (keval.ahir2019@gmail.com)'}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            log_error("Error fetching company data from SEC website.")
            st.error("Error fetching company data from SEC website.")
            return None
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        match = re.search(r"CIK#:\s*([0-9]+)", text)
        if match:
            cik = match.group(1).strip()
            return cik.zfill(10)
        else:
            log_error("CIK not found for the provided company.")
            st.error("CIK not found for the provided company.")
            return None
    except Exception as e:
        log_error(f"Error searching for CIK: {e}")
        st.error("Error searching for CIK.")
        return None

# ------------------------------
# 3. SEC Filings Function
# ------------------------------
@st.cache_data(ttl=300)
def get_sec_filings(query):
    cik = get_cik(query)
    if cik is None:
        st.error("Cannot retrieve CIK for the company. Please check your query.")
        return pd.DataFrame()
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {'User-Agent': 'Keval Ahir (keval.ahir2019@gmail.com)'}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            log_error("Error fetching SEC filings data from the SEC website.")
            st.error("Error fetching SEC filings data.")
            return pd.DataFrame()
        data = response.json()
        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            log_error("No recent filings found.")
            st.info("No recent SEC filings found for this company.")
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
        st.error("Error processing SEC filings data.")
        return pd.DataFrame()

# ------------------------------
# 4. OpenInsider Filings Function
# ------------------------------
@st.cache_data(ttl=300)
def get_openinsider_filings(query):
    ticker = query.upper()
    six_months_ago = (pd.Timestamp.today() - relativedelta(months=6)).strftime('%Y-%m-%d')
    today = pd.Timestamp.today().strftime('%Y-%m-%d')
    url = f"http://openinsider.com/screener?s={ticker}&fd={six_months_ago}&td={today}&f=html"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            log_error("Error fetching OpenInsider data.")
            st.error("Error fetching OpenInsider filings.")
            return pd.DataFrame()
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find("table", class_="tinytable")
        if table is None:
            log_error("No OpenInsider data found for this ticker.")
            st.info("No OpenInsider filings found for this ticker.")
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
        st.error("Error processing OpenInsider data.")
        return pd.DataFrame()

# ------------------------------
# 5. Sentiment & Technical Analysis Functions
# ------------------------------
def get_reddit_sentiment(query):
    subreddits = ['stocks', 'investing', 'wallstreetbets', 'StockMarket']
    posts = []
    for sub in subreddits:
        try:
            submissions = reddit.subreddit(sub).search(query, limit=15, time_filter='week')
            posts.extend([{'title': post.title, 'score': post.score, 'comments': post.num_comments}
                          for post in submissions])
        except Exception as e:
            st.warning(f"Error fetching data from r/{sub}: {e}")
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

def get_news_sentiment(query):
    try:
        url = f"https://gnews.io/api/v4/search?q={query}&lang=en&token={st.secrets['GNEWS_TOKEN']}"
        response = requests.get(url)
        articles = response.json().get('articles', [])[:10]
        sentiments = []
        for art in articles:
            title = art.get('title', '')
            description = art.get('description', '')
            title_sent = vader.polarity_scores(title)['compound']
            description_sent = vader.polarity_scores(description)['compound'] if description else 0
            combined_sent = (title_sent + description_sent) / 2
            sentiments.append(combined_sent)
        return np.mean(sentiments) if sentiments else 0
    except Exception as e:
        st.warning(f"Error fetching news data: {e}")
        return 0

def get_market_scenario():
    try:
        query = "stock market"
        url = f"https://gnews.io/api/v4/search?q={query}&lang=en&token={st.secrets['GNEWS_TOKEN']}"
        response = requests.get(url)
        articles = response.json().get('articles', [])[:10]
        sentiments = []
        for art in articles:
            title = art.get('title', '')
            description = art.get('description', '')
            title_sent = vader.polarity_scores(title)['compound']
            description_sent = vader.polarity_scores(description)['compound'] if description else 0
            combined_sent = (title_sent + description_sent) / 2
            sentiments.append(combined_sent)
        return np.mean(sentiments) if sentiments else 0
    except Exception as e:
        st.warning(f"Error fetching market scenario data: {e}")
        return 0

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else np.nan

# ------------------------------
# 6. Recommendation Engine
# ------------------------------
def generate_recommendations(budget):
    # Here for demonstration, we're using fixed queries.
    inst_df = get_sec_filings("Apple")  # Replace with your preferred method.
    insider_df = get_openinsider_filings("AAPL")
    inst_tickers = inst_df['accessionNumber'].tolist() if 'accessionNumber' in inst_df.columns else []
    insider_tickers = insider_df['Ticker'].tolist() if 'Ticker' in insider_df.columns else []
    tickers = list(set(inst_tickers + insider_tickers))
    
    recommendations = []
    for ticker in tickers[:15]:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")
            if hist.empty:
                continue
            reddit_sent = get_reddit_sentiment(ticker)
            news_sent = get_news_sentiment(ticker)
            rsi = calculate_rsi(hist['Close'])
            score = (reddit_sent * 0.4 + news_sent * 0.3 +
                     (0.15 if ticker in inst_tickers else 0) +
                     (0.15 if ticker in insider_tickers else 0) -
                     abs(rsi - 50) * 0.01)
            recommendations.append({
                'Ticker': ticker,
                'Price': hist['Close'].iloc[-1],
                'RSI': rsi,
                'Reddit Sentiment': reddit_sent,
                'News Sentiment': news_sent,
                'Institutional Activity': 1 if ticker in inst_tickers else 0,
                'Insider Activity': 1 if ticker in insider_tickers else 0,
                'Score': score
            })
        except Exception:
            continue
    df = pd.DataFrame(recommendations)
    if not df.empty and 'Score' in df.columns:
        df['Allocation (%)'] = (df['Score'] - df['Score'].min()) / (df['Score'].max() - df['Score'].min()) * 100
        df['Recommended Investment'] = (df['Allocation (%)'] / 100) * budget
        df = df.sort_values('Score', ascending=False)
    return df

# ------------------------------
# 7. AI Analysis Function
# ------------------------------
def get_ai_analysis(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception:
        return "AI analysis is currently unavailable."

# ------------------------------
# 8. Fundamental Analysis Function
# ------------------------------
@st.cache_data(ttl=300)
def get_fundamentals(ticker):
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
    except Exception:
        return {}

# ------------------------------
# 9. Trending Stocks Function (Placeholder)
# ------------------------------
def get_trending_stocks():
    trending = pd.DataFrame({
        'Ticker': ['AAPL', 'TSLA', 'MSFT', 'AMZN', 'NVDA'],
        'Change (%)': np.random.uniform(0, 5, 5)
    })
    return trending

# ------------------------------
# 10. Streamlit UI Setup & Page Configuration
# ------------------------------
st.set_page_config(page_title="AI Stock Analyst", layout="wide")
st.title("📈 Intelligent Stock Analysis Platform")

# ------------------------------
# 11. Initialize Portfolio in Session State
# ------------------------------
if 'portfolio' not in st.session_state or st.session_state.portfolio.empty:
    st.session_state.portfolio = pd.DataFrame(columns=['Ticker', 'Quantity', 'Purchase Price'])

# ------------------------------
# 12. Sidebar: Portfolio Management (Add/Delete/Edit)
# ------------------------------
with st.sidebar:
    st.header("💰 Portfolio Management")
    with st.form("add_stock"):
        ticker_input_sidebar = st.text_input("Stock Ticker").upper()
        qty = st.number_input("Quantity", min_value=0.01, value=1.0, step=0.01)
        cost = st.number_input("Purchase Price", min_value=0.01, value=1.0, step=0.01)
        if st.form_submit_button("Add to Portfolio") and (price := get_current_price(ticker_input_sidebar)):
            base_portfolio = st.session_state.portfolio[['Ticker', 'Quantity', 'Purchase Price']]
            new_entry = pd.DataFrame([[ticker_input_sidebar, qty, cost]], columns=base_portfolio.columns)
            st.session_state.portfolio = pd.concat([base_portfolio, new_entry], ignore_index=True)
            st.success(f"{ticker_input_sidebar} added!")
    if not st.session_state.portfolio.empty:
        tickers_in_portfolio = st.session_state.portfolio['Ticker'].unique().tolist()
        stock_to_remove = st.multiselect("Select stocks to remove", tickers_in_portfolio)
        if st.button("Remove Selected Stocks"):
            st.session_state.portfolio = st.session_state.portfolio[~st.session_state.portfolio['Ticker'].isin(stock_to_remove)]
            st.success("Selected stocks removed!")
    if not st.session_state.portfolio.empty and st.button("Clear Portfolio"):
        st.session_state.portfolio = pd.DataFrame(columns=['Ticker', 'Quantity', 'Purchase Price'])
        st.experimental_rerun()
    st.markdown("---")
    st.info(
        """
        **Data Sources:**
        - Reddit (r/stocks, r/investing, r/wallstreetbets)
        - GNews API
        - SEC EDGAR database
        - OpenInsider
        - Yahoo Finance
        """
    )

# ------------------------------
# 13. Main Tabs Layout
# ------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Portfolio", "Analysis", "Recommendations", "Market Intel", "Fundamentals", "Optimizer", "Trending"]
)

# ------------------------------
# 14. Tab 1: Portfolio Overview (Profit/Loss Column)
# ------------------------------
with tab1:
    if not st.session_state.portfolio.empty:
        portfolio_df = st.session_state.portfolio.copy()
        portfolio_df['Current Price'] = portfolio_df['Ticker'].apply(get_current_price)
        portfolio_df['Value'] = portfolio_df['Quantity'] * portfolio_df['Current Price']
        portfolio_df['Profit/Loss'] = (portfolio_df['Current Price'] - portfolio_df['Purchase Price']) * portfolio_df['Quantity']
        total_value = portfolio_df['Value'].sum()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Portfolio Value: ${total_value:,.2f}")
            st.dataframe(portfolio_df.style.format({
                'Current Price': '${:.2f}', 
                'Purchase Price': '${:.2f}', 
                'Value': '${:.2f}',
                'Profit/Loss': '${:.2f}'
            }))
        with col2:
            fig, ax = plt.subplots()
            portfolio_df.groupby('Ticker')['Value'].sum().plot.pie(ax=ax, autopct='%1.1f%%')
            ax.set_ylabel('')
            st.pyplot(fig)
    else:
        st.info("Add stocks using the sidebar.")

# ------------------------------
# 15. Tab 2: Technical & Sentiment Analysis for a Stock
# ------------------------------
with tab2:
    ticker_analysis = st.text_input("Enter ticker for analysis:").upper()
    if ticker_analysis:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Sentiment Analysis")
            reddit_sent = get_reddit_sentiment(ticker_analysis)
            news_sent = get_news_sentiment(ticker_analysis)
            st.metric("Reddit Sentiment", f"{reddit_sent:.2f}")
            st.metric("News Sentiment", f"{news_sent:.2f}")
            fig, ax = plt.subplots()
            ax.bar(['Reddit', 'News'], [reddit_sent, news_sent])
            ax.set_ylim(-1, 1)
            st.pyplot(fig)
        with col2:
            try:
                data = yf.Ticker(ticker_analysis).history(period="6mo")
                data['MA50'] = data['Close'].rolling(50).mean()
                data['MA200'] = data['Close'].rolling(200).mean()
                rsi = calculate_rsi(data['Close'])
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                ax1.plot(data['Close'], label='Price')
                ax1.plot(data['MA50'], label='50-day MA')
                ax1.plot(data['MA200'], label='200-day MA')
                ax1.set_title(f"{ticker_analysis} Price & Moving Averages")
                ax1.legend()
                ax2.plot(data.index, [rsi] * len(data), label='RSI')
                ax2.axhline(70, color='r', linestyle='--')
                ax2.axhline(30, color='g', linestyle='--')
                ax2.set_title("RSI")
                ax2.legend()
                st.pyplot(fig)
            except Exception:
                st.error("Error loading technical data.")

# ------------------------------
# 16. Tab 3: Recommendations & AI Analysis
# ------------------------------
with tab3:
    budget = st.number_input("Investment Budget ($)", min_value=0, value=1000000000, key="rec_budget")
    if st.button("Generate Recommendations"):
        with st.spinner("Analyzing opportunities..."):
            rec_df = generate_recommendations(budget)
            if not rec_df.empty:
                st.subheader("Recommended Allocation")
                st.dataframe(rec_df.style.format({
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
                    {rec_df.head(3).to_dict()}
                    """)
                st.subheader("AI Analysis")
                st.write(analysis)
            else:
                st.info("No recommendations available at this time.")

# ------------------------------
# 17. Tab 4: Market Intelligence
# ------------------------------
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        query_inst = st.text_input("Enter company name for SEC Filings (Institutional Activity):", key="inst_query")
        if query_inst:
            st.subheader("SEC Filings (Institutional Activity)")
            inst_df = get_sec_filings(query_inst)
            if not inst_df.empty:
                st.dataframe(inst_df)
            else:
                st.info("No SEC filings found in the last 6 months for this company.")
        else:
            st.info("Enter a company name to see institutional activity.")
            
        query_insider = st.text_input("Enter ticker for OpenInsider Filings (Insider Trades):", key="insider_query")
        if query_insider:
            st.subheader("OpenInsider Filings (Insider Trades)")
            insider_df = get_openinsider_filings(query_insider)
            if not insider_df.empty:
                st.dataframe(insider_df)
            else:
                st.info("No OpenInsider filings found in the last 6 months for this ticker.")
        else:
            st.info("Enter a ticker to see insider trades.")
    with col2:
        st.subheader("Sector Sentiment")
        fig, ax = plt.subplots()
        sectors = ['Tech', 'Finance', 'Healthcare', 'Energy', 'Consumer']
        ax.barh(sectors, [np.random.uniform(-0.5, 0.8) for _ in sectors])
        ax.set_xlim(-1, 1)
        st.pyplot(fig)
        st.subheader("Market Anxiety Index")
        st.metric("Fear & Greed Index", "38 (Fear)", "-12% week-over-week")

# ------------------------------
# 18. Tab 5: Fundamental Analysis
# ------------------------------
with tab5:
    ticker_fund = st.text_input("Enter ticker for fundamentals:", key="fund_ticker").upper()
    if ticker_fund:
        fundamentals = get_fundamentals(ticker_fund)
        if fundamentals:
            st.subheader(f"{ticker_fund} Fundamentals")
            for key, value in fundamentals.items():
                st.write(f"**{key}:** {value}")
        else:
            st.info("No fundamental data available.")

# ------------------------------
# 19. Tab 6: Portfolio Optimizer using Efficient Frontier
# ------------------------------
with tab6:
    st.subheader("Portfolio Optimization using Efficient Frontier")
    if not st.session_state.portfolio.empty:
        tickers_list = st.session_state.portfolio['Ticker'].unique().tolist()
        prices_df = pd.DataFrame()
        for t in tickers_list:
            data = yf.Ticker(t).history(period="1y")
            if not data.empty:
                prices_df[t] = data["Close"]
        if not prices_df.empty:
            mu = expected_returns.mean_historical_return(prices_df)
            S = risk_models.sample_cov(prices_df)
            ef = EfficientFrontier(mu, S)
            try:
                weights = ef.max_sharpe()
                cleaned_weights = ef.clean_weights()
                st.write("### Optimized Portfolio Weights:")
                st.write(pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Weight']))
                exp_return, volatility, sharpe = ef.portfolio_performance(verbose=True)
                st.write(f"Expected Annual Return: {exp_return*100:.2f}%")
                st.write(f"Annual Volatility: {volatility*100:.2f}%")
                st.write(f"Sharpe Ratio: {sharpe:.2f}")
                fig, ax = plt.subplots()
                ax.pie(list(cleaned_weights.values()), labels=list(cleaned_weights.keys()), autopct='%1.1f%%')
                st.pyplot(fig)
            except Exception:
                st.error("Error during optimization. Ensure sufficient data for all portfolio stocks.")
        else:
            st.info("Insufficient price history for optimization.")
    else:
        st.info("Add stocks to your portfolio to optimize.")

# ------------------------------
# 20. Tab 7: Trending & Emerging Stocks (Placeholder)
# ------------------------------
with tab7:
    st.subheader("Trending & Emerging Stocks")
    trending_df = get_trending_stocks()
    if not trending_df.empty:
        st.dataframe(trending_df.style.format({'Change (%)': '{:.2f}%'}))
    else:
        st.info("Trending stocks data not available.")

# ------------------------------
# Final Suggestions for Further Improvement:
# - Modularize the code (split functions into separate modules for data fetching, analysis, and UI).
# - Integrate interactive charting libraries (e.g., Plotly) for more dynamic visualizations.
# - Implement advanced error logging and monitoring.
# - Allow user customizations (e.g., upload CSV portfolios, adjust scoring weights).
# ------------------------------

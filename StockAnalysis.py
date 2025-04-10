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

# -----------------------------------
# 1. Configuration & API Setup
# -----------------------------------
# Set up API keys and endpoints.
openai.api_key = st.secrets["DEEPSEEK"]["API_KEY"]
openai.api_base = "https://api.deepseek.com"

reddit = praw.Reddit(
    client_id=st.secrets["REDDIT"]["CLIENT_ID"],
    client_secret=st.secrets["REDDIT"]["CLIENT_SECRET"],
    user_agent='Stock Analysis v2.0'
)

# Initialize VADER for sentiment analysis.
vader = SentimentIntensityAnalyzer()

# -----------------------------------
# 2. Caching Helpers for Performance
# These functions cache API responses to reduce external calls.
# -----------------------------------
@st.cache_data(ttl=60)
def get_current_price(ticker):
    """
    Retrieve the latest closing price for a given ticker.
    """
    try:
        data = yf.Ticker(ticker).history(period='1d')
        return data['Close'].iloc[-1]
    except Exception as e:
        st.error(f"Error fetching price for {ticker}: {e}")
        return None

@st.cache_data(ttl=3600)
def get_cik(query):
    """
    Given a company ticker or name, search the SEC EDGAR website for the company
    and extract its CIK. This enables searching for any company.
    """
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?company={query}&owner=exclude&action=getcompany"
    headers = {'User-Agent': 'YourName your-email@example.com'}  # Replace with your info
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            log_error("Error fetching company data from SEC website.")
            return None
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        # Look for the pattern "CIK#: <number>" using a regex
        match = re.search(r"CIK#:\s*([0-9]+)", text)
        if match:
            cik = match.group(1).strip()
            return cik.zfill(10)  # pad with zeros to 10 digits if necessary
        else:
            log_error("CIK not found for the provided company.")
            return None
    except Exception as e:
        log_error(f"Error searching for CIK: {e}")
        return None

# -----------------------------------
# SEC Filings Function
# Retrieves filings using the EDGAR JSON endpoint based on the company's CIK.
# Filters for filings in the last 6 months.
# -----------------------------------
@st.cache_data(ttl=300)
def get_sec_filings(query):
    """
    Search for SEC filings for the given company (by ticker or name).
    First, obtain the CIK then request the JSON submissions file.
    Returns a DataFrame of filings from the last 6 months.
    """
    cik = get_cik(query)
    if cik is None:
        st.error("Cannot retrieve CIK for the company. Please check your query.")
        return pd.DataFrame()
    
    # Build the URL for the JSON submissions file
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {'User-Agent': 'Keval Ahir your-keval.ahir2019@gmail.com'}  # update as necessary
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            log_error("Error fetching SEC filings data from the SEC website.")
            return pd.DataFrame()
        data = response.json()
        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            log_error("No recent filings found.")
            return pd.DataFrame()
        # Create a DataFrame from the recent filings
        df = pd.DataFrame({
            'accessionNumber': recent.get('accessionNumber', []),
            'filingDate': recent.get('filingDate', []),
            'form': recent.get('form', []),
            'reportDate': recent.get('reportDate', [])
        })
        # Convert filingDate to datetime
        df['filingDate'] = pd.to_datetime(df['filingDate'], errors='coerce')
        # Filter filings from the last six months.
        six_months_ago = pd.Timestamp.today() - relativedelta(months=6)
        df = df[df['filingDate'] >= six_months_ago]
        return df.sort_values('filingDate', ascending=False)
    except Exception as e:
        log_error(f"Error processing SEC filings: {e}")
        return pd.DataFrame()

# -----------------------------------
# OpenInsider Filings Function (Scraping)
# Retrieves insider-related filings from OpenInsider for the given ticker.
# Filters for filings in the last 6 months.
# -----------------------------------
@st.cache_data(ttl=300)
def get_openinsider_filings(query):
    """
    Scrape OpenInsider for filings related to the provided ticker.
    It expects a ticker string, so if the query is a company name, try using it directly.
    Filters results to the last six months.
    """
    ticker = query.upper()
    six_months_ago = (pd.Timestamp.today() - relativedelta(months=6)).strftime('%Y-%m-%d')
    today = pd.Timestamp.today().strftime('%Y-%m-%d')
    # Construct a URL that tries to filter for filings by this ticker and date range.
    # (Note: OpenInsider URL parameters may change over time.)
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
        # Skip header row; iterate through remaining rows
        for row in rows[1:]:
            cols = row.find_all("td")
            if len(cols) < 10:  # Make sure there are enough columns
                continue
            # Example: Use col indices based on known table layout
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
        # Drop rows with missing filing dates and filter to last six months.
        df = df.dropna(subset=['Filing Date'])
        six_months_ago_date = pd.Timestamp.today() - relativedelta(months=6)
        df = df[df['Filing Date'] >= six_months_ago_date]
        return df.sort_values('Filing Date', ascending=False)
    except Exception as e:
        log_error(f"Error processing OpenInsider data: {e}")
        return pd.DataFrame()

# -----------------------------------
# Main App: UI for Searching and Displaying Filings
# -----------------------------------
st.set_page_config(page_title="Company Filings", layout="wide")
st.title("SEC and OpenInsider Filings (Last 6 Months)")

st.markdown("""
This app lets you search for any company (by name or ticker) and displays:
- **SEC Filings:** Pulled from SEC EDGAR using the company's CIK.
- **OpenInsider Filings:** Insider trades and related filings.
""")

# Let user enter a company name or ticker.
user_query = st.text_input("Enter a company name or ticker (e.g., AAPL, Microsoft, Tesla):").strip()

if user_query:
    st.header(f"SEC Filings for {user_query}")
    sec_df = get_sec_filings(user_query)
    if not sec_df.empty:
        st.dataframe(sec_df)
    else:
        st.info("No SEC filings found in the last 6 months for this company.")

    st.header(f"OpenInsider Filings for {user_query}")
    insider_df = get_openinsider_filings(user_query)
    if not insider_df.empty:
        st.dataframe(insider_df)
    else:
        st.info("No OpenInsider filings found in the last 6 months for this company.")
# -----------------------------------
# 3. Sentiment & Technical Analysis Functions
# -----------------------------------
# Updated Function: Reddit Sentiment
# -----------------------------------
def get_reddit_sentiment(query):
    """
    Searches various subreddits for posts mentioning the query (which may be a ticker or company name)
    and calculates a weighted sentiment score using VADER.
    """
    subreddits = ['stocks', 'investing', 'wallstreetbets', 'StockMarket']
    posts = []
    for sub in subreddits:
        try:
            submissions = reddit.subreddit(sub).search(query, limit=15, time_filter='week')
            posts.extend([{
                'title': post.title,
                'score': post.score,
                'comments': post.num_comments
            } for post in submissions])
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

# -----------------------------------
# Updated Function: News Sentiment
# -----------------------------------
def get_news_sentiment(query):
    """
    Uses the GNews API to fetch news articles for the query (ticker or company name) and calculates
    a combined sentiment score using both the title and the description.
    """
    try:
        url = f"https://gnews.io/api/v4/search?q={query}&lang=en&token={st.secrets['GNEWS_TOKEN']}"
        response = requests.get(url)
        articles = response.json().get('articles', [])[:10]
        sentiments = []
        for art in articles:
            title = art.get('title', '')
            description = art.get('description', '')
            title_sent = vader.polarity_scores(title)['compound']
            # Use description sentiment if available.
            description_sent = vader.polarity_scores(description)['compound'] if description else 0
            combined_sent = (title_sent + description_sent) / 2
            sentiments.append(combined_sent)
        return np.mean(sentiments) if sentiments else 0
    except Exception as e:
        st.warning(f"Error fetching news data: {e}")
        return 0

# -----------------------------------
# Market Scenario Sentiment (for broader market conditions)
# -----------------------------------
def get_market_scenario():
    """
    Retrieves current general market sentiment using a generic query ("stock market")
    with the GNews API. Provides an overall sentiment score for broader market conditions.
    """
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

# -----------------------------------
# RSI Calculation (unchanged, with robustness improvements)
# -----------------------------------
def calculate_rsi(prices, window=14):
    """
    Calculate the Relative Strength Index (RSI) from a series of prices.
    Uses rolling means with a minimum period and safeguards against division by zero.
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else np.nan

# -----------------------------------
# Example Usage in a Streamlit App
# -----------------------------------
st.title("Enhanced Sentiment Analysis")

# Let the user enter a ticker or company name for sentiment analysis.
user_query = st.text_input("Enter a ticker or company name (e.g., AAPL or Apple):").strip()

if user_query:
    # Get sentiment from Reddit and News for the user query.
    reddit_sent = get_reddit_sentiment(user_query)
    news_sent = get_news_sentiment(user_query)
    market_sent = get_market_scenario()
    
    st.markdown(f"**Reddit Sentiment for '{user_query}':** {reddit_sent:.2f}")
    st.markdown(f"**News Sentiment for '{user_query}':** {news_sent:.2f}")
    st.markdown(f"**Overall Market Sentiment:** {market_sent:.2f}")
    
    # Optionally, if you wish to calculate RSI as well:
    try:
        import yfinance as yf
        price_data = yf.Ticker(user_query).history(period="6mo")
        if not price_data.empty:
            rsi_value = calculate_rsi(price_data['Close'])
            st.markdown(f"**RSI for '{user_query}':** {rsi_value:.2f}")
        else:
            st.warning("No price data available for RSI calculation.")
    except Exception as ex:
        st.error(f"Error calculating RSI: {ex}")
# -----------------------------------
# 4. Recommendation Engine
# -----------------------------------
def generate_recommendations(budget):
    """
    Generate stock recommendations by integrating sentiment, 
    institutional/insider data, and technical indicators (RSI).
    """
    inst_activity = get_institutional_activity()
    insider_trades = get_insider_trades()
    inst_tickers = inst_activity['Ticker'].tolist() if 'Ticker' in inst_activity.columns else []
    insider_tickers = insider_trades['Ticker'].tolist() if 'Ticker' in insider_trades.columns else []
    tickers = list(set(inst_tickers + insider_tickers))
    
    recommendations = []
    for ticker in tickers[:15]:  # Limit analysis for performance
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
        # Normalize the score to allocate investment proportions
        df['Allocation (%)'] = (df['Score'] - df['Score'].min()) / (df['Score'].max() - df['Score'].min()) * 100
        df['Recommended Investment'] = (df['Allocation (%)'] / 100) * budget
        df = df.sort_values('Score', ascending=False)
    return df

# -----------------------------------
# 5. AI Analysis Function
# -----------------------------------
def get_ai_analysis(prompt):
    """
    Uses OpenAI API to generate a market analysis based on a prompt.
    """
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

# -----------------------------------
# 6. Fundamental Analysis Function
# -----------------------------------
@st.cache_data(ttl=300)
def get_fundamentals(ticker):
    """
    Retrieve extensive fundamental data from Yahoo Finance.
    """
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

# -----------------------------------
# 7. Trending Stocks Function (Placeholder)
# -----------------------------------
def get_trending_stocks():
    """
    Placeholder: In the future, scrape real-time trending stock data from free sources.
    Currently returns a sample DataFrame.
    """
    trending = pd.DataFrame({
        'Ticker': ['AAPL', 'TSLA', 'MSFT', 'AMZN', 'NVDA'],
        'Change (%)': np.random.uniform(0, 5, 5)
    })
    return trending

# -----------------------------------
# 8. Streamlit UI Setup & Page Configuration
# -----------------------------------
st.set_page_config(page_title="AI Stock Analyst", layout="wide")
st.title("📈 Intelligent Stock Analysis Platform")

# -----------------------------------
# 9. Initialize Portfolio in Session State
# Use only the base columns: 'Ticker', 'Quantity', and 'Purchase Price'.
# -----------------------------------
if 'portfolio' not in st.session_state or st.session_state.portfolio.empty:
    st.session_state.portfolio = pd.DataFrame(columns=['Ticker', 'Quantity', 'Purchase Price'])

# -----------------------------------
# 10. Sidebar: Portfolio Management (Add/Delete/Edit)
# -----------------------------------
with st.sidebar:
    st.header("💰 Portfolio Management")
    
    # Form to add a new stock (allows decimal quantities)
    with st.form("add_stock"):
        ticker = st.text_input("Stock Ticker").upper()
        qty = st.number_input("Quantity", min_value=0.01, value=1.0, step=0.01)
        cost = st.number_input("Purchase Price", min_value=0.01, value=1.0, step=0.01)
        if st.form_submit_button("Add to Portfolio") and (price := get_current_price(ticker)):
            # Append the new entry to the base portfolio
            base_portfolio = st.session_state.portfolio[['Ticker', 'Quantity', 'Purchase Price']]
            new_entry = pd.DataFrame([[ticker, qty, cost]], columns=base_portfolio.columns)
            st.session_state.portfolio = pd.concat([base_portfolio, new_entry], ignore_index=True)
            st.success(f"{ticker} added!")
    
    # Option to delete individual stocks from portfolio
    if not st.session_state.portfolio.empty:
        tickers_in_portfolio = st.session_state.portfolio['Ticker'].unique().tolist()
        stock_to_remove = st.multiselect("Select stocks to remove", tickers_in_portfolio)
        if st.button("Remove Selected Stocks"):
            st.session_state.portfolio = st.session_state.portfolio[~st.session_state.portfolio['Ticker'].isin(stock_to_remove)]
            st.success("Selected stocks removed!")
    
    # Option to clear the entire portfolio
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

# -----------------------------------
# 11. Main Tabs Layout
# Tabs: Portfolio | Analysis | Recommendations | Market Intel | Fundamentals | Optimizer | Trending
# -----------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Portfolio", "Analysis", "Recommendations", "Market Intel", "Fundamentals", "Optimizer", "Trending"]
)

# -----------------------------------
# 12. Tab 1: Portfolio Overview (with Profit/Loss Column)
# -----------------------------------
with tab1:
    if not st.session_state.portfolio.empty:
        # Create a copy and compute additional columns for display only
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

# -----------------------------------
# 13. Tab 2: Technical & Sentiment Analysis for a Stock
# -----------------------------------
with tab2:
    ticker_input = st.text_input("Enter ticker for analysis:").upper()
    if ticker_input:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Sentiment Analysis")
            reddit_sent = get_reddit_sentiment(ticker_input)
            news_sent = get_news_sentiment(ticker_input)
            st.metric("Reddit Sentiment", f"{reddit_sent:.2f}")
            st.metric("News Sentiment", f"{news_sent:.2f}")
            fig, ax = plt.subplots()
            ax.bar(['Reddit', 'News'], [reddit_sent, news_sent])
            ax.set_ylim(-1, 1)
            st.pyplot(fig)
        with col2:
            try:
                data = yf.Ticker(ticker_input).history(period="6mo")
                data['MA50'] = data['Close'].rolling(50).mean()
                data['MA200'] = data['Close'].rolling(200).mean()
                rsi = calculate_rsi(data['Close'])
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                ax1.plot(data['Close'], label='Price')
                ax1.plot(data['MA50'], label='50-day MA')
                ax1.plot(data['MA200'], label='200-day MA')
                ax1.set_title(f"{ticker_input} Price & Moving Averages")
                ax1.legend()
                ax2.plot(data.index, [rsi] * len(data), label='RSI')
                ax2.axhline(70, color='r', linestyle='--')
                ax2.axhline(30, color='g', linestyle='--')
                ax2.set_title("RSI")
                ax2.legend()
                st.pyplot(fig)
            except Exception:
                st.error("Error loading technical data.")

# -----------------------------------
# 14. Tab 3: Recommendations & AI Analysis
# -----------------------------------
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

# -----------------------------------
# 15. Tab 4: Market Intelligence
# -----------------------------------
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

# -----------------------------------
# 16. Tab 5: Fundamental Analysis
# -----------------------------------
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

# -----------------------------------
# 17. Tab 6: Portfolio Optimizer using Efficient Frontier
# -----------------------------------
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
                # Display a pie chart of the optimized weights
                fig, ax = plt.subplots()
                ax.pie(list(cleaned_weights.values()), labels=list(cleaned_weights.keys()), autopct='%1.1f%%')
                st.pyplot(fig)
            except Exception:
                st.error("Error during optimization. Ensure sufficient data for all portfolio stocks.")
        else:
            st.info("Insufficient price history for optimization.")
    else:
        st.info("Add stocks to your portfolio to optimize.")

# -----------------------------------
# 18. Tab 7: Trending & Emerging Stocks
# -----------------------------------
with tab7:
    st.subheader("Trending & Emerging Stocks")
    trending_df = get_trending_stocks()
    if not trending_df.empty:
        st.dataframe(trending_df.style.format({'Change (%)': '{:.2f}%'}))
    else:
        st.info("Trending stocks data not available.")

# -----------------------------------
# Final Suggestions for Further Improvement:
# - Modularize the code (split functions into separate modules for data fetching, analysis, and UI).
# - Integrate interactive charting libraries (e.g. Plotly) for more dynamic visualizations.
# - Implement advanced error logging and monitoring.
# - Allow user customizations (e.g. upload CSV portfolios, adjust scoring weights).
# -----------------------------------

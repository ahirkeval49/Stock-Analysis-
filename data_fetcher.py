import requests
import streamlit as st
import pandas as pd
from auth import KeyManager

# Initialize KeyManager with keys from secrets
def get_key_manager():
    # Make sure you have these keys in your .streamlit/secrets.toml file
    keys = [st.secrets["AV_API_KEY"], st.secrets["AV_API_KEY1"]]
    return KeyManager(keys)

@st.cache_data(ttl=3600) # Cache price for 1 hour
def fetch_price_history(symbol):
    km = get_key_manager()
    api_key = km.get_active_key()
    
    if not api_key:
        return None, "API Limit Reached for all keys. Try again tomorrow."

    # Fetch full history once, then cache it
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        
        # Check for API Note/Error (Rate Limits)
        if "Note" in data:
            km.log_request(api_key) # Log the failed attempt to trigger rotation
            return None, "Rate Limit Hit (Note from API)"
        if "Error Message" in data:
            return None, "Invalid Symbol"
            
        km.log_request(api_key) # Log success
        
        # Parse Data
        ts_data = data.get("Time Series (Daily)", {})
        df = pd.DataFrame.from_dict(ts_data, orient='index')
        df = df.astype(float)
        df = df.rename(columns={
            "1. open": "open", "2. high": "high", 
            "3. low": "low", "4. close": "close", "5. volume": "volume"
        })
        df.index = pd.to_datetime(df.index)
        return df.sort_index(), "Success"
        
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=86400 * 30) # Cache fundamentals for 30 days [cite: 161]
def fetch_fundamentals(symbol, statement_type="OVERVIEW"):
    """
    statement_type options: OVERVIEW, INCOME_STATEMENT, BALANCE_SHEET, CASH_FLOW
    """
    km = get_key_manager()
    api_key = km.get_active_key()
    
    if not api_key:
        return {} # Return empty dict if no keys available

    url = f"https://www.alphavantage.co/query?function={statement_type}&symbol={symbol}&apikey={api_key}"
    try:
        r = requests.get(url)
        km.log_request(api_key)
        return r.json()
    except:
        return {}

@st.cache_data(ttl=12000) # Cache news for ~3 hours
def fetch_news(symbol):
    km = get_key_manager()
    api_key = km.get_active_key()
    if not api_key: return {}
    
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&limit=10&apikey={api_key}"
    try:
        r = requests.get(url)
        km.log_request(api_key)
        return r.json()
    except:
        return {}

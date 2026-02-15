import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import pandas_ta as ta
import time
import numpy as np
from datetime import datetime
from google import genai

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
st.set_page_config(layout="wide", page_title="Project Atlas", page_icon="📈")

# API Rate Limit Configuration
# Free tier allows 5 calls per minute. We force a small pause to be safe.
API_CALL_DELAY = 2 

# Custom CSS for "Bloomberg-Lite" Aesthetic
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    div[data-testid="stMetric"] {
        background-color: #262730;
        border: 1px solid #464B5C;
        padding: 15px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. KEY MANAGER (Smart Failover)
# ==========================================
class KeyManager:
    def __init__(self, keys):
        self.keys = keys
        # Initialize usage state in session if not present
        if 'api_usage' not in st.session_state:
            st.session_state.api_usage = {key: 0 for key in keys}
            st.session_state.key_status = {key: "active" for key in keys} # Track active/exhausted
            st.session_state.last_reset = datetime.now().date()
            
    def _check_reset(self):
        # Reset if new day (UTC check simplified)
        if datetime.now().date() > st.session_state.last_reset:
            st.session_state.api_usage = {key: 0 for key in self.keys}
            st.session_state.key_status = {key: "active" for key in self.keys}
            st.session_state.last_reset = datetime.now().date()

    def get_active_key(self):
        self._check_reset()
        # Find first key that is active and under limit
        for key in self.keys:
            if st.session_state.key_status[key] == "active":
                return key
        return None 

    def mark_key_as_exhausted(self, key):
        """Called when API actually returns a limit error."""
        st.session_state.key_status[key] = "exhausted"
        print(f"Key {key[:4]}... exhausted. Switching to next key.")

    def log_request(self, key):
        if key in st.session_state.api_usage:
            st.session_state.api_usage[key] += 1

def get_key_manager():
    # Helper to initialize KeyManager from secrets
    try:
        keys = [
            st.secrets["AV_API_KEY"], 
            st.secrets["AV_API_KEY1"],
            st.secrets.get("AV_API_KEY2", st.secrets["AV_API_KEY"]) # Fallback if key 3 missing
        ]
        return KeyManager(keys)
    except FileNotFoundError:
        st.error("Secrets file not found. Please set up .streamlit/secrets.toml")
        st.stop()


# ==========================================
# 3. MOCK DATA GENERATOR (Final Fallback)
# ==========================================
def get_mock_price_data(symbol):
    """Generates fake data so app doesn't crash when API is down."""
    st.warning(f"⚠️ API Limit Reached (or Error). Showing MOCK DATA for {symbol}.")
    dates = pd.date_range(end=datetime.now(), periods=100)
    base_price = 150.0
    prices = base_price + np.cumsum(np.random.randn(100))
    data = {
        'open': prices, 'high': prices + 2, 'low': prices - 2,
        'close': prices + np.random.randn(100),
        'volume': np.random.randint(1000000, 5000000, 100)
    }
    return pd.DataFrame(data, index=dates).sort_index()


# ==========================================
# 4. ROBUST DATA FETCHER (With Auto-Retry)
# ==========================================
def safe_api_call(url_builder_func):
    """
    Wrapper to handle rate limits with automatic retries across keys.
    url_builder_func: A lambda that takes an api_key and returns the URL.
    """
    km = get_key_manager()
    
    # Try up to 3 times (once per key)
    for _ in range(len(km.keys)):
        api_key = km.get_active_key()
        if not api_key:
            break # No keys left

        # Construct URL with the current active key
        url = url_builder_func(api_key)
        
        try:
            time.sleep(API_CALL_DELAY) # Respect speed limit
            response = requests.get(url)
            data = response.json()
            
            # CHECK FOR ERRORS
            if "Note" in data or "Information" in data:
                # Limit reached -> Kill this key and loop to try next one
                km.mark_key_as_exhausted(api_key)
                continue 
            
            # If success, log it and return
            km.log_request(api_key)
            return data, None
            
        except Exception as e:
            return None, f"Connection Error: {e}"
            
    return None, "Daily Limit Reached on ALL keys."

@st.cache_data(ttl=3600)
def fetch_price_history(symbol):
    # Lambda allows us to inject different keys on retry
    url_builder = lambda key: f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={key}"
    
    data, error = safe_api_call(url_builder)
    
    # Fallback to Mock Data if all keys fail
    if error or not data or not data.get("Time Series (Daily)"):
        return get_mock_price_data(symbol), "Showing Mock Data"

    ts_data = data.get("Time Series (Daily)")
    df = pd.DataFrame.from_dict(ts_data, orient='index')
    df = df.astype(float)
    df = df.rename(columns={
        "1. open": "open", "2. high": "high", 
        "3. low": "low", "4. close": "close", "5. volume": "volume"
    })
    df.index = pd.to_datetime(df.index)
    return df.sort_index(), "Success"

@st.cache_data(ttl=2592000) # Cache for 30 days
def fetch_fundamentals(symbol, statement_type="OVERVIEW"):
    url_builder = lambda key: f"https://www.alphavantage.co/query?function={statement_type}&symbol={symbol}&apikey={key}"
    data, error = safe_api_call(url_builder)
    return data if not error else {}

@st.cache_data(ttl=12000) # Cache for ~3 hours
def fetch_news(symbol):
    url_builder = lambda key: f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&limit=10&apikey={key}"
    data, error = safe_api_call(url_builder)
    return data if not error else {}


# ==========================================
# 5. ANALYTICS ENGINES
# ==========================================
def calculate_technicals(df):
    if df is None or df.empty: return df
    df['SMA_50'] = ta.sma(df['close'], length=50)
    df['SMA_200'] = ta.sma(df['close'], length=200)
    df['RSI'] = ta.rsi(df['close'], length=14)
    return df

def calculate_dcf(cash_flow_data, overview_data):
    try:
        if not cash_flow_data or not overview_data: return 0.0
        reports = cash_flow_data.get('annualReports', [])
        if not reports: return 0.0
        
        # Simple DCF Logic
        latest_report = reports[0]
        op_cash = float(latest_report.get('operatingCashflow', 0))
        capex = float(latest_report.get('capitalExpenditures', 0))
        fcf = op_cash - capex
        shares_outstanding = float(overview_data.get('SharesOutstanding', 1))
        
        # Projection
        growth_rate = 0.05
        discount_rate = 0.10
        terminal_growth = 0.025
        
        future_fcf = []
        projected_fcf = fcf
        for i in range(1, 6):
            projected_fcf = projected_fcf * (1 + growth_rate)
            discounted_fcf = projected_fcf / ((1 + discount_rate) ** i)
            future_fcf.append(discounted_fcf)
            
        terminal_val = (projected_fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
        terminal_val_discounted = terminal_val / ((1 + discount_rate) ** 5)
        
        total_value = sum(future_fcf) + terminal_val_discounted
        return round(total_value / shares_outstanding, 2)
    except:
        return 0.0

def generate_agent_analysis(agent_role, data_context, prompt_instruction):
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        full_prompt = f"ROLE: {agent_role}\nDATA: {data_context}\nTASK: {prompt_instruction}"
        response = client.models.generate_content(model="gemini-2.0-flash", contents=full_prompt)
        return response.text
    except Exception as e:
        return f"Agent Error: {str(e)}"


# ==========================================
# 6. MAIN APP LOGIC
# ==========================================
st.sidebar.title("Project Atlas 🌐")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
run_analysis = st.sidebar.button("Run Analysis")

if run_analysis:
    # --- PHASE 1: CHEAP DATA (1 API Call) ---
    with st.spinner(f"Fetching Price Action for {ticker}..."):
        df, msg = fetch_price_history(ticker)
        
        # Calculate Technicals locally
        df = calculate_technicals(df)
        current_price = df['close'].iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${current_price:.2f}")
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'))
        fig.update_layout(template='plotly_dark', height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    # --- PHASE 2: EXPENSIVE DATA (3 API Calls) ---
    st.markdown("---")
    st.info("💡 Tip: Check the box below to unlock AI Analysis & Fair Value (Uses 3 API Calls).")
    
    if st.checkbox("Unlock Deep Analysis (Fundamentals + News + AI)"):
        with st.spinner("Fetching financial statements and news..."):
            overview = fetch_fundamentals(ticker, "OVERVIEW")   
            cash_flow = fetch_fundamentals(ticker, "CASH_FLOW") 
            news = fetch_news(ticker)                           
            
            dcf_value = calculate_dcf(cash_flow, overview)
            
            col2.metric("Intrinsic Value (DCF)", f"${dcf_value}", delta=round(dcf_value - current_price, 2))
            col3.metric("Sector", overview.get("Sector", "N/A"))
            
            tab1, tab2 = st.tabs(["Fundamental Analysis", "Agent Reports"])
            
            with tab1:
                st.subheader("Company Overview")
                st.json(overview, expanded=False)
                
            with tab2:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("### 👮 The Value Investor")
                    analysis = generate_agent_analysis(
                        "Value Investor (Warren Buffett Persona)", 
                        f"Overview: {str(overview)[:1000]}... \nDCF Value: {dcf_value}", 
                        "Is this company a buy based on fundamentals? Focus on MOAT and Safety."
                    )
                    st.markdown(analysis)
                with col_b:
                    st.markdown("### 🤖 The Technician")
                    tech_context = df.tail(10).to_string()
                    analysis = generate_agent_analysis(
                        "Technical Analyst (Jim Simons Persona)",
                        f"Recent Price Action: {tech_context}",
                        "Analyze the trend. Bullish or Bearish? Look for momentum."
                    )
                    st.markdown(analysis)

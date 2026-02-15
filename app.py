import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import pandas_ta as ta
import time
from datetime import datetime
from google import genai

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
st.set_page_config(layout="wide", page_title="AiHedge", page_icon="📈")

# API Rate Limit Configuration
# Free tier allows 5 calls per minute. We force a 12s pause to be safe.
API_CALL_DELAY = 12 

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
# 2. KEY MANAGER (formerly auth.py)
# ==========================================
class KeyManager:
    def __init__(self, keys):
        self.keys = keys
        # Initialize usage state in session if not present
        if 'api_usage' not in st.session_state:
            st.session_state.api_usage = {key: 0 for key in keys}
            st.session_state.last_reset = datetime.now().date()
            
    def _check_reset(self):
        # Reset if new day (UTC check simplified)
        if datetime.now().date() > st.session_state.last_reset:
            st.session_state.api_usage = {key: 0 for key in self.keys}
            st.session_state.last_reset = datetime.now().date()

    def get_active_key(self):
        self._check_reset()
        # Check all keys to see if any are under the limit (25 calls)
        for key in self.keys:
            if st.session_state.api_usage[key] < 25: 
                return key
        return None # All keys exhausted

    def log_request(self, key):
        if key in st.session_state.api_usage:
            st.session_state.api_usage[key] += 1

def get_key_manager():
    # Helper to initialize KeyManager from secrets
    try:
        # UPDATED: Now includes your 3rd key
        keys = [
            st.secrets["AV_API_KEY"], 
            st.secrets["AV_API_KEY1"], 
            st.secrets["AV_API_KEY2"]  # Added M9254PYJTCJET3E0
        ]
        return KeyManager(keys)
    except FileNotFoundError:
        st.error("Secrets file not found. Please set up .streamlit/secrets.toml")
        st.stop()
    except KeyError:
        st.error("Missing a key in secrets.toml. Make sure AV_API_KEY, AV_API_KEY1, and AV_API_KEY2 are all defined.")
        st.stop()


# ==========================================
# 3. DATA FETCHER (formerly data_fetcher.py)
# ==========================================
def safe_api_call(url):
    """Wrapper to handle rate limits and errors gracefully."""
    time.sleep(API_CALL_DELAY) # Enforce speed limit
    try:
        response = requests.get(url)
        data = response.json()
        
        if "Note" in data:
            return None, "Rate Limit Hit (5 calls/min or Daily Limit exhausted)."
        if "Information" in data:
            return None, "Daily Limit Reached."
            
        return data, None
    except Exception as e:
        return None, f"Connection Error: {e}"

@st.cache_data(ttl=3600)
def fetch_price_history(symbol):
    km = get_key_manager()
    api_key = km.get_active_key()
    if not api_key: return None, "No API Keys available."

    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}"
    
    data, error = safe_api_call(url)
    if error: return None, error
    
    ts_data = data.get("Time Series (Daily)")
    if not ts_data: return None, "No price data found."

    km.log_request(api_key)
    
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
    km = get_key_manager()
    api_key = km.get_active_key()
    if not api_key: return {}

    url = f"https://www.alphavantage.co/query?function={statement_type}&symbol={symbol}&apikey={api_key}"
    data, error = safe_api_call(url)
    
    if error or not data: return {}
    km.log_request(api_key)
    return data

@st.cache_data(ttl=12000) # Cache for ~3 hours
def fetch_news(symbol):
    km = get_key_manager()
    api_key = km.get_active_key()
    if not api_key: return {}
    
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&limit=10&apikey={api_key}"
    data, error = safe_api_call(url)
    
    if error or not data: return {}
    km.log_request(api_key)
    return data


# ==========================================
# 4. ANALYTICS ENGINES (Technicals & DCF)
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
            
        latest_report = reports[0]
        op_cash = float(latest_report.get('operatingCashflow', 0))
        capex = float(latest_report.get('capitalExpenditures', 0))
        fcf = op_cash - capex
        
        growth_rate = 0.05
        discount_rate = 0.10
        terminal_growth = 0.025
        shares_outstanding = float(overview_data.get('SharesOutstanding', 1))
        
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


# ==========================================
# 5. GEMINI BRAIN (formerly gemini_brain.py)
# ==========================================
def generate_agent_analysis(agent_role, data_context, prompt_instruction):
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        full_prompt = f"""
        ROLE: {agent_role}
        DATA CONTEXT: {data_context}
        INSTRUCTION: {prompt_instruction}
        FORMAT: Provide a concise analysis in Markdown. Use bullet points.
        """
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=full_prompt
        )
        return response.text
    except Exception as e:
        return f"Agent Error: {str(e)}"


# ==========================================
# 6. MAIN APP LOGIC
# ==========================================
st.sidebar.title("AiHedge")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
run_analysis = st.sidebar.button("Run Analysis")

if run_analysis:
    # --- PHASE 1: CHEAP DATA (1 API Call) ---
    with st.spinner(f"Fetching Price Action for {ticker}..."):
        df, msg = fetch_price_history(ticker)
        
        if df is None or df.empty:
            st.error(msg)
            st.stop()
            
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

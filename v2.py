import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
# from dateutil.relativedelta import relativedelta # Unused, pd.DateOffset is used
from datetime import datetime, timedelta, timezone # Added timezone
import openai
from openai import OpenAI # Ensure this is at the top of your script
from dotenv import load_dotenv
import requests # For web scraping
from bs4 import BeautifulSoup # For web scraping
import re # For parsing text more effectively
from urllib.parse import urljoin # For handling relative URLs from scraping

# Load environment variables (if running locally)
load_dotenv()

# --------------------------------
# Data Fetchers
# --------------------------------
@st.cache_data
def fetch_price_history(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Fetches historical price data for a given ticker."""
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data
def fetch_ticker_info(ticker: str) -> dict:
    """Fetches comprehensive info from yfinance for a ticker."""
    try:
        info = yf.Ticker(ticker).info
        if not info or info.get('regularMarketPrice') is None and info.get('currentPrice') is None :
            return {}
        return {
            "marketCap": info.get("marketCap"),
            "freeCashflow": info.get("freeCashflow"),
            "forwardPE": info.get("forwardPE"),
            "trailingPE": info.get("trailingPE"),
            "priceToBook": info.get("priceToBook"),
            "enterpriseToRevenue": info.get("enterpriseToRevenue"),
            "enterpriseToEbitda": info.get("enterpriseToEbitda"),
            "returnOnEquity": info.get("returnOnEquity"),
            "debtToEquity": info.get("debtToEquity"),
            "beta": info.get("beta"),
            "targetMeanPrice": info.get("targetMeanPrice"),
            "recommendationKey":info.get("recommendationKey"),
            "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions"),
            "industry": info.get("industry"),
            "sector": info.get("sector"),
            "longName": info.get("longName"),
            "shortName": info.get("shortName"),
            "longBusinessSummary": info.get("longBusinessSummary"),
            "currentPrice": info.get("currentPrice") or info.get("regularMarketPrice"),
        }
    except Exception as e:
        return {}

@st.cache_data
def fetch_enriched_news(ticker: str) -> list[dict]:
    """
    Fetches news for a ticker and enriches it with company name,
    formatted publish time, and the ticker symbol itself.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        info_data = ticker_obj.info
        company_name = info_data.get('longName', info_data.get('shortName', ticker))
        raw_news = ticker_obj.news
        enriched_news_list = []

        if not raw_news:
            return []

        for news_item in raw_news:
            enriched_item = news_item.copy()
            enriched_item['ticker'] = ticker
            enriched_item['company_name'] = company_name
            if 'providerPublishTime' in news_item and news_item['providerPublishTime'] is not None:
                try:
                    timestamp = int(news_item['providerPublishTime'])
                    dt_object_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                    enriched_item['publish_datetime_utc'] = dt_object_utc
                    enriched_item['publish_time_readable'] = dt_object_utc.strftime('%Y-%m-%d %H:%M:%S %Z')
                except (ValueError, TypeError, OSError) as e:
                    enriched_item['publish_datetime_utc'] = None
                    enriched_item['publish_time_readable'] = "N/A"
                    enriched_item['publish_time_error'] = str(e)
            else: # Handle case where providerPublishTime might be missing
                enriched_item['publish_datetime_utc'] = None
                enriched_item['publish_time_readable'] = "N/A"

            enriched_item.setdefault('title', 'No Title')
            enriched_item.setdefault('publisher', 'N/A')
            enriched_item.setdefault('link', '#')
            enriched_item.setdefault('type', 'N/A')
            enriched_news_list.append(enriched_item)
            
        enriched_news_list.sort(key=lambda x: x.get('publish_datetime_utc', datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
        return enriched_news_list
    except Exception as e:
        return [{"error": f"Failed to fetch/enrich news for {ticker}: {e}"}]


@st.cache_data
def fetch_inst_filings(ticker: str) -> list[dict]:
    try:
        df = yf.Ticker(ticker).institutional_holders
        if df is not None:
            return df.to_dict("records")
        return []
    except Exception as e:
        return []


@st.cache_data
def fetch_insider_filings(ticker: str) -> list[dict]:
    try:
        df = yf.Ticker(ticker).get_insider_transactions()
        if df is None or df.empty:
            return []
        recs = df.to_dict("records")
        for r in recs:
            tx = r.get("Transaction", "").lower()
            if "purchase" in tx or "buy" in tx:
                r["type"] = "buy"
            elif "sale" in tx or "sell" in tx:
                r["type"] = "sell"
            else:
                r["type"] = "other"
        return recs
    except Exception as e:
        return []

@st.cache_data(ttl=3600)
def fetch_fair_value_from_value_trades(ticker: str, company_name: str) -> dict: # Added company_name
    """
    Attempts to fetch fair value from value-trades.com by constructing a likely stock page URL.
    NO LOGIN. Example target sentence for AAPL:
    "The fair value of Apple Inc (AAPL) is $179. The current price (<span class="math-inline">200\.85\) indicates stock is overvalued by 12\.21%\."
"""
if not company\_name\: 
company\_name\_for\_slug \= ticker 
else\:
company\_name\_for\_slug \= company\_name
\# Attempt to create a plausible slug\: "aapl\-apple\-inc"
name\_part \= company\_name\_for\_slug\.lower\(\)\.replace\('\.', ''\)\.replace\(',', ''\)\.replace\(' inc', ''\)\.replace\(' corp', ''\)\.replace\(' company', ''\)\.replace\(' incorporated', ''\)\.replace\(' limited', ''\)\.replace\(' ltd', ''\)
slug \= f"\{ticker\.lower\(\)\}\-\{'\-'\.join\(name\_part\.split\(\)\[\:3\]\)\}"\.replace\('\-\-','\-'\) \# Replace double hyphens
stock\_page\_url \= f"https\://value\-trades\.com/stock/\{slug\}/"
simple\_slug\_url \= f"https\://value\-trades\.com/stock/\{ticker\.lower\(\)\}/"
session \= requests\.Session\(\)
session\.headers\.update\(\{
'User\-Agent'\: 'Mozilla/5\.0 \(Windows NT 10\.0; Win64; x64\) AppleWebKit/537\.36 \(KHTML, like Gecko\) Chrome/100\.0\.0\.0 Safari/537\.36'
\}\)
response\_stock\_page \= None
final\_url\_attempted \= ""
try\:
\# Try complex slug first
final\_url\_attempted \= stock\_page\_url
response\_stock\_page \= session\.get\(stock\_page\_url, timeout\=15, allow\_redirects\=True\)
\# If complex slug fails \(e\.g\., 404 or doesn't look like a stock page\), try simple slug
if response\_stock\_page\.status\_code \!\= 200 or ticker\.upper\(\) not in response\_stock\_page\.text\[\:10000\]\: \# Check first 10k chars for ticker
if stock\_page\_url \!\= simple\_slug\_url\:
final\_url\_attempted \= simple\_slug\_url
response\_stock\_page \= session\.get\(simple\_slug\_url, timeout\=15, allow\_redirects\=True\)
response\_stock\_page\.raise\_for\_status\(\) \# Raise HTTPError for bad responses \(4XX or 5XX\)
\# Further check if the page content is relevant \(e\.g\., contains ticker name prominently\)
\# This is a heuristic to avoid parsing generic error pages that return 200 OK
if ticker\.upper\(\) not in response\_stock\_page\.text\[\:10000\]\: \# Check a larger chunk of text
return \{"error"\: f"VT\: Fetched page for \{ticker\} from \{final\_url\_attempted\} but content seems irrelevant\.",
"vt\_fair\_value"\: None, "vt\_current\_price"\: None, "vt\_valuation\_text"\: None\}
soup\_stock\_page \= BeautifulSoup\(response\_stock\_page\.content, 'html\.parser'\)
tags\_to\_search \= soup\_stock\_page\.find\_all\(\['p', 'h1', 'h2', 'h3', 'h4', 'div', 'span'\]\)
full\_valuation\_text \= None; fair\_value \= None; site\_current\_price \= None
valuation\_status \= None; valuation\_percentage \= None
ticker\_pattern \= re\.escape\(ticker\.upper\(\)\)
\# Regex updated to be more flexible with company name part and surrounding text
pattern\_str \= rf"The\\s\+fair\\s\+value\\s\+of\\s\+\(?\:\.\*?\)\\s\*\\\(\{ticker\_pattern\}\\\)\\s\*is\\s\*\\$\(\[\\d,\]\+\\\.?\\d\*\)\\s\*\\\.\\s\*The\\s\+current\\s\+price\\s\*\\\(\\s\*\\$\(\[\\d,\]\+\\\.?\\d\*\)\\s\*\\\)\\s\*indicates\\s\+stock\\s\+is\\s\*\(overvalued\|undervalued\)\\s\+by\\s\*\(\[\\d\\\.,\]\+\)%"
found\_sentence \= False
for tag in tags\_to\_search\:
text\_content \= tag\.get\_text\(separator\=" ", strip\=True\)
match \= re\.search\(pattern\_str, text\_content, re\.IGNORECASE\)
if match\:
full\_valuation\_text \= match\.group\(0\)
try\:
fair\_value \= float\(match\.group\(1\)\.replace\(',', ''\)\)
site\_current\_price \= float\(match\.group\(2\)\.replace\(',', ''\)\)
valuation\_status \= match\.group\(3\)\.lower\(\)
valuation\_percentage \= float\(match\.group\(4\)\.replace\(',', ''\)\)
found\_sentence \= True
break 
except ValueError as ve\:
return \{"error"\: f"VT\: Found sentence for \{ticker\}, but failed to parse numbers\: '\{full\_valuation\_text\}'\. Error\: \{ve\}", 
"vt\_fair\_value"\: None, "vt\_current\_price"\: None, "vt\_valuation\_text"\: full\_valuation\_text\}
if found\_sentence\:
return \{"vt\_fair\_value"\: fair\_value, "vt\_current\_price"\: site\_current\_price,
"vt\_valuation\_status"\: valuation\_status, "vt\_valuation\_percentage"\: valuation\_percentage,
"vt\_valuation\_text"\: full\_valuation\_text, "error"\: None\}
else\:
return \{"error"\: f"VT\: Fair value sentence not found for \{ticker\} on page \{final\_url\_attempted\}\.", 
"vt\_fair\_value"\: None, "vt\_current\_price"\: None, "vt\_valuation\_text"\: None\}
except requests\.exceptions\.Timeout\:
return \{"error"\: f"VT\: Timeout accessing \{final\_url\_attempted or initial\_search\_url\}", "vt\_fair\_value"\: None, "vt\_current\_price"\: None, "vt\_valuation\_text"\: None\}
except requests\.exceptions\.HTTPError as http\_err\:
return \{"error"\: f"VT HTTP error for \{ticker\} \(\{http\_err\.response\.status\_code\}\) on URL \{final\_url\_attempted or initial\_search\_url\}", 
"vt\_fair\_value"\: None, "vt\_current\_price"\: None, "vt\_valuation\_text"\: None\}
except requests\.exceptions\.RequestException as req\_err\:
return \{"error"\: f"VT Request error for \{ticker\}\: \{req\_err\}", "vt\_fair\_value"\: None, "vt\_current\_price"\: None, "vt\_valuation\_text"\: None\}
except Exception as e\:
return \{"error"\: f"VT Unexpected error for \{ticker\}\: \{e\}", "vt\_fair\_value"\: None, "vt\_current\_price"\: None, "vt\_valuation\_text"\: None\}
finally\:
if 'session' in locals\(\) and session\:
session\.close\(\)
@st\.cache\_data\(ttl\=3600\)
def fetch\_politician\_trades\(ticker\: str, days\_back\: int \= 365\) \-\> list\[dict\]\:
url \= f"https\://www\.capitoltrades\.com/trades?asset\=\{ticker\.upper\(\)\}&pageSize\=100&perPage\=100"
headers \= \{
'User\-Agent'\: 'Mozilla/5\.0 \(Windows NT 10\.0; Win64; x64\) AppleWebKit/537\.36 \(KHTML, like Gecko\) Chrome/100\.0\.0\.0 Safari/537\.36',
'Accept'\: 'text/html,application/xhtml\+xml,application/xml;q\=0\.9,image/webp,\*/\*;q\=0\.8',
'Accept\-Language'\: 'en\-US,en;q\=0\.5',
'Referer'\: 'https\://www\.capitoltrades\.com/'
\}
politician\_trades\_list \= \[\]
try\:
response \= requests\.get\(url, headers\=headers, timeout\=20\)
response\.raise\_for\_status\(\)
soup \= BeautifulSoup\(response\.content, 'html\.parser'\)
trade\_rows \= soup\.select\("a\[href^\='/trades/'\]\[class\*\='trade\-row'\]"\) \# Example more specific selector
if not trade\_rows\: \# Fallback
trade\_rows \= soup\.find\_all\('a', href\=lambda href\: href and href\.startswith\('/trades/'\)\)
if not trade\_rows\:
return \[\{"error"\: f"CT\: No trade rows found for \{ticker\}\. Site structure might have changed\."\}\]
for row\_link\_tag in trade\_rows\[\:20\]\: \# Limit parsing to avoid excessive time
politician\_name\_tag \= row\_link\_tag\.find\(\['div','span'\], class\_\=lambda x\: x and 'politician\-name' in x\)
tx\_type\_tag \= row\_link\_tag\.find\(\['div','span'\], class\_\=lambda x\: x and 'tx\-type' in x\)
value\_range\_tag \= row\_link\_tag\.find\(\['div','span'\], class\_\=lambda x\: x and 'tx\-value' in x\)
date\_tag \= row\_link\_tag\.find\(\['div','span'\], class\_\=lambda x\: x and 'tx\-date' in x\)
if all\(\[politician\_name\_tag, tx\_type\_tag, value\_range\_tag, date\_tag\]\)\:
name \= politician\_name\_tag\.text\.strip\(\)
tx\_type\_text \= tx\_type\_tag\.text\.strip\(\)\.lower\(\)
tx\_type \= "purchase" if "purchase" in tx\_type\_text else "sale" if "sale" in tx\_type\_text else "other"
value\_range \= value\_range\_tag\.text\.strip\(\)
date\_str \= date\_tag\.text\.strip\(\)
value\_estimate \= 0
value\_matches \= re\.findall\(r'\\$\(\[\\d,\]\+\)', value\_range\)
if value\_matches\:
try\: value\_estimate \= int\(value\_matches\[0\]\.replace\(',', ''\)\)
except ValueError\: pass
politician\_trades\_list\.append\(\{
"politician\_name"\: name, "transaction\_type"\: tx\_type,
"value\_range"\: value\_range, "value\_estimate\_lower"\: value\_estimate,
"date\_str"\: date\_str, "source\_url"\: "https\://www\.capitoltrades\.com" \+ row\_link\_tag\['href'\]
\}\)
if not politician\_trades\_list and trade\_rows\:
return \[\{"error"\: f"CT\: Found trade rows for \{ticker\} but failed to parse fields\. Selectors need update\."\}\]
return politician\_trades\_list
except requests\.exceptions\.Timeout\:
return \[\{"error"\: f"CT\: Timeout accessing CapitolTrades for \{ticker\}"\}\]
except requests\.exceptions\.HTTPError as http\_err\:
return \[\{"error"\: f"CT HTTP error for \{ticker\}\: \{http\_err\}"\}\]
except Exception as e\:
return \[\{"error"\: f"CT Parsing error for \{ticker\}\: \{e\}"\}\]
\# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-
\# LLM Client
\# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-
class ModelClient\:
def \_\_init\_\_\(self, api\_key\: str, provider\: str \= "openai"\)\:
self\.api\_key \= api\_key
self\.provider \= provider
OPENAI\_DEFAULT\_MODEL \= "gpt\-4o"
DEEPSEEK\_DEFAULT\_MODEL \= "deepseek\-reasoner"
if not api\_key\: raise ValueError\("API key required for ModelClient\."\)
if provider \=\= "deepseek"\:
self\.client \= OpenAI\(api\_key\=self\.api\_key, base\_url\="https\://api\.deepseek\.com/v1"\)
self\.model\_name \= DEEPSEEK\_DEFAULT\_MODEL
elif provider \=\= "openai"\:
self\.client \= OpenAI\(api\_key\=self\.api\_key\)
self\.model\_name \= OPENAI\_DEFAULT\_MODEL
else\: raise ValueError\(f"Unsupported LLM provider\: \{provider\}"\)
def embed\(self, texts\: list\[str\], model\_id\: str \= "text\-embedding\-ada\-002"\) \-\> list\[list\[float\]\]\:
actual\_embedding\_model \= model\_id
\# if self\.provider \=\= "deepseek"\: actual\_embedding\_model \= "deepseek\-embedder"
try\:
resp \= self\.client\.embeddings\.create\(input\=texts, model\=actual\_embedding\_model\)
return \[e\.embedding for e in resp\.data\]
except Exception as e\: raise Exception\(f"Embedding Error \(\{self\.provider\}, \{actual\_embedding\_model\}\)\: \{e\}"\)
def generate\(self, prompt\: str\) \-\> str\:
try\:
stream \= self\.client\.chat\.completions\.create\(model\=self\.model\_name, messages\=\[\{"role"\: "user", "content"\: prompt\}\], stream\=True\)
final\_content \= ""
for chunk in stream\:
if chunk\.choices and chunk\.choices\[0\]\.delta and chunk\.choices\[0\]\.delta\.content\:
final\_content \+\= chunk\.choices\[0\]\.delta\.content
return final\_content
except Exception as e\: raise Exception\(f"LLM Generation Error \(\{self\.provider\}, \{self\.model\_name\}\)\: \{e\}"\)
\# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-
\# Agents
\# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-
class PriceAgent\:
def run\(self, ticker\: str, price\_data\_slice\: pd\.DataFrame\) \-\> dict\:
if price\_data\_slice\.empty or len\(price\_data\_slice\) < 200\:
return \{"ticker"\: ticker, "price\_signal"\: "hold", "sma50"\: np\.nan, "sma200"\: np\.nan, "rsi14"\: np\.nan\}
df \= price\_data\_slice\.copy\(\); df\["SMA50"\]  \= df\["Close"\]\.rolling\(50\)\.mean\(\); df\["SMA200"\] \= df\["Close"\]\.rolling\(200\)\.mean\(\)
delta \= df\["Close"\]\.diff\(\); gain  \= delta\.clip\(lower\=0\)\.rolling\(14\)\.mean\(\); loss  \= \(\-delta\.clip\(upper\=0\)\)\.rolling\(14\)\.mean\(\)
rs \= gain / loss\.replace\(0, np\.nan\); df\["RSI14"\] \= 100 \- \(100 / \(1 \+ rs\)\); latest \= df\.iloc\[\-1\]; signal \= "hold"
if pd\.isna\(latest\.SMA50\) or pd\.isna\(latest\.SMA200\) or pd\.isna\(latest\.RSI14\)\: signal \= "hold"
elif latest\.SMA50 \> latest\.SMA200 and latest\.RSI14 < 70\: signal \= "buy"
elif latest\.SMA50 < latest\.SMA200 and latest\.RSI14 \> 30\: signal \= "sell"
return \{"ticker"\: ticker, "sma50"\: float\(latest\.SMA50\) if pd\.notna\(latest\.SMA50\) else np\.nan,
"sma200"\: float\(latest\.SMA200\) if pd\.notna\(latest\.SMA200\) else np\.nan,
"rsi14"\: float\(latest\.RSI14\) if pd\.notna\(latest\.RSI14\) else np\.nan, "price\_signal"\: signal\}
class MomentumAgent\:
def run\(self, ticker\: str, price\_data\_slice\: pd\.DataFrame\) \-\> dict\:
if price\_data\_slice\.empty or len\(price\_data\_slice\) < 252\:
return \{"ticker"\: ticker, "momentum\_signal"\: "hold", "momentum\_1m"\: 0, "momentum\_12m"\: 0\}
df \= price\_data\_slice; P\_t \= df\.Close\.iloc\[\-1\]
P\_1m \= df\.Close\.shift\(21\)\.iloc\[\-1\] if len\(df\) \> 21 else np\.nan
P\_12m \= df\.Close\.shift\(252\)\.iloc\[\-1\] if len\(df\) \> 252 else np\.nan
m1  \= \(P\_t/P\_1m\)\-1  if pd\.notna\(P\_1m\) and P\_1m \!\= 0 else 0
m12 \= \(P\_t/P\_12m\)\-1 if pd\.notna\(P\_12m\) and P\_12m \!\= 0 else 0; signal \= "hold"
if m12 \> 0\.01 and m1 \> 0\.01\: signal \= "buy"
elif m12 < \-0\.01 and m1 < \-0\.01\: signal \= "sell"
return \{"ticker"\: ticker, "momentum\_1m"\: float\(m1\), "momentum\_12m"\: float\(m12\), "momentum\_signal"\: signal\}
class VolatilityAgent\:
def run\(self, ticker\: str, data\: dict, price\_data\_slice\: pd\.DataFrame \= None\) \-\> dict\:
beta \= data\.get\("ticker\_info", \{\}\)\.get\("beta", 1\.0\); 
if beta is None\: beta \= 1\.0 \# Ensure beta is not None
sig  \= "sell" if beta \> 1\.5 else \("buy" if beta < 0\.8 else "hold"\)
ann\_vol \= np\.nan; weight \= 0\.0
if price\_data\_slice is not None and not price\_data\_slice\.empty and len\(price\_data\_slice\) \> 1\:
ret \= np\.log\(price\_data\_slice\.Close / price\_data\_slice\.Close\.shift\(1\)\)\.dropna\(\)
if not ret\.empty\: ann\_vol \= float\(ret\.std\(\) \* np\.sqrt\(252\)\); weight  \= float\(1/ann\_vol\) if ann\_vol \> 0 else 0\.0
return \{"ticker"\: ticker, "beta"\: beta, "annual\_vol"\: ann\_vol, "vol\_weight"\: weight, "volatility\_signal"\: sig\}
class SentimentAgent\:
def \_\_init\_\_\(self, client\)\: self\.client \= client
def run\(self, ticker\: str, data\: dict\) \-\> dict\:
enriched\_news\_items \= data\.get\("news", \[\]\)
if not enriched\_news\_items or \(isinstance\(enriched\_news\_items, list\) and len\(enriched\_news\_items\) \> 0 and isinstance\(enriched\_news\_items\[0\], dict\) and "error" in enriched\_news\_items\[0\]\)\:
return \{"ticker"\: ticker, "sentiment\_score"\: 0\.0, "sentiment\_signal"\: "hold",
"sentiment\_error"\: enriched\_news\_items\[0\]\.get\("error"\) if enriched\_news\_items and isinstance\(enriched\_news\_items, list\) and enriched\_news\_items\[0\] else "No news items found\."\}
headlines\_with\_context \= \[\]
company\_name\_overall \= ticker
if enriched\_news\_items and isinstance\(enriched\_news\_items, list\) and len\(enriched\_news\_items\) \> 0 and isinstance\(enriched\_news\_items\[0\], dict\) and "company\_name" in enriched\_news\_items\[0\]\:
company\_name\_overall \= enriched\_news\_items\[0\]\.get\("company\_name", ticker\)
for item in enriched\_news\_items\[\:10\]\:
if isinstance\(item, dict\)\:
title \= item\.get\('title', 'No Title'\)
publisher \= item\.get\('publisher', 'N/A'\)
company\_name \= item\.get\('company\_name', ticker\)
headlines\_with\_context\.append\(f"From \{publisher\} about \{company\_name\}\: \{title\}"\)
if not headlines\_with\_context\:
return \{"ticker"\: ticker, "sentiment\_score"\: 0\.0, "sentiment\_signal"\: "hold", "sentiment\_error"\: "No relevant headlines after processing\."\}
prompt \= \(
f"Analyze the sentiment of the following news headlines regarding \{company\_name\_overall\} \(\{ticker\}\)\. "
f"Provide a single floating\-point sentiment score between \-1\.0 \(very negative\) and \+1\.0 \(very positive\)\. "
f"Consider the source and content\. Output only the number\.\\n\\nHeadlines\:\\n"
\+ "\\n"\.join\(f"\- \{h\}" for h in headlines\_with\_context\)
\)
score \= 0\.0; error\_msg \= None
try\:
response \= self\.client\.generate\(prompt\)\.strip\(\)
if response\.startswith\("Error\:"\)\: error\_msg \= response
else\:
match \= re\.search\(r"\[\-\+\]?\\d\*\\\.\\d\+\|\\d\+", response\)
if match\: score \= float\(match\.group\(0\)\)
else\: error\_msg \= "LLM did not return a parsable number\."
except Exception as e\: error\_msg \= str\(e\)\[\:150\]
score \= max\(\-1\.0, min\(1\.0, score\)\)
sig \= "buy" if score \> 0\.25 else \("sell" if score < \-0\.25 else "hold"\)
return \{"ticker"\:ticker, "sentiment\_score"\:score, "sentiment\_signal"\:sig, "sentiment\_error"\: error\_msg\}
class FundamentalsAgent\:
def run\(self, ticker\: str, data\: dict\) \-\> dict\:
s \= data\.get\("ticker\_info", \{\}\); mcap \= s\.get\("marketCap"\) or 1; fcf \= s\.get\("freeCashflow"\) or 0
roe \= s\.get\("returnOnEquity"\) or 0; de \= s\.get\("debtToEquity"\); de \= 1000 if de is None else de
fcy \= fcf/mcap if mcap \!\= 0 else 0; piotroski\_score \= sum\(\[roe \> 0\.01, de < 100, fcf \> 0\]\)
sig  \= "buy" if piotroski\_score \>\= 2 else \("sell" if piotroski\_score \=\= 0 else "hold"\)
return \{"ticker"\: ticker, "fcf\_yield"\: float\(fcy\), "piotroski\_score"\: piotroski\_score, "fund\_signal"\: sig\}
class ValuationAgent\:
def run\(self, ticker\: str, data\: dict\) \-\> dict\:
stats \= data\.get\("ticker\_info", \{\}\); price\_history\_df \= data\.get\("price\_history"\)
price \= stats\.get\("currentPrice"\) or \(price\_history\_df\["Close"\]\.iloc\[\-1\] if price\_history\_df is not None and not price\_history\_df\.empty else None\)
if price is None\: return \{"ticker"\: ticker, "forward\_pe"\: None, "relative\_pe\_signal"\: "hold", "dcf\_fair\_price"\: np\.nan, "dcf\_signal"\: "hold"\}
pe \= stats\.get\("forwardPE"\); rel\_sig \= "hold";
if pe is not None\: rel\_sig \= "buy" if pe < 15 else "sell" if pe \> 25 else "hold"
fcf \= stats\.get\("freeCashflow"\); mcap \= stats\.get\("marketCap"\)
fcy \= \(fcf / mcap\) if fcf is not None and mcap is not None and mcap \!\= 0 else 0\.0
fair\_price \= price \* \(1 \+ fcy\); dcf\_sig \= "hold"
if fair\_price \> price \* 1\.15\: dcf\_sig \= "buy"
elif fair\_price < price \* 0\.85\: dcf\_sig \= "sell"
return \{"ticker"\: ticker, "forward\_pe"\: pe, "relative\_pe\_signal"\: rel\_sig, "dcf\_fair\_price"\: float\(fair\_price\), "dcf\_signal"\: dcf\_sig\}
class FilingsAgent\:
def run\(self, ticker\: str, data\: dict\) \-\> dict\:
insiders \= data\.get\("insider\_filings",\[\]\); net\_shares \= 0
if insiders\:
for r in insiders\:
shares\_val \= r\.get\("Shares",0\)
try\: shares \= int\(str\(shares\_val\)\.replace\(',',''\)\)
except ValueError\: shares \= 0
if r\.get\("type"\) \=\= "buy"\: net\_shares \+\= shares
elif r\.get\("type"\) \=\= "sell"\: net\_shares \-\= shares
sig \= "buy" if net\_shares \> 1000 else \("sell" if net\_shares < \-1000 else "hold"\)
return \{"ticker"\: ticker, "net\_insider\_shares"\: int\(net\_shares\), "filings\_signal"\: sig\}
class AnalystRatingAgent\:
def run\(self, ticker\: str, data\: dict\) \-\> dict\:
ticker\_info \= data\.get\("ticker\_info", \{\}\)
price\_history\_df \= data\.get\("price\_history"\)
current\_price \= ticker\_info\.get\("currentPrice"\) or \(price\_history\_df\["Close"\]\.iloc\[\-1\] if price\_history\_df is not None and not price\_history\_df\.empty else None\)
target\_mean\_price \= ticker\_info\.get\("targetMeanPrice"\)
recommendation \= str\(ticker\_info\.get\("recommendationKey", "hold"\)\)\.lower\(\)
upside \= 0\.0
if target\_mean\_price and current\_price and current\_price \> 0\:
try\: upside \= \(float\(target\_mean\_price\) / float\(current\_price\)\) \- 1
except\: upside \= 0\.0
sig \= "hold"
if recommendation in \["buy", "strong\_buy"\] and upside \> 0\.10\: sig \= "buy"
elif recommendation \=\= "buy" and upside \> 0\.05\: sig \= "buy"
elif recommendation in \["sell", "strong\_sell", "underperform"\] and upside < \-0\.05\: sig \= "sell"
elif upside \> 0\.20\: sig \= "buy"
elif upside < \-0\.15\: sig \= "sell"
buy\_pct\_inferred \= \{"strong\_buy"\: 0\.9, "buy"\: 0\.7, "hold"\: 0\.5, "underperform"\: 0\.3, "sell"\: 0\.1\}\.get\(recommendation, 0\.5\)
return \{"ticker"\: ticker, "analyst\_buy\_pct\_inferred"\: buy\_pct\_inferred, "target\_upside"\: float\(upside\),
"yfinance\_recommendation"\: recommendation, "analyst\_signal"\: sig\}
class PoliticianFilingsAgent\:
def run\(self, ticker\: str, data\: dict\) \-\> dict\:
trades \= data\.get\("politician\_trades", \[\]\)
net\_value\_estimate \= 0; buy\_count \= 0; sell\_count \= 0; error \= None
if trades and isinstance\(trades, list\) and len\(trades\)\>0 and isinstance\(trades\[0\], dict\) and "error" in trades\[0\]\:
error \= trades\[0\]\["error"\]
elif trades\:
for trade in trades\:
if isinstance\(trade, dict\)\:
value \= trade\.get\("value\_estimate\_lower", 0\)
if trade\.get\("transaction\_type"\) \=\= "purchase"\: net\_value\_estimate \+\= value; buy\_count \+\=1
elif trade\.get\("transaction\_type"\) \=\= "sale"\: net\_value\_estimate \-\= value; sell\_count \+\=1
signal \= "hold"
if not error\:
if buy\_count \> sell\_count and buy\_count \> 1 \: signal \= "buy"
elif sell\_count \> buy\_count and sell\_count \> 1\: signal \= "sell"
return \{"ticker"\: ticker, "politician\_net\_trade\_value\_estimate"\: net\_value\_estimate,
"politician\_buy\_tx\_count"\: buy\_count, "politician\_sell\_tx\_count"\: sell\_count,
"politician\_filings\_signal"\: signal, "politician\_data\_error"\: error\}
class FairValueAgentVT\:
def run\(self, ticker\: str, data\: dict\) \-\> dict\:
vt\_data \= data\.get\("value\_trades\_fair\_value\_data", \{\}\)
fair\_value \= vt\_data\.get\("vt\_fair\_value"\)
error \= vt\_data\.get\("error"\)
current\_price\_data \= data\.get\("ticker\_info", \{\}\)
current\_price \= current\_price\_data\.get\("currentPrice"\)
if current\_price is None and data\.get\("price\_history"\) is not None and not data\["price\_history"\]\.empty\:
current\_price \= data\["price\_history"\]\["Close"\]\.iloc\[\-1\]
signal \= "hold"; margin\_of\_safety \= 0\.20
error\_fv\_not\_found\_page \= f"VT\: FV not found on page for \{ticker\}\."
error\_specific\_sentence\_prefix \= f"VT\: Specific fair value sentence not found for \{ticker\} on page"
significant\_error \= False
if error\:
if error not in \[error\_fv\_not\_found\_page, "VT Configuration incomplete in secrets\.", "VT\: Skipped by user config\."\] and \\
not error\.startswith\(error\_specific\_sentence\_prefix\)\:
significant\_error \= True
if not significant\_error and fair\_value is not None and current\_price is not None and current\_price \> 0\:
if current\_price < fair\_value \* \(1 \- margin\_of\_safety\)\: signal \= "buy"
elif current\_price \> fair\_value \* \(1 \+ margin\_of\_safety\)\: signal \= "sell"
return \{"ticker"\: ticker, "vt\_fair\_value\_estimate"\: fair\_value, "vt\_fair\_value\_signal"\: signal, "vt\_data\_error"\: error,
"vt\_site\_current\_price"\: vt\_data\.get\("vt\_current\_price"\),
"vt\_valuation\_status"\: vt\_data\.get\("vt\_valuation\_status"\),
"vt\_valuation\_percentage"\: vt\_data\.get\("vt\_valuation\_percentage"\),
"vt\_valuation\_text\_display"\: vt\_data\.get\("vt\_valuation\_text"\)\}
class PortfolioAgent\:
WEIGHTS \= \{
"price"\: 1\.0, "momentum"\: 0\.8, "volatility"\: 0\.3, "sentiment"\: 0\.6, "fund"\: 0\.9,
"valuation\_dcf"\:0\.5, "valuation\_pe"\:0\.5, "filings"\: 0\.5, "analyst"\: 0\.7,
"politician\_filings"\: 0\.4, "vt\_fair\_value"\: 0\.8
\}
def run\(self, ticker\: str, signals\: list\[dict\], agent\_weights\: dict \= None\) \-\> dict\:
current\_weights \= agent\_weights or self\.WEIGHTS; total\_weighted\_score \= 0; sum\_of\_weights\_used \= 0
agg\_signals \= \{\};
for s\_dict in signals\:
if isinstance\(s\_dict, dict\)\: agg\_signals\.update\(s\_dict\)
signal\_map \= \{"price\_signal"\: "price", "momentum\_signal"\: "momentum", "volatility\_signal"\: "volatility",
"sentiment\_signal"\: "sentiment", "fund\_signal"\: "fund", "dcf\_signal"\: "valuation\_dcf",
"relative\_pe\_signal"\: "valuation\_pe", "filings\_signal"\: "filings", "analyst\_signal"\: "analyst",
"politician\_filings\_signal"\: "politician\_filings", "vt\_fair\_value\_signal"\: "vt\_fair\_value"\}
for signal\_key, weight\_key in signal\_map\.items\(\)\:
signal\_value \= agg\_signals\.get\(signal\_key\); weight \= current\_weights\.get\(weight\_key, 0\)
if signal\_value and weight \> 0 \:
if signal\_value in \["buy", "hold", "sell"\]\:
raw\_score \= \{"buy"\:1, "hold"\:0, "sell"\:\-1\}\.get\(signal\_value, 0\)
total\_weighted\_score \+\= raw\_score \* weight
sum\_of\_weights\_used \+\= weight
composite\_score \= \(total\_weighted\_score / sum\_of\_weights\_used\) if sum\_of\_weights\_used else 0\.0
final\_decision \= "buy" if composite\_score \> 0\.15 else "sell" if composite\_score < \-0\.15 else "hold"
return \{"ticker"\:ticker, "composite\_score"\:composite\_score, "final\_decision"\:final\_decision\}
\# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-
\# Orchestrator for Live Analysis
\# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-
def run\_live\_analysis\(tickers, history\_years, llm\_client, configs\)\:
results \= \{\}
for t in tickers\:
price\_history\_full \= fetch\_price\_history\(t, period\=f"\{history\_years\}y"\)
if price\_history\_full\.empty\:
results\[t\] \= \{"error"\: f"Failed to fetch price history for \{t\}\.", "ticker"\: t, "final\_decision"\:"error", "composite\_score"\:0\}
continue
ticker\_info \= fetch\_ticker\_info\(t\)
if not ticker\_info\:
results\[t\] \= \{"error"\: f"Failed to fetch ticker info for \{t\}\.", "ticker"\: t, "final\_decision"\:"error", "composite\_score"\:0\}
continue
current\_price\_for\_ticker \= ticker\_info\.get\("currentPrice"\) or \(price\_history\_full\["Close"\]\.iloc\[\-1\] if not price\_history\_full\.empty else None\)
news\_data\_list \= fetch\_enriched\_news\(t\) if configs\["use\_sentiment"\] else \[\]
politician\_trades\_list \= fetch\_politician\_trades\(t\) if configs\["use\_politician\_filings"\] else \[\]
data\_bundle \= \{
"price\_history"\: price\_history\_full, "ticker\_info"\: ticker\_info,
"news"\: news\_data\_list,
"insider\_filings"\: fetch\_insider\_filings\(t\) if configs\["use\_filings"\] else \[\],
"politician\_trades"\: politician\_trades\_list,
\# Corrected call for fetch\_fair\_value\_from\_value\_trades \(no company\_name arg as per simplified fetcher\)
"value\_trades\_fair\_value\_data"\: fetch\_fair\_value\_from\_value\_trades\(t\) if configs\["use\_value\_trades"\] else \\
\{"vt\_fair\_value"\: None, "error"\: "VT\: Skipped by user config\."\}
\}
all\_agents\_instances \= \[PriceAgent\(\), MomentumAgent\(\), VolatilityAgent\(\), FundamentalsAgent\(\), ValuationAgent\(\), AnalystRatingAgent\(\)\]
if configs\["use\_sentiment"\] and llm\_client\: all\_agents\_instances\.append\(SentimentAgent\(llm\_client\)\)
if configs\["use\_filings"\]\: all\_agents\_instances\.append\(FilingsAgent\(\)\)
if configs\["use\_politician\_filings"\]\: all\_agents\_instances\.append\(PoliticianFilingsAgent\(\)\)
if configs\["use\_value\_trades"\]\: all\_agents\_instances\.append\(FairValueAgentVT\(\)\)
agent\_results\_list \= \[\]
for agent\_instance in all\_agents\_instances\:
agent\_name \= agent\_instance\.\_\_class\_\_\.\_\_name\_\_
try\:
if isinstance\(agent\_instance, \(PriceAgent, MomentumAgent\)\)\: res\_agent \= agent\_instance\.run\(t, data\_bundle\["price\_history"\]\)
elif isinstance\(agent\_instance, VolatilityAgent\)\: res\_agent \= agent\_instance\.run\(t, data\_bundle, data\_bundle\["price\_history"\]\)
else\: res\_agent \= agent\_instance\.run\(t, data\_bundle\)
agent\_results\_list\.append\(res\_agent\)
except Exception as e\:
agent\_error\_key \= agent\_name\.lower\(\)\.replace\("agent",""\) \+ "\_error"
default\_signal\_key\_name \= agent\_name\.lower\(\)\.replace\("agent",""\) \+ "\_signal"
agent\_results\_list\.append\(\{default\_signal\_key\_name\: "error", agent\_error\_key\: f"Agent \{agent\_name\} error\: \{str\(e\)\[\:100\]\}"\}\)
final\_decision \= PortfolioAgent\(\)\.run\(t, agent\_results\_list\)
current\_result\_dict \= \{
"ticker"\: t, 
"current\_price\_display"\: current\_price\_for\_ticker,
"market\_cap\_display"\: ticker\_info\.get\("marketCap"\),
"industry\_display"\: ticker\_info\.get\("industry"\),
"sector\_display"\: ticker\_info\.get\("sector"\),
"ticker\_info"\: ticker\_info, 
"news\_headlines\_for\_popover"\: \[n\.get\('title', 'N/A'\) for n in news\_data\_list\[\:5\] if isinstance\(n,dict\)\],
"politician\_trades\_for\_popover"\: \[pt for pt in politician\_trades\_list\[\:5\] if isinstance\(pt, dict\) and "error" not in pt\]
\}
for res\_dict in agent\_results\_list\:
if isinstance\(res\_dict, dict\)\:
current\_result\_dict\.update\(res\_dict\)
current\_result\_dict\.update\(final\_decision\)
results\[t\] \= current\_result\_dict
return results
\# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-
\# Backtesting Engine
\# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-
def run\_backtest\(ticker, start\_date, end\_date, initial\_capital, llm\_client\_placeholder, backtest\_agent\_weights\)\:
s\_date\_obj \= datetime\.strptime\(start\_date, "%Y\-%m\-%d"\)
fetch\_start\_date \= \(s\_date\_obj \- pd\.DateOffset\(months\=18\)\)\.strftime\("%Y\-%m\-%d"\)
full\_price\_history \= fetch\_price\_history\(ticker, period\=None, interval\="1d"\)
if full\_price\_history\.empty\:
return \{"error"\: "Backtest failed\: Price history empty\."\}, pd\.DataFrame\(\)
price\_history \= full\_price\_history\[\(full\_price\_history\.index \>\= pd\.to\_datetime\(fetch\_start\_date\)\) & \(full\_price\_history\.index <\= pd\.to\_datetime\(end\_date\)\)\]\.copy\(\)
if price\_history\.empty or len\(price\_history\[price\_history\.index \>\= pd\.to\_datetime\(start\_date\)\]\) < 2\:
return \{"error"\: "Backtest failed\: Not enough data in range\."\}, pd\.DataFrame\(\)
ticker\_info\_for\_backtest \= fetch\_ticker\_info\(ticker\)
data\_bundle\_static \= \{"ticker\_info"\: ticker\_info\_for\_backtest\}
price\_agent \= PriceAgent\(\); momentum\_agent \= MomentumAgent\(\); volatility\_agent \= VolatilityAgent\(\); portfolio\_agent \= PortfolioAgent\(\)
portfolio\_log \= \[\]; cash \= initial\_capital; shares\_held \= 0; portfolio\_value \= initial\_capital
backtest\_run\_dates \= price\_history\[price\_history\.index \>\= pd\.to\_datetime\(start\_date\)\]\.index
for current\_date in backtest\_run\_dates\:
data\_slice \= price\_history\[price\_history\.index <\= current\_date\]
current\_price\_point \= data\_slice\.Close\.iloc\[\-1\] if not data\_slice\.empty else portfolio\_value / shares\_held if shares\_held else 0
if data\_slice\.empty or len\(data\_slice\) < 252\:
portfolio\_log\.append\(\{"date"\: current\_date, "cash"\: cash, "shares\_held"\: shares\_held, "price"\: current\_price\_point, "portfolio\_value"\: portfolio\_value, "signal"\: "hold \(insufficient data\)", "composite\_score"\:0\.0\}\); continue
current\_price \= data\_slice\.Close\.iloc\[\-1\]
pa\_res \= price\_agent\.run\(ticker, data\_slice\); ma\_res \= momentum\_agent\.run\(ticker, data\_slice\); va\_res \= volatility\_agent\.run\(ticker, data\_bundle\_static, data\_slice\)
final\_decision\_obj \= portfolio\_agent\.run\(ticker, \[pa\_res, ma\_res, va\_res\], agent\_weights\=backtest\_agent\_weights\)
final\_decision \= final\_decision\_obj\["final\_decision"\]
if final\_decision \=\= "buy" and cash \> current\_price \: shares\_to\_buy \= cash / current\_price; shares\_held \+\= shares\_to\_buy; cash \= 0
elif final\_decision \=\= "sell" and shares\_held \> 0\: cash \+\= shares\_held \* current\_price; shares\_held \= 0
portfolio\_value \= cash \+ shares\_held \* current\_price
portfolio\_log\.append\(\{"date"\: current\_date, "cash"\: cash, "shares\_held"\: shares\_held, "price"\: current\_price, "portfolio\_value"\: portfolio\_value, "signal"\: final\_decision, "composite\_score"\: final\_decision\_obj\["composite\_score"\]\}\)
log\_df \= pd\.DataFrame\(portfolio\_log\);
if not log\_df\.empty\: log\_df\.set\_index\("date", inplace\=True\)
if log\_df\.empty or len\(log\_df\) < 2\:
return \{"message"\:"Log too short to calculate performance metrics\."\}, pd\.DataFrame\(\)
total\_return \= \(log\_df\["portfolio\_value"\]\.iloc\[\-1\] / initial\_capital \- 1\) \* 100
num\_days \= \(log\_df\.index\[\-1\] \- log\_df\.index\[0\]\)\.days; num\_years \= num\_days / 365\.25 if num\_days \> 0 else 1/365\.25
annualized\_return \= \(\(log\_df\["portfolio\_value"\]\.iloc\[\-1\] / initial\_capital\) \*\* \(1/num\_years\) \- 1\) \* 100 if num\_years \> 0 else total\_return if num\_days \> 0 else 0
log\_df\["daily\_return"\] \= log\_df\["portfolio\_value"\]\.pct\_change\(\)\.fillna\(0\); annualized\_volatility \= log\_df\["daily\_return"\]\.std\(\) \* np\.sqrt\(252\) \* 100
sharpe\_ratio \= \(annualized\_return / annualized\_volatility\) if annualized\_volatility \!\= 0 else 0
log\_df\["cumulative\_max"\] \= log\_df\["portfolio\_value"\]\.cummax\(\); log\_df\["drawdown"\] \= \(log\_df\["portfolio\_value"\] \- log\_df\["cumulative\_max"\]\) / log\_df\["cumulative\_max"\]
max\_drawdown \= log\_df\["drawdown"\]\.min\(\) \* 100
metrics \= \{"Initial Capital"\: f"</span>{initial_capital:,.2f}", "Final Portfolio Value": f"<span class="math-inline">\{log\_df\['portfolio\_value'\]\.iloc\[\-1\]\:,\.2f\}",
"Total Return \(%\)"\: f"\{total\_return\:\.2f\}%", "Annualized Return \(%\)"\: f"\{annualized\_return\:\.2f\}%",
"Annualized Volatility \(%\)"\: f"\{annualized\_volatility\:\.2f\}%", "Sharpe Ratio"\: f"\{sharpe\_ratio\:\.2f\}",
"Max Drawdown \(%\)"\: f"\{max\_drawdown\:\.2f\}%", "Number of Trades \(approx\)"\: f"\{\(log\_df\['signal'\] \!\= log\_df\['signal'\]\.shift\(\)\)\.fillna\(False\)\.sum\(\) // 2\}"\}
return metrics, log\_df
\# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-
\# Streamlit UI
\# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-
st\.set\_page\_config\(page\_title\="AI Hedge Fund Simulator", layout\="wide"\)
st\.title\("🚀 AI Hedge Fund Simulator"\)
\# LLM Client Initialization
llm\_client \= None
try\:
deepseek\_key \= getattr\(st\.secrets, "DEEPSEEK\_API\_KEY", None\) if hasattr\(st\.secrets, "DEEPSEEK\_API\_KEY"\) else None
openai\_key \= getattr\(st\.secrets, "OPENAI\_API\_KEY", None\) if hasattr\(st\.secrets, "OPENAI\_API\_KEY"\) else None
if deepseek\_key\:
llm\_client \= ModelClient\(api\_key\=deepseek\_key, provider\="deepseek"\)
st\.sidebar\.caption\("✅ LLM\: DeepSeek Initialized"\)
elif openai\_key\:
llm\_client \= ModelClient\(api\_key\=openai\_key, provider\="openai"\)
st\.sidebar\.caption\("✅ LLM\: OpenAI Initialized"\)
else\:
st\.sidebar\.warning\("LLM API key missing\. Sentiment analysis disabled\."\)
except ValueError as e\: st\.sidebar\.error\(f"LLM Init Error\: \{e\}\. Check API Key\."\)
except Exception as e\: st\.sidebar\.error\(f"LLM Init Unexpected Error\: \{e\}"\)
\# \-\-\- Configuration Moved to Main Area \-\-\-
st\.header\("⚙️ Configuration"\)
config\_container \= st\.container\(border\=True\)
app\_mode \= "Live Analysis"
with config\_container\:
app\_mode \= st\.radio\("Select Mode\:", \["Live Analysis", "Backtesting"\], key\="app\_mode\_select\_main", horizontal\=True, index\=0\)
st\.markdown\("\-\-\-"\)
if app\_mode \=\= "Live Analysis"\:
st\.subheader\("Live Analysis Settings"\)
tickers\_in\_main \= st\.text\_input\("Tickers \(comma\-separated\)\:", "AAPL,MSFT,GOOG", key\="live\_tickers\_input\_main"\)
history\_years\_live\_main \= st\.slider\("Historical Data for Analysis \(Years\)\:", 1, 10, 5, key\="live\_history\_slider\_main"\)
st\.subheader\("Feature Toggles \(Live Analysis\)"\)
cols\_features \= st\.columns\(3\)
with cols\_features\[0\]\:
use\_sentiment\_live\_main \= st\.checkbox\("News Sentiment \(LLM\)", value\=True if llm\_client else False, disabled\=not llm\_client, key\="live\_sentiment\_cb\_main", help\="Uses LLM for news sentiment\. Requires API key\."\)
use\_filings\_live\_main \= st\.checkbox\("Insider Filings", value\=True, key\="live\_insider\_cb\_main", help\="Analyzes yfinance insider transaction data\."\)
with cols\_features\[1\]\:
use\_politician\_filings\_main \= st\.checkbox\("Politician Filings", value\=False, key\="live\_politician\_cb\_main", help\="EXPERIMENTAL\: Attempts to scrape CapitolTrades\.com\. May be slow/unreliable\."\)
use\_value\_trades\_main \= st\.checkbox\("Value\-Trades Fair Value", value\=False, key\="live\_vt\_cb\_main", help\="EXPERIMENTAL\: Scrapes public Value\-Trades\.com data\. CHECK ToS\!"\)
st\.markdown\(""\)
run\_button\_live\_main \= st\.button\("🚀 Run Live Analysis", use\_container\_width\=True, type\="primary", key\="run\_live\_btn\_main"\)
elif app\_mode \=\= "Backtesting"\:
st\.subheader\("Backtesting Settings"\)
bt\_ticker\_main \= st\.text\_input\("Ticker for Backtest\:", "AAPL", key\="bt\_ticker\_input\_main"\)\.upper\(\)
col1\_bt, col2\_bt \= st\.columns\(2\)
with col1\_bt\:
default\_bt\_end\_date\_main \= datetime\.now\(\) \- timedelta\(days\=1\)
default\_bt\_start\_date\_main \= default\_bt\_end\_date\_main \- pd\.DateOffset\(years\=3\)
bt\_start\_date\_main \= st\.date\_input\("Start Date\:", default\_bt\_start\_date\_main, max\_value\=default\_bt\_end\_date\_main \- timedelta\(days\=1\), key\="bt\_start\_date\_main"\)\.strftime\("%Y\-%m\-%d"\)
with col2\_bt\:
bt\_end\_date\_main \= st\.date\_input\("End Date\:", default\_bt\_end\_date\_main, min\_value\=datetime\.strptime\(bt\_start\_date\_main, "%Y\-%m\-%d"\) \+ timedelta\(days\=1\), key\="bt\_end\_date\_main"\)\.strftime\("%Y\-%m\-%d"\)
bt\_initial\_capital\_main \= st\.number\_input\("Initial Capital\:", 1000, 1000000, 10000, 1000, key\="bt\_capital\_input\_main", format\="%d"\)
with st\.expander\("Adjust Backtest Agent Weights \(Simplified Strategy\)", expanded\=False\)\:
st\.caption\("Backtest primarily uses Price, Momentum\. Volatility \(Beta part\) uses current data \(lookahead\)\. Other agents are off by default for backtesting due to point\-in\-time data challenges\."\)
bt\_weights\_price\_main \= st\.slider\("Price Signal Weight\:", 0\.0, 2\.0, 1\.0, 0\.1, key\="bt\_w\_price\_main"\)
bt\_weights\_momentum\_main \= st\.slider\("Momentum Signal Weight\:", 0\.0, 2\.0, 0\.8, 0\.1, key\="bt\_w\_momentum\_main"\)
bt\_weights\_volatility\_main \= st\.slider\("Volatility Signal Weight\:", 0\.0, 2\.0, 0\.2, 0\.1, key\="bt\_w\_vol\_main"\)
backtest\_portfolio\_weights\_main \= \{"price"\: bt\_weights\_price\_main, "momentum"\: bt\_weights\_momentum\_main, "volatility"\: bt\_weights\_volatility\_main,
"sentiment"\: 0\.0, "fund"\: 0\.0, "valuation\_dcf"\:0\.0, "valuation\_pe"\:0\.0,
"filings"\: 0\.0, "analyst"\: 0\.0, "politician\_filings"\: 0\.0, "vt\_fair\_value"\: 0\.0\}
st\.markdown\(""\)
run\_button\_backtest\_main \= st\.button\("📈 Run Backtest", use\_container\_width\=True, type\="primary", key\="run\_bt\_btn\_main"\)
\# Main App Logic \(Execution and Display of Results\)
st\.markdown\("\-\-\-"\)
if app\_mode \=\= "Live Analysis"\:
if 'run\_button\_live\_main' in locals\(\) and run\_button\_live\_main and 'tickers\_in\_main' in locals\(\) and tickers\_in\_main\:
live\_tickers\_list\_main \= \[t\.strip\(\)\.upper\(\) for t in tickers\_in\_main\.split\(","\) if t\.strip\(\)\]
if not live\_tickers\_list\_main\:
st\.error\("Please enter at least one valid ticker in the configuration above\."\)
else\:
live\_configs\_main \= \{"use\_sentiment"\: use\_sentiment\_live\_main, "use\_filings"\: use\_filings\_live\_main,
"use\_politician\_filings"\: use\_politician\_filings\_main,
"use\_value\_trades"\: use\_value\_trades\_main\}
if 'live\_output' not in st\.session\_state\: st\.session\_state\.live\_output \= \{\}
with st\.spinner\("⏳ Processing analysis\.\.\. Please wait\."\)\:
st\.session\_state\.live\_output \= run\_live\_analysis\(live\_tickers\_list\_main, history\_years\_live\_main, llm\_client, live\_configs\_main\)
st\.header\("📊 Live Analysis Summary"\)
num\_tickers \= len\(live\_tickers\_list\_main\)
cols\_per\_row \= min\(num\_tickers, 3\)
for i in range\(0, num\_tickers, cols\_per\_row\)\:
row\_tickers \= live\_tickers\_list\_main\[i\:i\+cols\_per\_row\]
cols \= st\.columns\(len\(row\_tickers\)\)
for idx, t\_symbol in enumerate\(row\_tickers\)\:
with cols\[idx\]\:
res \= st\.session\_state\.live\_output\.get\(t\_symbol\)
if not res or \("error" in res and res\["error"\] is not None\)\:
st\.error\(f"\*\*\{t\_symbol\}\*\*\: \{res\.get\('error', 'Unknown error'\) if res else 'No data'\}"\)
continue
dec \= res\.get\("final\_decision", "N/A"\)\.upper\(\); score \= res\.get\("composite\_score", float\('nan'\)\); price\_disp \= res\.get\("current\_price\_display"\)
card\_color\_map \= \{"BUY"\: "green", "SELL"\: "red", "HOLD"\: "\#FFA500"\}; card\_color \= card\_color\_map\.get\(dec, "\#D3D3D3"\)
st\.markdown\(f"""<div style\="border\: 1px solid \{card\_color\}; border\-radius\: 8px; padding\: 15px; margin\-bottom\: 10px; background\-color\: \{card\_color\}20;"\>
<h3 style\="margin\-bottom\: 5px; color\: \{card\_color\};"\>\{t\_symbol\}</h3\>
<p style\="font\-size\: 1\.6em; font\-weight\: bold; color\: \{card\_color\}; margin\-bottom\: 5px;"\>\{dec\}</p\>
<p style\="font\-size\: 0\.9em; margin\-bottom\: 3px;"\>Composite Score\: <strong style\="color\: \{card\_color\};"\>\{score\:\.2f\}</strong\></p\>
\{f'<p style\="font\-size\: 0\.9em;"\>Price\: <strong\></span>{price_disp:,.2f}</strong></p>' if price_disp is not None else ""}
                                    </div>""", unsafe_allow_html=True)
            st.markdown("---")
            for t_symbol in live_tickers_list_main:
                res = st.session_state.live_output.get(t_symbol)
                if not res or ("error" in res and res["error"] is not None): continue
                with st.expander(f"🔍 Detailed Analysis for {t_symbol}"):
                    tab_titles = ["📈 Chart & Core", "펀 Fundamentals", "💰 Valuation & Fair Value", "📰 News & Filings", "⚙️ All Signals"]
                    tabs = st.tabs(tab_titles)
                    with tabs[0]:
                        st.subheader("Price Performance & Core Signals")
                        price_history_df_display_exp = fetch_price_history(t_symbol, period=f"{history_years_live_main}y")
                        if not price_history_df_display_exp.empty: st.line_chart(price_history_df_display_exp["Close"], use_container_width=True)
                        core_s = {"Price Signal (SMA/RSI)": res.get("price_signal", "N/A").upper(), "SMA50 / SMA200": f"{res.get('sma50',0):.2f} / {res.get('sma200',0):.2f}", "RSI14": f"{res.get('rsi14',0):.2f}", "Momentum Signal (1M/12M)": res.get("momentum_signal", "N/A").upper(), "Momentum 1M / 12M (%)": f"{res.get('momentum_1m',0)*100:.1f}% / {res.get('momentum_12m',0)*100:.1f}%", "Volatility Signal (Beta)": res.get("volatility_signal", "N/A").upper(), "Beta / Annual Vol (%)": f"{res.get('beta',0):.2f} / {res.get('annual_vol',0)*100:.1f}%",}
                        st.dataframe(pd.Series(core_s, name="Value"), use_container_width=True)
                    with tabs[1]:
                        st.subheader(f"Fundamental Snapshot - {res.get('industry_display', 'N/A')}")
                        ticker_info_res = res.get("ticker_info", {})
                        fund_s = {"Market Cap": f"<span class="math-inline">\{res\.get\('market\_cap\_display',0\)\:,\}" if res\.get\('market\_cap\_display'\) else "N/A",
"FCF Yield"\: f"\{res\.get\('fcf\_yield',0\)\*100\:\.2f\}%",
"Piotroski Score"\: res\.get\('piotroski\_score'\),
"ROE / DebtToEquity"\: f"\{ticker\_info\_res\.get\('returnOnEquity',0\)\*100\:\.1f\}% / \{ticker\_info\_res\.get\('debtToEquity',0\)\:\.1f\}",
"Fundamental Signal"\: res\.get\("fund\_signal", "N/A"\)\.upper\(\)\}
st\.dataframe\(pd\.Series\(fund\_s, name\="Value"\), use\_container\_width\=True\)
business\_summary \= ticker\_info\_res\.get\("longBusinessSummary"\)
if business\_summary\:
with st\.popover\("View Business Summary"\)\:
st\.markdown\(business\_summary\)
with tabs\[2\]\:
st\.subheader\("Valuation Metrics"\)
val\_s \= \{"Forward P/E"\: f"\{res\.get\('forward\_pe',0\)\:\.1f\}", "Relative P/E Signal"\: res\.get\('relative\_pe\_signal', "N/A"\)\.upper\(\), "DCF Fair Price \(Simple Est\.\)"\: f"</span>{res.get('dcf_fair_price',0):.2f}" if res.get('dcf_fair_price') is not None else "N/A", "DCF Signal": res.get('dcf_signal', "N/A").upper()}
                        st.dataframe(pd.Series(val_s, name="Value"), use_container_width=True)
                        if live_configs_main["use_value_trades"]:
                            st.subheader("Value-Trades.com Analysis (Experimental)")
                            vt_scrape_status = res.get('vt_data_error') if res.get('vt_data_error') else "Success"
                            if res.get('vt_fair_value_estimate') is None and not res.get('vt_data_error'):
                                vt_scrape_status = "Data not found on page"
                            elif "VT Configuration incomplete" in str(vt_scrape_status) or "Skipped by user" in str(vt_scrape_status) :
                                vt_scrape_status = "Not Attempted (Check Config/Secrets)"

                            vt_full_text_display = res.get('vt_valuation_text_display')
                            if vt_full_text_display:
                                st.markdown(f"""> *"{vt_full_text_display}"*""")

                            vt_s = {
                                "VT Scraped Fair Value": f"<span class="math-inline">\{res\.get\('vt\_fair\_value\_estimate'\)\:\.2f\}" if res\.get\('vt\_fair\_value\_estimate'\) is not None else "N/A",
"VT Site Current Price"\: f"</span>{res.get('vt_site_current_price'):.2f}" if res.get('vt_site_current_price') is not None else "N/A",
                                "VT Site Valuation Status": str(res.get('vt_valuation_status', "N/A")).title(),
                                "VT Site Valuation (%)": f"{res.get('vt_valuation_percentage'):.2f}%" if res.get('vt_valuation_percentage') is not None else "N/A",
                                "VT Fair Value Signal": res.get('vt_fair_value_signal', "N/A").upper(),
                                "VT Scrape Status": vt_scrape_status
                            }
                            st.dataframe(pd.Series(vt_s, name="Value"), use_container_width=True)
                    with tabs[3]: # News & Filings Tab
                        if live_configs_main["use_sentiment"]:
                            st.subheader("News Sentiment (LLM)")
                            sent_error = res.get("sentiment_error")
                            sent_s = {"Sentiment Score": f"{res.get('sentiment_score',0):.2f}", "Sentiment Signal": res.get("sentiment_signal", "N/A").upper(), "LLM Status": "Error" if sent_error else "OK"}
                            st.dataframe(pd.Series(sent_s, name="Value"), use_container_width=True)
                            if sent_error: st.caption(f"LLM Error: {sent_error}")
                            news_headlines = res.get("news_headlines_for_popover")
                            if news_headlines:
                                with st.popover("View News Headlines"):
                                    for title in news_headlines: # Corrected Loop
                                        st.markdown(f"- {title}")
                        if live_configs_main["use_filings"]:
                            st.subheader("Insider Filings")
                            fil_s = {"Net Insider Shares (Recent)": f"{res.get('net_insider_shares',0):,}", "Insider Filings Signal": res.get("filings_signal", "N/A").upper()}
                            st.dataframe(pd.Series(fil_s, name="Value"), use_container_width=True)
                        if live_configs_main["use_politician_filings"]:
                            st.subheader("Politician Filings (Experimental Scrape)")
                            pol_scrape_error = res.get("politician_data_error")
                            pol_s = {"Net Trade Value Estimate (Recent)": f"${res.get('politician_net_trade_value_estimate',0):,}", "Buy/Sell Transactions": f"{res.get('politician_buy_tx_count',0)} / {res.get('politician_sell_tx_count',0)}", "Politician Filings Signal": res.get("politician_filings_signal", "N/A").upper(), "Scrape Status": "Error" if pol_scrape_error else "OK"}
                            st.dataframe(pd.Series(pol_s, name="Value"), use_container_width=True)
                            if pol_scrape_error: st.caption(f"Scraping Note: {pol_scrape_error}")
                            politician_trades_display = res.get("politician_trades_for_popover")
                            if politician_trades_display:
                                with st.popover("View Scraped Politician Trades (Max 5)"):
                                    for p_trade in politician_trades_display: # Corrected Loop
                                        st.markdown(f"**{p_trade.get('politician_name')}**: {p_trade.get('transaction_type')} ({p_trade.get('value_range')}) on {p_trade.get('date_str')}")
                    with tabs[4]:
                        st.subheader("All Agent Signals & Final Decision")
                        all_s_keys = [k for k in res if k.endswith("_signal")]; all_s_table = {k.replace("_signal","").replace("_"," ").title(): str(res[k]).upper() for k in all_s_keys}
                        all_s_table["Composite Score"] = f"{res.get('composite_score',0.0):.2f}"; all_s_table["Final Decision"] = res.get('final_decision',"").upper()
                        st.dataframe(pd.Series(all_s_table, name="Signal Value"), use_container_width=True)
                        with st.popover("View Full Raw JSON for this ticker"): st.json(res)

            with st.sidebar.expander("Portfolio Agent Weights (Live)", expanded=False):
                st.caption("These weights are used by the PortfolioAgent to combine individual signals.")
                st.json(PortfolioAgent.WEIGHTS)

elif app_mode == "Backtesting":
    if 'run_button_backtest_main' in locals() and run_button_backtest_main and 'bt_ticker_main' in locals() and bt_ticker_main:
        if 'bt_metrics' not in st.session_state: st.session_state.bt_metrics = None
        if 'bt_log_df' not in st.session_state: st.session_state.bt_log_df = pd.DataFrame()
        with st.spinner(f"⏳ Running backtest for {bt_ticker_main}... This may take a while."):
            st.session_state.bt_metrics, st.session_state.bt_log_df = run_backtest(
                bt_ticker_main, bt_start_date_main, bt_end_date_main,
                bt_initial_capital_main, llm_client, backtest_portfolio_weights_main)
        if st.session_state.bt_metrics and "message" not in st.session_state.bt_metrics and "error" not in st.session_state.bt_metrics:
            st.header(f"📈 Backtest Results for {bt_ticker_main}")
            metrics_df = pd.DataFrame.from_dict(st.session_state.bt_metrics, orient='index', columns=['Value'])
            st.table(metrics_df)
            if not st.session_state.bt_log_df.empty:
                st.subheader("Portfolio Value Over Time"); st.line_chart(st.session_state.bt_log_df["portfolio_value"])
                st.subheader("Drawdown Over Time"); st.area_chart(st.session_state.bt_log_df["drawdown"])
                with st.expander("View Backtest Log and Signals (Last 1000 rows)"): st.dataframe(st.session_state.bt_log_df[["price", "signal", "composite_score", "portfolio_value", "cash", "shares_held"]].tail(1000))
            else: st.warning("Backtest log empty, no charts.")
        else: st.error(f"Backtest failed: {st.session_state.bt_metrics.get('message', '') or st.session_state.bt_metrics.get('error', 'Unknown error') if st.session_state.bt_metrics else 'Unknown error'}")

# Sidebar for global info / disclaimers
st.sidebar.markdown("---")
st.sidebar.info("This simulator is for educational purposes only and does not constitute financial advice.")
st.sidebar.markdown("Experimental scraping features may be unreliable and are subject to website ToS.")
Remove the login details for the fair value section and make it for public. Is this something you can do with the whole code? Please provide the whole updated code.

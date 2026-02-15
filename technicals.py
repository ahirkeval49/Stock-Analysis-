import pandas_ta as ta
import pandas as pd

def calculate_technicals(df):
    if df is None or df.empty:
        return df
    
    # Simple Moving Averages
    df['SMA_50'] = ta.sma(df['close'], length=50)
    df['SMA_200'] = ta.sma(df['close'], length=200)
    
    # RSI (Relative Strength Index)
    df['RSI'] = ta.rsi(df['close'], length=14)
    
    # MACD (Moving Average Convergence Divergence)
    macd = ta.macd(df['close'])
    df = pd.concat([df, macd], axis=1)
    
    # Bollinger Bands
    bb = ta.bbands(df['close'], length=20)
    df = pd.concat([df, bb], axis=1)
    
    return df

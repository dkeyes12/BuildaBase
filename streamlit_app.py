import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
import plotly.express as px

# --- CONFIGURATION ---
st.set_page_config(page_title="OR-Tools Portfolio Optimizer", layout="wide")

# Default Universe (Starting point)
DEFAULT_TICKERS = [
    {"Ticker": "NVDA", "Sector": "Tech"}, {"Ticker": "AAPL", "Sector": "Tech"},
    {"Ticker": "JPM", "Sector": "Finance"}, {"Ticker": "XOM", "Sector": "Energy"},
    {"Ticker": "PG", "Sector": "Defensive"}, {"Ticker": "F", "Sector": "Industrial"}
]

# --- HELPER FUNCTIONS ---
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_data
def fetch_market_data(tickers, period="1y"):
    data = []
    
    # Extract just the symbol strings if tickers is a list of dicts, or use as is
    ticker_list = [t["Ticker"] for t in tickers] if isinstance(tickers[0], dict) else tickers
    
    with st.spinner(f"Fetching data for {len(ticker_list)} assets..."):
        for t in ticker_list:
            try:
                # Basic cleaning
                t = t.upper().strip()
                if not t: continue
                
                stock = yf.Ticker(t)
                hist = stock.history(period=period)
                
                if hist.empty:
                    continue

                # Metrics
                current_price = hist['Close'].iloc[-1]
                rsi = calculate_rsi(hist['Close']).iloc[-1]
                volatility = hist['Close'].pct_change().std() * np.sqrt(252)
                pct_return = (current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
                
                # Fetch PE
                pe = stock.info.get('trailingPE')
                if pe is None: 
                    pe = stock.info.get('forwardPE')
                
                # Filter: Exclude if no PE data
                if pe is not None and pe > 0:
                    data.append({
                        "Ticker": t,
                        "Price": current_price,
                        "PE": pe,
                        "RSI": rsi,
                        "Volatility": volatility,
                        "Return": pct_return
                    })
            except Exception as e:
                # st.warning(f"Could not fetch {t}") # Optional: show errors
                continue
                
    return pd.DataFrame(data)

# --- OPTIMIZATION ENGINE (OR-TOOLS) ---
def optimize_portfolio(df, objective_type, max_weight_per_asset):
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver: return None

    # Variables
    weights = []
    for i in range(len(df)):
        weights.append(solver.NumVar(0.0, max_weight_per_asset, f'w_{i}'))

    # Constraint: Total Weight == 100%
    constraint_sum = solver.Constraint(1.0, 1.0)
    for w in weights:
        constraint_sum.SetCoefficient(w, 1)

    # Objective
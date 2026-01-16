import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
import plotly.graph_objects as go # Changed to graph_objects for custom shapes

# --- CONFIGURATION ---
st.set_page_config(page_title="OR-Tools Portfolio Optimizer", layout="wide")

# Default Universe
DEFAULT_TICKERS = [
    {"Ticker": "NVDA", "Sector": "Tech"}, {"Ticker": "AAPL", "Sector": "Tech"},
    {"Ticker": "JPM", "Sector": "Finance"}, {"Ticker": "XOM", "Sector": "Energy"},
    {"Ticker": "PG", "Sector": "Defensive"}, {"Ticker": "F", "Sector": "Industrial"},
    {"Ticker": "TSLA", "Sector": "Auto"}, {"Ticker": "AMD", "Sector": "Tech"}
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
    ticker_list = [t["Ticker"] for t in tickers] if isinstance(tickers[0], dict) else tickers
    
    with st.spinner(f"Fetching data for {len(ticker_list)} assets..."):
        for t in ticker_list:
            try:
                t = t.upper().strip()
                if not t: continue
                
                stock = yf.Ticker(t)
                hist = stock.history(period=period)
                
                if hist.empty: continue

                current_price = hist['Close'].iloc[-1]
                rsi = calculate_rsi(hist['Close']).iloc[-1]
                volatility = hist['Close'].pct_change().std() * np.sqrt(252)
                
                pe = stock.info.get('trailingPE')
                if pe is None: pe = stock.info.get('forwardPE')
                
                if pe is not None and pe > 0:
                    data.append({
                        "Ticker": t,
                        "Price": current_price,
                        "PE": pe,
                        "RSI": rsi,
                        "Volatility": volatility
                    })
            except Exception:
                continue
                
    return pd.DataFrame(data)

# --- OPTIMIZATION ENGINE ---
def optimize_portfolio(df, objective_type, max_weight_per_asset):
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver: return None

    weights = []
    for i in range(len(df)):
        weights.append(solver.NumVar(0.0, max_weight_per_asset, f'w_{i}'))

    constraint_sum = solver.Constraint(1.0, 1.0)
    for w in weights:
        constraint_sum.SetCoefficient(w, 1)

    objective = solver.Objective()
    scores = (df['RSI'] / 100) + ((1 / df['PE']) * 50) 
    
    if objective_type == "Maximize Gain (Score)":
        for i, w in enumerate(weights):
            objective.SetCoefficient(w, scores.iloc[i])
        objective.SetMaximization()
    elif objective_type == "Minimize Loss (Volatility)":
        avg_score = scores.mean()
        constraint_quality = solver.Constraint(avg_score, solver.infinity())
        for i, w in enumerate(weights):
            constraint_quality.SetCoefficient(w, scores.iloc[i])
        for i, w in enumerate(weights):
            objective.SetCoefficient(w, df['Volatility'].iloc[i])
        objective.SetMinimization()

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        results = []
        for i, w in enumerate(weights):
            if w.solution_value() > 0.001: 
                results.append({
                    "Ticker": df['Ticker'].iloc[i],
                    "Weight": w.solution_value(),
                    "RSI": df['RSI'].iloc[i],
                    "PE": df['PE'].iloc[i],
                    "Volatility": df['Volatility'].iloc[i]
                })
        return pd.DataFrame(results)
    else:
        return pd.DataFrame()

# --- DASHBOARD UI ---
st.title("⚖️ Value-Momentum Portfolio Optimizer")

# 1. SIDEBAR
with st.sidebar:
    st.header("Settings")
    obj_choice = st.radio("Goal", ["Maximize Gain (Score)", "Minimize Loss (Volatility)"])
    max_concentration = st.slider("Max Allocation per Stock", 0.05, 1.0, 0.25, 0.05)
    st.info("Edit the table to define your stock universe.")

# 2. EDITABLE STOCK INPUT
st.subheader("1. Define Stock Universe")
col_input, col_action = st.columns([3, 1])

with col_input:
    if "user_tickers" not in st.session_state:
        st.session_state["user_tickers"] = pd.DataFrame(DEFAULT_TICKERS)

    # UPDATED: Column config for width control
    edited_df = st.data_editor(
        st.session_state["user_tickers"], 
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small", help="Stock Symbol"),
            "Sector": st.column_config.TextColumn("Sector", width="medium"),
        },
        num_rows="dynamic", 
        use_container_width=True,
        key="ticker_editor"
    )

with col_action:
    st.write("### ") 
    run_optimization = st.button("🚀 Run Optimization", type="primary", use_container_width=True)

# 3. EXECUTION
if run_optimization:
    ticker_list = edited_df["Ticker"].tolist()
    
    if len(ticker_list) < 2:
        st.error("Please enter at least 2 tickers.")
    else:
        df_market = fetch_market_data(ticker_list)

        if not df_market.empty:
            st.divider()
            
            with st.spinner("Optimizing..."):
                df_opt = optimize_portfolio(df_market, obj_choice, max_concentration)

            if not df_opt.empty:
                st.subheader(f"2. Optimal Allocation ({obj_choice})")
                
                # --- RESULTS TABLE ---
                # Simple display to avoid matplotlib/styling errors
                display_df = df_opt[["Ticker", "Weight", "PE", "RSI"]].copy()
                display_df["Weight"] = display_df["Weight"].apply(lambda x: f"{x:.1%}")
                display_df["PE"] = display_df["PE"].apply(lambda x: f"{x:.1f}")
                display_df["RSI"] = display_df["RSI"].apply(lambda x: f"{x:.1f}")
                st.dataframe(display_df, use_container_width=True)

                # --- VISUALIZATION: COLORED QUADRANTS ---
                st.subheader("3. Portfolio Analysis (Value vs Momentum)")
                
                # Parameters for plotting
                PE_THRESHOLD = 25  
                RSI_THRESHOLD = 50
                max_pe_in_data = df_market['PE'].max()
                max_x = max(50, max_pe_in_data * 1.1)

                fig_quad = go.Figure()

                # 1. Plot ALL Stocks (Gray/Ghost)
                fig_quad.add_trace(go.Scatter(
                    x=df_market['PE'], y=df_market['RSI'],
                    mode='markers+text',
                    text=df_market['Ticker'],
                    textposition="top center",
                    marker=dict(size=10, color='gray', opacity=0.5),
                    name='Universe'
                ))

                # 2. Plot SELECTED Portfolio Stocks (Blue/Highlighted)
                fig_quad.add_trace(go.Scatter(
                    x=df_opt['PE'], y=df_opt['RSI'],
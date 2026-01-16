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
    objective = solver.Objective()
    
    # Score = (RSI / 100) + (1 / PE * 50) 
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
st.title("⚖️ Custom Portfolio Optimizer")

# 1. SIDEBAR CONFIGURATION
with st.sidebar:
    st.header("Optimization Settings")
    obj_choice = st.radio("Goal", ["Maximize Gain (Score)", "Minimize Loss (Volatility)"])
    max_concentration = st.slider("Max Allocation per Stock", 0.05, 1.0, 0.25, 0.05)
    st.markdown("---")
    st.info("Edit the table on the right to define your stock universe.")

# 2. EDITABLE STOCK INPUT
st.subheader("1. Define Stock Universe")
col_input, col_action = st.columns([3, 1])

with col_input:
    # Initialize session state for tickers if not present
    if "user_tickers" not in st.session_state:
        st.session_state["user_tickers"] = pd.DataFrame(DEFAULT_TICKERS)

    # The Data Editor allows adding/deleting rows
    edited_df = st.data_editor(
        st.session_state["user_tickers"], 
        num_rows="dynamic", 
        use_container_width=True,
        key="ticker_editor"
    )

with col_action:
    st.write("### ") # Spacing
    run_optimization = st.button("🚀 Run Optimization", type="primary", use_container_width=True)

# 3. EXECUTION
if run_optimization:
    # Get list of tickers from the edited dataframe
    ticker_list = edited_df["Ticker"].tolist()
    
    if len(ticker_list) < 2:
        st.error("Please enter at least 2 tickers to optimize.")
    else:
        df_market = fetch_market_data(ticker_list)

        if not df_market.empty:
            st.divider()
            
            # --- MARKET ANALYSIS ---
            st.subheader("2. Market Data Analysis")
            st.dataframe(
                df_market.style.format({
                    "PE": "{:.2f}", 
                    "RSI": "{:.2f}", 
                    "Return": "{:.2%}", 
                    "Volatility": "{:.2%}"
                }), use_container_width=True
            )

            # --- OPTIMIZATION ---
            with st.spinner("Calculating optimal weights using OR-Tools..."):
                df_opt = optimize_portfolio(df_market, obj_choice, max_concentration)

            if not df_opt.empty:
                st.subheader(f"3. Optimal Portfolio ({obj_choice})")
                
                res_col1, res_col2 = st.columns([1, 1])
                
                with res_col1:
                    # Sort and display allocation
                    df_opt = df_opt.sort_values(by="Weight", ascending=False)
                    
                    # Create a nice summary table
                    st.dataframe(
                        df_opt[["Ticker", "Weight", "PE", "RSI"]].style.format({
                            "Weight": "{:.1%}", 
                            "PE": "{:.1f}", 
                            "RSI": "{:.1f}"
                        }).background_gradient(subset=['Weight'], cmap='Greens'), 
                        use_container_width=True
                    )

                with res_col2:
                    # Allocation Pie Chart
                    fig = px.pie(df_opt, values='Weight', names='Ticker', hole=0.4)
                    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)
                
                # Visual Scatter Plot
                fig_scatter = px.scatter(df_market, x="PE", y="RSI", text="Ticker", 
                                        color="Ticker", size_max=40)
                
                # Highlight Selected Stocks
                fig_scatter.add_trace(px.scatter(df_opt, x="PE", y="RSI").data[0])
                fig_scatter.update_traces(marker=dict(size=12), textposition='top center')
                
                # Draw Ideal Zone
                fig_scatter.add_shape(type="rect", x0=0, y0=50, x1=25, y1=100,
                                     fillcolor="green", opacity=0.1, layer="below", line_width=0)
                
                fig_scatter.update_layout(
                    title="Portfolio Selection Map (Green Zone = Ideal)",
                    xaxis_title="P/E Ratio (Lower is Better)",
                    yaxis_title="RSI Momentum (Higher is Better)",
                    showlegend=False
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            else:
                st.error("No valid solution found. Try increasing the 'Max Allocation' slider.")
        else:
            st.error("Could not fetch data for the provided tickers. Please check spelling.")
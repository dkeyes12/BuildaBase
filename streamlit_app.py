import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
import plotly.express as px

# --- CONFIGURATION ---
st.set_page_config(page_title="OR-Tools Portfolio Optimizer", layout="wide")

# Universe of assets (Mix of Value, Growth, and Defensive for demonstration)
TICKERS = [
    "NVDA", "AAPL", "GOOGL", "MSFT", "AMZN", "META", # Tech
    "JPM", "BAC", "V", "MA",                         # Finance
    "XOM", "CVX", "COP",                             # Energy
    "PG", "KO", "PEP", "JNJ",                        # Defensive
    "F", "GM"                                        # Industrial/Value
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
    
    with st.spinner(f"Fetching data for {len(tickers)} assets..."):
        for t in tickers:
            try:
                stock = yf.Ticker(t)
                hist = stock.history(period=period)
                
                if hist.empty:
                    continue

                # Metrics
                current_price = hist['Close'].iloc[-1]
                rsi = calculate_rsi(hist['Close']).iloc[-1]
                volatility = hist['Close'].pct_change().std() * np.sqrt(252) # Annualized Vol
                pct_return = (current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
                
                # Fetch PE (Handling missing data with a fallback)
                pe = stock.info.get('trailingPE')
                if pe is None: 
                    pe = stock.info.get('forwardPE')
                
                # Filter: Exclude if no PE data (cannot evaluate Value)
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
                continue
                
    return pd.DataFrame(data)

# --- OPTIMIZATION ENGINE (OR-TOOLS) ---
def optimize_portfolio(df, objective_type, max_weight_per_asset):
    # 1. Create the linear solver with the GLOP backend.
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return None

    # 2. Define Variables
    # w[i] is the weight of asset i in the portfolio (0.0 to max_weight)
    weights = []
    for i in range(len(df)):
        weights.append(solver.NumVar(0.0, max_weight_per_asset, f'w_{i}'))

    # 3. Define Constraints
    
    # Constraint A: Total Weight must equal 100% (1.0)
    # sum(weights) == 1
    constraint_sum = solver.Constraint(1.0, 1.0)
    for w in weights:
        constraint_sum.SetCoefficient(w, 1)

    # 4. Define Objective Function
    objective = solver.Objective()
    
    # SCORING LOGIC:
    # We create a composite score favoring Low PE and High RSI.
    # Score = (RSI / 100) + (1 / PE * 100)  <-- Higher is better
    # Note: 1/PE is "Earnings Yield". We multiply by 100 to normalize scale with RSI.
    scores = (df['RSI'] / 100) + ((1 / df['PE']) * 50) 
    
    if objective_type == "Maximize Gain (Score)":
        # Maximize the weighted sum of our Custom Value/Momentum Score
        for i, w in enumerate(weights):
            objective.SetCoefficient(w, scores.iloc[i])
        objective.SetMaximization()
        
    elif objective_type == "Minimize Loss (Volatility)":
        # Minimize weighted Volatility
        # But we must add a constraint that the Portfolio Score is at least average
        # otherwise it will just pick the lowest volatility stock regardless of PE/RSI.
        avg_score = scores.mean()
        constraint_quality = solver.Constraint(avg_score, solver.infinity())
        for i, w in enumerate(weights):
            constraint_quality.SetCoefficient(w, scores.iloc[i])
            
        for i, w in enumerate(weights):
            objective.SetCoefficient(w, df['Volatility'].iloc[i])
        objective.SetMinimization()

    # 5. Solve
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        results = []
        for i, w in enumerate(weights):
            if w.solution_value() > 0.001: # Filter out near-zero weights
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
st.title("⚖️ Optimal Value-Momentum Portfolio")
st.markdown("Generates a portfolio maximizing **Low PE & High RSI** traits using **Google OR-Tools**.")

# Sidebar Controls
obj_choice = st.sidebar.radio("Optimization Goal", ["Maximize Gain (Score)", "Minimize Loss (Volatility)"])
max_concentration = st.sidebar.slider("Max Weight per Asset", 0.05, 1.0, 0.20, 0.05)
st.sidebar.info("The 'Score' is a composite of Earnings Yield (Low PE) and RSI Momentum.")

# Main Execution
df_market = fetch_market_data(TICKERS)

if not df_market.empty:
    # Show Universe Stats
    st.subheader("1. Asset Universe Analysis")
    st.dataframe(df_market.style.format({"PE": "{:.2f}", "RSI": "{:.2f}", "Return": "{:.2%}", "Volatility": "{:.2%}"}))

    # Run Optimization
    st.subheader("2. OR-Tools Optimization Result")
    
    # 
    # The solver finds the point on the efficient frontier that satisfies our constraints.
    
    with st.spinner("Solving linear constraints..."):
        df_opt = optimize_portfolio(df_market, obj_choice, max_concentration)

    if not df_opt.empty:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.success(f"**Optimal Solution Found!**")
            st.write("Allocations:")
            # Sort by weight
            df_opt = df_opt.sort_values(by="Weight", ascending=False)
            
            # Display list
            for _, row in df_opt.iterrows():
                st.write(f"**{row['Ticker']}**: {row['Weight']:.1%} (PE: {row['PE']:.1f}, RSI: {row['RSI']:.1f})")

        with col2:
            # 
            fig = px.pie(df_opt, values='Weight', names='Ticker', title=f"Optimal Allocation: {obj_choice}", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter of Portfolio vs Universe
            fig_scatter = px.scatter(df_market, x="PE", y="RSI", hover_name="Ticker", color="Ticker", size_max=10)
            
            # Highlight chosen
            fig_scatter.add_scatter(x=df_opt['PE'], y=df_opt['RSI'], mode='markers', 
                                   marker=dict(size=15, color='black', symbol='circle-open', line=dict(width=2)),
                                   name="Selected")
            
            fig_scatter.update_layout(title="Selected Assets in PE/RSI Space (Top Left is Best)")
            fig_scatter.add_vline(x=20, line_dash="dash", line_color="green", annotation_text="Value Zone")
            fig_scatter.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Momentum Zone")
            st.plotly_chart(fig_scatter, use_container_width=True)

    else:
        st.error("No optimal solution found. Try relaxing the constraints (increase Max Weight).")
else:
    st.warning("Failed to fetch market data.")
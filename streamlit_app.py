# 3. EXECUTION
if run_optimization:
    # Get list of tickers from the edited dataframe
    ticker_list = edited_df["Ticker"].tolist()
    
    if len(ticker_list) < 2:
        st.error("Please enter at least 2 tickers.")
    else:
        df_market = fetch_market_data(ticker_list)

        if not df_market.empty:
            st.divider()
            
            # Run Optimization
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
                # Ensure this line is closed properly:
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
                    mode='markers+text',
                    text=df_opt['Ticker'],
                    textposition="top center",
                    marker=dict(size=18, color='blue', line=dict(width=2, color='white')),
                    name='Selected Portfolio'
                ))

                # 3. Add Colored Quadrant Backgrounds
                # Q1: Top Left (Value + Momentum) - Green
                fig_quad.add_shape(type="rect", x0=0, y0=RSI_THRESHOLD, x1=PE_THRESHOLD, y1=100,
                                   fillcolor="green", opacity=0.1, layer="below", line_width=0)
                
                # Q2: Top Right (Growth/Expensive) - Yellow
                fig_quad.add_shape(type="rect", x0=PE_THRESHOLD, y0=RSI_THRESHOLD, x1=max_x, y1=100,
                                   fillcolor="yellow", opacity=0.1, layer="below", line_width=0)

                # Q3: Bottom Left (Value Trap/Weak) - Yellow
                fig_quad.add_shape(type="rect", x0=0, y0=0, x1=PE_THRESHOLD, y1=RSI_THRESHOLD,
                                   fillcolor="yellow", opacity=0.1, layer="below", line_width=0)
                
                # Q4: Bottom Right (Expensive & Weak) - Red
                fig_quad.add_shape(type="rect", x0=PE_THRESHOLD, y0=0, x1=max_x, y1=RSI_THRESHOLD,
                                   fillcolor="red", opacity=0.1, layer="below", line_width=0)

                # Add Crosshair Lines
                fig_quad.add_vline(x=PE_THRESHOLD, line_width=1, line_dash="dash", line_color="gray")
                fig_quad.add_hline(y=RSI_THRESHOLD, line_width=1, line_dash="dash", line_color="gray")

                # Quadrant Labels
                fig_quad.add_annotation(x=PE_THRESHOLD/2, y=90, text="VALUE + MOMENTUM", showarrow=False, font=dict(color="green", size=14, weight="bold"))
                fig_quad.add_annotation(x=PE_THRESHOLD*1.2, y=90, text="EXPENSIVE MOMENTUM", showarrow=False, font=dict(color="orange", size=10))
                fig_quad.add_annotation(x=PE_THRESHOLD/2, y=10, text="WEAK / VALUE TRAP", showarrow=False, font=dict(color="orange", size=10))
                fig_quad.add_annotation(x=PE_THRESHOLD*1.2, y=10, text="EXPENSIVE & WEAK", showarrow=False, font=dict(color="red", size=14, weight="bold"))

                fig_quad.update_xaxes(title_text="P/E Ratio (Value)", range=[0, max_x])
                fig_quad.update_yaxes(title_text="RSI (Momentum)", range=[0, 100])
                fig_quad.update_layout(height=600, title="Market Universe & Selection")

                st.plotly_chart(fig_quad, use_container_width=True)

            else:
                st.error("No optimal solution found. Try relaxing the constraints (increase Max Weight).")
        else:
            st.error("Could not fetch data.")
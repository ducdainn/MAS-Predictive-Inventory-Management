"""
Forecast component - Demand prediction and forecasting
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

def render(orchestrator):
    """
    Render forecast page
    
    Args:
        orchestrator: OrchestratorAgent instance
    """
    st.title("🔮 Demand Forecasting")
    st.markdown("AI-powered demand prediction for inventory planning")
    
    # Input section
    st.subheader("📝 Forecast Configuration")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        template = st.selectbox(
            "Quick Templates:",
            [
                "Custom Question",
                "Dự báo doanh số 30 ngày tới",
                "Dự báo nhu cầu gạch 30x60",
                "Dự báo doanh số chi nhánh Đà Nẵng",
                "Dự báo nhu cầu tổng thể"
            ]
        )
        
        if template == "Custom Question":
            question = st.text_input(
                "Your Question:",
                placeholder="e.g., Dự báo doanh số 60 ngày tới",
                help="Ask about demand forecasting"
            )
        else:
            question = template
            st.text_input("Your Question:", value=question, disabled=True)
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        horizon = st.number_input(
            "Horizon (days)",
            min_value=7,
            max_value=90,
            value=30,
            help="Number of days to forecast"
        )
    
    run_button = st.button(
        "🚀 Generate Forecast",
        type="primary",
        use_container_width=True,
        disabled=not question
    )
    
    st.markdown("---")
    
    # Run forecast
    if run_button and question:
        with st.spinner("🔮 Generating forecast..."):
            try:
                result = orchestrator.process_query(question, forced_intent="FORECAST")
                
                st.session_state.last_forecast_result = result
                st.success(f"✅ Forecast complete in {result.get('elapsed_seconds', 0):.2f}s")
                
            except Exception as e:
                st.error(f"❌ Forecast failed: {e}")
                st.exception(e)
                return
    
    # Display results
    if st.session_state.get('last_forecast_result'):
        result = st.session_state.last_forecast_result
        
        # Debug: Show what we received
        intent = result.get('intent', 'UNKNOWN')
        success = result.get('success', False)
        
        # Check if this is a forecast result
        if success and intent == 'FORECAST':
            display_forecast_results(result)
        elif intent == 'FORECAST' and not success:
            # Forecast was attempted but failed
            st.error("❌ Forecast generation failed")
            
            # Get error details
            error_msg = result.get('error', 'Unknown error')
            nested_result = result.get('result', {})
            nested_error = nested_result.get('error', '') if isinstance(nested_result, dict) else ''
            
            # Show error message
            st.warning(f"**Error:** {error_msg or nested_error or 'No data available for forecasting'}")
            
            # Troubleshooting tips
            with st.expander("💡 Troubleshooting Tips"):
                st.markdown("""
                **Possible causes:**
                1. **No historical data** - The query returned no sales data
                2. **Insufficient data** - Need at least 2 days of data for forecasting
                3. **SQL error** - The generated query may have issues
                4. **Branch/Product not found** - Check if the branch/product exists
                
                **Solutions:**
                - Try a broader query (e.g., "Dự báo doanh số tổng thể")
                - Check if the branch/product has recent sales data
                - Try a different time period or product
                """)
            
            # Debug info
            with st.expander("🔍 Debug Info"):
                st.write(f"**Intent:** `{intent}`")
                st.write(f"**Success:** `{success}`")
                st.write(f"**SQL Query:**")
                st.code(result.get('sql', 'N/A'), language='sql')
                if nested_result:
                    st.write(f"**Result Details:**")
                    st.json(nested_result)
        else:
            # Wrong intent
            st.warning("⚠️ Last result is not a forecast. Please run a new forecast.")
            with st.expander("🔍 Debug Info"):
                st.write(f"**Intent detected:** `{intent}`")
                st.write(f"**Success status:** `{success}`")
                st.write(f"**Expected:** Intent = 'FORECAST' and Success = True")
                if intent != 'FORECAST':
                    st.info(f"💡 This query was classified as **{intent}**. To see forecast results, please ask a forecasting question like 'Dự báo doanh số 30 ngày tới'.")


def display_forecast_results(result):
    """Display forecast results"""
    
    data = result['result']
    
    # Metrics
    if data.get('metrics'):
        st.subheader("📊 Forecast Metrics")
        
        metrics = data['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Recent Avg (Daily)",
                f"{metrics['recent_avg_daily']:.1f}",
                help="Average daily sales (recent period)"
            )
        
        with col2:
            st.metric(
                "Forecast Avg (Daily)",
                f"{metrics['forecast_avg_daily']:.1f}",
                delta=f"{((metrics['forecast_avg_daily'] / max(metrics['recent_avg_daily'], 0.1) - 1) * 100):.1f}%",
                help="Predicted average daily sales"
            )
        
        with col3:
            st.metric(
                "Total Forecast",
                f"{metrics['forecast_total']:.0f}",
                help="Total predicted sales"
            )
        
        with col4:
            trend = metrics['trend']
            trend_icon = "📈" if trend == "increasing" else "📉" if trend == "decreasing" else "➡️"
            st.metric(
                "Trend",
                f"{trend_icon} {trend.title()}",
                help="Demand trend direction"
            )
    
    st.markdown("---")
    
    # Chart
    if data.get('chart'):
        st.subheader("📈 Forecast Visualization")
        
        try:
            img = Image.open(data['chart'])
            st.image(img, use_container_width=True, caption="Demand Forecast")
        except Exception as e:
            st.warning(f"Could not load chart: {e}")
    
    # Interactive Plotly chart
    if data.get('historical_data') is not None and data.get('forecast') is not None:
        st.markdown("---")
        st.subheader("📊 Interactive Forecast")
        
        historical = data['historical_data']
        forecast = data['forecast']
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical.index,
            y=historical['value'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='steelblue', width=2),
            marker=dict(size=4)
        ))
        
        # Forecast data
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast['forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='orange', width=2, dash='dash'),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title='Historical vs Forecast',
            xaxis_title='Date',
            yaxis_title='Quantity',
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary
    if data.get('summary'):
        st.markdown("---")
        st.subheader("📝 Forecast Summary")
        st.text(data['summary'])
    
    # Download
    if data.get('forecast') is not None:
        st.markdown("---")
        forecast_df = data['forecast'].reset_index()
        csv = forecast_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name="forecast_result.csv",
            mime="text/csv"
        )


"""
Analytics component - Interactive data analysis and visualization
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

def render(orchestrator):
    """
    Render analytics page
    
    Args:
        orchestrator: OrchestratorAgent instance
    """
    st.title("📊 Analytics & Insights")
    st.markdown("Interactive data analysis and visualization")
    
    # Input section
    st.subheader("📝 Ask Your Question")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        template = st.selectbox(
            "Quick Templates:",
            [
                "Custom Question",
                "Top 10 sản phẩm bán chạy nhất tháng này",
                "Phân tích doanh số theo vùng miền",
                "Thống kê tồn kho theo chi nhánh",
                "Top chi nhánh có doanh thu cao nhất"
            ]
        )
        
        if template == "Custom Question":
            question = st.text_input(
                "Your Question:",
                placeholder="e.g., Show sales by product category",
                help="Ask about sales, inventory, trends, distributions"
            )
        else:
            question = template
            st.text_input("Your Question:", value=question, disabled=True)
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_button = st.button(
            "🚀 Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=not question
        )
    
    st.markdown("---")
    
    # Run analytics
    if run_button and question:
        with st.spinner("🤖 Analyzing data..."):
            try:
                result = orchestrator.process_query(question)
                
                st.session_state.last_analytics_result = result
                st.success(f"✅ Analysis complete in {result.get('elapsed_seconds', 0):.2f}s")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                st.exception(e)
                return
    
    # Display results
    if st.session_state.get('last_analytics_result'):
        result = st.session_state.last_analytics_result
        
        if result.get('success'):
            display_analytics_results(result)
        else:
            st.error("Analysis failed. Please try a different question.")


def display_analytics_results(result):
    """Display analytics results"""
    
    data = result['result']
    
    # Show data
    if data.get('data') is not None and not data['data'].empty:
        st.subheader("📋 Data")
        
        df = data['data']
        
        # Data summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            if len(df.select_dtypes(include='number').columns) > 0:
                numeric_col = df.select_dtypes(include='number').columns[0]
                st.metric("Total", f"{df[numeric_col].sum():,.0f}")
        
        # Show data
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="analytics_result.csv",
            mime="text/csv"
        )
    
    # Show chart
    if data.get('charts') and len(data['charts']) > 0:
        st.markdown("---")
        st.subheader("📈 Visualization")
        
        for chart_path in data['charts']:
            try:
                img = Image.open(chart_path)
                st.image(img, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load chart: {e}")
    
    # Show summary
    if data.get('summary'):
        st.markdown("---")
        st.subheader("📝 Summary")
        st.text(data['summary'])


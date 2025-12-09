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
                result = orchestrator.process_query(question, forced_intent="ANALYTICS")
                
                st.session_state.last_analytics_result = result
                st.success(f"✅ Analysis complete in {result.get('elapsed_seconds', 0):.2f}s")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                st.exception(e)
                return
    
    # Display results
    if st.session_state.get('last_analytics_result'):
        result = st.session_state.last_analytics_result
        
        if result.get('success') and result.get('result', {}).get('success', True):
            display_analytics_results(result)
        else:
            # Show detailed error message
            error_msg = "Analysis failed. Please try a different question."
            
            if result.get('error'):
                error_msg = f"Error: {result.get('error')}"
            elif result.get('result', {}).get('message'):
                error_msg = result['result']['message']
            elif result.get('result', {}).get('error'):
                error_msg = f"Error: {result['result']['error']}"
            
            st.error(f"❌ {error_msg}")
            
            # Show SQL for debugging
            if result.get('sql'):
                with st.expander("🔍 Debug: Generated SQL"):
                    st.code(result['sql'], language='sql')


def display_analytics_results(result):
    """Display analytics results with interactive Plotly charts"""
    
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
        
        # Interactive Visualization
        st.markdown("---")
        st.subheader("📈 Interactive Visualization")
        
        _render_interactive_chart(df)
    
    # Show summary
    if data.get('summary'):
        st.markdown("---")
        st.subheader("📝 Summary")
        st.text(data['summary'])


def _render_interactive_chart(df: pd.DataFrame):
    """Render interactive Plotly chart based on data structure"""
    import plotly.graph_objects as go
    
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = []
    
    # Detect date columns
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                date_cols.append(col)
                if col in categorical_cols:
                    categorical_cols.remove(col)
            except:
                pass
    
    if not numeric_cols:
        st.info("No numeric columns to visualize")
        return
    
    # Chart type selection
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        chart_type = st.selectbox(
            "Chart Type",
            ["Bar", "Line", "Area", "Pie", "Scatter", "Horizontal Bar"],
            key="analytics_chart_type"
        )
    
    with col2:
        y_col = st.selectbox(
            "Value (Y-axis)",
            numeric_cols,
            key="analytics_y_col"
        )
    
    with col3:
        x_options = categorical_cols + date_cols + ["Index"]
        x_col = st.selectbox(
            "Category (X-axis)",
            x_options if x_options else ["Index"],
            key="analytics_x_col"
        )
    
    with col4:
        sort_option = st.selectbox(
            "Sort By",
            ["Value (Desc)", "Value (Asc)", "Category (A-Z)", "Category (Z-A)", "None"],
            index=0,  # Default: Value (Desc)
            key="analytics_sort"
        )
    
    # Prepare data
    if x_col == "Index":
        plot_df = df.reset_index()
        x_col = "index"
    else:
        plot_df = df.copy()
    
    # Apply sorting
    if sort_option == "Value (Desc)":
        plot_df = plot_df.sort_values(y_col, ascending=False)
    elif sort_option == "Value (Asc)":
        plot_df = plot_df.sort_values(y_col, ascending=True)
    elif sort_option == "Category (A-Z)":
        plot_df = plot_df.sort_values(x_col, ascending=True)
    elif sort_option == "Category (Z-A)":
        plot_df = plot_df.sort_values(x_col, ascending=False)
    
    # Limit data for better visualization
    if len(plot_df) > 50 and chart_type in ["Bar", "Horizontal Bar", "Pie"]:
        plot_df = plot_df.head(20)
        st.caption("📌 Showing top 20 items")
    
    # Create chart
    try:
        if chart_type == "Bar":
            fig = px.bar(
                plot_df, x=x_col, y=y_col,
                color=y_col,
                color_continuous_scale="Blues",
                title=f"{y_col} by {x_col}"
            )
            fig.update_layout(showlegend=False)
            
        elif chart_type == "Horizontal Bar":
            fig = px.bar(
                plot_df, x=y_col, y=x_col,
                orientation='h',
                color=y_col,
                color_continuous_scale="Blues",
                title=f"{y_col} by {x_col}"
            )
            # Preserve sort order from dataframe
            fig.update_layout(showlegend=False, yaxis={'categoryorder': 'trace'})
            
        elif chart_type == "Line":
            fig = px.line(
                plot_df, x=x_col, y=y_col,
                markers=True,
                title=f"{y_col} over {x_col}"
            )
            
        elif chart_type == "Area":
            fig = px.area(
                plot_df, x=x_col, y=y_col,
                title=f"{y_col} over {x_col}"
            )
            
        elif chart_type == "Pie":
            fig = px.pie(
                plot_df, values=y_col, names=x_col,
                title=f"Distribution of {y_col}",
                hole=0.3
            )
            
        elif chart_type == "Scatter":
            if len(numeric_cols) >= 2:
                x_num = st.selectbox("X Numeric", numeric_cols, key="scatter_x")
                y_num = st.selectbox("Y Numeric", [c for c in numeric_cols if c != x_num], key="scatter_y")
                fig = px.scatter(
                    plot_df, x=x_num, y=y_num,
                    color=categorical_cols[0] if categorical_cols else None,
                    title=f"{y_num} vs {x_num}"
                )
            else:
                fig = px.scatter(
                    plot_df, x=x_col, y=y_col,
                    title=f"{y_col} vs {x_col}"
                )
        
        # Common layout updates
        fig.update_layout(
            template="plotly_white",
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            font=dict(family="Arial", size=12),
            title_font_size=16,
            title_x=0.5
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Could not create chart: {e}")


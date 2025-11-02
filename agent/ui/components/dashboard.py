"""
Dashboard component - Main overview page
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def render(orchestrator):
    """
    Render main dashboard with overview metrics
    
    Args:
        orchestrator: OrchestratorAgent instance
    """
    st.title("📊 Dashboard Overview")
    st.markdown("Real-time inventory and sales insights")
    
    # Get database stats
    try:
        db = orchestrator.db_manager
        
        # Row 1: Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Total branches
            branches_df = db.execute_query("SELECT COUNT(DISTINCT branch_code) as count FROM branch")
            total_branches = int(branches_df['count'].iloc[0])
            st.metric(
                label="🏢 Total Branches",
                value=total_branches,
                help="Number of warehouse branches"
            )
        
        with col2:
            # Total products
            products_df = db.execute_query("SELECT COUNT(DISTINCT product_code) as count FROM product")
            total_products = int(products_df['count'].iloc[0])
            st.metric(
                label="📦 Total Products",
                value=f"{total_products:,}",
                help="Number of unique products"
            )
        
        with col3:
            # Total inventory
            inventory_df = db.execute_query("SELECT SUM(quantity) as total FROM inventory")
            total_inventory = int(inventory_df['total'].iloc[0])
            st.metric(
                label="📊 Total Stock",
                value=f"{total_inventory:,}",
                help="Total inventory quantity"
            )
        
        with col4:
            # Sales today
            sales_df = db.execute_query("""
                SELECT COALESCE(SUM(quantity), 0) as total 
                FROM sales 
                WHERE date = CURRENT_DATE
            """)
            sales_today = int(sales_df['total'].iloc[0]) if not sales_df.empty else 0
            st.metric(
                label="💰 Sales Today",
                value=f"{sales_today:,}",
                help="Total sales quantity today"
            )
        
        st.markdown("---")
        
        # Row 2: Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Sales Trend (Last 30 Days)")
            
            sales_trend_df = db.execute_query("""
                SELECT date, SUM(quantity) as total_sales
                FROM sales
                WHERE date >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY date
                ORDER BY date
            """)
            
            if not sales_trend_df.empty:
                fig = px.line(
                    sales_trend_df,
                    x='date',
                    y='total_sales',
                    title='Daily Sales Volume',
                    labels={'total_sales': 'Quantity', 'date': 'Date'}
                )
                fig.update_traces(line_color='#1f77b4', line_width=3)
                fig.update_layout(
                    height=350,
                    showlegend=False,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sales data available for the last 30 days")
        
        with col2:
            st.subheader("🏆 Top 10 Products (This Month)")
            
            top_products_df = db.execute_query("""
                SELECT 
                    p.product_name,
                    SUM(s.quantity) as total_sold
                FROM sales s
                JOIN product p ON s.product_code = p.product_code
                WHERE s.date >= DATE_TRUNC('month', CURRENT_DATE)
                GROUP BY p.product_name
                ORDER BY total_sold DESC
                LIMIT 10
            """)
            
            if not top_products_df.empty:
                fig = px.bar(
                    top_products_df,
                    x='total_sold',
                    y='product_name',
                    orientation='h',
                    title='Best Selling Products',
                    labels={'total_sold': 'Quantity Sold', 'product_name': ''}
                )
                fig.update_traces(marker_color='#2ecc71')
                fig.update_layout(
                    height=350,
                    showlegend=False,
                    yaxis={'categoryorder':'total ascending'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sales data available for this month")
        
        st.markdown("---")
        
        # Row 3: Regional Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🗺️ Sales by Region")
            
            region_df = db.execute_query("""
                SELECT 
                    b.region,
                    COUNT(DISTINCT s.id) as sales_count,
                    SUM(s.quantity) as total_quantity
                FROM sales s
                JOIN branch b ON s.branch_code = b.branch_code
                WHERE s.date >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY b.region
                ORDER BY total_quantity DESC
            """)
            
            if not region_df.empty:
                fig = px.pie(
                    region_df,
                    values='total_quantity',
                    names='region',
                    title='Sales Distribution by Region',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No regional sales data available")
        
        with col2:
            st.subheader("⚠️ Low Stock Alerts")
            
            # Simple low stock check (< 10 units)
            low_stock_df = db.execute_query("""
                SELECT 
                    i.product_name,
                    b.branch_name,
                    i.quantity,
                    i.unit
                FROM inventory i
                JOIN branch b ON i.branch_code = b.branch_code
                WHERE i.quantity < 10 AND i.quantity > 0
                ORDER BY i.quantity ASC
                LIMIT 10
            """)
            
            if not low_stock_df.empty:
                st.dataframe(
                    low_stock_df,
                    use_container_width=True,
                    hide_index=True,
                    height=318
                )
            else:
                st.success("✅ No critical low stock items!")
        
        st.markdown("---")
        
        # Recent Activity
        st.subheader("🕐 Recent Query History")
        
        if st.session_state.get('query_history'):
            history_df = pd.DataFrame(st.session_state.query_history)
            st.dataframe(
                history_df[['timestamp', 'intent', 'question', 'success']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No queries yet. Try running an analysis!")
        
    except Exception as e:
        st.error(f"Error loading dashboard: {e}")
        st.exception(e)


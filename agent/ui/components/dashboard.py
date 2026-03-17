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
    st.markdown("""
    <div style='color: #718096; font-size: 1.05rem; margin-bottom: 2rem; font-weight: 500;'>
        Real-time inventory and sales insights
    </div>
    """, unsafe_allow_html=True)
    
    # Get database stats
    try:
        db = orchestrator.db_manager
        
        # Row 1: Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Total branches
            branches_df = db.execute_query("SELECT COUNT(DISTINCT branch_code) as count FROM branch")
            total_branches = int(branches_df['count'].iloc[0])
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 1.5rem; border-radius: 16px; color: white; text-align: center;
                        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);'>
                <div style='font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem;'>{total_branches}</div>
                <div style='font-size: 0.95rem; font-weight: 600; opacity: 0.9;'>🏢 Total Branches</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Total products
            products_df = db.execute_query("SELECT COUNT(DISTINCT product_code) as count FROM product")
            total_products = int(products_df['count'].iloc[0])
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                        padding: 1.5rem; border-radius: 16px; color: white; text-align: center;
                        box-shadow: 0 8px 25px rgba(240, 147, 251, 0.3);'>
                <div style='font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem;'>{total_products:,}</div>
                <div style='font-size: 0.95rem; font-weight: 600; opacity: 0.9;'>📦 Total Products</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Total inventory
            inventory_df = db.execute_query("SELECT SUM(quantity) as total FROM inventory")
            total_inventory = int(inventory_df['total'].iloc[0])
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                        padding: 1.5rem; border-radius: 16px; color: white; text-align: center;
                        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);'>
                <div style='font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem;'>{total_inventory:,}</div>
                <div style='font-size: 0.95rem; font-weight: 600; opacity: 0.9;'>📊 Total Stock</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Sales today
            sales_df = db.execute_query("""
                SELECT COALESCE(SUM(quantity), 0) as total 
                FROM sales 
                WHERE date = CURRENT_DATE
            """)
            sales_today = int(sales_df['total'].iloc[0]) if not sales_df.empty else 0
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
                        padding: 1.5rem; border-radius: 16px; color: white; text-align: center;
                        box-shadow: 0 8px 25px rgba(67, 233, 123, 0.3);'>
                <div style='font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem;'>{sales_today:,}</div>
                <div style='font-size: 0.95rem; font-weight: 600; opacity: 0.9;'>💰 Sales Today</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Row 2: Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='margin-bottom: 1rem;'>
                <h3 style='color: #2d3748; font-size: 1.3rem; font-weight: 700; margin-bottom: 1rem;'>
                    📈 Sales Trend (Last 30 Days)
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            sales_trend_df = db.execute_query("""
                SELECT date, SUM(quantity) as total_sales
                FROM sales
                WHERE date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
                  AND date < DATE_TRUNC('month', CURRENT_DATE)
                GROUP BY date
                ORDER BY date
            """)
            
            if not sales_trend_df.empty:
                fig = px.line(
                    sales_trend_df,
                    x='date',
                    y='total_sales',
                    title='',
                    labels={'total_sales': 'Quantity', 'date': 'Date'}
                )
                fig.update_traces(
                    line_color='#667eea', 
                    line_width=3,
                    fill='tonexty',
                    fillcolor='rgba(102, 126, 234, 0.1)'
                )
                fig.update_layout(
                    height=350,
                    showlegend=False,
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2d3748'),
                    xaxis=dict(gridcolor='rgba(0,0,0,0.05)'),
                    yaxis=dict(gridcolor='rgba(0,0,0,0.05)')
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sales data available for the last 30 days")
        
        with col2:
            st.markdown("""
            <div style='margin-bottom: 1rem;'>
                <h3 style='color: #2d3748; font-size: 1.3rem; font-weight: 700; margin-bottom: 1rem;'>
                    🏆 Top 10 Products (This Month)
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
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
                    title='',
                    labels={'total_sold': 'Quantity Sold', 'product_name': ''}
                )
                fig.update_traces(
                    marker_color='#48bb78',
                    marker_line_color='#38a169',
                    marker_line_width=1.5
                )
                fig.update_layout(
                    height=350,
                    showlegend=False,
                    yaxis=dict(
                        categoryorder='total ascending',
                        gridcolor='rgba(0,0,0,0.05)'
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2d3748'),
                    xaxis=dict(gridcolor='rgba(0,0,0,0.05)')
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


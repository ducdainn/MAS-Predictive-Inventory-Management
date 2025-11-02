"""
Inventory Optimization component
Main feature: ROP, Safety Stock, Restock/Transfer recommendations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from PIL import Image

def render(orchestrator):
    """
    Render inventory optimization page
    
    Args:
        orchestrator: OrchestratorAgent instance
    """
    st.title("🎯 Inventory Optimization")
    st.markdown("AI-powered restock and transfer recommendations")
    
    # Input section
    st.subheader("📝 Enter Your Question")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Quick templates
        template = st.selectbox(
            "Quick Templates:",
            [
                "Custom Question",
                "Tối ưu hóa tồn kho của chi nhánh đà nẵng",
                "Tối ưu hóa tồn kho của chi nhánh bình chánh",
                "Kiểm tra tồn kho miền bắc",
                "Tối ưu hóa tồn kho tất cả chi nhánh"
            ]
        )
        
        if template == "Custom Question":
            question = st.text_input(
                "Your Question:",
                placeholder="e.g., Tối ưu hóa tồn kho của chi nhánh hà nội",
                help="Ask about inventory optimization for specific branches or regions"
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
    
    # Run optimization
    if run_button and question:
        with st.spinner("🤖 AI Agents are working..."):
            try:
                # Progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Step 1/6: Classifying intent...")
                progress_bar.progress(16)
                
                status_text.text("Step 2/6: Extracting entities...")
                progress_bar.progress(33)
                
                status_text.text("Step 3/6: Analyzing inventory...")
                progress_bar.progress(50)
                
                status_text.text("Step 4/6: Generating forecasts...")
                progress_bar.progress(66)
                
                status_text.text("Step 5/6: Finding opportunities...")
                progress_bar.progress(83)
                
                # Run optimization
                result = orchestrator.process_query(question)
                
                status_text.text("Step 6/6: Creating insights...")
                progress_bar.progress(100)
                
                # Store result
                st.session_state.last_result = result
                
                # Add to history
                if 'query_history' not in st.session_state:
                    st.session_state.query_history = []
                
                st.session_state.query_history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'intent': result.get('intent', 'Unknown'),
                    'question': question,
                    'success': result.get('success', False)
                })
                
                progress_bar.empty()
                status_text.empty()
                
                # Show success
                st.success(f"✅ Analysis complete in {result.get('elapsed_seconds', 0):.2f}s")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                st.exception(e)
                return
    
    # Display results
    if st.session_state.get('last_result'):
        result = st.session_state.last_result
        
        if result.get('success') and result.get('intent') == 'INVENTORY_OPTIMIZATION':
            display_optimization_results(result)
        else:
            st.warning("Last result is not an inventory optimization. Please run a new analysis.")


def display_optimization_results(result):
    """Display optimization results with tabs"""
    
    data = result['result']
    
    # Summary metrics
    st.subheader("📊 Summary")
    
    if data.get('action_plan'):
        summary = data['action_plan']['summary']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Actions",
                summary['total_actions'],
                help="Total number of recommended actions"
            )
        
        with col2:
            st.metric(
                "Restock Orders",
                summary['restock_actions'],
                help="Items that need external procurement"
            )
        
        with col3:
            st.metric(
                "Internal Transfers",
                summary['transfer_actions'],
                help="Items that can be transferred between branches"
            )
        
        with col4:
            st.metric(
                "High Priority",
                summary['high_priority_actions'],
                delta=None,
                delta_color="inverse",
                help="Urgent actions needed"
            )
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Kế Hoạch Hành Động",
        "📈 Biểu Đồ",
        "🧠 Phân Tích AI",
        "📦 Dữ Liệu Chi Tiết",
        "💾 Xuất File"
    ])
    
    with tab1:
        display_action_plan_tab(data)
    
    with tab2:
        display_charts_tab(data)
    
    with tab3:
        display_insights_tab(data)
    
    with tab4:
        display_data_tab(data)
    
    with tab5:
        display_export_tab(data)


def display_action_plan_tab(data):
    """Display action plan with priority grouping"""
    
    if not data.get('action_plan'):
        st.warning("No action plan available")
        return
    
    actions = data['action_plan']['actions']
    
    if not actions:
        st.success("✅ All items are at optimal stock levels!")
        return
    
    # Group by priority
    high = [a for a in actions if a['priority'] == 'HIGH']
    medium = [a for a in actions if a['priority'] == 'MEDIUM']
    low = [a for a in actions if a['priority'] == 'LOW']
    
    # High Priority
    if high:
        st.markdown("### 🔴 HIGH PRIORITY")
        st.error(f"{len(high)} urgent actions needed")
        
        for i, action in enumerate(high[:10], 1):
            with st.expander(f"{i}. {action['action_type']}: {action.get('product_name', 'Product')[:50]}"):
                if action['action_type'] == 'RESTOCK':
                    st.write(f"**Branch:** {action['branch_name']}")
                    st.write(f"**Quantity:** {action['quantity']:.0f} {action['unit']}")
                    st.write(f"**Reason:** {action['reason']}")
                else:  # TRANSFER
                    st.write(f"**From:** {action['source_branch_name']}")
                    st.write(f"**To:** {action['dest_branch_name']}")
                    st.write(f"**Quantity:** {action['quantity']:.0f} {action['unit']}")
                    st.write(f"**Distance:** {action['distance_km']:.1f} km")
        
        if len(high) > 10:
            st.info(f"... and {len(high) - 10} more high priority actions")
    
    # Medium Priority
    if medium:
        st.markdown("### 🟡 MEDIUM PRIORITY")
        st.warning(f"{len(medium)} actions recommended")
        
        if st.checkbox("Show Medium Priority Details", key="show_medium"):
            for i, action in enumerate(medium[:5], 1):
                with st.expander(f"{i}. {action['action_type']}: {action.get('product_name', 'Product')[:50]}"):
                    if action['action_type'] == 'RESTOCK':
                        st.write(f"**Branch:** {action['branch_name']}")
                        st.write(f"**Quantity:** {action['quantity']:.0f} {action['unit']}")
                    else:
                        st.write(f"**From → To:** {action['source_branch_name']} → {action['dest_branch_name']}")
                        st.write(f"**Quantity:** {action['quantity']:.0f} {action['unit']}")
    
    # Low Priority
    if low:
        st.markdown("### 🟢 LOW PRIORITY")
        st.info(f"{len(low)} actions (can be scheduled)")


def display_charts_tab(data):
    """Display visualizations"""
    
    st.markdown("### 📊 Optimization Charts")
    
    # Show matplotlib chart
    if data.get('chart'):
        try:
            img = Image.open(data['chart'])
            st.image(img, use_container_width=True, caption="Inventory Optimization Analysis")
        except Exception as e:
            st.warning(f"Could not load chart: {e}")
    
    # Interactive Plotly charts
    if data.get('recommendations') is not None and not data['recommendations'].empty:
        recs = data['recommendations']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Action distribution
            action_counts = recs['action'].value_counts()
            fig = px.pie(
                values=action_counts.values,
                names=action_counts.index,
                title="Action Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Priority distribution
            priority_counts = recs[recs['action'] != 'OK']['priority'].value_counts()
            fig = px.bar(
                x=priority_counts.index,
                y=priority_counts.values,
                title="Priority Distribution",
                labels={'x': 'Priority', 'y': 'Count'},
                color=priority_counts.index,
                color_discrete_map={'HIGH': '#e74c3c', 'MEDIUM': '#f39c12', 'LOW': '#f1c40f'}
            )
            st.plotly_chart(fig, use_container_width=True)


def display_insights_tab(data):
    """Display AI-generated insights"""
    
    st.markdown("### 🧠 Phân Tích Thông Minh Từ AI")
    
    if data.get('smart_insights'):
        # Style the insights with better colors
        insights_html = f"""
        <div style="
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            color: #262730;
            line-height: 1.8;
            font-size: 0.95rem;
        ">
            <pre style="
                white-space: pre-wrap;
                word-wrap: break-word;
                background-color: transparent;
                color: #262730;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 0;
            ">{data['smart_insights']}</pre>
        </div>
        """
        st.markdown(insights_html, unsafe_allow_html=True)
    else:
        st.info("Không có phân tích AI")


def display_data_tab(data):
    """Display detailed data tables"""
    
    st.markdown("### 📦 Detailed Recommendations")
    
    if data.get('recommendations') is not None and not data['recommendations'].empty:
        recs = data['recommendations']
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            actions_filter = st.multiselect(
                "Filter by Action",
                options=recs['action'].unique(),
                default=None
            )
        
        with col2:
            priority_filter = st.multiselect(
                "Filter by Priority",
                options=['HIGH', 'MEDIUM', 'LOW'],
                default=None
            )
        
        with col3:
            branch_filter = st.multiselect(
                "Filter by Branch",
                options=recs['branch_name'].unique(),
                default=None
            )
        
        # Apply filters
        filtered = recs.copy()
        if actions_filter:
            filtered = filtered[filtered['action'].isin(actions_filter)]
        if priority_filter:
            filtered = filtered[filtered['priority'].isin(priority_filter)]
        if branch_filter:
            filtered = filtered[filtered['branch_name'].isin(branch_filter)]
        
        st.dataframe(
            filtered[[
                'product_name', 'branch_name', 'current_stock',
                'reorder_point', 'action', 'priority', 'quantity_needed'
            ]],
            use_container_width=True,
            hide_index=True
        )
        
        st.caption(f"Showing {len(filtered)} of {len(recs)} items")
    else:
        st.info("No recommendations data available")


def display_export_tab(data):
    """Display export options"""
    
    st.markdown("### 💾 Export Results")
    
    st.info("Click buttons below to download results in different formats")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Excel", use_container_width=True):
            try:
                from agent.improved_mas import export_inventory_plan_to_excel
                export_inventory_plan_to_excel(
                    {'success': True, 'result': data},
                    "inventory_optimization_plan.xlsx"
                )
                st.success("✅ Exported to inventory_optimization_plan.xlsx")
            except Exception as e:
                st.error(f"Export failed: {e}")
    
    with col2:
        if st.button("📄 Export Forecasts CSV", use_container_width=True):
            try:
                from agent.improved_mas import export_forecasts_to_csv
                export_forecasts_to_csv(
                    {'success': True, 'result': data},
                    "forecasts_detail.csv"
                )
                st.success("✅ Exported to forecasts_detail.csv")
            except Exception as e:
                st.error(f"Export failed: {e}")
    
    with col3:
        if st.button("📋 Export Recommendations CSV", use_container_width=True):
            try:
                from agent.improved_mas import export_recommendations_to_csv
                export_recommendations_to_csv(
                    {'success': True, 'result': data},
                    "recommendations_detail.csv"
                )
                st.success("✅ Exported to recommendations_detail.csv")
            except Exception as e:
                st.error(f"Export failed: {e}")
    
    st.markdown("---")
    
    st.markdown("""
    **Files will be saved to:**
    - `inventory_optimization_plan.xlsx` - Complete plan (5 sheets)
    - `forecasts_detail.csv` - Forecast comparisons
    - `recommendations_detail.csv` - All metrics
    """)


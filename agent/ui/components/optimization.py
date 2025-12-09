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
                result = orchestrator.process_query(question, forced_intent="INVENTORY_OPTIMIZATION")
                
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
        
        # Debug info
        with st.expander("🔍 Debug: Result Check", expanded=False):
            st.write(f"**Success:** `{result.get('success')}`")
            st.write(f"**Intent:** `{result.get('intent')}`")
            st.write(f"**Has result key:** `{'result' in result}`")
            if 'result' in result:
                st.write(f"**Result success:** `{result['result'].get('success')}`")
            st.write(f"**Keys in result:** `{list(result.keys())}`")
        
        # Check if result is valid inventory optimization
        is_valid = (
            result.get('success') and 
            result.get('intent') == 'INVENTORY_OPTIMIZATION' and
            'result' in result
        )
        
        if is_valid:
            display_optimization_results(result)
        else:
            st.warning("Last result is not an inventory optimization. Please run a new analysis.")
            
            # Show error details if available
            if result.get('error'):
                st.error(f"Error: {result.get('error')}")
            elif 'result' in result and result['result'].get('message'):
                st.error(f"Error: {result['result'].get('message')}")


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
    
    # Interactive Plotly charts - use raw data for filtering
    if data.get('recommendations_raw') is not None and not data['recommendations_raw'].empty:
        recs = data['recommendations_raw']  # Use raw data with English columns
        
        # 1) Biểu đồ interactive: Tồn kho hiện tại vs ROP vs Tồn kho an toàn (Top 10)
        st.markdown("#### 📦 Top 10: Tồn kho vs ROP & Tồn kho an toàn")
        try:
            top_10 = recs.nlargest(10, 'current_stock')
            if not top_10.empty:
                labels = [
                    f"{row['product_name'][:30]}\n({row['branch_name'][:20]})"
                    for _, row in top_10.iterrows()
                ]
                fig_top10 = go.Figure()
                fig_top10.add_bar(
                    x=labels,
                    y=top_10['current_stock'],
                    name='Tồn Kho Hiện Tại',
                    marker_color='steelblue',
                )
                fig_top10.add_bar(
                    x=labels,
                    y=top_10['reorder_point'],
                    name='Điểm Đặt Hàng (ROP)',
                    marker_color='orange',
                )
                fig_top10.add_bar(
                    x=labels,
                    y=top_10['safety_stock'],
                    name='Tồn Kho An Toàn',
                    marker_color='green',
                )
                fig_top10.update_layout(
                    barmode='group',
                    xaxis_title='Sản Phẩm @ Chi Nhánh',
                    yaxis_title='Số Lượng',
                    height=450,
                    legend_title_text='Chỉ Tiêu',
                    xaxis_tickangle=-45,
                )
                st.plotly_chart(fig_top10, use_container_width=True)
        except Exception as e:
            st.warning(f"Không thể vẽ biểu đồ Top 10 tồn kho: {e}")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            # 2) Phân phối hành động (interactive)
            action_counts = recs['action'].value_counts()
            
            # Translate action labels to Vietnamese
            action_translation = {
                'OK': 'Đủ Hàng',
                'RESTOCK': 'Cần Nhập',
                'URGENT_RESTOCK': 'Nhập Gấp',
                'SURPLUS': 'Thừa Hàng',
                'TRANSFER': 'Chuyển Kho'
            }
            translated_index = [action_translation.get(x, x) for x in action_counts.index]
            
            fig = px.pie(
                values=action_counts.values,
                names=translated_index,
                title="Phân Phối Hành Động",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 3) Phân phối ưu tiên (interactive)
            priority_counts = recs[recs['action'] != 'OK']['priority'].value_counts()
            
            # Translate priority labels
            priority_translation = {'HIGH': 'Cao', 'MEDIUM': 'Trung Bình', 'LOW': 'Thấp'}
            translated_priority = [priority_translation.get(x, x) for x in priority_counts.index]
            
            fig = px.bar(
                x=translated_priority,
                y=priority_counts.values,
                title="Phân Phối Ưu Tiên",
                labels={'x': 'Mức Độ Ưu Tiên', 'y': 'Số Lượng'},
                color=priority_counts.index,
                color_discrete_map={'HIGH': '#e74c3c', 'MEDIUM': '#f39c12', 'LOW': '#f1c40f'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # 4) Biểu đồ interactive: Tổng nhu cầu dự báo 30 ngày (aggregate)
        per_item_forecasts = data.get('per_item_forecasts', {})
        if per_item_forecasts:
            st.markdown("---")
            st.markdown("#### 📈 Tổng Nhu Cầu Dự Báo 30 Ngày")
            try:
                first_fc = next(iter(per_item_forecasts.values()))
                fc_df = first_fc.get('forecast_df')
                if fc_df is not None and not fc_df.empty and 'forecast' in fc_df.columns:
                    base_dates = pd.to_datetime([d.date() for d in fc_df.index])
                    total_series = pd.Series(0.0, index=base_dates)
                    for fc in per_item_forecasts.values():
                        fdf = fc.get('forecast_df')
                        if fdf is None or fdf.empty or 'forecast' not in fdf.columns:
                            continue
                        dates_norm = pd.to_datetime([d.date() for d in fdf.index])
                        s = pd.Series(fdf['forecast'].values, index=dates_norm)
                        total_series = total_series.add(s, fill_value=0)
                    agg_df = total_series.reset_index()
                    agg_df.columns = ['date', 'forecast']
                    fig_agg = px.line(
                        agg_df,
                        x='date',
                        y='forecast',
                        title='Tổng Nhu Cầu Dự Báo 30 Ngày',
                        labels={'date': 'Ngày', 'forecast': 'Số Lượng'}
                    )
                    fig_agg.update_traces(line_color='orange')
                    st.plotly_chart(fig_agg, use_container_width=True)
            except Exception as e:
                st.warning(f"Không thể vẽ biểu đồ tổng nhu cầu dự báo: {e}")


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
    
    # Use formatted data for display
    if data.get('recommendations') is not None and not data['recommendations'].empty:
        recs_display = data['recommendations']  # Formatted (Vietnamese columns)
        recs_raw = data.get('recommendations_raw', recs_display)  # Raw for filtering
        
        # Filters (using raw data for options)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            actions_filter = st.multiselect(
                "Lọc theo Hành Động",
                options=recs_raw['action'].unique() if 'action' in recs_raw.columns else [],
                default=None
            )
        
        with col2:
            priority_filter = st.multiselect(
                "Lọc theo Ưu Tiên",
                options=['HIGH', 'MEDIUM', 'LOW'],
                default=None
            )
        
        with col3:
            branch_filter = st.multiselect(
                "Lọc theo Chi Nhánh",
                options=recs_raw['branch_name'].unique() if 'branch_name' in recs_raw.columns else [],
                default=None
            )
        
        # Apply filters on raw data, then get indices for display data
        filtered_indices = recs_raw.index
        if actions_filter:
            filtered_indices = filtered_indices.intersection(recs_raw[recs_raw['action'].isin(actions_filter)].index)
        if priority_filter:
            filtered_indices = filtered_indices.intersection(recs_raw[recs_raw['priority'].isin(priority_filter)].index)
        if branch_filter:
            filtered_indices = filtered_indices.intersection(recs_raw[recs_raw['branch_name'].isin(branch_filter)].index)
        
        # Display formatted data
        filtered_display = recs_display.loc[filtered_indices]
        filtered_raw = recs_raw.loc[filtered_indices]
        
        # Get forecast data from per_item_forecasts if available
        per_item_forecasts = data.get('per_item_forecasts', {})
        
        # Add forecast columns to display
        forecast_info = []
        for idx, row in filtered_raw.iterrows():
            product_code = row.get('product_code', '')
            branch_code = row.get('branch_code', '')
            key = (product_code, branch_code)
            
            forecast_data = per_item_forecasts.get(key, {})
            metrics = forecast_data.get('metrics', {})
            
            forecast_info.append({
                'recent_avg_daily': metrics.get('recent_avg_daily', 0),
                'forecast_avg_daily': metrics.get('forecast_avg_daily', 0),
                'forecast_total_30d': metrics.get('forecast_total', row.get('forecast_demand_30d', 0)),
                'trend': metrics.get('trend', 'unknown')
            })
        
        # Create forecast DataFrame
        if forecast_info:
            forecast_df = pd.DataFrame(forecast_info, index=filtered_display.index)
            
            # Merge forecast info with display data
            # Map English forecast columns to Vietnamese for display
            forecast_display_cols = {
                'recent_avg_daily': 'Nhu Cầu TB Ngày (Lịch Sử)',
                'forecast_avg_daily': 'Nhu Cầu TB Ngày (Dự Báo)',
                'forecast_total_30d': 'Tổng Nhu Cầu 30 Ngày',
                'trend': 'Xu Hướng'
            }
            forecast_df_display = forecast_df.rename(columns=forecast_display_cols)
            
            # Format numbers
            for col in ['Nhu Cầu TB Ngày (Lịch Sử)', 'Nhu Cầu TB Ngày (Dự Báo)', 'Tổng Nhu Cầu 30 Ngày']:
                if col in forecast_df_display.columns:
                    forecast_df_display[col] = forecast_df_display[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "0.00")
        
        # Select columns to display (Vietnamese names) - include forecast columns
        display_cols = []
        
        # Priority columns (always show)
        priority_keywords = ['sản phẩm', 'chi nhánh', 'tồn kho', 'điểm đặt', 'hành động', 'ưu tiên']
        for col in recs_display.columns:
            if any(x in col.lower() for x in priority_keywords):
                display_cols.append(col)
        
        # Add forecast columns if available
        if forecast_info and 'forecast_df_display' in locals():
            for col in forecast_df_display.columns:
                if col not in display_cols:
                    display_cols.append(col)
        
        # Add forecast demand columns from recommendations if not already included
        forecast_keywords = ['nhu cầu', 'dự báo', 'forecast', 'demand']
        for col in recs_display.columns:
            if any(x in col.lower() for x in forecast_keywords) and col not in display_cols:
                display_cols.append(col)
        
        # Combine display data with forecast info
        if forecast_info and 'forecast_df_display' in locals():
            combined_display = pd.concat([filtered_display, forecast_df_display], axis=1)
        else:
            combined_display = filtered_display
        
        # Display with selected columns
        if display_cols:
            # Filter to only show columns that exist
            available_cols = [col for col in display_cols if col in combined_display.columns]
            if available_cols:
                st.dataframe(
                    combined_display[available_cols],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.dataframe(combined_display, use_container_width=True, hide_index=True)
        else:
            st.dataframe(combined_display, use_container_width=True, hide_index=True)
        
        st.caption(f"Hiển thị {len(filtered_display)} / {len(recs_display)} sản phẩm")
        
        # Add forecast summary metrics
        if forecast_info:
            st.markdown("---")
            st.markdown("### 📊 Tóm Tắt Nhu Cầu Dự Báo")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_forecast = sum(f.get('forecast_total_30d', 0) for f in forecast_info)
                st.metric(
                    "Tổng Nhu Cầu 30 Ngày",
                    f"{total_forecast:,.0f}",
                    help="Tổng nhu cầu dự báo cho tất cả sản phẩm trong 30 ngày tới"
                )
            
            with col2:
                avg_daily = sum(f.get('forecast_avg_daily', 0) for f in forecast_info) / len(forecast_info) if forecast_info else 0
                st.metric(
                    "Nhu Cầu TB Ngày",
                    f"{avg_daily:.2f}",
                    help="Nhu cầu trung bình hàng ngày từ dự báo"
                )
            
            with col3:
                increasing_count = sum(1 for f in forecast_info if f.get('trend') == 'increasing')
                st.metric(
                    "Xu Hướng Tăng",
                    f"{increasing_count}",
                    help="Số sản phẩm có xu hướng nhu cầu tăng"
                )
            
            with col4:
                recent_avg = sum(f.get('recent_avg_daily', 0) for f in forecast_info) / len(forecast_info) if forecast_info else 0
                st.metric(
                    "Nhu Cầu TB (Lịch Sử)",
                    f"{recent_avg:.2f}",
                    help="Nhu cầu trung bình hàng ngày từ dữ liệu lịch sử"
                )
    else:
        st.info("Không có dữ liệu recommendations")


def display_export_tab(data):
    """Display export options"""
    
    st.markdown("### 💾 Export Results")
    
    st.info("Click buttons below to download results in different formats")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Excel", use_container_width=True):
            try:
                from agent.agents.orchestrator_agent import export_inventory_plan_to_excel
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
                from agent.agents.orchestrator_agent import export_forecasts_to_csv
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
                from agent.agents.orchestrator_agent import export_recommendations_to_csv
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


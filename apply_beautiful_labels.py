"""
Apply Beautiful Labels - Comprehensive Update
Updates ALL displays in the system with beautiful Vietnamese labels:
- Charts (matplotlib/seaborn)
- DataFrames (pandas)
- Summary text
- Console output
- Excel export
- UI components
"""

import re
from pathlib import Path

# Complete Vietnamese label mappings
LABEL_MAPPINGS = {
    # Core columns
    'date': 'Ngày',
    'month': 'Tháng',
    'year': 'Năm',
    'quarter': 'Quý',
    'week': 'Tuần',
    
    # Sales columns
    'total_sales_quantity': 'Tổng Số Lượng Bán Hàng',
    'total_quantity': 'Tổng Số Lượng',
    'quantity': 'Số Lượng',
    'total_revenue': 'Tổng Doanh Thu',
    'revenue': 'Doanh Thu',
    'selling_price': 'Giá Bán',
    'cost_price': 'Giá Vốn',
    'profit': 'Lợi Nhuận',
    'total_sales': 'Tổng Doanh Số',
    'sales_quantity': 'Số Lượng Bán',
    
    # Product columns
    'product_name': 'Tên Sản Phẩm',
    'product_code': 'Mã Sản Phẩm',
    'category_name': 'Danh Mục',
    'category': 'Danh Mục',
    'category_id': 'Mã Danh Mục',
    
    # Branch columns
    'branch_name': 'Tên Chi Nhánh',
    'branch_code': 'Mã Chi Nhánh',
    'region': 'Khu Vực',
    
    # Inventory columns
    'quantity_on_hand': 'Tồn Kho Hiện Tại',
    'reorder_point': 'Điểm Đặt Hàng Lại',
    'safety_stock': 'Tồn Kho An Toàn',
    'current_stock': 'Tồn Kho Hiện Tại',
    'forecast_demand': 'Dự Báo Nhu Cầu',
    'recommended_restock': 'Đề Xuất Nhập Hàng',
    'priority': 'Mức Độ Ưu Tiên',
    'action': 'Hành Động',
    
    # Aggregations
    'count': 'Số Lượng',
    'avg': 'Trung Bình',
    'sum': 'Tổng',
    'min': 'Nhỏ Nhất',
    'max': 'Lớn Nhất',
    'total': 'Tổng Cộng',
    'mean': 'Trung Bình',
    'median': 'Trung Vị',
    
    # Common terms
    'transaction_count': 'Số Giao Dịch',
    'product_count': 'Số Sản Phẩm',
    'customer_count': 'Số Khách Hàng',
    'row_count': 'Số Dòng',
    'forecast': 'Dự Báo',
    'historical': 'Lịch Sử',
    'value': 'Giá Trị',
    
    # Distance/Transfer
    'distance_km': 'Khoảng Cách (km)',
    'from_branch': 'Từ Chi Nhánh',
    'to_branch': 'Đến Chi Nhánh',
    'transfer_quantity': 'Số Lượng Chuyển',
}

def create_updated_improved_mas():
    """Create updated improved_mas.py with beautiful labels everywhere."""
    
    print("=" * 80)
    print("APPLYING BEAUTIFUL LABELS TO ALL DISPLAYS")
    print("=" * 80)
    print()
    
    # Read original file
    mas_file = Path("agent/improved_mas.py")
    print(f"📖 Reading: {mas_file}")
    
    with open(mas_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"✓ File read: {len(content)} characters")
    print()
    
    # Backup
    backup_file = Path("agent/improved_mas.py.backup_labels")
    print(f"💾 Creating backup: {backup_file}")
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✓ Backup created")
    print()
    
    # Apply updates
    print("🔧 Applying updates...")
    print()
    
    updates = []
    
    # 1. Add import at top
    if 'from agent.label_formatter import' not in content:
        import_line = "from agent.label_formatter import QuickLabelFormatter, format_axis_label\n"
        # Find first import line
        first_import = content.find('import')
        if first_import > 0:
            content = content[:first_import] + import_line + content[first_import:]
            updates.append("✓ Added label_formatter import")
    
    # 2. Update AnalyticsAgent._plot_time_series
    old_plot_ts = r'def _plot_time_series\(self, df: pd\.DataFrame, date_col: str, value_col: str\) -> str:.*?return filepath'
    new_plot_ts = '''def _plot_time_series(self, df: pd.DataFrame, date_col: str, value_col: str) -> str:
        """Plot time series with beautiful labels."""
        from agent.label_formatter import QuickLabelFormatter
        
        formatter = QuickLabelFormatter()
        labels = formatter.format_chart_labels(date_col, value_col)
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x=date_col, y=value_col, linewidth=2.5, color='steelblue')
        
        plt.xlabel(labels['x'], fontsize=12, fontweight='bold')
        plt.ylabel(labels['y'], fontsize=12, fontweight='bold')
        plt.title(labels['title'], fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"timeseries_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath'''
    
    if re.search(old_plot_ts, content, re.DOTALL):
        content = re.sub(old_plot_ts, new_plot_ts, content, flags=re.DOTALL)
        updates.append("✓ Updated AnalyticsAgent._plot_time_series")
    
    # 3. Add DataFrame column renaming helper
    dataframe_helper = '''
# ============================================================================
# DATAFRAME FORMATTER HELPER
# ============================================================================

def format_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Format DataFrame column names to beautiful Vietnamese labels."""
    from agent.label_formatter import QuickLabelFormatter
    
    if df is None or df.empty:
        return df
    
    formatter = QuickLabelFormatter()
    
    # Rename columns
    new_columns = {}
    for col in df.columns:
        new_columns[col] = formatter.format_label(str(col))
    
    return df.rename(columns=new_columns)
'''
    
    # Insert before ANALYTICS AGENT section
    analytics_marker = "# ANALYTICS AGENT"
    if analytics_marker in content and 'def format_dataframe_columns' not in content:
        pos = content.find(analytics_marker)
        content = content[:pos] + dataframe_helper + "\n\n" + content[pos:]
        updates.append("✓ Added format_dataframe_columns helper")
    
    # 4. Write updated file
    print("\n💾 Writing updated file...")
    with open(mas_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ File written")
    print()
    
    # Summary
    print("=" * 80)
    print("UPDATES APPLIED")
    print("=" * 80)
    for update in updates:
        print(update)
    
    if not updates:
        print("⚠️  No automatic updates applied. See manual instructions below.")
    
    print()
    print("=" * 80)
    print("MANUAL UPDATES REQUIRED")
    print("=" * 80)
    print()
    print("Copy these code blocks into agent/improved_mas.py:")
    print()
    
    # Print all manual update instructions
    print_manual_instructions()

def print_manual_instructions():
    """Print detailed manual update instructions."""
    
    print("""
## 1. ADD IMPORTS (at top of file, after existing imports)

from agent.label_formatter import QuickLabelFormatter, format_axis_label


## 2. ADD HELPER FUNCTION (before AnalyticsAgent class)

def format_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Format DataFrame column names to beautiful Vietnamese labels.\"\"\"
    from agent.label_formatter import QuickLabelFormatter
    
    if df is None or df.empty:
        return df
    
    formatter = QuickLabelFormatter()
    new_columns = {}
    for col in df.columns:
        new_columns[col] = formatter.format_label(str(col))
    
    return df.rename(columns=new_columns)


## 3. UPDATE AnalyticsAgent.analyze() (around line 791)

Find:
    def analyze(self, sql: str, question: str) -> Dict[str, Any]:
        df = self.db.execute_query(sql)
        ...
        return {
            "data": df,
            ...
        }

Replace with:
    def analyze(self, sql: str, question: str) -> Dict[str, Any]:
        df = self.db.execute_query(sql)
        
        # Format column names for display
        df_display = format_dataframe_columns(df)
        
        ...
        return {
            "data": df_display,  # Use formatted df
            ...
        }


## 4. UPDATE AnalyticsAgent._plot_bar_chart() (around line 864)

def _plot_bar_chart(self, df: pd.DataFrame, cat_col: str, value_col: str) -> str:
    \"\"\"Plot bar chart with beautiful labels.\"\"\"
    from agent.label_formatter import QuickLabelFormatter
    
    formatter = QuickLabelFormatter()
    labels = formatter.format_chart_labels(cat_col, value_col)
    
    # Limit to top 20
    if len(df) > 20:
        df = df.nlargest(20, value_col)
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(df)), df[value_col], color='steelblue', alpha=0.8)
    
    plt.xticks(range(len(df)), df[cat_col], rotation=45, ha='right')
    plt.xlabel(labels['x'], fontsize=12, fontweight='bold')
    plt.ylabel(labels['y'], fontsize=12, fontweight='bold')
    plt.title(labels['title'], fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, df[value_col])):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:,.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    filename = f"barchart_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(self.output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath


## 5. UPDATE ForecastAgent._plot_forecast() (around line 1019)

def _plot_forecast(self, df_ts: pd.DataFrame, forecast_df: pd.DataFrame, value_name: str) -> str:
    \"\"\"Plot forecast with beautiful labels.\"\"\"
    from agent.label_formatter import QuickLabelFormatter
    
    formatter = QuickLabelFormatter()
    y_label = formatter.format_label(value_name)
    
    plt.figure(figsize=(14, 7))
    
    recent = df_ts.tail(90)
    plt.plot(recent.index, recent['value'], label='Dữ Liệu Lịch Sử', 
            linewidth=2.5, color='steelblue', marker='o', markersize=3)
    plt.plot(forecast_df.index, forecast_df['forecast'], label='Dự Báo', 
            linewidth=2.5, color='orange', linestyle='--', marker='o', markersize=4)
    
    # Confidence interval
    forecast_upper = forecast_df['forecast'] * 1.1
    forecast_lower = forecast_df['forecast'] * 0.9
    plt.fill_between(forecast_df.index, forecast_lower, forecast_upper, 
                    color='orange', alpha=0.2, label='Khoảng Tin Cậy ±10%')
    
    plt.xlabel('Ngày', fontsize=12, fontweight='bold')
    plt.ylabel(y_label, fontsize=12, fontweight='bold')
    plt.title(f'Dự Báo {y_label}', fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filename = f"forecast_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(self.output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath


## 6. UPDATE InventoryOptimizationAgent._plot_inventory_optimization()

Find legends (around line 1986-1990):
    bars1 = ax1.bar(..., label='Current Stock', ...)
    bars2 = ax1.bar(..., label='Reorder Point (ROP)', ...)
    bars3 = ax1.bar(..., label='Safety Stock', ...)

Replace with:
    bars1 = ax1.bar(..., label='Tồn Kho Hiện Tại', ...)
    bars2 = ax1.bar(..., label='Điểm Đặt Hàng Lại', ...)
    bars3 = ax1.bar(..., label='Tồn Kho An Toàn', ...)


## 7. UPDATE export_inventory_plan_to_excel() (around line 2519)

Find:
    writer = pd.ExcelWriter(filepath, engine='openpyxl')
    
    # Summary
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

Add before to_excel:
    # Format column names
    summary_df = format_dataframe_columns(summary_df)
    actions_df = format_dataframe_columns(actions_df)
    recommendations_df = format_dataframe_columns(recommendations_df)
    transfers_df = format_dataframe_columns(transfers_df)


## 8. UPDATE display_action_plan() (around line 2574)

Find console output like:
    print(f"  Product: {action['product_name']}")
    print(f"  Branch: {action['branch_name']}")
    print(f"  Action: {action['action']}")

Replace with Vietnamese:
    print(f"  Sản phẩm: {action['product_name']}")
    print(f"  Chi nhánh: {action['branch_name']}")
    print(f"  Hành động: {action['action']}")


## 9. UPDATE Streamlit UI (agent/ui/components/optimization.py)

Find DataFrame displays like:
    st.dataframe(data['recommendations'])

Replace with:
    from agent.improved_mas import format_dataframe_columns
    st.dataframe(format_dataframe_columns(data['recommendations']))

""")

def main():
    """Main execution."""
    try:
        create_updated_improved_mas()
        
        print()
        print("=" * 80)
        print("✅ SCRIPT COMPLETE!")
        print("=" * 80)
        print()
        print("NEXT STEPS:")
        print("1. Review the manual instructions above")
        print("2. Apply changes to agent/improved_mas.py")
        print("3. Test with: python -c \"from agent.improved_mas import initialize_system\"")
        print("4. Run UI: python run_ui.py")
        print()
        
    except Exception as e:
        print()
        print(f"❌ Error: {e}")
        print()
        print("Please apply changes manually using the instructions above.")

if __name__ == "__main__":
    main()



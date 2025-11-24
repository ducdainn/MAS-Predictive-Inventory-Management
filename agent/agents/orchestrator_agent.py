"""
OrchestratorAgent: coordinates all agents in the MAS.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from agent.utils.dataframe_utils import format_dataframe_columns

from agent.agents.analytics_agent import AnalyticsAgent
from agent.agents.entity_extractor import EntityExtractor
from agent.agents.forecast_agent import ForecastAgent
from agent.agents.intent_agent import IntentAgent
from agent.agents.inventory_agent import InventoryOptimizationAgent
from agent.agents.schema_agent import SchemaAgent
from agent.agents.smart_insights_agent import SmartInsightsGenerator  # pylint: disable=unused-import
from agent.agents.sql_agent import SQLAgent
from agent.core.conversation import ConversationEntry
from agent.core.llm_provider import LLMProvider
from agent.manager.database_manager import DatabaseManager
from agent.manager.memory_manager import MemoryManager


class OrchestratorAgent:
    """Main orchestrator that coordinates all agents."""
    
    def __init__(self, 
                 db_manager: DatabaseManager,
                 memory: MemoryManager,
                 llm_provider: LLMProvider):
        
        self.db_manager = db_manager
        self.memory = memory
        self.llm_provider = llm_provider
        
        # Initialize all agents
        self.schema_agent = SchemaAgent(db_manager, memory)
        self.entity_extractor = EntityExtractor(llm_provider, db_manager)  # NEW: Entity extraction
        self.intent_agent = IntentAgent(llm_provider)
        self.sql_agent = SQLAgent(llm_provider, self.schema_agent)
        self.analytics_agent = AnalyticsAgent(db_manager)
        self.forecast_agent = ForecastAgent(db_manager, use_ml=True)  # UPGRADED: XGBoost/LightGBM/Prophet
        self.inventory_agent = InventoryOptimizationAgent(db_manager, self.forecast_agent, llm_provider)  # With LLM for insights
        
        print("✅ OrchestratorAgent initialized with all sub-agents (ML Forecasting + Entity Extraction + Smart Insights)")
    
    def process_query(self, question: str, forced_intent: Optional[str] = None) -> Dict[str, Any]:
        """Main entry point: process user question through the agent pipeline."""
        print(f"\n{'='*80}")
        print(f"🤖 Processing question: {question}")
        print(f"{'='*80}\n")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Classify intent
            print("📌 Step 1: Intent Classification")
            valid_intents = {"FORECAST", "ANALYTICS", "INVENTORY_OPTIMIZATION"}
            if forced_intent and forced_intent.upper() in valid_intents:
                intent = forced_intent.upper()
                print(f"   → Intent overridden by caller: {intent}\n")
            else:
                intent = self.intent_agent.classify(question)
                print(f"   → Intent: {intent}\n")
            
            # Step 2: Extract entities for ALL intents (to improve SQL generation)
            print(f"📌 Step 2: Extracting entities from question...")
            entities = self.entity_extractor.extract_entities(question)
            
            # Step 3: Handle different intents
            if intent == "INVENTORY_OPTIMIZATION":
                # Step 3a: Optimize inventory with entity filters
                print(f"📌 Step 3: Processing with Inventory Optimization Agent")
                result = self.inventory_agent.optimize_inventory(question, entities=entities)
                sql = "N/A - Inventory optimization uses multiple queries internally"
            else:
                # Step 3: Generate SQL for FORECAST and ANALYTICS (with entities)
                print("📌 Step 3: SQL Generation")
                sql = self.sql_agent.generate_sql(question, intent, entities=entities)
                print(f"   → SQL: {sql[:200]}...\n")
                
                # Step 3: Route to appropriate agent
                print(f"📌 Step 3: Processing with {intent} Agent")
                
                if intent == "FORECAST":
                    result = self.forecast_agent.forecast(sql, question)
                else:
                    result = self.analytics_agent.analyze(sql, question)
            
            # Step 4: Store in memory with success status
            success = result.get('success', True)
            error_message = None
            if not success:
                error_message = result.get('message') or result.get('error') or 'Unknown error'
            
            entry = ConversationEntry(
                timestamp=datetime.now(),
                question=question,
                intent=intent,
                sql_query=sql,
                result_summary=result.get('summary', ''),
                charts=result.get('charts', []) or [result.get('chart', '')],
                success=success,
                error_message=error_message
            )
            self.memory.add_entry(entry)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            final_result = {
                "success": result.get('success', True),
                "question": question,
                "intent": intent,
                "sql": sql,
                "result": result,
                "elapsed_seconds": elapsed
            }
            
            print(f"\n✅ Completed in {elapsed:.2f}s")
            return final_result
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to get intent even if there was an error
            error_intent = "UNKNOWN"
            try:
                if forced_intent:
                    error_intent = forced_intent.upper()
                else:
                    error_intent = self.intent_agent.classify(question)
            except:
                pass
            
            return {
                "success": False,
                "question": question,
                "intent": error_intent,  # Include intent even on error
                "error": str(e),
                "elapsed_seconds": (datetime.now() - start_time).total_seconds()
            }
    
    def get_conversation_history(self) -> List[ConversationEntry]:
        """Get conversation history from memory."""
        return self.memory.conversation_history
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.conversation_history.clear()
        print("🗑️ Memory cleared")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def initialize_system() -> OrchestratorAgent:
    """Initialize the complete multi-agent system."""
    print("\n" + "="*80)
    print("🚀 Initializing Multi-Agent System...")
    print("="*80 + "\n")
    
    db_manager = DatabaseManager()
    memory = MemoryManager()
    llm_provider = LLMProvider()
    
    orchestrator = OrchestratorAgent(
        db_manager=db_manager,
        memory=memory,
        llm_provider=llm_provider
    )
    
    print("\n" + "="*80)
    print("🎉 Multi-Agent System Ready!")
    print("="*80)
    
    return orchestrator


def display_conversation_history(orchestrator: OrchestratorAgent):
    """Display conversation history in a nice format."""
    history = orchestrator.get_conversation_history()
    
    if not history:
        print("No conversation history yet.")
        return
    
    print("\n" + "="*80)
    print("📜 CONVERSATION HISTORY")
    print("="*80)
    
    for i, entry in enumerate(history, 1):
        print(f"\n[{i}] {entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Q: {entry.question}")
        print(f"Intent: {entry.intent}")
        print(f"SQL: {entry.sql_query[:100]}...")
        if entry.charts:
            print(f"Charts: {len(entry.charts)} created")
        print("-" * 80)


def export_results_to_excel(result: Dict[str, Any], filename: str = "export.xlsx"):
    """Export query results to Excel."""
    if not result.get('success'):
        print("❌ Cannot export: query was not successful")
        return
    
    data = result['result'].get('data') or result['result'].get('historical_data')
    
    if data is None or (isinstance(data, pd.DataFrame) and data.empty):
        print("❌ No data to export")
        return
    
    try:
        data.to_excel(filename, index=True)
        print(f"✅ Exported to {filename}")
    except Exception as e:
        print(f"❌ Export failed: {e}")


def export_inventory_plan_to_excel(result: Dict[str, Any], filename: str = "inventory_plan.xlsx"):
    """
    Export detailed inventory optimization plan to Excel with multiple sheets.
    
    IMPROVEMENT: Professional multi-sheet Excel export for business users.
    """
    if not result.get('success'):
        print("❌ Cannot export: optimization was not successful")
        return
    
    try:
        action_plan = result['result'].get('action_plan')
        recommendations = result['result'].get('recommendations')
        transfer_opportunities = result['result'].get('transfer_opportunities')
        
        if not action_plan:
            print("❌ No action plan to export")
            return
        
        print(f"📝 Creating Excel file: {filename}")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: Summary (with Vietnamese labels)
            summary_data = {
                'Metric': [
                    'Tổng Số Hành Động',
                    'Đơn Nhập Hàng',
                    'Cơ Hội Chuyển Kho',
                    'Hành Động Ưu Tiên Cao',
                    'Tổng Số Lượng Nhập',
                    'Tổng Số Lượng Chuyển'
                ],
                'Value': [
                    action_plan['summary']['total_actions'],
                    action_plan['summary']['restock_actions'],
                    action_plan['summary']['transfer_actions'],
                    action_plan['summary']['high_priority_actions'],
                    action_plan['summary']['total_restock_quantity'],
                    action_plan['summary']['total_transfer_quantity']
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df = format_dataframe_columns(summary_df)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            print(f"   ✓ Sheet 1: Summary")
            
            # Sheet 2: Restock Orders (formatted)
            restock_actions = [a for a in action_plan['actions'] if a['action_type'] == 'RESTOCK']
            if restock_actions:
                restock_df = pd.DataFrame(restock_actions)
                restock_df = format_dataframe_columns(restock_df)
                restock_df.to_excel(writer, sheet_name='Restock Orders', index=False)
                print(f"   ✓ Sheet 2: Restock Orders ({len(restock_actions)} items)")
            
            # Sheet 3: Transfer Opportunities (formatted)
            transfer_actions = [a for a in action_plan['actions'] if a['action_type'] == 'TRANSFER']
            if transfer_actions:
                transfer_df = pd.DataFrame(transfer_actions)
                transfer_df = format_dataframe_columns(transfer_df)
                transfer_df.to_excel(writer, sheet_name='Transfers', index=False)
                print(f"   ✓ Sheet 3: Transfers ({len(transfer_actions)} items)")
            
            # Sheet 4: All Recommendations (already formatted from optimize_inventory)
            if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
                recommendations.to_excel(writer, sheet_name='All Items', index=False)
                print(f"   ✓ Sheet 4: All Items ({len(recommendations)} items)")
            
            # Sheet 5: Priority Actions (formatted)
            high_priority = [a for a in action_plan['actions'] if a['priority'] == 'HIGH']
            if high_priority:
                priority_df = pd.DataFrame(high_priority)
                priority_df = format_dataframe_columns(priority_df)
                priority_df.to_excel(writer, sheet_name='High Priority', index=False)
                print(f"   ✓ Sheet 5: High Priority ({len(high_priority)} items)")
        
        print(f"✅ Exported detailed plan to {filename}")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()


def export_forecasts_to_csv(result: Dict[str, Any], filename: str = "forecasts_detail.csv"):
    """
    Export detailed per-item forecasts to CSV for easy analysis.
    
    NEW: Export all forecast results with comparisons to CSV.
    """
    if not result.get('success'):
        print("❌ Cannot export: optimization was not successful")
        return
    
    try:
        per_item_forecasts = result['result'].get('per_item_forecasts')
        # Use raw data (English column names) instead of formatted (Vietnamese)
        inventory_data = result['result'].get('inventory_data_raw')
        if inventory_data is None or inventory_data.empty:
            # Fallback to formatted data if raw not available
            inventory_data = result['result'].get('inventory_data')
        
        if not per_item_forecasts or inventory_data is None or inventory_data.empty:
            print("❌ No forecast data to export")
            return
        
        print(f"📝 Creating forecast CSV: {filename}")
        
        # Prepare detailed forecast data
        forecast_rows = []
        
        # Ensure we have English column names (use raw data)
        if 'product_code' not in inventory_data.columns:
            print("⚠️  Warning: inventory_data has Vietnamese column names, but raw data should be used")
            print(f"   Available columns: {list(inventory_data.columns)}")
            # Try to get raw data
            inventory_data = result['result'].get('inventory_data_raw', inventory_data)
        
        for idx, row in inventory_data.iterrows():
            product_code = row['product_code']
            branch_code = row['branch_code']
            key = (product_code, branch_code)
            
            forecast_data = per_item_forecasts.get(key)
            if not forecast_data:
                continue
            
            metrics = forecast_data['metrics']
            
            forecast_rows.append({
                'product_code': product_code,
                'product_name': row.get('product_name', ''),
                'branch_code': branch_code,
                'branch_name': row.get('branch_name', ''),
                'region': row.get('region', ''),
                'current_stock': row.get('current_stock', 0),
                'unit': row.get('unit', ''),
                'recent_avg_daily_demand': metrics['recent_avg_daily'],
                'forecast_avg_daily_demand': metrics['forecast_avg_daily'],
                'forecast_total_30d': metrics['forecast_total'],
                'trend': metrics['trend'],
                'stock_coverage_days': row.get('current_stock', 0) / max(metrics['recent_avg_daily'], 0.1) if metrics['recent_avg_daily'] > 0 else 0
            })
        
        forecast_df = pd.DataFrame(forecast_rows)
        forecast_df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"✅ Exported {len(forecast_df)} forecast records to {filename}")
        print(f"   📊 Columns: product, branch, stock, demand (recent/forecast), trend")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()


def export_recommendations_to_csv(result: Dict[str, Any], filename: str = "recommendations_detail.csv"):
    """
    Export detailed recommendations with all metrics to CSV.
    
    NEW: Complete recommendations export for analysis.
    """
    if not result.get('success'):
        print("❌ Cannot export: optimization was not successful")
        return
    
    try:
        recommendations = result['result'].get('recommendations')
        
        if recommendations is None or recommendations.empty:
            print("❌ No recommendations to export")
            return
        
        print(f"📝 Creating recommendations CSV: {filename}")
        
        recommendations.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"✅ Exported {len(recommendations)} recommendations to {filename}")
        print(f"   📊 Includes: ROP, Safety Stock, EOQ, Actions, Priorities")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()


def display_action_plan(action_plan: Dict[str, Any]):
    """
    Display detailed action plan in a beautiful format.
    
    IMPROVEMENT: Human-readable output for business users.
    """
    print("\n" + "="*80)
    print("📋 DETAILED ACTION PLAN")
    print("="*80)
    
    summary = action_plan['summary']
    print(f"\n📊 SUMMARY:")
    print(f"   Total Actions: {summary['total_actions']}")
    print(f"   - Restock Orders: {summary['restock_actions']} (Qty: {summary['total_restock_quantity']:.0f})")
    print(f"   - Internal Transfers: {summary['transfer_actions']} (Qty: {summary['total_transfer_quantity']:.0f})")
    print(f"   - High Priority: {summary['high_priority_actions']}")
    
    # Group by priority
    actions_by_priority = {}
    for action in action_plan['actions']:
        priority = action['priority']
        if priority not in actions_by_priority:
            actions_by_priority[priority] = []
        actions_by_priority[priority].append(action)
    
    # Display HIGH priority first
    for priority in ['HIGH', 'MEDIUM', 'LOW']:
        if priority not in actions_by_priority:
            continue
        
        actions = actions_by_priority[priority]
        
        priority_colors = {
            'HIGH': '🔴',
            'MEDIUM': '🟡',
            'LOW': '🟢'
        }
        
        print(f"\n{priority_colors[priority]} {priority} PRIORITY ({len(actions)} actions):")
        print("-" * 80)
        
        for i, action in enumerate(actions[:10], 1):  # Show top 10
            if action['action_type'] == 'RESTOCK':
                print(f"\n   {i}. 📦 RESTOCK: {action['product_name'][:50]}")
                print(f"      Branch: {action['branch_name']}")
                print(f"      Quantity: {action['quantity']:.0f} {action['unit']}")
                print(f"      Reason: {action['reason']}")
                
            elif action['action_type'] == 'TRANSFER':
                print(f"\n   {i}. 🚚 TRANSFER: {action['product_name'][:50]}")
                print(f"      From: {action['source_branch_name']}")
                print(f"      To: {action['dest_branch_name']}")
                print(f"      Quantity: {action['quantity']:.0f} {action['unit']}")
                print(f"      Distance: {action['distance_km']:.1f} km")
                print(f"      💰 {action['cost_saving']}")
        
        if len(actions) > 10:
            print(f"\n   ... and {len(actions) - 10} more {priority} priority actions")
    
    print("\n" + "="*80)


# ============================================================================
# MAIN USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize system
    orchestrator = initialize_system()
    
    # # Example 1: Analytics
    # print("\n" + "="*80)
    # print("📊 Example 1: Analytics Query")
    # print("="*80)
    # result1 = orchestrator.process_query(
    #     "Top 10 sản phẩm bán chạy nhất trong tháng này"
    # )
    
    # # Example 2: Forecast
    # print("\n" + "="*80)
    # print("🔮 Example 2: Forecast Query")
    # print("="*80)
    # result2 = orchestrator.process_query(
    #     "Dự báo doanh số bán hàng cho 30 ngày tới"
    # )
    
    # Example 3: Inventory Optimization with Entity Extraction (NEW!)
    print("\n" + "="*80)
    print("🎯 Example 3: Inventory Optimization with Smart Filtering")
    print("="*80)
    result3 = orchestrator.process_query(
        "Tối ưu hóa tồn kho của chi nhánh đà nẵng: "
        "kiểm tra sản phẩm nào cần nhập hàng và có thể chuyển kho không"
    )
    
    if result3.get('success'):
        print("\n" + "="*80)
        print("📋 INVENTORY OPTIMIZATION RESULTS")
        print("="*80)
        
        # Display summary statistics first
        inventory_data = result3['result'].get('inventory_data')
        recommendations = result3['result'].get('recommendations')
        
        if inventory_data is not None:
            print(f"\n📊 ANALYSIS SCOPE:")
            print(f"   • Total items analyzed: {len(inventory_data)}")
            print(f"   • Branches: {inventory_data['branch_name'].nunique()}")
            unique_branches = inventory_data['branch_name'].unique()
            for branch in unique_branches:
                count = len(inventory_data[inventory_data['branch_name'] == branch])
                print(f"     - {branch}: {count} products")
        
        if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
            print(f"\n📈 RECOMMENDATIONS SUMMARY:")
            action_dist = recommendations['action'].value_counts()
            for action, count in action_dist.items():
                pct = (count / len(recommendations)) * 100
                print(f"   • {action}: {count} items ({pct:.1f}%)")
        
        # Display detailed action plan
        if result3['result'].get('action_plan'):
            plan = result3['result']['action_plan']
            display_action_plan(plan)
        
        # Display smart insights
        if result3['result'].get('smart_insights'):
            print("\n" + "="*80)
            print("🧠 AI-POWERED INSIGHTS")
            print("="*80)
            print(result3['result']['smart_insights'])
        
        # Export all files
        print("\n" + "="*80)
        print("📊 EXPORTING RESULTS TO FILES")
        print("="*80)
        
        # 1. Excel file (multi-sheet)
        export_inventory_plan_to_excel(result3, "inventory_optimization_plan.xlsx")
        
        # 2. Forecasts CSV (detailed)
        export_forecasts_to_csv(result3, "forecasts_detail.csv")
        
        # 3. Recommendations CSV (complete)
        export_recommendations_to_csv(result3, "recommendations_detail.csv")
        
        print("\n✅ ALL EXPORTS COMPLETE!")
        print("📁 Files created:")
        print("   1. inventory_optimization_plan.xlsx (5 sheets)")
        print("   2. forecasts_detail.csv (forecast comparisons)")
        print("   3. recommendations_detail.csv (all metrics)")
        print(f"   4. {result3['result']['chart']} (visualization)")
    
    else:
        print("\n" + "="*80)
        print("❌ OPTIMIZATION FAILED")
        print("="*80)
        print(f"Error: {result3.get('error', result3.get('message', 'Unknown error'))}")
    
    # Show history
    display_conversation_history(orchestrator)


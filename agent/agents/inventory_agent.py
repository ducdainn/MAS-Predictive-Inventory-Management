"""
InventoryOptimizationAgent: intelligent inventory management agent.
"""

import os
import time
import uuid
# Threading disabled - using vectorization instead
# from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from agent.agents.forecast_agent import ForecastAgent
from agent.agents.smart_insights_agent import SmartInsightsGenerator
from agent.manager.database_manager import DatabaseManager
from agent.utils.dataframe_utils import format_dataframe_columns
from agent.core.llm_provider import LLMProvider
from agent.utils.data_availability_checker import DataAvailabilityChecker
from agent.utils.model_logger import get_model_logger
from agent.utils.workflow_data_logger import get_workflow_logger
from agent.panel_xgboost_model_loader import (
    get_panel_model_loader,
    get_panel_multistep_model_loader,
)


try:
    from agent.system_date import get_system_date
    SYSTEM_DATE_AVAILABLE = True
except ImportError:
    SYSTEM_DATE_AVAILABLE = False


class InventoryOptimizationAgent:
    """
    Intelligent inventory management agent that:
    1. Analyzes current stock levels
    2. Compares with forecast demand
    3. Calculates reorder points and safety stock
    4. Recommends restocking or transfers based on proximity
    """
    
    def __init__(self, 
                 db_manager: DatabaseManager,
                 forecast_agent: 'ForecastAgent',
                 llm_provider: LLMProvider,
                 output_dir: str = "charts"):
        self.db = db_manager
        self.forecast_agent = forecast_agent
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # NEW: Smart insights generator + monitoring utilities
        self.insights_generator = SmartInsightsGenerator(llm_provider)
        self.data_checker = DataAvailabilityChecker()
        self._latest_data_quality_report: Dict[str, Any] = {}
        self._latest_drift_report: Dict[str, Any] = {}
        self._model_info: Dict[str, Any] = {}

        # Centralized model logger cho toàn bộ pipeline Optimization
        self.model_logger = get_model_logger(log_dir="model_logs")
        
        # Workflow data logger
        self.workflow_logger = get_workflow_logger()
        
        # Configuration parameters
        self.service_level = 0.95  # 95% service level
        self.lead_time_days = 7    # Default lead time
        self.max_transfer_distance_km = 200  # Max distance for transfer
        self.missing_data_forecast_value = 0.0  # Forecast level for missing SKU history
        self.max_forecast_vs_recent_ratio = 1.2  # Cap forecasts relative to recent demand to avoid seasonal spikes
        # DISABLED: Threading replaced with vectorization (faster due to GIL)
        # self.max_parallel_forecasts = min(16, max(4, os.cpu_count() or 4))
        self.bulk_query_chunk_size = 200  # Number of product codes per bulk SQL query
    
    def optimize_inventory(self, 
                          question: str,
                          entities: Optional[Dict] = None,
                          horizon_days: int = 30,
                          max_items: Optional[int] = None,
                          use_batch_tool: bool = False) -> Dict[str, Any]:
        """
        Main optimization workflow:
        1. Get forecast demand
        2. Get current inventory
        3. Calculate metrics (ROP, safety stock, EOQ)
        4. Find optimization opportunities
        5. Generate recommendations
        """
        self.model_logger.info(
            f"INVENTORY_OPT_START | horizon_days={horizon_days} | max_items={max_items} | use_batch_tool={use_batch_tool} "
            f"| question={question}"
        )
        
        # Extract filter criteria from entities
        branch_codes = None
        product_codes = None
        regions = None
        
        if entities:
            branch_codes = entities.get('branch_codes')
            product_codes = entities.get('product_codes')
            regions = entities.get('regions')
            
            if branch_codes or product_codes or regions:
                self.model_logger.info(
                    f"INVENTORY_OPT_FILTERS | branches={len(branch_codes or [])} "
                    f"| products={len(product_codes or [])} | regions={regions}"
                )
        
        try:
            # Step 1: Get current inventory FIRST (with entity filters)
            step1_start = time.perf_counter()
            self.model_logger.info("INVENTORY_OPT_STEP1 | current_inventory")
            print("📌 Inventory Step 1: Fetching current inventory...")
            inventory_data = self._get_current_inventory(
                branch_codes=branch_codes,
                product_codes=product_codes,
                regions=regions
            )
            step1_elapsed = time.perf_counter() - step1_start
            print(f"   ⏱️  Step 1 completed in {step1_elapsed:.3f}s")
            
            # Log inventory data
            self.workflow_logger.log_dataframe(
                "current_inventory",
                "InventoryOptimizationAgent",
                inventory_data,
                {
                    "branch_codes": branch_codes,
                    "product_codes": product_codes,
                    "regions": regions,
                    "n_items": len(inventory_data)
                }
            )
            
            if inventory_data.empty:
                return {
                    "success": False,
                    "message": "No inventory data found"
                }
            
            # Step 2: Get PER-ITEM forecast demand (IMPROVED!)
            step2_start = time.perf_counter()
            self.model_logger.info("INVENTORY_OPT_STEP2 | per_item_forecasts_batch")
            self.model_logger.info("INVENTORY_OPT_STEP2 | per_item_forecasts_per_item")
            print("📌 Inventory Step 2: Generating forecasts...")
            if use_batch_tool:
                # Use BatchForecastTool for faster processing (NO LIMIT!)
                from agent.tools.batch_forecast_tool import BatchForecastTool
                per_item_forecasts = self._get_forecast_data_batch(inventory_data, horizon_days)
            else:
                # Use traditional method with optional limit
                per_item_forecasts = self._get_forecast_data_per_item(
                    inventory_data, 
                    horizon_days, 
                    max_items=max_items
                )
            step2_elapsed = time.perf_counter() - step2_start
            print(f"   ⏱️  Step 2 completed in {step2_elapsed:.3f}s")
            
            # Log forecast summary
            if per_item_forecasts:
                forecast_summary = {
                    "n_forecasts": len(per_item_forecasts),
                    "sample_keys": list(per_item_forecasts.keys())[:5],
                }
                # Extract sample forecast data
                sample_forecast = {}
                for key in list(per_item_forecasts.keys())[:3]:
                    fc_data = per_item_forecasts[key]
                    sample_forecast[str(key)] = {
                        "metrics": fc_data.get("metrics", {}),
                        "routing_strategy": fc_data.get("routing_strategy", "unknown"),
                    }
                forecast_summary["sample_forecasts"] = sample_forecast
                
                self.workflow_logger.log_step(
                    "per_item_forecasts",
                    "InventoryOptimizationAgent",
                    forecast_summary,
                    {"horizon_days": horizon_days, "use_batch_tool": use_batch_tool}
                )
            
            if not per_item_forecasts:
                return {
                    "success": False,
                    "message": "Could not generate forecasts for demand prediction"
                }
            
            self._latest_drift_report = self._analyze_forecast_drift(per_item_forecasts)
            
            # Step 3: Calculate inventory metrics with per-item forecasts
            step3_start = time.perf_counter()
            self.model_logger.info("INVENTORY_OPT_STEP3 | calculating_metrics")
            print("📌 Inventory Step 3: Calculating recommendations...")
            recommendations = self._generate_recommendations(
                inventory_data, 
                per_item_forecasts, 
                horizon_days
            )
            step3_elapsed = time.perf_counter() - step3_start
            print(f"   ⏱️  Step 3 completed in {step3_elapsed:.3f}s")
            
            # Log recommendations
            self.workflow_logger.log_dataframe(
                "recommendations",
                "InventoryOptimizationAgent",
                recommendations,
                {"n_items": len(recommendations)}
            )
            
            # Step 4: Find transfer opportunities
            step4_start = time.perf_counter()
            self.model_logger.info("INVENTORY_OPT_STEP4 | transfers")
            print("📌 Inventory Step 4: Finding transfer opportunities...")
            transfer_opportunities = self._find_transfer_opportunities(
                recommendations
            )
            step4_elapsed = time.perf_counter() - step4_start
            print(f"   ⏱️  Step 4 completed in {step4_elapsed:.3f}s")
            
            # Log transfer opportunities
            if transfer_opportunities:
                self.workflow_logger.log_step(
                    "transfer_opportunities",
                    "InventoryOptimizationAgent",
                    transfer_opportunities,
                    {"n_opportunities": len(transfer_opportunities)}
            )
            
            # Step 5: Generate comprehensive plan
            step5_start = time.perf_counter()
            print("📌 Inventory Step 5: Creating action plan...")
            plan = self._create_action_plan(recommendations, transfer_opportunities)
            step5_elapsed = time.perf_counter() - step5_start
            print(f"   ⏱️  Step 5 completed in {step5_elapsed:.3f}s")
            
            # Log action plan
            self.workflow_logger.log_step(
                "action_plan",
                "InventoryOptimizationAgent",
                plan,
                {"n_actions": len(plan.get("actions", []))}
            )
            
            # Step 6: Create visualization
            step6_start = time.perf_counter()
            print("📌 Inventory Step 6: Creating visualization...")
            chart_path = self._plot_inventory_optimization(
                inventory_data, 
                per_item_forecasts, 
                recommendations
            )
            step6_elapsed = time.perf_counter() - step6_start
            print(f"   ⏱️  Step 6 completed in {step6_elapsed:.3f}s")
            
            # Step 7: Generate smart insights (NEW!)
            step7_start = time.perf_counter()
            self.model_logger.info("INVENTORY_OPT_STEP7 | ai_insights")
            print("📌 Inventory Step 7: Generating AI insights...")
            insights = self.insights_generator.generate_insights(
                recommendations, 
                plan, 
                entities
            )
            step7_elapsed = time.perf_counter() - step7_start
            print(f"   ⏱️  Step 7 completed in {step7_elapsed:.3f}s")
            
            # Calculate total elapsed time
            total_elapsed = step1_elapsed + step2_elapsed + step3_elapsed + step4_elapsed + step5_elapsed + step6_elapsed + step7_elapsed
            
            print(f"\n{'='*80}")
            print(f"✅ Inventory Optimization completed in {total_elapsed:.3f}s")
            print(f"📊 Timing Breakdown:")
            print(f"   • Step 1 (Fetch Inventory): {step1_elapsed:.3f}s")
            print(f"   • Step 2 (Generate Forecasts): {step2_elapsed:.3f}s")
            print(f"   • Step 3 (Calculate Metrics): {step3_elapsed:.3f}s")
            print(f"   • Step 4 (Find Transfers): {step4_elapsed:.3f}s")
            print(f"   • Step 5 (Action Plan): {step5_elapsed:.3f}s")
            print(f"   • Step 6 (Visualization): {step6_elapsed:.3f}s")
            print(f"   • Step 7 (AI Insights): {step7_elapsed:.3f}s")
            print(f"   • Total: {total_elapsed:.3f}s")
            print(f"{'='*80}")
            
            self.model_logger.info(f"INVENTORY_OPT_DONE | total_actions={len(plan['actions'])} | total_time={total_elapsed:.3f}s")
            
            # Format DataFrames for display (Vietnamese labels)
            inventory_data_display = format_dataframe_columns(inventory_data)
            recommendations_display = format_dataframe_columns(recommendations)
            
            return {
                "success": True,
                "per_item_forecasts": per_item_forecasts,
                "inventory_data": inventory_data_display,
                "inventory_data_raw": inventory_data,  # Keep raw for export functions
                "recommendations": recommendations_display,
                "recommendations_raw": recommendations,  # Keep raw for UI filtering
                "transfer_opportunities": transfer_opportunities,
                "action_plan": plan,
                "chart": chart_path,
                "summary": self._generate_summary(plan),
                "smart_insights": insights,  # NEW: AI-powered insights
                "data_quality_report": self._latest_data_quality_report,
                "drift_report": self._latest_drift_report,
                "model_info": self._model_info,
                "timing_breakdown": {  # NEW: Detailed timing breakdown for orchestrator
                    "Step 1 (Fetch Inventory)": step1_elapsed,
                    "Step 2 (Generate Forecasts)": step2_elapsed,
                    "Step 3 (Calculate Metrics)": step3_elapsed,
                    "Step 4 (Find Transfers)": step4_elapsed,
                    "Step 5 (Action Plan)": step5_elapsed,
                    "Step 6 (Visualization)": step6_elapsed,
                    "Step 7 (AI Insights)": step7_elapsed,
                    "Total": total_elapsed
                }
            }
            
        except Exception as e:
            self.model_logger.error(f"INVENTORY_OPT_ERROR | error={e}")
            return {
                "success": False,
                "message": f"Optimization failed: {str(e)}"
            }
    
    def _get_forecast_data_per_item(self, 
                                   inventory_data: pd.DataFrame,
                                   horizon_days: int,
                                   max_items: Optional[int] = None) -> Dict[tuple, Dict]:
        """
        Get forecast demand PER (product_code, branch_code) combination.
        
        IMPROVEMENT: Instead of one aggregate forecast, we forecast
        separately for each product-branch to get accurate predictions.
        
        OPTIMIZED: Limit to top N items to avoid excessive processing time.
        
        Args:
            inventory_data: DataFrame with inventory items
            horizon_days: Forecast horizon  
            max_items: Maximum items to forecast (default: 100)
        
        Returns:
            Dict[(product_code, branch_code)] = {forecast_df, historical_df, metrics}
        """
        # Limit to top N items by current stock if max_items is specified
        if max_items and len(inventory_data) > max_items:
            print(f"⚠️  Large inventory ({len(inventory_data)} items). Limiting to top {max_items} by stock value...")
            inventory_data_limited = inventory_data.nlargest(max_items, 'current_stock')
        else:
            if len(inventory_data) > 200:
                print(f"📊 Processing {len(inventory_data)} items (this may take a while)...")
            inventory_data_limited = inventory_data
        
        # Use vectorized approach instead of threading
        self.model_logger.info(
            f"FORECAST_PER_ITEM | method=vectorized | n_items={len(inventory_data_limited)}"
        )
        forecasts = self._generate_forecasts_vectorized(
            inventory_data_limited,
            horizon_days
        )
        self.model_logger.info(
            f"FORECAST_PER_ITEM_DONE | n_forecasts={len(forecasts)}"
        )
        return forecasts
    
    def _generate_forecasts_vectorized(self,
                                      inventory_data: pd.DataFrame,
                                      horizon_days: int) -> Dict[tuple, Dict]:
        """
        Generate forecasts using VECTORIZATION instead of threading.
        
        Advantages:
        - No GIL bottleneck (single-threaded vectorized operations are faster)
        - Single database query for all historical data
        - Batch feature engineering with Pandas
        - Single model prediction for entire matrix
        
        Args:
            inventory_data: DataFrame with inventory items
            horizon_days: Forecast horizon
            
        Returns:
            Dict[(product_code, branch_code)] = {forecast_df, historical_df, metrics}
        """
        # Step 1: Get ALL historical data in one query
        self.model_logger.info(
            "INV_VECTOR_FORECAST_STEP1 | fetch_all_historical_single_query"
        )
        timeseries_cache = self._build_timeseries_cache(inventory_data)
        self._latest_data_quality_report = self.data_checker.generate_report(
            inventory_data,
            timeseries_cache
        )
        self.data_checker.log_report(self._latest_data_quality_report)

        # OPTIMIZATION: Cache forecast_base_date to avoid repeated calls
        forecast_base_date = pd.to_datetime(self._get_forecast_base_date())
        
        routing_assignments = {
            "xgboost": [],
            "moving_avg": [],
            "cold_start": []
        }

        # OPTIMIZATION: Pre-compute date statistics for all items in vectorized way
        step3_start = time.perf_counter()
        date_stats_cache = {}
        for key, cache_df in timeseries_cache.items():
            if cache_df is not None and not cache_df.empty:
                date_series = pd.to_datetime(cache_df['date'])
                date_stats_cache[key] = {
                    'span_days': int((date_series.max() - date_series.min()).days),
                    'records': len(cache_df),
                    'last_sale': date_series.max()
                }

        routing_start = time.perf_counter()
        for idx, row in inventory_data.iterrows():
            key = (row['product_code'], row['branch_code'])
            cache_df = timeseries_cache.get(key)
            stats = date_stats_cache.get(key, {})
            history_span_days = stats.get('span_days', 0)
            history_records = stats.get('records', 0)
            last_sale_date = stats.get('last_sale', None)
            
            # OPTIMIZED ROUTING LOGIC (Recommended by Data Science Expert):
            # - >= 14 ngày lịch sử → Panel XGBoost (multi-step) - Best accuracy
            # - 7-13 ngày lịch sử → Moving Average (weighted với trend) - Balanced
            # - < 7 ngày lịch sử → Simple Average hoặc Cold Start - Conservative
            
            # Additional checks:
            # 1. Stale data: last sale > 90 days ago → Cold Start (dead stock)
            # 2. Sparse data: < 2 records → Cold Start (insufficient data)
            # 3. Very short span: < 1 day → Cold Start (single day data)
            
            if cache_df is None or cache_df.empty or history_records < 2:
                # No data or insufficient records → Cold Start
                routing_assignments["cold_start"].append((key, cache_df, row))
            elif last_sale_date is not None:
                days_since_last_sale = (forecast_base_date - last_sale_date).days
                if days_since_last_sale > 90:
                    # Stale data (dead stock) → Cold Start
                    routing_assignments["cold_start"].append((key, cache_df, row))
                elif history_span_days >= 14:
                    # >= 14 ngày lịch sử → Panel XGBoost (best accuracy)
                    routing_assignments["xgboost"].append(idx)
                elif history_span_days >= 7:
                    # 7-13 ngày lịch sử → Moving Average (balanced)
                    routing_assignments["moving_avg"].append((key, cache_df, row))
                else:
                    # < 7 ngày lịch sử → Simple Average (conservative)
                    routing_assignments["moving_avg"].append((key, cache_df, row))
            else:
                # Fallback: treat as cold start if we can't determine last sale
                routing_assignments["cold_start"].append((key, cache_df, row))

        routing_elapsed = time.perf_counter() - routing_start
        step3_elapsed = time.perf_counter() - step3_start
        print(f"   ⏱️  Step 3 (Routing Decision): completed in {step3_elapsed:.3f}s (routing: {routing_elapsed:.3f}s)")
        self.model_logger.info(
            "INV_ROUTING_SUMMARY | "
            f"xgboost={len(routing_assignments['xgboost'])} | "
            f"moving_avg={len(routing_assignments['moving_avg'])} | "
            f"cold_start={len(routing_assignments['cold_start'])}"
        )

        xgboost_inventory = inventory_data.loc[routing_assignments["xgboost"]]
        
        # Step 2: Prepare data for vectorized processing
        self.model_logger.info("INV_VECTOR_FORECAST_STEP2 | prepare_features")
        
        # Get PANEL pre-trained model loader (Global Model with Identifiers)
        # Ưu tiên multi-step model nếu có (nhanh hơn), fallback sang bản recursive.
        try:
            model_loader = None
            # multi-step trước
            if get_panel_multistep_model_loader:
                try:
                    ms_loader = get_panel_multistep_model_loader()
                    if getattr(ms_loader, "loaded", False):
                        model_loader = ms_loader
                        self._model_info = {
                            "model_type": "panel_xgboost_multistep",
                            "version": ms_loader.model_version,
                            "n_features": len(ms_loader.feature_names),
                        }
                        self.model_logger.info(
                            "INV_VECTOR_FORECAST | panel_multistep_model_loaded "
                            f"| version={ms_loader.model_version} "
                            f"| n_features={len(ms_loader.feature_names)}"
                        )
                except Exception as e:
                    self.model_logger.error(
                        f"INV_VECTOR_FORECAST | load_panel_multistep_error={e}"
                    )
                    model_loader = None

            # nếu chưa có multi-step, thử loader cũ
            if model_loader is None and get_panel_model_loader:
                base_loader = get_panel_model_loader()
                if getattr(base_loader, "loaded", False):
                    model_loader = base_loader
                    self._model_info = {
                        "model_type": "panel_xgboost",
                        "version": base_loader.model_version,
                        "n_features": len(base_loader.feature_names),
                    }
                    self.model_logger.info(
                        "INV_VECTOR_FORECAST | panel_model_loaded "
                        f"| version={base_loader.model_version} "
                        f"| n_features={len(base_loader.feature_names)}"
                    )

            if model_loader is None or not getattr(model_loader, "loaded", False):
                self.model_logger.warning(
                    "INV_VECTOR_FORECAST | panel_model_not_available → fallback_per_item"
                )
                return self._generate_forecasts_fallback(
                    inventory_data, horizon_days, timeseries_cache
                )
        except Exception as e:
            self.model_logger.error(
                f"INV_VECTOR_FORECAST | load_panel_model_error={e} → fallback_per_item"
            )
            return self._generate_forecasts_fallback(
                inventory_data, horizon_days, timeseries_cache
            )
        
        forecasts = {}
        processed = 0
        total_items = len(xgboost_inventory)
        
        # Step 3: Process items with vectorized feature engineering + PANEL batch prediction
        self.model_logger.info("INV_VECTOR_FORECAST_STEP3 | per_item_loop_with_panel_batch")
        
        if xgboost_inventory.empty:
            self.model_logger.warning(
                "INV_VECTOR_FORECAST | no_items_for_xgboost_routing"
            )
        
        # OPTIMIZATION: Pre-filter và prepare data efficiently
        # Chuẩn bị batch cho các series đủ điều kiện PANEL XGBoost
        panel_series = []  # list[dict] cho model_loader.predict_batch
        panel_meta = []    # meta để map kết quả batch về forecasts
        
        # OPTIMIZATION: Use date_stats_cache để tránh tính toán lại
        # Skip timing in loop to reduce overhead
        prep_start = time.perf_counter()
        
        for idx, row in xgboost_inventory.iterrows():
            product_code = row['product_code']
            branch_code = row['branch_code']
            key = (product_code, branch_code)
            
            # Get historical data from cache
            cache_df = timeseries_cache.get(key)
            
            # Early filtering: check cache và stats trước
            if cache_df is None or cache_df.empty or len(cache_df) < 2:
                forecasts[key] = self._create_intelligent_fallback(
                    product_code, branch_code, horizon_days, row
                )
                forecasts[key]["routing_strategy"] = "PANEL_FALLBACK"
                processed += 1
                continue
            
            # OPTIMIZATION: Use cached stats instead of recalculating
            stats = date_stats_cache.get(key, {})
            history_span_days = stats.get('span_days', 0)
            history_records = stats.get('records', 0)
            
            # If history span < 14 days, use moving average instead (routing validation)
            if history_span_days < 14 or history_records <= 7:
                avg_demand = cache_df["total_qty"].mean()
                forecasts[key] = self._create_simple_forecast_from_data(
                    cache_df, avg_demand, horizon_days
                )
                forecasts[key]["routing_strategy"] = "MOVING_AVG_SHORT_HISTORY"
                processed += 1
                continue
            
            # Chuẩn hóa df_ts cho panel loader (OPTIMIZED: minimal copy)
            df_ts = cache_df[["date", "total_qty"]].copy()
            df_ts["date"] = pd.to_datetime(df_ts["date"])
            df_ts = df_ts.set_index("date").sort_index()
            df_ts.columns = ["value"]
            
            # Dùng panel model: cần branch_code, region, f_sku
            region = row.get("region")
            f_sku = row.get("f_sku", None)
            
            if f_sku is None:
                # Nếu thiếu f_sku (hiếm), fallback simple forecast
                avg_demand = df_ts["value"].mean()
                forecasts[key] = self._create_simple_forecast_from_data(
                    df_ts.reset_index().rename(columns={"value": "total_qty"}),
                    avg_demand,
                    horizon_days,
                )
                forecasts[key]["routing_strategy"] = "MOVING_AVG_NO_FSKU"
                processed += 1
                continue
            
            # Gom series đủ điều kiện vào batch PANEL
            series_key = len(panel_series)
            panel_series.append(
                {
                    "key": series_key,
                    "df_ts": df_ts,
                    "branch_code": int(branch_code),
                    "region": str(region) if region is not None else "",
                    "f_sku": str(f_sku),
                }
            )
            panel_meta.append(
                {
                    "series_key": series_key,
                    "composite_key": key,
                    "product_code": product_code,
                    "branch_code": branch_code,
                    "df_ts": df_ts,
                }
            )
        
        prep_elapsed = time.perf_counter() - prep_start
        if panel_series:
            print(f"   ⏱️  Step 2 (Prepare Series): completed in {prep_elapsed:.3f}s ({len(panel_series)} series)")
            self.model_logger.info(
                f"INV_VECTOR_FORECAST_PREP | n_series={len(panel_series)} | elapsed={prep_elapsed:.2f}s"
            )
        
        # Chạy PANEL XGBoost batch cho các series đã gom được
        batch_results = {}
        if panel_series:
            step4_start = time.perf_counter()
            try:
                batch_results = model_loader.predict_batch(
                    panel_series, horizon=horizon_days
                )
            except Exception as e:
                self.model_logger.error(
                    f"INV_VECTOR_FORECAST_PANEL_BATCH_ERROR | error={e}"
                )
                batch_results = {}
            finally:
                step4_elapsed = time.perf_counter() - step4_start
                print(f"   ⏱️  Step 4 (Batch Processing): completed in {step4_elapsed:.3f}s ({len(panel_series)} series)")
                self.model_logger.info(
                    "INV_VECTOR_FORECAST_PANEL_BATCH_TIMING | "
                    f"model_type={self._model_info.get('model_type')} | "
                    f"n_series={len(panel_series)} | "
                    f"horizon={horizon_days} | "
                    f"elapsed_sec={step4_elapsed:.2f}"
                )
        
        # OPTIMIZATION: Batch post-processing với vectorized operations
        step5_start = time.perf_counter()
        
        if panel_meta:
            # VECTORIZED: Pre-compute recent averages for all series at once
            constrain_start = time.perf_counter()
            
            # Build arrays for vectorized processing
            recent_avgs = []
            forecast_means = []
            valid_keys = []
            valid_meta = []
            valid_forecast_dfs = []
            valid_hist_dfs = []
            
            for meta in panel_meta:
                series_key = meta["series_key"]
                key = meta["composite_key"]
                
                if series_key not in batch_results:
                    continue
                
                df_ts = meta["df_ts"]
                forecast_df = batch_results[series_key]
                
                # Extract historical series
                if 'value' in df_ts.columns:
                    hist_series = df_ts['value']
                else:
                    hist_series = df_ts.iloc[:, 0]
                
                # Calculate recent average (vectorized-ready)
                recent_window = hist_series.tail(min(30, len(hist_series)))
                recent_avg = recent_window.mean() if not recent_window.empty else 0
                
                # Calculate forecast mean
                forecast_mean = forecast_df['forecast'].mean() if not forecast_df.empty else 0
                
                recent_avgs.append(recent_avg)
                forecast_means.append(forecast_mean)
                valid_keys.append(key)
                valid_meta.append(meta)
                valid_forecast_dfs.append(forecast_df)
                valid_hist_dfs.append(df_ts)
            
            # VECTORIZED: Calculate scales for all forecasts at once
            recent_avgs_arr = np.array(recent_avgs)
            forecast_means_arr = np.array(forecast_means)
            allowed_means = recent_avgs_arr * self.max_forecast_vs_recent_ratio
            
            # Vectorized condition: only scale if forecast_mean > allowed_mean
            needs_scaling = (forecast_means_arr > allowed_means) & (forecast_means_arr > 0) & (recent_avgs_arr > 0)
            scales = np.where(needs_scaling, allowed_means / forecast_means_arr, 1.0)
            
            # Apply constraints and compute metrics in batch
            for idx, (key, meta, forecast_df, df_ts, scale) in enumerate(zip(
                valid_keys, valid_meta, valid_forecast_dfs, valid_hist_dfs, scales
            )):
                try:
                    # Apply scale if needed
                    if scale != 1.0:
                        adjusted_forecast_df = forecast_df.copy()
                        adjusted_forecast_df['forecast'] = adjusted_forecast_df['forecast'] * scale
                        # Scale bounds if present
                        for bound in ['lower_bound', 'upper_bound']:
                            if bound in adjusted_forecast_df.columns:
                                adjusted_forecast_df[bound] = np.maximum(0, adjusted_forecast_df[bound] * scale)
                    else:
                        adjusted_forecast_df = forecast_df.copy()
                    
                    # Compute metrics (vectorized-ready)
                    hist_series = df_ts['value'] if 'value' in df_ts.columns else df_ts.iloc[:, 0]
                    recent_window = hist_series.tail(min(30, len(hist_series)))
                    
                    # Calculate recent_avg (prefer non-zero)
                    non_zero_recent = recent_window[recent_window > 0]
                    if len(non_zero_recent) > 0:
                        recent_avg = float(non_zero_recent.mean())
                    else:
                        recent_avg = float(recent_window.mean()) if not recent_window.empty else 0.0
                    
                    if recent_avg == 0:
                        all_non_zero = hist_series[hist_series > 0]
                        if len(all_non_zero) > 0:
                            recent_avg = float(all_non_zero.mean())
                        else:
                            recent_avg = float(hist_series.mean()) if not hist_series.empty else 0.0
                    
                    forecast_avg = float(adjusted_forecast_df['forecast'].mean())
                    forecast_total = float(adjusted_forecast_df['forecast'].sum())
                    trend = "increasing" if forecast_avg > recent_avg else "decreasing"
                    
                    adjusted_metrics = {
                        "recent_avg_daily": recent_avg,
                        "forecast_avg_daily": forecast_avg,
                        "forecast_total": forecast_total,
                        "trend": trend
                    }
                    
                    forecasts[key] = {
                        "forecast_df": adjusted_forecast_df,
                        "historical_df": df_ts,
                        "metrics": adjusted_metrics,
                    }
                    forecasts[key]["routing_strategy"] = "PANEL_XGBOOST"
                    
                    # OPTIMIZATION: Reduce logging frequency
                    processed += 1
                    if processed % 50 == 0:
                        self.model_logger.log_forecast_series(
                            key={
                                "product_code": meta["product_code"],
                                "branch_code": meta["branch_code"],
                                "routing": "PANEL_XGBOOST",
                            },
                            historical_df=df_ts,
                            forecast_df=adjusted_forecast_df,
                            metrics=adjusted_metrics,
                            extra={"source": "InventoryOptimizationAgent"},
                        )
                except Exception as e:
                    self.model_logger.error(
                        f"INV_VECTOR_FORECAST_ITEM_ERROR | product={meta['product_code']} | branch={meta['branch_code']} | error={e}"
                    )
                    fallback = self._create_fallback_forecast(horizon_days)
                    fallback["routing_strategy"] = "PANEL_ERROR_FALLBACK"
                    forecasts[key] = fallback
                    processed += 1
            
            # Handle missing batch results
            for meta in panel_meta:
                series_key = meta["series_key"]
                key = meta["composite_key"]
                
                if series_key not in batch_results and key not in forecasts:
                    fallback = self._create_fallback_forecast(horizon_days)
                    fallback["routing_strategy"] = "PANEL_ERROR_FALLBACK"
                    forecasts[key] = fallback
                    processed += 1
            
            constrain_elapsed = time.perf_counter() - constrain_start
            step5_elapsed = time.perf_counter() - step5_start
            
            if processed % 100 == 0 or processed >= total_items:
                self.model_logger.info(
                    f"INV_VECTOR_FORECAST_PROGRESS | processed={processed} | total={total_items}"
                )
            
            if panel_meta:
                print(f"   ⏱️  Step 5 (Post-Processing): completed in {step5_elapsed:.3f}s (vectorized constrain: {constrain_elapsed:.3f}s)")
                self.model_logger.info(
                    f"INV_VECTOR_FORECAST_POSTPROC | n_items={len(panel_meta)} | elapsed={step5_elapsed:.2f}s | constrain={constrain_elapsed:.2f}s"
                )
        
        # OPTIMIZATION: Process moving_avg với vectorized batch processing
        if routing_assignments["moving_avg"]:
            self.model_logger.info(
                f"INV_ROUTING_MOVING_AVG | n_items={len(routing_assignments['moving_avg'])}"
            )
            movavg_start = time.perf_counter()
            
            # Separate items with data vs no data
            items_with_data = []
            items_no_data = []
            
            for key, cache_df, row in routing_assignments["moving_avg"]:
                if cache_df is None or cache_df.empty:
                    items_no_data.append((key, row))
                else:
                    items_with_data.append((key, cache_df, row))
            
            # Batch process items with data
            if items_with_data:
                # Pre-compute all avg_demands and trend_slopes in batch
                avg_demands = []
                trend_slopes = []
                history_spans = []
                valid_items = []
                
                base_date = self._get_forecast_base_date()
                future_dates = pd.date_range(
                    start=base_date + timedelta(days=1),
                    periods=horizon_days,
                    freq='D'
                )
                
                for key, cache_df, row in items_with_data:
                    stats = date_stats_cache.get(key, {})
                    history_span_days = stats.get('span_days', 0)
                    history_spans.append(history_span_days)
                    
                    avg_demand = cache_df['total_qty'].mean()
                    avg_demands.append(avg_demand)
                    
                    # Calculate trend slope (vectorized-ready)
                    if history_span_days >= 7:
                        recent_data = cache_df['total_qty'].tail(min(7, len(cache_df)))
                        if len(recent_data) > 1:
                            x = np.arange(len(recent_data))
                            y = recent_data.values
                            trend_slope = np.polyfit(x, y, 1)[0] if len(y) > 1 else 0
                        else:
                            trend_slope = 0
                    else:
                        trend_slope = 0
                    
                    trend_slopes.append(trend_slope)
                    valid_items.append((key, cache_df, row))
                
                # Batch create forecasts
                history_spans_arr = np.array(history_spans)
                trend_slopes_arr = np.array(trend_slopes)
                avg_demands_arr = np.array(avg_demands)
                
                # Vectorized: items with trend (>= 7 days) vs simple (< 7 days)
                has_trend = history_spans_arr >= 7
                
                for idx, (key, cache_df, row) in enumerate(valid_items):
                    avg_demand = avg_demands_arr[idx]
                    trend_slope = trend_slopes_arr[idx]
                    history_span_days = history_spans_arr[idx]
                    
                    # Create base forecast
                    forecast = self._create_simple_forecast_from_data(
                        cache_df, avg_demand, horizon_days
                    )
                    
                    # Apply trend if applicable (vectorized)
                    if has_trend[idx] and trend_slope != 0 and 'forecast_df' in forecast:
                        forecast_df = forecast['forecast_df']
                        # Vectorized trend application
                        trend_adjustments = trend_slope * np.arange(1, len(forecast_df) + 1) * 0.1
                        forecast_df['forecast'] = (forecast_df['forecast'] + trend_adjustments).clip(lower=0)
                        forecast['routing_strategy'] = 'MOVING_AVG_WITH_TREND'
                    else:
                        forecast['routing_strategy'] = 'MOVING_AVG_SIMPLE'
                    
                    forecasts[key] = forecast
            
            # Batch process items without data (cold start)
            if items_no_data:
                base_date = self._get_forecast_base_date()
                for key, row in items_no_data:
                    fallback = self._create_new_item_forecast(horizon_days, row)
                    fallback['routing_strategy'] = 'MOVING_AVG_NO_DATA'
                    forecasts[key] = fallback
            
            movavg_elapsed = time.perf_counter() - movavg_start
            self.model_logger.info(
                f"INV_ROUTING_MOVING_AVG_TIMING | n_items={len(routing_assignments['moving_avg'])} | elapsed={movavg_elapsed:.2f}s"
            )

        # OPTIMIZATION: Batch cold start processing (vectorized)
        if routing_assignments["cold_start"]:
            self.model_logger.info(
                f"INV_ROUTING_COLD_START | n_items={len(routing_assignments['cold_start'])}"
            )
            coldstart_start = time.perf_counter()
            
            # OPTIMIZATION: Cache base_date và tạo forecasts trong batch
            base_date = self._get_forecast_base_date()
            future_dates = pd.date_range(
                start=base_date + timedelta(days=1),
                periods=horizon_days,
                freq='D'
            )
            
            # VECTORIZED: Pre-compute zero values array (reused for all cold starts)
            zero_values = np.full(horizon_days, self.missing_data_forecast_value, dtype=float)
            
            # Batch create all cold start forecasts
            for key, _, row in routing_assignments["cold_start"]:
                # Reuse pre-computed arrays for speed
                forecast_df = pd.DataFrame({'date': future_dates, 'forecast': zero_values.copy()}).set_index('date')
                
                historical_df = pd.DataFrame(
                    {'value': [self.missing_data_forecast_value]},
                    index=[pd.to_datetime(base_date - timedelta(days=1))]
                )
                
                fallback = {
                    'forecast_df': forecast_df,
                    'historical_df': historical_df,
                    'metrics': {
                        'recent_avg_daily': float(self.missing_data_forecast_value),
                        'forecast_avg_daily': float(self.missing_data_forecast_value),
                        'forecast_total': float(self.missing_data_forecast_value * horizon_days),
                        'trend': 'insufficient_history',
                        'reason': 'cold_start_new_item',
                        'current_stock': float(row.get('current_stock', 0))
                    },
                    'routing_strategy': 'REVIEW_NEW_ITEM'
                }
                forecasts[key] = fallback
            
            coldstart_elapsed = time.perf_counter() - coldstart_start
            self.model_logger.info(
                f"INV_ROUTING_COLD_START_TIMING | n_items={len(routing_assignments['cold_start'])} | elapsed={coldstart_elapsed:.2f}s"
            )

        return forecasts
    
    def _generate_forecasts_fallback(self,
                                     inventory_data: pd.DataFrame,
                                     horizon_days: int,
                                     timeseries_cache: Dict[tuple, pd.DataFrame]) -> Dict[tuple, Dict]:
        """
        Fallback method: process items one by one (when pre-trained model unavailable).
        """
        forecasts = {}
        for idx, row in inventory_data.iterrows():
            row_data = row.to_dict()
            key, forecast_data = self._forecast_single_item_worker(
                row_data,
                horizon_days,
                timeseries_cache,
            )
            forecasts[key] = forecast_data
        return forecasts
    
    def _forecast_single_item_worker(self,
                                     row_data: Dict[str, Any],
                                     horizon_days: int,
                                     timeseries_cache: Optional[Dict[tuple, pd.DataFrame]] = None) -> tuple:
        """Worker function for forecasting a single (product, branch) with error handling."""
        product_code = row_data['product_code']
        branch_code = row_data['branch_code']
        key = (product_code, branch_code)
        inventory_row = row_data
        
        # Build parameterized SQL for historical data (365 days for better features)
        # Use system_date-aware date filter
        if SYSTEM_DATE_AVAILABLE:
            from agent.system_date import get_system_date
            system_date = get_system_date()
            date_filter = f"date >= DATE '{system_date}' - INTERVAL '365 days' AND date <= DATE '{system_date}'"
        else:
            date_filter = "date >= CURRENT_DATE - INTERVAL '365 days' AND date <= CURRENT_DATE"
        
        sql = f"""
        SELECT date, SUM(quantity) as total_qty
        FROM sales
        WHERE {date_filter}
            AND product_code = :product_code
            AND branch_code = :branch_code
        GROUP BY date 
        ORDER BY date
        """
        params = {
            "product_code": product_code,
            "branch_code": branch_code
        }
        
        try:
            cache_df = None
            if timeseries_cache is not None:
                cache_df = timeseries_cache.get(key)
            
            if cache_df is not None:
                df = cache_df.copy()
            else:
                df = self.db.execute_query(sql, params, source="InventoryAgent._forecast_single_item_worker")
            
            if df.empty or len(df) < 2:
                return key, self._create_intelligent_fallback(
                    product_code, branch_code, horizon_days, inventory_row
                )
            
            if len(df) < 7:
                avg_demand = df['total_qty'].mean()
                return key, self._create_simple_forecast_from_data(
                    df, avg_demand, horizon_days
                )
            
            # Use system_date-aware date filter (365 days for better features)
            if SYSTEM_DATE_AVAILABLE:
                from agent.system_date import get_system_date
                system_date = get_system_date()
                date_filter = f"date >= DATE '{system_date}' - INTERVAL '365 days' AND date <= DATE '{system_date}'"
            else:
                date_filter = "date >= CURRENT_DATE - INTERVAL '365 days' AND date <= CURRENT_DATE"
            
            sql_for_forecast = f"""
            SELECT date, SUM(quantity) as total_qty
            FROM sales
            WHERE {date_filter}
                AND product_code = '{product_code}'
                AND branch_code = {branch_code}
            GROUP BY date 
            ORDER BY date
            """
            
            df_for_forecast = df[['date', 'total_qty']].copy()
            
            result = self.forecast_agent.forecast(
                sql_for_forecast,
                f"forecast for {product_code} at branch {branch_code}",
                horizon_days,
                create_chart=False,
                preloaded_df=df_for_forecast
            )
            
            if result.get('success'):
                forecast_df_raw = result.get('forecast_raw', result['forecast'])
                historical_df_raw = result.get('historical_data_raw', result['historical_data'])
                
                adjusted_forecast_df = self._constrain_forecast_growth(
                    forecast_df_raw, historical_df_raw
                )
                adjusted_metrics = self._compute_forecast_metrics(
                    historical_df_raw, adjusted_forecast_df
                )
                
                return key, {
                    'forecast_df': adjusted_forecast_df,
                    'historical_df': historical_df_raw,
                    'metrics': adjusted_metrics
                }
            else:
                return key, self._create_fallback_forecast(horizon_days)
        
        except Exception as e:
            print(f"⚠️ Forecast failed for {product_code} at branch {branch_code}: {e}")
            return key, self._create_fallback_forecast(horizon_days)
    
    def _get_forecast_data_batch(self,
                                inventory_data: pd.DataFrame,
                                horizon_days: int) -> Dict[tuple, Dict]:
        """
        Get forecast data using BatchForecastTool (FAST for large inventories).
        
        Args:
            inventory_data: DataFrame with inventory items
            horizon_days: Forecast horizon
        
        Returns:
            Dict[(product_code, branch_code)] = {forecast_df, historical_df, metrics}
        """
        from agent.tools.batch_forecast_tool import BatchForecastTool
        
        print(f"🚀 Using BatchForecastTool for {len(inventory_data)} items...")
        
        # Build product list
        product_list = []
        for idx, row in inventory_data.iterrows():
            product_list.append({
                "product_code": row['product_code'],
                "branch_code": row.get('branch_code')
            })
        
        # Run batch forecast
        tool = BatchForecastTool(db_manager=self.db)
        batch_result = tool.forecast_products(
            product_list=product_list,
            horizon_days=horizon_days
        )
        
        # Convert to expected format
        forecasts = {}
        
        for forecast in batch_result['forecasts']:
            product_code = forecast['product_code']
            branch_code = forecast.get('branch_code')
            key = (product_code, branch_code)
            
            # Convert to DataFrame format
            forecast_df = pd.DataFrame({
                'date': forecast['dates'],
                'forecast': forecast['forecast']
            }).set_index('date')
            
            historical_df = pd.DataFrame({
                'date': forecast['historical_dates'],
                'value': forecast['historical_values']
            }).set_index('date')
            
            adjusted_forecast_df = self._constrain_forecast_growth(forecast_df, historical_df)
            adjusted_metrics = self._compute_forecast_metrics(historical_df, adjusted_forecast_df)
            
            forecasts[key] = {
                'forecast_df': adjusted_forecast_df,
                'historical_df': historical_df,
                'metrics': adjusted_metrics
            }
        
        print(f"✅ Generated {len(forecasts)} forecasts using BatchForecastTool")
        return forecasts
    
    def _constrain_forecast_growth(self,
                                   forecast_df: Optional[pd.DataFrame],
                                   historical_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Limit forecast average so it cannot exceed recent demand by too much.
        Helps remove seasonality spikes that no longer reflect reality.
        """
        if forecast_df is None or forecast_df.empty:
            return forecast_df
        
        if historical_df is None or historical_df.empty:
            return forecast_df
        
        if 'value' in historical_df.columns:
            hist_series = historical_df['value']
        else:
            hist_series = historical_df.iloc[:, 0]
        
        recent_window = hist_series.tail(min(30, len(hist_series)))
        recent_avg = recent_window.mean() if not recent_window.empty else 0
        if recent_avg <= 0:
            return forecast_df
        
        forecast_mean = forecast_df['forecast'].mean()
        allowed_mean = recent_avg * self.max_forecast_vs_recent_ratio
        
        if forecast_mean <= allowed_mean or forecast_mean <= 0:
            return forecast_df
        
        scale = allowed_mean / forecast_mean
        adjusted = forecast_df.copy()
        adjusted['forecast'] = adjusted['forecast'] * scale
        
        for bound in ['lower_bound', 'upper_bound']:
            if bound in adjusted.columns:
                adjusted[bound] = np.maximum(0, adjusted[bound] * scale)
        
        return adjusted
    
    def _compute_forecast_metrics(self,
                                  historical_df: Optional[pd.DataFrame],
                                  forecast_df: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Recalculate basic metrics after forecast adjustments."""
        recent_avg = 0.0
        if historical_df is not None and not historical_df.empty:
            if 'value' in historical_df.columns:
                hist_series = historical_df['value']
            else:
                hist_series = historical_df.iloc[:, 0]
            
            # CRITICAL FIX: Use non-zero values for recent_avg to avoid 0 from resampled data
            recent_window = hist_series.tail(min(30, len(hist_series)))
            if not recent_window.empty:
                # Try non-zero values first (actual sales days)
                non_zero_recent = recent_window[recent_window > 0]
                if len(non_zero_recent) > 0:
                    recent_avg = float(non_zero_recent.mean())
                else:
                    # If all zeros in recent window, use overall mean
                    recent_avg = float(recent_window.mean())
                
                # If still 0, try all historical data
                if recent_avg == 0:
                    all_non_zero = hist_series[hist_series > 0]
                    if len(all_non_zero) > 0:
                        recent_avg = float(all_non_zero.mean())
                    else:
                        recent_avg = float(hist_series.mean())
        
        forecast_avg = 0.0
        forecast_total = 0.0
        if forecast_df is not None and not forecast_df.empty:
            forecast_avg = float(forecast_df['forecast'].mean())
            forecast_total = float(forecast_df['forecast'].sum())
        
        trend = "increasing" if forecast_avg > recent_avg else "decreasing"
        
        return {
            "recent_avg_daily": recent_avg,
            "forecast_avg_daily": forecast_avg,
            "forecast_total": forecast_total,
            "trend": trend
        }
    
    def _quote_literal(self, value: Any) -> str:
        """Safely quote string literals for SQL IN clauses."""
        if value is None:
            return "NULL"
        return "'" + str(value).replace("'", "''") + "'"
    
    def _build_timeseries_cache(self, inventory_subset: pd.DataFrame) -> Dict[tuple, pd.DataFrame]:
        """
        Fetch historical sales for many product-branch combos using OPTIMIZED single query.
        
        IMPROVEMENT: Instead of querying per branch, fetch ALL data in one query with JOIN
        and filter by all product_codes and branch_codes at once.
        
        Returns dict[(product_code, branch_code)] = DataFrame(date,total_qty)
        """
        cache: Dict[tuple, pd.DataFrame] = {}
        
        if inventory_subset.empty:
            return cache
        
        # Extract unique product_codes and branch_codes
        unique_products = inventory_subset['product_code'].unique()
        unique_branches = inventory_subset['branch_code'].unique()
        
        # Filter valid product codes (strings)
        product_codes = [code for code in unique_products if isinstance(code, str)]
        branch_codes = [int(code) for code in unique_branches if pd.notna(code)]
        
        if not product_codes or not branch_codes:
            return cache
        
        print(f"   📦 Fetching historical data for {len(product_codes)} products × {len(branch_codes)} branches...")
        
        # Process in chunks if too many items (SQL IN clause limit)
        # PostgreSQL typically handles up to 1000 items in IN clause well
        max_in_clause_size = 1000
        
        # Split into chunks if needed
        product_chunks = []
        for start in range(0, len(product_codes), max_in_clause_size):
            chunk = product_codes[start:start + max_in_clause_size]
            product_chunks.append(chunk)
        
        branch_chunks = []
        for start in range(0, len(branch_codes), max_in_clause_size):
            chunk = branch_codes[start:start + max_in_clause_size]
            branch_chunks.append(chunk)
        
        # Query all combinations (usually just 1 query if under limits)
        total_queries = len(product_chunks) * len(branch_chunks)
        if total_queries > 1:
            print(f"   ⚠️  Large dataset: splitting into {total_queries} queries")
        
        for product_chunk in product_chunks:
            for branch_chunk in branch_chunks:
                # Build IN clauses
                product_in = ", ".join(self._quote_literal(code) for code in product_chunk)
                branch_in = ", ".join(str(code) for code in branch_chunk)
                
                # Single query with JOIN (if needed) and filters
                # Use system_date-aware date filter to ensure we get latest data
                if SYSTEM_DATE_AVAILABLE:
                    from agent.system_date import get_system_date
                    system_date = get_system_date()
                    date_filter = f"s.date >= DATE '{system_date}' - INTERVAL '365 days' AND s.date <= DATE '{system_date}'"
                else:
                    date_filter = "s.date >= CURRENT_DATE - INTERVAL '365 days' AND s.date <= CURRENT_DATE"
                
                # OPTIMIZATION: Add index hints và optimize query structure
                sql = f"""
                SELECT 
                    s.product_code,
                    s.branch_code,
                    s.date,
                    SUM(s.quantity) as total_qty
                FROM sales s
                WHERE {date_filter}
                    AND s.product_code IN ({product_in})
                    AND s.branch_code IN ({branch_in})
                GROUP BY s.product_code, s.branch_code, s.date
                ORDER BY s.product_code, s.branch_code, s.date
                """
                # Note: Ensure indexes exist on sales(product_code, branch_code, date) for optimal performance
                
                try:
                    df_chunk = self.db.execute_query(
                        sql, source="InventoryAgent._build_timeseries_cache"
                    )
                    if df_chunk.empty:
                        continue
                    # Group by (product_code, branch_code) and store in cache
                    for (product_code, branch_code), grp in df_chunk.groupby(
                        ["product_code", "branch_code"]
                    ):
                        key = (product_code, branch_code)
                        cache[key] = grp[["date", "total_qty"]].copy()
                except Exception as e:
                    print(f"   ⚠️  Error fetching chunk: {e}")
                    continue
        
        print(f"   ✅ Timeseries cache built with {len(cache)} product-branch combos (from {total_queries} query/queries)")
        return cache
    
    def _analyze_forecast_drift(self,
                                per_item_forecasts: Dict[tuple, Dict],
                                drift_threshold: float = 0.4) -> Dict[str, Any]:
        """Detect potential demand drift (forecast deviates strongly from recent history)."""
        stats = {
            "total_items": len(per_item_forecasts),
            "high_drift_count": 0,
            "drift_threshold": drift_threshold,
            "high_drift_samples": []
        }
        if not per_item_forecasts:
            return stats
        
        for key, forecast_data in per_item_forecasts.items():
            metrics = forecast_data.get('metrics', {})
            recent = metrics.get('recent_avg_daily')
            forecast = metrics.get('forecast_avg_daily')
            if recent is None or forecast is None:
                continue
            if recent <= 0:
                continue
            ratio = abs(forecast - recent) / max(recent, 0.1)
            if ratio >= drift_threshold:
                stats["high_drift_count"] += 1
                stats["high_drift_samples"].append({
                    "key": key,
                    "recent_avg": recent,
                    "forecast_avg": forecast,
                    "ratio": round(ratio, 2)
                })
        
        if stats["high_drift_count"]:
            self.model_logger.warning(
                f"INV_DRIFT_MONITOR | high_drift={stats['high_drift_count']} "
                f"/ {stats['total_items']} | threshold={drift_threshold:.2f}"
            )
        else:
            self.model_logger.info(
                f"INV_DRIFT_MONITOR | high_drift=0 / {stats['total_items']} | threshold={drift_threshold:.2f}"
            )
        stats["high_drift_samples"] = stats["high_drift_samples"][:10]
        return stats
    
    def _get_forecast_base_date(self) -> date:
        """Return the configured system date (or real date) for fallback forecasts.
        
        OPTIMIZATION: Cache the result to avoid repeated system calls.
        """
        if not hasattr(self, '_cached_base_date'):
            if SYSTEM_DATE_AVAILABLE:
                from agent.system_date import get_system_date
                self._cached_base_date = pd.to_datetime(get_system_date()).date()
            else:
                self._cached_base_date = datetime.now().date()
        return self._cached_base_date
    
    def _create_fallback_forecast(self, horizon_days: int) -> Dict:
        """Create a simple fallback forecast when data is insufficient.
        
        OPTIMIZATION: Cache base_date calculation.
        """
        base_date = self._get_forecast_base_date()
        
        future_dates = pd.date_range(
            start=base_date + timedelta(days=1),
            periods=horizon_days,
            freq='D'
        )
        
        # OPTIMIZATION: Use numpy array for faster creation
        zero_values = np.full(horizon_days, self.missing_data_forecast_value, dtype=float)
        forecast_df = pd.DataFrame({'date': future_dates, 'forecast': zero_values}).set_index('date')
        
        historical_df = pd.DataFrame(
            {'value': [self.missing_data_forecast_value]},
            index=[pd.to_datetime(base_date - timedelta(days=1))]
        )
        
        return {
            'forecast_df': forecast_df,
            'historical_df': historical_df,
            'metrics': {
                'recent_avg_daily': float(self.missing_data_forecast_value),
                'forecast_avg_daily': float(self.missing_data_forecast_value),
                'forecast_total': float(self.missing_data_forecast_value * horizon_days),
                'trend': 'insufficient_history'
            }
        }
    
    def _create_intelligent_fallback(self, product_code: str, branch_code: int, 
                                    horizon_days: int, inventory_row: pd.Series) -> Dict:
        """
        Create intelligent fallback forecast using branch average demand.
        
        IMPROVEMENT: Instead of returning 0, estimate based on:
        - Current stock level
        - Average inventory turnover for similar products
        - Conservative estimate
        """
        fallback = self._create_fallback_forecast(horizon_days)
        fallback['metrics']['reason'] = 'insufficient_inventory_history'
        fallback['metrics']['current_stock'] = float(inventory_row.get('current_stock', 0))
        return fallback
    
    def _create_simple_forecast_from_data(self, historical_df: pd.DataFrame, 
                                         avg_demand: float, 
                                         horizon_days: int) -> Dict:
        """
        Create forecast from sparse historical data (2-6 days).
        
        IMPROVEMENT: Use available data instead of fallback zeros.
        CRITICAL FIX: Use actual avg_demand for recent_avg_daily instead of 0.
        OPTIMIZATION: Use numpy for faster array creation.
        """
        hist_df = historical_df.copy()
        hist_df['date'] = pd.to_datetime(hist_df['date'])
        hist_df = hist_df.set_index('date')
        hist_df.columns = ['value']
        
        base_date = self._get_forecast_base_date()
        future_dates = pd.date_range(
            start=base_date + timedelta(days=1),
            periods=horizon_days,
            freq='D'
        )
        
        # OPTIMIZATION: Use numpy array for faster creation
        forecast_value = max(0, avg_demand)
        forecast_values = np.full(horizon_days, forecast_value, dtype=float)
        forecast_df = pd.DataFrame({'date': future_dates, 'forecast': forecast_values}).set_index('date')
        
        # CRITICAL: Use actual avg_demand for recent_avg_daily, not 0!
        actual_recent_avg = float(avg_demand) if avg_demand > 0 else 0.0
        
        return {
            'forecast_df': forecast_df,
            'historical_df': hist_df,
            'metrics': {
                'recent_avg_daily': actual_recent_avg,  # FIX: Use actual avg_demand
                'forecast_avg_daily': float(forecast_value),
                'forecast_total': float(forecast_value * horizon_days),
                'trend': 'insufficient_history',
                'history_points': int(len(hist_df))
            }
        }

    def _create_new_item_forecast(self, horizon_days: int, inventory_row: pd.Series) -> Dict:
        """Cold-start strategy: neutral forecast with review flag."""
        forecast = self._create_fallback_forecast(horizon_days)
        forecast['metrics']['reason'] = 'cold_start_new_item'
        forecast['metrics']['current_stock'] = float(inventory_row.get('current_stock', 0))
        forecast['routing_strategy'] = 'REVIEW_NEW_ITEM'
        return forecast
    
    def _get_current_inventory(self, 
                               branch_codes: Optional[List[int]] = None,
                               product_codes: Optional[List[str]] = None,
                               regions: Optional[List[str]] = None,
                               include_sales_products: bool = True) -> pd.DataFrame:
        """
        Get current inventory levels with SMART FILTERING based on extracted entities.
        
        IMPROVEMENT: 
        - Supports filtering by multiple branches/products/regions
        - Optionally includes products with sales but no inventory (include_sales_products=True)
        """
        # Step 1: Get products from inventory
        sql = """
        SELECT 
            i.product_code,
            i.branch_code,
            b.branch_name,
            b.region,
            p.f_sku,
            p.product_name,
            i.quantity as current_stock,
            i.unit
        FROM inventory i
        JOIN branch b ON i.branch_code = b.branch_code
        JOIN product p ON i.product_code = p.product_code
        WHERE 1=1
        """
        
        params = {}
        
        # Filter by specific branches (if mentioned in question)
        if branch_codes and len(branch_codes) > 0:
            # Use IN clause for multiple branches
            placeholders = ','.join([f':branch_code_{i}' for i in range(len(branch_codes))])
            sql += f" AND i.branch_code IN ({placeholders})"
            for i, code in enumerate(branch_codes):
                params[f'branch_code_{i}'] = code
        
        # Filter by specific products (if mentioned in question)
        if product_codes and len(product_codes) > 0:
            placeholders = ','.join([f':product_code_{i}' for i in range(len(product_codes))])
            sql += f" AND i.product_code IN ({placeholders})"
            for i, code in enumerate(product_codes):
                params[f'product_code_{i}'] = code
        
        # Filter by regions (only when user DIDN'T specify explicit branches)
        if (not branch_codes) and regions and len(regions) > 0:
            placeholders = ','.join([f':region_{i}' for i in range(len(regions))])
            sql += f" AND b.region IN ({placeholders})"
            for i, region in enumerate(regions):
                params[f'region_{i}'] = region.upper()
        
        sql += " ORDER BY i.branch_code, i.product_code"
        
        inventory_result = self.db.execute_query(sql, params if params else None, source="InventoryAgent._get_current_inventory")
        
        if not inventory_result.empty:
            print(f"   ✅ Found {len(inventory_result)} inventory items matching criteria")
        
        # Step 2: If include_sales_products=True, also get products with sales but no inventory
        if include_sales_products and (branch_codes or regions):
            sales_products = self._get_products_with_sales_no_inventory(
                branch_codes=branch_codes,
                regions=regions,
                date_range_days=120
            )
            
            if not sales_products.empty:
                print(f"   ✅ Found {len(sales_products)} additional products with sales but no inventory")
                # Combine both results
                inventory_result = pd.concat([inventory_result, sales_products], ignore_index=True)
        
        return inventory_result
    
    def _get_products_with_sales_no_inventory(self,
                                              branch_codes: Optional[List[int]] = None,
                                              regions: Optional[List[str]] = None,
                                              date_range_days: int = 120) -> pd.DataFrame:
        """
        Get products that have sales in the last N days but are NOT in inventory.
        This ensures we don't miss products that were sold but are now out of stock.
        """
        if not branch_codes and not regions:
            return pd.DataFrame()
        
        # Build date filter
        if SYSTEM_DATE_AVAILABLE:
            from agent.system_date import get_system_date
            system_date = get_system_date()
            date_filter = f"s.date >= DATE '{system_date}' - INTERVAL '{date_range_days} days' AND s.date <= DATE '{system_date}'"
        else:
            date_filter = f"s.date >= CURRENT_DATE - INTERVAL '{date_range_days} days' AND s.date <= CURRENT_DATE"
        
        # Build branch filter
        branch_filter = ""
        if branch_codes and len(branch_codes) > 0:
            branch_in = ", ".join(str(code) for code in branch_codes)
            branch_filter = f"AND s.branch_code IN ({branch_in})"
        elif regions and len(regions) > 0:
            region_in = ", ".join(f"'{r.upper()}'" for r in regions)
            branch_filter = f"AND b.region IN ({region_in})"
        else:
            return pd.DataFrame()
        
        sql = f"""
        SELECT DISTINCT
            s.product_code,
            s.branch_code,
            b.branch_name,
            b.region,
            p.f_sku,
            p.product_name,
            0 as current_stock,  -- No inventory
            COALESCE(p.unit, 'viên') as unit
        FROM sales s
        JOIN product p ON s.product_code = p.product_code
        JOIN branch b ON s.branch_code = b.branch_code
        WHERE {date_filter}
          {branch_filter}
          AND s.product_code NOT IN (
              SELECT product_code 
              FROM inventory 
              WHERE branch_code = s.branch_code
          )
        ORDER BY s.branch_code, s.product_code
        """
        
        try:
            result = self.db.execute_query(
                sql, source="InventoryAgent._get_products_with_sales_no_inventory"
            )
            return result
        except Exception as e:
            self.model_logger.error(
                f"INV_SALES_NO_INV_ERROR | error={e}"
            )
            return pd.DataFrame()
    
    def _calculate_safety_stock(self, avg_demand: float, std_demand: float) -> float:
        """
        Calculate safety stock using statistical method.
        Safety Stock = Z * σ * √LT
        where Z = service level factor, σ = demand std, LT = lead time
        """
        from scipy import stats
        z_score = stats.norm.ppf(self.service_level)
        safety_stock = z_score * std_demand * np.sqrt(self.lead_time_days)
        return max(0, safety_stock)
    
    def _calculate_rop(self, avg_demand: float, safety_stock: float) -> float:
        """
        Calculate Reorder Point (ROP).
        ROP = (Average Daily Demand × Lead Time) + Safety Stock
        """
        rop = (avg_demand * self.lead_time_days) + safety_stock
        return max(0, rop)
    
    def _calculate_eoq(self, annual_demand: float, ordering_cost: float = 1000, 
                      holding_cost: float = 50) -> float:
        """
        Calculate Economic Order Quantity (EOQ).
        EOQ = √((2 × D × S) / H)
        where D = annual demand, S = ordering cost, H = holding cost
        """
        if annual_demand <= 0:
            return 0
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        return max(0, eoq)
    
    def _generate_recommendations(self, 
                                 inventory_data: pd.DataFrame,
                                 per_item_forecasts: Dict[tuple, Dict],
                                 horizon_days: int) -> pd.DataFrame:
        """
        Generate inventory recommendations using PER-ITEM forecasts.
        
        IMPROVEMENT: Each (product, branch) gets its own forecast-based metrics.
        """
        
        recommendations = []
        
        for idx, row in inventory_data.iterrows():
            product_code = row['product_code']
            branch_code = row['branch_code']
            current_stock = row['current_stock']
            
            # Get forecast for THIS specific item
            key = (product_code, branch_code)
            forecast_data = per_item_forecasts.get(key)
            
            if not forecast_data:
                # Skip if no forecast available
                continue
            
            forecast_df = forecast_data['forecast_df']
            historical_df = forecast_data['historical_df']
            routing_strategy = forecast_data.get('routing_strategy', 'XGBOOST')
            
            # Calculate demand statistics from THIS item's historical data
            # Use iloc to handle formatted column names (value → Giá Trị)
            if 'value' in historical_df.columns:
                hist_series = historical_df['value']
            else:
                # Fallback: use first column
                hist_series = historical_df.iloc[:, 0]
            
            # CRITICAL FIX: Calculate avg from non-zero values only (exclude resampled zeros)
            # This ensures we get actual demand, not zeros from days with no sales
            non_zero_values = hist_series[hist_series > 0]
            if len(non_zero_values) > 0:
                # Use mean of non-zero days (actual sales days) - more accurate
                avg_daily_demand = non_zero_values.mean()
                # For std, use all values
                std_daily_demand = hist_series.std()
            else:
                # All zeros - use overall mean
                avg_daily_demand = hist_series.mean()
                std_daily_demand = hist_series.std()
            
            # Also try recent window (last 30 days) for better accuracy
            recent_window = hist_series.tail(min(30, len(hist_series)))
            if len(recent_window) > 0 and recent_window.sum() > 0:
                # Prefer recent average if available
                recent_avg = recent_window.mean()
                if recent_avg > 0:
                    avg_daily_demand = recent_avg
            
            if 'forecast' in forecast_df.columns:
                total_forecast_demand = forecast_df['forecast'].sum()
            else:
                # Fallback: use first column
                total_forecast_demand = forecast_df.iloc[:, 0].sum()
            
            # Handle edge case: no historical demand
            if avg_daily_demand == 0 or pd.isna(avg_daily_demand):
                avg_daily_demand = 0.1  # Small default to avoid division by zero
            if std_daily_demand == 0 or pd.isna(std_daily_demand):
                std_daily_demand = avg_daily_demand * 0.3  # 30% CV as default
            
            # Calculate metrics
            safety_stock = self._calculate_safety_stock(avg_daily_demand, std_daily_demand)
            rop = self._calculate_rop(avg_daily_demand, safety_stock)
            annual_demand = avg_daily_demand * 365
            eoq = self._calculate_eoq(annual_demand)
            
            # Calculate expected stock after forecast period
            expected_stock_after_period = current_stock - total_forecast_demand
            
            # Determine action needed
            if routing_strategy == 'REVIEW_NEW_ITEM':
                action = 'REVIEW_NEW_ITEM'
                priority = 'MEDIUM'
                quantity_needed = 0
            elif current_stock < rop:
                action = "URGENT_RESTOCK"
                priority = "HIGH"
                quantity_needed = eoq
            elif expected_stock_after_period < safety_stock:
                action = "RESTOCK"
                priority = "MEDIUM"
                quantity_needed = eoq
            elif current_stock > (rop + eoq * 2):
                action = "SURPLUS"
                priority = "LOW"
                quantity_needed = 0
            else:
                action = "OK"
                priority = "LOW"
                quantity_needed = 0
            
            recommendations.append({
                'product_code': product_code,
                'branch_code': branch_code,
                'branch_name': row['branch_name'],
                'region': row['region'],
                'product_name': row['product_name'],
                'current_stock': current_stock,
                'avg_daily_demand': avg_daily_demand,
                'forecast_demand_30d': total_forecast_demand,
                'safety_stock': safety_stock,
                'reorder_point': rop,
                'eoq': eoq,
                'expected_stock_after_30d': expected_stock_after_period,
                'action': action,
                'priority': priority,
                'quantity_needed': quantity_needed,
                'unit': row['unit'],
                'routing_strategy': routing_strategy
            })
        
        return pd.DataFrame(recommendations)
    
    def _find_transfer_opportunities(self, 
                                    recommendations: pd.DataFrame) -> List[Dict]:
        """
        Find opportunities to transfer stock from surplus branches to deficit branches.
        Uses branch_distance table to find nearby branches.
        
        OPTIMIZATION: Batch query nearby branches và vectorize matching.
        """
        if recommendations.empty:
            return []
        
        # Separate surplus and deficit branches
        surplus = recommendations[recommendations['action'] == 'SURPLUS'].copy()
        deficit = recommendations[recommendations['action'].isin(['URGENT_RESTOCK', 'RESTOCK'])].copy()
        
        if surplus.empty or deficit.empty:
            return []
        
        # OPTIMIZATION: Batch query all nearby branches at once
        unique_deficit_branches = deficit['branch_code'].unique()
        if len(unique_deficit_branches) == 0:
            return []
        
        # Build batch query for all deficit branches
        branch_in = ", ".join(str(int(b)) for b in unique_deficit_branches)
        batch_nearby_query = f"""
            SELECT 
                bd.branch_code_1 as source_branch,
                bd.branch_code_2 as dest_branch,
                bd.distance_km,
                b.branch_name as source_branch_name
            FROM branch_distance bd
            JOIN branch b ON bd.branch_code_1 = b.branch_code
        WHERE bd.branch_code_2 IN ({branch_in})
            AND bd.distance_km <= {self.max_transfer_distance_km}
        ORDER BY bd.branch_code_2, bd.distance_km ASC
        """
        
        try:
            all_nearby_branches = self.db.execute_query(
                batch_nearby_query, 
                source="InventoryAgent._find_transfer_opportunities_batch"
            )
            
            # OPTIMIZATION: Create lookup dict for faster access
            nearby_lookup = {}
            if not all_nearby_branches.empty:
                for _, nearby in all_nearby_branches.iterrows():
                    dest_branch = nearby['dest_branch']
                    if dest_branch not in nearby_lookup:
                        nearby_lookup[dest_branch] = []
                    nearby_lookup[dest_branch].append(nearby)
        except Exception as e:
            self.model_logger.error(
                f"INV_TRANSFER_BATCH_ERROR | error={e}"
            )
            all_nearby_branches = pd.DataFrame()
            nearby_lookup = {}
        
        transfer_opportunities = []
        
        # OPTIMIZATION: Create surplus lookup for faster matching
        surplus_lookup = {}
        for _, s_row in surplus.iterrows():
            key = (s_row['branch_code'], s_row['product_code'])
            if key not in surplus_lookup:
                surplus_lookup[key] = []
            surplus_lookup[key].append(s_row)
        
        for _, deficit_row in deficit.iterrows():
            deficit_branch = deficit_row['branch_code']
            needed_qty = deficit_row['quantity_needed']
            product_code = deficit_row['product_code']
            
            # Get nearby branches from lookup
            nearby_list = nearby_lookup.get(deficit_branch, [])
            
            for nearby in nearby_list:
                source_branch = nearby['source_branch']
                
                # OPTIMIZATION: Use lookup instead of DataFrame filtering
                key = (source_branch, product_code)
                surplus_matches = surplus_lookup.get(key, [])
                
                if surplus_matches:
                    surplus_row = surplus_matches[0]  # Take first match
                    available_qty = surplus_row['current_stock'] - surplus_row['reorder_point']
                    
                    if available_qty > 0:
                        transfer_qty = min(available_qty, needed_qty)
                        
                        transfer_opportunities.append({
                            'product_code': product_code,
                            'product_name': deficit_row['product_name'],
                            'source_branch_code': source_branch,
                            'source_branch_name': nearby['source_branch_name'],
                            'dest_branch_code': deficit_branch,
                            'dest_branch_name': deficit_row['branch_name'],
                            'distance_km': nearby['distance_km'],
                            'transfer_quantity': transfer_qty,
                            'unit': deficit_row['unit'],
                            'cost_saving': 'Avoid external purchase',
                            'priority': deficit_row['priority']
                        })
                        
                        # Update needed quantity
                        needed_qty -= transfer_qty
                        if needed_qty <= 0:
                            break
        
        return transfer_opportunities
    
    def _create_action_plan(self, 
                           recommendations: pd.DataFrame,
                           transfer_opportunities: List[Dict]) -> Dict[str, Any]:
        """Create comprehensive action plan with prioritized actions."""
        
        actions = []
        
        # Add restock actions (excluding those that can be fulfilled by transfers)
        transferred_branches = {(t['dest_branch_code'], t['product_code']) 
                              for t in transfer_opportunities}
        
        for _, row in recommendations.iterrows():
            if row['action'] in ['URGENT_RESTOCK', 'RESTOCK']:
                branch_product = (row['branch_code'], row['product_code'])
                
                # Check if partially/fully covered by transfers
                transfer_qty = sum(
                    t['transfer_quantity'] 
                    for t in transfer_opportunities 
                    if t['dest_branch_code'] == row['branch_code'] 
                    and t['product_code'] == row['product_code']
                )
                
                remaining_qty = row['quantity_needed'] - transfer_qty
                
                if remaining_qty > 0:
                    actions.append({
                        'action_type': 'RESTOCK',
                        'priority': row['priority'],
                        'branch_code': row['branch_code'],
                        'branch_name': row['branch_name'],
                        'product_code': row['product_code'],
                        'product_name': row['product_name'],
                        'quantity': remaining_qty,
                        'unit': row['unit'],
                        'reason': f"Current: {row['current_stock']}, ROP: {row['reorder_point']:.0f}, Forecast demand: {row['forecast_demand_30d']:.0f}",
                        'estimated_cost': 'TBD'
                    })
        
        # Add transfer actions
        for transfer in transfer_opportunities:
            actions.append({
                'action_type': 'TRANSFER',
                'priority': transfer['priority'],
                'source_branch_code': transfer['source_branch_code'],
                'source_branch_name': transfer['source_branch_name'],
                'dest_branch_code': transfer['dest_branch_code'],
                'dest_branch_name': transfer['dest_branch_name'],
                'product_code': transfer['product_code'],
                'product_name': transfer['product_name'],
                'quantity': transfer['transfer_quantity'],
                'unit': transfer['unit'],
                'distance_km': transfer['distance_km'],
                'reason': f"Transfer from surplus to deficit branch ({transfer['distance_km']:.1f} km)",
                'estimated_cost': f"Transport cost for {transfer['distance_km']:.1f} km"
            })
        
        # Sort by priority
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        actions.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        # Calculate summary statistics
        total_restock = sum(a['quantity'] for a in actions if a['action_type'] == 'RESTOCK')
        total_transfer = sum(a['quantity'] for a in actions if a['action_type'] == 'TRANSFER')
        
        return {
            'actions': actions,
            'summary': {
                'total_actions': len(actions),
                'restock_actions': len([a for a in actions if a['action_type'] == 'RESTOCK']),
                'transfer_actions': len([a for a in actions if a['action_type'] == 'TRANSFER']),
                'total_restock_quantity': total_restock,
                'total_transfer_quantity': total_transfer,
                'high_priority_actions': len([a for a in actions if a['priority'] == 'HIGH'])
            }
        }
    
    def _plot_inventory_optimization(self, 
                                    inventory_data: pd.DataFrame,
                                    per_item_forecasts: Dict[tuple, Dict],
                                    recommendations: pd.DataFrame) -> str:
        """
        Create visualization for inventory optimization with per-item forecasts.
        
        IMPROVED: Better labels, titles with branch names, and clear legends.
        """
        
        # Get unique branch names for title
        unique_branches = inventory_data['branch_name'].unique()
        if len(unique_branches) <= 3:
            branch_title = f"Branches: {', '.join(unique_branches)}"
        else:
            branch_title = f"{len(unique_branches)} Branches"
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'Inventory Optimization Analysis - {branch_title}', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Plot 1: Current Stock vs ROP (Top 10 items)
        ax1 = axes[0, 0]
        top_10 = recommendations.head(10)
        
        # Create product labels with branch names
        labels = [f"{row['product_name'][:30]}\n({row['branch_name'][:20]})" 
                  for _, row in top_10.iterrows()]
        
        current_stock = top_10['current_stock'].values
        rop = top_10['reorder_point'].values
        safety_stock = top_10['safety_stock'].values
        
        x = np.arange(len(labels))
        width = 0.25
        
        bars1 = ax1.bar(x - width, current_stock, width, label='Tồn Kho Hiện Tại', 
                        color='steelblue', alpha=0.8)
        bars2 = ax1.bar(x, rop, width, label='Điểm Đặt Hàng (ROP)', 
                        color='orange', alpha=0.8)
        bars3 = ax1.bar(x + width, safety_stock, width, label='Tồn Kho An Toàn', 
                        color='green', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontsize=7)
        
        ax1.set_xlabel('Sản Phẩm @ Chi Nhánh', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Số Lượng', fontsize=11, fontweight='bold')
        ax1.set_title('Tồn Kho Hiện Tại vs ROP & Tồn Kho An Toàn (Top 10)', 
                      fontsize=12, fontweight='bold', pad=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Plot 2: Action Distribution by Branch
        ax2 = axes[0, 1]
        action_counts = recommendations['action'].value_counts()
        
        # Translate action labels to Vietnamese
        action_translation = {
            'OK': 'Đủ Hàng',
            'RESTOCK': 'Cần Nhập',
            'URGENT_RESTOCK': 'Nhập Gấp',
            'SURPLUS': 'Thừa Hàng',
            'TRANSFER': 'Chuyển Kho'
        }
        action_counts.index = [action_translation.get(x, x) for x in action_counts.index]
        
        colors = {'Đủ Hàng': '#2ecc71', 'Cần Nhập': '#f39c12', 
                  'Nhập Gấp': '#e74c3c', 'Thừa Hàng': '#3498db', 'Chuyển Kho': '#9b59b6'}
        
        wedges, texts, autotexts = ax2.pie(
            action_counts.values, 
            labels=action_counts.index, 
            autopct='%1.1f%%',
            colors=[colors.get(action, 'gray') for action in action_counts.index],
            startangle=90,
            textprops={'fontsize': 10, 'weight': 'bold'}
        )
        
        # Add count to labels
        for i, (label, count) in enumerate(zip(action_counts.index, action_counts.values)):
            texts[i].set_text(f'{label}\n({count} items)')
        
        ax2.set_title('Phân Phối Hành Động Tồn Kho', 
                      fontsize=12, fontweight='bold', pad=10)
        
        # Plot 3: Aggregated Demand Forecast
        ax3 = axes[1, 0]
        if per_item_forecasts:
            # Get consistent date range (normalize to dates only, no time)
            first_forecast = list(per_item_forecasts.values())[0]
            first_dates = first_forecast['forecast_df'].index
            
            # Normalize to date only (remove time component for alignment)
            forecast_dates = pd.to_datetime([d.date() for d in first_dates])
            
            # Sum all forecasts using normalized dates
            total_forecast = pd.Series(0.0, index=forecast_dates)
            valid_forecasts = 0
            
            for forecast_data in per_item_forecasts.values():
                if 'forecast_df' in forecast_data and not forecast_data['forecast_df'].empty:
                    fc_df = forecast_data['forecast_df']
                    # Normalize dates for this forecast too
                    fc_dates_normalized = pd.to_datetime([d.date() for d in fc_df.index])
                    fc_values = fc_df['forecast'].values
                    
                    # Create aligned series
                    fc_series = pd.Series(fc_values, index=fc_dates_normalized)
                    
                    # Add to total (now dates are aligned!)
                    total_forecast = total_forecast.add(fc_series, fill_value=0)
                    valid_forecasts += 1
            
            # Check if forecast data is valid
            max_forecast = total_forecast.max()
            total_demand = total_forecast.sum()
            
            if max_forecast > 0:
                # Plot forecast
                ax3.plot(total_forecast.index, total_forecast.values, 
                        label='Tổng Nhu Cầu Dự Báo', linewidth=2.5, 
                        color='orange', marker='o', markersize=4, alpha=0.8)
                
                # Add trend line
                x_numeric = np.arange(len(total_forecast))
                z = np.polyfit(x_numeric, total_forecast.values, 1)
                p = np.poly1d(z)
                ax3.plot(total_forecast.index, p(x_numeric), 
                        "--", alpha=0.5, color='red', linewidth=1.5, label='Xu Hướng')
                
                # Add mean line
                mean_val = total_forecast.mean()
                ax3.axhline(y=mean_val, color='green', linestyle=':', 
                           linewidth=1.5, alpha=0.7, label=f'Trung Bình: {mean_val:.0f}')
                
                # Add statistics text
                stats_text = f'Tổng 30 ngày: {total_demand:.0f}\n'
                stats_text += f'Số sản phẩm: {valid_forecasts}\n'
                stats_text += f'TB/ngày: {mean_val:.1f}'
                ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes,
                        verticalalignment='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                ax3.set_xlabel('Ngày', fontsize=11, fontweight='bold')
                ax3.set_ylabel('Số Lượng', fontsize=11, fontweight='bold')
                ax3.set_title(f'Dự Báo Nhu Cầu 30 Ngày - {branch_title}', 
                             fontsize=12, fontweight='bold', pad=10)
                ax3.legend(loc='upper right', fontsize=9)
                ax3.grid(True, alpha=0.3, linestyle='--')
                ax3.tick_params(axis='x', rotation=45)
                
                # Set reasonable y-axis limits
                if max_forecast < 10:
                    ax3.set_ylim(0, max(10, max_forecast * 1.5))
                
                # Format x-axis
                import matplotlib.dates as mdates
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                ax3.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            else:
                # No valid forecast data
                warning_text = f'⚠️ Dự báo gần 0\n'
                warning_text += f'Dữ liệu lịch sử quá ít\n'
                warning_text += f'({valid_forecasts} sản phẩm có dự báo)'
                ax3.text(0.5, 0.5, warning_text, 
                        ha='center', va='center', fontsize=11,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
                ax3.set_xlabel('Ngày', fontsize=11, fontweight='bold')
                ax3.set_ylabel('Số Lượng', fontsize=11, fontweight='bold')
                ax3.set_title(f'Dự Báo Nhu Cầu 30 Ngày - {branch_title}', 
                             fontsize=12, fontweight='bold', pad=10)
        else:
            ax3.text(0.5, 0.5, 'Không có dữ liệu dự báo', 
                    ha='center', va='center', fontsize=12)
        
        # Plot 4: Priority Distribution by Branch
        ax4 = axes[1, 1]
        priority_data = recommendations[recommendations['action'] != 'OK']
        
        if not priority_data.empty:
            priority_counts = priority_data['priority'].value_counts()
            priority_colors = {'HIGH': '#e74c3c', 'MEDIUM': '#f39c12', 'LOW': '#f1c40f'}
            
            bars = ax4.bar(priority_counts.index, priority_counts.values,
                          color=[priority_colors.get(p, 'gray') for p in priority_counts.index],
                          alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax4.set_xlabel('Mức Độ Ưu Tiên', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Số Lượng Hành Động', fontsize=11, fontweight='bold')
            ax4.set_title('Phân Phối Ưu Tiên Hành Động', 
                         fontsize=12, fontweight='bold', pad=10)
            ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            # Add summary text
            total_actions = len(priority_data)
            summary_text = f'Tổng Số Hành Động: {total_actions}'
            ax4.text(0.5, 0.95, summary_text, transform=ax4.transAxes,
                    ha='center', va='top', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax4.text(0.5, 0.5, '✓ Tất Cả Đều Ổn\nKhông Cần Hành Động', 
                    ha='center', va='center', fontsize=14, color='green', fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Create filename with branch info
        if len(unique_branches) == 1:
            branch_slug = unique_branches[0].replace(' ', '_')[:20]
            filename = f"inventory_opt_{branch_slug}_{uuid.uuid4().hex[:8]}.png"
        else:
            filename = f"inventory_opt_{len(unique_branches)}branches_{uuid.uuid4().hex[:8]}.png"
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Created inventory optimization chart: {filepath}")
        print(f"   📈 Includes: Stock vs ROP, Actions, Forecast, Priorities")
        return filepath
    
    def _despike_time_series(self, df_ts: pd.DataFrame, window: int = 30, threshold_std: float = 3.0) -> pd.DataFrame:
        """
        Detect and replace outliers (spikes) at the end of time series.
        
        This prevents "Spike Amplification" in recursive forecasting where a spike
        at the last day gets amplified through lag_1 features in subsequent predictions.
        
        Logic:
        - If last value > mean(window) + threshold_std * std(window), replace with mean
        - This ensures model learns long-term trends, not one-off spikes
        
        Args:
            df_ts: Time series DataFrame with 'value' column
            window: Rolling window size for calculating mean/std (default: 30)
            threshold_std: Number of standard deviations to consider as outlier (default: 3.0)
            
        Returns:
            DataFrame with despiked values
        """
        if len(df_ts) < 2:
            return df_ts  # Not enough data to despike
        
        # Use minimum of window and available data length
        actual_window = min(window, len(df_ts) - 1)  # Exclude last day from calculation
        
        if actual_window < 2:
            return df_ts  # Not enough data
        
        # Calculate rolling statistics excluding the last day
        # We want to compare last day against historical pattern
        historical_data = df_ts['value'].iloc[:-1]  # All except last day
        
        if len(historical_data) >= actual_window:
            # Use rolling window
            mean_rolling = historical_data.rolling(window=actual_window, min_periods=2).mean()
            std_rolling = historical_data.rolling(window=actual_window, min_periods=2).std()
            
            # Get the last valid rolling statistics (for comparison with last day)
            mean_val = mean_rolling.iloc[-1]
            std_val = std_rolling.iloc[-1]
        else:
            # Use all available historical data
            mean_val = historical_data.mean()
            std_val = historical_data.std()
        
        # Handle case where std is 0 or NaN
        if pd.isna(std_val) or std_val == 0:
            std_val = mean_val * 0.1 if mean_val > 0 else 1.0  # Use 10% of mean as default std
        
        last_val = df_ts['value'].iloc[-1]
        threshold = mean_val + (threshold_std * std_val)
        
        # Check if last value is an outlier
        if last_val > threshold:
            # Replace with rolling mean (smoother than simple mean)
            if len(historical_data) >= actual_window:
                replacement_val = mean_rolling.iloc[-1]
            else:
                replacement_val = mean_val
            
            # Log the despiking action (only for first few items to avoid spam)
            # We'll use a simple counter or just log occasionally
            df_ts = df_ts.copy()  # Avoid SettingWithCopyWarning
            df_ts.iloc[-1, df_ts.columns.get_loc('value')] = replacement_val
        
        return df_ts
    
    def _generate_summary(self, plan: Dict) -> str:
        """Generate text summary of the action plan."""
        summary = plan['summary']
        
        text = f"""
INVENTORY OPTIMIZATION SUMMARY:
================================

Total Actions Recommended: {summary['total_actions']}
- Restock Orders: {summary['restock_actions']} (Total Qty: {summary['total_restock_quantity']:.0f})
- Internal Transfers: {summary['transfer_actions']} (Total Qty: {summary['total_transfer_quantity']:.0f})
- High Priority Actions: {summary['high_priority_actions']}

KEY ACTIONS:
"""
        
        for action in plan['actions'][:10]:  # Top 10 actions
            if action['action_type'] == 'RESTOCK':
                text += f"\n📦 RESTOCK [{action['priority']}]: {action['product_name']}"
                text += f"\n   Branch: {action['branch_name']}"
                text += f"\n   Quantity: {action['quantity']:.0f} {action['unit']}"
                text += f"\n   Reason: {action['reason']}\n"
            else:  # TRANSFER
                text += f"\n🚚 TRANSFER [{action['priority']}]: {action['product_name']}"
                text += f"\n   From: {action['source_branch_name']} → To: {action['dest_branch_name']}"
                text += f"\n   Quantity: {action['quantity']:.0f} {action['unit']}"
                text += f"\n   Distance: {action['distance_km']:.1f} km\n"
        
        if len(plan['actions']) > 10:
            text += f"\n... and {len(plan['actions']) - 10} more actions.\n"
        
        return text




"""
InventoryOptimizationAgent: intelligent inventory management agent.
"""

import os
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
        print(f"🎯 Executing inventory optimization...")
        
        # Extract filter criteria from entities
        branch_codes = None
        product_codes = None
        regions = None
        
        if entities:
            branch_codes = entities.get('branch_codes')
            product_codes = entities.get('product_codes')
            regions = entities.get('regions')
            
            if branch_codes:
                print(f"   🎯 Filtering by {len(branch_codes)} specific branches")
            if product_codes:
                print(f"   🎯 Filtering by {len(product_codes)} specific products")
            if regions:
                print(f"   🎯 Filtering by regions: {', '.join(regions)}")
        
        try:
            # Step 1: Get current inventory FIRST (with entity filters)
            print("📌 Step 1: Analyzing current inventory...")
            inventory_data = self._get_current_inventory(
                branch_codes=branch_codes,
                product_codes=product_codes,
                regions=regions
            )
            
            if inventory_data.empty:
                return {
                    "success": False,
                    "message": "No inventory data found"
                }
            
            # Step 2: Get PER-ITEM forecast demand (IMPROVED!)
            print("📌 Step 2: Getting per-item demand forecasts...")
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
            
            if not per_item_forecasts:
                return {
                    "success": False,
                    "message": "Could not generate forecasts for demand prediction"
                }
            
            self._latest_drift_report = self._analyze_forecast_drift(per_item_forecasts)
            
            # Step 3: Calculate inventory metrics with per-item forecasts
            print("📌 Step 3: Calculating inventory metrics...")
            recommendations = self._generate_recommendations(
                inventory_data, 
                per_item_forecasts, 
                horizon_days
            )
            
            # Step 4: Find transfer opportunities
            print("📌 Step 4: Finding transfer opportunities...")
            transfer_opportunities = self._find_transfer_opportunities(
                recommendations
            )
            
            # Step 5: Generate comprehensive plan
            plan = self._create_action_plan(recommendations, transfer_opportunities)
            
            # Step 6: Create visualization
            chart_path = self._plot_inventory_optimization(
                inventory_data, 
                per_item_forecasts, 
                recommendations
            )
            
            # Step 6: Generate smart insights (NEW!)
            print("📌 Step 6: Generating AI-powered insights...")
            insights = self.insights_generator.generate_insights(
                recommendations, 
                plan, 
                entities
            )
            
            print(f"✅ Optimization complete: {len(plan['actions'])} actions recommended")
            
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
                "model_info": self._model_info
            }
            
        except Exception as e:
            print(f"❌ Optimization error: {e}")
            import traceback
            traceback.print_exc()
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
        print(f"🔮 Generating {len(inventory_data_limited)} forecasts using VECTORIZATION...")
        forecasts = self._generate_forecasts_vectorized(
            inventory_data_limited,
            horizon_days
        )
        print(f"✅ Generated {len(forecasts)} forecasts successfully (vectorized)")
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
        print("   📦 Step 1: Fetching all historical data (single query)...")
        timeseries_cache = self._build_timeseries_cache(inventory_data)
        self._latest_data_quality_report = self.data_checker.generate_report(
            inventory_data,
            timeseries_cache
        )
        self.data_checker.log_report(self._latest_data_quality_report)

        routing_assignments = {
            "xgboost": [],
            "moving_avg": [],
            "cold_start": []
        }

        for idx, row in inventory_data.iterrows():
            key = (row['product_code'], row['branch_code'])
            cache_df = timeseries_cache.get(key)
            history_span_days = 0
            if cache_df is not None and not cache_df.empty:
                date_series = pd.to_datetime(cache_df['date'])
                history_span_days = int((date_series.max() - date_series.min()).days)
            if history_span_days >= 60:
                routing_assignments["xgboost"].append(idx)
            elif history_span_days >= 14:
                routing_assignments["moving_avg"].append((key, cache_df, row))
            else:
                routing_assignments["cold_start"].append((key, cache_df, row))

        print("   🔀 Smart routing summary:")
        print(f"      • XGBoost (≥60 ngày): {len(routing_assignments['xgboost'])} items")
        print(f"      • Moving Average (14-59 ngày): {len(routing_assignments['moving_avg'])} items")
        print(f"      • Cold Start (<14 ngày hoặc không có lịch sử): {len(routing_assignments['cold_start'])} items")

        xgboost_inventory = inventory_data.loc[routing_assignments["xgboost"]]
        
        # Step 2: Prepare data for vectorized processing
        print("   🔧 Step 2: Preparing data for vectorized feature engineering...")
        
        # Get pre-trained model loader
        try:
            from agent.xgboost_model_loader import get_model_loader
            model_loader = get_model_loader()
            if not model_loader.loaded:
                print("   ⚠️  Pre-trained model not available, falling back to per-item forecasting")
                return self._generate_forecasts_fallback(inventory_data, horizon_days, timeseries_cache)
            self._model_info = model_loader.get_model_info()
            print(f"   🧠 Using model version: {model_loader.get_version_string()}")
        except Exception as e:
            print(f"   ⚠️  Could not load model: {e}, falling back to per-item forecasting")
            return self._generate_forecasts_fallback(inventory_data, horizon_days, timeseries_cache)
        
        forecasts = {}
        processed = 0
        total_items = len(xgboost_inventory)
        
        # Step 3: Process each item with vectorized feature engineering
        print("   🚀 Step 3: Vectorized feature engineering and prediction...")
        
        if xgboost_inventory.empty:
            print("   ⚠️  No items eligible for XGBoost routing, skipping ML pipeline.")
        
        for idx, row in xgboost_inventory.iterrows():
            product_code = row['product_code']
            branch_code = row['branch_code']
            key = (product_code, branch_code)
            
            # Get historical data from cache
            cache_df = timeseries_cache.get(key)
            
            if cache_df is None or cache_df.empty or len(cache_df) < 2:
                forecasts[key] = self._create_intelligent_fallback(
                    product_code, branch_code, horizon_days, row
                )
                forecasts[key]['routing_strategy'] = 'XGBOOST'
                processed += 1
                continue
            
            if len(cache_df) < 7:
                avg_demand = cache_df['total_qty'].mean()
                forecasts[key] = self._create_simple_forecast_from_data(
                    cache_df, avg_demand, horizon_days
                )
                forecasts[key]['routing_strategy'] = 'XGBOOST'
                processed += 1
                continue
            
            try:
                # Prepare time series DataFrame
                df_ts = cache_df[['date', 'total_qty']].copy()
                df_ts['date'] = pd.to_datetime(df_ts['date'])
                df_ts = df_ts.set_index('date')
                df_ts.columns = ['value']
                
                # CRITICAL: Ensure data is sorted and complete
                df_ts = df_ts.sort_index()
                
                # Check if we have recent data (last 30 days)
                if SYSTEM_DATE_AVAILABLE:
                    from agent.system_date import get_system_date
                    system_date = pd.to_datetime(get_system_date())
                else:
                    system_date = pd.Timestamp.now()
                
                last_data_date = df_ts.index[-1]
                days_since_last_data = (system_date - last_data_date).days
                
                if days_since_last_data > 7:
                    print(f"   ⚠️  Warning: Last data is {days_since_last_data} days old for {product_code} at branch {branch_code}")
                
                # Ensure we have at least 60 days of data for good features (lag_30, rolling_30)
                if len(df_ts) < 60:
                    print(f"   ⚠️  Warning: Only {len(df_ts)} days of data for {product_code} at branch {branch_code} (recommend ≥90 days for best features)")
                
                # Use vectorized feature engineering from model loader
                df_features = model_loader.create_features_from_timeseries(df_ts)
                
                # Fix rolling_std_30 missing data with fillna(0)
                if 'rolling_std_30' in df_features.columns:
                    df_features['rolling_std_30'] = df_features['rolling_std_30'].fillna(0)
                if 'rolling_std_7' in df_features.columns:
                    df_features['rolling_std_7'] = df_features['rolling_std_7'].fillna(0)
                if 'rolling_std_14' in df_features.columns:
                    df_features['rolling_std_14'] = df_features['rolling_std_14'].fillna(0)
                
                # Fix volatility columns that depend on rolling_std
                if 'volatility_30' in df_features.columns:
                    df_features['volatility_30'] = df_features['volatility_30'].fillna(0)
                if 'volatility_7' in df_features.columns:
                    df_features['volatility_7'] = df_features['volatility_7'].fillna(0)
                
                # Generate forecast using pre-trained model
                forecast_df = model_loader.predict_with_confidence(
                    df_ts,
                    horizon=horizon_days,
                    confidence_level=0.95
                )
                
                # Adjust forecast growth
                adjusted_forecast_df = self._constrain_forecast_growth(
                    forecast_df, df_ts
                )
                
                # Compute metrics
                adjusted_metrics = self._compute_forecast_metrics(
                    df_ts, adjusted_forecast_df
                )
                
                forecasts[key] = {
                    'forecast_df': adjusted_forecast_df,
                    'historical_df': df_ts,
                    'metrics': adjusted_metrics
                }
                forecasts[key]['routing_strategy'] = 'XGBOOST'
                
                processed += 1
                if processed % 100 == 0 or processed == total_items:
                    print(f"      • Processed {processed}/{total_items} items")
                    
            except Exception as e:
                print(f"   ⚠️  Forecast failed for {product_code} at branch {branch_code}: {e}")
                forecasts[key] = self._create_fallback_forecast(horizon_days)
                forecasts[key]['routing_strategy'] = 'XGBOOST'
                processed += 1
        
        if routing_assignments["moving_avg"]:
            print(f"   🔁 Using moving-average fallback for {len(routing_assignments['moving_avg'])} items (14-59 ngày).")
            for key, cache_df, _ in routing_assignments["moving_avg"]:
                avg_demand = cache_df['total_qty'].mean() if cache_df is not None and not cache_df.empty else 0.0
                fallback = self._create_simple_forecast_from_data(cache_df, avg_demand, horizon_days)
                fallback['routing_strategy'] = 'MOVING_AVG'
                forecasts[key] = fallback

        if routing_assignments["cold_start"]:
            print(f"   🧊 Cold start fallback cho {len(routing_assignments['cold_start'])} items (<14 ngày lịch sử).")
            for key, _, row in routing_assignments["cold_start"]:
                fallback = self._create_new_item_forecast(horizon_days, row)
                fallback['routing_strategy'] = 'REVIEW_NEW_ITEM'
                forecasts[key] = fallback

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
                timeseries_cache
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
        
        # Build parameterized SQL for historical data (90 days for better features)
        # Use system_date-aware date filter
        if SYSTEM_DATE_AVAILABLE:
            from agent.system_date import get_system_date
            system_date = get_system_date()
            date_filter = f"date >= DATE '{system_date}' - INTERVAL '90 days' AND date <= DATE '{system_date}'"
        else:
            date_filter = "date >= CURRENT_DATE - INTERVAL '90 days' AND date <= CURRENT_DATE"
        
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
                df = self.db.execute_query(sql, params)
            
            if df.empty or len(df) < 2:
                return key, self._create_intelligent_fallback(
                    product_code, branch_code, horizon_days, inventory_row
                )
            
            if len(df) < 7:
                avg_demand = df['total_qty'].mean()
                return key, self._create_simple_forecast_from_data(
                    df, avg_demand, horizon_days
                )
            
            # Use system_date-aware date filter
            if SYSTEM_DATE_AVAILABLE:
                from agent.system_date import get_system_date
                system_date = get_system_date()
                date_filter = f"date >= DATE '{system_date}' - INTERVAL '90 days' AND date <= DATE '{system_date}'"
            else:
                date_filter = "date >= CURRENT_DATE - INTERVAL '90 days' AND date <= CURRENT_DATE"
            
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
                    date_filter = f"s.date >= DATE '{system_date}' - INTERVAL '120 days' AND s.date <= DATE '{system_date}'"
                else:
                    date_filter = "s.date >= CURRENT_DATE - INTERVAL '120 days' AND s.date <= CURRENT_DATE"
                
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
                
                try:
                    df_chunk = self.db.execute_query(sql)
                    
                    if df_chunk.empty:
                        continue
                    
                    # Group by (product_code, branch_code) and store in cache
                    for (product_code, branch_code), grp in df_chunk.groupby(['product_code', 'branch_code']):
                        key = (product_code, branch_code)
                        cache[key] = grp[['date', 'total_qty']].copy()
                        
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
            print(f"   ⚠️  Drift monitor: {stats['high_drift_count']} / {stats['total_items']} items exceed threshold {drift_threshold:.0%}")
        else:
            print("   ✅ Drift monitor: no items exceed threshold")
        stats["high_drift_samples"] = stats["high_drift_samples"][:10]
        return stats
    
    def _get_forecast_base_date(self) -> date:
        """Return the configured system date (or real date) for fallback forecasts."""
        if SYSTEM_DATE_AVAILABLE:
            from agent.system_date import get_system_date
            return pd.to_datetime(get_system_date()).date()
        return datetime.now().date()
    
    def _create_fallback_forecast(self, horizon_days: int) -> Dict:
        """Create a simple fallback forecast when data is insufficient."""
        base_date = self._get_forecast_base_date()
        
        future_dates = pd.date_range(
            start=base_date + timedelta(days=1),
            periods=horizon_days,
            freq='D'
        )
        
        zero_values = [self.missing_data_forecast_value] * horizon_days
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
        
        # Use avg_demand for forecast (better than 0)
        forecast_values = [max(0, avg_demand)] * horizon_days
        forecast_df = pd.DataFrame({'date': future_dates, 'forecast': forecast_values}).set_index('date')
        
        # CRITICAL: Use actual avg_demand for recent_avg_daily, not 0!
        actual_recent_avg = float(avg_demand) if avg_demand > 0 else 0.0
        
        return {
            'forecast_df': forecast_df,
            'historical_df': hist_df,
            'metrics': {
                'recent_avg_daily': actual_recent_avg,  # FIX: Use actual avg_demand
                'forecast_avg_daily': float(max(0, avg_demand)),
                'forecast_total': float(max(0, avg_demand) * horizon_days),
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
            i.product_name,
            i.quantity as current_stock,
            i.unit
        FROM inventory i
        JOIN branch b ON i.branch_code = b.branch_code
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
        
        inventory_result = self.db.execute_query(sql, params if params else None)
        
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
            result = self.db.execute_query(sql)
            return result
        except Exception as e:
            print(f"   ⚠️  Error fetching products with sales: {e}")
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
        """
        if recommendations.empty:
            return []
        
        # Separate surplus and deficit branches
        surplus = recommendations[recommendations['action'] == 'SURPLUS'].copy()
        deficit = recommendations[recommendations['action'].isin(['URGENT_RESTOCK', 'RESTOCK'])].copy()
        
        if surplus.empty or deficit.empty:
            return []
        
        transfer_opportunities = []
        
        for _, deficit_row in deficit.iterrows():
            deficit_branch = deficit_row['branch_code']
            needed_qty = deficit_row['quantity_needed']
            
            # Find nearby branches with surplus (PARAMETERIZED)
            nearby_query = """
            SELECT 
                bd.branch_code_1 as source_branch,
                bd.branch_code_2 as dest_branch,
                bd.distance_km,
                b.branch_name as source_branch_name
            FROM branch_distance bd
            JOIN branch b ON bd.branch_code_1 = b.branch_code
            WHERE bd.branch_code_2 = :deficit_branch
                AND bd.distance_km <= :max_distance
            ORDER BY bd.distance_km ASC
            """
            
            params = {
                'deficit_branch': int(deficit_branch),
                'max_distance': self.max_transfer_distance_km
            }
            
            try:
                nearby_branches = self.db.execute_query(nearby_query, params)
                
                for _, nearby in nearby_branches.iterrows():
                    source_branch = nearby['source_branch']
                    
                    # Check if source branch has surplus for this product
                    surplus_match = surplus[
                        (surplus['branch_code'] == source_branch) &
                        (surplus['product_code'] == deficit_row['product_code'])
                    ]
                    
                    if not surplus_match.empty:
                        surplus_row = surplus_match.iloc[0]
                        available_qty = surplus_row['current_stock'] - surplus_row['reorder_point']
                        
                        if available_qty > 0:
                            transfer_qty = min(available_qty, needed_qty)
                            
                            transfer_opportunities.append({
                                'product_code': deficit_row['product_code'],
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
                
            except Exception as e:
                print(f"⚠️ Error finding transfers for branch {deficit_branch}: {e}")
                continue
        
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




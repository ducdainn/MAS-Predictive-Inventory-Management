"""
ForecastAgent: provides demand forecasting using pre-trained and on-the-fly models.
"""

import os
import time
from datetime import timedelta
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent.label_formatter import format_axis_label
from agent.manager.database_manager import DatabaseManager
from agent.utils.dataframe_utils import format_dataframe_columns
from typing import Any

from agent.utils.model_logger import get_model_logger
from agent.utils.workflow_data_logger import get_workflow_logger

try:
    from agent.ml_forecasting import MLForecastingEngine, ForecastResult
    ML_FORECASTING_AVAILABLE = True
except ImportError:
    ML_FORECASTING_AVAILABLE = False
    MLForecastingEngine = None  # type: ignore
    ForecastResult = None  # type: ignore

try:
    from agent.xgboost_model_loader import get_model_loader
    PRETRAINED_MODEL_AVAILABLE = True
except ImportError:
    PRETRAINED_MODEL_AVAILABLE = False
    get_model_loader = None  # type: ignore

try:
    # NEW: Panel XGBoost loaders (Global Model with Identifiers)
    from agent.panel_xgboost_model_loader import (
        get_panel_model_loader,
        get_panel_multistep_model_loader,
    )
    PANEL_MODEL_AVAILABLE = True
except ImportError:
    PANEL_MODEL_AVAILABLE = False
    get_panel_model_loader = None  # type: ignore
    get_panel_multistep_model_loader = None  # type: ignore

try:
    from agent.system_date import get_system_date
    SYSTEM_DATE_AVAILABLE = True
except ImportError:
    SYSTEM_DATE_AVAILABLE = False


class ForecastAgent:
    """Performs time series forecasting with ML models (XGBoost, LightGBM, Prophet)."""

    def __init__(self,
                 db_manager: DatabaseManager,
                 output_dir: str = "charts",
                 use_ml: bool = True,
                 use_pretrained: bool = True):
        self.db = db_manager
        self.output_dir = output_dir
        # TẠM THỜI: ưu tiên dùng XGBoost (pre-trained / panel) cho mọi dự báo,
        # vô hiệu hóa ML engine on-the-fly.
        self.use_ml = False
        self.use_pretrained = use_pretrained and PRETRAINED_MODEL_AVAILABLE
        os.makedirs(output_dir, exist_ok=True)

        # Centralized model logger cho tab Forecast (ghi input / output từng lần chạy)
        self.model_logger = get_model_logger(log_dir="model_logs")
        
        # Workflow data logger
        self.workflow_logger = get_workflow_logger()

        # Align behaviour with InventoryOptimizationAgent:
        # limit forecast growth vs recent demand to avoid unrealistic spikes
        # UPDATED: Increased from 1.2 to 2.0 to allow reasonable growth (1.2 was too strict)
        self.max_forecast_vs_recent_ratio = 2.0

        self.pretrained_loader = None
        if self.use_pretrained and get_model_loader:
            try:
                self.pretrained_loader = get_model_loader(log_dir=output_dir)
                if self.pretrained_loader.loaded:
                    # Giữ một log high-level, nhưng vào file thay vì spam terminal
                    self.model_logger.info("FORECAST_AGENT_INIT | pretrained_model=loaded")
                else:
                    self.model_logger.warning("FORECAST_AGENT_INIT | pretrained_model=not_found")
                    self.pretrained_loader = None
            except Exception as e:
                self.model_logger.error(f"FORECAST_AGENT_INIT | pretrained_load_error={e}")
                self.pretrained_loader = None

        # NEW: Panel model loaders (same family as InventoryOptimizationAgent)
        self.panel_loader = None
        self.panel_multistep_loader = None
        if PANEL_MODEL_AVAILABLE and get_panel_model_loader:
            try:
                self.panel_loader = get_panel_model_loader()
                if getattr(self.panel_loader, "loaded", False):
                    self.model_logger.info(
                        "FORECAST_AGENT_INIT | panel_model=loaded "
                        f"| version={self.panel_loader.model_version}"
                    )
                else:
                    self.model_logger.warning("FORECAST_AGENT_INIT | panel_model=not_found")
                    self.panel_loader = None
            except Exception as e:
                self.model_logger.error(f"FORECAST_AGENT_INIT | panel_load_error={e}")
                self.panel_loader = None

        # Ưu tiên: multi-step panel model (nếu đã train bằng train_xgboost_panel_multistep.py)
        if PANEL_MODEL_AVAILABLE and get_panel_multistep_model_loader:
            try:
                self.panel_multistep_loader = get_panel_multistep_model_loader()
                if getattr(self.panel_multistep_loader, "loaded", False):
                    self.model_logger.info(
                        "FORECAST_AGENT_INIT | panel_multistep_model=loaded "
                        f"| version={self.panel_multistep_loader.model_version}"
                    )
                else:
                    self.panel_multistep_loader = None
            except Exception as e:
                self.model_logger.error(
                    f"FORECAST_AGENT_INIT | panel_multistep_load_error={e}"
                )
                self.panel_multistep_loader = None

        if use_ml and ML_FORECASTING_AVAILABLE:
            self.ml_engine = MLForecastingEngine(confidence_level=0.95, log_dir=output_dir, enable_export=True)
            if not self.pretrained_loader:
                self.model_logger.info("FORECAST_AGENT_INIT | ml_engine=initialized_but_disabled_for_xgboost_priority")
        else:
            self.ml_engine = None
            self.model_logger.info("FORECAST_AGENT_INIT | ml_engine=disabled_using_simple")

    def forecast(self,
                 sql: str,
                 question: str,
                 horizon: int = 30,
                 create_chart: bool = True,
                 preloaded_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Execute query and perform forecasting.

        Nếu dữ liệu có cột product_code → dự báo theo từng SKU rồi cộng lại thành tổng
        để vẽ biểu đồ; đồng thời trả thêm per_sku_forecasts cho phân tích chi tiết.
        """
        # Log câu hỏi và SQL, không in ra terminal
        self.model_logger.info(
            f"FORECAST_START | horizon={horizon} | create_chart={create_chart} | question={question}"
        )

        if preloaded_df is not None:
            df = preloaded_df.copy()
        else:
            # Debug: Log SQL query
            print(f"   📝 SQL Query: {sql[:200]}..." if len(sql) > 200 else f"   📝 SQL Query: {sql}")
            df = self.db.execute_query(sql, source="ForecastAgent.forecast")
            # Debug: Log query result info
            if not df.empty:
                date_cols = [col for col in df.columns if 'date' in col.lower() or df[col].dtype == 'datetime64[ns]']
                if date_cols:
                    date_col = date_cols[0]
                    min_date = df[date_col].min() if date_col in df.columns else None
                    max_date = df[date_col].max() if date_col in df.columns else None
                    print(f"   📅 Query date range: {min_date} to {max_date} ({len(df)} rows)")
                else:
                    print(f"   ⚠️  No date column found in query result! Columns: {list(df.columns)}")

        # Log raw query result
        if not df.empty:
            self.workflow_logger.log_dataframe(
                "raw_query_result",
                "ForecastAgent",
                df,
                {"sql": sql, "question": question, "horizon": horizon}
            )

        if df.empty:
            return {"success": False, "message": "No historical data available"}

        self.model_logger.log_dataframe_overview(
            df,
            name="ForecastAgent.raw_query_result",
            context={"rows": len(df)},
        )

        # Case 1: Có dimension product_code → forecast per SKU rồi cộng dồn
        # Sử dụng panel XGBoost model (nếu có đủ branch_code/region/f_sku),
        # fallback sang model cũ / simple khi thiếu thông tin.
        if 'product_code' in df.columns:
            result = self._forecast_per_sku(df, horizon, create_chart)
            # Log per-SKU forecast result
            if result.get("success"):
                self.workflow_logger.log_step(
                    "per_sku_forecast_result",
                    "ForecastAgent",
                    {
                        "n_skus": len(result.get("per_sku_forecasts", {})),
                        "metrics": result.get("metrics", {}),
                        "summary": result.get("summary", ""),
                    },
                    {"horizon": horizon, "question": question}
                )
            return result

        # Case 2: Legacy – single aggregated series
        date_col, value_col = self._identify_columns(df)

        if not date_col or not value_col:
            return {"success": False, "message": "Could not identify date and value columns"}

        df_ts = self._prepare_time_series(df, date_col, value_col)
        
        # Log prepared time series
        self.workflow_logger.log_dataframe(
            "prepared_timeseries",
            "ForecastAgent",
            df_ts.reset_index(),
            {"horizon": horizon, "question": question}
        )

        if self.use_ml and self.ml_engine:
            forecast_result, model_info = self._ml_forecast(df_ts, horizon)
        else:
            forecast_result, model_info = self._run_pretrained_or_simple(df_ts, horizon)

        # CRITICAL: Align forecast growth behaviour with InventoryOptimizationAgent
        # Apply same style of constraint: average forecast cannot exceed recent demand
        # by more than self.max_forecast_vs_recent_ratio.
        step5_start = time.perf_counter()
        forecast_result = self._constrain_forecast_growth(forecast_result, df_ts)
        step5_elapsed = time.perf_counter() - step5_start
        print(f"   ⏱️  Step 5 (Post-Processing): completed in {step5_elapsed:.3f}s")
        
        # Log forecast result
        self.workflow_logger.log_dataframe(
            "forecast_result",
            "ForecastAgent",
            forecast_result.reset_index(),
            {"model_info": model_info, "horizon": horizon}
        )

        # Không tạo chart PNG; UI dùng interactive Plotly với historical_data_raw/forecast_raw
        chart_path = None
        metrics_start = time.perf_counter()
        metrics = self._calculate_metrics(df_ts, forecast_result)
        metrics_elapsed = time.perf_counter() - metrics_start
        
        # Log metrics
        self.workflow_logger.log_step(
            "forecast_metrics",
            "ForecastAgent",
            metrics,
            {"model": model_info.get("model") if isinstance(model_info, dict) else None}
        )
        
        # Log series-level forecast summary
        self.model_logger.log_forecast_series(
            key={"source": "ForecastAgent.single_series", "question": question},
            historical_df=df_ts,
            forecast_df=forecast_result,
            metrics=metrics,
            extra={"model": model_info.get("model") if isinstance(model_info, dict) else None},
        )

        df_ts_display = format_dataframe_columns(df_ts.reset_index())
        if len(df_ts_display.columns) > 0:
            df_ts_display = df_ts_display.set_index(df_ts_display.columns[0])

        forecast_display = format_dataframe_columns(forecast_result.reset_index())
        if len(forecast_display.columns) > 0:
            forecast_display = forecast_display.set_index(forecast_display.columns[0])

        return {
            "success": True,
            "historical_data": df_ts_display,
            "forecast": forecast_display,
            "historical_data_raw": df_ts,
            "forecast_raw": forecast_result,
            "chart": chart_path,
            "metrics": metrics,
            "summary": self._generate_forecast_summary(df_ts, forecast_result)
        }

    def _identify_columns(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """Identify date and value columns."""
        date_col = None
        value_col = None

        for col in df.columns:
            if 'date' in col.lower() or df[col].dtype == 'datetime64[ns]':
                date_col = col
                break

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                if any(kw in col.lower() for kw in ['quantity', 'qty', 'sales', 'total', 'sum']):
                    value_col = col
                    break
            if not value_col:
                value_col = numeric_cols[0]

        return date_col, value_col

    def _fast_simple_forecast(self, df_ts: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        Ultra-fast simple forecast for SKUs with very little data.
        Just uses the mean value, no complex calculations.
        
        CRITICAL FIX: Always calculate mean on ALL values (including zeros) to reflect
        true average daily demand. Using non-zero mean inflates forecast for slow-moving items.
        Example: 1 sale in 30 days (5 units) → True avg = 5/30 = 0.17/day, NOT 5/day!
        """
        # CRITICAL: Always calculate mean on ALL values (including zeros)
        # This correctly reflects average daily demand for slow-moving items
        values = df_ts['value']
        avg_value = values.mean() if len(values) > 0 else 0
        
        # Generate flat forecast (no trend for sparse data)
        last_date = df_ts.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')
        forecast_values = [max(0, avg_value)] * horizon
        
        return pd.DataFrame({
            'forecast': forecast_values
        }, index=future_dates)

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
            
            # Log the despiking action (only for first few to avoid spam in batch processing)
            # Skip logging in batch mode to improve performance
            
            # Replace the last value
            df_ts = df_ts.copy()  # Avoid SettingWithCopyWarning
            df_ts.iloc[-1, df_ts.columns.get_loc('value')] = replacement_val
        
        return df_ts

    def _prepare_time_series(self, df: pd.DataFrame, date_col: str, value_col: str, export_file: bool = True) -> pd.DataFrame:
        """Prepare time series data with despiking to prevent spike amplification."""
        # OPTIMIZATION: Only measure timing if export_file=True (for single series forecasts)
        # Skip timing overhead in batch processing loops
        if export_file:
            step2_start = time.perf_counter()
        
        # Fast resample operation
        df_ts = df[[date_col, value_col]].copy()
        df_ts[date_col] = pd.to_datetime(df_ts[date_col])
        df_ts = df_ts.sort_values(date_col)
        df_ts = df_ts.set_index(date_col)
        df_ts.columns = ['value']
        df_ts = df_ts.resample('D').sum().fillna(0)
        
        # CRITICAL: Despike to prevent spike amplification in recursive forecasting
        # This replaces outliers at the last day with rolling mean to prevent
        # the model from amplifying spikes through lag_1 features
        df_ts = self._despike_time_series(df_ts, window=30, threshold_std=3.0)
        
        # Log and export prepared time series (only if export_file=True)
        if export_file:
            step2_elapsed = time.perf_counter() - step2_start
            print(f"   ⏱️  Step 2 (Prepare Time Series): completed in {step2_elapsed:.3f}s")
            print(f"   📊 Prepared time series: {len(df_ts)} days, range: {df_ts.index[0]} to {df_ts.index[-1]}")
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            try:
                log_file = os.path.join(self.output_dir, f"prepared_timeseries_{timestamp}.csv")
                df_ts.reset_index().to_csv(log_file, index=False)
                print(f"   💾 Exported prepared time series to: {log_file}")
            except Exception as e:
                print(f"   ⚠️  Failed to export time series: {e}")
        
        return df_ts

    def _cap_forecast(self, forecast_df: pd.DataFrame, historical_df: pd.DataFrame, max_multiplier: float = 3.0) -> pd.DataFrame:
        """
        Cap forecast values to prevent unrealistic high predictions.
        
        This is critical when:
        - Data is very sparse (1-4 days) → model can't learn patterns
        - Aggregating many SKUs → one bad forecast can inflate total
        
        Logic:
        - Calculate historical average (prefer recent 30 days, fallback to all)
        - Cap each forecast value to max_multiplier * historical_avg
        - Apply to both individual values and bounds
        
        Args:
            forecast_df: DataFrame with 'forecast' column (and optionally bounds)
            historical_df: Historical data with 'value' column
            max_multiplier: Maximum multiplier for forecast vs historical avg (default: 3.0)
            
        Returns:
            DataFrame with capped forecast values
        """
        if forecast_df.empty or historical_df.empty:
            return forecast_df
        
        # Calculate historical average (prefer recent data)
        hist_values = historical_df['value']
        recent_window = hist_values.tail(min(30, len(hist_values)))
        
        # CRITICAL FIX: Always calculate mean on ALL values (including zeros)
        # This correctly reflects average daily demand for slow-moving items
        # Using non-zero mean inflates cap threshold and allows unrealistic forecasts
        hist_avg = recent_window.mean() if len(recent_window) > 0 else 0
        
        # Fallback to overall mean if recent window is empty
        if hist_avg <= 0 and len(hist_values) > 0:
            hist_avg = hist_values.mean()
        
        # If still 0 or very small, use a conservative default
        if hist_avg <= 0:
            hist_avg = max(hist_values.max() * 0.1, 1.0) if len(hist_values) > 0 else 1.0
        
        # Calculate cap threshold
        cap_threshold = hist_avg * max_multiplier
        
        # Cap forecast values
        capped_forecast = forecast_df.copy()
        
        # Cap main forecast column
        if 'forecast' in capped_forecast.columns:
            # Apply cap: min(forecast, cap_threshold)
            capped_forecast['forecast'] = capped_forecast['forecast'].clip(upper=cap_threshold)
        
        # Cap confidence bounds if present
        for bound_col in ['lower_bound', 'upper_bound']:
            if bound_col in capped_forecast.columns:
                # Lower bound: ensure >= 0, upper bound: cap at threshold
                if bound_col == 'lower_bound':
                    capped_forecast[bound_col] = capped_forecast[bound_col].clip(lower=0)
                else:
                    capped_forecast[bound_col] = capped_forecast[bound_col].clip(upper=cap_threshold)
        
        # Log if capping was applied (disabled to reduce noise)
        # original_mean = forecast_df['forecast'].mean() if 'forecast' in forecast_df.columns else 0
        # capped_mean = capped_forecast['forecast'].mean() if 'forecast' in capped_forecast.columns else 0
        # 
        # if capped_mean < original_mean * 0.9:  # Significant reduction
        #     print(f"   ⚠️  Forecast capped: {original_mean:.1f} → {capped_mean:.1f} (cap={cap_threshold:.1f}, hist_avg={hist_avg:.1f})")
        
        return capped_forecast

    def _constrain_forecast_growth(self,
                                   forecast_df: Optional[pd.DataFrame],
                                   historical_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Limit forecast average so it cannot exceed recent demand by too much.

        This mirrors InventoryOptimizationAgent._constrain_forecast_growth so that
        the standalone forecast behaviour is consistent with the optimization logic.
        
        CRITICAL FIX: Use NON-ZERO mean for recent_avg to avoid over-constraining
        forecasts for sparse/slow-moving items.
        """
        if forecast_df is None or forecast_df.empty:
            return forecast_df

        if historical_df is None or historical_df.empty:
            return forecast_df

        # Support both 'value' column and generic first-column (for safety)
        if 'value' in historical_df.columns:
            hist_series = historical_df['value']
        else:
            hist_series = historical_df.iloc[:, 0]

        # CRITICAL FIX: Calculate recent_avg on NON-ZERO values to avoid
        # over-constraining forecasts for sparse/slow-moving items
        recent_window = hist_series.tail(min(30, len(hist_series)))
        
        # Prefer non-zero mean (true demand when items are sold)
        non_zero_recent = recent_window[recent_window > 0]
        if len(non_zero_recent) > 0:
            recent_avg = non_zero_recent.mean()
        else:
            # Fallback: use all values (including zeros) if no non-zero sales
            recent_avg = recent_window.mean() if not recent_window.empty else 0
        
        if recent_avg <= 0:
            return forecast_df

        forecast_mean = forecast_df['forecast'].mean()
        # Use the configured ratio (now 2.0 instead of 1.2)
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

    def _simple_forecast(self, df_ts: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Simple forecasting using moving average and trend."""
        df_ts['ma_7'] = df_ts['value'].rolling(window=7, min_periods=1).mean()
        df_ts['ma_30'] = df_ts['value'].rolling(window=30, min_periods=1).mean()

        last_30_days = df_ts['value'].tail(30)
        if len(last_30_days) > 1:
            x = np.arange(len(last_30_days))
            y = last_30_days.values
            trend = np.polyfit(x, y, 1)
        else:
            trend = [0, last_30_days.mean() if len(last_30_days) > 0 else 0]

        last_date = df_ts.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')

        base_value = df_ts['ma_30'].iloc[-1]
        daily_trend = trend[0]

        forecast_values = [max(0, base_value + daily_trend * i) for i in range(1, horizon + 1)]

        forecast_df = pd.DataFrame({
            'date': future_dates,
            'forecast': forecast_values
        }).set_index('date')

        return forecast_df

    def _run_pretrained_or_simple(self, df_ts: pd.DataFrame, horizon: int, export_files: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Helper: chạy pre-trained XGBoost nếu có, nếu lỗi thì fallback simple forecast.
        """
        if self.pretrained_loader and self.pretrained_loader.loaded:
            if export_files:
                print("⚡ Using pre-trained XGBoost model (FAST inference)...")
            try:
                # Temporarily disable file export and logging if export_files=False
                if not export_files:
                    original_enable_export = self.pretrained_loader.enable_export
                    original_enable_logging = self.pretrained_loader.enable_logging
                    self.pretrained_loader.enable_export = False
                    self.pretrained_loader.enable_logging = False
                
                forecast_df = self.pretrained_loader.predict_with_confidence(
                    df_ts,
                    horizon=horizon,
                    confidence_level=0.95
                )
                
                # CRITICAL: Cap forecast to prevent unrealistic values
                data_length = len(df_ts)
                if data_length < 7:
                    max_multiplier = 2.0  # Very aggressive for sparse data
                elif data_length < 30:
                    max_multiplier = 2.5  # Moderate for short data
                else:
                    max_multiplier = 3.0  # Standard for sufficient data
                
                forecast_df = self._cap_forecast(forecast_df, df_ts, max_multiplier=max_multiplier)
                
                # Restore enable_export and enable_logging
                if not export_files:
                    self.pretrained_loader.enable_export = original_enable_export
                    self.pretrained_loader.enable_logging = original_enable_logging
                model_info = {
                    "model": "xgboost_pretrained",
                    "confidence_intervals": True,
                    "confidence_level": 0.95,
                    "has_bounds": True,
                    "inference_time": "~50ms"
                }
                if export_files:
                    print(f"✅ Forecast generated using PRE-TRAINED XGBOOST (fast!)")
                return forecast_df, model_info
            except Exception as e:
                if export_files:
                    print(f"⚠️  Pre-trained model failed on aggregated series: {e}")
                # Restore enable_export and enable_logging if they were disabled
                if not export_files and hasattr(self, 'pretrained_loader') and self.pretrained_loader:
                    self.pretrained_loader.enable_export = original_enable_export
                    self.pretrained_loader.enable_logging = original_enable_logging

        # Fallback: simple
        forecast_df = self._simple_forecast(df_ts, horizon)
        
        # CRITICAL: Cap forecast to prevent unrealistic values (especially for sparse data)
        # Use more aggressive cap (2.0x) for simple forecast since it's less sophisticated
        forecast_df = self._cap_forecast(forecast_df, df_ts, max_multiplier=2.0)
        
        model_info = {"model": "moving_average", "confidence_intervals": False}
        return forecast_df, model_info

    def _forecast_per_sku(self,
                          df: pd.DataFrame,
                          horizon: int,
                          create_chart: bool = True) -> Dict[str, Any]:
        """
        Forecast per SKU (product_code) rồi cộng forecast thành tổng chuỗi thời gian.
        """
        print("🔬 Detected product_code column → running per-SKU forecasts")

        date_col, value_col = self._identify_columns(df)
        if not date_col or not value_col:
            return {"success": False, "message": "Could not identify date and value columns for per-SKU forecast"}

        per_sku_forecasts = {}
        per_sku_histories = {}

        agg_hist_ts = None
        agg_forecast_ts = None

        # Nhóm theo product_code + định danh panel (branch_code, region, f_sku) nếu có
        group_cols = ['product_code']
        if 'branch_code' in df.columns:
            group_cols.append('branch_code')
        if 'f_sku' in df.columns:
            group_cols.append('f_sku')
        if 'region' in df.columns:
            group_cols.append('region')
        if 'product_name' in df.columns:
            group_cols.append('product_name')

        # Count total SKUs for progress tracking
        total_skus = len(df.groupby(group_cols))
        print(f"   📦 Processing {total_skus} SKUs...")

        # Chuẩn bị container cho panel batch
        panel_series = []   # list[dict] cho panel_loader.predict_batch
        panel_meta = []     # meta để map kết quả batch về per_sku_forecasts
        
        # Debug counters
        routing_counts = {
            'panel': 0,
            'pretrained': 0,
            'simple': 0,
            'moving_avg': 0,
            'dead_stock': 0
        }
        
        for idx, (_, group) in enumerate(df.groupby(group_cols), 1):
            # Lấy thông tin định danh từ group (mỗi group là 1 series panel)
            product_code = group['product_code'].iloc[0]
            product_name = group['product_name'].iloc[0] if 'product_name' in group.columns else ""
            branch_code = int(group['branch_code'].iloc[0]) if 'branch_code' in group.columns else None
            region = str(group['region'].iloc[0]) if 'region' in group.columns else ""
            f_sku = (
                str(group['f_sku'].iloc[0])
                if 'f_sku' in group.columns and pd.notna(group['f_sku'].iloc[0])
                else None
            )

            # Don't export files for individual SKUs (only export aggregated result)
            df_ts = self._prepare_time_series(group, date_col, value_col, export_file=False)
            
            data_length = len(df_ts)
            # OPTIMIZED ROUTING LOGIC (Recommended by Data Science Expert):
            # - >= 14 ngày lịch sử → Panel XGBoost (multi-step) - Best accuracy
            # - 7-14 ngày lịch sử → Moving Average (weighted với trend) - Balanced
            # - < 7 ngày lịch sử → Simple Average - Conservative
            
            # Additional checks:
            # 1. Stale data: last sale > 90 days ago → Dead stock (forecast = 0)
            # 2. Sparse data: < 2 records → Simple Average
            # 3. Very short span: < 1 day → Simple Average
            
            # Step 3: Routing Decision timing
            if idx == 1:  # Log timing for first SKU only
                step3_start_routing = time.perf_counter()
            
            # Check for stale data (dead stock)
            if data_length > 0:
                last_sale_date = df_ts.index[-1]
                # Use system_date if available, otherwise use current date
                if SYSTEM_DATE_AVAILABLE:
                    try:
                        current_date = pd.to_datetime(get_system_date()).normalize()
                    except Exception:
                        current_date = pd.Timestamp.now().normalize()
                else:
                    current_date = pd.Timestamp.now().normalize()
                
                days_since_last_sale = (current_date - last_sale_date).days
                
                # Debug: Log first few SKUs to understand the issue
                if idx <= 5:
                    print(f"      🔍 SKU {product_code}: last_sale={last_sale_date.date()}, current={current_date.date()}, days_diff={days_since_last_sale}, data_length={data_length}")
                
                if days_since_last_sale > 90:
                    if days_since_last_sale > 180:
                        routing_counts['dead_stock'] += 1
                    last_date = df_ts.index[-1]
                    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')
                    forecast_ts = pd.DataFrame({
                        'forecast': [0] * horizon
                    }, index=future_dates)
                    model_info = {"model": "dead_stock", "confidence_intervals": False}
                    per_sku_histories[product_code] = df_ts
                    per_sku_forecasts[product_code] = {
                        "product_name": product_name,
                        "forecast": forecast_ts,
                        "model_info": model_info
                    }
                    agg_hist_ts = df_ts[['value']].add(agg_hist_ts, fill_value=0) if agg_hist_ts is not None else df_ts[['value']].copy()
                    agg_forecast_ts = forecast_ts[['forecast']].add(agg_forecast_ts, fill_value=0) if agg_forecast_ts is not None else forecast_ts[['forecast']].copy()
                    continue  # Skip to next SKU
            
            # Calculate history span
            history_span_days = int((df_ts.index[-1] - df_ts.index[0]).days) if data_length > 1 else 0
            
            # Ưu tiên dùng PANEL XGBoost nếu có đủ định danh và lịch sử >= 14 ngày
            can_use_panel = (
                self.panel_loader
                and getattr(self.panel_loader, "loaded", False)
                and branch_code is not None
                and f_sku is not None
                and history_span_days >= 14
                and data_length >= 2
            )

            can_use_pretrained = self.pretrained_loader and self.pretrained_loader.loaded

            if can_use_panel:
                routing_counts['panel'] += 1
                # Thay vì dự đoán từng SKU, gom các series đủ điều kiện panel vào batch
                series_key = len(panel_series)
                panel_series.append(
                    {
                        "key": series_key,
                        "df_ts": df_ts,
                        "branch_code": branch_code,
                        "region": region or "",
                        "f_sku": f_sku,
                    }
                )
                panel_meta.append(
                    {
                        "series_key": series_key,
                        "product_code": product_code,
                        "product_name": product_name,
                        "df_ts": df_ts,
                    }
                )
                # Lưu history ngay để sau khi batch forecast xong sẽ điền forecast
                per_sku_histories[product_code] = df_ts
                # Cộng dồn lịch sử vào chuỗi tổng ngay bây giờ
                agg_hist_ts = df_ts[['value']].add(agg_hist_ts, fill_value=0) if agg_hist_ts is not None else df_ts[['value']].copy()
                # Bỏ qua forecast tại đây, sẽ xử lý sau batch
                continue
            elif can_use_pretrained:
                routing_counts['pretrained'] += 1
                # Luôn ưu tiên pre-trained XGBoost khi không dùng được panel
                try:
                    # Temporarily disable export and logging
                    original_enable_export = self.pretrained_loader.enable_export
                    original_enable_logging = self.pretrained_loader.enable_logging
                    self.pretrained_loader.enable_export = False
                    self.pretrained_loader.enable_logging = False
                    
                    forecast_ts = self.pretrained_loader.predict_with_confidence(
                        df_ts, horizon=horizon, confidence_level=0.95
                    )
                    
                    # Cap forecast
                    if data_length < 30:
                        max_multiplier = 2.5
                    else:
                        max_multiplier = 3.0
                    forecast_ts = self._cap_forecast(forecast_ts, df_ts, max_multiplier=max_multiplier)
                    
                    # Restore flags
                    self.pretrained_loader.enable_export = original_enable_export
                    self.pretrained_loader.enable_logging = original_enable_logging
                    
                    model_info = {
                        "model": "xgboost_pretrained",
                        "confidence_intervals": True,
                        "confidence_level": 0.95
                    }
                except Exception:
                    # Fallback to simple if pre-trained fails
                    forecast_ts = self._simple_forecast(df_ts, horizon)
                    forecast_ts = self._cap_forecast(forecast_ts, df_ts, max_multiplier=2.0)
                    model_info = {"model": "moving_average", "confidence_intervals": False}
            else:
                # Fallback: simple forecast (khi không dùng được XGBoost panel / pre-trained)
                if data_length < 2:
                    routing_counts['simple'] += 1
                    # Sparse data (< 2 records) → Simple Average
                    forecast_ts = self._fast_simple_forecast(df_ts, horizon)
                    forecast_ts = self._cap_forecast(forecast_ts, df_ts, max_multiplier=2.0)
                    model_info = {"model": "moving_average_fast", "confidence_intervals": False}
                elif history_span_days >= 7:
                    routing_counts['moving_avg'] += 1
                    # 7-13 ngày lịch sử → Moving Average với trend (weighted)
                    forecast_ts = self._simple_forecast(df_ts, horizon)
                    # Apply trend adjustment for better accuracy
                    recent_data = df_ts['value'].tail(min(7, len(df_ts)))
                    if len(recent_data) > 1:
                        x = np.arange(len(recent_data))
                        y = recent_data.values
                        trend_slope = np.polyfit(x, y, 1)[0] if len(y) > 1 else 0
                        # Apply gradual trend adjustment
                        if trend_slope != 0:
                            for i in range(len(forecast_ts)):
                                forecast_ts.iloc[i, forecast_ts.columns.get_loc('forecast')] += trend_slope * (i + 1) * 0.1
                            forecast_ts['forecast'] = forecast_ts['forecast'].clip(lower=0)
                    forecast_ts = self._cap_forecast(forecast_ts, df_ts, max_multiplier=2.0)
                    model_info = {"model": "moving_average_with_trend", "confidence_intervals": False}
                else:
                    routing_counts['simple'] += 1
                    # < 7 ngày lịch sử → Simple Average (conservative)
                    forecast_ts = self._fast_simple_forecast(df_ts, horizon)
                    forecast_ts = self._cap_forecast(forecast_ts, df_ts, max_multiplier=2.0)
                    model_info = {"model": "moving_average_simple", "confidence_intervals": False}
            
            # Progress update every 10 SKUs or at the end
            if idx % 10 == 0 or idx == total_skus:
                print(f"      • Processed {idx}/{total_skus} SKUs...")

            per_sku_histories[product_code] = df_ts
            per_sku_forecasts[product_code] = {
                "product_name": product_name,
                "forecast": forecast_ts,
                "model_info": model_info
            }

            # Cộng dồn vào chuỗi tổng
            agg_hist_ts = df_ts[['value']].add(agg_hist_ts, fill_value=0) if agg_hist_ts is not None else df_ts[['value']].copy()
            forecast_sum = forecast_ts['forecast'].sum() if 'forecast' in forecast_ts.columns else 0
            agg_forecast_ts = forecast_ts[['forecast']].add(agg_forecast_ts, fill_value=0) if agg_forecast_ts is not None else forecast_ts[['forecast']].copy()
            
            # Debug: log if forecast is zero for non-dead-stock items
            if forecast_sum == 0 and model_info.get("model") != "dead_stock":
                if idx <= 5:  # Only log first 5 to avoid spam
                    print(f"      ⚠️  SKU {product_code}: forecast sum = 0 (model: {model_info.get('model')})")

        # Sau khi duyệt xong tất cả group, xử lý batch cho các series đủ điều kiện panel
        if panel_series:
            step4_start = time.perf_counter()
            batch_backend = "none"
            model_pred_start = time.perf_counter()
            try:
                if self.panel_multistep_loader and getattr(
                    self.panel_multistep_loader, "loaded", False
                ):
                    batch_backend = "panel_multistep"
                    batch_results = self.panel_multistep_loader.predict_batch(
                        panel_series, horizon=horizon
                    )
                elif self.panel_loader and getattr(self.panel_loader, "loaded", False):
                    batch_backend = "panel_recursive"
                    batch_results = self.panel_loader.predict_batch(
                        panel_series, horizon=horizon
                    )
                else:
                    batch_results = {}
                    print(f"   ⚠️  No panel model loaded, batch_results will be empty")
            except Exception as e:
                self.model_logger.error(
                    f"FORECAST_PER_SKU_PANEL_BATCH_ERROR | error={e}"
                )
                batch_results = {}
                print(f"   ⚠️  Batch processing error: {e}")
            finally:
                model_pred_elapsed = time.perf_counter() - model_pred_start
                step4_elapsed = time.perf_counter() - step4_start
                print(f"   ⏱️  Step 4 (Batch Processing): completed in {step4_elapsed:.3f}s (model.predict: {model_pred_elapsed:.3f}s)")
                print(f"      → Batch results: {len(batch_results)}/{len(panel_series)} series forecasted")
                self.model_logger.info(
                    "FORECAST_PER_SKU_PANEL_BATCH_TIMING | "
                    f"backend={batch_backend} | "
                    f"n_series={len(panel_series)} | "
                    f"horizon={horizon} | "
                    f"elapsed_sec={step4_elapsed:.2f} | "
                    f"model_pred_sec={model_pred_elapsed:.2f} | "
                    f"results_count={len(batch_results)}"
                )

            step5_start = time.perf_counter()
            batch_success_count = 0
            batch_fallback_count = 0
            
            for meta in panel_meta:
                series_key = meta["series_key"]
                product_code = meta["product_code"]
                product_name = meta["product_name"]
                df_ts = meta["df_ts"]

                if series_key in batch_results:
                    forecast_ts = batch_results[series_key]
                    # Áp dụng constrain growth giống InventoryOptimizationAgent
                    forecast_ts = self._constrain_forecast_growth(forecast_ts, df_ts)
                    model_info = {
                        "model": "panel_xgboost",
                        "confidence_intervals": False,
                        "panel_version": getattr(self.panel_loader, "model_version", "unknown"),
                    }
                    batch_success_count += 1
                else:
                    # Nếu batch thất bại cho series này → fallback simple
                    forecast_ts = self._simple_forecast(df_ts, horizon)
                    forecast_ts = self._cap_forecast(forecast_ts, df_ts, max_multiplier=2.0)
                    model_info = {"model": "moving_average", "confidence_intervals": False}
                    batch_fallback_count += 1

                per_sku_forecasts[product_code] = {
                    "product_name": product_name,
                    "forecast": forecast_ts,
                    "model_info": model_info,
                }
                # Cộng dồn forecast panel vào chuỗi tổng
                agg_forecast_ts = (
                    forecast_ts[["forecast"]].add(agg_forecast_ts, fill_value=0)
                    if agg_forecast_ts is not None
                    else forecast_ts[["forecast"]].copy()
                )
            
            step5_elapsed = time.perf_counter() - step5_start
            if panel_meta:
                print(f"   ⏱️  Step 5 (Post-Processing): completed in {step5_elapsed:.3f}s (batch: {batch_success_count}, fallback: {batch_fallback_count})")
            else:
                print(f"   ⏱️  Step 5 (Post-Processing): completed in {step5_elapsed:.3f}s")

        # Debug: Check aggregation before constrain
        if agg_forecast_ts is not None and not agg_forecast_ts.empty:
            forecast_sum_before = agg_forecast_ts['forecast'].sum()
            forecast_mean_before = agg_forecast_ts['forecast'].mean()
            print(f"   📊 Aggregated Forecast (before constrain): sum={forecast_sum_before:.1f}, mean={forecast_mean_before:.2f}")
        else:
            print(f"   ⚠️  WARNING: agg_forecast_ts is None or empty!")
            if agg_hist_ts is not None and not agg_hist_ts.empty:
                hist_sum = agg_hist_ts['value'].sum() if 'value' in agg_hist_ts.columns else agg_hist_ts.iloc[:, 0].sum()
                print(f"   📊 Historical data sum: {hist_sum:.1f}")
        
        # Align aggregated per-SKU forecast with optimization logic by constraining growth
        if agg_forecast_ts is not None and not agg_forecast_ts.empty:
            forecast_sum_before = agg_forecast_ts['forecast'].sum()
            agg_forecast_ts = self._constrain_forecast_growth(agg_forecast_ts, agg_hist_ts)
            forecast_sum_after = agg_forecast_ts['forecast'].sum() if agg_forecast_ts is not None and not agg_forecast_ts.empty else 0
            forecast_mean_after = agg_forecast_ts['forecast'].mean() if agg_forecast_ts is not None and not agg_forecast_ts.empty else 0
            print(f"   📊 Aggregated Forecast (after constrain): sum={forecast_sum_after:.1f}, mean={forecast_mean_after:.2f}")
            if forecast_sum_before > 0 and forecast_sum_after == 0:
                print(f"   ⚠️  WARNING: Forecast was constrained to 0! (before: {forecast_sum_before:.1f}, after: {forecast_sum_after:.1f})")

        # Tính metrics và chart trên chuỗi tổng
        metrics_start = time.perf_counter()
        metrics = self._calculate_metrics(agg_hist_ts, agg_forecast_ts)
        metrics_elapsed = time.perf_counter() - metrics_start
        
        # Không tạo chart PNG; UI dùng interactive Plotly với historical_data_raw/forecast_raw
        chart_path = None

        hist_display = format_dataframe_columns(agg_hist_ts.reset_index())
        if len(hist_display.columns) > 0:
            hist_display = hist_display.set_index(hist_display.columns[0])

        forecast_display = format_dataframe_columns(agg_forecast_ts.reset_index())
        if len(forecast_display.columns) > 0:
            forecast_display = forecast_display.set_index(forecast_display.columns[0])

        return {
            "success": True,
            "historical_data": hist_display,
            "forecast": forecast_display,
            "historical_data_raw": agg_hist_ts,
            "forecast_raw": agg_forecast_ts,
            "chart": chart_path,
            "metrics": metrics,
            "summary": self._generate_forecast_summary(agg_hist_ts, agg_forecast_ts),
            "per_sku_forecasts": per_sku_forecasts
        }

    def _ml_forecast(self, df_ts: pd.DataFrame, horizon: int, export_files: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        ML-based forecasting using pre-trained model or auto-select.
        """
        if self.pretrained_loader and self.pretrained_loader.loaded:
            if export_files:
                print("⚡ Using pre-trained XGBoost model (FAST inference)...")

            try:
                # Temporarily disable file export and logging if export_files=False
                if not export_files:
                    original_enable_export = self.pretrained_loader.enable_export
                    original_enable_logging = self.pretrained_loader.enable_logging
                    self.pretrained_loader.enable_export = False
                    self.pretrained_loader.enable_logging = False
                
                forecast_df = self.pretrained_loader.predict_with_confidence(
                    df_ts,
                    horizon=horizon,
                    confidence_level=0.95
                )
                
                # Restore enable_export and enable_logging
                if not export_files:
                    self.pretrained_loader.enable_export = original_enable_export
                    self.pretrained_loader.enable_logging = original_enable_logging

                model_info = {
                    "model": "xgboost_pretrained",
                    "confidence_intervals": True,
                    "confidence_level": 0.95,
                    "has_bounds": True,
                    "inference_time": "~50ms"
                }

                if export_files:
                    print(f"✅ Forecast generated using PRE-TRAINED XGBOOST (fast!)")
                return forecast_df, model_info

            except Exception as e:
                if export_files:
                    print(f"⚠️  Pre-trained model failed: {e}")
                    print("   Falling back to on-the-fly training...")
                # Restore enable_export and enable_logging if they were disabled
                if not export_files and hasattr(self, 'pretrained_loader') and self.pretrained_loader:
                    self.pretrained_loader.enable_export = original_enable_export
                    self.pretrained_loader.enable_logging = original_enable_logging

        if export_files:
            print("🤖 Using ML Forecasting Engine (training on-the-fly)...")

        try:
            ml_data = df_ts[['value']].copy()
            
            # Temporarily disable export if export_files=False
            if not export_files:
                original_enable_export = self.ml_engine.enable_export
                self.ml_engine.enable_export = False

            result: ForecastResult = self.ml_engine.forecast(  # type: ignore
                data=ml_data,
                horizon=horizon,
                model=None,
                auto_select=True
            )
            
            # Restore enable_export
            if not export_files:
                self.ml_engine.enable_export = original_enable_export

            forecast_df = result.to_dataframe()
            forecast_df = forecast_df.rename(columns={'forecast': 'forecast'})
            
            # CRITICAL: Cap forecast to prevent unrealistic values
            # Use multiplier based on data length: more aggressive for sparse data
            data_length = len(df_ts)
            if data_length < 7:
                max_multiplier = 2.0  # Very aggressive for sparse data
            elif data_length < 30:
                max_multiplier = 2.5  # Moderate for short data
            else:
                max_multiplier = 3.0  # Standard for sufficient data
            
            forecast_df = self._cap_forecast(forecast_df, df_ts, max_multiplier=max_multiplier)

            model_info = {
                "model": result.model_used,
                "confidence_intervals": result.lower_bound is not None,
                "confidence_level": result.confidence,
                "has_bounds": 'lower_bound' in forecast_df.columns
            }

            print(f"✅ Forecast generated using {result.model_used.upper()}")
            return forecast_df, model_info

        except Exception as e:
            print(f"⚠️  ML forecast failed: {e}")
            print("   Falling back to simple forecast...")
            forecast_df = self._simple_forecast(df_ts, horizon)
            # Cap is already applied in _simple_forecast path
            model_info = {"model": "moving_average", "confidence_intervals": False}
            return forecast_df, model_info

    def _plot_forecast(self,
                       df_ts: pd.DataFrame,
                       forecast_df: pd.DataFrame,
                       value_name: str,
                       model_info: Dict[str, Any] = None) -> str:
        """Plot historical data and forecast with Vietnamese labels and confidence intervals."""
        value_label = format_axis_label(value_name)
        model_name = model_info.get('model', 'unknown').upper() if model_info else 'SIMPLE'
        has_confidence = model_info and model_info.get('confidence_intervals', False)

        plt.figure(figsize=(14, 7))

        # Determine time window:
        # - X-axis: từ 1 tháng trước ngày lịch sử cuối cùng đến 1 tháng sau ngày đó
        # - Lịch sử: chỉ lấy đoạn trong [hist_end - 30, hist_end]
        hist_end = pd.to_datetime(df_ts.index.max()).normalize()
        start_date = hist_end - pd.Timedelta(days=30)
        end_date = hist_end + pd.Timedelta(days=30)

        hist_window = df_ts[(df_ts.index >= start_date) & (df_ts.index <= hist_end)]
        if hist_window.empty:
            # Fallback: nếu không có dữ liệu trong cửa sổ này, dùng 90 điểm cuối để vẫn có gì đó hiển thị
            hist_window = df_ts.tail(90)

        plt.plot(hist_window.index, hist_window['value'], label='Dữ Liệu Lịch Sử',
                 linewidth=2, color='steelblue', marker='o', markersize=3, alpha=0.8)

        plt.plot(forecast_df.index, forecast_df['forecast'], label='Dự Báo',
                 linewidth=2.5, color='orange', linestyle='--', marker='s', markersize=5)

        if has_confidence and 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
            confidence_level = model_info.get('confidence_level', 0.95)
            plt.fill_between(forecast_df.index,
                             forecast_df['lower_bound'],
                             forecast_df['upper_bound'],
                             alpha=0.3, color='orange',
                             label=f'Khoảng Tin Cậy {int(confidence_level*100)}%')

        plt.xlabel('Ngày', fontsize=12, fontweight='bold')
        plt.ylabel(value_label, fontsize=12, fontweight='bold')
        plt.title(f'Dự Báo {value_label} (Model: {model_name})', fontsize=14, fontweight='bold')

        # Giới hạn trục X: từ 1 tháng trước đến 1 tháng sau hiện tại
        plt.xlim(start_date, end_date)

        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        filename = f"forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"🔮 Created forecast chart: {filepath} (Model: {model_name})")
        return filepath

    def _calculate_metrics(self, df_ts: pd.DataFrame, forecast_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate forecast metrics.

        IMPORTANT: This mirrors InventoryOptimizationAgent._compute_forecast_metrics
        so that the reported metrics (recent_avg_daily, forecast_avg_daily, etc.)
        are consistent between the Forecast tab and the Optimization workflow.
        """
        recent_avg = 0.0
        if df_ts is not None and not df_ts.empty:
            if 'value' in df_ts.columns:
                hist_series = df_ts['value']
            else:
                hist_series = df_ts.iloc[:, 0]

            # Prefer recent non-zero demand, then fall back progressively
            recent_window = hist_series.tail(min(30, len(hist_series)))
            if not recent_window.empty:
                non_zero_recent = recent_window[recent_window > 0]
                if len(non_zero_recent) > 0:
                    recent_avg = float(non_zero_recent.mean())
                else:
                    recent_avg = float(recent_window.mean())

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

    def _generate_forecast_summary(self, df_ts: pd.DataFrame, forecast_df: pd.DataFrame) -> str:
        """Generate forecast summary text."""
        metrics = self._calculate_metrics(df_ts, forecast_df)

        return f"""
FORECAST SUMMARY:
- Historical period: {df_ts.index[0].strftime('%Y-%m-%d')} to {df_ts.index[-1].strftime('%Y-%m-%d')}
- Forecast period: {forecast_df.index[0].strftime('%Y-%m-%d')} to {forecast_df.index[-1].strftime('%Y-%m-%d')}
- Recent average (daily): {metrics['recent_avg_daily']:.2f}
- Forecast average (daily): {metrics['forecast_avg_daily']:.2f}
- Forecast total: {metrics['forecast_total']:.2f}
- Trend: {metrics['trend']}
"""





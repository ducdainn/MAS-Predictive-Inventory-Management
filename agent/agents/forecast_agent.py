"""
ForecastAgent: provides demand forecasting using pre-trained and on-the-fly models.
"""

import os
from datetime import timedelta
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent.label_formatter import format_axis_label
from agent.manager.database_manager import DatabaseManager
from agent.utils.dataframe_utils import format_dataframe_columns
from typing import Any

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


class ForecastAgent:
    """Performs time series forecasting with ML models (XGBoost, LightGBM, Prophet)."""

    def __init__(self,
                 db_manager: DatabaseManager,
                 output_dir: str = "charts",
                 use_ml: bool = True,
                 use_pretrained: bool = True):
        self.db = db_manager
        self.output_dir = output_dir
        self.use_ml = use_ml and ML_FORECASTING_AVAILABLE
        self.use_pretrained = use_pretrained and PRETRAINED_MODEL_AVAILABLE
        os.makedirs(output_dir, exist_ok=True)

        self.pretrained_loader = None
        if self.use_pretrained and get_model_loader:
            try:
                self.pretrained_loader = get_model_loader()
                if self.pretrained_loader.loaded:
                    print("✅ Pre-trained XGBoost model loaded (FAST inference!)")
                else:
                    print("ℹ️  No pre-trained model found, will train on-the-fly")
                    self.pretrained_loader = None
            except Exception as e:
                print(f"⚠️  Could not load pre-trained model: {e}")
                self.pretrained_loader = None

        if self.use_ml and ML_FORECASTING_AVAILABLE:
            self.ml_engine = MLForecastingEngine(confidence_level=0.95)
            if not self.pretrained_loader:
                print("✅ ML Forecasting Engine initialized (XGBoost, LightGBM, Prophet)")
        else:
            self.ml_engine = None
            print("ℹ️  Using simple forecasting (moving average + trend)")

    def forecast(self,
                 sql: str,
                 question: str,
                 horizon: int = 30,
                 create_chart: bool = True,
                 preloaded_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Execute query and perform forecasting."""
        print(f"🔮 Executing forecast query...")

        if preloaded_df is not None:
            df = preloaded_df.copy()
        else:
            df = self.db.execute_query(sql)

        if df.empty:
            return {"success": False, "message": "No historical data available"}

        print(f"✅ Retrieved {len(df)} historical data points")

        date_col, value_col = self._identify_columns(df)

        if not date_col or not value_col:
            return {"success": False, "message": "Could not identify date and value columns"}

        df_ts = self._prepare_time_series(df, date_col, value_col)

        if self.use_ml and self.ml_engine:
            forecast_result, model_info = self._ml_forecast(df_ts, horizon)
        else:
            forecast_result = self._simple_forecast(df_ts, horizon)
            model_info = {"model": "moving_average", "confidence_intervals": False}

        chart_path = None
        if create_chart:
            chart_path = self._plot_forecast(df_ts, forecast_result, value_col, model_info)
        metrics = self._calculate_metrics(df_ts, forecast_result)

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

    def _prepare_time_series(self, df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
        """Prepare time series data."""
        df_ts = df[[date_col, value_col]].copy()
        df_ts[date_col] = pd.to_datetime(df_ts[date_col])
        df_ts = df_ts.sort_values(date_col)
        df_ts = df_ts.set_index(date_col)
        df_ts.columns = ['value']
        df_ts = df_ts.resample('D').sum().fillna(0)
        return df_ts

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

    def _ml_forecast(self, df_ts: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        ML-based forecasting using pre-trained model or auto-select.
        """
        if self.pretrained_loader and self.pretrained_loader.loaded:
            print("⚡ Using pre-trained XGBoost model (FAST inference)...")

            try:
                forecast_df = self.pretrained_loader.predict_with_confidence(
                    df_ts,
                    horizon=horizon,
                    confidence_level=0.95
                )

                model_info = {
                    "model": "xgboost_pretrained",
                    "confidence_intervals": True,
                    "confidence_level": 0.95,
                    "has_bounds": True,
                    "inference_time": "~50ms"
                }

                print(f"✅ Forecast generated using PRE-TRAINED XGBOOST (fast!)")
                return forecast_df, model_info

            except Exception as e:
                print(f"⚠️  Pre-trained model failed: {e}")
                print("   Falling back to on-the-fly training...")

        print("🤖 Using ML Forecasting Engine (training on-the-fly)...")

        try:
            ml_data = df_ts[['value']].copy()

            result: ForecastResult = self.ml_engine.forecast(  # type: ignore
                data=ml_data,
                horizon=horizon,
                model=None,
                auto_select=True
            )

            forecast_df = result.to_dataframe()
            forecast_df = forecast_df.rename(columns={'forecast': 'forecast'})

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

        recent = df_ts.tail(90)
        plt.plot(recent.index, recent['value'], label='Dữ Liệu Lịch Sử',
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
        """Calculate forecast metrics."""
        recent_mean = df_ts['value'].tail(30).mean()
        forecast_mean = forecast_df['forecast'].mean()

        return {
            "recent_avg_daily": float(recent_mean),
            "forecast_avg_daily": float(forecast_mean),
            "forecast_total": float(forecast_df['forecast'].sum()),
            "trend": "increasing" if forecast_mean > recent_mean else "decreasing"
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





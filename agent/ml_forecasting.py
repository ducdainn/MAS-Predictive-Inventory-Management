"""
🤖 Advanced ML-Based Forecasting Engine
========================================

Multi-model ensemble forecasting system for inventory demand prediction.

Models:
1. Prophet - Facebook's robust forecaster (handles seasonality, holidays)
2. SARIMA - Seasonal ARIMA (statistical approach)
3. Exponential Smoothing - Simple but effective
4. Moving Average - Baseline (existing)

Features:
- Auto-model selection based on data characteristics
- Confidence intervals
- Model performance tracking
- Handles sparse/missing data
- Seasonality detection

"""

import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# Try to import advanced models (install if needed)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("WARNING: Prophet not installed. Install with: pip install prophet")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("WARNING: statsmodels not installed. Install with: pip install statsmodels")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")


@dataclass
class ForecastResult:
    """Forecast result with metadata."""
    dates: pd.DatetimeIndex
    forecast: np.ndarray
    lower_bound: Optional[np.ndarray] = None
    upper_bound: Optional[np.ndarray] = None
    model_used: str = "moving_average"
    confidence: float = 0.95
    metrics: Dict = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        df = pd.DataFrame({
            'date': self.dates,
            'forecast': self.forecast
        })
        
        if self.lower_bound is not None:
            df['lower_bound'] = self.lower_bound
        if self.upper_bound is not None:
            df['upper_bound'] = self.upper_bound
            
        return df.set_index('date')


@dataclass
class DataCharacteristics:
    """Time series data characteristics."""
    length: int
    has_trend: bool
    has_seasonality: bool
    seasonality_period: Optional[int]
    volatility: float
    missing_pct: float
    is_sparse: bool
    
    def recommend_model(self) -> str:
        """Recommend best model based on characteristics."""
        # Sparse data → Simple models
        if self.is_sparse or self.length < 14:
            return 'moving_average'
        
        # Short data → Exponential Smoothing
        if self.length < 30:
            return 'exponential_smoothing'
        
        # XGBoost for medium to long data with patterns
        if XGBOOST_AVAILABLE and self.length >= 45:
            if self.volatility < 0.5:  # Stable patterns
                return 'xgboost'
        
        # Has clear seasonality → Prophet or SARIMA
        if self.has_seasonality:
            if PROPHET_AVAILABLE and self.length >= 60:
                return 'prophet'
            elif STATSMODELS_AVAILABLE:
                return 'sarima'
            else:
                return 'exponential_smoothing'
        
        # Medium data with trend → Exponential Smoothing
        if self.has_trend and STATSMODELS_AVAILABLE:
            return 'exponential_smoothing'
        
        # Default
        return 'moving_average'


class MLForecastingEngine:
    """
    Advanced ML forecasting engine with multiple models.
    
    Usage:
        engine = MLForecastingEngine()
        result = engine.forecast(df, horizon=30)
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize forecasting engine.
        
        Args:
            confidence_level: Confidence level for intervals (default: 0.95)
        """
        self.confidence_level = confidence_level
        self.models = {
            'xgboost': self._xgboost_forecast if XGBOOST_AVAILABLE else None,
            'prophet': self._prophet_forecast if PROPHET_AVAILABLE else None,
            'sarima': self._sarima_forecast if STATSMODELS_AVAILABLE else None,
            'exponential_smoothing': self._exp_smoothing_forecast if STATSMODELS_AVAILABLE else None,
            'moving_average': self._moving_average_forecast
        }
        
        # Remove unavailable models
        self.models = {k: v for k, v in self.models.items() if v is not None}
    
    def forecast(
        self, 
        data: pd.DataFrame, 
        horizon: int = 30,
        model: Optional[str] = None,
        auto_select: bool = True
    ) -> ForecastResult:
        """
        Generate forecast with best model.
        
        Args:
            data: DataFrame with datetime index and 'value' column
            horizon: Number of periods to forecast
            model: Specific model to use (None for auto-selection)
            auto_select: Auto-select best model based on data
            
        Returns:
            ForecastResult with predictions and metadata
        """
        # Analyze data characteristics
        characteristics = self._analyze_data(data)
        
        # Select model
        if model is None and auto_select:
            model = characteristics.recommend_model()
        elif model is None:
            model = 'moving_average'
        
        print(f"🤖 Using model: {model.upper()}")
        print(f"   Data length: {characteristics.length} days")
        print(f"   Trend: {'Yes' if characteristics.has_trend else 'No'}")
        print(f"   Seasonality: {'Yes' if characteristics.has_seasonality else 'No'}")
        print(f"   Volatility: {characteristics.volatility:.2f}")
        
        # Check if model is available
        if model not in self.models:
            print(f"⚠️  Model '{model}' not available, falling back to moving_average")
            model = 'moving_average'
        
        # Generate forecast
        try:
            result = self.models[model](data, horizon, characteristics)
            result.model_used = model
            return result
        except Exception as e:
            print(f"❌ {model} failed: {e}")
            print(f"   Falling back to moving_average")
            result = self._moving_average_forecast(data, horizon, characteristics)
            result.model_used = 'moving_average'
            return result
    
    def _analyze_data(self, data: pd.DataFrame) -> DataCharacteristics:
        """Analyze time series characteristics."""
        values = data['value'].values
        length = len(values)
        
        # Missing data
        missing_pct = (values == 0).sum() / length
        is_sparse = missing_pct > 0.5 or length < 7
        
        # Trend detection (simple linear regression)
        if length > 3:
            x = np.arange(length)
            slope = np.polyfit(x, values, 1)[0]
            has_trend = abs(slope) > (np.std(values) / length)
        else:
            has_trend = False
        
        # Seasonality detection (simple autocorrelation)
        has_seasonality = False
        seasonality_period = None
        
        if length >= 14 and not is_sparse:
            try:
                # Check for weekly seasonality (7 days)
                autocorr_7 = pd.Series(values).autocorr(lag=7)
                if autocorr_7 > 0.3:
                    has_seasonality = True
                    seasonality_period = 7
            except:
                pass
        
        # Volatility (coefficient of variation)
        mean_val = np.mean(values[values > 0]) if any(values > 0) else 1
        std_val = np.std(values)
        volatility = std_val / mean_val if mean_val > 0 else 0
        
        return DataCharacteristics(
            length=length,
            has_trend=has_trend,
            has_seasonality=has_seasonality,
            seasonality_period=seasonality_period,
            volatility=volatility,
            missing_pct=missing_pct,
            is_sparse=is_sparse
        )
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create time series features for XGBoost.
        
        Features:
        - Lag features (t-1, t-2, t-3, t-7, t-14, t-30)
        - Rolling statistics (MA7, MA14, MA30, std7, std14)
        - Date features (day_of_week, day_of_month, month, quarter)
        - Trend feature
        """
        df = data.copy()
        df['value'] = data['value'].values
        
        # Lag features
        for lag in [1, 2, 3, 7, 14, 30]:
            if len(df) > lag:
                df[f'lag_{lag}'] = df['value'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 30]:
            if len(df) > window:
                df[f'rolling_mean_{window}'] = df['value'].rolling(window=window, min_periods=1).mean()
                df[f'rolling_std_{window}'] = df['value'].rolling(window=window, min_periods=1).std()
        
        # Date features
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Trend feature
        df['trend'] = np.arange(len(df))
        
        # Fill NaN values (from lag features)
        df = df.fillna(method='bfill').fillna(0)
        
        return df
    
    def _xgboost_forecast(
        self,
        data: pd.DataFrame,
        horizon: int,
        characteristics: DataCharacteristics
    ) -> ForecastResult:
        """Forecast using XGBoost with feature engineering."""
        print("   📊 Training XGBoost model...")
        print("      🔧 Feature engineering...")
        
        # Create features
        df_features = self._create_features(data)
        
        # Prepare train data
        feature_cols = [col for col in df_features.columns if col != 'value']
        X_train = df_features[feature_cols].values
        y_train = df_features['value'].values
        
        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Train model
        print("      🚀 Training...")
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        
        # Generate forecast
        print("      🔮 Generating forecast...")
        forecasts = []
        lower_bounds = []
        upper_bounds = []
        
        # Extend data for forecasting
        last_date = data.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        
        # Create a temporary dataframe for recursive forecasting
        temp_df = data.copy()
        
        for i, future_date in enumerate(forecast_dates):
            # Add dummy row
            temp_df.loc[future_date] = 0
            
            # Create features for this point
            df_feat = self._create_features(temp_df)
            X_future = df_feat[feature_cols].iloc[-1:].values
            
            # Predict
            pred = model.predict(X_future)[0]
            pred = max(0, pred)  # No negative demand
            
            # Update temp_df with prediction
            temp_df.loc[future_date] = pred
            
            forecasts.append(pred)
            
            # Calculate confidence intervals (using training residuals)
            train_pred = model.predict(X_train)
            residuals = y_train - train_pred
            std_err = np.std(residuals)
            z_score = 1.96  # 95% confidence
            
            lower_bounds.append(max(0, pred - z_score * std_err))
            upper_bounds.append(pred + z_score * std_err)
        
        print(f"      ✅ Forecast complete (mean: {np.mean(forecasts):.1f})")
        
        return ForecastResult(
            dates=forecast_dates,
            forecast=np.array(forecasts),
            lower_bound=np.array(lower_bounds),
            upper_bound=np.array(upper_bounds),
            model_used='xgboost',
            confidence=self.confidence_level
        )
    
    def _prophet_forecast(
        self, 
        data: pd.DataFrame, 
        horizon: int,
        characteristics: DataCharacteristics
    ) -> ForecastResult:
        """Forecast using Facebook Prophet."""
        print("   📊 Training Prophet model...")
        
        # Prepare data for Prophet (needs 'ds' and 'y' columns)
        prophet_df = pd.DataFrame({
            'ds': data.index,
            'y': data['value'].values
        })
        
        # Initialize Prophet with appropriate parameters
        model = Prophet(
            yearly_seasonality=False,  # Not enough data typically
            weekly_seasonality=characteristics.has_seasonality,
            daily_seasonality=False,
            seasonality_mode='multiplicative' if characteristics.volatility > 0.5 else 'additive',
            interval_width=self.confidence_level
        )
        
        # Fit model
        model.fit(prophet_df)
        
        # Generate future dataframe
        future = model.make_future_dataframe(periods=horizon, freq='D')
        
        # Predict
        forecast = model.predict(future)
        
        # Extract forecast for horizon only
        forecast_only = forecast.tail(horizon)
        
        return ForecastResult(
            dates=pd.DatetimeIndex(forecast_only['ds']),
            forecast=forecast_only['yhat'].values,
            lower_bound=forecast_only['yhat_lower'].values,
            upper_bound=forecast_only['yhat_upper'].values,
            model_used='prophet',
            confidence=self.confidence_level
        )
    
    def _sarima_forecast(
        self,
        data: pd.DataFrame,
        horizon: int,
        characteristics: DataCharacteristics
    ) -> ForecastResult:
        """Forecast using SARIMA."""
        print("   📊 Training SARIMA model...")
        
        values = data['value'].values
        
        # SARIMA parameters (p, d, q) x (P, D, Q, s)
        # Simplified auto-selection
        if characteristics.has_seasonality:
            order = (1, 1, 1)  # (p, d, q)
            seasonal_order = (1, 1, 1, characteristics.seasonality_period or 7)  # (P, D, Q, s)
        else:
            order = (1, 1, 1)
            seasonal_order = (0, 0, 0, 0)
        
        # Fit model
        model = SARIMAX(
            values,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted = model.fit(disp=False)
        
        # Forecast
        forecast_result = fitted.forecast(steps=horizon)
        
        # Calculate confidence intervals (simple approach)
        std_err = np.std(fitted.resid)
        z_score = 1.96  # 95% confidence
        
        return ForecastResult(
            dates=pd.date_range(
                start=data.index[-1] + timedelta(days=1),
                periods=horizon,
                freq='D'
            ),
            forecast=forecast_result.values,
            lower_bound=forecast_result.values - z_score * std_err,
            upper_bound=forecast_result.values + z_score * std_err,
            model_used='sarima',
            confidence=self.confidence_level
        )
    
    def _exp_smoothing_forecast(
        self,
        data: pd.DataFrame,
        horizon: int,
        characteristics: DataCharacteristics
    ) -> ForecastResult:
        """Forecast using Exponential Smoothing."""
        print("   📊 Training Exponential Smoothing model...")
        
        values = data['value'].values
        
        # Determine seasonal parameter
        seasonal = None
        seasonal_periods = None
        
        if characteristics.has_seasonality and len(values) >= 14:
            seasonal = 'add'
            seasonal_periods = characteristics.seasonality_period or 7
        
        # Fit model
        model = ExponentialSmoothing(
            values,
            trend='add' if characteristics.has_trend else None,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        )
        
        fitted = model.fit()
        
        # Forecast
        forecast_result = fitted.forecast(steps=horizon)
        
        # Calculate confidence intervals (simple approach)
        std_err = np.std(fitted.resid)
        z_score = 1.96
        
        return ForecastResult(
            dates=pd.date_range(
                start=data.index[-1] + timedelta(days=1),
                periods=horizon,
                freq='D'
            ),
            forecast=forecast_result.values,
            lower_bound=forecast_result.values - z_score * std_err,
            upper_bound=forecast_result.values + z_score * std_err,
            model_used='exponential_smoothing',
            confidence=self.confidence_level
        )
    
    def _moving_average_forecast(
        self,
        data: pd.DataFrame,
        horizon: int,
        characteristics: DataCharacteristics
    ) -> ForecastResult:
        """Forecast using Moving Average + Trend (baseline)."""
        print("   📊 Using Moving Average + Trend...")
        
        values = data['value'].values
        
        # Calculate moving averages
        if len(values) >= 30:
            ma_window = 30
        elif len(values) >= 7:
            ma_window = 7
        else:
            ma_window = max(2, len(values))
        
        # Moving average
        ma = pd.Series(values).rolling(window=ma_window, min_periods=1).mean()
        
        # Trend estimation
        if len(values) > 3:
            recent_values = values[-min(30, len(values)):]
            x = np.arange(len(recent_values))
            trend_coef = np.polyfit(x, recent_values, 1)
            daily_trend = trend_coef[0]
        else:
            daily_trend = 0
        
        # Generate forecast
        base_value = ma.iloc[-1]
        forecast = np.array([
            max(0, base_value + daily_trend * i)
            for i in range(1, horizon + 1)
        ])
        
        # Simple confidence intervals (±20%)
        std = np.std(values) if len(values) > 1 else base_value * 0.2
        
        return ForecastResult(
            dates=pd.date_range(
                start=data.index[-1] + timedelta(days=1),
                periods=horizon,
                freq='D'
            ),
            forecast=forecast,
            lower_bound=np.maximum(0, forecast - 1.96 * std),
            upper_bound=forecast + 1.96 * std,
            model_used='moving_average',
            confidence=self.confidence_level
        )
    
    def evaluate_model(
        self,
        data: pd.DataFrame,
        test_size: int = 30,
        model: Optional[str] = None
    ) -> Dict:
        """
        Evaluate model performance using train-test split.
        
        Args:
            data: Historical data
            test_size: Number of periods for testing
            model: Model to evaluate (None for auto-selection)
            
        Returns:
            Dict with evaluation metrics
        """
        if len(data) < test_size + 14:
            return {
                'error': 'Insufficient data for evaluation',
                'min_required': test_size + 14,
                'available': len(data)
            }
        
        # Split data
        train = data.iloc[:-test_size]
        test = data.iloc[-test_size:]
        
        # Generate forecast
        result = self.forecast(train, horizon=test_size, model=model)
        
        # Calculate metrics
        actual = test['value'].values
        predicted = result.forecast[:len(actual)]  # Match lengths
        
        # MAE, RMSE, MAPE
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted)**2))
        
        # MAPE (avoid division by zero)
        mape = np.mean(np.abs((actual - predicted) / np.where(actual != 0, actual, 1))) * 100
        
        # Bias
        bias = np.mean(predicted - actual)
        
        return {
            'model': result.model_used,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'bias': bias,
            'test_size': test_size,
            'train_size': len(train)
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compare_models(
    data: pd.DataFrame,
    horizon: int = 30,
    models: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare performance of different models.
    
    Args:
        data: Historical data
        horizon: Forecast horizon
        models: List of models to compare (None for all available)
        
    Returns:
        DataFrame with comparison results
    """
    engine = MLForecastingEngine()
    
    if models is None:
        models = list(engine.models.keys())
    
    results = []
    
    for model in models:
        print(f"\n🧪 Evaluating {model.upper()}...")
        try:
            metrics = engine.evaluate_model(data, test_size=min(30, len(data) // 4), model=model)
            results.append(metrics)
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('mape')  # Best MAPE first
        return df
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    """Example usage and testing."""
    print("="*80)
    print("ML FORECASTING ENGINE - TEST")
    print("="*80)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
    
    # Synthetic demand with trend and weekly seasonality
    trend = np.linspace(100, 150, 90)
    seasonality = 20 * np.sin(np.arange(90) * 2 * np.pi / 7)  # Weekly pattern
    noise = np.random.normal(0, 10, 90)
    demand = trend + seasonality + noise
    demand = np.maximum(0, demand)  # No negative demand
    
    df = pd.DataFrame({'value': demand}, index=dates)
    
    print(f"\n📊 Test Data:")
    print(f"   Length: {len(df)} days")
    print(f"   Range: {df.index[0]} to {df.index[-1]}")
    print(f"   Mean: {df['value'].mean():.1f}")
    print(f"   Std: {df['value'].std():.1f}")
    
    # Test forecasting
    engine = MLForecastingEngine()
    
    print(f"\n{'='*80}")
    print(f"TEST 1: AUTO-SELECT MODEL")
    print(f"{'='*80}")
    
    result = engine.forecast(df, horizon=30)
    
    print(f"\n✅ Forecast Generated:")
    print(f"   Model: {result.model_used.upper()}")
    print(f"   Horizon: {len(result.forecast)} days")
    print(f"   Mean forecast: {result.forecast.mean():.1f}")
    print(f"   Has confidence intervals: {result.lower_bound is not None}")
    
    # Compare models
    if len(engine.models) > 1:
        print(f"\n{'='*80}")
        print(f"TEST 2: MODEL COMPARISON")
        print(f"{'='*80}")
        
        comparison = compare_models(df, horizon=30)
        
        if not comparison.empty:
            print(f"\n📊 Model Performance:")
            print(comparison.to_string(index=False))
            print(f"\n🏆 Best Model: {comparison.iloc[0]['model'].upper()} (MAPE: {comparison.iloc[0]['mape']:.1f}%)")
    
    print(f"\n{'='*80}")
    print("✅ ML Forecasting Engine Ready!")
    print("="*80)


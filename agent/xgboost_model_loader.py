"""
XGBoost Model Loader for Production Inference
==============================================

Fast inference using pre-trained XGBoost models.
"""

import os
import json
import pickle
from typing import Dict, Optional
from datetime import datetime

import pandas as pd
import numpy as np

# Suppress pandas fillna deprecation warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='.*fillna.*')


class XGBoostModelLoader:
    """Load and use pre-trained XGBoost models for fast inference."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.model = None
        self.feature_names = []
        self.metadata = {}
        self.loaded = False
        self.model_version = "unknown"
    
    def load_latest_model(self) -> bool:
        """Load the most recent model version."""
        try:
            # Find all model files (support both naming conventions)
            model_files = [f for f in os.listdir(self.models_dir) 
                          if (f.startswith('xgboost_model_') or f.startswith('xgboost_forecast_')) 
                          and f.endswith('.pkl')]
            
            if not model_files:
                print("⚠️  No pre-trained models found")
                return False
            
            # Sort by modification time (most recent first)
            model_files.sort(key=lambda x: os.path.getmtime(
                os.path.join(self.models_dir, x)
            ), reverse=True)
            
            latest_model = model_files[0]
            model_path = os.path.join(self.models_dir, latest_model)
            self.model_version = latest_model.replace('.pkl', '')
            
            # Load model directly
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Extract timestamp from filename
            timestamp = latest_model.replace('xgboost_model_', '').replace('xgboost_forecast_', '').replace('.pkl', '')
            
            # Try to load metadata
            metadata_files = [f for f in os.listdir(self.models_dir)
                            if f.startswith('metadata_') and f.endswith('.json')]
            
            if metadata_files:
                # Find matching metadata by timestamp
                metadata_file = [f for f in metadata_files if timestamp in f]
                if metadata_file:
                    metadata_path = os.path.join(self.models_dir, metadata_file[0])
                    with open(metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                    
                    # Extract features from metadata
                    if 'features' in self.metadata:
                        self.feature_names = self.metadata['features']
            
            # If no features in metadata, get from model
            if not self.feature_names:
                if hasattr(self.model, 'feature_names_in_'):
                    self.feature_names = list(self.model.feature_names_in_)
                elif hasattr(self.model, 'get_booster'):
                    self.feature_names = self.model.get_booster().feature_names
            
            self.loaded = True
            
            print(f"✅ Loaded XGBoost model: {latest_model}")
            print(f"   • Model type: {type(self.model).__name__}")
            print(f"   • Features: {len(self.feature_names)}")
            if 'metrics' in self.metadata:
                test_r2 = self.metadata['metrics'].get('test_r2', 0)
                print(f"   • Test R²: {test_r2:.4f}")
            if 'trained_at' in self.metadata:
                print(f"   • Trained at: {self.metadata['trained_at']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_model(self, version: str) -> bool:
        """
        Load specific model version.
        
        Args:
            version: Model version (e.g., 'v1', 'v20241114_123456')
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            # Load model
            model_path = os.path.join(self.models_dir, f'xgboost_forecast_{version}.pkl')
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load metadata
            metadata_path = os.path.join(self.models_dir, f'model_metadata_{version}.json')
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Load feature names
            features_path = os.path.join(self.models_dir, f'feature_names_{version}.txt')
            with open(features_path, 'r') as f:
                self.feature_names = [line.strip() for line in f]
            
            self.loaded = True
            
            print(f"✅ Loaded XGBoost model: {version}")
            print(f"   • Trained: {self.metadata.get('trained_at', 'Unknown')}")
            print(f"   • Features: {len(self.feature_names)}")
            print(f"   • R² Score: {self.metadata.get('final_metrics', {}).get('r2', 0):.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model {version}: {e}")
            return False
    
    def create_features_from_timeseries(self, df_ts: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from time series data matching training script.
        
        Args:
            df_ts: DataFrame with DatetimeIndex and 'value' column
            
        Returns:
            DataFrame with features matching training
        """
        df = df_ts.copy()
        
        # Rename 'value' to 'quantity' for consistency
        if 'value' in df.columns:
            df['quantity'] = df['value']
        
        # Date features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['day_of_year'] = df.index.dayofyear
        
        # Lag features (matching training: [1, 7, 14, 30])
        # CRITICAL: Fill NaN with forward fill (use most recent value) to avoid model predicting 0
        for lag in [1, 7, 14, 30]:
            if len(df) > lag:
                df[f'lag_{lag}'] = df['quantity'].shift(lag)
                # Fill NaN with forward fill (most recent available value) then backward fill
                df[f'lag_{lag}'] = df[f'lag_{lag}'].ffill().bfill()
                # If still NaN (no data at all), use 0
                df[f'lag_{lag}'] = df[f'lag_{lag}'].fillna(0)
        
        # Rolling statistics (matching training: [7, 14, 30])
        for window in [7, 14, 30]:
            if len(df) > window:
                df[f'rolling_mean_{window}'] = df['quantity'].rolling(
                    window=window, min_periods=1
                ).mean()
                df[f'rolling_std_{window}'] = df['quantity'].rolling(
                    window=window, min_periods=1
                ).std().fillna(0)  # Fix: fillna(0) instead of training model
                df[f'rolling_min_{window}'] = df['quantity'].rolling(
                    window=window, min_periods=1
                ).min()
                df[f'rolling_max_{window}'] = df['quantity'].rolling(
                    window=window, min_periods=1
                ).max()
        
        # Change features - fill NaN with 0 (no change)
        df['change_1'] = df['quantity'].diff(1).fillna(0)
        df['change_7'] = df['quantity'].diff(7).fillna(0)
        df['pct_change_1'] = df['quantity'].pct_change(1).fillna(0)
        df['pct_change_7'] = df['quantity'].pct_change(7).fillna(0)
        
        # Trend features
        def trend(series):
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            return np.polyfit(x, series, 1)[0]
        
        df['trend_7'] = df['quantity'].rolling(7, min_periods=2).apply(trend, raw=True)
        df['trend_30'] = df['quantity'].rolling(30, min_periods=2).apply(trend, raw=True)
        
        # Volatility
        df['volatility_7'] = (df['rolling_std_7'] / df['rolling_mean_7']).fillna(0)
        df['volatility_30'] = (df['rolling_std_30'] / df['rolling_mean_30']).fillna(0)
        
        # Fill remaining NaN values systematically
        # 1. Forward fill (use previous value)
        df = df.ffill()
        # 2. Backward fill (use next value)
        df = df.bfill()
        # 3. Fill any remaining NaN with 0
        df = df.fillna(0)
        
        # CRITICAL: Ensure lag features are never NaN for last row (used for prediction)
        # If last row has NaN lag features, model will predict 0
        for lag in [1, 7, 14, 30]:
            lag_col = f'lag_{lag}'
            if lag_col in df.columns:
                if pd.isna(df[lag_col].iloc[-1]):
                    # Use most recent non-null value or 0
                    last_valid = df[lag_col].dropna()
                    if len(last_valid) > 0:
                        df[lag_col].iloc[-1] = last_valid.iloc[-1]
                    else:
                        df[lag_col].iloc[-1] = 0
        
        return df
    
    def predict(self, 
                df_ts: pd.DataFrame, 
                horizon: int = 30) -> pd.DataFrame:
        """
        Generate forecast using pre-trained model.
        
        Args:
            df_ts: Historical time series with 'value' column
            horizon: Number of days to forecast
            
        Returns:
            DataFrame with forecast
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Create features from historical data
        df_features = self.create_features_from_timeseries(df_ts)
        
        # Generate forecast recursively
        forecasts = []
        
        # Extend dataframe for forecasting
        last_date = df_ts.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        
        # Create temporary dataframe for recursive forecasting
        temp_df = df_ts.copy()
        
        for future_date in forecast_dates:
            # Add dummy row with previous value (better than 0)
            last_value = temp_df['value'].iloc[-1] if len(temp_df) > 0 else 0
            temp_df.loc[future_date] = last_value
            
            # Create features
            df_feat = self.create_features_from_timeseries(temp_df)
            
            # Get features for prediction (last row)
            # CRITICAL: Ensure no NaN in features for prediction
            X_future = df_feat[self.feature_names].iloc[-1:].values
            
            # Check for NaN in features
            if pd.isna(X_future).any():
                # Fill NaN with 0 or mean
                X_future = np.nan_to_num(X_future, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Predict
            pred = self.model.predict(X_future)[0]
            pred = max(0, pred)  # No negative demand
            
            # Update temp_df with prediction
            temp_df.loc[future_date, 'value'] = pred
            
            forecasts.append(pred)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'forecast': forecasts
        }, index=forecast_dates)
        
        return forecast_df
    
    def predict_with_confidence(self,
                               df_ts: pd.DataFrame,
                               horizon: int = 30,
                               confidence_level: float = 0.95) -> pd.DataFrame:
        """
        Generate forecast with confidence intervals.
        
        Uses historical residuals to estimate uncertainty.
        """
        # Get point forecast
        forecast_df = self.predict(df_ts, horizon)
        
        # Calculate residuals on historical data
        df_features = self.create_features_from_timeseries(df_ts)
        X_hist = df_features[self.feature_names].values
        y_hist = df_ts['value'].values
        
        # Predict on historical data
        y_pred_hist = self.model.predict(X_hist)
        residuals = y_hist - y_pred_hist
        
        # Calculate standard error
        std_err = np.std(residuals)
        
        # Calculate confidence intervals
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        forecast_df['lower_bound'] = np.maximum(
            forecast_df['forecast'] - z_score * std_err, 0
        )
        forecast_df['upper_bound'] = forecast_df['forecast'] + z_score * std_err
        
        return forecast_df
    
    def get_model_info(self) -> Dict:
        """Get model metadata."""
        if not self.loaded:
            return {"error": "Model not loaded"}
        
        return {
            "loaded": True,
            "model_type": self.metadata.get('model_type', 'Unknown'),
            "version": self.metadata.get('version', self.model_version),
            "trained_at": self.metadata.get('trained_at', 'Unknown'),
            "n_features": len(self.feature_names),
            "metrics": self.metadata.get('final_metrics', {}),
            "feature_names": self.feature_names[:10]  # Top 10 for display
        }

    def get_version_string(self) -> str:
        """Return a human-readable model version identifier."""
        info = self.get_model_info()
        return f"{info.get('version', 'unknown')} (trained_at={info.get('trained_at', 'N/A')})"


# Singleton instance for production use
_model_loader_instance = None

def get_model_loader(models_dir: str = "models") -> XGBoostModelLoader:
    """Get or create singleton model loader instance."""
    global _model_loader_instance
    
    if _model_loader_instance is None:
        _model_loader_instance = XGBoostModelLoader(models_dir)
        _model_loader_instance.load_latest_model()
    
    return _model_loader_instance


# Example usage
if __name__ == "__main__":
    print("Testing XGBoost Model Loader...")
    
    # Load model
    loader = XGBoostModelLoader()
    
    if loader.load_latest_model():
        print("\n✅ Model loaded successfully!")
        print("\nModel Info:")
        info = loader.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test prediction with dummy data
        print("\n🧪 Testing prediction...")
        
        # Create dummy time series
        dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
        values = np.random.randint(50, 150, size=90)
        df_ts = pd.DataFrame({'value': values}, index=dates)
        
        # Predict
        forecast = loader.predict(df_ts, horizon=30)
        
        print(f"\n✅ Forecast generated!")
        print(f"   • Forecast length: {len(forecast)} days")
        print(f"   • Forecast range: [{forecast['forecast'].min():.1f}, {forecast['forecast'].max():.1f}]")
        print(f"   • Mean forecast: {forecast['forecast'].mean():.1f}")
        
        # With confidence intervals
        forecast_ci = loader.predict_with_confidence(df_ts, horizon=30)
        print(f"\n✅ Forecast with confidence intervals!")
        print(f"   • Lower bound: {forecast_ci['lower_bound'].mean():.1f}")
        print(f"   • Forecast: {forecast_ci['forecast'].mean():.1f}")
        print(f"   • Upper bound: {forecast_ci['upper_bound'].mean():.1f}")
    else:
        print("\n❌ No model available")
        print("\nTo train a model:")
        print("  python agent/train_xgboost_model.py")



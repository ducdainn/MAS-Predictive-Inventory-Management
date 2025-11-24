"""
XGBoost Model Training for Predictive Inventory Management
Professional implementation with OOT testing and comprehensive analysis

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import json
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
import xgboost as xgb
import joblib
import os

# Visualization setup
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_path': r'D:\Study\KLTN\BrickDemand\rawData\Data_FSKU_cleanedDate.csv',
    'model_dir': 'models',
    'plots_dir': 'plots',
    'random_seed': 42,
    
    # OOT Split (Out-Of-Time Testing)
    'train_end_date': '2025-05-31',  # Train: 2023-01-02 to 2025-05-31
    'test_start_date': '2025-06-01',  # Test:  2025-06-01 to 2025-06-30 (1 month OOT)
    
    # Feature engineering
    'lag_features': [1, 7, 14, 30],
    'rolling_windows': [7, 14, 30],
    
    # Model hyperparameters (tuned)
    'xgb_params': {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'min_child_weight': 3,
        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1
    }
}

# Create directories
os.makedirs(CONFIG['model_dir'], exist_ok=True)
os.makedirs(CONFIG['plots_dir'], exist_ok=True)

print("="*80)
print("🤖 XGBOOST MODEL TRAINING FOR INVENTORY FORECASTING")
print("="*80)


# ============================================================================
# 1. DATA LOADING & EDA
# ============================================================================

def load_and_explore_data(file_path):
    """Load data and perform initial exploration."""
    print("\n📊 Step 1: Loading Data...")
    
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    print(f"   ✅ Loaded {len(df):,} records")
    print(f"   Columns: {df.shape[1]}")
    print(f"\n   📋 Column Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"      {i:2d}. {col}")
    
    # Parse date
    if 'Ngày' in df.columns:
        df['date'] = pd.to_datetime(df['Ngày'])
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    print(f"\n   📅 Date Range:")
    print(f"      Start: {df['date'].min()}")
    print(f"      End:   {df['date'].max()}")
    print(f"      Days:  {(df['date'].max() - df['date'].min()).days}")
    
    # Target variable
    if 'Số lượng' in df.columns:
        df['quantity'] = df['Số lượng']
    
    print(f"\n   📊 Target (Quantity) Statistics:")
    print(f"      Mean:   {df['quantity'].mean():.2f}")
    print(f"      Median: {df['quantity'].median():.2f}")
    print(f"      Std:    {df['quantity'].std():.2f}")
    print(f"      Min:    {df['quantity'].min():.2f}")
    print(f"      Max:    {df['quantity'].max():.2f}")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n   ⚠️  Missing Values:")
        for col, count in missing[missing > 0].items():
            print(f"      {col}: {count} ({count/len(df)*100:.1f}%)")
    else:
        print(f"\n   ✅ No missing values")
    
    return df


# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

def create_time_series_features(df):
    """
    Create comprehensive time series features.
    
    Features:
    - Date features (day of week, month, quarter, etc.)
    - Lag features (t-1, t-7, t-14, t-30)
    - Rolling statistics (mean, std, min, max)
    - Trend and seasonality indicators
    """
    print("\n🔧 Step 2: Feature Engineering...")
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Date features
    print("   Creating date features...")
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    # Aggregate by date for time series
    print("   Aggregating daily totals...")
    daily_df = df.groupby('date').agg({
        'quantity': 'sum'
    }).reset_index()
    
    # Lag features
    print(f"   Creating lag features: {CONFIG['lag_features']}...")
    for lag in CONFIG['lag_features']:
        daily_df[f'lag_{lag}'] = daily_df['quantity'].shift(lag)
    
    # Rolling statistics
    print(f"   Creating rolling features: {CONFIG['rolling_windows']}...")
    for window in CONFIG['rolling_windows']:
        daily_df[f'rolling_mean_{window}'] = daily_df['quantity'].rolling(window).mean()
        daily_df[f'rolling_std_{window}'] = daily_df['quantity'].rolling(window).std()
        daily_df[f'rolling_min_{window}'] = daily_df['quantity'].rolling(window).min()
        daily_df[f'rolling_max_{window}'] = daily_df['quantity'].rolling(window).max()
    
    # Advanced features
    print("   Creating advanced features...")
    # Rate of change
    daily_df['change_1'] = daily_df['quantity'].diff(1)
    daily_df['change_7'] = daily_df['quantity'].diff(7)
    daily_df['pct_change_1'] = daily_df['quantity'].pct_change(1)
    daily_df['pct_change_7'] = daily_df['quantity'].pct_change(7)
    
    # Trend (linear regression coefficient over window)
    def trend(series):
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0]
    
    daily_df['trend_7'] = daily_df['quantity'].rolling(7).apply(trend, raw=True)
    daily_df['trend_30'] = daily_df['quantity'].rolling(30).apply(trend, raw=True)
    
    # Volatility (coefficient of variation)
    daily_df['volatility_7'] = (daily_df['rolling_std_7'] / daily_df['rolling_mean_7']).fillna(0)
    daily_df['volatility_30'] = (daily_df['rolling_std_30'] / daily_df['rolling_mean_30']).fillna(0)
    
    # Date features for daily_df
    daily_df['day_of_week'] = pd.to_datetime(daily_df['date']).dt.dayofweek
    daily_df['month'] = pd.to_datetime(daily_df['date']).dt.month
    daily_df['quarter'] = pd.to_datetime(daily_df['date']).dt.quarter
    daily_df['day_of_year'] = pd.to_datetime(daily_df['date']).dt.dayofyear
    
    # Drop NaN rows (from lag/rolling calculations)
    print("   Removing NaN rows from feature engineering...")
    initial_rows = len(daily_df)
    daily_df = daily_df.dropna()
    removed_rows = initial_rows - len(daily_df)
    print(f"   Removed {removed_rows} rows with NaN values")
    
    print(f"   ✅ Final dataset: {len(daily_df)} rows, {len(daily_df.columns)} columns")
    
    return daily_df


# ============================================================================
# 3. FEATURE SELECTION
# ============================================================================

def select_important_features(X_train, y_train, threshold=0.001):
    """
    Select important features using XGBoost feature importance.
    
    Args:
        X_train: Training features
        y_train: Training target
        threshold: Minimum importance score to keep feature
    
    Returns:
        List of selected feature names
    """
    print("\n🎯 Step 3: Feature Selection...")
    
    # Train a quick model for feature importance
    print("   Training preliminary model for feature selection...")
    model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=CONFIG['random_seed']
    )
    
    model.fit(X_train, y_train)
    
    # Get feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Select features above threshold
    selected_features = importance[importance['importance'] >= threshold]['feature'].tolist()
    
    print(f"\n   📊 Feature Importance (Top 20):")
    for i, row in importance.head(20).iterrows():
        bar = '█' * int(row['importance'] * 100)
        print(f"      {row['feature']:25s} {row['importance']:.4f} {bar}")
    
    print(f"\n   ✅ Selected {len(selected_features)}/{len(X_train.columns)} features")
    print(f"      (importance >= {threshold})")
    
    return selected_features, importance


# ============================================================================
# 4. TRAIN/TEST SPLIT (OOT)
# ============================================================================

def split_train_test_oot(df, train_end, test_start):
    """
    Split data into train/test using Out-Of-Time approach.
    
    Args:
        df: DataFrame with features
        train_end: Last date for training (inclusive)
        test_start: First date for testing (inclusive)
    
    Returns:
        X_train, X_test, y_train, y_test, train_df, test_df
    """
    print("\n✂️  Step 4: Train/Test Split (OOT)...")
    
    # Split
    train_df = df[df['date'] <= train_end].copy()
    test_df = df[df['date'] >= test_start].copy()
    
    print(f"   📅 Training Period:")
    print(f"      Start: {train_df['date'].min()}")
    print(f"      End:   {train_df['date'].max()}")
    print(f"      Days:  {len(train_df)}")
    print(f"      %:     {len(train_df)/len(df)*100:.1f}%")
    
    print(f"\n   📅 Testing Period (OOT):")
    print(f"      Start: {test_df['date'].min()}")
    print(f"      End:   {test_df['date'].max()}")
    print(f"      Days:  {len(test_df)}")
    print(f"      %:     {len(test_df)/len(df)*100:.1f}%")
    
    # Features to exclude
    exclude_cols = ['date', 'quantity']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df['quantity']
    X_test = test_df[feature_cols]
    y_test = test_df['quantity']
    
    print(f"\n   ✅ Features: {len(feature_cols)}")
    print(f"   ✅ Train samples: {len(X_train)}")
    print(f"   ✅ Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, train_df, test_df, feature_cols


# ============================================================================
# 5. MODEL TRAINING
# ============================================================================

def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Train XGBoost model with validation."""
    print("\n🚀 Step 5: Training XGBoost Model...")
    
    print("   Hyperparameters:")
    for key, value in CONFIG['xgb_params'].items():
        print(f"      {key:20s}: {value}")
    
    # Train with early stopping
    model = xgb.XGBRegressor(**CONFIG['xgb_params'])
    
    print("\n   Training in progress...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    print("\n   📊 Training Metrics:")
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
    
    print(f"      RMSE:  {train_rmse:.2f}")
    print(f"      MAE:   {train_mae:.2f}")
    print(f"      R²:    {train_r2:.4f}")
    print(f"      MAPE:  {train_mape*100:.2f}%")
    
    print("\n   📊 Testing Metrics (OOT):")
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
    
    print(f"      RMSE:  {test_rmse:.2f}")
    print(f"      MAE:   {test_mae:.2f}")
    print(f"      R²:    {test_r2:.4f}")
    print(f"      MAPE:  {test_mape*100:.2f}%")
    
    metrics = {
        'train': {'rmse': train_rmse, 'mae': train_mae, 'r2': train_r2, 'mape': train_mape},
        'test': {'rmse': test_rmse, 'mae': test_mae, 'r2': test_r2, 'mape': test_mape},
        'predictions': {
            'y_train': y_train,
            'y_train_pred': y_train_pred,
            'y_test': y_test,
            'y_test_pred': y_test_pred
        }
    }
    
    print("\n   ✅ Model trained successfully!")
    
    return model, metrics


# ============================================================================
# 6. VISUALIZATION
# ============================================================================

def plot_comprehensive_analysis(model, metrics, importance_df, train_df, test_df):
    """Create comprehensive visualization suite."""
    print("\n📊 Step 6: Creating Visualizations...")
    
    # Figure 1: Training Results (2x2)
    fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle('XGBoost Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1.1 Actual vs Predicted (Train)
    ax = axes[0, 0]
    ax.scatter(metrics['predictions']['y_train'], 
               metrics['predictions']['y_train_pred'],
               alpha=0.3, s=20)
    ax.plot([metrics['predictions']['y_train'].min(), metrics['predictions']['y_train'].max()],
            [metrics['predictions']['y_train'].min(), metrics['predictions']['y_train'].max()],
            'r--', lw=2)
    ax.set_xlabel('Actual Quantity', fontsize=12)
    ax.set_ylabel('Predicted Quantity', fontsize=12)
    ax.set_title(f"Training Set: Actual vs Predicted\nR² = {metrics['train']['r2']:.4f}", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 1.2 Actual vs Predicted (Test)
    ax = axes[0, 1]
    ax.scatter(metrics['predictions']['y_test'], 
               metrics['predictions']['y_test_pred'],
               alpha=0.5, s=30, c='orange')
    ax.plot([metrics['predictions']['y_test'].min(), metrics['predictions']['y_test'].max()],
            [metrics['predictions']['y_test'].min(), metrics['predictions']['y_test'].max()],
            'r--', lw=2)
    ax.set_xlabel('Actual Quantity', fontsize=12)
    ax.set_ylabel('Predicted Quantity', fontsize=12)
    ax.set_title(f"Test Set (OOT): Actual vs Predicted\nR² = {metrics['test']['r2']:.4f}", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 1.3 Residuals (Train)
    ax = axes[1, 0]
    residuals_train = metrics['predictions']['y_train'] - metrics['predictions']['y_train_pred']
    ax.scatter(metrics['predictions']['y_train_pred'], residuals_train, alpha=0.3, s=20)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Quantity', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title(f"Training Set: Residual Plot\nRMSE = {metrics['train']['rmse']:.2f}", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 1.4 Residuals (Test)
    ax = axes[1, 1]
    residuals_test = metrics['predictions']['y_test'] - metrics['predictions']['y_test_pred']
    ax.scatter(metrics['predictions']['y_test_pred'], residuals_test, alpha=0.5, s=30, c='orange')
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Quantity', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title(f"Test Set (OOT): Residual Plot\nRMSE = {metrics['test']['rmse']:.2f}", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig1.savefig(f"{CONFIG['plots_dir']}/01_model_performance.png", dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: 01_model_performance.png")
    
    # Figure 2: Time Series Predictions
    fig2, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig2.suptitle('Time Series Forecast: Actual vs Predicted', fontsize=16, fontweight='bold')
    
    # 2.1 Training Period
    ax = axes[0]
    train_dates = train_df['date'].values[-len(metrics['predictions']['y_train']):]
    ax.plot(train_dates, metrics['predictions']['y_train'], label='Actual', linewidth=2, alpha=0.7)
    ax.plot(train_dates, metrics['predictions']['y_train_pred'], label='Predicted', linewidth=2, alpha=0.7)
    ax.fill_between(train_dates, metrics['predictions']['y_train'], metrics['predictions']['y_train_pred'], 
                     alpha=0.2, color='gray')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Quantity', fontsize=12)
    ax.set_title(f"Training Period | MAE = {metrics['train']['mae']:.2f} | MAPE = {metrics['train']['mape']*100:.2f}%", 
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 2.2 Testing Period (OOT)
    ax = axes[1]
    test_dates = test_df['date'].values
    ax.plot(test_dates, metrics['predictions']['y_test'], label='Actual', linewidth=2, alpha=0.7, color='green')
    ax.plot(test_dates, metrics['predictions']['y_test_pred'], label='Predicted', linewidth=2, alpha=0.7, color='red')
    ax.fill_between(test_dates, metrics['predictions']['y_test'], metrics['predictions']['y_test_pred'], 
                     alpha=0.2, color='gray')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Quantity', fontsize=12)
    ax.set_title(f"Test Period (OOT) | MAE = {metrics['test']['mae']:.2f} | MAPE = {metrics['test']['mape']*100:.2f}%", 
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    fig2.savefig(f"{CONFIG['plots_dir']}/02_time_series_forecast.png", dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: 02_time_series_forecast.png")
    
    # Figure 3: Feature Importance
    fig3, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig3.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
    
    # 3.1 Top 20 Features
    ax = axes[0]
    top_features = importance_df.head(20)
    ax.barh(range(len(top_features)), top_features['importance'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Top 20 Most Important Features', fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # 3.2 Feature Importance Distribution
    ax = axes[1]
    ax.hist(importance_df['importance'], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Feature Importance Distribution', fontsize=12)
    ax.axvline(importance_df['importance'].mean(), color='r', linestyle='--', 
               linewidth=2, label=f'Mean = {importance_df["importance"].mean():.4f}')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig3.savefig(f"{CONFIG['plots_dir']}/03_feature_importance.png", dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: 03_feature_importance.png")
    
    # Figure 4: Error Analysis
    fig4, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig4.suptitle('Error Analysis', fontsize=16, fontweight='bold')
    
    # 4.1 Error Distribution (Train)
    ax = axes[0, 0]
    errors_train = metrics['predictions']['y_train'] - metrics['predictions']['y_train_pred']
    ax.hist(errors_train, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Prediction Error', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f"Training: Error Distribution\nMean Error = {errors_train.mean():.2f}", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 4.2 Error Distribution (Test)
    ax = axes[0, 1]
    errors_test = metrics['predictions']['y_test'] - metrics['predictions']['y_test_pred']
    ax.hist(errors_test, bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Prediction Error', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f"Test (OOT): Error Distribution\nMean Error = {errors_test.mean():.2f}", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 4.3 Percentage Error Distribution (Train)
    ax = axes[1, 0]
    pct_errors_train = ((metrics['predictions']['y_train'] - metrics['predictions']['y_train_pred']) / 
                        metrics['predictions']['y_train'] * 100)
    pct_errors_train = pct_errors_train[np.isfinite(pct_errors_train)]  # Remove inf
    ax.hist(pct_errors_train, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Percentage Error (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f"Training: Percentage Error Distribution", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 4.4 Percentage Error Distribution (Test)
    ax = axes[1, 1]
    pct_errors_test = ((metrics['predictions']['y_test'] - metrics['predictions']['y_test_pred']) / 
                       metrics['predictions']['y_test'] * 100)
    pct_errors_test = pct_errors_test[np.isfinite(pct_errors_test)]  # Remove inf
    ax.hist(pct_errors_test, bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Percentage Error (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f"Test (OOT): Percentage Error Distribution", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig4.savefig(f"{CONFIG['plots_dir']}/04_error_analysis.png", dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: 04_error_analysis.png")
    
    plt.close('all')
    
    print(f"\n   📊 All plots saved to: {CONFIG['plots_dir']}/")


# ============================================================================
# 7. MODEL SAVING
# ============================================================================

def save_model_and_metadata(model, feature_cols, importance_df, metrics):
    """Save model and associated metadata."""
    print("\n💾 Step 7: Saving Model...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"xgboost_model_{timestamp}.pkl"
    model_path = os.path.join(CONFIG['model_dir'], model_filename)
    
    # Save model
    joblib.dump(model, model_path)
    print(f"   ✅ Model saved: {model_path}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_filename': model_filename,
        'features': feature_cols,
        'n_features': len(feature_cols),
        'metrics': {
            'train_rmse': metrics['train']['rmse'],
            'train_mae': metrics['train']['mae'],
            'train_r2': metrics['train']['r2'],
            'train_mape': metrics['train']['mape'],
            'test_rmse': metrics['test']['rmse'],
            'test_mae': metrics['test']['mae'],
            'test_r2': metrics['test']['r2'],
            'test_mape': metrics['test']['mape']
        },
        'hyperparameters': CONFIG['xgb_params'],
        'data_info': {
            'train_end': CONFIG['train_end_date'],
            'test_start': CONFIG['test_start_date']
        }
    }
    
    metadata_path = os.path.join(CONFIG['model_dir'], f"metadata_{timestamp}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Metadata saved: {metadata_path}")
    
    # Save feature importance
    importance_path = os.path.join(CONFIG['model_dir'], f"feature_importance_{timestamp}.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"   ✅ Feature importance saved: {importance_path}")
    
    return model_path, metadata_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    
    try:
        # 1. Load data
        df = load_and_explore_data(CONFIG['data_path'])
        
        # 2. Feature engineering
        df_features = create_time_series_features(df)
        
        # 3. Train/test split (OOT)
        X_train, X_test, y_train, y_test, train_df, test_df, feature_cols = split_train_test_oot(
            df_features,
            CONFIG['train_end_date'],
            CONFIG['test_start_date']
        )
        
        # 4. Feature selection
        selected_features, importance_df = select_important_features(X_train, y_train)
        
        # Use selected features
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        # 5. Train model
        model, metrics = train_xgboost_model(
            X_train_selected, y_train,
            X_test_selected, y_test
        )
        
        # 6. Visualizations
        plot_comprehensive_analysis(model, metrics, importance_df, train_df, test_df)
        
        # 7. Save model
        model_path, metadata_path = save_model_and_metadata(
            model, selected_features, importance_df, metrics
        )
        
        # Final summary
        print("\n" + "="*80)
        print("🎉 TRAINING COMPLETE!")
        print("="*80)
        
        print(f"\n📊 Final Metrics:")
        print(f"   Training   - RMSE: {metrics['train']['rmse']:.2f} | R²: {metrics['train']['r2']:.4f}")
        print(f"   Test (OOT) - RMSE: {metrics['test']['rmse']:.2f} | R²: {metrics['test']['r2']:.4f}")
        
        print(f"\n💾 Saved Files:")
        print(f"   Model: {model_path}")
        print(f"   Metadata: {metadata_path}")
        print(f"   Plots: {CONFIG['plots_dir']}/")
        
        print(f"\n✅ Model ready for production use!")
        print("="*80)
        
        return model, metrics
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    model, metrics = main()


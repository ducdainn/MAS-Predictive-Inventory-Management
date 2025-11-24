"""
Train and Save XGBoost Model for Production Forecasting
=======================================================

This script trains an XGBoost model on historical sales data
and saves it for fast inference in production.

Usage:
    python agent/train_xgboost_model.py
    
Output:
    - models/xgboost_forecast_v1.pkl  (trained model)
    - models/feature_scaler_v1.pkl    (optional scaler)
    - models/model_metadata_v1.json   (model info)
"""

import os
import sys
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.manager.database_manager import DatabaseManager


class XGBoostModelTrainer:
    """Train and manage XGBoost forecasting models."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.model = None
        self.feature_names = []
        self.model_metadata = {}
        
        # Create models directory
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
    
    def load_training_data(self, 
                          days_back: int = 365,
                          min_sales_per_item: int = 30) -> pd.DataFrame:
        """
        Load historical sales data for training.
        
        Args:
            days_back: How many days of history to load
            min_sales_per_item: Minimum sales records per product
            
        Returns:
            DataFrame with aggregated daily sales
        """
        print(f"\n{'='*70}")
        print("📊 LOADING TRAINING DATA")
        print(f"{'='*70}\n")
        
        # Query to get aggregated daily sales
        query = f"""
        SELECT 
            date,
            product_code,
            branch_code,
            SUM(quantity) as quantity
        FROM sales
        WHERE date >= CURRENT_DATE - INTERVAL '{days_back} days'
        GROUP BY date, product_code, branch_code
        HAVING SUM(quantity) > 0
        ORDER BY date, product_code, branch_code
        """
        
        print(f"Loading {days_back} days of sales data...")
        df = self.db.execute_query(query)
        
        print(f"✅ Loaded {len(df):,} sales records")
        print(f"   • Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   • Unique products: {df['product_code'].nunique()}")
        print(f"   • Unique branches: {df['branch_code'].nunique()}")
        
        # Filter products with sufficient history
        product_counts = df.groupby('product_code').size()
        valid_products = product_counts[product_counts >= min_sales_per_item].index
        
        df_filtered = df[df['product_code'].isin(valid_products)]
        
        print(f"✅ Filtered to {len(df_filtered):,} records")
        print(f"   • Products with {min_sales_per_item}+ records: {len(valid_products)}")
        
        return df_filtered
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time series features for XGBoost.
        
        Features:
        - Lag features (1, 2, 3, 7, 14, 30 days)
        - Rolling statistics (MA7, MA14, MA30, std)
        - Date features (day_of_week, month, quarter)
        - Trend features
        """
        print(f"\n{'='*70}")
        print("🔧 FEATURE ENGINEERING")
        print(f"{'='*70}\n")
        
        df = df.sort_values(['product_code', 'branch_code', 'date'])
        
        features_list = []
        
        # Group by product-branch combination
        groups = df.groupby(['product_code', 'branch_code'])
        total_groups = len(groups)
        
        print(f"Creating features for {total_groups} product-branch combinations...")
        
        for i, ((product, branch), group) in enumerate(groups, 1):
            if i % 100 == 0:
                print(f"   Progress: {i}/{total_groups} ({i/total_groups*100:.1f}%)")
            
            # Set date as index
            group = group.set_index('date').sort_index()
            
            # Create features
            feat_df = pd.DataFrame(index=group.index)
            feat_df['product_code'] = product
            feat_df['branch_code'] = branch
            feat_df['quantity'] = group['quantity']
            
            # Lag features
            for lag in [1, 2, 3, 7, 14, 30]:
                feat_df[f'lag_{lag}'] = feat_df['quantity'].shift(lag)
            
            # Rolling statistics
            for window in [7, 14, 30]:
                feat_df[f'rolling_mean_{window}'] = feat_df['quantity'].rolling(
                    window=window, min_periods=1
                ).mean()
                feat_df[f'rolling_std_{window}'] = feat_df['quantity'].rolling(
                    window=window, min_periods=1
                ).std()
            
            # Date features
            feat_df['day_of_week'] = feat_df.index.dayofweek
            feat_df['day_of_month'] = feat_df.index.day
            feat_df['month'] = feat_df.index.month
            feat_df['quarter'] = feat_df.index.quarter
            feat_df['is_weekend'] = (feat_df.index.dayofweek >= 5).astype(int)
            
            # Trend feature (days since start)
            feat_df['trend'] = np.arange(len(feat_df))
            
            # Fill NaN values
            feat_df = feat_df.fillna(method='bfill').fillna(0)
            
            features_list.append(feat_df.reset_index())
        
        # Combine all features
        features_df = pd.concat(features_list, ignore_index=True)
        
        print(f"\n✅ Feature engineering complete")
        print(f"   • Total samples: {len(features_df):,}")
        print(f"   • Total features: {len(features_df.columns) - 4}")  # Exclude meta columns
        
        return features_df
    
    def train_model(self, 
                   features_df: pd.DataFrame,
                   test_size: int = 30,
                   cv_folds: int = 3) -> Dict:
        """
        Train XGBoost model with cross-validation.
        
        Args:
            features_df: DataFrame with features
            test_size: Days to hold out for testing
            cv_folds: Number of CV folds
            
        Returns:
            Dict with training results
        """
        print(f"\n{'='*70}")
        print("🚀 TRAINING XGBOOST MODEL")
        print(f"{'='*70}\n")
        
        # Prepare features and target
        meta_cols = ['date', 'product_code', 'branch_code', 'quantity']
        feature_cols = [col for col in features_df.columns if col not in meta_cols]
        
        X = features_df[feature_cols].values
        y = features_df['quantity'].values
        
        print(f"Training data:")
        print(f"   • Samples: {len(X):,}")
        print(f"   • Features: {len(feature_cols)}")
        print(f"   • Target range: [{y.min():.0f}, {y.max():.0f}]")
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        print(f"\n⏳ Training with {cv_folds}-fold time series cross-validation...")
        
        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Train with CV
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            print(f"\n   Fold {fold}/{cv_folds}:")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, verbose=False)
            
            # Validate
            y_pred = model.predict(X_val)
            y_pred = np.maximum(y_pred, 0)  # No negative predictions
            
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            print(f"      RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
            
            cv_scores.append({'rmse': rmse, 'mae': mae, 'r2': r2})
        
        # Train final model on all data
        print(f"\n🎯 Training final model on full dataset...")
        
        final_model = xgb.XGBRegressor(**params)
        final_model.fit(X, y, verbose=False)
        
        # Calculate final metrics
        y_pred_final = final_model.predict(X)
        y_pred_final = np.maximum(y_pred_final, 0)
        
        final_rmse = np.sqrt(mean_squared_error(y, y_pred_final))
        final_mae = mean_absolute_error(y, y_pred_final)
        final_r2 = r2_score(y, y_pred_final)
        
        print(f"\n✅ Final model trained!")
        print(f"   • RMSE: {final_rmse:.2f}")
        print(f"   • MAE: {final_mae:.2f}")
        print(f"   • R²: {final_r2:.4f}")
        
        # Store model and metadata
        self.model = final_model
        self.feature_names = feature_cols
        
        self.model_metadata = {
            'model_type': 'XGBRegressor',
            'version': '1.0',
            'trained_at': datetime.now().isoformat(),
            'training_samples': len(X),
            'n_features': len(feature_cols),
            'feature_names': feature_cols,
            'cv_scores': cv_scores,
            'final_metrics': {
                'rmse': float(final_rmse),
                'mae': float(final_mae),
                'r2': float(final_r2)
            },
            'hyperparameters': params,
            'target_range': [float(y.min()), float(y.max())]
        }
        
        return self.model_metadata
    
    def save_model(self, version: str = "v1"):
        """Save trained model to disk."""
        print(f"\n{'='*70}")
        print("💾 SAVING MODEL")
        print(f"{'='*70}\n")
        
        # Save model
        model_path = os.path.join(self.models_dir, f'xgboost_forecast_{version}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✅ Model saved: {model_path}")
        
        # Save metadata
        metadata_path = os.path.join(self.models_dir, f'model_metadata_{version}.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        print(f"✅ Metadata saved: {metadata_path}")
        
        # Save feature names
        features_path = os.path.join(self.models_dir, f'feature_names_{version}.txt')
        with open(features_path, 'w') as f:
            f.write('\n'.join(self.feature_names))
        print(f"✅ Feature names saved: {features_path}")
        
        print(f"\n📦 Model package ready for production!")
        print(f"   • Model: {model_path}")
        print(f"   • Metadata: {metadata_path}")
        print(f"   • Features: {features_path}")
    
    def evaluate_model(self, features_df: pd.DataFrame, sample_size: int = 1000):
        """Evaluate model on sample data."""
        print(f"\n{'='*70}")
        print("📊 MODEL EVALUATION")
        print(f"{'='*70}\n")
        
        # Sample data
        if len(features_df) > sample_size:
            sample_df = features_df.sample(n=sample_size, random_state=42)
        else:
            sample_df = features_df
        
        meta_cols = ['date', 'product_code', 'branch_code', 'quantity']
        X_sample = sample_df[[f for f in self.feature_names if f in sample_df.columns]].values
        y_sample = sample_df['quantity'].values
        
        # Predict
        y_pred = self.model.predict(X_sample)
        y_pred = np.maximum(y_pred, 0)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_sample, y_pred))
        mae = mean_absolute_error(y_sample, y_pred)
        r2 = r2_score(y_sample, y_pred)
        
        # MAPE
        mask = y_sample > 0
        mape = np.mean(np.abs((y_sample[mask] - y_pred[mask]) / y_sample[mask])) * 100
        
        print(f"Evaluation on {len(sample_df)} samples:")
        print(f"   • RMSE: {rmse:.2f}")
        print(f"   • MAE: {mae:.2f}")
        print(f"   • R²: {r2:.4f}")
        print(f"   • MAPE: {mape:.2f}%")
        
        # Feature importance
        print(f"\n🎯 Top 10 Most Important Features:")
        importances = self.model.feature_importances_
        feature_imp = sorted(zip(self.feature_names, importances), 
                           key=lambda x: x[1], reverse=True)
        
        for i, (feat, imp) in enumerate(feature_imp[:10], 1):
            bar = '█' * int(imp * 50)
            print(f"   {i:2d}. {feat:20s} {imp:.4f} {bar}")


def main():
    """Main training pipeline."""
    print("\n" + "="*70)
    print("🎯 XGBOOST MODEL TRAINING PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now()}")
    print("="*70 + "\n")
    
    # Initialize
    print("Initializing database connection...")
    try:
        db_manager = DatabaseManager()
        print("✅ Connected to database\n")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("\nPlease ensure:")
        print("  1. Docker Desktop is running")
        print("  2. PostgreSQL container is up: docker-compose up -d")
        return
    
    # Create trainer
    trainer = XGBoostModelTrainer(db_manager)
    
    # Step 1: Load data
    try:
        df = trainer.load_training_data(days_back=365, min_sales_per_item=30)
        
        if df.empty:
            print("❌ No training data available")
            return
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Step 2: Create features
    try:
        features_df = trainer.create_features(df)
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return
    
    # Step 3: Train model
    try:
        metadata = trainer.train_model(features_df, test_size=30, cv_folds=3)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Evaluate
    try:
        trainer.evaluate_model(features_df, sample_size=1000)
    except Exception as e:
        print(f"⚠️ Evaluation failed: {e}")
    
    # Step 5: Save model
    try:
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trainer.save_model(version=version)
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        return
    
    # Summary
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*70}\n")
    
    print("📊 Model Summary:")
    print(f"   • Version: {version}")
    print(f"   • Samples: {metadata['training_samples']:,}")
    print(f"   • Features: {metadata['n_features']}")
    print(f"   • R² Score: {metadata['final_metrics']['r2']:.4f}")
    print(f"   • RMSE: {metadata['final_metrics']['rmse']:.2f}")
    print(f"   • MAE: {metadata['final_metrics']['mae']:.2f}")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Review model performance above")
    print(f"   2. Test model: python agent/test_saved_model.py")
    print(f"   3. Deploy to production: Update ForecastAgent to load this model")
    print(f"   4. Monitor performance and retrain periodically")
    
    print(f"\n💾 Model Files:")
    print(f"   • models/xgboost_forecast_{version}.pkl")
    print(f"   • models/model_metadata_{version}.json")
    print(f"   • models/feature_names_{version}.txt")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()



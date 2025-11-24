"""
Comprehensive XGBoost Visualization & Analysis
==============================================

Detailed feature importance, predictions, and model diagnostics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Run the benchmark first to get data
import sys
sys.path.append('.')
from agent.benchmark_models import (
    load_sample_data_from_db, 
    DataExplorer,
    FeatureEngineer,
    ModelBenchmark
)

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class XGBoostAnalyzer:
    """Comprehensive XGBoost analysis and visualization."""
    
    def __init__(self, model, X_train, X_test, y_train, y_test, 
                 feature_names, test_dates, data_characteristics):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.test_dates = test_dates
        self.data_chars = data_characteristics
        
        # Generate predictions
        self.train_pred = model.predict(X_train)
        self.test_pred = model.predict(X_test)
        self.test_pred = np.maximum(self.test_pred, 0)  # No negatives
        
    def create_all_plots(self):
        """Generate all visualization plots."""
        print("\n" + "="*70)
        print("📊 GENERATING XGBOOST VISUALIZATIONS")
        print("="*70 + "\n")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams['font.size'] = 10
        
        # 1. Feature Importance (3 different views)
        self.plot_feature_importance()
        
        # 2. Predictions Analysis
        self.plot_predictions_detailed()
        
        # 3. Residual Analysis
        self.plot_residual_analysis()
        
        # 4. Error Distribution
        self.plot_error_distribution()
        
        # 5. Learning Curves (if available)
        self.plot_learning_curves()
        
        # 6. Feature Correlation
        self.plot_feature_correlations()
        
        # 7. Actual vs Predicted Scatter
        self.plot_actual_vs_predicted()
        
        # 8. Time-based Error Analysis
        self.plot_temporal_errors()
        
        print("\n✅ All visualizations saved!")
        print("="*70 + "\n")
    
    def plot_feature_importance(self):
        """Plot feature importance in 3 different ways."""
        print("1️⃣  Feature Importance Analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('XGBoost Feature Importance Analysis', 
                     fontsize=16, fontweight='bold')
        
        # Get feature importance
        importance_gain = self.model.get_booster().get_score(importance_type='gain')
        importance_weight = self.model.get_booster().get_score(importance_type='weight')
        importance_cover = self.model.get_booster().get_score(importance_type='cover')
        
        # Map feature names (f0, f1, ... to actual names)
        def map_features(importance_dict):
            mapped = {}
            for key, value in importance_dict.items():
                idx = int(key[1:])  # Remove 'f' prefix
                if idx < len(self.feature_names):
                    mapped[self.feature_names[idx]] = value
            return mapped
        
        importance_gain = map_features(importance_gain)
        importance_weight = map_features(importance_weight)
        importance_cover = map_features(importance_cover)
        
        # 1. Gain (most important)
        ax1 = axes[0, 0]
        top_gain = sorted(importance_gain.items(), key=lambda x: x[1], reverse=True)[:15]
        features_gain = [x[0] for x in top_gain]
        values_gain = [x[1] for x in top_gain]
        
        bars1 = ax1.barh(range(len(features_gain)), values_gain, color='#2E86AB')
        ax1.set_yticks(range(len(features_gain)))
        ax1.set_yticklabels(features_gain)
        ax1.set_xlabel('Gain (Information Gain)', fontweight='bold')
        ax1.set_title('Top 15 Features by Gain\n(Most Important for Accuracy)', 
                     fontweight='bold')
        ax1.invert_yaxis()
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, values_gain)):
            ax1.text(val, i, f' {val:.0f}', va='center', fontweight='bold')
        
        # 2. Weight (frequency)
        ax2 = axes[0, 1]
        top_weight = sorted(importance_weight.items(), key=lambda x: x[1], reverse=True)[:15]
        features_weight = [x[0] for x in top_weight]
        values_weight = [x[1] for x in top_weight]
        
        bars2 = ax2.barh(range(len(features_weight)), values_weight, color='#A23B72')
        ax2.set_yticks(range(len(features_weight)))
        ax2.set_yticklabels(features_weight)
        ax2.set_xlabel('Weight (Frequency)', fontweight='bold')
        ax2.set_title('Top 15 Features by Weight\n(Most Frequently Used)', 
                     fontweight='bold')
        ax2.invert_yaxis()
        
        for i, (bar, val) in enumerate(zip(bars2, values_weight)):
            ax2.text(val, i, f' {val:.0f}', va='center', fontweight='bold')
        
        # 3. Cover (samples affected)
        ax3 = axes[1, 0]
        top_cover = sorted(importance_cover.items(), key=lambda x: x[1], reverse=True)[:15]
        features_cover = [x[0] for x in top_cover]
        values_cover = [x[1] for x in top_cover]
        
        bars3 = ax3.barh(range(len(features_cover)), values_cover, color='#F18F01')
        ax3.set_yticks(range(len(features_cover)))
        ax3.set_yticklabels(features_cover)
        ax3.set_xlabel('Cover (Samples Affected)', fontweight='bold')
        ax3.set_title('Top 15 Features by Cover\n(Affects Most Samples)', 
                     fontweight='bold')
        ax3.invert_yaxis()
        
        for i, (bar, val) in enumerate(zip(bars3, values_cover)):
            ax3.text(val, i, f' {val:.0f}', va='center', fontweight='bold')
        
        # 4. Normalized comparison of top 10
        ax4 = axes[1, 1]
        top_10_features = [x[0] for x in top_gain[:10]]
        
        # Normalize to 0-100 scale
        def normalize_dict(d, features):
            values = [d.get(f, 0) for f in features]
            max_val = max(values) if max(values) > 0 else 1
            return [v / max_val * 100 for v in values]
        
        gain_norm = normalize_dict(importance_gain, top_10_features)
        weight_norm = normalize_dict(importance_weight, top_10_features)
        cover_norm = normalize_dict(importance_cover, top_10_features)
        
        x = np.arange(len(top_10_features))
        width = 0.25
        
        ax4.bar(x - width, gain_norm, width, label='Gain', color='#2E86AB')
        ax4.bar(x, weight_norm, width, label='Weight', color='#A23B72')
        ax4.bar(x + width, cover_norm, width, label='Cover', color='#F18F01')
        
        ax4.set_ylabel('Normalized Importance (0-100)', fontweight='bold')
        ax4.set_title('Top 10 Features: Comparison Across Metrics', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(top_10_features, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('xgboost_feature_importance.png', dpi=150, bbox_inches='tight')
        print("   ✅ Saved: xgboost_feature_importance.png")
        plt.close()
        
        # Print detailed importance table
        self._print_feature_importance_table(importance_gain, importance_weight, importance_cover)
    
    def _print_feature_importance_table(self, gain, weight, cover):
        """Print detailed feature importance table."""
        print("\n" + "="*70)
        print("📋 DETAILED FEATURE IMPORTANCE TABLE")
        print("="*70)
        
        # Combine all metrics
        all_features = set(list(gain.keys()) + list(weight.keys()) + list(cover.keys()))
        
        importance_data = []
        for feat in all_features:
            importance_data.append({
                'Feature': feat,
                'Gain': gain.get(feat, 0),
                'Weight': weight.get(feat, 0),
                'Cover': cover.get(feat, 0)
            })
        
        # Sort by gain
        importance_data = sorted(importance_data, key=lambda x: x['Gain'], reverse=True)
        
        # Print top 20
        print(f"\n{'Rank':<5} {'Feature':<22} {'Gain':>12} {'Weight':>10} {'Cover':>12}")
        print("-" * 70)
        
        for i, item in enumerate(importance_data[:20], 1):
            print(f"{i:<5} {item['Feature']:<22} {item['Gain']:>12.1f} "
                  f"{item['Weight']:>10.0f} {item['Cover']:>12.1f}")
        
        print("-" * 70)
        print(f"Total features: {len(importance_data)}")
        print("="*70 + "\n")
    
    def plot_predictions_detailed(self):
        """Detailed prediction analysis."""
        print("2️⃣  Predictions Analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('XGBoost Predictions: Detailed Analysis', 
                     fontsize=16, fontweight='bold')
        
        # 1. Test set predictions
        ax1 = axes[0, 0]
        ax1.plot(self.test_dates, self.y_test, 'o-', label='Actual', 
                linewidth=2.5, markersize=8, color='#2E86AB')
        ax1.plot(self.test_dates, self.test_pred, 's--', label='Predicted', 
                linewidth=2.5, markersize=8, color='#F18F01', alpha=0.8)
        ax1.fill_between(self.test_dates, self.y_test, self.test_pred, 
                         alpha=0.2, color='gray')
        ax1.set_xlabel('Date', fontweight='bold')
        ax1.set_ylabel('Value', fontweight='bold')
        ax1.set_title('Test Set: Actual vs Predicted', fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(self.y_test, self.test_pred))
        mae = mean_absolute_error(self.y_test, self.test_pred)
        r2 = r2_score(self.y_test, self.test_pred)
        
        # Add metrics box
        metrics_text = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.4f}'
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.8), fontsize=11, fontweight='bold')
        
        # 2. Error over time
        ax2 = axes[0, 1]
        errors = self.test_pred - self.y_test
        colors = ['red' if e > 0 else 'blue' for e in errors]
        ax2.bar(range(len(errors)), errors, color=colors, alpha=0.6)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Test Sample Index', fontweight='bold')
        ax2.set_ylabel('Prediction Error (Predicted - Actual)', fontweight='bold')
        ax2.set_title('Prediction Errors Over Time\n(Red=Over, Blue=Under)', 
                     fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Cumulative error
        ax3 = axes[1, 0]
        cumulative_error = np.cumsum(np.abs(errors))
        ax3.plot(cumulative_error, linewidth=2.5, color='#A23B72', marker='o')
        ax3.set_xlabel('Test Sample Index', fontweight='bold')
        ax3.set_ylabel('Cumulative Absolute Error', fontweight='bold')
        ax3.set_title('Cumulative Error Accumulation', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.fill_between(range(len(cumulative_error)), cumulative_error, 
                        alpha=0.3, color='#A23B72')
        
        # 4. Prediction confidence (using train residuals as proxy)
        ax4 = axes[1, 1]
        train_errors = self.train_pred - self.y_train
        std_err = np.std(train_errors)
        
        ax4.plot(self.test_dates, self.y_test, 'o', label='Actual', 
                markersize=10, color='#2E86AB')
        ax4.plot(self.test_dates, self.test_pred, 's', label='Predicted', 
                markersize=10, color='#F18F01')
        
        # Add confidence bands (±1.96 std for 95% CI)
        upper_bound = self.test_pred + 1.96 * std_err
        lower_bound = np.maximum(self.test_pred - 1.96 * std_err, 0)
        
        ax4.fill_between(self.test_dates, lower_bound, upper_bound, 
                        alpha=0.3, color='#F18F01', label='95% Confidence')
        ax4.set_xlabel('Date', fontweight='bold')
        ax4.set_ylabel('Value', fontweight='bold')
        ax4.set_title('Predictions with Confidence Intervals', fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('xgboost_predictions.png', dpi=150, bbox_inches='tight')
        print("   ✅ Saved: xgboost_predictions.png")
        plt.close()
    
    def plot_residual_analysis(self):
        """Residual analysis plots."""
        print("3️⃣  Residual Analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('XGBoost Residual Analysis', fontsize=16, fontweight='bold')
        
        residuals = self.test_pred - self.y_test
        
        # 1. Residuals vs Predicted
        ax1 = axes[0, 0]
        ax1.scatter(self.test_pred, residuals, alpha=0.6, s=100, color='#2E86AB')
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Values', fontweight='bold')
        ax1.set_ylabel('Residuals', fontweight='bold')
        ax1.set_title('Residuals vs Predicted Values', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(self.test_pred, residuals, 1)
        p = np.poly1d(z)
        ax1.plot(sorted(self.test_pred), p(sorted(self.test_pred)), 
                "r-", linewidth=2, alpha=0.8, label=f'Trend: y={z[0]:.3f}x+{z[1]:.2f}')
        ax1.legend()
        
        # 2. Q-Q Plot
        ax2 = axes[0, 1]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normality Check)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Residuals histogram
        ax3 = axes[1, 0]
        ax3.hist(residuals, bins=15, edgecolor='black', alpha=0.7, color='#A23B72')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Residuals', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Residuals Distribution', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        ax3.text(0.02, 0.98, f'Mean: {mean_res:.2f}\nStd: {std_res:.2f}', 
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=11, fontweight='bold')
        
        # 4. Absolute residuals vs Predicted (heteroscedasticity check)
        ax4 = axes[1, 1]
        abs_residuals = np.abs(residuals)
        ax4.scatter(self.test_pred, abs_residuals, alpha=0.6, s=100, color='#F18F01')
        ax4.set_xlabel('Predicted Values', fontweight='bold')
        ax4.set_ylabel('Absolute Residuals', fontweight='bold')
        ax4.set_title('Scale-Location Plot\n(Homoscedasticity Check)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add trend
        z = np.polyfit(self.test_pred, abs_residuals, 1)
        p = np.poly1d(z)
        ax4.plot(sorted(self.test_pred), p(sorted(self.test_pred)), 
                "r-", linewidth=2, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig('xgboost_residuals.png', dpi=150, bbox_inches='tight')
        print("   ✅ Saved: xgboost_residuals.png")
        plt.close()
    
    def plot_error_distribution(self):
        """Error distribution analysis."""
        print("4️⃣  Error Distribution...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('XGBoost Error Distribution Analysis', 
                     fontsize=16, fontweight='bold')
        
        errors = self.test_pred - self.y_test
        abs_errors = np.abs(errors)
        pct_errors = (errors / self.y_test) * 100
        
        # 1. Error magnitude distribution
        ax1 = axes[0, 0]
        ax1.hist(errors, bins=20, edgecolor='black', alpha=0.7, color='#2E86AB')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax1.axvline(x=np.mean(errors), color='green', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
        ax1.set_xlabel('Error (Predicted - Actual)', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('Error Distribution', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Absolute error distribution
        ax2 = axes[0, 1]
        ax2.hist(abs_errors, bins=20, edgecolor='black', alpha=0.7, color='#F18F01')
        ax2.axvline(x=np.mean(abs_errors), color='red', linestyle='--', 
                   linewidth=2, label=f'MAE: {np.mean(abs_errors):.2f}')
        ax2.axvline(x=np.median(abs_errors), color='blue', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(abs_errors):.2f}')
        ax2.set_xlabel('Absolute Error', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('Absolute Error Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Percentage error distribution
        ax3 = axes[1, 0]
        ax3.hist(pct_errors, bins=20, edgecolor='black', alpha=0.7, color='#A23B72')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax3.axvline(x=np.mean(pct_errors), color='green', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(pct_errors):.1f}%')
        ax3.set_xlabel('Percentage Error (%)', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Percentage Error Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Error statistics box plot
        ax4 = axes[1, 1]
        box_data = [errors, abs_errors]
        bp = ax4.boxplot(box_data, labels=['Errors', 'Abs Errors'], 
                        patch_artist=True, widths=0.6)
        
        # Color the boxes
        colors = ['#2E86AB', '#F18F01']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax4.set_ylabel('Error Magnitude', fontweight='bold')
        ax4.set_title('Error Statistics (Box Plot)', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('xgboost_error_distribution.png', dpi=150, bbox_inches='tight')
        print("   ✅ Saved: xgboost_error_distribution.png")
        plt.close()
    
    def plot_learning_curves(self):
        """Plot learning curves if eval set was used."""
        print("5️⃣  Learning Curves...")
        
        # Note: This requires training with eval_set
        # For now, create a placeholder explaining this
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.text(0.5, 0.5, 
                'Learning Curves\n\n'
                'To enable learning curves, retrain XGBoost with:\n\n'
                'model.fit(X_train, y_train,\n'
                '          eval_set=[(X_train, y_train), (X_val, y_val)],\n'
                '          early_stopping_rounds=50,\n'
                '          verbose=False)\n\n'
                'This will track training and validation loss over iterations.',
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Learning Curves (Requires Retraining)', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('xgboost_learning_curves.png', dpi=150, bbox_inches='tight')
        print("   ✅ Saved: xgboost_learning_curves.png")
        plt.close()
    
    def plot_feature_correlations(self):
        """Plot top feature correlations."""
        print("6️⃣  Feature Correlations...")
        
        # Get top 15 features by importance
        importance = self.model.feature_importances_
        top_indices = np.argsort(importance)[-15:]
        top_features = [self.feature_names[i] for i in top_indices]
        
        # Create correlation matrix
        X_test_df = pd.DataFrame(self.X_test, columns=self.feature_names)
        corr_matrix = X_test_df[top_features].corr()
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax, vmin=-1, vmax=1)
        
        ax.set_title('Top 15 Features: Correlation Matrix', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('xgboost_feature_correlations.png', dpi=150, bbox_inches='tight')
        print("   ✅ Saved: xgboost_feature_correlations.png")
        plt.close()
    
    def plot_actual_vs_predicted(self):
        """Scatter plot of actual vs predicted."""
        print("7️⃣  Actual vs Predicted Scatter...")
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Scatter plot
        ax.scatter(self.y_test, self.test_pred, alpha=0.6, s=150, 
                  color='#2E86AB', edgecolors='black', linewidths=1)
        
        # Perfect prediction line
        min_val = min(self.y_test.min(), self.test_pred.min())
        max_val = max(self.y_test.max(), self.test_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', linewidth=2.5, label='Perfect Prediction', alpha=0.8)
        
        # Calculate R²
        r2 = r2_score(self.y_test, self.test_pred)
        
        ax.set_xlabel('Actual Values', fontweight='bold', fontsize=14)
        ax.set_ylabel('Predicted Values', fontweight='bold', fontsize=14)
        ax.set_title(f'Actual vs Predicted Values\nR² = {r2:.4f}', 
                    fontweight='bold', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add R² text box
        textstr = f'R² = {r2:.4f}\nRMSE = {np.sqrt(mean_squared_error(self.y_test, self.test_pred)):.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
               verticalalignment='top', bbox=props, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('xgboost_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
        print("   ✅ Saved: xgboost_actual_vs_predicted.png")
        plt.close()
    
    def plot_temporal_errors(self):
        """Time-based error analysis."""
        print("8️⃣  Temporal Error Analysis...")
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('Temporal Error Analysis', fontsize=16, fontweight='bold')
        
        errors = self.test_pred - self.y_test
        
        # 1. Errors over time with bands
        ax1 = axes[0]
        ax1.plot(self.test_dates, errors, 'o-', linewidth=2.5, markersize=8,
                color='#2E86AB', label='Prediction Error')
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax1.axhline(y=np.mean(errors), color='green', linestyle='--', 
                   linewidth=2, label=f'Mean Error: {np.mean(errors):.2f}')
        
        # Add ±1 std bands
        std_err = np.std(errors)
        ax1.fill_between(self.test_dates, 
                        np.mean(errors) - std_err, 
                        np.mean(errors) + std_err,
                        alpha=0.3, color='green', label='±1 Std Dev')
        
        ax1.set_xlabel('Date', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Prediction Error', fontweight='bold', fontsize=12)
        ax1.set_title('Prediction Errors Over Time', fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Percentage errors over time
        ax2 = axes[1]
        pct_errors = (errors / self.y_test) * 100
        
        colors = ['red' if e > 0 else 'blue' for e in pct_errors]
        ax2.bar(self.test_dates, pct_errors, color=colors, alpha=0.6, width=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        ax2.set_xlabel('Date', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Percentage Error (%)', fontweight='bold', fontsize=12)
        ax2.set_title('Percentage Errors Over Time (Red=Over-predict, Blue=Under-predict)', 
                     fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('xgboost_temporal_errors.png', dpi=150, bbox_inches='tight')
        print("   ✅ Saved: xgboost_temporal_errors.png")
        plt.close()


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("🎯 XGBOOST COMPREHENSIVE ANALYSIS")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data...")
    data = load_sample_data_from_db()
    
    # Explore
    explorer = DataExplorer(data)
    report = explorer.explore()
    
    # Engineer features
    engineer = FeatureEngineer(data)
    features_df = engineer.create_features()
    
    # Prepare data
    benchmark = ModelBenchmark(data, test_size=30)
    X_train, X_test, y_train, y_test, feature_cols, test_dates = \
        benchmark.prepare_data(features_df)
    
    # Train XGBoost
    print("\n" + "="*70)
    print("🚀 TRAINING XGBOOST")
    print("="*70 + "\n")
    
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training model...")
    xgb_model.fit(X_train, y_train, verbose=False)
    print("✅ Training complete!\n")
    
    # Evaluate
    y_pred = xgb_model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("📊 PERFORMANCE METRICS:")
    print(f"   • RMSE: {rmse:.2f}")
    print(f"   • MAE:  {mae:.2f}")
    print(f"   • R²:   {r2:.4f}")
    
    # Create analyzer
    analyzer = XGBoostAnalyzer(
        model=xgb_model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_cols,
        test_dates=test_dates,
        data_characteristics=report
    )
    
    # Generate all plots
    analyzer.create_all_plots()
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print("\n📊 Generated Files:")
    print("   1. xgboost_feature_importance.png")
    print("   2. xgboost_predictions.png")
    print("   3. xgboost_residuals.png")
    print("   4. xgboost_error_distribution.png")
    print("   5. xgboost_learning_curves.png")
    print("   6. xgboost_feature_correlations.png")
    print("   7. xgboost_actual_vs_predicted.png")
    print("   8. xgboost_temporal_errors.png")
    print("\n🎯 Next: Open the PNG files to view detailed analysis!\n")


if __name__ == "__main__":
    main()



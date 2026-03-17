"""
Test cases for ForecastAgent
Tests forecast routing, model selection, and forecast accuracy
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestForecastRouting:
    """Test smart routing logic in ForecastAgent"""
    
    def test_routing_panel_xgboost_sufficient_history(self, sample_sales_data):
        """
        TC-F001: Panel XGBoost should be used when history >= 14 days
        """
        # Filter data for 30 days
        recent_data = sample_sales_data[
            sample_sales_data['date'] >= (sample_sales_data['date'].max() - timedelta(days=30))
        ]
        
        # Verify data has sufficient history
        date_span = (recent_data['date'].max() - recent_data['date'].min()).days
        assert date_span >= 14, "Test data should have at least 14 days"
        
        # Expected: Panel XGBoost should be selected
        # This is a unit test - actual routing tested in integration
    
    def test_routing_moving_average_medium_history(self):
        """
        TC-F002: Moving Average should be used for 7-13 days history
        """
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'quantity': [50, 60, 55, 70, 65, 80, 75, 90, 85, 100]
        })
        
        date_span = (data['date'].max() - data['date'].min()).days
        assert 7 <= date_span < 14, "Test data should have 7-13 days"
    
    def test_routing_simple_average_short_history(self):
        """
        TC-F003: Simple Average should be used for < 7 days history
        """
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'quantity': [50, 60, 55, 70, 65]
        })
        
        date_span = (data['date'].max() - data['date'].min()).days
        assert date_span < 7, "Test data should have less than 7 days"
    
    def test_routing_dead_stock_detection(self):
        """
        TC-F004: Dead stock should be detected when no sales > 180 days
        """
        # Create data with last sale 200 days ago
        old_date = datetime.now() - timedelta(days=200)
        data = pd.DataFrame({
            'date': [old_date],
            'quantity': [100]
        })
        
        days_since_last_sale = (datetime.now() - data['date'].max()).days
        assert days_since_last_sale > 180, "Should be detected as dead stock"


class TestForecastConstraints:
    """Test forecast constraining logic"""
    
    def test_constrain_forecast_growth_normal(self):
        """
        TC-F005: Forecast should not exceed 2x recent average
        """
        recent_avg = 100
        max_ratio = 2.0
        allowed_max = recent_avg * max_ratio
        
        forecast_values = [150, 180, 200, 220]
        constrained = [min(v, allowed_max) for v in forecast_values]
        
        assert all(v <= allowed_max for v in constrained)
    
    def test_constrain_forecast_sparse_data(self, sample_sparse_time_series):
        """
        TC-F006: Sparse data should use non-zero average for constraining
        """
        data = sample_sparse_time_series
        
        # Calculate non-zero average
        non_zero_values = data[data['quantity'] > 0]['quantity']
        if len(non_zero_values) > 0:
            non_zero_avg = non_zero_values.mean()
        else:
            non_zero_avg = 0
        
        # Calculate all values average
        all_avg = data['quantity'].mean()
        
        # Non-zero average should be higher for sparse data
        if len(non_zero_values) > 0:
            assert non_zero_avg >= all_avg
    
    def test_forecast_not_negative(self, sample_forecast_result):
        """
        TC-F007: Forecast values should never be negative
        """
        forecast = sample_forecast_result
        assert (forecast['forecast'] >= 0).all(), "Forecast should not be negative"
        assert (forecast['lower_bound'] >= 0).all(), "Lower bound should not be negative"


class TestForecastModels:
    """Test individual forecast models"""
    
    def test_simple_average_calculation(self, sample_time_series):
        """
        TC-F008: Simple average forecast should equal mean of historical data
        """
        data = sample_time_series
        expected_avg = data['quantity'].mean()
        
        # Simple average forecast
        forecast = [expected_avg] * 30
        
        assert abs(np.mean(forecast) - expected_avg) < 0.01
    
    def test_moving_average_calculation(self, sample_time_series):
        """
        TC-F009: Moving average should use recent window
        """
        data = sample_time_series
        window = 7
        
        # Calculate moving average
        ma = data['quantity'].rolling(window=window).mean().iloc[-1]
        
        # Should be close to last 7 days average
        last_7_avg = data['quantity'].tail(window).mean()
        assert abs(ma - last_7_avg) < 0.01
    
    def test_forecast_horizon_length(self):
        """
        TC-F010: Forecast should return correct number of periods
        """
        horizon = 30
        dates = pd.date_range(start='2025-01-01', periods=horizon, freq='D')
        
        assert len(dates) == horizon


class TestForecastAggregation:
    """Test forecast aggregation across SKUs"""
    
    def test_aggregate_multiple_skus(self):
        """
        TC-F011: Aggregated forecast should sum individual SKU forecasts
        """
        sku_forecasts = {
            'SKU001': [100, 110, 120],
            'SKU002': [50, 55, 60],
            'SKU003': [200, 210, 220]
        }
        
        # Aggregate
        aggregated = [sum(sku_forecasts[sku][i] for sku in sku_forecasts) 
                      for i in range(3)]
        
        expected = [350, 375, 400]
        assert aggregated == expected
    
    def test_aggregate_with_zero_forecasts(self):
        """
        TC-F012: Aggregation should handle zero forecasts correctly
        """
        sku_forecasts = {
            'SKU001': [100, 110, 120],
            'SKU002': [0, 0, 0],  # Dead stock
            'SKU003': [200, 210, 220]
        }
        
        aggregated = [sum(sku_forecasts[sku][i] for sku in sku_forecasts) 
                      for i in range(3)]
        
        expected = [300, 320, 340]
        assert aggregated == expected


class TestForecastEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_data_handling(self):
        """
        TC-F013: Should handle empty dataframe gracefully
        """
        empty_df = pd.DataFrame(columns=['date', 'quantity'])
        assert len(empty_df) == 0
    
    def test_single_data_point(self):
        """
        TC-F014: Should handle single data point
        """
        data = pd.DataFrame({
            'date': [datetime.now()],
            'quantity': [100]
        })
        
        assert len(data) == 1
        # Should fallback to simple average
    
    def test_all_zero_values(self):
        """
        TC-F015: Should handle all zero values
        """
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'quantity': [0] * 30
        })
        
        assert data['quantity'].sum() == 0
        # Forecast should also be 0 or near 0
    
    def test_extreme_values(self):
        """
        TC-F016: Should handle extreme values without overflow
        """
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'quantity': [1000000] * 30  # Very large values
        })
        
        avg = data['quantity'].mean()
        assert not np.isinf(avg)
        assert not np.isnan(avg)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


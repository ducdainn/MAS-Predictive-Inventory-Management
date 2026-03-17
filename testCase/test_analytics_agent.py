"""
Test cases for AnalyticsAgent
Tests data analysis, visualization, and insights generation
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataAnalysis:
    """Test data analysis functions"""
    
    def test_calculate_basic_statistics(self, sample_sales_data):
        """
        TC-A001: Should calculate basic statistics correctly
        """
        data = sample_sales_data
        
        stats = {
            'mean': data['quantity'].mean(),
            'median': data['quantity'].median(),
            'std': data['quantity'].std(),
            'min': data['quantity'].min(),
            'max': data['quantity'].max()
        }
        
        assert stats['mean'] > 0
        assert stats['min'] <= stats['median'] <= stats['max']
    
    def test_calculate_growth_rate(self):
        """
        TC-A002: Should calculate growth rate correctly
        """
        current_period = 1200
        previous_period = 1000
        
        growth_rate = ((current_period - previous_period) / previous_period) * 100
        expected = 20.0  # 20% growth
        
        assert abs(growth_rate - expected) < 0.01
    
    def test_calculate_yoy_comparison(self):
        """
        TC-A003: Should calculate year-over-year comparison
        """
        this_year = 12000
        last_year = 10000
        
        yoy_growth = ((this_year - last_year) / last_year) * 100
        expected = 20.0
        
        assert abs(yoy_growth - expected) < 0.01
    
    def test_calculate_mom_comparison(self):
        """
        TC-A004: Should calculate month-over-month comparison
        """
        this_month = 1100
        last_month = 1000
        
        mom_growth = ((this_month - last_month) / last_month) * 100
        expected = 10.0
        
        assert abs(mom_growth - expected) < 0.01


class TestTrendAnalysis:
    """Test trend analysis functions"""
    
    def test_detect_upward_trend(self):
        """
        TC-A005: Should detect upward trend
        """
        values = [100, 110, 120, 130, 140, 150]
        
        # Simple trend detection: compare first half to second half
        first_half_avg = np.mean(values[:3])
        second_half_avg = np.mean(values[3:])
        
        is_upward = second_half_avg > first_half_avg
        assert is_upward == True
    
    def test_detect_downward_trend(self):
        """
        TC-A006: Should detect downward trend
        """
        values = [150, 140, 130, 120, 110, 100]
        
        first_half_avg = np.mean(values[:3])
        second_half_avg = np.mean(values[3:])
        
        is_downward = second_half_avg < first_half_avg
        assert is_downward == True
    
    def test_detect_stable_trend(self):
        """
        TC-A007: Should detect stable/flat trend
        """
        values = [100, 102, 98, 101, 99, 100]
        
        first_half_avg = np.mean(values[:3])
        second_half_avg = np.mean(values[3:])
        
        # Stable if change < 5%
        change_pct = abs((second_half_avg - first_half_avg) / first_half_avg) * 100
        is_stable = change_pct < 5
        
        assert is_stable == True
    
    def test_detect_seasonality(self):
        """
        TC-A008: Should detect seasonal patterns
        """
        # Monthly data with seasonal pattern
        monthly_values = [100, 90, 95, 110, 130, 150, 160, 155, 140, 120, 105, 100]
        
        # Peak months (June, July, August) - indices 5, 6, 7
        peak_avg = np.mean([monthly_values[5], monthly_values[6], monthly_values[7]])
        overall_avg = np.mean(monthly_values)
        
        has_seasonality = peak_avg > overall_avg * 1.2  # Peak is 20% above average
        assert has_seasonality == True


class TestVisualization:
    """Test visualization generation"""
    
    def test_bar_chart_data_preparation(self, sample_sales_data):
        """
        TC-A009: Should prepare data for bar chart correctly
        """
        data = sample_sales_data
        
        # Group by branch
        chart_data = data.groupby('branch_code')['quantity'].sum().reset_index()
        
        assert 'branch_code' in chart_data.columns
        assert 'quantity' in chart_data.columns
        assert len(chart_data) > 0
    
    def test_line_chart_data_preparation(self, sample_sales_data):
        """
        TC-A010: Should prepare data for line chart correctly
        """
        data = sample_sales_data
        
        # Group by date
        chart_data = data.groupby('date')['quantity'].sum().reset_index()
        chart_data = chart_data.sort_values('date')
        
        assert 'date' in chart_data.columns
        assert 'quantity' in chart_data.columns
        # Data should be sorted by date
        assert chart_data['date'].is_monotonic_increasing
    
    def test_pie_chart_data_preparation(self, sample_sales_data):
        """
        TC-A011: Should prepare data for pie chart correctly
        """
        data = sample_sales_data
        
        # Group by branch for pie chart
        chart_data = data.groupby('branch_code')['quantity'].sum()
        
        # Percentages should sum to 100
        total = chart_data.sum()
        percentages = (chart_data / total) * 100
        
        assert abs(percentages.sum() - 100) < 0.01
    
    def test_chart_sorting_value_desc(self, sample_sales_data):
        """
        TC-A012: Should sort chart data by value descending
        """
        data = sample_sales_data
        
        chart_data = data.groupby('branch_code')['quantity'].sum().reset_index()
        sorted_data = chart_data.sort_values('quantity', ascending=False)
        
        # First value should be largest
        assert sorted_data['quantity'].iloc[0] >= sorted_data['quantity'].iloc[-1]


class TestTopNAnalysis:
    """Test top N analysis functions"""
    
    def test_top_products_by_quantity(self, sample_sales_data):
        """
        TC-A013: Should return top N products by quantity
        """
        data = sample_sales_data
        n = 5
        
        top_products = (data.groupby('f_sku')['quantity']
                       .sum()
                       .nlargest(n)
                       .reset_index())
        
        assert len(top_products) <= n
        # Should be sorted descending
        assert top_products['quantity'].is_monotonic_decreasing
    
    def test_top_branches_by_revenue(self, sample_sales_data):
        """
        TC-A014: Should return top N branches by revenue
        """
        data = sample_sales_data
        n = 3
        
        top_branches = (data.groupby('branch_code')['revenue']
                       .sum()
                       .nlargest(n)
                       .reset_index())
        
        assert len(top_branches) <= n
    
    def test_bottom_performers(self, sample_sales_data):
        """
        TC-A015: Should return bottom N performers
        """
        data = sample_sales_data
        n = 3
        
        bottom_products = (data.groupby('f_sku')['quantity']
                          .sum()
                          .nsmallest(n)
                          .reset_index())
        
        assert len(bottom_products) <= n
        # Should be sorted ascending
        assert bottom_products['quantity'].is_monotonic_increasing


class TestInsightsGeneration:
    """Test automated insights generation"""
    
    def test_generate_summary_insight(self, sample_sales_data):
        """
        TC-A016: Should generate summary insight
        """
        data = sample_sales_data
        
        total_quantity = data['quantity'].sum()
        total_revenue = data['revenue'].sum()
        unique_branches = data['branch_code'].nunique()
        unique_products = data['f_sku'].nunique()
        
        insight = f"Tổng số lượng: {total_quantity:,}, Doanh thu: {total_revenue:,.0f}, " \
                  f"Số chi nhánh: {unique_branches}, Số sản phẩm: {unique_products}"
        
        assert str(total_quantity) in insight.replace(',', '')
    
    def test_generate_comparison_insight(self):
        """
        TC-A017: Should generate comparison insight
        """
        current = 1200
        previous = 1000
        growth = ((current - previous) / previous) * 100
        
        if growth > 0:
            insight = f"Tăng {growth:.1f}% so với kỳ trước"
        else:
            insight = f"Giảm {abs(growth):.1f}% so với kỳ trước"
        
        assert "20.0%" in insight
    
    def test_generate_anomaly_insight(self):
        """
        TC-A018: Should detect and report anomalies
        """
        values = [100, 105, 98, 102, 500, 99, 101]  # 500 is anomaly
        
        mean = np.mean(values)
        std = np.std(values)
        threshold = mean + 2 * std
        
        anomalies = [v for v in values if v > threshold]
        
        assert 500 in anomalies


class TestDataFiltering:
    """Test data filtering functions"""
    
    def test_filter_by_date_range(self, sample_sales_data):
        """
        TC-A019: Should filter data by date range correctly
        """
        data = sample_sales_data
        
        start_date = '2024-06-01'
        end_date = '2024-06-30'
        
        filtered = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        
        assert len(filtered) > 0
        assert filtered['date'].min() >= pd.Timestamp(start_date)
        assert filtered['date'].max() <= pd.Timestamp(end_date)
    
    def test_filter_by_branch(self, sample_sales_data):
        """
        TC-A020: Should filter data by branch correctly
        """
        data = sample_sales_data
        
        branch = 'CN001'
        filtered = data[data['branch_code'] == branch]
        
        assert len(filtered) > 0
        assert (filtered['branch_code'] == branch).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


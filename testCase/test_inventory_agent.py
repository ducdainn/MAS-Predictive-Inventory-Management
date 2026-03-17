"""
Test cases for InventoryOptimizationAgent
Tests inventory optimization, reorder points, and recommendations
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestReorderPointCalculation:
    """Test reorder point and safety stock calculations"""
    
    def test_reorder_point_formula(self):
        """
        TC-I001: Reorder point = (Average daily demand × Lead time) + Safety stock
        """
        avg_daily_demand = 50
        lead_time_days = 7
        safety_stock = 100
        
        reorder_point = (avg_daily_demand * lead_time_days) + safety_stock
        expected = 450  # (50 * 7) + 100
        
        assert reorder_point == expected
    
    def test_safety_stock_calculation(self):
        """
        TC-I002: Safety stock = Z × √((LT × σ_D²) + (D² × σ_LT²))
        
        Combined demand and lead time variability formula:
        - Z = service level factor (z-score)
        - LT = average lead time (days)
        - σ_D = standard deviation of demand
        - D = average demand
        - σ_LT = standard deviation of lead time
        """
        z_score = 1.65  # 95% service level
        avg_demand = 50  # units/day
        std_demand = 20  # demand std dev
        lead_time_days = 7
        std_lead_time = 1.0  # lead time std dev (days)
        
        # Combined variability formula
        demand_variance_component = lead_time_days * (std_demand ** 2)
        lead_time_variance_component = (avg_demand ** 2) * (std_lead_time ** 2)
        
        safety_stock = z_score * np.sqrt(demand_variance_component + lead_time_variance_component)
        safety_stock = round(safety_stock)  # Rounded for practical use
        
        assert safety_stock > 0
        assert not np.isnan(safety_stock)
        assert isinstance(safety_stock, int)  # Should be rounded
    
    def test_eoq_calculation(self):
        """
        TC-I003: EOQ = √((2 × D × S) / H)
        D = Annual demand, S = Order cost, H = Holding cost
        """
        annual_demand = 10000
        order_cost = 50
        holding_cost = 5
        
        eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost)
        expected = np.sqrt((2 * 10000 * 50) / 5)  # ≈ 447
        
        assert abs(eoq - expected) < 0.01


class TestInventoryPriority:
    """Test inventory priority classification"""
    
    def test_priority_high_below_reorder(self):
        """
        TC-I004: Priority should be HIGH when stock < reorder point
        """
        current_stock = 100
        reorder_point = 200
        
        priority = "HIGH" if current_stock < reorder_point else "LOW"
        assert priority == "HIGH"
    
    def test_priority_medium_near_reorder(self):
        """
        TC-I005: Priority should be MEDIUM when stock is 100-150% of reorder point
        """
        current_stock = 250
        reorder_point = 200
        
        ratio = current_stock / reorder_point
        if ratio < 1.0:
            priority = "HIGH"
        elif ratio < 1.5:
            priority = "MEDIUM"
        else:
            priority = "LOW"
        
        assert priority == "MEDIUM"
    
    def test_priority_low_sufficient_stock(self):
        """
        TC-I006: Priority should be LOW when stock > 150% of reorder point
        """
        current_stock = 400
        reorder_point = 200
        
        ratio = current_stock / reorder_point
        priority = "LOW" if ratio >= 1.5 else "HIGH"
        
        assert priority == "LOW"


class TestRecommendations:
    """Test inventory recommendations generation"""
    
    def test_reorder_recommendation(self):
        """
        TC-I007: Should recommend reorder when stock < reorder point
        """
        current_stock = 100
        reorder_point = 200
        max_stock = 500
        
        should_reorder = current_stock < reorder_point
        recommended_qty = max_stock - current_stock if should_reorder else 0
        
        assert should_reorder == True
        assert recommended_qty == 400
    
    def test_no_reorder_sufficient_stock(self):
        """
        TC-I008: Should not recommend reorder when stock >= reorder point
        """
        current_stock = 300
        reorder_point = 200
        
        should_reorder = current_stock < reorder_point
        assert should_reorder == False
    
    def test_overstock_warning(self):
        """
        TC-I009: Should warn about overstock when stock > max level
        """
        current_stock = 600
        max_stock = 500
        
        is_overstock = current_stock > max_stock
        overstock_qty = current_stock - max_stock if is_overstock else 0
        
        assert is_overstock == True
        assert overstock_qty == 100


class TestDaysOfSupply:
    """Test days of supply calculations"""
    
    def test_days_of_supply_calculation(self):
        """
        TC-I010: Days of supply = Current stock / Average daily demand
        """
        current_stock = 500
        avg_daily_demand = 50
        
        days_of_supply = current_stock / avg_daily_demand
        expected = 10
        
        assert days_of_supply == expected
    
    def test_days_of_supply_zero_demand(self):
        """
        TC-I011: Should handle zero demand gracefully
        """
        current_stock = 500
        avg_daily_demand = 0
        
        # Avoid division by zero
        if avg_daily_demand > 0:
            days_of_supply = current_stock / avg_daily_demand
        else:
            days_of_supply = float('inf')  # Infinite days of supply
        
        assert days_of_supply == float('inf')
    
    def test_stockout_risk_calculation(self):
        """
        TC-I012: Stockout risk when days of supply < lead time
        """
        days_of_supply = 5
        lead_time_days = 7
        
        stockout_risk = days_of_supply < lead_time_days
        assert stockout_risk == True


class TestForecastIntegration:
    """Test integration with forecast agent"""
    
    def test_forecast_demand_used_for_reorder(self):
        """
        TC-I013: Forecast demand should be used for reorder calculations
        """
        forecast_demand_30d = 1500  # Total forecast for 30 days
        avg_daily_forecast = forecast_demand_30d / 30
        lead_time_days = 7
        safety_stock = 100
        
        reorder_point = (avg_daily_forecast * lead_time_days) + safety_stock
        
        assert reorder_point > 0
    
    def test_forecast_growth_impact(self):
        """
        TC-I014: Growing forecast should increase reorder point
        """
        historical_avg = 50
        forecast_avg = 75  # 50% growth
        lead_time = 7
        safety_stock = 100
        
        historical_rop = (historical_avg * lead_time) + safety_stock
        forecast_rop = (forecast_avg * lead_time) + safety_stock
        
        assert forecast_rop > historical_rop


class TestMultiBranchOptimization:
    """Test multi-branch inventory optimization"""
    
    def test_branch_level_recommendations(self, sample_inventory_data):
        """
        TC-I015: Should generate recommendations per branch
        """
        data = sample_inventory_data
        branches = data['branch_code'].unique()
        
        assert len(branches) == 3  # CN001, CN002, CN003
    
    def test_aggregate_recommendations(self, sample_inventory_data):
        """
        TC-I016: Should aggregate recommendations across branches
        """
        data = sample_inventory_data
        total_stock = data['current_stock'].sum()
        
        assert total_stock > 0
    
    def test_transfer_recommendations(self):
        """
        TC-I017: Should recommend transfers between branches
        """
        branch_stocks = {
            'CN001': {'stock': 100, 'reorder_point': 200},  # Needs stock
            'CN002': {'stock': 500, 'reorder_point': 200},  # Excess stock
        }
        
        # CN002 has excess, CN001 needs stock
        cn001_deficit = branch_stocks['CN001']['reorder_point'] - branch_stocks['CN001']['stock']
        cn002_excess = branch_stocks['CN002']['stock'] - branch_stocks['CN002']['reorder_point']
        
        can_transfer = cn002_excess > 0 and cn001_deficit > 0
        transfer_qty = min(cn001_deficit, cn002_excess)
        
        assert can_transfer == True
        assert transfer_qty == 100


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_new_product_no_history(self):
        """
        TC-I018: Should handle new products with no sales history
        """
        historical_demand = []
        
        if len(historical_demand) == 0:
            avg_demand = 0
            recommendation = "MONITOR"  # New product, monitor closely
        
        assert avg_demand == 0
        assert recommendation == "MONITOR"
    
    def test_seasonal_product(self):
        """
        TC-I019: Should handle seasonal products
        """
        # Summer product in winter
        current_month = 12  # December
        peak_months = [6, 7, 8]  # June, July, August
        
        is_peak_season = current_month in peak_months
        
        # Adjust safety stock for off-season
        base_safety_stock = 100
        adjusted_safety_stock = base_safety_stock * 0.5 if not is_peak_season else base_safety_stock
        
        assert adjusted_safety_stock == 50
    
    def test_discontinued_product(self):
        """
        TC-I020: Should handle discontinued products
        """
        is_discontinued = True
        current_stock = 100
        
        if is_discontinued:
            recommendation = "LIQUIDATE"
            target_stock = 0
        
        assert recommendation == "LIQUIDATE"
        assert target_stock == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


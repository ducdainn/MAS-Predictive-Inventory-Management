"""
Batch Forecast Tool for Agent
===============================

Tool cho agent để dự báo nhu cầu cho nhiều products cùng lúc.
Tối ưu hóa bằng cách sử dụng XGBoost model và batch processing.

"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agent.xgboost_model_loader import XGBoostModelLoader
from agent.manager.database_manager import DatabaseManager


class BatchForecastTool:
    """
    Tool để dự báo nhu cầu cho nhiều products/branches cùng lúc.
    
    Features:
    - Batch processing để tăng tốc
    - Sử dụng pre-trained XGBoost model
    - Hỗ trợ forecast theo product, branch, hoặc cả hai
    - Trả về kết quả structured để agent/tool khác xử lý tiếp
    """
    
    def __init__(self, 
                 db_manager: Optional[DatabaseManager] = None,
                 model_dir: str = "agent/models"):
        """
        Initialize tool.
        
        Args:
            db_manager: Database manager instance (tạo mới nếu None)
            model_dir: Directory chứa pre-trained models
        """
        self.db = db_manager if db_manager else DatabaseManager()
        
        # Load XGBoost model
        self.model_loader = XGBoostModelLoader(models_dir=model_dir)
        self.model_loaded = self.model_loader.load_latest_model()
        
        if self.model_loaded:
            print(f"✅ BatchForecastTool initialized with XGBoost model")
        else:
            print(f"⚠️  XGBoost model not available. Using fallback methods.")
    
    def forecast_products(self,
                         product_list: List[Dict[str, Any]],
                         horizon_days: int = 30,
                         branch_filter: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Dự báo nhu cầu cho list các products.
        
        Args:
            product_list: List of dicts với format:
                [
                    {"product_code": "10.L1.3060.4566", "branch_code": 1},
                    {"product_code": "10.L1.3060.4567", "branch_code": 2},
                    ...
                ]
                Hoặc chỉ cần product_code nếu muốn aggregate tất cả branches:
                [
                    {"product_code": "10.L1.3060.4566"},
                    ...
                ]
            
            horizon_days: Số ngày dự báo (mặc định 30)
            branch_filter: List branch_codes để filter (optional)
        
        Returns:
            Dict với format:
                {
                    "success": bool,
                    "total_items": int,
                    "forecasts": [
                        {
                            "product_code": str,
                            "branch_code": int (optional),
                            "forecast_values": [float, ...],  # List giá trị dự báo
                            "forecast_dates": [str, ...],     # List ngày dự báo
                            "total_forecast": float,           # Tổng forecast
                            "avg_daily_forecast": float,       # Trung bình mỗi ngày
                            "confidence_lower": [float, ...],  # Lower bound (nếu có)
                            "confidence_upper": [float, ...],  # Upper bound (nếu có)
                            "model_used": str,                 # Model đã dùng
                            "historical_avg": float            # Trung bình lịch sử
                        },
                        ...
                    ],
                    "summary": {
                        "total_forecast_all": float,
                        "avg_forecast_per_item": float,
                        "processing_time_seconds": float
                    }
                }
        """
        start_time = datetime.now()
        
        print(f"\n🔮 BatchForecastTool: Processing {len(product_list)} items...")
        
        results = []
        failed_items = []
        
        for idx, item in enumerate(product_list, 1):
            product_code = item.get('product_code')
            branch_code = item.get('branch_code')  # Optional
            
            if not product_code:
                failed_items.append({"item": item, "reason": "Missing product_code"})
                continue
            
            try:
                # Forecast cho item này
                forecast_result = self._forecast_single_item(
                    product_code=product_code,
                    branch_code=branch_code,
                    horizon_days=horizon_days,
                    branch_filter=branch_filter
                )
                
                if forecast_result:
                    results.append(forecast_result)
                    
                    # Progress indicator
                    if idx % 10 == 0:
                        print(f"   Progress: {idx}/{len(product_list)} items processed...")
                else:
                    failed_items.append({
                        "item": item, 
                        "reason": "No historical data or forecast failed"
                    })
                    
            except Exception as e:
                failed_items.append({
                    "item": item,
                    "reason": f"Error: {str(e)}"
                })
        
        # Calculate summary
        total_forecast = sum(r['total_forecast'] for r in results)
        avg_forecast = total_forecast / len(results) if results else 0
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"\n✅ BatchForecastTool: Completed!")
        print(f"   Successful: {len(results)}/{len(product_list)}")
        print(f"   Failed: {len(failed_items)}")
        print(f"   Total forecast: {total_forecast:,.0f}")
        print(f"   Processing time: {elapsed:.2f}s")
        
        return {
            "success": True,
            "total_items": len(product_list),
            "successful_forecasts": len(results),
            "failed_forecasts": len(failed_items),
            "forecasts": results,
            "failed_items": failed_items,
            "summary": {
                "total_forecast_all": total_forecast,
                "avg_forecast_per_item": avg_forecast,
                "processing_time_seconds": elapsed,
                "items_per_second": len(product_list) / elapsed if elapsed > 0 else 0
            }
        }
    
    def _forecast_single_item(self,
                             product_code: str,
                             branch_code: Optional[int],
                             horizon_days: int,
                             branch_filter: Optional[List[int]] = None) -> Optional[Dict]:
        """
        Dự báo cho 1 item duy nhất.
        
        Returns:
            Dict với forecast results hoặc None nếu thất bại
        """
        # Build SQL query để lấy historical data (90 days for better features)
        # Use system_date-aware date filter
        try:
            from agent.system_date import get_system_date, SYSTEM_DATE_AVAILABLE
            if SYSTEM_DATE_AVAILABLE:
                system_date = get_system_date()
                date_filter = f"date >= DATE '{system_date}' - INTERVAL '90 days' AND date <= DATE '{system_date}'"
            else:
                date_filter = "date >= CURRENT_DATE - INTERVAL '90 days' AND date <= CURRENT_DATE"
        except ImportError:
            date_filter = "date >= CURRENT_DATE - INTERVAL '90 days' AND date <= CURRENT_DATE"
        
        if branch_code:
            # Specific product-branch combination
            sql = f"""
            SELECT date, SUM(quantity) as total_qty
            FROM sales
            WHERE {date_filter}
                AND product_code = '{product_code}'
                AND branch_code = {branch_code}
            GROUP BY date
            ORDER BY date ASC
            """
            item_key = f"{product_code}_B{branch_code}"
        elif branch_filter:
            # Product across specific branches
            branch_list = ','.join(map(str, branch_filter))
            sql = f"""
            SELECT date, SUM(quantity) as total_qty
            FROM sales
            WHERE {date_filter}
                AND product_code = '{product_code}'
                AND branch_code IN ({branch_list})
            GROUP BY date
            ORDER BY date ASC
            """
            item_key = f"{product_code}_Branches{len(branch_filter)}"
        else:
            # Product across all branches
            sql = f"""
            SELECT date, SUM(quantity) as total_qty
            FROM sales
            WHERE {date_filter}
                AND product_code = '{product_code}'
            GROUP BY date
            ORDER BY date ASC
            """
            item_key = f"{product_code}_ALL"
        
        # Execute query
        try:
            df = self.db.execute_query(sql)
        except Exception as e:
            print(f"   ⚠️  Query failed for {item_key}: {e}")
            return None
        
        if df.empty or len(df) < 2:
            return None
        
        # Prepare time series
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df.columns = ['value']
        
        # Resample to daily (fill missing days with 0)
        df = df.resample('D').sum().fillna(0)
        
        # Calculate historical average
        historical_avg = df['value'].mean()
        
        if not self.model_loaded:
            print(f"   ⚠️  XGBoost model not loaded. Skipping {item_key}")
            return None
        
        if len(df) < 30:
            print(f"   ⚠️  Not enough historical data for {item_key} (need ≥30 days, have {len(df)})")
            return None
        
        # Ensure we have at least 60 days for good lag features (lag_30, rolling_30)
        if len(df) < 60:
            print(f"   ⚠️  Limited historical data for {item_key} (have {len(df)} days, recommend ≥90 days for best features)")
        
        # Generate forecast using XGBoost only
        try:
            forecast_df = self.model_loader.predict_with_confidence(
                df, 
                horizon=horizon_days,
                confidence_level=0.95
            )
            
            model_used = "XGBoost (Pre-trained)"
            has_confidence = True
            
        except Exception as e:
            print(f"   ⚠️  XGBoost forecast failed for {item_key}: {e}")
            return None
        
        # Extract results
        forecast_values = forecast_df['forecast'].tolist()
        forecast_dates = [d.strftime('%Y-%m-%d') for d in forecast_df.index]
        
        result = {
            "product_code": product_code,
            "branch_code": branch_code,
            "item_key": item_key,
            "forecast_values": forecast_values,
            "forecast_dates": forecast_dates,
            "total_forecast": sum(forecast_values),
            "avg_daily_forecast": np.mean(forecast_values),
            "model_used": model_used,
            "historical_avg": float(historical_avg),
            "historical_days": len(df)
        }
        
        # Add confidence intervals if available
        if has_confidence:
            result["confidence_lower"] = forecast_df['lower_bound'].tolist()
            result["confidence_upper"] = forecast_df['upper_bound'].tolist()
        
        return result
    
    def aggregate_forecasts(self, 
                           forecasts: List[Dict],
                           group_by: str = "product_code") -> Dict[str, Any]:
        """
        Aggregate forecasts theo product_code hoặc branch_code.
        
        Args:
            forecasts: List forecast results từ forecast_products()
            group_by: "product_code" hoặc "branch_code"
        
        Returns:
            Dict với aggregated results
        """
        if group_by not in ["product_code", "branch_code"]:
            raise ValueError("group_by must be 'product_code' or 'branch_code'")
        
        # Group forecasts
        grouped = {}
        for forecast in forecasts:
            key = forecast.get(group_by)
            if key is None:
                continue
            
            if key not in grouped:
                grouped[key] = {
                    group_by: key,
                    "total_forecast": 0,
                    "items_count": 0,
                    "avg_daily_forecast": 0,
                    "forecast_values_sum": None
                }
            
            grouped[key]["total_forecast"] += forecast["total_forecast"]
            grouped[key]["items_count"] += 1
            
            # Sum forecast values day by day
            if grouped[key]["forecast_values_sum"] is None:
                grouped[key]["forecast_values_sum"] = forecast["forecast_values"].copy()
            else:
                for i, val in enumerate(forecast["forecast_values"]):
                    grouped[key]["forecast_values_sum"][i] += val
        
        # Calculate averages
        for key, data in grouped.items():
            data["avg_daily_forecast"] = data["total_forecast"] / len(data["forecast_values_sum"])
        
        return {
            "group_by": group_by,
            "groups": list(grouped.values()),
            "total_groups": len(grouped)
        }


# ============================================================================
# HELPER FUNCTIONS FOR AGENT
# ============================================================================

def forecast_products_batch(product_list: List[Dict[str, Any]],
                           horizon_days: int = 30,
                           branch_filter: Optional[List[int]] = None,
                           db_manager: Optional[DatabaseManager] = None) -> Dict[str, Any]:
    """
    Helper function để agent gọi trực tiếp.
    """
    tool = BatchForecastTool(db_manager=db_manager)
    return tool.forecast_products(
        product_list=product_list,
        horizon_days=horizon_days,
        branch_filter=branch_filter
    )

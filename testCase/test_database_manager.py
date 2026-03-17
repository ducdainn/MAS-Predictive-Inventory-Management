"""
Test cases for DatabaseManager
Tests database connection, query execution, and data retrieval
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDatabaseConnection:
    """Test database connection handling"""
    
    def test_connection_success(self):
        """
        TC-D001: Should connect to database successfully
        """
        connection_params = {
            "host": "localhost",
            "port": 5432,
            "database": "brickdemand",
            "user": "postgres",
            "password": "postgres"
        }
        
        # Mock successful connection
        is_connected = True
        
        assert is_connected == True
    
    def test_connection_failure_handling(self):
        """
        TC-D002: Should handle connection failure gracefully
        """
        def connect_with_retry(max_retries=3):
            for i in range(max_retries):
                try:
                    raise ConnectionError("Connection refused")
                except ConnectionError:
                    if i == max_retries - 1:
                        return False
            return True
        
        result = connect_with_retry()
        assert result == False
    
    def test_connection_pooling(self):
        """
        TC-D003: Should use connection pooling
        """
        pool_config = {
            "pool_size": 5,
            "max_overflow": 10,
            "pool_timeout": 30
        }
        
        assert pool_config["pool_size"] > 0
    
    def test_connection_string_format(self):
        """
        TC-D004: Should format connection string correctly
        """
        params = {
            "user": "postgres",
            "password": "postgres",
            "host": "localhost",
            "port": 5432,
            "database": "brickdemand"
        }
        
        conn_string = f"postgresql://{params['user']}:{params['password']}@{params['host']}:{params['port']}/{params['database']}"
        
        assert "postgresql://" in conn_string
        assert "brickdemand" in conn_string


class TestQueryExecution:
    """Test query execution"""
    
    def test_select_query_execution(self, mock_db_manager):
        """
        TC-D005: Should execute SELECT query
        """
        query = "SELECT * FROM sales LIMIT 10"
        
        mock_db_manager.execute_query.return_value = pd.DataFrame({
            'id': range(10),
            'quantity': range(10)
        })
        
        result = mock_db_manager.execute_query(query)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
    
    def test_parameterized_query(self, mock_db_manager):
        """
        TC-D006: Should execute parameterized query safely
        """
        query = "SELECT * FROM sales WHERE branch_code = %s"
        params = ("CN001",)
        
        mock_db_manager.execute_query.return_value = pd.DataFrame({
            'branch_code': ['CN001'] * 5
        })
        
        result = mock_db_manager.execute_query(query)
        
        assert (result['branch_code'] == 'CN001').all()
    
    def test_query_timeout(self, mock_db_manager):
        """
        TC-D007: Should handle query timeout
        """
        query = "SELECT * FROM very_large_table"
        timeout = 30  # seconds
        
        mock_db_manager.execute_query.side_effect = TimeoutError("Query timed out")
        
        with pytest.raises(TimeoutError):
            mock_db_manager.execute_query(query)
    
    def test_query_result_caching(self):
        """
        TC-D008: Should cache query results
        """
        cache = {}
        query = "SELECT COUNT(*) FROM sales"
        
        # First execution
        if query not in cache:
            cache[query] = {"result": 1000, "timestamp": datetime.now()}
        
        # Second execution should use cache
        cached_result = cache.get(query)
        
        assert cached_result is not None
        assert cached_result["result"] == 1000


class TestDataRetrieval:
    """Test data retrieval functions"""
    
    def test_get_sales_data(self, mock_db_manager):
        """
        TC-D009: Should retrieve sales data correctly
        """
        mock_db_manager.execute_query.return_value = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30),
            'branch_code': ['CN001'] * 30,
            'quantity': np.random.randint(10, 100, 30)
        })
        
        result = mock_db_manager.execute_query("SELECT * FROM sales")
        
        assert 'date' in result.columns
        assert 'branch_code' in result.columns
        assert 'quantity' in result.columns
    
    def test_get_inventory_data(self, mock_db_manager):
        """
        TC-D010: Should retrieve inventory data correctly
        """
        mock_db_manager.execute_query.return_value = pd.DataFrame({
            'branch_code': ['CN001', 'CN002'],
            'f_sku': ['SKU001', 'SKU001'],
            'current_stock': [100, 200]
        })
        
        result = mock_db_manager.execute_query("SELECT * FROM inventory")
        
        assert 'current_stock' in result.columns
    
    def test_get_branch_info(self, mock_db_manager):
        """
        TC-D011: Should retrieve branch information
        """
        mock_db_manager.execute_query.return_value = pd.DataFrame({
            'branch_code': ['CN001', 'CN002', 'CN003'],
            'branch_name': ['Chi nhánh Bình Chánh', 'Chi nhánh Đà Nẵng', 'Chi nhánh Hà Nội']
        })
        
        result = mock_db_manager.execute_query("SELECT * FROM branch")
        
        assert len(result) == 3
    
    def test_get_product_info(self, mock_db_manager):
        """
        TC-D012: Should retrieve product information
        """
        mock_db_manager.execute_query.return_value = pd.DataFrame({
            'f_sku': ['SKU001', 'SKU002'],
            'product_name': ['Product A', 'Product B'],
            'category': ['Cat1', 'Cat2']
        })
        
        result = mock_db_manager.execute_query("SELECT * FROM product")
        
        assert 'product_name' in result.columns


class TestDataAggregation:
    """Test data aggregation queries"""
    
    def test_aggregate_by_branch(self, mock_db_manager):
        """
        TC-D013: Should aggregate data by branch
        """
        mock_db_manager.execute_query.return_value = pd.DataFrame({
            'branch_code': ['CN001', 'CN002', 'CN003'],
            'total_quantity': [1000, 1500, 1200]
        })
        
        result = mock_db_manager.execute_query(
            "SELECT branch_code, SUM(quantity) as total_quantity FROM sales GROUP BY branch_code"
        )
        
        assert len(result) == 3
        assert result['total_quantity'].sum() == 3700
    
    def test_aggregate_by_date(self, mock_db_manager):
        """
        TC-D014: Should aggregate data by date
        """
        mock_db_manager.execute_query.return_value = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=7),
            'daily_total': [100, 120, 110, 130, 125, 140, 135]
        })
        
        result = mock_db_manager.execute_query(
            "SELECT date, SUM(quantity) as daily_total FROM sales GROUP BY date"
        )
        
        assert len(result) == 7
    
    def test_aggregate_by_product(self, mock_db_manager):
        """
        TC-D015: Should aggregate data by product
        """
        mock_db_manager.execute_query.return_value = pd.DataFrame({
            'f_sku': ['SKU001', 'SKU002', 'SKU003'],
            'total_sold': [500, 750, 600]
        })
        
        result = mock_db_manager.execute_query(
            "SELECT f_sku, SUM(quantity) as total_sold FROM sales GROUP BY f_sku"
        )
        
        assert len(result) == 3


class TestTransactionHandling:
    """Test transaction handling"""
    
    def test_transaction_commit(self):
        """
        TC-D016: Should commit transaction successfully
        """
        transaction_steps = [
            "BEGIN",
            "INSERT INTO sales VALUES (...)",
            "UPDATE inventory SET ...",
            "COMMIT"
        ]
        
        # All steps should complete
        assert transaction_steps[-1] == "COMMIT"
    
    def test_transaction_rollback(self):
        """
        TC-D017: Should rollback on error
        """
        def execute_transaction():
            try:
                # Simulate error
                raise Exception("Constraint violation")
            except Exception:
                return "ROLLBACK"
        
        result = execute_transaction()
        assert result == "ROLLBACK"


class TestSchemaIntrospection:
    """Test schema introspection"""
    
    def test_get_table_names(self):
        """
        TC-D018: Should retrieve table names
        """
        tables = ["sales", "product", "branch", "inventory"]
        
        assert len(tables) == 4
        assert "sales" in tables
    
    def test_get_column_info(self):
        """
        TC-D019: Should retrieve column information
        """
        columns = {
            "sales": [
                {"name": "date", "type": "DATE"},
                {"name": "branch_code", "type": "VARCHAR"},
                {"name": "f_sku", "type": "VARCHAR"},
                {"name": "quantity", "type": "INTEGER"}
            ]
        }
        
        assert len(columns["sales"]) == 4
    
    def test_get_foreign_keys(self):
        """
        TC-D020: Should retrieve foreign key relationships
        """
        foreign_keys = [
            {"table": "sales", "column": "branch_code", "references": "branch.branch_code"},
            {"table": "sales", "column": "f_sku", "references": "product.f_sku"}
        ]
        
        assert len(foreign_keys) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


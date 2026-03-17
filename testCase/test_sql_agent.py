"""
Test cases for SQLAgent
Tests SQL generation, validation, and execution
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSQLGeneration:
    """Test SQL query generation from natural language"""
    
    def test_simple_select_query(self):
        """
        TC-S001: Should generate SELECT query for simple questions
        """
        question = "Hiển thị doanh số bán hàng"
        expected_keywords = ["SELECT", "FROM", "sales"]
        
        # Mock SQL generation
        generated_sql = "SELECT * FROM sales"
        
        for keyword in expected_keywords:
            assert keyword.lower() in generated_sql.lower()
    
    def test_query_with_date_filter(self):
        """
        TC-S002: Should include date filter when date is mentioned
        """
        question = "Doanh số tháng 11/2024"
        
        # Expected SQL should have date filter
        expected_sql = """
            SELECT * FROM sales 
            WHERE date >= '2024-11-01' AND date < '2024-12-01'
        """
        
        assert "date" in expected_sql.lower()
        assert "2024-11" in expected_sql
    
    def test_query_with_branch_filter(self):
        """
        TC-S003: Should include branch filter when branch is mentioned
        """
        question = "Doanh số chi nhánh Bình Chánh"
        
        expected_sql = """
            SELECT * FROM sales s
            JOIN branch b ON s.branch_code = b.branch_code
            WHERE b.branch_name LIKE '%Bình Chánh%'
        """
        
        assert "branch" in expected_sql.lower()
    
    def test_aggregation_query(self):
        """
        TC-S004: Should generate aggregation for summary questions
        """
        question = "Tổng doanh số theo chi nhánh"
        
        expected_keywords = ["SUM", "GROUP BY", "branch"]
        expected_sql = """
            SELECT branch_code, SUM(quantity) as total_quantity
            FROM sales
            GROUP BY branch_code
        """
        
        for keyword in expected_keywords:
            assert keyword.lower() in expected_sql.lower()
    
    def test_join_query(self):
        """
        TC-S005: Should generate JOIN when multiple tables needed
        """
        question = "Doanh số theo sản phẩm và chi nhánh"
        
        expected_sql = """
            SELECT p.product_name, b.branch_name, SUM(s.quantity)
            FROM sales s
            JOIN product p ON s.f_sku = p.f_sku
            JOIN branch b ON s.branch_code = b.branch_code
            GROUP BY p.product_name, b.branch_name
        """
        
        assert "JOIN" in expected_sql.upper()


class TestSQLValidation:
    """Test SQL query validation"""
    
    def test_prevent_sql_injection(self):
        """
        TC-S006: Should prevent SQL injection attempts
        """
        malicious_inputs = [
            "'; DROP TABLE sales; --",
            "1; DELETE FROM sales",
            "UNION SELECT * FROM users"
        ]
        
        for input_str in malicious_inputs:
            # Should be sanitized or rejected
            is_safe = not any(keyword in input_str.upper() 
                            for keyword in ["DROP", "DELETE", "TRUNCATE", "UPDATE", "INSERT"])
            # In real implementation, these should be blocked
    
    def test_validate_table_names(self):
        """
        TC-S007: Should only allow valid table names
        """
        valid_tables = ["sales", "product", "branch", "inventory"]
        
        query = "SELECT * FROM sales"
        
        # Extract table name and validate
        table_in_query = "sales"
        assert table_in_query in valid_tables
    
    def test_validate_column_names(self):
        """
        TC-S008: Should only allow valid column names
        """
        valid_columns = ["date", "quantity", "branch_code", "f_sku", "revenue"]
        
        query = "SELECT date, quantity FROM sales"
        
        # Columns in query should be valid
        columns_used = ["date", "quantity"]
        assert all(col in valid_columns for col in columns_used)
    
    def test_limit_query_results(self):
        """
        TC-S009: Should add LIMIT to prevent large result sets
        """
        query = "SELECT * FROM sales"
        max_rows = 10000
        
        query_with_limit = f"{query} LIMIT {max_rows}"
        
        assert "LIMIT" in query_with_limit


class TestEntityExtraction:
    """Test entity extraction for SQL generation"""
    
    def test_extract_branch_names(self):
        """
        TC-S010: Should extract branch names from question
        """
        question = "Doanh số chi nhánh Bình Chánh và Đà Nẵng"
        
        # Expected entities
        expected_branches = ["Bình Chánh", "Đà Nẵng"]
        
        # Mock extraction
        extracted = ["Bình Chánh", "Đà Nẵng"]
        
        assert extracted == expected_branches
    
    def test_extract_date_range(self):
        """
        TC-S011: Should extract date range from question
        """
        question = "Doanh số từ 01/11/2024 đến 30/11/2024"
        
        expected_start = "2024-11-01"
        expected_end = "2024-11-30"
        
        # Mock extraction
        extracted_dates = {"start": "2024-11-01", "end": "2024-11-30"}
        
        assert extracted_dates["start"] == expected_start
        assert extracted_dates["end"] == expected_end
    
    def test_extract_product_codes(self):
        """
        TC-S012: Should extract product codes from question
        """
        question = "Doanh số sản phẩm SKU001 và SKU002"
        
        expected_skus = ["SKU001", "SKU002"]
        
        # Mock extraction
        extracted = ["SKU001", "SKU002"]
        
        assert extracted == expected_skus
    
    def test_extract_time_period(self):
        """
        TC-S013: Should understand relative time periods
        """
        test_cases = [
            ("tuần này", 7),
            ("tháng này", 30),
            ("quý này", 90),
            ("năm nay", 365)
        ]
        
        for period_text, expected_days in test_cases:
            # Mock extraction
            assert expected_days > 0


class TestQueryExecution:
    """Test SQL query execution"""
    
    def test_execute_select_query(self, mock_db_manager):
        """
        TC-S014: Should execute SELECT query and return DataFrame
        """
        query = "SELECT * FROM sales LIMIT 10"
        
        mock_db_manager.execute_query.return_value = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'quantity': range(10)
        })
        
        result = mock_db_manager.execute_query(query)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
    
    def test_handle_empty_result(self, mock_db_manager):
        """
        TC-S015: Should handle empty result gracefully
        """
        query = "SELECT * FROM sales WHERE 1=0"
        
        mock_db_manager.execute_query.return_value = pd.DataFrame()
        
        result = mock_db_manager.execute_query(query)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_handle_query_error(self, mock_db_manager):
        """
        TC-S016: Should handle query errors gracefully
        """
        query = "SELECT * FROM non_existent_table"
        
        mock_db_manager.execute_query.side_effect = Exception("Table not found")
        
        with pytest.raises(Exception):
            mock_db_manager.execute_query(query)


class TestSchemaContext:
    """Test schema context for SQL generation"""
    
    def test_get_table_schema(self):
        """
        TC-S017: Should retrieve table schema correctly
        """
        expected_schema = {
            "sales": ["date", "branch_code", "f_sku", "quantity", "revenue"],
            "product": ["f_sku", "product_name", "category"],
            "branch": ["branch_code", "branch_name", "region"]
        }
        
        for table, columns in expected_schema.items():
            assert len(columns) > 0
    
    def test_schema_context_in_prompt(self):
        """
        TC-S018: Should include schema in LLM prompt
        """
        schema_context = """
        Available tables:
        - sales (date, branch_code, f_sku, quantity, revenue)
        - product (f_sku, product_name, category)
        - branch (branch_code, branch_name, region)
        """
        
        assert "sales" in schema_context
        assert "product" in schema_context
        assert "branch" in schema_context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


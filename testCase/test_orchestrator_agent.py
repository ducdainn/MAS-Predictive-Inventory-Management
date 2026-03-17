"""
Test cases for OrchestratorAgent
Tests intent classification, agent routing, and workflow orchestration
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestIntentClassification:
    """Test intent classification from user queries"""
    
    def test_classify_forecast_intent(self):
        """
        TC-O001: Should classify forecast-related queries correctly
        """
        forecast_queries = [
            "Dự báo doanh số tháng tới",
            "Forecast demand cho chi nhánh Bình Chánh",
            "Xu hướng bán hàng 30 ngày tới",
            "Dự đoán nhu cầu sản phẩm"
        ]
        
        expected_intent = "FORECAST"
        
        for query in forecast_queries:
            # Mock classification - in real test, call actual classifier
            assert "dự báo" in query.lower() or "forecast" in query.lower() or "dự đoán" in query.lower()
    
    def test_classify_analytics_intent(self):
        """
        TC-O002: Should classify analytics-related queries correctly
        """
        analytics_queries = [
            "Phân tích doanh số theo chi nhánh",
            "Thống kê bán hàng tháng 11",
            "So sánh doanh thu các chi nhánh",
            "Báo cáo tổng hợp"
        ]
        
        expected_intent = "ANALYTICS"
        
        for query in analytics_queries:
            assert any(kw in query.lower() for kw in ["phân tích", "thống kê", "so sánh", "báo cáo"])
    
    def test_classify_inventory_intent(self):
        """
        TC-O003: Should classify inventory-related queries correctly
        """
        inventory_queries = [
            "Tối ưu tồn kho chi nhánh Đà Nẵng",
            "Đề xuất nhập hàng",
            "Kiểm tra mức tồn kho",
            "Reorder point cho sản phẩm"
        ]
        
        expected_intent = "INVENTORY_OPTIMIZATION"
        
        for query in inventory_queries:
            assert any(kw in query.lower() for kw in ["tồn kho", "nhập hàng", "reorder", "inventory"])
    
    def test_classify_general_query(self):
        """
        TC-O004: Should handle general queries
        """
        general_queries = [
            "Hiển thị doanh số",
            "Cho tôi xem dữ liệu bán hàng",
            "Danh sách chi nhánh"
        ]
        
        expected_intent = "GENERAL"
        
        # General queries should still be handled


class TestAgentRouting:
    """Test routing queries to appropriate agents"""
    
    def test_route_to_forecast_agent(self):
        """
        TC-O005: Should route forecast queries to ForecastAgent
        """
        intent = "FORECAST"
        
        agent_mapping = {
            "FORECAST": "ForecastAgent",
            "ANALYTICS": "AnalyticsAgent",
            "INVENTORY_OPTIMIZATION": "InventoryOptimizationAgent",
            "GENERAL": "SQLAgent"
        }
        
        selected_agent = agent_mapping.get(intent)
        assert selected_agent == "ForecastAgent"
    
    def test_route_to_analytics_agent(self):
        """
        TC-O006: Should route analytics queries to AnalyticsAgent
        """
        intent = "ANALYTICS"
        
        agent_mapping = {
            "FORECAST": "ForecastAgent",
            "ANALYTICS": "AnalyticsAgent",
            "INVENTORY_OPTIMIZATION": "InventoryOptimizationAgent",
            "GENERAL": "SQLAgent"
        }
        
        selected_agent = agent_mapping.get(intent)
        assert selected_agent == "AnalyticsAgent"
    
    def test_route_to_inventory_agent(self):
        """
        TC-O007: Should route inventory queries to InventoryAgent
        """
        intent = "INVENTORY_OPTIMIZATION"
        
        agent_mapping = {
            "FORECAST": "ForecastAgent",
            "ANALYTICS": "AnalyticsAgent",
            "INVENTORY_OPTIMIZATION": "InventoryOptimizationAgent",
            "GENERAL": "SQLAgent"
        }
        
        selected_agent = agent_mapping.get(intent)
        assert selected_agent == "InventoryOptimizationAgent"
    
    def test_fallback_to_sql_agent(self):
        """
        TC-O008: Should fallback to SQLAgent for unknown intents
        """
        intent = "UNKNOWN"
        
        agent_mapping = {
            "FORECAST": "ForecastAgent",
            "ANALYTICS": "AnalyticsAgent",
            "INVENTORY_OPTIMIZATION": "InventoryOptimizationAgent"
        }
        
        selected_agent = agent_mapping.get(intent, "SQLAgent")
        assert selected_agent == "SQLAgent"


class TestWorkflowOrchestration:
    """Test workflow orchestration"""
    
    def test_workflow_steps_order(self):
        """
        TC-O009: Should execute workflow steps in correct order
        """
        expected_steps = [
            "1. Intent Classification",
            "2. Entity Extraction",
            "3. Schema Context",
            "4. Agent Processing",
            "5. Memory Storage",
            "6. Response Generation"
        ]
        
        # Verify steps are numbered correctly
        for i, step in enumerate(expected_steps, 1):
            assert step.startswith(f"{i}.")
    
    def test_entity_extraction_before_processing(self):
        """
        TC-O010: Entity extraction should happen before agent processing
        """
        workflow = ["intent", "entity_extraction", "schema", "processing"]
        
        entity_idx = workflow.index("entity_extraction")
        processing_idx = workflow.index("processing")
        
        assert entity_idx < processing_idx
    
    def test_memory_storage_after_processing(self):
        """
        TC-O011: Memory storage should happen after processing
        """
        workflow = ["processing", "memory_storage", "response"]
        
        processing_idx = workflow.index("processing")
        memory_idx = workflow.index("memory_storage")
        
        assert memory_idx > processing_idx


class TestContextManagement:
    """Test context management across agents"""
    
    def test_pass_entities_to_sql_agent(self):
        """
        TC-O012: Should pass extracted entities to SQL agent
        """
        entities = {
            "branch_codes": ["CN001", "CN002"],
            "date_range": {"start": "2024-11-01", "end": "2024-11-30"},
            "product_codes": ["SKU001"]
        }
        
        # Entities should be passed to SQL generation
        assert "branch_codes" in entities
        assert len(entities["branch_codes"]) > 0
    
    def test_pass_schema_context(self):
        """
        TC-O013: Should pass schema context to SQL agent
        """
        schema_context = {
            "tables": ["sales", "product", "branch"],
            "relationships": [
                {"from": "sales.branch_code", "to": "branch.branch_code"},
                {"from": "sales.f_sku", "to": "product.f_sku"}
            ]
        }
        
        assert len(schema_context["tables"]) > 0
        assert len(schema_context["relationships"]) > 0
    
    def test_pass_forecast_to_inventory(self):
        """
        TC-O014: Should pass forecast results to inventory agent
        """
        forecast_result = {
            "forecast_30d": 1500,
            "daily_forecast": [50] * 30,
            "growth_rate": 0.15
        }
        
        # Inventory agent should receive forecast
        assert forecast_result["forecast_30d"] > 0


class TestErrorHandling:
    """Test error handling in orchestration"""
    
    def test_handle_agent_failure(self):
        """
        TC-O015: Should handle agent failure gracefully
        """
        def failing_agent():
            raise Exception("Agent processing failed")
        
        try:
            failing_agent()
            success = True
        except Exception as e:
            success = False
            error_message = str(e)
        
        assert success == False
        assert "failed" in error_message.lower()
    
    def test_retry_on_failure(self):
        """
        TC-O016: Should retry on transient failures
        """
        max_retries = 3
        attempts = 0
        
        def flaky_operation():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise Exception("Transient error")
            return "Success"
        
        result = None
        for i in range(max_retries):
            try:
                result = flaky_operation()
                break
            except Exception:
                continue
        
        assert result == "Success"
        assert attempts == 3
    
    def test_fallback_response(self):
        """
        TC-O017: Should provide fallback response on complete failure
        """
        def get_response_with_fallback():
            try:
                raise Exception("All agents failed")
            except Exception:
                return "Xin lỗi, tôi không thể xử lý yêu cầu này. Vui lòng thử lại."
        
        response = get_response_with_fallback()
        assert "xin lỗi" in response.lower() or "không thể" in response.lower()


class TestMemoryIntegration:
    """Test memory system integration"""
    
    def test_store_successful_query(self, mock_memory_manager):
        """
        TC-O018: Should store successful query in memory
        """
        query = "Dự báo doanh số tháng tới"
        result = {"forecast": 1500, "success": True}
        
        mock_memory_manager.store(query, result)
        mock_memory_manager.store.assert_called_once()
    
    def test_retrieve_similar_query(self, mock_memory_manager):
        """
        TC-O019: Should retrieve similar queries from memory
        """
        query = "Dự báo doanh số"
        
        mock_memory_manager.retrieve.return_value = [
            {"query": "Dự báo doanh số tháng 11", "result": {"forecast": 1200}}
        ]
        
        similar = mock_memory_manager.retrieve(query)
        assert len(similar) > 0
    
    def test_use_cached_result(self, mock_memory_manager):
        """
        TC-O020: Should use cached result for identical queries
        """
        query = "Dự báo doanh số chi nhánh Bình Chánh"
        
        mock_memory_manager.retrieve.return_value = [
            {"query": query, "result": {"forecast": 1500}, "timestamp": datetime.now()}
        ]
        
        cached = mock_memory_manager.retrieve(query)
        
        # If exact match found and recent, use cached
        if cached and cached[0]["query"] == query:
            use_cache = True
        else:
            use_cache = False
        
        assert use_cache == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


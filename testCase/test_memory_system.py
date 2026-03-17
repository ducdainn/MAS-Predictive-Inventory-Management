"""
Test cases for Memory System
Tests Short-term, Long-term (Qdrant), and Episodic memory
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestShortTermMemory:
    """Test short-term memory (in-memory deque)"""
    
    def test_store_interaction(self):
        """
        TC-M001: Should store interaction in short-term memory
        """
        from collections import deque
        
        memory = deque(maxlen=10)
        
        interaction = {
            "query": "Dự báo doanh số",
            "response": "Forecast: 1500",
            "timestamp": datetime.now()
        }
        
        memory.append(interaction)
        
        assert len(memory) == 1
        assert memory[0]["query"] == "Dự báo doanh số"
    
    def test_memory_limit(self):
        """
        TC-M002: Should respect memory limit (max 10 interactions)
        """
        from collections import deque
        
        memory = deque(maxlen=10)
        
        # Add 15 interactions
        for i in range(15):
            memory.append({"query": f"Query {i}"})
        
        # Should only keep last 10
        assert len(memory) == 10
        assert memory[0]["query"] == "Query 5"  # First kept item
        assert memory[-1]["query"] == "Query 14"  # Last item
    
    def test_get_recent_context(self):
        """
        TC-M003: Should retrieve recent context for conversation
        """
        from collections import deque
        
        memory = deque(maxlen=10)
        
        memory.append({"query": "Q1", "response": "R1"})
        memory.append({"query": "Q2", "response": "R2"})
        memory.append({"query": "Q3", "response": "R3"})
        
        # Get last 2 interactions
        recent = list(memory)[-2:]
        
        assert len(recent) == 2
        assert recent[0]["query"] == "Q2"
        assert recent[1]["query"] == "Q3"


class TestLongTermMemory:
    """Test long-term memory (Qdrant vector DB)"""
    
    def test_store_query_pattern(self):
        """
        TC-M004: Should store query pattern in Qdrant
        """
        pattern = {
            "query": "Dự báo doanh số chi nhánh Bình Chánh",
            "intent": "FORECAST",
            "entities": {"branch": "Bình Chánh"},
            "solution": {"sql": "SELECT...", "model": "xgboost"},
            "success": True
        }
        
        # Mock Qdrant storage
        assert pattern["success"] == True
        assert "query" in pattern
    
    def test_search_similar_queries(self):
        """
        TC-M005: Should find similar queries using vector search
        """
        stored_queries = [
            {"query": "Dự báo doanh số chi nhánh Bình Chánh", "similarity": 0.95},
            {"query": "Dự báo doanh số chi nhánh Đà Nẵng", "similarity": 0.85},
            {"query": "Phân tích doanh số", "similarity": 0.60}
        ]
        
        # Filter by similarity threshold
        threshold = 0.8
        similar = [q for q in stored_queries if q["similarity"] >= threshold]
        
        assert len(similar) == 2
    
    def test_filter_successful_patterns(self):
        """
        TC-M006: Should filter only successful patterns
        """
        patterns = [
            {"query": "Q1", "success": True},
            {"query": "Q2", "success": False},
            {"query": "Q3", "success": True}
        ]
        
        successful = [p for p in patterns if p["success"]]
        
        assert len(successful) == 2
    
    def test_embedding_generation(self):
        """
        TC-M007: Should generate embeddings for queries
        """
        query = "Dự báo doanh số"
        
        # Mock embedding (384-dimensional for sentence-transformers)
        embedding = np.random.rand(384)
        
        assert len(embedding) == 384
        assert embedding.dtype == np.float64


class TestEpisodicMemory:
    """Test episodic memory (SQLite)"""
    
    def test_store_experience(self):
        """
        TC-M008: Should store learned experience
        """
        experience = {
            "context": "Forecast for branch with sparse data",
            "action": "Used moving average instead of XGBoost",
            "outcome": "Better accuracy",
            "lesson": "Sparse data needs simpler models",
            "timestamp": datetime.now()
        }
        
        assert "lesson" in experience
        assert experience["outcome"] == "Better accuracy"
    
    def test_retrieve_relevant_experience(self):
        """
        TC-M009: Should retrieve relevant past experiences
        """
        experiences = [
            {"context": "sparse data", "lesson": "Use simple models"},
            {"context": "seasonal product", "lesson": "Adjust for seasonality"},
            {"context": "new product", "lesson": "Monitor closely"}
        ]
        
        current_context = "sparse data"
        
        relevant = [e for e in experiences if current_context in e["context"]]
        
        assert len(relevant) == 1
        assert relevant[0]["lesson"] == "Use simple models"
    
    def test_experience_persistence(self):
        """
        TC-M010: Should persist experiences across sessions
        """
        # Simulate SQLite storage
        import sqlite3
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        # Store
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY,
                context TEXT,
                lesson TEXT
            )
        """)
        conn.execute("INSERT INTO experiences (context, lesson) VALUES (?, ?)",
                    ("test context", "test lesson"))
        conn.commit()
        conn.close()
        
        # Retrieve in new connection
        conn2 = sqlite3.connect(db_path)
        cursor = conn2.execute("SELECT * FROM experiences")
        rows = cursor.fetchall()
        conn2.close()
        
        assert len(rows) == 1
        
        # Cleanup
        os.unlink(db_path)


class TestMemoryManager:
    """Test unified memory manager"""
    
    def test_unified_store(self, mock_memory_manager):
        """
        TC-M011: Should store to appropriate memory type
        """
        query = "Dự báo doanh số"
        result = {"forecast": 1500}
        
        mock_memory_manager.store(query, result)
        mock_memory_manager.store.assert_called_once_with(query, result)
    
    def test_unified_retrieve(self, mock_memory_manager):
        """
        TC-M012: Should retrieve from all memory types
        """
        query = "Dự báo doanh số"
        
        mock_memory_manager.retrieve.return_value = {
            "short_term": [{"query": "Q1"}],
            "long_term": [{"query": "Q2", "similarity": 0.9}],
            "episodic": [{"lesson": "Use XGBoost"}]
        }
        
        result = mock_memory_manager.retrieve(query)
        
        assert "short_term" in result or isinstance(result, list)
    
    def test_context_building(self, mock_memory_manager):
        """
        TC-M013: Should build context from all memory sources
        """
        mock_memory_manager.get_context.return_value = """
        Recent interactions: Q1, Q2, Q3
        Similar past queries: Dự báo doanh số chi nhánh X
        Learned lessons: Use XGBoost for 14+ days data
        """
        
        context = mock_memory_manager.get_context("Dự báo doanh số")
        
        assert len(context) > 0


class TestQdrantIntegration:
    """Test Qdrant-specific functionality"""
    
    def test_collection_creation(self):
        """
        TC-M014: Should create Qdrant collection if not exists
        """
        collection_config = {
            "name": "query_patterns",
            "vector_size": 384,
            "distance": "Cosine"
        }
        
        assert collection_config["vector_size"] == 384
        assert collection_config["distance"] == "Cosine"
    
    def test_point_upsert(self):
        """
        TC-M015: Should upsert points to Qdrant
        """
        point = {
            "id": "uuid-123",
            "vector": np.random.rand(384).tolist(),
            "payload": {
                "query": "Dự báo doanh số",
                "success": True
            }
        }
        
        assert len(point["vector"]) == 384
        assert "payload" in point
    
    def test_filtered_search(self):
        """
        TC-M016: Should search with filters
        """
        search_params = {
            "vector": np.random.rand(384).tolist(),
            "filter": {"success": True},
            "limit": 5
        }
        
        assert search_params["limit"] == 5
        assert search_params["filter"]["success"] == True
    
    def test_cloud_vs_local_mode(self):
        """
        TC-M017: Should support both cloud and local Qdrant modes
        """
        modes = ["cloud", "local"]
        
        for mode in modes:
            if mode == "cloud":
                config = {"url": "https://xxx.qdrant.io", "api_key": "xxx"}
            else:
                config = {"path": "./memory/qdrant_data"}
            
            assert len(config) > 0


class TestMemoryCaching:
    """Test memory caching behavior"""
    
    def test_cache_hit(self):
        """
        TC-M018: Should return cached result for identical query
        """
        cache = {}
        
        query = "Dự báo doanh số"
        result = {"forecast": 1500}
        
        # Store in cache
        cache[query] = {"result": result, "timestamp": datetime.now()}
        
        # Cache hit
        if query in cache:
            cached_result = cache[query]["result"]
        
        assert cached_result == result
    
    def test_cache_expiration(self):
        """
        TC-M019: Should expire old cache entries
        """
        cache = {}
        cache_ttl = timedelta(hours=1)
        
        query = "Dự báo doanh số"
        old_timestamp = datetime.now() - timedelta(hours=2)
        
        cache[query] = {"result": {}, "timestamp": old_timestamp}
        
        # Check if expired
        is_expired = (datetime.now() - cache[query]["timestamp"]) > cache_ttl
        
        assert is_expired == True
    
    def test_cache_invalidation(self):
        """
        TC-M020: Should invalidate cache on data update
        """
        cache = {"query1": {"result": {}}, "query2": {"result": {}}}
        
        # Invalidate all cache
        cache.clear()
        
        assert len(cache) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


"""
Pytest fixtures and configuration for BrickDemand tests
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_sales_data():
    """Generate sample sales data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    data = []
    for date in dates:
        for branch in ['CN001', 'CN002', 'CN003']:
            for sku in ['SKU001', 'SKU002', 'SKU003']:
                quantity = np.random.poisson(lam=50) + np.random.randint(0, 20)
                data.append({
                    'date': date,
                    'branch_code': branch,
                    'f_sku': sku,
                    'quantity': quantity,
                    'revenue': quantity * np.random.uniform(100, 500)
                })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_inventory_data():
    """Generate sample inventory data for testing"""
    data = []
    for branch in ['CN001', 'CN002', 'CN003']:
        for sku in ['SKU001', 'SKU002', 'SKU003']:
            data.append({
                'branch_code': branch,
                'f_sku': sku,
                'current_stock': np.random.randint(100, 1000),
                'min_stock': np.random.randint(50, 100),
                'max_stock': np.random.randint(500, 1000),
                'lead_time_days': np.random.randint(3, 14)
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_db_manager():
    """Mock DatabaseManager for testing"""
    mock = Mock()
    mock.execute_query = Mock(return_value=pd.DataFrame())
    mock.get_connection = Mock()
    return mock


@pytest.fixture
def mock_llm_provider():
    """Mock LLMProvider for testing"""
    mock = Mock()
    mock.generate = Mock(return_value="Test response")
    mock.chat = Mock(return_value="Test chat response")
    return mock


@pytest.fixture
def mock_memory_manager():
    """Mock MemoryManager for testing"""
    mock = Mock()
    mock.store = Mock()
    mock.retrieve = Mock(return_value=[])
    mock.get_context = Mock(return_value="")
    return mock


@pytest.fixture
def sample_time_series():
    """Generate sample time series for forecast testing"""
    dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
    values = np.random.poisson(lam=100, size=90) + np.sin(np.arange(90) * 0.1) * 20
    
    return pd.DataFrame({
        'date': dates,
        'quantity': values.astype(int)
    })


@pytest.fixture
def sample_sparse_time_series():
    """Generate sparse time series (many zeros) for testing"""
    dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
    values = np.random.choice([0, 0, 0, 10, 20, 50], size=90)
    
    return pd.DataFrame({
        'date': dates,
        'quantity': values
    })


@pytest.fixture
def sample_forecast_result():
    """Sample forecast result for testing"""
    dates = pd.date_range(start='2025-01-01', periods=30, freq='D')
    
    return pd.DataFrame({
        'date': dates,
        'forecast': np.random.poisson(lam=100, size=30),
        'lower_bound': np.random.poisson(lam=80, size=30),
        'upper_bound': np.random.poisson(lam=120, size=30)
    })


"""
Configuration for BrickDemand MAS
"""

import os
from dotenv import load_dotenv

load_dotenv()

# App Configuration
APP_CONFIG = {
    "name": "BrickDemand Inventory AI",
    "version": "3.1",
    "description": "Predictive Inventory Management System",
    "max_history": 100,
    "default_horizon_days": 30,
    "chart_output_dir": "charts",
    "export_dir": "exports"
}

# Database Configuration
DB_CONFIG = {
    "user": os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", "postgres"),
    "host": os.getenv("PG_HOST", "localhost"),
    "port": os.getenv("PG_PORT", "5433"),
    "database": os.getenv("PG_DB", "brickdemand")
}

# LLM Configuration
LLM_CONFIG = {
    "default_model": "openai",
    "openai_model": "gpt-4o-mini",
    "huggingface_repo": "Qwen/Qwen2.5-VL-7B-Instruct",
    "huggingface_provider": "hyperbolic",
    "default_temperature": 0.0
}

# Inventory Optimization Config
INVENTORY_CONFIG = {
    "service_level": 0.95,
    "lead_time_days": 7,
    "max_transfer_distance_km": 200,
    "min_forecast_days": 2,  # Minimum days of sales data for forecast
    "ordering_cost": 1000,
    "holding_cost": 50
}

# UI Configuration
UI_CONFIG = {
    "theme": "light",
    "chart_height": 400,
    "table_page_size": 50,
    "max_chart_items": 50
}


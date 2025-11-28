"""
Database manager responsible for PostgreSQL connectivity and safe query execution.
"""

import os
from typing import Dict, Optional

import pandas as pd
from sqlalchemy import create_engine, text

try:
    from agent.system_date import get_system_date
    SYSTEM_DATE_AVAILABLE = True
except ImportError:
    SYSTEM_DATE_AVAILABLE = False


class DatabaseManager:
    """
    Manages database connections and queries.

    Improvements:
    - Supports PARAMETERIZED QUERIES to prevent SQL injection
    - Uses SQLAlchemy text() with bound parameters
    """

    def __init__(self):
        self.PG_USER = os.getenv("PG_USER", "postgres")
        self.PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
        self.PG_HOST = os.getenv("PG_HOST", "localhost")
        self.PG_PORT = os.getenv("PG_PORT", "5433")
        self.PG_DB = os.getenv("PG_DB", "brickdemand")

        uri = (
            f"postgresql+psycopg2://{self.PG_USER}:{self.PG_PASSWORD}"
            f"@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DB}"
        )
        self.engine = create_engine(uri, pool_pre_ping=True, pool_size=5)
        print(f"✅ Connected to database: {self.PG_DB}")

    def _replace_current_date(self, query: str) -> str:
        """
        Replace CURRENT_DATE with system date in SQL.

        This ensures queries use the configured system date (e.g., 2025-12-06)
        instead of the real current date.
        """
        if SYSTEM_DATE_AVAILABLE:
            system_date = get_system_date()
            query = query.replace("CURRENT_DATE", f"DATE '{system_date}'")
        return query

    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute SQL and return DataFrame with PARAMETERIZED QUERIES.

        Args:
            query: SQL query with :param_name placeholders
            params: Dict of parameters {param_name: value}

        Example:
            db.execute_query(
                "SELECT * FROM sales WHERE branch_code = :branch AND date >= :date",
                {"branch": 101, "date": "2024-01-01"}
            )
        """
        try:
            query = self._replace_current_date(query)
            preview = query if len(query) <= 800 else query[:800] + "... [truncated]"
            print("\n📝 Executing SQL:")
            print(preview)
            if params:
                print(f"   ↪ Params: {params}")

            with self.engine.connect() as conn:
                if params:
                    result = pd.read_sql(text(query), conn, params=params)
                else:
                    result = pd.read_sql(text(query), conn)
            return result
        except Exception as e:
            print(f"❌ Query error: {e}")
            print(f"Query: {query[:200]}...")
            if params:
                print(f"Params: {params}")
            raise




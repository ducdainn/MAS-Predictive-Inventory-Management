"""
🤖 Improved Multi-Agent System for BrickDemand Analytics & Forecasting

Architecture:
    User → Orchestrator → [Intent, Schema, SQL, Analytics/Forecast] → Results

Agents:
    1. SchemaAgent: DB schema understanding
    2. IntentAgent: Question classification (FORECAST vs ANALYTICS)
    3. SQLAgent: Smart SQL generation
    4. AnalyticsAgent: Visualization & analysis
    5. ForecastAgent: Time series forecasting
    6. OrchestratorAgent: Main coordinator

Author: AI Assistant
Date: 2025-10-23
"""

import os
import re
import json
import uuid
import warnings
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
load_dotenv()


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ConversationEntry:
    """Single conversation entry in memory."""
    timestamp: datetime
    question: str
    intent: str
    sql_query: Optional[str] = None
    result_summary: Optional[str] = None
    charts: List[str] = field(default_factory=list)


# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """Manages database connections and queries."""
    
    def __init__(self):
        self.PG_USER = os.getenv("PG_USER", "postgres")
        self.PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
        self.PG_HOST = os.getenv("PG_HOST", "localhost")
        self.PG_PORT = os.getenv("PG_PORT", "5433")
        self.PG_DB = os.getenv("PG_DB", "brickdemand")
        
        uri = f"postgresql+psycopg2://{self.PG_USER}:{self.PG_PASSWORD}@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DB}"
        self.engine = create_engine(uri, pool_pre_ping=True, pool_size=5)
        print(f"✅ Connected to database: {self.PG_DB}")
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL and return DataFrame."""
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql(text(query), conn)
            return result
        except Exception as e:
            print(f"❌ Query error: {e}")
            print(f"Query: {query[:200]}...")
            raise


# ============================================================================
# MEMORY MANAGER
# ============================================================================

class MemoryManager:
    """Manages conversation history and context."""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversation_history: List[ConversationEntry] = []
        self.schema_cache: Dict[str, Any] = {}
        self._initialize_schema_cache()
    
    def _initialize_schema_cache(self):
        """Cache schema info and common patterns."""
        self.schema_cache = {
            "tables": ["branch", "product", "inventory", "sales"],
            "key_columns": {
                "branch": ["branch_code", "region", "branch_name"],
                "product": ["product_code", "product_name", "category", "unit"],
                "inventory": ["product_code", "branch_code", "quantity"],
                "sales": ["date", "branch_code", "product_code", "quantity", "square_meters"]
            },
            "relationships": [
                "sales JOIN branch ON sales.branch_code = branch.branch_code",
                "sales JOIN product ON sales.product_code = product.product_code",
                "inventory JOIN branch ON inventory.branch_code = branch.branch_code",
                "inventory JOIN product ON inventory.product_code = product.product_code"
            ]
        }
    
    def add_entry(self, entry: ConversationEntry):
        """Add conversation entry to history."""
        self.conversation_history.append(entry)
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
    
    def get_recent_context(self, n: int = 3) -> str:
        """Get recent conversation context."""
        recent = self.conversation_history[-n:]
        if not recent:
            return "No previous context."
        
        context = "Recent conversation:\n"
        for entry in recent:
            context += f"- Q: {entry.question[:100]}... (Intent: {entry.intent})\n"
        return context
    
    def get_similar_queries(self, question: str, top_k: int = 2) -> List[str]:
        """Find similar past queries using keyword matching."""
        if not self.conversation_history:
            return []
        
        keywords = set(question.lower().split())
        similar = []
        
        for entry in self.conversation_history:
            entry_keywords = set(entry.question.lower().split())
            overlap = len(keywords & entry_keywords)
            if overlap > 2 and entry.sql_query:
                similar.append((overlap, entry.sql_query))
        
        similar.sort(reverse=True, key=lambda x: x[0])
        return [sql for _, sql in similar[:top_k]]


# ============================================================================
# LLM PROVIDER
# ============================================================================

class LLMProvider:
    """Manages LLM instances with caching."""
    
    def __init__(self):
        self._llm_cache = {}
    
    def get_llm(self, model_type: str = "openai", temperature: float = 0.0):
        """Get LLM instance with caching."""
        cache_key = f"{model_type}_{temperature}"
        
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]
        
        if model_type == "openai":
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
        elif model_type == "huggingface":
            endpoint = HuggingFaceEndpoint(
                repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
                task="text-generation",
                provider="hyperbolic"
            )
            llm = ChatHuggingFace(llm=endpoint)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self._llm_cache[cache_key] = llm
        return llm


# ============================================================================
# SCHEMA AGENT
# ============================================================================

class SchemaAgent:
    """Understands database schema and provides context."""
    
    def __init__(self, db_manager: DatabaseManager, memory: MemoryManager):
        self.db = db_manager
        self.memory = memory
        self.schema_summary = self._build_schema_summary()
    
    def _build_schema_summary(self) -> str:
        """Build comprehensive schema summary with examples."""
        return """
DATABASE SCHEMA FOR BRICKDEMAND:

1. BRANCH (Chi nhánh)
   - branch_code: INTEGER (PK)
   - region: TEXT (MIỀN BẮC, MIỀN TRUNG, MIỀN NAM)
   - branch_name: TEXT

2. PRODUCT (Sản phẩm)
   - product_code: VARCHAR(128) (PK)
   - product_name: TEXT (e.g., 'Gạch 30x60 MS 4566 Loại 2')
   - category: TEXT
   - unit: TEXT (e.g., 'viên', 'thùng')

3. INVENTORY (Tồn kho)
   - product_code, branch_code (Composite PK, FKs)
   - quantity: INTEGER (≥ 0)

4. SALES (Bán hàng)
   - id: BIGSERIAL (PK)
   - date: DATE
   - branch_code, product_code (FKs)
   - quantity: INTEGER
   - square_meters: NUMERIC(12,2)

5. BRANCH_DISTANCE (Khoảng cách giữa các chi nhánh)
   - branch_code_1: INTEGER (FK → branch)
   - branch_code_2: INTEGER (FK → branch)
   - distance_km: NUMERIC(12,2) (khoảng cách km)
   Used for: Finding nearby branches for inventory transfer

KEY RELATIONSHIPS:
- sales ⋈ branch ON sales.branch_code = branch.branch_code
- sales ⋈ product ON sales.product_code = product.product_code
- inventory ⋈ branch ON inventory.branch_code = branch.branch_code
- inventory ⋈ product ON inventory.product_code = product.product_code
- branch_distance: Symmetric relation for branch proximity

COMMON PATTERNS:
- Time filters: WHERE date >= CURRENT_DATE - INTERVAL '30 days'
- Region filter: WHERE region IN ('MIỀN BẮC', 'MIỀN TRUNG', 'MIỀN NAM')
- Aggregations: GROUP BY date/branch_code/product_code
- Distance query: Find branches within X km for transfer optimization
"""
    
    def get_schema_context(self, question: str) -> str:
        """Get relevant schema context based on question."""
        context = self.schema_summary
        
        # Add similar queries if available
        similar = self.memory.get_similar_queries(question)
        if similar:
            context += "\n\nSIMILAR PAST QUERIES:\n"
            for i, sql in enumerate(similar, 1):
                context += f"{i}. {sql}\n"
        
        return context


# ============================================================================
# INTENT AGENT
# ============================================================================

class IntentAgent:
    """Classifies user intent: FORECAST vs ANALYTICS."""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider.get_llm("openai", temperature=0.0)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", "{question}")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def _get_system_prompt(self) -> str:
        return """You are an intent classifier for a brick sales analytics system.

Classify the user's question into ONE of these categories:

1. FORECAST - Predicting future demand/sales
   Keywords: dự báo, forecast, dự đoán, predict, tương lai, future, nhu cầu

2. ANALYTICS - Data analysis and visualization
   Keywords: biểu đồ, chart, phân tích, analysis, thống kê, distribution, top, ranking

3. INVENTORY_OPTIMIZATION - Inventory management, restock, transfer decisions
   Keywords: tồn kho, inventory, nhập hàng, restock, chuyển kho, transfer, ROP, safety stock, 
            kế hoạch, plan, tối ưu, optimize, stock level

Return ONLY one word: FORECAST, ANALYTICS, or INVENTORY_OPTIMIZATION"""
    
    def classify(self, question: str) -> str:
        """Classify question intent using hybrid approach."""
        question_lower = question.lower()
        
        # Fast heuristic check
        forecast_keywords = ['dự báo', 'forecast', 'dự đoán', 'predict', 'tương lai', 'future', 'nhu cầu']
        analytics_keywords = ['biểu đồ', 'chart', 'plot', 'phân tích', 'analysis', 'top', 'thống kê', 'distribution']
        inventory_keywords = ['tồn kho', 'inventory', 'nhập hàng', 'restock', 'chuyển kho', 'transfer', 
                             'rop', 'safety stock', 'kế hoạch', 'plan', 'tối ưu', 'optimize', 'stock level',
                             'order', 'replenish', 'surplus', 'shortage', 'stockout']
        
        forecast_score = sum(1 for kw in forecast_keywords if kw in question_lower)
        analytics_score = sum(1 for kw in analytics_keywords if kw in question_lower)
        inventory_score = sum(1 for kw in inventory_keywords if kw in question_lower)
        
        # Priority: INVENTORY_OPTIMIZATION > FORECAST > ANALYTICS
        if inventory_score > 0 and inventory_score >= max(forecast_score, analytics_score):
            return "INVENTORY_OPTIMIZATION"
        elif forecast_score > analytics_score and forecast_score > 0:
            return "FORECAST"
        elif analytics_score > 0:
            return "ANALYTICS"
        
        # Use LLM if unclear
        try:
            result = self.chain.invoke({"question": question}).strip().upper()
            if result in ["FORECAST", "ANALYTICS", "INVENTORY_OPTIMIZATION"]:
                return result
        except:
            pass
        
        return "ANALYTICS"  # Default


# ============================================================================
# SQL AGENT
# ============================================================================

class SQLAgent:
    """Generates SQL queries from natural language."""
    
    def __init__(self, llm_provider: LLMProvider, schema_agent: SchemaAgent):
        self.llm = llm_provider.get_llm("openai", temperature=0.0)
        self.schema_agent = schema_agent
    
    def generate_sql(self, question: str, intent: str) -> str:
        """Generate SQL query with schema context."""
        schema_context = self.schema_agent.get_schema_context(question)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt(intent)),
            ("human", "Schema:\n{schema}\n\nQuestion: {question}\n\nSQL Query:")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            raw_sql = chain.invoke({"schema": schema_context, "question": question})
            sql = self._clean_sql(raw_sql)
            self._validate_sql(sql)
            return sql
        except Exception as e:
            print(f"⚠️ SQL generation failed: {e}")
            return self._retry_generate_sql(question, schema_context, str(e))
    
    def _get_system_prompt(self, intent: str) -> str:
        base = """You are a PostgreSQL expert. Generate ONLY a valid SELECT query.

RULES:
1. Return ONLY the SQL query - no explanations, no markdown, no quotes
2. Start with SELECT or WITH
3. Use proper JOINs as shown in schema
4. Use date filters with INTERVAL notation
5. Always include relevant columns in SELECT
6. Use meaningful aliases
7. No INSERT/UPDATE/DELETE/DROP allowed
"""
        
        if intent == "FORECAST":
            base += """
FOR FORECAST QUERIES:
- Include historical data (at least last 90 days)
- Group by date to get time series
- Include product_code and branch_code for filtering
- Order by date ASC
Example: SELECT date, SUM(quantity) as total_qty FROM sales 
         WHERE date >= CURRENT_DATE - INTERVAL '90 days' 
         GROUP BY date ORDER BY date"""
        else:
            base += """
FOR ANALYTICS QUERIES:
- Use appropriate aggregations (SUM, AVG, COUNT)
- Include dimension for grouping (branch_name, product_name, region)
- Add ORDER BY for ranking queries
- LIMIT results if requesting "top N" """
        
        return base
    
    def _clean_sql(self, sql: str) -> str:
        """Clean and extract SQL from LLM output."""
        sql = re.sub(r"```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"```\s*$", "", sql)
        sql = re.sub(r"^\s*(?:SQL Query:|Query:)\s*", "", sql, flags=re.IGNORECASE)
        
        match = re.search(r"((?:WITH|SELECT)\b.*?)(?:;|$)", sql, re.IGNORECASE | re.DOTALL)
        if match:
            sql = match.group(1)
        
        return sql.strip()
    
    def _validate_sql(self, sql: str):
        """Validate SQL for safety."""
        sql_upper = sql.upper()
        
        forbidden = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'TRUNCATE', 'ALTER', 'CREATE']
        for keyword in forbidden:
            if re.search(rf'\b{keyword}\b', sql_upper):
                raise ValueError(f"Forbidden keyword: {keyword}")
        
        if not re.match(r'^\s*(SELECT|WITH)\b', sql_upper):
            raise ValueError("Query must start with SELECT or WITH")
    
    def _retry_generate_sql(self, question: str, schema: str, error: str) -> str:
        """Retry SQL generation with error context."""
        retry_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a PostgreSQL expert. Fix the SQL query based on the error."),
            ("human", "Schema: {schema}\n\nQuestion: {question}\n\nPrevious error: {error}\n\nGenerate a valid SELECT query:")
        ])
        
        chain = retry_prompt | self.llm | StrOutputParser()
        raw_sql = chain.invoke({"schema": schema, "question": question, "error": error})
        
        sql = self._clean_sql(raw_sql)
        self._validate_sql(sql)
        return sql


# ============================================================================
# ANALYTICS AGENT
# ============================================================================

class AnalyticsAgent:
    """Creates visualizations and analytics."""
    
    def __init__(self, db_manager: DatabaseManager, output_dir: str = "charts"):
        self.db = db_manager
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def analyze(self, sql: str, question: str) -> Dict[str, Any]:
        """Execute query and create visualizations."""
        print(f"📊 Executing analytics query...")
        
        df = self.db.execute_query(sql)
        
        if df.empty:
            return {"success": False, "message": "No data returned", "data": df}
        
        print(f"✅ Retrieved {len(df)} rows")
        
        charts = self._create_charts(df, question)
        summary = self._generate_summary(df)
        
        return {
            "success": True,
            "data": df,
            "summary": summary,
            "charts": charts,
            "row_count": len(df)
        }
    
    def _create_charts(self, df: pd.DataFrame, question: str) -> List[str]:
        """Create appropriate charts based on data."""
        charts = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Try to convert string dates
        for col in categorical_cols:
            if 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols.append(col)
                    categorical_cols.remove(col)
                except:
                    pass
        
        # Time series chart
        if date_cols and numeric_cols:
            chart_path = self._plot_time_series(df, date_cols[0], numeric_cols[0])
            charts.append(chart_path)
        # Bar chart for categories
        elif categorical_cols and numeric_cols and len(df) <= 50:
            chart_path = self._plot_bar_chart(df, categorical_cols[0], numeric_cols[0])
            charts.append(chart_path)
        # Distribution
        elif len(numeric_cols) >= 1:
            chart_path = self._plot_distribution(df, numeric_cols[0])
            charts.append(chart_path)
        
        return charts
    
    def _plot_time_series(self, df: pd.DataFrame, date_col: str, value_col: str) -> str:
        """Create time series plot."""
        plt.figure(figsize=(12, 6))
        plt.plot(df[date_col], df[value_col], marker='o', linewidth=2)
        plt.xlabel(date_col, fontsize=12)
        plt.ylabel(value_col, fontsize=12)
        plt.title(f'Time Series: {value_col} over {date_col}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"timeseries_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Created time series chart: {filepath}")
        return filepath
    
    def _plot_bar_chart(self, df: pd.DataFrame, cat_col: str, value_col: str) -> str:
        """Create bar chart."""
        df_plot = df.nlargest(20, value_col) if len(df) > 20 else df
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(df_plot)), df_plot[value_col], color='steelblue')
        plt.xticks(range(len(df_plot)), df_plot[cat_col], rotation=45, ha='right')
        plt.xlabel(cat_col, fontsize=12)
        plt.ylabel(value_col, fontsize=12)
        plt.title(f'{value_col} by {cat_col}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        filename = f"bar_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Created bar chart: {filepath}")
        return filepath
    
    def _plot_distribution(self, df: pd.DataFrame, col: str) -> str:
        """Create distribution plot."""
        plt.figure(figsize=(10, 6))
        plt.hist(df[col].dropna(), bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Distribution of {col}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        filename = f"dist_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📉 Created distribution chart: {filepath}")
        return filepath
    
    def _generate_summary(self, df: pd.DataFrame) -> str:
        """Generate text summary of results."""
        summary = f"Retrieved {len(df)} rows with {len(df.columns)} columns.\n\n"
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary += "Numeric summary:\n"
            for col in numeric_cols[:3]:
                summary += f"  - {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}\n"
        
        summary += f"\nFirst 5 rows:\n{df.head().to_string()}\n"
        return summary


# ============================================================================
# FORECAST AGENT
# ============================================================================

class ForecastAgent:
    """Performs time series forecasting."""
    
    def __init__(self, db_manager: DatabaseManager, output_dir: str = "charts"):
        self.db = db_manager
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def forecast(self, sql: str, question: str, horizon: int = 30) -> Dict[str, Any]:
        """Execute query and perform forecasting."""
        print(f"🔮 Executing forecast query...")
        
        df = self.db.execute_query(sql)
        
        if df.empty:
            return {"success": False, "message": "No historical data available"}
        
        print(f"✅ Retrieved {len(df)} historical data points")
        
        date_col, value_col = self._identify_columns(df)
        
        if not date_col or not value_col:
            return {"success": False, "message": "Could not identify date and value columns"}
        
        df_ts = self._prepare_time_series(df, date_col, value_col)
        forecast_result = self._simple_forecast(df_ts, horizon)
        chart_path = self._plot_forecast(df_ts, forecast_result, value_col)
        metrics = self._calculate_metrics(df_ts, forecast_result)
        
        return {
            "success": True,
            "historical_data": df_ts,
            "forecast": forecast_result,
            "chart": chart_path,
            "metrics": metrics,
            "summary": self._generate_forecast_summary(df_ts, forecast_result)
        }
    
    def _identify_columns(self, df: pd.DataFrame) -> tuple:
        """Identify date and value columns."""
        date_col = None
        value_col = None
        
        for col in df.columns:
            if 'date' in col.lower() or df[col].dtype == 'datetime64[ns]':
                date_col = col
                break
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                if any(kw in col.lower() for kw in ['quantity', 'qty', 'sales', 'total', 'sum']):
                    value_col = col
                    break
            if not value_col:
                value_col = numeric_cols[0]
        
        return date_col, value_col
    
    def _prepare_time_series(self, df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
        """Prepare time series data."""
        df_ts = df[[date_col, value_col]].copy()
        df_ts[date_col] = pd.to_datetime(df_ts[date_col])
        df_ts = df_ts.sort_values(date_col)
        df_ts = df_ts.set_index(date_col)
        df_ts.columns = ['value']
        df_ts = df_ts.resample('D').sum().fillna(0)
        return df_ts
    
    def _simple_forecast(self, df_ts: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Simple forecasting using moving average and trend."""
        df_ts['ma_7'] = df_ts['value'].rolling(window=7, min_periods=1).mean()
        df_ts['ma_30'] = df_ts['value'].rolling(window=30, min_periods=1).mean()
        
        last_30_days = df_ts['value'].tail(30)
        if len(last_30_days) > 1:
            x = np.arange(len(last_30_days))
            y = last_30_days.values
            trend = np.polyfit(x, y, 1)
        else:
            trend = [0, last_30_days.mean()]
        
        last_date = df_ts.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')
        
        base_value = df_ts['ma_30'].iloc[-1]
        daily_trend = trend[0]
        
        forecast_values = [max(0, base_value + daily_trend * i) for i in range(1, horizon + 1)]
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'forecast': forecast_values
        }).set_index('date')
        
        return forecast_df
    
    def _plot_forecast(self, df_ts: pd.DataFrame, forecast_df: pd.DataFrame, value_name: str) -> str:
        """Plot historical data and forecast."""
        plt.figure(figsize=(14, 7))
        
        recent = df_ts.tail(90)
        plt.plot(recent.index, recent['value'], label='Historical', linewidth=2, color='steelblue')
        plt.plot(forecast_df.index, forecast_df['forecast'], label='Forecast', 
                linewidth=2, color='orange', linestyle='--', marker='o', markersize=4)
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel(value_name, fontsize=12)
        plt.title('Sales Forecast', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = f"forecast_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"🔮 Created forecast chart: {filepath}")
        return filepath
    
    def _calculate_metrics(self, df_ts: pd.DataFrame, forecast_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate forecast metrics."""
        recent_mean = df_ts['value'].tail(30).mean()
        forecast_mean = forecast_df['forecast'].mean()
        
        return {
            "recent_avg_daily": float(recent_mean),
            "forecast_avg_daily": float(forecast_mean),
            "forecast_total": float(forecast_df['forecast'].sum()),
            "trend": "increasing" if forecast_mean > recent_mean else "decreasing"
        }
    
    def _generate_forecast_summary(self, df_ts: pd.DataFrame, forecast_df: pd.DataFrame) -> str:
        """Generate forecast summary text."""
        metrics = self._calculate_metrics(df_ts, forecast_df)
        
        return f"""
FORECAST SUMMARY:
- Historical period: {df_ts.index[0].strftime('%Y-%m-%d')} to {df_ts.index[-1].strftime('%Y-%m-%d')}
- Forecast period: {forecast_df.index[0].strftime('%Y-%m-%d')} to {forecast_df.index[-1].strftime('%Y-%m-%d')}
- Recent average (daily): {metrics['recent_avg_daily']:.2f}
- Forecast average (daily): {metrics['forecast_avg_daily']:.2f}
- Forecast total: {metrics['forecast_total']:.2f}
- Trend: {metrics['trend']}
"""


# ============================================================================
# INVENTORY OPTIMIZATION AGENT
# ============================================================================

class InventoryOptimizationAgent:
    """
    Intelligent inventory management agent that:
    1. Analyzes current stock levels
    2. Compares with forecast demand
    3. Calculates reorder points and safety stock
    4. Recommends restocking or transfers based on proximity
    """
    
    def __init__(self, 
                 db_manager: DatabaseManager,
                 forecast_agent: 'ForecastAgent',
                 output_dir: str = "charts"):
        self.db = db_manager
        self.forecast_agent = forecast_agent
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configuration parameters
        self.service_level = 0.95  # 95% service level
        self.lead_time_days = 7    # Default lead time
        self.max_transfer_distance_km = 200  # Max distance for transfer
    
    def optimize_inventory(self, 
                          question: str,
                          product_code: Optional[str] = None,
                          branch_code: Optional[int] = None,
                          horizon_days: int = 30) -> Dict[str, Any]:
        """
        Main optimization workflow:
        1. Get forecast demand
        2. Get current inventory
        3. Calculate metrics (ROP, safety stock, EOQ)
        4. Find optimization opportunities
        5. Generate recommendations
        """
        print(f"🎯 Executing inventory optimization...")
        
        try:
            # Step 1: Get forecast demand
            print("📌 Step 1: Getting demand forecast...")
            forecast_data = self._get_forecast_data(product_code, branch_code, horizon_days)
            
            if not forecast_data:
                return {
                    "success": False,
                    "message": "Could not generate forecast for demand prediction"
                }
            
            # Step 2: Get current inventory
            print("📌 Step 2: Analyzing current inventory...")
            inventory_data = self._get_current_inventory(product_code, branch_code)
            
            if inventory_data.empty:
                return {
                    "success": False,
                    "message": "No inventory data found"
                }
            
            # Step 3: Calculate inventory metrics
            print("📌 Step 3: Calculating inventory metrics...")
            recommendations = self._generate_recommendations(
                inventory_data, 
                forecast_data, 
                horizon_days
            )
            
            # Step 4: Find transfer opportunities
            print("📌 Step 4: Finding transfer opportunities...")
            transfer_opportunities = self._find_transfer_opportunities(
                recommendations, 
                product_code
            )
            
            # Step 5: Generate comprehensive plan
            plan = self._create_action_plan(recommendations, transfer_opportunities)
            
            # Step 6: Create visualization
            chart_path = self._plot_inventory_optimization(
                inventory_data, 
                forecast_data, 
                recommendations
            )
            
            print(f"✅ Optimization complete: {len(plan['actions'])} actions recommended")
            
            return {
                "success": True,
                "forecast_data": forecast_data,
                "inventory_data": inventory_data,
                "recommendations": recommendations,
                "transfer_opportunities": transfer_opportunities,
                "action_plan": plan,
                "chart": chart_path,
                "summary": self._generate_summary(plan)
            }
            
        except Exception as e:
            print(f"❌ Optimization error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "message": f"Optimization failed: {str(e)}"
            }
    
    def _get_forecast_data(self, 
                          product_code: Optional[str],
                          branch_code: Optional[int],
                          horizon_days: int) -> Optional[Dict]:
        """Get forecast demand using ForecastAgent."""
        # Build forecast query
        sql = f"""
        SELECT date, SUM(quantity) as total_qty
        FROM sales
        WHERE date >= CURRENT_DATE - INTERVAL '90 days'
        """
        
        conditions = []
        if product_code:
            conditions.append(f"AND product_code = '{product_code}'")
        if branch_code:
            conditions.append(f"AND branch_code = {branch_code}")
        
        if conditions:
            sql += " " + " ".join(conditions)
        
        sql += " GROUP BY date ORDER BY date"
        
        # Get forecast from ForecastAgent
        try:
            result = self.forecast_agent.forecast(sql, "forecast for optimization", horizon_days)
            if result.get('success'):
                return {
                    'forecast_df': result['forecast'],
                    'historical_df': result['historical_data'],
                    'metrics': result['metrics']
                }
        except Exception as e:
            print(f"⚠️ Forecast failed: {e}")
        
        return None
    
    def _get_current_inventory(self, 
                               product_code: Optional[str],
                               branch_code: Optional[int]) -> pd.DataFrame:
        """Get current inventory levels from database."""
        sql = """
        SELECT 
            i.product_code,
            i.branch_code,
            b.branch_name,
            b.region,
            i.product_name,
            i.quantity as current_stock,
            i.unit
        FROM inventory i
        JOIN branch b ON i.branch_code = b.branch_code
        WHERE 1=1
        """
        
        if product_code:
            sql += f" AND i.product_code = '{product_code}'"
        if branch_code:
            sql += f" AND i.branch_code = {branch_code}"
        
        sql += " ORDER BY i.branch_code, i.product_code"
        
        return self.db.execute_query(sql)
    
    def _calculate_safety_stock(self, avg_demand: float, std_demand: float) -> float:
        """
        Calculate safety stock using statistical method.
        Safety Stock = Z * σ * √LT
        where Z = service level factor, σ = demand std, LT = lead time
        """
        from scipy import stats
        z_score = stats.norm.ppf(self.service_level)
        safety_stock = z_score * std_demand * np.sqrt(self.lead_time_days)
        return max(0, safety_stock)
    
    def _calculate_rop(self, avg_demand: float, safety_stock: float) -> float:
        """
        Calculate Reorder Point (ROP).
        ROP = (Average Daily Demand × Lead Time) + Safety Stock
        """
        rop = (avg_demand * self.lead_time_days) + safety_stock
        return max(0, rop)
    
    def _calculate_eoq(self, annual_demand: float, ordering_cost: float = 1000, 
                      holding_cost: float = 50) -> float:
        """
        Calculate Economic Order Quantity (EOQ).
        EOQ = √((2 × D × S) / H)
        where D = annual demand, S = ordering cost, H = holding cost
        """
        if annual_demand <= 0:
            return 0
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        return max(0, eoq)
    
    def _generate_recommendations(self, 
                                 inventory_data: pd.DataFrame,
                                 forecast_data: Dict,
                                 horizon_days: int) -> pd.DataFrame:
        """Generate inventory recommendations for each product-branch combination."""
        
        forecast_df = forecast_data['forecast_df']
        historical_df = forecast_data['historical_df']
        
        # Calculate demand statistics from historical data
        avg_daily_demand = historical_df['value'].mean()
        std_daily_demand = historical_df['value'].std()
        total_forecast_demand = forecast_df['forecast'].sum()
        
        recommendations = []
        
        for idx, row in inventory_data.iterrows():
            current_stock = row['current_stock']
            
            # Calculate metrics
            safety_stock = self._calculate_safety_stock(avg_daily_demand, std_daily_demand)
            rop = self._calculate_rop(avg_daily_demand, safety_stock)
            annual_demand = avg_daily_demand * 365
            eoq = self._calculate_eoq(annual_demand)
            
            # Calculate expected stock after forecast period
            expected_stock_after_period = current_stock - total_forecast_demand
            
            # Determine action needed
            if current_stock < rop:
                action = "URGENT_RESTOCK"
                priority = "HIGH"
                quantity_needed = eoq
            elif expected_stock_after_period < safety_stock:
                action = "RESTOCK"
                priority = "MEDIUM"
                quantity_needed = eoq
            elif current_stock > (rop + eoq * 2):
                action = "SURPLUS"
                priority = "LOW"
                quantity_needed = 0
            else:
                action = "OK"
                priority = "LOW"
                quantity_needed = 0
            
            recommendations.append({
                'product_code': row['product_code'],
                'branch_code': row['branch_code'],
                'branch_name': row['branch_name'],
                'region': row['region'],
                'product_name': row['product_name'],
                'current_stock': current_stock,
                'avg_daily_demand': avg_daily_demand,
                'forecast_demand_30d': total_forecast_demand,
                'safety_stock': safety_stock,
                'reorder_point': rop,
                'eoq': eoq,
                'expected_stock_after_30d': expected_stock_after_period,
                'action': action,
                'priority': priority,
                'quantity_needed': quantity_needed,
                'unit': row['unit']
            })
        
        return pd.DataFrame(recommendations)
    
    def _find_transfer_opportunities(self, 
                                    recommendations: pd.DataFrame,
                                    product_code: Optional[str]) -> List[Dict]:
        """
        Find opportunities to transfer stock from surplus branches to deficit branches.
        Uses branch_distance table to find nearby branches.
        """
        if recommendations.empty:
            return []
        
        # Separate surplus and deficit branches
        surplus = recommendations[recommendations['action'] == 'SURPLUS'].copy()
        deficit = recommendations[recommendations['action'].isin(['URGENT_RESTOCK', 'RESTOCK'])].copy()
        
        if surplus.empty or deficit.empty:
            return []
        
        transfer_opportunities = []
        
        for _, deficit_row in deficit.iterrows():
            deficit_branch = deficit_row['branch_code']
            needed_qty = deficit_row['quantity_needed']
            
            # Find nearby branches with surplus
            nearby_query = f"""
            SELECT 
                bd.branch_code_1 as source_branch,
                bd.branch_code_2 as dest_branch,
                bd.distance_km,
                b.branch_name as source_branch_name
            FROM branch_distance bd
            JOIN branch b ON bd.branch_code_1 = b.branch_code
            WHERE bd.branch_code_2 = {deficit_branch}
                AND bd.distance_km <= {self.max_transfer_distance_km}
            ORDER BY bd.distance_km ASC
            """
            
            try:
                nearby_branches = self.db.execute_query(nearby_query)
                
                for _, nearby in nearby_branches.iterrows():
                    source_branch = nearby['source_branch']
                    
                    # Check if source branch has surplus for this product
                    surplus_match = surplus[
                        (surplus['branch_code'] == source_branch) &
                        (surplus['product_code'] == deficit_row['product_code'])
                    ]
                    
                    if not surplus_match.empty:
                        surplus_row = surplus_match.iloc[0]
                        available_qty = surplus_row['current_stock'] - surplus_row['reorder_point']
                        
                        if available_qty > 0:
                            transfer_qty = min(available_qty, needed_qty)
                            
                            transfer_opportunities.append({
                                'product_code': deficit_row['product_code'],
                                'product_name': deficit_row['product_name'],
                                'source_branch_code': source_branch,
                                'source_branch_name': nearby['source_branch_name'],
                                'dest_branch_code': deficit_branch,
                                'dest_branch_name': deficit_row['branch_name'],
                                'distance_km': nearby['distance_km'],
                                'transfer_quantity': transfer_qty,
                                'unit': deficit_row['unit'],
                                'cost_saving': 'Avoid external purchase',
                                'priority': deficit_row['priority']
                            })
                            
                            # Update needed quantity
                            needed_qty -= transfer_qty
                            if needed_qty <= 0:
                                break
                
            except Exception as e:
                print(f"⚠️ Error finding transfers for branch {deficit_branch}: {e}")
                continue
        
        return transfer_opportunities
    
    def _create_action_plan(self, 
                           recommendations: pd.DataFrame,
                           transfer_opportunities: List[Dict]) -> Dict[str, Any]:
        """Create comprehensive action plan with prioritized actions."""
        
        actions = []
        
        # Add restock actions (excluding those that can be fulfilled by transfers)
        transferred_branches = {(t['dest_branch_code'], t['product_code']) 
                              for t in transfer_opportunities}
        
        for _, row in recommendations.iterrows():
            if row['action'] in ['URGENT_RESTOCK', 'RESTOCK']:
                branch_product = (row['branch_code'], row['product_code'])
                
                # Check if partially/fully covered by transfers
                transfer_qty = sum(
                    t['transfer_quantity'] 
                    for t in transfer_opportunities 
                    if t['dest_branch_code'] == row['branch_code'] 
                    and t['product_code'] == row['product_code']
                )
                
                remaining_qty = row['quantity_needed'] - transfer_qty
                
                if remaining_qty > 0:
                    actions.append({
                        'action_type': 'RESTOCK',
                        'priority': row['priority'],
                        'branch_code': row['branch_code'],
                        'branch_name': row['branch_name'],
                        'product_code': row['product_code'],
                        'product_name': row['product_name'],
                        'quantity': remaining_qty,
                        'unit': row['unit'],
                        'reason': f"Current: {row['current_stock']}, ROP: {row['reorder_point']:.0f}, Forecast demand: {row['forecast_demand_30d']:.0f}",
                        'estimated_cost': 'TBD'
                    })
        
        # Add transfer actions
        for transfer in transfer_opportunities:
            actions.append({
                'action_type': 'TRANSFER',
                'priority': transfer['priority'],
                'source_branch_code': transfer['source_branch_code'],
                'source_branch_name': transfer['source_branch_name'],
                'dest_branch_code': transfer['dest_branch_code'],
                'dest_branch_name': transfer['dest_branch_name'],
                'product_code': transfer['product_code'],
                'product_name': transfer['product_name'],
                'quantity': transfer['transfer_quantity'],
                'unit': transfer['unit'],
                'distance_km': transfer['distance_km'],
                'reason': f"Transfer from surplus to deficit branch ({transfer['distance_km']:.1f} km)",
                'estimated_cost': f"Transport cost for {transfer['distance_km']:.1f} km"
            })
        
        # Sort by priority
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        actions.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        # Calculate summary statistics
        total_restock = sum(a['quantity'] for a in actions if a['action_type'] == 'RESTOCK')
        total_transfer = sum(a['quantity'] for a in actions if a['action_type'] == 'TRANSFER')
        
        return {
            'actions': actions,
            'summary': {
                'total_actions': len(actions),
                'restock_actions': len([a for a in actions if a['action_type'] == 'RESTOCK']),
                'transfer_actions': len([a for a in actions if a['action_type'] == 'TRANSFER']),
                'total_restock_quantity': total_restock,
                'total_transfer_quantity': total_transfer,
                'high_priority_actions': len([a for a in actions if a['priority'] == 'HIGH'])
            }
        }
    
    def _plot_inventory_optimization(self, 
                                    inventory_data: pd.DataFrame,
                                    forecast_data: Dict,
                                    recommendations: pd.DataFrame) -> str:
        """Create visualization for inventory optimization."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Plot 1: Current Stock vs ROP
        ax1 = axes[0, 0]
        branches = recommendations['branch_name'].head(10)
        current_stock = recommendations['current_stock'].head(10)
        rop = recommendations['reorder_point'].head(10)
        safety_stock = recommendations['safety_stock'].head(10)
        
        x = np.arange(len(branches))
        width = 0.25
        
        ax1.bar(x - width, current_stock, width, label='Current Stock', color='steelblue')
        ax1.bar(x, rop, width, label='Reorder Point', color='orange')
        ax1.bar(x + width, safety_stock, width, label='Safety Stock', color='green')
        
        ax1.set_xlabel('Branch', fontsize=10)
        ax1.set_ylabel('Quantity', fontsize=10)
        ax1.set_title('Current Stock vs ROP & Safety Stock (Top 10 Branches)', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(branches, rotation=45, ha='right', fontsize=8)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Action Distribution
        ax2 = axes[0, 1]
        action_counts = recommendations['action'].value_counts()
        colors = {'OK': 'green', 'RESTOCK': 'orange', 'URGENT_RESTOCK': 'red', 'SURPLUS': 'blue'}
        ax2.pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%',
                colors=[colors.get(action, 'gray') for action in action_counts.index])
        ax2.set_title('Inventory Action Distribution', fontsize=12, fontweight='bold')
        
        # Plot 3: Forecast vs Expected Stock
        ax3 = axes[1, 0]
        forecast_df = forecast_data['forecast_df']
        ax3.plot(forecast_df.index, forecast_df['forecast'], 
                label='Forecasted Demand', linewidth=2, color='orange', marker='o')
        ax3.set_xlabel('Date', fontsize=10)
        ax3.set_ylabel('Quantity', fontsize=10)
        ax3.set_title('30-Day Demand Forecast', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Priority Distribution
        ax4 = axes[1, 1]
        priority_counts = recommendations[recommendations['action'] != 'OK']['priority'].value_counts()
        priority_colors = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'yellow'}
        ax4.bar(priority_counts.index, priority_counts.values,
                color=[priority_colors.get(p, 'gray') for p in priority_counts.index])
        ax4.set_xlabel('Priority', fontsize=10)
        ax4.set_ylabel('Count', fontsize=10)
        ax4.set_title('Action Priority Distribution', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filename = f"inventory_opt_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Created inventory optimization chart: {filepath}")
        return filepath
    
    def _generate_summary(self, plan: Dict) -> str:
        """Generate text summary of the action plan."""
        summary = plan['summary']
        
        text = f"""
INVENTORY OPTIMIZATION SUMMARY:
================================

Total Actions Recommended: {summary['total_actions']}
- Restock Orders: {summary['restock_actions']} (Total Qty: {summary['total_restock_quantity']:.0f})
- Internal Transfers: {summary['transfer_actions']} (Total Qty: {summary['total_transfer_quantity']:.0f})
- High Priority Actions: {summary['high_priority_actions']}

KEY ACTIONS:
"""
        
        for action in plan['actions'][:10]:  # Top 10 actions
            if action['action_type'] == 'RESTOCK':
                text += f"\n📦 RESTOCK [{action['priority']}]: {action['product_name']}"
                text += f"\n   Branch: {action['branch_name']}"
                text += f"\n   Quantity: {action['quantity']:.0f} {action['unit']}"
                text += f"\n   Reason: {action['reason']}\n"
            else:  # TRANSFER
                text += f"\n🚚 TRANSFER [{action['priority']}]: {action['product_name']}"
                text += f"\n   From: {action['source_branch_name']} → To: {action['dest_branch_name']}"
                text += f"\n   Quantity: {action['quantity']:.0f} {action['unit']}"
                text += f"\n   Distance: {action['distance_km']:.1f} km\n"
        
        if len(plan['actions']) > 10:
            text += f"\n... and {len(plan['actions']) - 10} more actions.\n"
        
        return text


# ============================================================================
# ORCHESTRATOR AGENT
# ============================================================================

class OrchestratorAgent:
    """Main orchestrator that coordinates all agents."""
    
    def __init__(self, 
                 db_manager: DatabaseManager,
                 memory: MemoryManager,
                 llm_provider: LLMProvider):
        
        self.db_manager = db_manager
        self.memory = memory
        self.llm_provider = llm_provider
        
        # Initialize all agents
        self.schema_agent = SchemaAgent(db_manager, memory)
        self.intent_agent = IntentAgent(llm_provider)
        self.sql_agent = SQLAgent(llm_provider, self.schema_agent)
        self.analytics_agent = AnalyticsAgent(db_manager)
        self.forecast_agent = ForecastAgent(db_manager)
        self.inventory_agent = InventoryOptimizationAgent(db_manager, self.forecast_agent)
        
        print("✅ OrchestratorAgent initialized with all sub-agents (including Inventory Optimization)")
    
    def process_query(self, question: str) -> Dict[str, Any]:
        """Main entry point: process user question through the agent pipeline."""
        print(f"\n{'='*80}")
        print(f"🤖 Processing question: {question}")
        print(f"{'='*80}\n")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Classify intent
            print("📌 Step 1: Intent Classification")
            intent = self.intent_agent.classify(question)
            print(f"   → Intent: {intent}\n")
            
            # Step 2: Handle different intents
            if intent == "INVENTORY_OPTIMIZATION":
                # For inventory optimization, we don't need SQL generation first
                print(f"📌 Step 2-3: Processing with Inventory Optimization Agent")
                result = self.inventory_agent.optimize_inventory(question)
                sql = "N/A - Inventory optimization uses multiple queries internally"
            else:
                # Step 2: Generate SQL for FORECAST and ANALYTICS
                print("📌 Step 2: SQL Generation")
                sql = self.sql_agent.generate_sql(question, intent)
                print(f"   → SQL: {sql[:200]}...\n")
                
                # Step 3: Route to appropriate agent
                print(f"📌 Step 3: Processing with {intent} Agent")
                
                if intent == "FORECAST":
                    result = self.forecast_agent.forecast(sql, question)
                else:
                    result = self.analytics_agent.analyze(sql, question)
            
            # Step 4: Store in memory
            entry = ConversationEntry(
                timestamp=datetime.now(),
                question=question,
                intent=intent,
                sql_query=sql,
                result_summary=result.get('summary', ''),
                charts=result.get('charts', []) or [result.get('chart', '')]
            )
            self.memory.add_entry(entry)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            final_result = {
                "success": result.get('success', True),
                "question": question,
                "intent": intent,
                "sql": sql,
                "result": result,
                "elapsed_seconds": elapsed
            }
            
            print(f"\n✅ Completed in {elapsed:.2f}s")
            return final_result
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "question": question,
                "error": str(e),
                "elapsed_seconds": (datetime.now() - start_time).total_seconds()
            }
    
    def get_conversation_history(self) -> List[ConversationEntry]:
        """Get conversation history from memory."""
        return self.memory.conversation_history
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.conversation_history.clear()
        print("🗑️ Memory cleared")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def initialize_system() -> OrchestratorAgent:
    """Initialize the complete multi-agent system."""
    print("\n" + "="*80)
    print("🚀 Initializing Multi-Agent System...")
    print("="*80 + "\n")
    
    db_manager = DatabaseManager()
    memory = MemoryManager()
    llm_provider = LLMProvider()
    
    orchestrator = OrchestratorAgent(
        db_manager=db_manager,
        memory=memory,
        llm_provider=llm_provider
    )
    
    print("\n" + "="*80)
    print("🎉 Multi-Agent System Ready!")
    print("="*80)
    
    return orchestrator


def display_conversation_history(orchestrator: OrchestratorAgent):
    """Display conversation history in a nice format."""
    history = orchestrator.get_conversation_history()
    
    if not history:
        print("No conversation history yet.")
        return
    
    print("\n" + "="*80)
    print("📜 CONVERSATION HISTORY")
    print("="*80)
    
    for i, entry in enumerate(history, 1):
        print(f"\n[{i}] {entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Q: {entry.question}")
        print(f"Intent: {entry.intent}")
        print(f"SQL: {entry.sql_query[:100]}...")
        if entry.charts:
            print(f"Charts: {len(entry.charts)} created")
        print("-" * 80)


def export_results_to_excel(result: Dict[str, Any], filename: str = "export.xlsx"):
    """Export query results to Excel."""
    if not result.get('success'):
        print("❌ Cannot export: query was not successful")
        return
    
    data = result['result'].get('data') or result['result'].get('historical_data')
    
    if data is None or (isinstance(data, pd.DataFrame) and data.empty):
        print("❌ No data to export")
        return
    
    try:
        data.to_excel(filename, index=True)
        print(f"✅ Exported to {filename}")
    except Exception as e:
        print(f"❌ Export failed: {e}")


# ============================================================================
# MAIN USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize system
    orchestrator = initialize_system()
    
    # # Example 1: Analytics
    # print("\n" + "="*80)
    # print("📊 Example 1: Analytics Query")
    # print("="*80)
    # result1 = orchestrator.process_query(
    #     "Top 10 sản phẩm bán chạy nhất trong tháng này"
    # )
    
    # # Example 2: Forecast
    # print("\n" + "="*80)
    # print("🔮 Example 2: Forecast Query")
    # print("="*80)
    # result2 = orchestrator.process_query(
    #     "Dự báo doanh số bán hàng cho 30 ngày tới"
    # )
    
    # Example 3: Inventory Optimization (NEW!)
    print("\n" + "="*80)
    print("🎯 Example 3: Inventory Optimization Query")
    print("="*80)
    result3 = orchestrator.process_query(
        "Tối ưu hóa tồn kho: kiểm tra sản phẩm nào cần nhập hàng và có thể chuyển kho không"
    )
    
    if result3.get('success'):
        print("\n" + "="*80)
        print("📋 INVENTORY OPTIMIZATION RESULTS")
        print("="*80)
        print(result3['result']['summary'])
        
        if result3['result'].get('action_plan'):
            plan = result3['result']['action_plan']
            print(f"\n💡 {plan['summary']['total_actions']} total actions recommended")
            print(f"   - {plan['summary']['restock_actions']} restock orders")
            print(f"   - {plan['summary']['transfer_actions']} transfer opportunities")
    
    # Show history
    display_conversation_history(orchestrator)


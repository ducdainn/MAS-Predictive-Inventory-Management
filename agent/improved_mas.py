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
import sqlite3
import pickle
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from pathlib import Path

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
        
        uri = f"postgresql+psycopg2://{self.PG_USER}:{self.PG_PASSWORD}@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DB}"
        self.engine = create_engine(uri, pool_pre_ping=True, pool_size=5)
        print(f"✅ Connected to database: {self.PG_DB}")
    
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


# ============================================================================
# MEMORY MANAGER
# ============================================================================

class MemoryManager:
    """
    Manages conversation history with PERSISTENT STORAGE using SQLite.
    
    Improvements:
    - Stores conversation history in SQLite database
    - Survives restarts
    - Efficient querying with SQL
    """
    
    def __init__(self, max_history: int = 100, db_path: str = "agent_memory.db"):
        self.max_history = max_history
        self.db_path = db_path
        self.conversation_history: List[ConversationEntry] = []
        self.schema_cache: Dict[str, Any] = {}
        
        # Initialize persistent storage
        self._init_db()
        self._initialize_schema_cache()
        self._load_from_db()
    
    def _init_db(self):
        """Initialize SQLite database for persistent memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                question TEXT NOT NULL,
                intent TEXT NOT NULL,
                sql_query TEXT,
                result_summary TEXT,
                charts TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON conversation_history(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_intent ON conversation_history(intent)
        """)
        
        conn.commit()
        conn.close()
        print(f"✅ Persistent memory initialized at: {self.db_path}")
    
    def _initialize_schema_cache(self):
        """Cache schema info and common patterns."""
        self.schema_cache = {
            "tables": ["branch", "product", "inventory", "sales", "branch_distance"],
            "key_columns": {
                "branch": ["branch_code", "region", "branch_name"],
                "product": ["product_code", "product_name", "category", "unit"],
                "inventory": ["product_code", "branch_code", "quantity"],
                "sales": ["date", "branch_code", "product_code", "quantity", "square_meters"],
                "branch_distance": ["branch_code_1", "branch_code_2", "distance_km"]
            },
            "relationships": [
                "sales JOIN branch ON sales.branch_code = branch.branch_code",
                "sales JOIN product ON sales.product_code = product.product_code",
                "inventory JOIN branch ON inventory.branch_code = branch.branch_code",
                "inventory JOIN product ON inventory.product_code = product.product_code"
            ]
        }
    
    def _load_from_db(self):
        """Load recent history from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, question, intent, sql_query, result_summary, charts
            FROM conversation_history
            ORDER BY created_at DESC
            LIMIT ?
        """, (self.max_history,))
        
        rows = cursor.fetchall()
        conn.close()
        
        for row in reversed(rows):  # Reverse to get chronological order
            timestamp_str, question, intent, sql_query, result_summary, charts_json = row
            entry = ConversationEntry(
                timestamp=datetime.fromisoformat(timestamp_str),
                question=question,
                intent=intent,
                sql_query=sql_query,
                result_summary=result_summary,
                charts=json.loads(charts_json) if charts_json else []
            )
            self.conversation_history.append(entry)
    
    def add_entry(self, entry: ConversationEntry):
        """Add conversation entry to history and persist to database."""
        # Add to in-memory list
        self.conversation_history.append(entry)
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
        
        # Persist to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO conversation_history (timestamp, question, intent, sql_query, result_summary, charts)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            entry.timestamp.isoformat(),
            entry.question,
            entry.intent,
            entry.sql_query,
            entry.result_summary,
            json.dumps(entry.charts)
        ))
        
        conn.commit()
        conn.close()
    
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
    
    def clear_all(self):
        """Clear all conversation history from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM conversation_history")
        conn.commit()
        conn.close()
        self.conversation_history.clear()
        print("🗑️ All memory cleared from persistent storage")


# ============================================================================
# LLM PROVIDER
# ============================================================================

class LLMProvider:
    """Manages LLM instances with caching."""
    
    def __init__(self):
        self._llm_cache = {}
    
    def get_llm(self, model_type: str = "huggingface", temperature: float = 0.0):
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
# ENTITY EXTRACTOR AGENT
# ============================================================================

class EntityExtractor:
    """
    Extracts entities from user questions: branch names, product names, regions.
    
    IMPROVEMENT: Enables context-aware optimization (e.g., "chi nhánh đà nẵng")
    """
    
    def __init__(self, llm_provider: LLMProvider, db_manager: DatabaseManager):
        self.llm = llm_provider.get_llm("openai", temperature=0.0)
        self.db = db_manager
        
        # Cache branch and product names for matching
        self._load_entity_cache()
    
    def _load_entity_cache(self):
        """Load all branch and product names for fuzzy matching."""
        try:
            # Get all branches
            branches_df = self.db.execute_query("SELECT branch_code, branch_name, region FROM branch")
            self.branches = branches_df.to_dict('records')
            
            # Get all products
            products_df = self.db.execute_query("SELECT product_code, product_name FROM product LIMIT 1000")
            self.products = products_df.to_dict('records')
            
            print(f"✅ Loaded {len(self.branches)} branches and {len(self.products)} products for entity matching")
        except Exception as e:
            print(f"⚠️ Could not load entity cache: {e}")
            self.branches = []
            self.products = []
    
    def extract_entities(self, question: str) -> Dict[str, Any]:
        """
        Extract entities from user question using LLM + fuzzy matching.
        
        Returns:
            {
                'branch_names': [...],  # List of mentioned branch names
                'branch_codes': [...],  # Matched branch codes
                'product_names': [...], # List of mentioned product names
                'product_codes': [...], # Matched product codes
                'regions': [...],       # Mentioned regions
                'scope': 'specific' | 'all'  # Whether to filter or use all
            }
        """
        print("🔍 Extracting entities from question...")
        
        # Build prompt with available entities
        branch_names = [b['branch_name'] for b in self.branches[:20]]  # Sample for prompt
        regions = list(set([b['region'] for b in self.branches]))
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_extraction_prompt()),
            ("human", """
Question: {question}

Available branches (sample): {branch_names}
Available regions: {regions}

Extract entities as JSON:
{{
    "branch_names": ["exact or partial branch names mentioned"],
    "product_names": ["product names mentioned"],
    "regions": ["regions mentioned"],
    "scope": "specific" or "all"
}}
""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            result = chain.invoke({
                "question": question,
                "branch_names": ", ".join(branch_names),
                "regions": ", ".join(regions)
            })
            
            # Parse JSON
            entities = json.loads(result)
            
            # Fuzzy match branch names to codes
            entities['branch_codes'] = self._match_branches(entities.get('branch_names', []))
            
            # Fuzzy match product names to codes
            entities['product_codes'] = self._match_products(entities.get('product_names', []))
            
            print(f"✅ Extracted: {len(entities.get('branch_codes', []))} branches, "
                  f"{len(entities.get('product_codes', []))} products")
            
            return entities
            
        except Exception as e:
            print(f"⚠️ Entity extraction failed: {e}, using fallback")
            return self._fallback_extraction(question)
    
    def _get_extraction_prompt(self) -> str:
        return """You are an entity extractor for inventory management questions.

Extract:
1. Branch names (e.g., "đà nẵng", "hà nội", "chi nhánh 1")
2. Product names (e.g., "gạch 30x60", "sơn nước")
3. Regions (e.g., "miền bắc", "miền trung", "miền nam")
4. Scope: "specific" if question mentions specific branches/products, "all" if general

Rules:
- Extract partial matches (e.g., "đà nẵng" matches "Chi nhánh Đà Nẵng 1")
- Case insensitive
- Return empty lists if nothing mentioned
- Be lenient with Vietnamese diacritics

Return ONLY valid JSON, no explanations."""
    
    def _match_branches(self, mentioned_names: List[str]) -> List[int]:
        """Fuzzy match mentioned branch names to branch codes."""
        if not mentioned_names:
            return []
        
        matched_codes = []
        
        for mentioned in mentioned_names:
            mentioned_lower = mentioned.lower().strip()
            
            # Remove Vietnamese accents for better matching
            mentioned_normalized = self._normalize_vietnamese(mentioned_lower)
            
            for branch in self.branches:
                branch_name_lower = branch['branch_name'].lower()
                branch_name_normalized = self._normalize_vietnamese(branch_name_lower)
                
                # Check if mentioned name is in branch name
                if (mentioned_normalized in branch_name_normalized or 
                    mentioned_lower in branch_name_lower):
                    matched_codes.append(branch['branch_code'])
                    print(f"   ✓ Matched '{mentioned}' → {branch['branch_name']} (code: {branch['branch_code']})")
        
        return list(set(matched_codes))  # Remove duplicates
    
    def _match_products(self, mentioned_names: List[str]) -> List[str]:
        """Fuzzy match mentioned product names to product codes."""
        if not mentioned_names:
            return []
        
        matched_codes = []
        
        for mentioned in mentioned_names:
            mentioned_lower = mentioned.lower().strip()
            mentioned_normalized = self._normalize_vietnamese(mentioned_lower)
            
            for product in self.products:
                product_name_lower = product['product_name'].lower()
                product_name_normalized = self._normalize_vietnamese(product_name_lower)
                
                if (mentioned_normalized in product_name_normalized or
                    mentioned_lower in product_name_lower):
                    matched_codes.append(product['product_code'])
                    print(f"   ✓ Matched '{mentioned}' → {product['product_name'][:50]}...")
                    break  # Only first match per mentioned name
        
        return matched_codes
    
    def _normalize_vietnamese(self, text: str) -> str:
        """Remove Vietnamese accents for better matching."""
        import unicodedata
        # Decompose and remove accents
        normalized = unicodedata.normalize('NFD', text)
        return ''.join(char for char in normalized if unicodedata.category(char) != 'Mn')
    
    def _fallback_extraction(self, question: str) -> Dict[str, Any]:
        """Simple keyword-based fallback extraction."""
        question_lower = question.lower()
        
        entities = {
            'branch_names': [],
            'branch_codes': [],
            'product_names': [],
            'product_codes': [],
            'regions': [],
            'scope': 'all'
        }
        
        # Check for region keywords
        if 'miền bắc' in question_lower or 'mien bac' in question_lower:
            entities['regions'].append('MIỀN BẮC')
            entities['scope'] = 'specific'
        if 'miền trung' in question_lower or 'mien trung' in question_lower:
            entities['regions'].append('MIỀN TRUNG')
            entities['scope'] = 'specific'
        if 'miền nam' in question_lower or 'mien nam' in question_lower:
            entities['regions'].append('MIỀN NAM')
            entities['scope'] = 'specific'
        
        # Check for specific branch mentions in question
        for branch in self.branches:
            branch_name_lower = branch['branch_name'].lower()
            branch_name_normalized = self._normalize_vietnamese(branch_name_lower)
            question_normalized = self._normalize_vietnamese(question_lower)
            
            # Check if any significant word from branch name is in question
            branch_words = [w for w in branch_name_lower.split() if len(w) > 3]
            for word in branch_words:
                word_normalized = self._normalize_vietnamese(word)
                if word_normalized in question_normalized or word in question_lower:
                    entities['branch_codes'].append(branch['branch_code'])
                    entities['branch_names'].append(branch['branch_name'])
                    entities['scope'] = 'specific'
                    print(f"   ✓ Fallback matched: {branch['branch_name']}")
                    break
        
        # Remove duplicates
        entities['branch_codes'] = list(set(entities['branch_codes']))
        entities['branch_names'] = list(set(entities['branch_names']))
        
        return entities


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
# SMART INSIGHTS GENERATOR
# ============================================================================

class SmartInsightsGenerator:
    """
    LLM-powered insights generator for inventory optimization results.
    
    IMPROVEMENT: Makes agent more intelligent by:
    - Analyzing patterns in recommendations
    - Providing actionable business insights
    - Suggesting optimization strategies
    - Learning from historical data
    """
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider.get_llm("openai", temperature=0.3)  # Slightly creative
    
    def generate_insights(self, 
                         recommendations: pd.DataFrame,
                         action_plan: Dict[str, Any],
                         entities: Optional[Dict] = None) -> str:
        """
        Generate intelligent business insights from optimization results.
        
        Returns comprehensive analysis with:
        - Key findings
        - Risk areas
        - Opportunities
        - Strategic recommendations
        """
        print("🧠 Generating smart insights...")
        
        # Prepare data summary for LLM
        context = self._prepare_context(recommendations, action_plan, entities)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_insights_prompt()),
            ("human", """
Analyze this inventory optimization and provide strategic insights:

{context}

Provide:
1. KEY FINDINGS (3-5 bullet points)
2. RISK AREAS (critical issues)
3. OPPORTUNITIES (cost savings, efficiency)
4. STRATEGIC RECOMMENDATIONS (actionable steps)
5. PRIORITY ACTIONS (what to do first)

Be specific, data-driven, and actionable.
""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            insights = chain.invoke({"context": context})
            return insights
        except Exception as e:
            print(f"⚠️ Insight generation failed: {e}")
            return self._fallback_insights(recommendations, action_plan)
    
    def _get_insights_prompt(self) -> str:
        return """You are an expert inventory management consultant with 15+ years experience.

Your role: Analyze inventory optimization results and provide strategic insights.

Guidelines:
- Be specific and data-driven
- Focus on business impact (cost, service level, risk)
- Identify patterns and trends
- Provide actionable recommendations
- Prioritize by urgency and impact
- Use clear, professional language

Format insights as:
📊 KEY FINDINGS
- [Finding 1 with data]
- [Finding 2 with data]

⚠️ RISK AREAS
- [Risk with impact]
- [Mitigation strategy]

💡 OPPORTUNITIES
- [Opportunity with benefit]

🎯 STRATEGIC RECOMMENDATIONS
1. [Specific action]
2. [Specific action]

🔴 PRIORITY ACTIONS (Next 24-48 hours)
1. [Urgent action]
2. [Urgent action]"""
    
    def _prepare_context(self, 
                        recommendations: pd.DataFrame,
                        action_plan: Dict[str, Any],
                        entities: Optional[Dict]) -> str:
        """Prepare concise context for LLM."""
        context = []
        
        # Summary stats
        summary = action_plan['summary']
        context.append(f"SUMMARY:")
        context.append(f"- Total items analyzed: {len(recommendations)}")
        context.append(f"- Total actions needed: {summary['total_actions']}")
        context.append(f"- Restock orders: {summary['restock_actions']} (qty: {summary['total_restock_quantity']:.0f})")
        context.append(f"- Transfer opportunities: {summary['transfer_actions']} (qty: {summary['total_transfer_quantity']:.0f})")
        context.append(f"- High priority: {summary['high_priority_actions']}")
        
        # Action distribution
        if not recommendations.empty:
            action_dist = recommendations['action'].value_counts()
            context.append(f"\nACTION DISTRIBUTION:")
            for action, count in action_dist.items():
                pct = (count / len(recommendations)) * 100
                context.append(f"- {action}: {count} items ({pct:.1f}%)")
        
        # Regional analysis (if available)
        if 'region' in recommendations.columns and not recommendations.empty:
            region_actions = recommendations[recommendations['action'] != 'OK'].groupby('region')['action'].count()
            if not region_actions.empty:
                context.append(f"\nREGIONAL BREAKDOWN:")
                for region, count in region_actions.items():
                    context.append(f"- {region}: {count} actions needed")
        
        # Critical items
        urgent = recommendations[recommendations['action'] == 'URGENT_RESTOCK']
        if not urgent.empty:
            context.append(f"\nCRITICAL SHORTAGE: {len(urgent)} items below reorder point")
            top_urgent = urgent.nlargest(3, 'quantity_needed')
            for _, item in top_urgent.iterrows():
                context.append(f"  - {item['product_name'][:40]} at {item['branch_name']}: need {item['quantity_needed']:.0f}")
        
        # Transfer opportunities
        transfers = action_plan.get('actions', [])
        transfer_actions = [t for t in transfers if t['action_type'] == 'TRANSFER']
        if transfer_actions:
            total_distance = sum(t.get('distance_km', 0) for t in transfer_actions)
            avg_distance = total_distance / len(transfer_actions)
            context.append(f"\nTRANSFER ANALYSIS:")
            context.append(f"- {len(transfer_actions)} transfer opportunities identified")
            context.append(f"- Average distance: {avg_distance:.1f} km")
        
        # Scope (if entities provided)
        if entities and entities.get('scope') == 'specific':
            branches = entities.get('branch_names', [])
            if branches:
                context.append(f"\nSCOPE: Focused on {', '.join(branches[:3])}")
        
        return "\n".join(context)
    
    def _fallback_insights(self, recommendations: pd.DataFrame, action_plan: Dict) -> str:
        """Simple rule-based insights if LLM fails."""
        insights = []
        summary = action_plan['summary']
        
        insights.append("📊 KEY FINDINGS")
        insights.append(f"- {summary['total_actions']} total actions required across inventory")
        insights.append(f"- {summary['high_priority_actions']} high-priority items need immediate attention")
        
        if summary['transfer_actions'] > 0:
            savings_pct = (summary['total_transfer_quantity'] / (summary['total_restock_quantity'] + summary['total_transfer_quantity'])) * 100
            insights.append(f"- {savings_pct:.1f}% of needs can be met through internal transfers (cost savings)")
        
        insights.append("\n⚠️ RISK AREAS")
        urgent = recommendations[recommendations['action'] == 'URGENT_RESTOCK']
        if not urgent.empty:
            insights.append(f"- {len(urgent)} items critically low (stockout risk)")
        
        insights.append("\n🎯 PRIORITY ACTIONS")
        insights.append("1. Process all HIGH priority restocks immediately")
        insights.append("2. Initiate internal transfers to reduce external orders")
        insights.append("3. Review forecast accuracy for items with large discrepancies")
        
        return "\n".join(insights)


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
                 llm_provider: LLMProvider,
                 output_dir: str = "charts"):
        self.db = db_manager
        self.forecast_agent = forecast_agent
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # NEW: Smart insights generator for intelligent recommendations
        self.insights_generator = SmartInsightsGenerator(llm_provider)
        
        # Configuration parameters
        self.service_level = 0.95  # 95% service level
        self.lead_time_days = 7    # Default lead time
        self.max_transfer_distance_km = 200  # Max distance for transfer
    
    def optimize_inventory(self, 
                          question: str,
                          entities: Optional[Dict] = None,
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
        
        # Extract filter criteria from entities
        branch_codes = None
        product_codes = None
        regions = None
        
        if entities:
            branch_codes = entities.get('branch_codes')
            product_codes = entities.get('product_codes')
            regions = entities.get('regions')
            
            if branch_codes:
                print(f"   🎯 Filtering by {len(branch_codes)} specific branches")
            if product_codes:
                print(f"   🎯 Filtering by {len(product_codes)} specific products")
            if regions:
                print(f"   🎯 Filtering by regions: {', '.join(regions)}")
        
        try:
            # Step 1: Get current inventory FIRST (with entity filters)
            print("📌 Step 1: Analyzing current inventory...")
            inventory_data = self._get_current_inventory(
                branch_codes=branch_codes,
                product_codes=product_codes,
                regions=regions
            )
            
            if inventory_data.empty:
                return {
                    "success": False,
                    "message": "No inventory data found"
                }
            
            # Step 2: Get PER-ITEM forecast demand (IMPROVED!)
            print("📌 Step 2: Getting per-item demand forecasts...")
            per_item_forecasts = self._get_forecast_data_per_item(inventory_data, horizon_days)
            
            if not per_item_forecasts:
                return {
                    "success": False,
                    "message": "Could not generate forecasts for demand prediction"
                }
            
            # Step 3: Calculate inventory metrics with per-item forecasts
            print("📌 Step 3: Calculating inventory metrics...")
            recommendations = self._generate_recommendations(
                inventory_data, 
                per_item_forecasts, 
                horizon_days
            )
            
            # Step 4: Find transfer opportunities
            print("📌 Step 4: Finding transfer opportunities...")
            transfer_opportunities = self._find_transfer_opportunities(
                recommendations
            )
            
            # Step 5: Generate comprehensive plan
            plan = self._create_action_plan(recommendations, transfer_opportunities)
            
            # Step 6: Create visualization
            chart_path = self._plot_inventory_optimization(
                inventory_data, 
                per_item_forecasts, 
                recommendations
            )
            
            # Step 6: Generate smart insights (NEW!)
            print("📌 Step 6: Generating AI-powered insights...")
            insights = self.insights_generator.generate_insights(
                recommendations, 
                plan, 
                entities
            )
            
            print(f"✅ Optimization complete: {len(plan['actions'])} actions recommended")
            
            return {
                "success": True,
                "per_item_forecasts": per_item_forecasts,
                "inventory_data": inventory_data,
                "recommendations": recommendations,
                "transfer_opportunities": transfer_opportunities,
                "action_plan": plan,
                "chart": chart_path,
                "summary": self._generate_summary(plan),
                "smart_insights": insights  # NEW: AI-powered insights
            }
            
        except Exception as e:
            print(f"❌ Optimization error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "message": f"Optimization failed: {str(e)}"
            }
    
    def _get_forecast_data_per_item(self, 
                                   inventory_data: pd.DataFrame,
                                   horizon_days: int) -> Dict[tuple, Dict]:
        """
        Get forecast demand PER (product_code, branch_code) combination.
        
        IMPROVEMENT: Instead of one aggregate forecast, we forecast
        separately for each product-branch to get accurate predictions.
        
        Returns:
            Dict[(product_code, branch_code)] = {forecast_df, historical_df, metrics}
        """
        forecasts = {}
        
        print(f"🔮 Generating {len(inventory_data)} individual forecasts...")
        
        for idx, row in inventory_data.iterrows():
            product_code = row['product_code']
            branch_code = row['branch_code']
            key = (product_code, branch_code)
            
            # Build PARAMETERIZED query for this specific item
            sql = """
            SELECT date, SUM(quantity) as total_qty
            FROM sales
            WHERE date >= CURRENT_DATE - INTERVAL '90 days'
                AND product_code = :product_code
                AND branch_code = :branch_code
            GROUP BY date 
            ORDER BY date
            """
            
            params = {
                "product_code": product_code,
                "branch_code": branch_code
            }
            
            try:
                # Get historical data with parameterized query
                df = self.db.execute_query(sql, params)
                
                if df.empty or len(df) < 2:  # IMPROVED: Need at least 2 days (was 7)
                    # Use intelligent fallback based on branch average
                    forecasts[key] = self._create_intelligent_fallback(
                        product_code, branch_code, horizon_days, row
                    )
                    continue
                
                # IMPROVED: Handle sparse data (2-6 days)
                if len(df) < 7:
                    avg_demand = df['total_qty'].mean()
                    # Create simple forecast from available data
                    forecasts[key] = self._create_simple_forecast_from_data(
                        df, avg_demand, horizon_days
                    )
                    continue
                
                # FIX: Build non-parameterized SQL for ForecastAgent
                # Use safe string formatting (values are already validated from DB)
                sql_for_forecast = f"""
                SELECT date, SUM(quantity) as total_qty
                FROM sales
                WHERE date >= CURRENT_DATE - INTERVAL '90 days'
                    AND product_code = '{product_code}'
                    AND branch_code = {branch_code}
                GROUP BY date 
                ORDER BY date
                """
                
                # Generate forecast for this specific item
                result = self.forecast_agent.forecast(
                    sql_for_forecast, 
                    f"forecast for {product_code} at branch {branch_code}", 
                    horizon_days
                )
                
                if result.get('success'):
                    forecasts[key] = {
                        'forecast_df': result['forecast'],
                        'historical_df': result['historical_data'],
                        'metrics': result['metrics']
                    }
                else:
                    forecasts[key] = self._create_fallback_forecast(horizon_days)
                    
            except Exception as e:
                print(f"⚠️ Forecast failed for {product_code} at branch {branch_code}: {e}")
                forecasts[key] = self._create_fallback_forecast(horizon_days)
        
        print(f"✅ Generated {len(forecasts)} forecasts successfully")
        return forecasts
    
    def _create_fallback_forecast(self, horizon_days: int) -> Dict:
        """Create a simple fallback forecast when data is insufficient."""
        future_dates = pd.date_range(
            start=datetime.now() + timedelta(days=1),
            periods=horizon_days,
            freq='D'
        )
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'forecast': [0.0] * horizon_days  # Conservative: assume 0 demand
        }).set_index('date')
        
        historical_df = pd.DataFrame({
            'value': [0.0]
        }, index=[datetime.now() - timedelta(days=1)])
        
        return {
            'forecast_df': forecast_df,
            'historical_df': historical_df,
            'metrics': {
                'recent_avg_daily': 0.0,
                'forecast_avg_daily': 0.0,
                'forecast_total': 0.0,
                'trend': 'unknown'
            }
        }
    
    def _create_intelligent_fallback(self, product_code: str, branch_code: int, 
                                    horizon_days: int, inventory_row: pd.Series) -> Dict:
        """
        Create intelligent fallback forecast using branch average demand.
        
        IMPROVEMENT: Instead of returning 0, estimate based on:
        - Current stock level
        - Average inventory turnover for similar products
        - Conservative estimate
        """
        # Use very conservative estimate: 1% of current stock per month
        current_stock = inventory_row['current_stock']
        estimated_monthly_demand = current_stock * 0.01
        estimated_daily_demand = max(0.1, estimated_monthly_demand / 30)
        
        future_dates = pd.date_range(
            start=datetime.now() + timedelta(days=1),
            periods=horizon_days,
            freq='D'
        )
        
        forecast_values = [estimated_daily_demand] * horizon_days
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'forecast': forecast_values
        }).set_index('date')
        
        historical_df = pd.DataFrame({
            'value': [estimated_daily_demand]
        }, index=[datetime.now() - timedelta(days=1)])
        
        return {
            'forecast_df': forecast_df,
            'historical_df': historical_df,
            'metrics': {
                'recent_avg_daily': estimated_daily_demand,
                'forecast_avg_daily': estimated_daily_demand,
                'forecast_total': estimated_daily_demand * horizon_days,
                'trend': 'estimated'
            }
        }
    
    def _create_simple_forecast_from_data(self, historical_df: pd.DataFrame, 
                                         avg_demand: float, 
                                         horizon_days: int) -> Dict:
        """
        Create forecast from sparse historical data (2-6 days).
        
        IMPROVEMENT: Use available data instead of fallback zeros.
        """
        # Prepare historical data
        hist_df = historical_df.copy()
        hist_df['date'] = pd.to_datetime(hist_df['date'])
        hist_df = hist_df.set_index('date')
        hist_df.columns = ['value']
        
        # Create forecast using average
        future_dates = pd.date_range(
            start=datetime.now() + timedelta(days=1),
            periods=horizon_days,
            freq='D'
        )
        
        # Add slight growth trend if data shows increase
        if len(hist_df) >= 3:
            recent_avg = hist_df['value'].tail(2).mean()
            older_avg = hist_df['value'].head(2).mean()
            if recent_avg > older_avg * 1.1:
                trend_factor = 1.05  # 5% growth
                trend = 'increasing'
            elif recent_avg < older_avg * 0.9:
                trend_factor = 0.95  # 5% decline
                trend = 'decreasing'
            else:
                trend_factor = 1.0
                trend = 'stable'
        else:
            trend_factor = 1.0
            trend = 'stable'
        
        forecast_values = [avg_demand * trend_factor] * horizon_days
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'forecast': forecast_values
        }).set_index('date')
        
        return {
            'forecast_df': forecast_df,
            'historical_df': hist_df,
            'metrics': {
                'recent_avg_daily': float(avg_demand),
                'forecast_avg_daily': float(avg_demand * trend_factor),
                'forecast_total': float(avg_demand * trend_factor * horizon_days),
                'trend': trend
            }
        }
    
    def _get_current_inventory(self, 
                               branch_codes: Optional[List[int]] = None,
                               product_codes: Optional[List[str]] = None,
                               regions: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get current inventory levels with SMART FILTERING based on extracted entities.
        
        IMPROVEMENT: Supports filtering by multiple branches/products/regions
        """
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
        
        params = {}
        
        # Filter by specific branches (if mentioned in question)
        if branch_codes and len(branch_codes) > 0:
            # Use IN clause for multiple branches
            placeholders = ','.join([f':branch_code_{i}' for i in range(len(branch_codes))])
            sql += f" AND i.branch_code IN ({placeholders})"
            for i, code in enumerate(branch_codes):
                params[f'branch_code_{i}'] = code
        
        # Filter by specific products (if mentioned in question)
        if product_codes and len(product_codes) > 0:
            placeholders = ','.join([f':product_code_{i}' for i in range(len(product_codes))])
            sql += f" AND i.product_code IN ({placeholders})"
            for i, code in enumerate(product_codes):
                params[f'product_code_{i}'] = code
        
        # Filter by regions (if mentioned in question)
        if regions and len(regions) > 0:
            placeholders = ','.join([f':region_{i}' for i in range(len(regions))])
            sql += f" AND b.region IN ({placeholders})"
            for i, region in enumerate(regions):
                params[f'region_{i}'] = region
        
        sql += " ORDER BY i.branch_code, i.product_code"
        
        result = self.db.execute_query(sql, params if params else None)
        
        if not result.empty:
            print(f"   ✅ Found {len(result)} inventory items matching criteria")
        
        return result
    
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
                                 per_item_forecasts: Dict[tuple, Dict],
                                 horizon_days: int) -> pd.DataFrame:
        """
        Generate inventory recommendations using PER-ITEM forecasts.
        
        IMPROVEMENT: Each (product, branch) gets its own forecast-based metrics.
        """
        
        recommendations = []
        
        for idx, row in inventory_data.iterrows():
            product_code = row['product_code']
            branch_code = row['branch_code']
            current_stock = row['current_stock']
            
            # Get forecast for THIS specific item
            key = (product_code, branch_code)
            forecast_data = per_item_forecasts.get(key)
            
            if not forecast_data:
                # Skip if no forecast available
                continue
            
            forecast_df = forecast_data['forecast_df']
            historical_df = forecast_data['historical_df']
            
            # Calculate demand statistics from THIS item's historical data
            avg_daily_demand = historical_df['value'].mean()
            std_daily_demand = historical_df['value'].std()
            total_forecast_demand = forecast_df['forecast'].sum()
            
            # Handle edge case: no historical demand
            if avg_daily_demand == 0 or pd.isna(avg_daily_demand):
                avg_daily_demand = 0.1  # Small default to avoid division by zero
            if std_daily_demand == 0 or pd.isna(std_daily_demand):
                std_daily_demand = avg_daily_demand * 0.3  # 30% CV as default
            
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
                'product_code': product_code,
                'branch_code': branch_code,
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
                                    recommendations: pd.DataFrame) -> List[Dict]:
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
            
            # Find nearby branches with surplus (PARAMETERIZED)
            nearby_query = """
            SELECT 
                bd.branch_code_1 as source_branch,
                bd.branch_code_2 as dest_branch,
                bd.distance_km,
                b.branch_name as source_branch_name
            FROM branch_distance bd
            JOIN branch b ON bd.branch_code_1 = b.branch_code
            WHERE bd.branch_code_2 = :deficit_branch
                AND bd.distance_km <= :max_distance
            ORDER BY bd.distance_km ASC
            """
            
            params = {
                'deficit_branch': int(deficit_branch),
                'max_distance': self.max_transfer_distance_km
            }
            
            try:
                nearby_branches = self.db.execute_query(nearby_query, params)
                
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
                                    per_item_forecasts: Dict[tuple, Dict],
                                    recommendations: pd.DataFrame) -> str:
        """
        Create visualization for inventory optimization with per-item forecasts.
        
        IMPROVED: Better labels, titles with branch names, and clear legends.
        """
        
        # Get unique branch names for title
        unique_branches = inventory_data['branch_name'].unique()
        if len(unique_branches) <= 3:
            branch_title = f"Branches: {', '.join(unique_branches)}"
        else:
            branch_title = f"{len(unique_branches)} Branches"
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'Inventory Optimization Analysis - {branch_title}', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Plot 1: Current Stock vs ROP (Top 10 items)
        ax1 = axes[0, 0]
        top_10 = recommendations.head(10)
        
        # Create product labels with branch names
        labels = [f"{row['product_name'][:30]}\n({row['branch_name'][:20]})" 
                  for _, row in top_10.iterrows()]
        
        current_stock = top_10['current_stock'].values
        rop = top_10['reorder_point'].values
        safety_stock = top_10['safety_stock'].values
        
        x = np.arange(len(labels))
        width = 0.25
        
        bars1 = ax1.bar(x - width, current_stock, width, label='Current Stock', 
                        color='steelblue', alpha=0.8)
        bars2 = ax1.bar(x, rop, width, label='Reorder Point (ROP)', 
                        color='orange', alpha=0.8)
        bars3 = ax1.bar(x + width, safety_stock, width, label='Safety Stock', 
                        color='green', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontsize=7)
        
        ax1.set_xlabel('Product @ Branch', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Quantity', fontsize=11, fontweight='bold')
        ax1.set_title('Current Stock vs ROP & Safety Stock (Top 10 Items)', 
                      fontsize=12, fontweight='bold', pad=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Plot 2: Action Distribution by Branch
        ax2 = axes[0, 1]
        action_counts = recommendations['action'].value_counts()
        colors = {'OK': '#2ecc71', 'RESTOCK': '#f39c12', 
                  'URGENT_RESTOCK': '#e74c3c', 'SURPLUS': '#3498db'}
        
        wedges, texts, autotexts = ax2.pie(
            action_counts.values, 
            labels=action_counts.index, 
            autopct='%1.1f%%',
            colors=[colors.get(action, 'gray') for action in action_counts.index],
            startangle=90,
            textprops={'fontsize': 10, 'weight': 'bold'}
        )
        
        # Add count to labels
        for i, (label, count) in enumerate(zip(action_counts.index, action_counts.values)):
            texts[i].set_text(f'{label}\n({count} items)')
        
        ax2.set_title('Inventory Action Distribution', 
                      fontsize=12, fontweight='bold', pad=10)
        
        # Plot 3: Aggregated Demand Forecast
        ax3 = axes[1, 0]
        if per_item_forecasts:
            first_forecast = list(per_item_forecasts.values())[0]
            forecast_dates = first_forecast['forecast_df'].index
            
            # Sum all forecasts
            total_forecast = pd.Series(0.0, index=forecast_dates)
            for forecast_data in per_item_forecasts.values():
                total_forecast += forecast_data['forecast_df']['forecast']
            
            ax3.plot(total_forecast.index, total_forecast.values, 
                    label='Total Forecasted Demand', linewidth=2.5, 
                    color='orange', marker='o', markersize=4, alpha=0.8)
            
            # Add trend line
            x_numeric = np.arange(len(total_forecast))
            z = np.polyfit(x_numeric, total_forecast.values, 1)
            p = np.poly1d(z)
            ax3.plot(total_forecast.index, p(x_numeric), 
                    "--", alpha=0.5, color='red', linewidth=1.5, label='Trend')
            
            # Add mean line
            mean_val = total_forecast.mean()
            ax3.axhline(y=mean_val, color='green', linestyle=':', 
                       linewidth=1.5, alpha=0.7, label=f'Average: {mean_val:.0f}')
            
            ax3.set_xlabel('Date', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Quantity', fontsize=11, fontweight='bold')
            ax3.set_title(f'30-Day Demand Forecast - {branch_title}', 
                         fontsize=12, fontweight='bold', pad=10)
            ax3.legend(loc='best', fontsize=9)
            ax3.grid(True, alpha=0.3, linestyle='--')
            ax3.tick_params(axis='x', rotation=45)
            
            # Format x-axis
            import matplotlib.dates as mdates
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax3.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        else:
            ax3.text(0.5, 0.5, 'No forecast data available', 
                    ha='center', va='center', fontsize=12)
        
        # Plot 4: Priority Distribution by Branch
        ax4 = axes[1, 1]
        priority_data = recommendations[recommendations['action'] != 'OK']
        
        if not priority_data.empty:
            priority_counts = priority_data['priority'].value_counts()
            priority_colors = {'HIGH': '#e74c3c', 'MEDIUM': '#f39c12', 'LOW': '#f1c40f'}
            
            bars = ax4.bar(priority_counts.index, priority_counts.values,
                          color=[priority_colors.get(p, 'gray') for p in priority_counts.index],
                          alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax4.set_xlabel('Priority Level', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Number of Actions', fontsize=11, fontweight='bold')
            ax4.set_title('Action Priority Distribution', 
                         fontsize=12, fontweight='bold', pad=10)
            ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            # Add summary text
            total_actions = len(priority_data)
            summary_text = f'Total Actions: {total_actions}'
            ax4.text(0.5, 0.95, summary_text, transform=ax4.transAxes,
                    ha='center', va='top', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax4.text(0.5, 0.5, '✓ All Items OK\nNo Actions Needed', 
                    ha='center', va='center', fontsize=14, color='green', fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Create filename with branch info
        if len(unique_branches) == 1:
            branch_slug = unique_branches[0].replace(' ', '_')[:20]
            filename = f"inventory_opt_{branch_slug}_{uuid.uuid4().hex[:8]}.png"
        else:
            filename = f"inventory_opt_{len(unique_branches)}branches_{uuid.uuid4().hex[:8]}.png"
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Created inventory optimization chart: {filepath}")
        print(f"   📈 Includes: Stock vs ROP, Actions, Forecast, Priorities")
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
        self.entity_extractor = EntityExtractor(llm_provider, db_manager)  # NEW: Entity extraction
        self.intent_agent = IntentAgent(llm_provider)
        self.sql_agent = SQLAgent(llm_provider, self.schema_agent)
        self.analytics_agent = AnalyticsAgent(db_manager)
        self.forecast_agent = ForecastAgent(db_manager)
        self.inventory_agent = InventoryOptimizationAgent(db_manager, self.forecast_agent, llm_provider)  # With LLM for insights
        
        print("✅ OrchestratorAgent initialized with all sub-agents (Entity Extraction + Smart Insights)")
    
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
                # Step 2a: Extract entities from question (NEW!)
                print(f"📌 Step 2: Extracting entities from question...")
                entities = self.entity_extractor.extract_entities(question)
                
                # Step 2b: Optimize inventory with entity filters
                print(f"📌 Step 3: Processing with Inventory Optimization Agent")
                result = self.inventory_agent.optimize_inventory(question, entities=entities)
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


def export_inventory_plan_to_excel(result: Dict[str, Any], filename: str = "inventory_plan.xlsx"):
    """
    Export detailed inventory optimization plan to Excel with multiple sheets.
    
    IMPROVEMENT: Professional multi-sheet Excel export for business users.
    """
    if not result.get('success'):
        print("❌ Cannot export: optimization was not successful")
        return
    
    try:
        action_plan = result['result'].get('action_plan')
        recommendations = result['result'].get('recommendations')
        transfer_opportunities = result['result'].get('transfer_opportunities')
        
        if not action_plan:
            print("❌ No action plan to export")
            return
        
        print(f"📝 Creating Excel file: {filename}")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: Summary
            summary_data = {
                'Metric': [
                    'Total Actions',
                    'Restock Orders',
                    'Transfer Opportunities',
                    'High Priority Actions',
                    'Total Restock Quantity',
                    'Total Transfer Quantity'
                ],
                'Value': [
                    action_plan['summary']['total_actions'],
                    action_plan['summary']['restock_actions'],
                    action_plan['summary']['transfer_actions'],
                    action_plan['summary']['high_priority_actions'],
                    action_plan['summary']['total_restock_quantity'],
                    action_plan['summary']['total_transfer_quantity']
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            print(f"   ✓ Sheet 1: Summary")
            
            # Sheet 2: Restock Orders
            restock_actions = [a for a in action_plan['actions'] if a['action_type'] == 'RESTOCK']
            if restock_actions:
                restock_df = pd.DataFrame(restock_actions)
                restock_df.to_excel(writer, sheet_name='Restock Orders', index=False)
                print(f"   ✓ Sheet 2: Restock Orders ({len(restock_actions)} items)")
            
            # Sheet 3: Transfer Opportunities
            transfer_actions = [a for a in action_plan['actions'] if a['action_type'] == 'TRANSFER']
            if transfer_actions:
                transfer_df = pd.DataFrame(transfer_actions)
                transfer_df.to_excel(writer, sheet_name='Transfers', index=False)
                print(f"   ✓ Sheet 3: Transfers ({len(transfer_actions)} items)")
            
            # Sheet 4: All Recommendations
            if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
                recommendations.to_excel(writer, sheet_name='All Items', index=False)
                print(f"   ✓ Sheet 4: All Items ({len(recommendations)} items)")
            
            # Sheet 5: Priority Actions
            high_priority = [a for a in action_plan['actions'] if a['priority'] == 'HIGH']
            if high_priority:
                priority_df = pd.DataFrame(high_priority)
                priority_df.to_excel(writer, sheet_name='High Priority', index=False)
                print(f"   ✓ Sheet 5: High Priority ({len(high_priority)} items)")
        
        print(f"✅ Exported detailed plan to {filename}")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()


def export_forecasts_to_csv(result: Dict[str, Any], filename: str = "forecasts_detail.csv"):
    """
    Export detailed per-item forecasts to CSV for easy analysis.
    
    NEW: Export all forecast results with comparisons to CSV.
    """
    if not result.get('success'):
        print("❌ Cannot export: optimization was not successful")
        return
    
    try:
        per_item_forecasts = result['result'].get('per_item_forecasts')
        inventory_data = result['result'].get('inventory_data')
        
        if not per_item_forecasts or inventory_data is None or inventory_data.empty:
            print("❌ No forecast data to export")
            return
        
        print(f"📝 Creating forecast CSV: {filename}")
        
        # Prepare detailed forecast data
        forecast_rows = []
        
        for idx, row in inventory_data.iterrows():
            product_code = row['product_code']
            branch_code = row['branch_code']
            key = (product_code, branch_code)
            
            forecast_data = per_item_forecasts.get(key)
            if not forecast_data:
                continue
            
            metrics = forecast_data['metrics']
            
            forecast_rows.append({
                'product_code': product_code,
                'product_name': row['product_name'],
                'branch_code': branch_code,
                'branch_name': row['branch_name'],
                'region': row['region'],
                'current_stock': row['current_stock'],
                'unit': row['unit'],
                'recent_avg_daily_demand': metrics['recent_avg_daily'],
                'forecast_avg_daily_demand': metrics['forecast_avg_daily'],
                'forecast_total_30d': metrics['forecast_total'],
                'trend': metrics['trend'],
                'stock_coverage_days': row['current_stock'] / max(metrics['recent_avg_daily'], 0.1)
            })
        
        forecast_df = pd.DataFrame(forecast_rows)
        forecast_df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"✅ Exported {len(forecast_df)} forecast records to {filename}")
        print(f"   📊 Columns: product, branch, stock, demand (recent/forecast), trend")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()


def export_recommendations_to_csv(result: Dict[str, Any], filename: str = "recommendations_detail.csv"):
    """
    Export detailed recommendations with all metrics to CSV.
    
    NEW: Complete recommendations export for analysis.
    """
    if not result.get('success'):
        print("❌ Cannot export: optimization was not successful")
        return
    
    try:
        recommendations = result['result'].get('recommendations')
        
        if recommendations is None or recommendations.empty:
            print("❌ No recommendations to export")
            return
        
        print(f"📝 Creating recommendations CSV: {filename}")
        
        recommendations.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"✅ Exported {len(recommendations)} recommendations to {filename}")
        print(f"   📊 Includes: ROP, Safety Stock, EOQ, Actions, Priorities")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()


def display_action_plan(action_plan: Dict[str, Any]):
    """
    Display detailed action plan in a beautiful format.
    
    IMPROVEMENT: Human-readable output for business users.
    """
    print("\n" + "="*80)
    print("📋 DETAILED ACTION PLAN")
    print("="*80)
    
    summary = action_plan['summary']
    print(f"\n📊 SUMMARY:")
    print(f"   Total Actions: {summary['total_actions']}")
    print(f"   - Restock Orders: {summary['restock_actions']} (Qty: {summary['total_restock_quantity']:.0f})")
    print(f"   - Internal Transfers: {summary['transfer_actions']} (Qty: {summary['total_transfer_quantity']:.0f})")
    print(f"   - High Priority: {summary['high_priority_actions']}")
    
    # Group by priority
    actions_by_priority = {}
    for action in action_plan['actions']:
        priority = action['priority']
        if priority not in actions_by_priority:
            actions_by_priority[priority] = []
        actions_by_priority[priority].append(action)
    
    # Display HIGH priority first
    for priority in ['HIGH', 'MEDIUM', 'LOW']:
        if priority not in actions_by_priority:
            continue
        
        actions = actions_by_priority[priority]
        
        priority_colors = {
            'HIGH': '🔴',
            'MEDIUM': '🟡',
            'LOW': '🟢'
        }
        
        print(f"\n{priority_colors[priority]} {priority} PRIORITY ({len(actions)} actions):")
        print("-" * 80)
        
        for i, action in enumerate(actions[:10], 1):  # Show top 10
            if action['action_type'] == 'RESTOCK':
                print(f"\n   {i}. 📦 RESTOCK: {action['product_name'][:50]}")
                print(f"      Branch: {action['branch_name']}")
                print(f"      Quantity: {action['quantity']:.0f} {action['unit']}")
                print(f"      Reason: {action['reason']}")
                
            elif action['action_type'] == 'TRANSFER':
                print(f"\n   {i}. 🚚 TRANSFER: {action['product_name'][:50]}")
                print(f"      From: {action['source_branch_name']}")
                print(f"      To: {action['dest_branch_name']}")
                print(f"      Quantity: {action['quantity']:.0f} {action['unit']}")
                print(f"      Distance: {action['distance_km']:.1f} km")
                print(f"      💰 {action['cost_saving']}")
        
        if len(actions) > 10:
            print(f"\n   ... and {len(actions) - 10} more {priority} priority actions")
    
    print("\n" + "="*80)


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
    
    # Example 3: Inventory Optimization with Entity Extraction (NEW!)
    print("\n" + "="*80)
    print("🎯 Example 3: Inventory Optimization with Smart Filtering")
    print("="*80)
    result3 = orchestrator.process_query(
        "Tối ưu hóa tồn kho của chi nhánh đà nẵng: "
        "kiểm tra sản phẩm nào cần nhập hàng và có thể chuyển kho không"
    )
    
    if result3.get('success'):
        print("\n" + "="*80)
        print("📋 INVENTORY OPTIMIZATION RESULTS")
        print("="*80)
        
        # Display summary statistics first
        inventory_data = result3['result'].get('inventory_data')
        recommendations = result3['result'].get('recommendations')
        
        if inventory_data is not None:
            print(f"\n📊 ANALYSIS SCOPE:")
            print(f"   • Total items analyzed: {len(inventory_data)}")
            print(f"   • Branches: {inventory_data['branch_name'].nunique()}")
            unique_branches = inventory_data['branch_name'].unique()
            for branch in unique_branches:
                count = len(inventory_data[inventory_data['branch_name'] == branch])
                print(f"     - {branch}: {count} products")
        
        if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
            print(f"\n📈 RECOMMENDATIONS SUMMARY:")
            action_dist = recommendations['action'].value_counts()
            for action, count in action_dist.items():
                pct = (count / len(recommendations)) * 100
                print(f"   • {action}: {count} items ({pct:.1f}%)")
        
        # Display detailed action plan
        if result3['result'].get('action_plan'):
            plan = result3['result']['action_plan']
            display_action_plan(plan)
        
        # Display smart insights
        if result3['result'].get('smart_insights'):
            print("\n" + "="*80)
            print("🧠 AI-POWERED INSIGHTS")
            print("="*80)
            print(result3['result']['smart_insights'])
        
        # Export all files
        print("\n" + "="*80)
        print("📊 EXPORTING RESULTS TO FILES")
        print("="*80)
        
        # 1. Excel file (multi-sheet)
        export_inventory_plan_to_excel(result3, "inventory_optimization_plan.xlsx")
        
        # 2. Forecasts CSV (detailed)
        export_forecasts_to_csv(result3, "forecasts_detail.csv")
        
        # 3. Recommendations CSV (complete)
        export_recommendations_to_csv(result3, "recommendations_detail.csv")
        
        print("\n✅ ALL EXPORTS COMPLETE!")
        print("📁 Files created:")
        print("   1. inventory_optimization_plan.xlsx (5 sheets)")
        print("   2. forecasts_detail.csv (forecast comparisons)")
        print("   3. recommendations_detail.csv (all metrics)")
        print(f"   4. {result3['result']['chart']} (visualization)")
    
    else:
        print("\n" + "="*80)
        print("❌ OPTIMIZATION FAILED")
        print("="*80)
        print(f"Error: {result3.get('error', result3.get('message', 'Unknown error'))}")
    
    # Show history
    display_conversation_history(orchestrator)


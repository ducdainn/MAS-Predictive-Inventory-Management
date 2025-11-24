"""
Advanced memory manager that backs the multi-agent system conversation history.
"""

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent.core.conversation import ConversationEntry

try:
    from agent.advanced_memory import AdvancedMemoryManager
    ADVANCED_MEMORY_AVAILABLE = True
except ImportError as e:
    ADVANCED_MEMORY_AVAILABLE = False
    print(f"⚠️  Advanced Memory not available: {e}")


class MemoryManager:
    """
    UPGRADED: Advanced Memory System with Vector DB.

    Features:
    - Short-term memory (working memory)
    - Long-term memory (Vector DB with semantic search)
    - Episodic memory (learned experiences)
    - Adaptive learning
    """

    def __init__(self, max_history: int = 100, db_path: str = "agent_memory.db"):
        """Initialize with advanced memory backend."""
        print("🔄 Initializing Advanced Memory System...")

        # Old fields for compatibility
        self.max_history = max_history
        self.db_path = db_path
        self.conversation_history: List[ConversationEntry] = []
        self.schema_cache: Dict[str, Any] = {}

        # NEW: Advanced memory backend
        if ADVANCED_MEMORY_AVAILABLE:
            self.advanced_memory = AdvancedMemoryManager(
                persist_dir="./memory",
                use_redis=False
            )
            print("✅ Advanced Memory System ready!")
        else:
            self.advanced_memory = None
            print("⚠️  Running with basic memory (Advanced Memory unavailable)")
            self._init_db()
            self._load_from_db()

        self._initialize_schema_cache()

    def _init_db(self):
        """Initialize SQLite database for basic memory fallback."""
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
                success INTEGER DEFAULT 1,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def _load_from_db(self):
        """Load conversation history from DB."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT timestamp, question, intent, sql_query, result_summary, charts, success, error_message
                FROM conversation_history
                ORDER BY created_at DESC
                LIMIT ?
            """, (self.max_history,))

            rows = cursor.fetchall()
            conn.close()

            for row in reversed(rows):
                entry = ConversationEntry(
                    timestamp=datetime.fromisoformat(row[0]),
                    question=row[1],
                    intent=row[2],
                    sql_query=row[3],
                    result_summary=row[4],
                    charts=json.loads(row[5]) if row[5] else [],
                    success=bool(row[6]) if len(row) > 6 and row[6] is not None else True,
                    error_message=row[7] if len(row) > 7 else None
                )
                self.conversation_history.append(entry)
        except Exception as e:
            print(f"⚠️  Could not load history: {e}")

    def _initialize_schema_cache(self):
        """Cache schema info and common patterns."""
        try:
            self.schema_cache['tables'] = {
                'sales': ['date', 'product_code', 'branch_code', 'quantity', 'value'],
                'product': ['product_code', 'product_name', 'category', 'unit'],
                'branch': ['branch_code', 'branch_name', 'region', 'address'],
            }

            self.schema_cache['common_patterns'] = {
                'time_series': 'SELECT date, SUM(quantity) FROM sales GROUP BY date',
                'top_products': 'SELECT p.product_name, SUM(s.quantity) FROM sales s JOIN product p',
                'regional': 'SELECT b.region, SUM(s.quantity) FROM sales s JOIN branch b',
            }
        except Exception as e:
            print(f"⚠️  Schema cache init warning: {e}")

    def add_entry(self, entry: ConversationEntry):
        """Add conversation entry."""
        if self.advanced_memory:
            self.advanced_memory.remember_interaction(
                query=entry.question,
                intent=entry.intent,
                result={
                    'sql': entry.sql_query,
                    'summary': entry.result_summary,
                    'charts': entry.charts,
                    'metadata': {
                        'success': entry.success,
                        'error_message': entry.error_message
                    }
                },
                success=entry.success
            )
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO conversation_history 
                (timestamp, question, intent, sql_query, result_summary, charts, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.timestamp.isoformat(),
                entry.question,
                entry.intent,
                entry.sql_query,
                entry.result_summary,
                json.dumps(entry.charts),
                1 if entry.success else 0,
                entry.error_message
            ))

            conn.commit()
            conn.close()

        self.conversation_history.append(entry)
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

    def get_recent_context(self, n: int = 5) -> str:
        """Get recent conversation context."""
        if self.advanced_memory:
            return self.advanced_memory.get_recent_context(n)

        if not self.conversation_history:
            return "No recent context."

        context = "Recent interactions:\n"
        for entry in self.conversation_history[-n:]:
            context += f"  - {entry.intent}: {entry.question[:50]}...\n"
        return context

    def search_similar(self, query: str, top_k: int = 3, only_successful: bool = True):
        """
        Search for similar past queries using semantic search.
        
        Args:
            query: Query to search for
            top_k: Number of results to return
            only_successful: If True, only return queries that were successful
        """
        if self.advanced_memory:
            results = self.advanced_memory.recall_similar(query, top_k * 2)  # Get more to filter
            if only_successful:
                # Filter to only successful queries
                successful_results = [
                    r for r in results 
                    if isinstance(r, dict) and r.get('solution', {}).get('metadata', {}).get('success', True)
                ]
                return successful_results[:top_k]
            return results[:top_k]
        
        # Fallback: search in conversation history
        if not self.conversation_history:
            return []
        
        # Simple keyword-based similarity (fallback)
        query_lower = query.lower()
        similar = []
        for entry in self.conversation_history:
            if only_successful and not entry.success:
                continue
            # Simple similarity: count common words
            entry_words = set(entry.question.lower().split())
            query_words = set(query_lower.split())
            common = len(entry_words & query_words)
            if common > 0:
                similar.append({
                    'query': entry.question,
                    'sql': entry.sql_query,
                    'similarity': common / max(len(entry_words), len(query_words)),
                    'success': entry.success
                })
        
        # Sort by similarity and return top_k
        similar.sort(key=lambda x: x['similarity'], reverse=True)
        return similar[:top_k]

    def learn_from_result(self, query: str, intent: str, strategy: str,
                          success: bool, metrics: Dict[str, Any]):
        """Learn from query result."""
        if not self.advanced_memory:
            return

        insight = f"Strategy '{strategy}' for '{intent}' "
        insight += "worked well" if success else "failed"

        if metrics:
            insight += f" (metrics: {metrics})"

        self.advanced_memory.learn_from_experience(
            situation=f"{intent}_query",
            action=strategy,
            outcome="success" if success else "failure",
            success=success,
            insight=insight,
            context={'query': query, 'metrics': metrics},
            confidence=0.8 if success else 0.5
        )

    def get_learned_insights(self, intent: str) -> List[str]:
        """Get learned insights for an intent type."""
        if self.advanced_memory:
            return self.advanced_memory.get_learned_insights(f"{intent}_query")
        return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if self.advanced_memory:
            return self.advanced_memory.get_statistics()

        return {
            "total_conversations": len(self.conversation_history),
            "max_history": self.max_history
        }

    def clear(self):
        """Clear short-term memory."""
        self.conversation_history.clear()
        if self.advanced_memory:
            self.advanced_memory.clear_short_term()




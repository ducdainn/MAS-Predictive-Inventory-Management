"""
Advanced Memory System with Vector Database

Architecture:
    - Short-term: Working memory (last 10 interactions)
    - Long-term: Vector DB for semantic search (Qdrant)
    - Episodic: Learned experiences from successes/failures
    - Semantic: Domain knowledge (products, branches, patterns)

"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import deque
import hashlib
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Load .env file from agent directory
if load_dotenv:
    agent_dir = Path(__file__).resolve().parent
    load_dotenv(agent_dir / ".env")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, Query, QueryFilter
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("⚠️  Qdrant not installed. Run: pip install qdrant-client")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError as e:
    EMBEDDINGS_AVAILABLE = False
    print(f"⚠️  sentence-transformers import error: {e}")
except Exception as e:
    EMBEDDINGS_AVAILABLE = False
    print(f"⚠️  sentence-transformers error: {e}")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("ℹ️  Redis not available (optional). Install with: pip install redis")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ShortTermMemoryEntry:
    """Single entry in short-term memory (working memory)."""
    timestamp: datetime
    query: str
    intent: str
    result_summary: str
    success: bool
    metadata: Dict[str, Any]


@dataclass
class EpisodicMemoryEntry:
    """Learned experience from past interactions."""
    id: str
    timestamp: datetime
    query_pattern: str
    context: Dict[str, Any]
    action_taken: str
    outcome: str
    success: bool
    learned_insight: str
    confidence: float  # How confident we are in this learning


# ============================================================================
# SHORT-TERM MEMORY (Working Memory)
# ============================================================================

class ShortTermMemory:
    """
    Working memory for current session.
    
    Stores:
    - Last 10-20 interactions
    - Active goals & sub-goals
    - Scratchpad for reasoning
    - Pending actions
    """
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.entries = deque(maxlen=max_size)
        self.scratchpad = {}  # For reasoning steps
        self.active_goals = []
        
    def add(self, entry: ShortTermMemoryEntry):
        """Add entry to short-term memory."""
        self.entries.append(entry)
    
    def get_recent(self, n: int = 5) -> List[ShortTermMemoryEntry]:
        """Get n most recent entries."""
        return list(self.entries)[-n:]
    
    def get_context(self) -> str:
        """Get formatted context for LLM."""
        if not self.entries:
            return "No recent context."
        
        context = "Recent interactions:\n"
        for entry in list(self.entries)[-5:]:
            status = "✅" if entry.success else "❌"
            context += f"{status} {entry.intent}: {entry.query[:50]}...\n"
        
        return context
    
    def update_scratchpad(self, key: str, value: Any):
        """Update reasoning scratchpad."""
        self.scratchpad[key] = value
    
    def clear(self):
        """Clear short-term memory (e.g., new session)."""
        self.entries.clear()
        self.scratchpad.clear()
        self.active_goals.clear()


# ============================================================================
# LONG-TERM MEMORY (Vector Database)
# ============================================================================

class LongTermMemory:
    """
    Persistent memory with semantic search.
    
    Uses Qdrant for vector storage and retrieval.
    Stores embeddings of:
    - Successful query patterns
    - Solutions that worked
    - User preferences
    """
    
    def __init__(self, 
                 qdrant_url: Optional[str] = None,
                 qdrant_api_key: Optional[str] = None,
                 use_redis: bool = False,
                 local_path: Optional[str] = None):
        """
        Initialize Qdrant client.
        
        Args:
            qdrant_url: Qdrant server URL (for cloud instance)
            qdrant_api_key: Qdrant API key (for cloud instance)
            use_redis: Whether to use Redis cache
            local_path: Path for local Qdrant storage (if using local mode)
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant required. Install: pip install qdrant-client")
        
        # Load Qdrant configuration from environment variables
        qdrant_mode = os.getenv("QDRANT_MODE", "cloud")
        
        if qdrant_mode == "local":
            # LOCAL MODE: Use local file storage (no server needed)
            if local_path is None:
                local_path = os.getenv("QDRANT_LOCAL_PATH", "./memory/qdrant_data")
            
            os.makedirs(local_path, exist_ok=True)
            self.client = QdrantClient(path=local_path)
            print(f"✅ Qdrant initialized in LOCAL mode: {local_path}")
        else:
            # CLOUD MODE: Connect to Qdrant Cloud
            if qdrant_url is None:
                qdrant_url = os.getenv("QDRANT_URL")
            
            if qdrant_api_key is None:
                qdrant_api_key = os.getenv("QDRANT_API_KEY")
            
            if not qdrant_url or not qdrant_api_key:
                raise ValueError(
                    "QDRANT_URL and QDRANT_API_KEY must be set for cloud mode. "
                    "Or set QDRANT_MODE=local for local storage."
                )
            
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key
            )
            print(f"✅ Qdrant initialized in CLOUD mode: {qdrant_url}")
        
        # Embedding model (multilingual for Vietnamese)
        # Model dimension: 384 for paraphrase-multilingual-MiniLM-L12-v2
        self.embedding_dim = 384
        if EMBEDDINGS_AVAILABLE:
            self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("✅ Loaded multilingual embedding model")
        else:
            self.encoder = None
            print("⚠️  No embedding model available")
        
        # Collection names
        self.query_collection_name = "query_patterns"
        self.user_prefs_collection_name = "user_preferences"
        
        # Create collections if they don't exist
        self._ensure_collections()
        
        # Redis cache (optional, for faster retrieval)
        self.redis_client = None
        if use_redis and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host='localhost', 
                    port=6379, 
                    decode_responses=True
                )
                self.redis_client.ping()
                print("✅ Redis cache connected")
            except:
                self.redis_client = None
                print("ℹ️  Redis not available, using Qdrant only")
    
    def _ensure_collections(self):
        """Create collections if they don't exist."""
        try:
            # Check if collections exist
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            # Create query_patterns collection
            if self.query_collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.query_collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Created collection: {self.query_collection_name}")
            
            # Create user_preferences collection
            if self.user_prefs_collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.user_prefs_collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Created collection: {self.user_prefs_collection_name}")
        except Exception as e:
            print(f"⚠️  Error ensuring collections: {e}")
    
    def add_query_pattern(self, 
                         query: str,
                         intent: str,
                         solution: Dict[str, Any],
                         success: bool,
                         metadata: Optional[Dict] = None):
        """
        Store successful query pattern with solution.
        
        Args:
            query: User query text
            intent: Classified intent
            solution: What worked (SQL, model, strategy)
            success: Whether it succeeded
            metadata: Additional context
        """
        if not self.encoder:
            return
        
        # Generate embedding
        embedding = self.encoder.encode(query).tolist()
        
        # Create unique ID
        query_id = hashlib.md5(
            f"{query}_{intent}_{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        # Prepare payload (metadata in Qdrant)
        payload = {
            "query": query,
            "intent": intent,
            "solution": json.dumps(solution),
            "success": success,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {})
        }
        
        # Store in Qdrant
        self.client.upsert(
            collection_name=self.query_collection_name,
            points=[
                PointStruct(
                    id=query_id,
                    vector=embedding,
                    payload=payload
                )
            ]
        )
        
        print(f"   💾 Stored query pattern: {query[:50]}...")
    
    def search_similar_queries(self, 
                               query: str, 
                               top_k: int = 3,
                               min_similarity: float = 0.7,
                               only_successful: bool = True) -> List[Dict]:
        """
        Find similar past queries using semantic search.
        
        Args:
            query: Current query
            top_k: Number of results
            min_similarity: Minimum similarity threshold (0-1)
            only_successful: If True, only return queries that were successful
        
        Returns:
            List of similar queries with solutions
        """
        if not self.encoder:
            return []
        
        # Check Redis cache first
        if self.redis_client:
            cache_key = f"query:{hashlib.md5(query.encode()).hexdigest()}"
            cached = self.redis_client.get(cache_key)
            if cached:
                print("   ⚡ Cache hit!")
                return json.loads(cached)
        
        # Generate query embedding
        query_embedding = self.encoder.encode(query).tolist()
        
        # Build filter for successful queries only
        query_filter = None
        if only_successful:
            query_filter = QueryFilter(
                must=[
                    FieldCondition(
                        key="success",
                        match=MatchValue(value=True)
                    )
                ]
            )
        
        # Search in Qdrant using query method
        results = self.client.query(
            collection_name=self.query_collection_name,
            query=Query(
                vector=query_embedding,
                filter=query_filter,
                top=top_k * 2 if only_successful else top_k
            )
        )
        
        # Process results
        similar_queries = []
        for result in results:
            # Qdrant returns score (higher = more similar), convert to similarity
            # Cosine distance: score ranges from 0 to 1 (1 = identical)
            similarity = result.score
            
            if similarity >= min_similarity:
                payload = result.payload
                similar_queries.append({
                    "query": payload.get("query", ""),
                    "intent": payload.get("intent", ""),
                    "solution": json.loads(payload.get("solution", "{}")),
                    "similarity": round(similarity, 3),
                    "timestamp": payload.get("timestamp", "")
                })
        
        # Limit to top_k
        similar_queries = similar_queries[:top_k]
        
        # Cache in Redis
        if self.redis_client and similar_queries:
            cache_key = f"query:{hashlib.md5(query.encode()).hexdigest()}"
            self.redis_client.setex(
                cache_key, 
                3600,  # 1 hour TTL
                json.dumps(similar_queries)
            )
        
        return similar_queries
    
    def store_user_preference(self, 
                             user_id: str,
                             preference_type: str,
                             preference_value: Any):
        """Store user-specific preferences."""
        pref_id = f"{user_id}_{preference_type}"
        
        # Generate dummy embedding (user prefs don't need semantic search)
        dummy_embedding = [0.0] * self.embedding_dim
        
        payload = {
            "user_id": user_id,
            "type": preference_type,
            "value": json.dumps(preference_value),
            "timestamp": datetime.now().isoformat()
        }
        
        self.client.upsert(
            collection_name=self.user_prefs_collection_name,
            points=[
                PointStruct(
                    id=pref_id,
                    vector=dummy_embedding,
                    payload=payload
                )
            ]
        )
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Retrieve all preferences for a user."""
        # Scroll through collection with filter
        results = self.client.scroll(
            collection_name=self.user_prefs_collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            ),
            limit=100
        )
        
        preferences = {}
        for point in results[0]:  # results is (points, next_page_offset)
            payload = point.payload
            pref_type = payload.get("type", "")
            pref_value = json.loads(payload.get("value", "{}"))
            preferences[pref_type] = pref_value
        
        return preferences
    
    def get_stats(self) -> Dict[str, int]:
        """Get memory statistics."""
        try:
            query_count = self.client.get_collection(self.query_collection_name).points_count
            prefs_count = self.client.get_collection(self.user_prefs_collection_name).points_count
        except:
            query_count = 0
            prefs_count = 0
        
        return {
            "query_patterns_count": query_count,
            "user_preferences_count": prefs_count,
        }


# ============================================================================
# EPISODIC MEMORY (Experience Learning)
# ============================================================================

class EpisodicMemory:
    """
    Learn from past experiences (successes and failures).
    
    Stores:
    - What worked and what didn't
    - Model performance comparisons
    - Strategy effectiveness
    - Error patterns and fixes
    """
    
    def __init__(self, db_path: str = "episodic_memory.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for episodic memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                query_pattern TEXT NOT NULL,
                context TEXT,
                action_taken TEXT NOT NULL,
                outcome TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                learned_insight TEXT,
                confidence REAL,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_success ON episodes(success)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pattern ON episodes(query_pattern)
        """)
        
        conn.commit()
        conn.close()
        print(f"✅ Episodic memory initialized: {self.db_path}")
    
    def add_episode(self, 
                   query_pattern: str,
                   context: Dict[str, Any],
                   action_taken: str,
                   outcome: str,
                   success: bool,
                   learned_insight: str,
                   confidence: float = 0.8):
        """
        Record an episode (experience).
        
        Example:
            add_episode(
                query_pattern="forecast_product_at_branch",
                context={"product": "X", "branch": "Y", "model": "XGBoost"},
                action_taken="Used XGBoost with lag features",
                outcome="RMSE=85, better than Prophet (RMSE=120)",
                success=True,
                learned_insight="XGBoost works better for product X",
                confidence=0.9
            )
        """
        episode_id = hashlib.md5(
            f"{query_pattern}_{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO episodes 
            (id, timestamp, query_pattern, context, action_taken, outcome, 
             success, learned_insight, confidence, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            episode_id,
            datetime.now().isoformat(),
            query_pattern,
            json.dumps(context),
            action_taken,
            outcome,
            success,
            learned_insight,
            confidence,
            json.dumps({})
        ))
        
        conn.commit()
        conn.close()
        
        print(f"   📚 Learned: {learned_insight}")
    
    def recall_similar_episodes(self, 
                               query_pattern: str,
                               context: Optional[Dict] = None,
                               only_successful: bool = True,
                               limit: int = 5) -> List[EpisodicMemoryEntry]:
        """
        Recall similar past episodes.
        
        Returns lessons learned from similar situations.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT id, timestamp, query_pattern, context, action_taken, outcome,
                   success, learned_insight, confidence
            FROM episodes
            WHERE query_pattern LIKE ?
        """
        
        if only_successful:
            query += " AND success = 1"
        
        query += " ORDER BY confidence DESC, timestamp DESC LIMIT ?"
        
        cursor.execute(query, (f"%{query_pattern}%", limit))
        rows = cursor.fetchall()
        conn.close()
        
        episodes = []
        for row in rows:
            episodes.append(EpisodicMemoryEntry(
                id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                query_pattern=row[2],
                context=json.loads(row[3]),
                action_taken=row[4],
                outcome=row[5],
                success=bool(row[6]),
                learned_insight=row[7],
                confidence=row[8]
            ))
        
        return episodes
    
    def get_best_strategy(self, situation: str) -> Optional[str]:
        """Get the best known strategy for a situation."""
        episodes = self.recall_similar_episodes(
            situation, 
            only_successful=True, 
            limit=1
        )
        
        if episodes:
            return episodes[0].learned_insight
        return None
    
    def get_stats(self) -> Dict[str, int]:
        """Get episodic memory statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM episodes")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM episodes WHERE success = 1")
        successes = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_episodes": total,
            "successful_episodes": successes,
            "failure_episodes": total - successes,
            "success_rate": round(successes / total, 2) if total > 0 else 0.0
        }


# ============================================================================
# ADVANCED MEMORY MANAGER (Orchestrator)
# ============================================================================

class AdvancedMemoryManager:
    """
    Orchestrates all memory layers.
    
    Provides unified interface for:
    - Short-term (working memory)
    - Long-term (vector DB)
    - Episodic (learned experiences)
    """
    
    def __init__(self, 
                 persist_dir: str = "./memory",
                 use_redis: bool = False):
        
        print("\n🧠 Initializing Advanced Memory System...")
        
        # Create directories
        os.makedirs(persist_dir, exist_ok=True)
        
        # Initialize memory layers
        self.short_term = ShortTermMemory(max_size=10)
        print("   ✅ Short-term memory initialized")
        
        try:
            self.long_term = LongTermMemory(
                use_redis=use_redis
            )
            print("   ✅ Long-term memory (Vector DB) initialized")
        except ImportError as e:
            print(f"   ⚠️  Long-term memory unavailable: {e}")
            self.long_term = None
        except Exception as e:
            print(f"   ⚠️  Long-term memory initialization error: {e}")
            self.long_term = None
        
        self.episodic = EpisodicMemory(
            db_path=os.path.join(persist_dir, "episodic.db")
        )
        print("   ✅ Episodic memory initialized")
        
        print("🎉 Advanced Memory System ready!\n")
    
    def remember_interaction(self,
                           query: str,
                           intent: str,
                           result: Dict[str, Any],
                           success: bool):
        """
        Store interaction across all memory layers.
        
        Args:
            query: User query
            intent: Classified intent
            result: Query result with solution details
            success: Whether query succeeded
        """
        # Short-term memory
        self.short_term.add(ShortTermMemoryEntry(
            timestamp=datetime.now(),
            query=query,
            intent=intent,
            result_summary=str(result.get('summary', ''))[:200],
            success=success,
            metadata=result.get('metadata', {})
        ))
        
        # Long-term memory (only successful patterns)
        if success and self.long_term:
            solution = {
                "sql": result.get('sql', ''),
                "model": result.get('model', ''),
                "strategy": result.get('strategy', ''),
                "metrics": result.get('metrics', {})
            }
            
            self.long_term.add_query_pattern(
                query=query,
                intent=intent,
                solution=solution,
                success=success,
                metadata={"elapsed": result.get('elapsed_seconds', 0)}
            )
    
    def learn_from_experience(self,
                            situation: str,
                            action: str,
                            outcome: str,
                            success: bool,
                            insight: str,
                            context: Optional[Dict] = None,
                            confidence: float = 0.8):
        """
        Learn from an experience and store in episodic memory.
        
        Example:
            learn_from_experience(
                situation="forecast_sparse_data",
                action="used_prophet_instead_of_xgboost",
                outcome="better_accuracy",
                success=True,
                insight="Prophet handles sparse data better",
                context={"data_points": 5, "product": "X"},
                confidence=0.9
            )
        """
        self.episodic.add_episode(
            query_pattern=situation,
            context=context or {},
            action_taken=action,
            outcome=outcome,
            success=success,
            learned_insight=insight,
            confidence=confidence
        )
    
    def recall_similar(self, query: str, top_k: int = 3, only_successful: bool = True) -> List[Dict]:
        """
        Find similar past queries from long-term memory.
        
        Args:
            query: Query to search for
            top_k: Number of results
            only_successful: If True, only return successful queries
        
        Returns:
            List of similar queries with solutions
        """
        if not self.long_term:
            return []
        
        similar = self.long_term.search_similar_queries(
            query, 
            top_k=top_k,
            only_successful=only_successful
        )
        
        if similar:
            print(f"\n   🔍 Found {len(similar)} similar past queries:")
            for s in similar:
                success_indicator = "✅" if s.get('solution', {}).get('metadata', {}).get('success', True) else "❌"
                print(f"      {success_indicator} {s['query'][:50]}... (similarity: {s['similarity']})")
        
        return similar
    
    def get_learned_insights(self, situation: str) -> List[str]:
        """Get learned insights for a situation."""
        episodes = self.episodic.recall_similar_episodes(situation, limit=3)
        return [ep.learned_insight for ep in episodes]
    
    def get_best_strategy(self, situation: str) -> Optional[str]:
        """Get best known strategy for a situation."""
        return self.episodic.get_best_strategy(situation)
    
    def get_recent_context(self, n: int = 5) -> str:
        """Get recent conversation context."""
        return self.short_term.get_context()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        stats = {
            "short_term": {
                "size": len(self.short_term.entries),
                "max_size": self.short_term.max_size
            },
            "episodic": self.episodic.get_stats()
        }
        
        if self.long_term:
            stats["long_term"] = self.long_term.get_stats()
        
        return stats
    
    def clear_short_term(self):
        """Clear short-term memory (new session)."""
        self.short_term.clear()
        print("   🗑️ Short-term memory cleared")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def initialize_advanced_memory(persist_dir: str = "./memory",
                              use_redis: bool = False) -> AdvancedMemoryManager:
    """Initialize advanced memory system."""
    return AdvancedMemoryManager(persist_dir=persist_dir, use_redis=use_redis)


if __name__ == "__main__":
    # Test memory system
    print("="*80)
    print("🧪 TESTING ADVANCED MEMORY SYSTEM")
    print("="*80)
    
    # Initialize
    memory = initialize_advanced_memory()
    
    # Test short-term memory
    print("\n1️⃣  Testing Short-term Memory...")
    memory.short_term.add(ShortTermMemoryEntry(
        timestamp=datetime.now(),
        query="Dự báo chi nhánh Đà Nẵng",
        intent="FORECAST",
        result_summary="Successfully forecasted 30 days",
        success=True,
        metadata={"model": "XGBoost"}
    ))
    print(memory.get_recent_context())
    
    # Test long-term memory
    if memory.long_term:
        print("\n2️⃣  Testing Long-term Memory...")
        memory.long_term.add_query_pattern(
            query="Dự báo doanh số 30 ngày",
            intent="FORECAST",
            solution={"model": "XGBoost", "rmse": 85},
            success=True
        )
        
        similar = memory.recall_similar("Dự báo chi nhánh Hà Nội")
        print(f"   Found {len(similar)} similar queries")
    
    # Test episodic memory
    print("\n3️⃣  Testing Episodic Memory...")
    memory.learn_from_experience(
        situation="forecast_with_sparse_data",
        action="switched_from_xgboost_to_prophet",
        outcome="improved_accuracy_by_20_percent",
        success=True,
        insight="Prophet handles sparse data better than XGBoost",
        context={"data_points": 5},
        confidence=0.9
    )
    
    insights = memory.get_learned_insights("forecast")
    print(f"   Learned insights: {insights}")
    
    # Statistics
    print("\n4️⃣  Memory Statistics:")
    stats = memory.get_statistics()
    print(json.dumps(stats, indent=2))
    
    print("\n" + "="*80)
    print("✅ ADVANCED MEMORY SYSTEM TEST COMPLETE")
    print("="*80)


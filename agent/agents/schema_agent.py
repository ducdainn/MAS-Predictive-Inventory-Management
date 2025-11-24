"""
SchemaAgent: understands database schema and provides context to SQL generator.
"""

from typing import Optional

from agent.manager.database_manager import DatabaseManager
from agent.manager.memory_manager import MemoryManager


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
   - region: TEXT (MIỀN TRUNG, TÂY NAM BỘ, TÂY NGUYÊN, ĐÔNG NAM BỘ, HỒ CHÍ MINH)
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
- Region filter: WHERE region IN ('MIỀN TRUNG', 'TÂY NAM BỘ', 'TÂY NGUYÊN', 'ĐÔNG NAM BỘ', 'HỒ CHÍ MINH') (MIỀN TRUNG GỒM: 'MIỀN TRUNG', 'TÂY NGUYÊN', MIỀN NAM GỒM: 'TÂY NAM BỘ', 'ĐÔNG NAM BỘ', 'HỒ CHÍ MINH')
- Aggregations: GROUP BY date/branch_code/product_code
- Distance query: Find branches within X km for transfer optimization
"""

    def get_schema_context(self, question: str) -> str:
        """Get relevant schema context based on question."""
        context = self.schema_summary

        # Only get successful queries to avoid repeating failed patterns
        similar = self.memory.search_similar(question, top_k=3, only_successful=True)
        if similar:
            context += "\n\nSIMILAR PAST QUERIES (successful only):\n"
            for i, s in enumerate(similar, 1):
                # Handle both advanced memory format and fallback format
                if isinstance(s, dict):
                    sql = s.get('solution', {}).get('sql', '') or s.get('sql', '')
                    if sql:
                        context += f"{i}. {sql[:200]}...\n"
                elif hasattr(s, 'sql_query') and s.sql_query:
                    context += f"{i}. {s.sql_query[:200]}...\n"

        return context




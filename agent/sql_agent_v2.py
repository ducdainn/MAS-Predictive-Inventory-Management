"""
SQL Agent V2 - Drop-in replacement for improved_mas.py SQLAgent class
Copy this class definition and replace the SQLAgent class in improved_mas.py (lines 680-776)
"""

import re
from typing import Dict, List, Optional
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser


class SQLAgent:
    """
    Advanced SQL Agent with intelligent query generation and validation.
    
    Improvements over V1:
    1. Multi-retry mechanism (up to 3 attempts)
    2. Dry-run validation with EXPLAIN
    3. Parse DB errors for intelligent refinement
    4. Template-based generation for common patterns
    5. Schema-aware validation
    6. Graceful degradation (never fails completely)
    """
    
    def __init__(self, llm_provider, schema_agent, db_manager):
        """
        Initialize SQL Agent with LLM, schema, and database access.
        
        Args:
            llm_provider: LLMProvider instance
            schema_agent: SchemaAgent instance
            db_manager: DatabaseManager instance for dry-run validation
        """
        self.llm = llm_provider.get_llm("openai", temperature=0.0)
        self.schema_agent = schema_agent
        self.db = db_manager
        self.max_retries = 3
        
        # Common error patterns for classification
        self.error_patterns = {
            'column.*does not exist': 'Column name mismatch',
            'relation.*does not exist': 'Table name mismatch',
            'table.*does not exist': 'Table name mismatch',
            'syntax error': 'SQL syntax error',
            'must appear in the group by': 'Missing GROUP BY clause',
            'aggregate function': 'Missing GROUP BY clause',
            'ambiguous column': 'Ambiguous column reference',
            'division by zero': 'Division by zero error',
        }
        
        # Query templates for common patterns (fast path)
        self.query_templates = {
            'top_products': """
SELECT 
    p.product_name,
    p.product_code,
    SUM(s.quantity) as total_quantity,
    SUM(s.quantity * s.selling_price) as total_revenue
FROM sales s
JOIN product p ON s.product_code = p.product_code
WHERE s.date >= CURRENT_DATE - INTERVAL '{days} days'
GROUP BY p.product_code, p.product_name
ORDER BY total_revenue DESC
LIMIT {limit}
""",
            'branch_performance': """
SELECT 
    b.branch_name,
    b.region,
    COUNT(DISTINCT s.product_code) as product_count,
    SUM(s.quantity) as total_quantity,
    SUM(s.quantity * s.selling_price) as total_revenue
FROM sales s
JOIN branch b ON s.branch_code = b.branch_code
WHERE s.date >= CURRENT_DATE - INTERVAL '{days} days'
GROUP BY b.branch_code, b.branch_name, b.region
ORDER BY total_revenue DESC
LIMIT 20
""",
            'inventory_low_stock': """
SELECT 
    p.product_name,
    b.branch_name,
    i.quantity_on_hand,
    i.reorder_point,
    (i.reorder_point - i.quantity_on_hand) as shortage
FROM inventory i
JOIN product p ON i.product_code = p.product_code
JOIN branch b ON i.branch_code = b.branch_code
WHERE i.quantity_on_hand <= i.reorder_point
ORDER BY shortage DESC
LIMIT 50
""",
        }
    
    def generate_sql(self, question: str, intent: str) -> str:
        """
        Generate SQL query with advanced validation and retry mechanism.
        
        Flow:
        1. Try template matching (fast path)
        2. Generate SQL using LLM
        3. Clean and validate syntax
        4. Dry-run with EXPLAIN to catch DB errors
        5. If error, parse and retry with refinement
        6. Return validated query or best attempt
        
        Args:
            question: User's natural language question
            intent: FORECAST or ANALYTICS
            
        Returns:
            Valid SQL SELECT query string
        """
        print(f"🔍 SQL Agent: Processing question (intent={intent})")
        
        # Step 1: Try template matching (fast path)
        template_sql = self._try_template_matching(question, intent)
        if template_sql:
            print("✅ SQL Agent: Using template (fast path)")
            return template_sql
        
        # Step 2: Generate with LLM (with retry)
        last_sql = None
        last_error = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"📝 SQL Agent: Attempt {attempt}/{self.max_retries}")
                
                # Generate query
                if attempt == 1:
                    sql = self._generate_initial_query(question, intent)
                else:
                    # Retry with error context
                    sql = self._generate_refined_query(
                        question, intent, last_sql, last_error
                    )
                
                # Clean
                sql = self._clean_sql(sql)
                
                # Validate syntax
                self._validate_sql(sql)
                print("   ✓ Syntax valid")
                
                # Dry-run with EXPLAIN (catch DB errors early)
                self._dry_run_query(sql)
                print("   ✓ Dry-run passed")
                
                print(f"✅ SQL Agent: Success on attempt {attempt}")
                return sql
                
            except Exception as e:
                last_sql = sql if 'sql' in locals() else None
                last_error = str(e)
                error_type = self._classify_error(last_error)
                
                print(f"   ❌ {error_type}")
                if len(last_error) < 150:
                    print(f"   {last_error}")
                
                if attempt < self.max_retries:
                    print(f"   🔄 Retrying with error feedback...")
                else:
                    # Graceful degradation
                    print(f"   ⚠️ Max retries reached, using last attempt")
                    if last_sql and self._is_safe_query(last_sql):
                        return last_sql
                    else:
                        return self._generate_fallback_query(question)
    
    def _try_template_matching(self, question: str, intent: str) -> Optional[str]:
        """Try to match question to a predefined template (fast path)."""
        q_lower = question.lower()
        
        # Top products pattern
        if any(kw in q_lower for kw in ['top', 'best', 'bán chạy', 'nhiều nhất', 'cao nhất']):
            if any(kw in q_lower for kw in ['product', 'sản phẩm', 'hàng', 'mặt hàng']):
                days = self._extract_time_days(question)
                limit = self._extract_limit(question, default=10)
                return self.query_templates['top_products'].format(days=days, limit=limit)
        
        # Branch performance pattern
        if any(kw in q_lower for kw in ['chi nhánh', 'branch', 'cửa hàng', 'store']):
            if any(kw in q_lower for kw in ['doanh thu', 'revenue', 'performance', 'hiệu suất', 'bán']):
                days = self._extract_time_days(question)
                return self.query_templates['branch_performance'].format(days=days)
        
        # Low stock pattern
        if any(kw in q_lower for kw in ['tồn kho thấp', 'hết hàng', 'low stock', 'out of stock', 'thiếu hàng']):
            return self.query_templates['inventory_low_stock']
        
        return None
    
    def _extract_time_days(self, text: str) -> int:
        """Extract time period in days from text."""
        text_lower = text.lower()
        
        # Extract number + unit
        if 'ngày' in text_lower or 'day' in text_lower:
            match = re.search(r'(\d+)\s*(ngày|day)', text_lower)
            if match:
                return int(match.group(1))
        
        if 'tháng' in text_lower or 'month' in text_lower:
            match = re.search(r'(\d+)\s*(tháng|month)', text_lower)
            if match:
                return int(match.group(1)) * 30
        
        if 'tuần' in text_lower or 'week' in text_lower:
            match = re.search(r'(\d+)\s*(tuần|week)', text_lower)
            if match:
                return int(match.group(1)) * 7
        
        # Default
        return 30
    
    def _extract_limit(self, text: str, default: int = 10) -> int:
        """Extract LIMIT value from text."""
        match = re.search(r'top\s*(\d+)|(\d+)\s*top', text, re.IGNORECASE)
        if match:
            return int(match.group(1) or match.group(2))
        
        # Look for any number
        numbers = re.findall(r'\d+', text)
        if numbers:
            first_num = int(numbers[0])
            if 5 <= first_num <= 100:  # Reasonable range
                return first_num
        
        return default
    
    def _generate_initial_query(self, question: str, intent: str) -> str:
        """Generate initial SQL query using LLM."""
        schema_context = self.schema_agent.get_schema_context(question)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt(intent)),
            ("human", """Schema:
{schema}

Question: {question}

Generate a PostgreSQL SELECT query. Rules:
1. Return ONLY the SQL query (no explanations)
2. Start with SELECT or WITH
3. Use EXACT table/column names from schema
4. Use proper JOINs with ON conditions
5. Use table aliases (s, p, b, i) for clarity

SQL Query:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"schema": schema_context, "question": question})
    
    def _generate_refined_query(self, question: str, intent: str, 
                                last_sql: str, last_error: str) -> str:
        """Generate refined query based on previous error."""
        schema_context = self.schema_agent.get_schema_context(question)
        error_hints = self._get_error_hints(last_error, last_sql)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a PostgreSQL expert. Fix the failed query based on error feedback."),
            ("human", """Schema:
{schema}

Question: {question}

Previous SQL (FAILED):
{last_sql}

Error: {error_msg}

Hints to fix:
{hints}

Generate a CORRECTED query. Use EXACT names from schema.
Return ONLY the SQL query.

SQL Query:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "schema": schema_context,
            "question": question,
            "last_sql": last_sql,
            "error_msg": last_error[:250],
            "hints": error_hints
        })
    
    def _get_system_prompt(self, intent: str) -> str:
        """Get system prompt for SQL generation."""
        base = """You are a PostgreSQL expert for inventory management systems.

CRITICAL RULES:
1. Return ONLY a valid SELECT query
2. No explanations, markdown, or code fences
3. Start with SELECT or WITH
4. Use EXACT table/column names from schema
5. Use proper JOINs: JOIN table ON condition
6. Use table aliases: FROM sales s, JOIN product p
7. GROUP BY all non-aggregated columns when using SUM/AVG/COUNT
8. ORDER BY for sorted results
9. LIMIT for top-N queries
10. Date filters: WHERE date >= CURRENT_DATE - INTERVAL '30 days'

Tables: sales, product, branch, inventory, branch_distance
"""
        
        if intent == "FORECAST":
            base += """
FORECAST QUERIES:
- Must include date column
- Filter: >= CURRENT_DATE - INTERVAL '90 days'
- GROUP BY date (for time series)
- ORDER BY date ASC
- Include quantity or revenue metrics

Example:
SELECT date, SUM(quantity) as total_qty 
FROM sales 
WHERE date >= CURRENT_DATE - INTERVAL '90 days' 
GROUP BY date 
ORDER BY date ASC
"""
        else:
            base += """
ANALYTICS QUERIES:
- Use aggregations: SUM(quantity), COUNT(*), AVG(price)
- GROUP BY dimensions: branch_name, product_name, region
- JOIN to get readable names
- ORDER BY aggregate DESC
- LIMIT for top-N

Example:
SELECT p.product_name, SUM(s.quantity) as total
FROM sales s
JOIN product p ON s.product_code = p.product_code
WHERE s.date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY p.product_name
ORDER BY total DESC
LIMIT 10
"""
        
        return base
    
    def _clean_sql(self, sql: str) -> str:
        """Clean and extract SQL from LLM output."""
        # Remove markdown code fences
        sql = re.sub(r'```(?:sql)?\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'```\s*$', '', sql)
        
        # Remove common prefixes
        sql = re.sub(r'^\s*(?:SQL Query:|Query:|SQL:)\s*', '', sql, flags=re.IGNORECASE)
        
        # Extract SELECT/WITH to end or semicolon
        match = re.search(r'((?:WITH|SELECT)\b.*?)(?:;|\Z)', sql, re.IGNORECASE | re.DOTALL)
        if match:
            sql = match.group(1)
        
        return sql.strip()
    
    def _validate_sql(self, sql: str):
        """Validate SQL syntax and safety."""
        sql_upper = sql.upper()
        
        # Check forbidden keywords (security)
        forbidden = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'TRUNCATE', 
                    'ALTER', 'CREATE', 'GRANT', 'REVOKE']
        for keyword in forbidden:
            if re.search(rf'\b{keyword}\b', sql_upper):
                raise ValueError(f"Forbidden keyword: {keyword}")
        
        # Must start with SELECT or WITH
        if not re.match(r'^\s*(SELECT|WITH)\b', sql_upper):
            raise ValueError("Query must start with SELECT or WITH")
        
        # Basic syntax checks
        if sql.count('(') != sql.count(')'):
            raise ValueError("Mismatched parentheses")
        
        if len(sql.strip()) < 15:
            raise ValueError("Query too short")
    
    def _dry_run_query(self, sql: str):
        """
        Dry-run query using EXPLAIN to catch errors without executing.
        Raises exception if query has errors.
        """
        try:
            explain_sql = f"EXPLAIN {sql}"
            result = self.db.execute_query(explain_sql)
            return result
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Parse PostgreSQL errors for better messages
            if 'column' in error_msg and 'does not exist' in error_msg:
                col_match = re.search(r'column "([^"]+)"', str(e), re.IGNORECASE)
                col_name = col_match.group(1) if col_match else 'unknown'
                raise ValueError(f"Column '{col_name}' does not exist in schema")
            
            elif 'relation' in error_msg and 'does not exist' in error_msg:
                table_match = re.search(r'relation "([^"]+)"', str(e), re.IGNORECASE)
                table_name = table_match.group(1) if table_match else 'unknown'
                raise ValueError(f"Table '{table_name}' does not exist")
            
            elif 'must appear in the group by' in error_msg:
                raise ValueError("Aggregate error: Add missing columns to GROUP BY")
            
            elif 'ambiguous' in error_msg:
                raise ValueError("Ambiguous column: Use table aliases (e.g., s.date)")
            
            else:
                raise ValueError(f"Query validation error: {str(e)[:200]}")
    
    def _is_safe_query(self, sql: str) -> bool:
        """Check if query is safe (no forbidden keywords)."""
        try:
            self._validate_sql(sql)
            return True
        except:
            return False
    
    def _classify_error(self, error: str) -> str:
        """Classify error type for better handling."""
        error_lower = error.lower()
        
        for pattern, error_type in self.error_patterns.items():
            if re.search(pattern, error_lower):
                return error_type
        
        return "Query error"
    
    def _analyze_error(self, error: str, sql: str) -> str:
        """Analyze error and provide diagnosis."""
        error_type = self._classify_error(error)
        
        if error_type == "Column name mismatch":
            return "Column name does not match schema. Check spelling and use exact names."
        elif error_type == "Table name mismatch":
            return "Table name does not match schema. Valid: sales, product, branch, inventory"
        elif error_type == "Missing GROUP BY clause":
            return "Using aggregates requires GROUP BY. Add all non-aggregated columns to GROUP BY."
        elif error_type == "Ambiguous column reference":
            return "Column exists in multiple tables. Use table alias: e.g., s.date, p.name"
        else:
            return "Check query syntax and schema carefully."
    
    def _get_error_hints(self, error: str, sql: str) -> str:
        """Get specific hints to fix the error."""
        error_type = self._classify_error(error)
        
        if error_type == "Column name mismatch":
            return "- Check exact column names in schema\n- Verify spelling\n- Use table alias if needed"
        elif error_type == "Table name mismatch":
            return "- Use: sales, product, branch, inventory, branch_distance\n- Check spelling"
        elif error_type == "Missing GROUP BY clause":
            return "- Add: GROUP BY column1, column2, ...\n- Include all SELECT columns except aggregates"
        elif error_type == "Ambiguous column reference":
            return "- Add table aliases: FROM sales s\n- Qualify columns: s.date, p.product_name"
        else:
            return "- Review PostgreSQL syntax\n- Compare with schema structure"
    
    def _generate_fallback_query(self, question: str) -> str:
        """Generate a safe fallback query if all retries fail."""
        print("⚠️ SQL Agent: Using fallback query")
        
        # Simple safe query that always works
        return """
SELECT 
    date,
    SUM(quantity) as total_quantity,
    COUNT(*) as transaction_count,
    SUM(quantity * selling_price) as total_revenue
FROM sales
WHERE date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY date
ORDER BY date DESC
LIMIT 100
"""
    
    # ============================================================================
    # LEGACY COMPATIBILITY (for backward compatibility)
    # ============================================================================
    
    def _retry_generate_sql(self, question: str, schema: str, error: str) -> str:
        """Legacy retry method (for backward compatibility)."""
        return self._generate_refined_query(question, "ANALYTICS", 
                                           "SELECT 1", error)



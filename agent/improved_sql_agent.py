"""
Improved SQL Agent with Advanced Query Generation and Validation
Features:
- Multi-retry mechanism with intelligent error recovery
- Dry-run validation using EXPLAIN
- Schema-aware query refinement
- DB error parsing and feedback
- Query complexity analysis
- Template-based generation for common patterns
"""

import re
from typing import Dict, List, Optional, Tuple
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser


class ImprovedSQLAgent:
    """
    Advanced SQL Agent with intelligent query generation and validation.
    
    Improvements over basic SQL Agent:
    1. Multi-level retry (up to 3 attempts)
    2. Dry-run validation with EXPLAIN
    3. Parse DB errors for intelligent refinement
    4. Schema-aware validation
    5. Query complexity checks
    6. Template-based generation for common patterns
    """
    
    def __init__(self, llm_provider, schema_agent, db_manager):
        self.llm = llm_provider.get_llm("openai", temperature=0.0)
        self.schema_agent = schema_agent
        self.db = db_manager
        self.max_retries = 3
        
        # Common error patterns and solutions
        self.error_patterns = {
            'column.*does not exist': 'Column name mismatch',
            'table.*does not exist': 'Table name mismatch',
            'syntax error': 'SQL syntax error',
            'aggregate function': 'Missing GROUP BY clause',
            'ambiguous column': 'Need table alias or qualification',
            'division by zero': 'Need NULLIF or CASE for division',
        }
        
        # Query templates for common patterns
        self.query_templates = {
            'top_products': """
SELECT 
    p.product_name,
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
""",
            'inventory_status': """
SELECT 
    p.product_name,
    b.branch_name,
    i.quantity_on_hand,
    i.reorder_point,
    CASE 
        WHEN i.quantity_on_hand <= i.reorder_point THEN 'LOW'
        WHEN i.quantity_on_hand <= i.reorder_point * 1.5 THEN 'MEDIUM'
        ELSE 'GOOD'
    END as stock_status
FROM inventory i
JOIN product p ON i.product_code = p.product_code
JOIN branch b ON i.branch_code = b.branch_code
WHERE i.quantity_on_hand <= i.reorder_point * 2
ORDER BY stock_status, i.quantity_on_hand
"""
        }
    
    def generate_sql(self, question: str, intent: str) -> str:
        """
        Generate SQL query with advanced validation and retry mechanism.
        
        Flow:
        1. Check if question matches a template pattern
        2. Generate SQL using LLM
        3. Clean and validate syntax
        4. Dry-run with EXPLAIN to catch DB errors
        5. If error, parse and retry with refinement
        6. Return validated query or raise detailed error
        """
        print(f"🔍 Generating SQL for: {question[:80]}...")
        
        # Step 1: Try template matching (fast path)
        template_sql = self._try_template_matching(question, intent)
        if template_sql:
            print("✅ Using query template (fast path)")
            return template_sql
        
        # Step 2: Generate with LLM (with retry)
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"📝 Attempt {attempt}/{self.max_retries}: Generating SQL...")
                
                # Generate query
                if attempt == 1:
                    sql = self._generate_initial_query(question, intent)
                else:
                    # Retry with error context
                    sql = self._generate_refined_query(
                        question, intent, last_sql, last_error, attempt
                    )
                
                # Clean
                sql = self._clean_sql(sql)
                print(f"   SQL: {sql[:100]}...")
                
                # Validate syntax
                self._validate_sql_syntax(sql)
                print("   ✓ Syntax valid")
                
                # Validate semantics
                self._validate_sql_semantics(sql)
                print("   ✓ Semantics valid")
                
                # Dry-run with EXPLAIN
                explain_result = self._dry_run_query(sql)
                print("   ✓ Dry-run successful")
                
                # Check complexity
                complexity = self._check_query_complexity(explain_result)
                if complexity > 1000:
                    print(f"   ⚠️ High complexity: {complexity} (but acceptable)")
                
                print(f"✅ SQL generated successfully on attempt {attempt}")
                return sql
                
            except Exception as e:
                last_sql = sql if 'sql' in locals() else None
                last_error = str(e)
                error_type = self._classify_error(last_error)
                
                print(f"   ❌ Error: {error_type}")
                print(f"   Details: {last_error[:100]}...")
                
                if attempt < self.max_retries:
                    print(f"   🔄 Retrying with error feedback...")
                else:
                    print(f"   ❌ Max retries reached")
                    raise ValueError(
                        f"Failed to generate valid SQL after {self.max_retries} attempts.\n"
                        f"Last error: {last_error}\n"
                        f"Last SQL: {last_sql}"
                    )
    
    def _try_template_matching(self, question: str, intent: str) -> Optional[str]:
        """Try to match question to a predefined template."""
        question_lower = question.lower()
        
        # Top products pattern
        if any(kw in question_lower for kw in ['top', 'best', 'sản phẩm bán chạy', 'best seller']):
            if any(kw in question_lower for kw in ['product', 'sản phẩm', 'hàng']):
                # Extract days and limit
                days = self._extract_number(question, default=30, keywords=['day', 'ngày', 'tháng'])
                if any(kw in question_lower for kw in ['tháng', 'month']):
                    days = days * 30
                limit = self._extract_number(question, default=10, keywords=['top'])
                
                return self.query_templates['top_products'].format(days=days, limit=limit)
        
        # Branch performance pattern
        if any(kw in question_lower for kw in ['branch', 'chi nhánh', 'cửa hàng', 'store']):
            if any(kw in question_lower for kw in ['performance', 'revenue', 'doanh thu', 'hiệu suất']):
                days = self._extract_number(question, default=30, keywords=['day', 'ngày', 'tháng'])
                if any(kw in question_lower for kw in ['tháng', 'month']):
                    days = days * 30
                
                return self.query_templates['branch_performance'].format(days=days)
        
        # Inventory status pattern
        if any(kw in question_lower for kw in ['inventory', 'tồn kho', 'stock', 'low stock']):
            if any(kw in question_lower for kw in ['status', 'level', 'mức', 'tình trạng', 'thấp']):
                return self.query_templates['inventory_status']
        
        return None
    
    def _extract_number(self, text: str, default: int, keywords: List[str]) -> int:
        """Extract number from text, return default if not found."""
        # Try to find number near keywords
        for keyword in keywords:
            pattern = rf'(\d+)\s*{keyword}|{keyword}\s*(\d+)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                num = match.group(1) or match.group(2)
                return int(num)
        
        # Try to find any number
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[0])
        
        return default
    
    def _generate_initial_query(self, question: str, intent: str) -> str:
        """Generate initial SQL query."""
        schema_context = self.schema_agent.get_schema_context(question)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt(intent)),
            ("human", """Schema:
{schema}

Question: {question}

Generate a PostgreSQL query. Follow these rules:
1. Return ONLY the SQL query
2. No explanations, no markdown, no code fences
3. Start with SELECT or WITH
4. Use proper JOINs based on schema relationships
5. Include appropriate WHERE clauses for filtering
6. Use meaningful column aliases
7. Add ORDER BY for rankings
8. Add LIMIT for top-N queries

SQL Query:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"schema": schema_context, "question": question})
    
    def _generate_refined_query(self, question: str, intent: str, 
                                last_sql: str, last_error: str, attempt: int) -> str:
        """Generate refined query based on previous error."""
        schema_context = self.schema_agent.get_schema_context(question)
        error_analysis = self._analyze_error(last_error, last_sql)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_refinement_prompt(intent)),
            ("human", """Schema:
{schema}

Question: {question}

Previous SQL (FAILED):
{last_sql}

Error Type: {error_type}
Error Message: {error_msg}

Error Analysis:
{error_analysis}

Suggestions to fix:
{suggestions}

Generate a CORRECTED PostgreSQL query that fixes the error.
Return ONLY the SQL query, no explanations.

SQL Query:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "schema": schema_context,
            "question": question,
            "last_sql": last_sql,
            "error_type": self._classify_error(last_error),
            "error_msg": last_error[:300],
            "error_analysis": error_analysis,
            "suggestions": self._get_error_suggestions(last_error, last_sql)
        })
    
    def _get_system_prompt(self, intent: str) -> str:
        """Get system prompt for initial generation."""
        base = """You are an expert PostgreSQL query generator for inventory management.

CRITICAL RULES:
1. Return ONLY a valid SELECT query - no explanations, no markdown
2. Start with SELECT or WITH
3. Use EXACT table and column names from schema
4. Always use proper JOINs with ON conditions
5. Use table aliases for clarity
6. Include WHERE clauses with date filters using INTERVAL
7. GROUP BY when using aggregates (SUM, AVG, COUNT)
8. ORDER BY for sorted results
9. LIMIT for top-N queries
10. Handle NULL values with COALESCE or NULLIF

Common PostgreSQL syntax:
- Date filter: WHERE date >= CURRENT_DATE - INTERVAL '30 days'
- Safe division: quantity / NULLIF(total, 0)
- String matching: WHERE name ILIKE '%pattern%'
- Top N: ORDER BY value DESC LIMIT 10
"""
        
        if intent == "FORECAST":
            base += """
FORECAST QUERIES:
- Must include time series data (date column)
- Filter: at least 90 days of history
- Group by date to get daily/monthly aggregates
- Include product_code and branch_code for granularity
- Order by date ASC (chronological)
- Include quantity, revenue, or metrics needed for forecasting

Example:
SELECT 
    date,
    product_code,
    branch_code,
    SUM(quantity) as total_quantity,
    SUM(quantity * selling_price) as revenue
FROM sales
WHERE date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY date, product_code, branch_code
ORDER BY date ASC
"""
        else:
            base += """
ANALYTICS QUERIES:
- Use appropriate aggregations (SUM, COUNT, AVG)
- Group by dimensions (branch_name, product_name, region, category)
- Include JOINs to get descriptive names
- Add ORDER BY for rankings (DESC for top, ASC for bottom)
- LIMIT results for top-N queries
- Use CASE WHEN for categorization

Example:
SELECT 
    b.branch_name,
    p.product_name,
    SUM(s.quantity) as total_sold,
    SUM(s.quantity * s.selling_price) as revenue
FROM sales s
JOIN branch b ON s.branch_code = b.branch_code
JOIN product p ON s.product_code = p.product_code
WHERE s.date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY b.branch_name, p.product_name
ORDER BY revenue DESC
LIMIT 20
"""
        
        return base
    
    def _get_refinement_prompt(self, intent: str) -> str:
        """Get system prompt for query refinement."""
        return """You are a PostgreSQL error fixing expert.

Your task: Fix a failed SQL query based on the error message.

CRITICAL RULES:
1. Analyze the error carefully
2. Check column and table names match schema EXACTLY
3. Fix syntax errors
4. Add missing GROUP BY for aggregates
5. Add table aliases if columns are ambiguous
6. Use NULLIF for divisions to avoid divide-by-zero
7. Return ONLY the corrected SQL query
8. No explanations, no markdown

Common fixes:
- "column does not exist" → Check exact column name in schema
- "table does not exist" → Check exact table name in schema
- "aggregate function" → Add GROUP BY clause
- "ambiguous column" → Add table alias (e.g., s.date instead of date)
- "syntax error near" → Check PostgreSQL syntax
- "division by zero" → Use NULLIF: value / NULLIF(divisor, 0)
"""
    
    def _clean_sql(self, sql: str) -> str:
        """Clean and extract SQL from LLM output."""
        # Remove markdown code fences
        sql = re.sub(r'```(?:sql)?\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'```\s*$', '', sql)
        
        # Remove common prefixes
        sql = re.sub(r'^\s*(?:SQL Query:|Query:|SQL:)\s*', '', sql, flags=re.IGNORECASE)
        
        # Extract SELECT/WITH to end or semicolon
        match = re.search(r'((?:WITH|SELECT)\b.*?)(?:;|$)', sql, re.IGNORECASE | re.DOTALL)
        if match:
            sql = match.group(1)
        
        # Clean whitespace
        sql = sql.strip()
        
        # Remove trailing semicolon if present
        sql = sql.rstrip(';').strip()
        
        return sql
    
    def _validate_sql_syntax(self, sql: str):
        """Validate SQL syntax and safety."""
        sql_upper = sql.upper()
        
        # 1. Check forbidden keywords
        forbidden = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'TRUNCATE', 
                    'ALTER', 'CREATE', 'GRANT', 'REVOKE', 'EXEC']
        for keyword in forbidden:
            if re.search(rf'\b{keyword}\b', sql_upper):
                raise ValueError(f"Forbidden keyword detected: {keyword}")
        
        # 2. Must start with SELECT or WITH
        if not re.match(r'^\s*(SELECT|WITH)\b', sql_upper):
            raise ValueError("Query must start with SELECT or WITH")
        
        # 3. Check for obvious syntax errors
        if sql.count('(') != sql.count(')'):
            raise ValueError("Mismatched parentheses")
        
        # 4. Check for empty query
        if len(sql.strip()) < 10:
            raise ValueError("Query too short, likely invalid")
    
    def _validate_sql_semantics(self, sql: str):
        """Validate SQL semantics against schema."""
        sql_upper = sql.upper()
        
        # Get all tables mentioned in query
        table_pattern = r'\b(?:FROM|JOIN)\s+(\w+)'
        mentioned_tables = re.findall(table_pattern, sql_upper)
        
        # Valid tables from schema
        valid_tables = ['SALES', 'PRODUCT', 'BRANCH', 'INVENTORY', 'BRANCH_DISTANCE']
        
        for table in mentioned_tables:
            if table not in valid_tables:
                raise ValueError(
                    f"Unknown table: {table}. "
                    f"Valid tables: {', '.join(valid_tables)}"
                )
    
    def _dry_run_query(self, sql: str) -> Dict:
        """
        Dry-run query using EXPLAIN to catch errors without executing.
        Returns EXPLAIN output for complexity analysis.
        """
        try:
            explain_sql = f"EXPLAIN (FORMAT JSON) {sql}"
            result = self.db.execute_query(explain_sql)
            
            if result.empty:
                raise ValueError("EXPLAIN returned empty result")
            
            # Extract plan from result
            plan = result.iloc[0, 0] if not result.empty else {}
            return plan
            
        except Exception as e:
            error_msg = str(e)
            
            # Parse PostgreSQL error
            if 'column' in error_msg.lower() and 'does not exist' in error_msg.lower():
                # Extract column name
                col_match = re.search(r'column "([^"]+)"', error_msg, re.IGNORECASE)
                col_name = col_match.group(1) if col_match else 'unknown'
                raise ValueError(
                    f"Column '{col_name}' does not exist. "
                    f"Check schema for correct column names."
                )
            
            elif 'table' in error_msg.lower() and 'does not exist' in error_msg.lower():
                # Extract table name
                table_match = re.search(r'relation "([^"]+)"', error_msg, re.IGNORECASE)
                table_name = table_match.group(1) if table_match else 'unknown'
                raise ValueError(
                    f"Table '{table_name}' does not exist. "
                    f"Valid tables: sales, product, branch, inventory, branch_distance"
                )
            
            elif 'must appear in the group by' in error_msg.lower():
                raise ValueError(
                    "Aggregate function error: All non-aggregated columns must be in GROUP BY clause. "
                    "Add missing columns to GROUP BY."
                )
            
            elif 'ambiguous' in error_msg.lower():
                raise ValueError(
                    "Ambiguous column reference. Use table aliases (e.g., s.date instead of date)."
                )
            
            elif 'syntax error' in error_msg.lower():
                raise ValueError(f"SQL syntax error: {error_msg}")
            
            else:
                # Unknown error, pass through
                raise ValueError(f"Query validation failed: {error_msg}")
    
    def _check_query_complexity(self, explain_plan: Dict) -> int:
        """
        Estimate query complexity from EXPLAIN plan.
        Returns estimated cost (higher = more complex).
        """
        try:
            if isinstance(explain_plan, list) and len(explain_plan) > 0:
                plan = explain_plan[0].get('Plan', {})
                total_cost = plan.get('Total Cost', 0)
                return int(total_cost)
        except:
            pass
        
        return 0
    
    def _classify_error(self, error: str) -> str:
        """Classify error type for better handling."""
        error_lower = error.lower()
        
        for pattern, error_type in self.error_patterns.items():
            if re.search(pattern, error_lower):
                return error_type
        
        return "Unknown error"
    
    def _analyze_error(self, error: str, sql: str) -> str:
        """Analyze error and provide detailed diagnosis."""
        error_type = self._classify_error(error)
        analysis = []
        
        if error_type == "Column name mismatch":
            # Try to find mentioned column
            col_match = re.search(r'column "([^"]+)"', error, re.IGNORECASE)
            if col_match:
                bad_col = col_match.group(1)
                analysis.append(f"Column '{bad_col}' not found in schema.")
                analysis.append("Check schema for correct column names.")
                # Suggest similar columns
                analysis.append("Hint: Use exact names from schema (e.g., 'product_name', 'branch_code')")
        
        elif error_type == "Table name mismatch":
            table_match = re.search(r'relation "([^"]+)"', error, re.IGNORECASE)
            if table_match:
                bad_table = table_match.group(1)
                analysis.append(f"Table '{bad_table}' not found.")
                analysis.append("Valid tables: sales, product, branch, inventory, branch_distance")
        
        elif error_type == "Missing GROUP BY clause":
            analysis.append("Using aggregate functions (SUM, AVG, COUNT) requires GROUP BY.")
            analysis.append("All columns in SELECT that are not aggregated must be in GROUP BY.")
        
        elif error_type == "Need table alias or qualification":
            analysis.append("Column name is ambiguous (exists in multiple tables).")
            analysis.append("Use table alias: e.g., 's.date' instead of 'date'")
        
        elif error_type == "SQL syntax error":
            analysis.append("PostgreSQL syntax error detected.")
            analysis.append("Check: commas, parentheses, JOIN syntax, WHERE clause")
        
        return "\n".join(analysis) if analysis else "Error analysis not available."
    
    def _get_error_suggestions(self, error: str, sql: str) -> str:
        """Get specific suggestions to fix the error."""
        error_type = self._classify_error(error)
        suggestions = []
        
        if error_type == "Column name mismatch":
            suggestions.append("1. Check schema for exact column names")
            suggestions.append("2. Verify table has the column you're trying to use")
            suggestions.append("3. Use table alias if ambiguous: table_alias.column_name")
        
        elif error_type == "Table name mismatch":
            suggestions.append("1. Use exact table names: sales, product, branch, inventory, branch_distance")
            suggestions.append("2. Check spelling and case")
        
        elif error_type == "Missing GROUP BY clause":
            suggestions.append("1. Add all non-aggregated SELECT columns to GROUP BY")
            suggestions.append("2. Example: GROUP BY product_code, branch_code, date")
        
        elif error_type == "Need table alias or qualification":
            suggestions.append("1. Add table aliases in FROM/JOIN: FROM sales s")
            suggestions.append("2. Qualify columns: s.date, p.product_name")
        
        elif error_type == "SQL syntax error":
            suggestions.append("1. Check parentheses are balanced")
            suggestions.append("2. Verify commas between columns")
            suggestions.append("3. Check JOIN syntax: JOIN table ON condition")
            suggestions.append("4. Verify WHERE clause syntax")
        
        else:
            suggestions.append("1. Review query syntax carefully")
            suggestions.append("2. Compare with schema structure")
            suggestions.append("3. Check for typos in table/column names")
        
        return "\n".join(suggestions)


# ============================================================================
# INTEGRATION HELPER
# ============================================================================

def upgrade_sql_agent(orchestrator):
    """
    Helper function to upgrade an existing OrchestratorAgent's SQL agent.
    
    Usage:
        from agent.improved_sql_agent import upgrade_sql_agent
        upgrade_sql_agent(orchestrator)
    """
    improved_sql = ImprovedSQLAgent(
        llm_provider=orchestrator.sql_agent.llm.__class__.__name__,  # Assuming LLMProvider access
        schema_agent=orchestrator.schema_agent,
        db_manager=orchestrator.db
    )
    
    # Replace old SQL agent
    orchestrator.sql_agent = improved_sql
    
    print("✅ SQL Agent upgraded to ImprovedSQLAgent")
    print("   - Multi-retry mechanism: 3 attempts")
    print("   - Dry-run validation with EXPLAIN")
    print("   - Intelligent error recovery")
    print("   - Template-based fast path")
    
    return orchestrator



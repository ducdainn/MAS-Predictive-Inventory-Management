"""
SQLAgent: generates safe SQL queries from natural language questions.
"""

import re
from typing import Any, Dict, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent.agents.schema_agent import SchemaAgent
from agent.core.llm_provider import LLMProvider


class SQLAgent:
    """Generates SQL queries from natural language."""

    def __init__(self, llm_provider: LLMProvider, schema_agent: SchemaAgent):
        self.llm = llm_provider.get_llm("openai", temperature=0.0)
        self.schema_agent = schema_agent

    def generate_sql(
        self,
        question: str,
        intent: str,
        entities: Optional[Dict] = None,
        analysis_plan: Optional[Dict[str, Any]] = None,
        schema_context: Optional[str] = None
    ) -> str:
        """
        Generate SQL query with schema context and entity information.
        
        Args:
            question: User's natural language question
            intent: FORECAST or ANALYTICS
            entities: Optional dict with branch_codes, product_codes, regions (from EntityExtractor)
        """
        schema_context = schema_context or self.schema_agent.get_schema_context(question)
        
        # Build entity context for prompt
        entity_context = ""
        if entities:
            if entities.get('branch_codes'):
                branch_codes = entities['branch_codes']
                entity_context += f"\nEXTRACTED BRANCH CODES: {branch_codes}\n"
                entity_context += "→ USE branch.branch_code IN ({}) instead of LIKE for branch_name!\n".format(
                    ', '.join(map(str, branch_codes))
                )
            if entities.get('product_codes'):
                product_codes = entities['product_codes']
                entity_context += f"\nEXTRACTED PRODUCT CODES: {product_codes}\n"
                entity_context += "→ USE product.product_code IN ({}) instead of LIKE for product_name!\n".format(
                    ', '.join([f"'{code}'" for code in product_codes])
                )
            if entities.get('regions'):
                regions = entities['regions']
                entity_context += f"\nEXTRACTED REGIONS: {regions}\n"
                entity_context += "→ USE branch.region IN ({}) for exact match!\n".format(
                    ', '.join([f"'{r}'" for r in regions])
                )

        plan_context = self._build_plan_context(analysis_plan)

        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt(intent)),
            ("human", "Schema:\n{schema}\n\n{entity_info}{plan_info}Question: {question}\n\nSQL Query:")
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            raw_sql = chain.invoke({
                "schema": schema_context, 
                "question": question,
                "entity_info": entity_context,
                "plan_info": plan_context
            })
            sql = self._clean_sql(raw_sql)
            self._validate_sql(sql)
            print("\n🧾 Generated SQL ({intent}):".format(intent=intent))
            print(sql)
            return sql
        except Exception as e:
            print(f"⚠️ SQL generation failed: {e}")
            return self._retry_generate_sql(
                question,
                schema_context,
                str(e),
                entities,
                analysis_plan
            )

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

CRITICAL - PRODUCT FILTERING:
- product_code: VARCHAR like '10.L2.3060.4566' (unique ID)
- product_name: TEXT like 'Gạch 30x60 MS 4566 Loại 2' (full name)
- When user mentions product NAME → use product.product_name = '...'
- When user mentions product CODE → use product.product_code = '...'
- ALWAYS JOIN with product table when filtering by product name!

CRITICAL - BRANCH FILTERING:
- branch_code: INTEGER (unique ID) - PREFERRED when available
- branch_name: TEXT like 'Chi nhánh Hà Nội UN', 'Chi nhánh Đà Nẵng UC'
- IF EXTRACTED BRANCH CODES ARE PROVIDED → ALWAYS use branch.branch_code IN (codes) - MUCH MORE ACCURATE!
- If no branch codes extracted → User says 'Đà Nẵng' but DB has 'Chi nhánh Đà Nẵng UN' → use LIKE '%Đà Nẵng%'
- When user mentions branch NAME without codes → use branch.branch_name LIKE '%name%' (NOT =!)
- When user mentions branch CODE → use branch.branch_code = code
- ALWAYS JOIN with branch table when filtering by branch!
- PRIORITY: branch_code IN (...) > branch_name LIKE '%...%'


"""

        if intent == "FORECAST":
            base += """
FOR FORECAST QUERIES:
- Include historical data (at least last 90 days)
- ALWAYS include product.product_code and product.product_name in SELECT when possible.
- Group by date to get time series
- For general forecasts (toàn hệ thống): GROUP BY date
- For branch-specific forecasts: JOIN branch AND product, GROUP BY date, product.product_code, product.product_name, branch.branch_code
- For product-specific: JOIN product and GROUP BY date, product.product_code, product.product_name
- Order by date ASC

IMPORTANT: User says "Đà Nẵng" but branch_name is "Chi nhánh Đà Nẵng UN"
→ ALWAYS use LIKE '%Đà Nẵng%' NOT = 'Đà Nẵng'!
"""
        else:
            base += """
FOR ANALYTICS QUERIES:
- Use appropriate aggregations (SUM, AVG, COUNT)
- Include dimension for grouping (branch_name, product_name, region)
- Add ORDER BY for ranking queries
- LIMIT results if requesting "top N"

CRITICAL - DATE FILTERING:
- User says "tháng 10-2024" → use: date >= '2024-10-01' AND date < '2024-11-01'
- User says "tháng 10" → use current year or most recent year with data
- Always use DATE format: 'YYYY-MM-DD'
- For month ranges: date >= 'YYYY-MM-01' AND date < 'YYYY-MM+1-01'

CRITICAL - REGION FILTERING:
- region values are: 'MIỀN TRUNG', 'TÂY NAM BỘ', 'TÂY NGUYÊN', 'ĐÔNG NAM BỘ', 'HỒ CHÍ MINH' (uppercase with diacritics)
- User says "miền nam" → use: region = 'ĐÔNG NAM BỘ', 'TÂY NAM BỘ', 'HỒ CHÍ MINH'
- User says "miền bắc" → Dữ liệu không có miền bắc
- User says "miền trung" → use: region = 'MIỀN TRUNG', 'TÂY NGUYÊN'
- ALWAYS use exact match with uppercase: region = 'MIỀN TRUNG', 'TÂY NGUYÊN' (NOT LIKE!)

CRITICAL - REVENUE/DOANH THU:
- Doanh thu = sales.quantity * (price if available, or use quantity as proxy)
- If no price column, use SUM(sales.quantity) as total_sales or total_revenue
- CURRENT SCHEMA DOES NOT PROVIDE price COLUMN → NEVER reference price in queries!
- For revenue by branch: GROUP BY branch.branch_name, SUM(sales.quantity)"""

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

    def _retry_generate_sql(
        self,
        question: str,
        schema: str,
        error: str,
        entities: Optional[Dict] = None,
        analysis_plan: Optional[Dict[str, Any]] = None
    ) -> str:
        """Retry SQL generation with error context."""
        entity_context = ""
        if entities:
            if entities.get('branch_codes'):
                entity_context += f"\nEXTRACTED BRANCH CODES: {entities['branch_codes']}\n"
                entity_context += "→ USE branch.branch_code IN ({}) instead of LIKE!\n".format(
                    ', '.join(map(str, entities['branch_codes']))
                )
        plan_context = self._build_plan_context(analysis_plan)
        
        retry_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a PostgreSQL expert. Fix the SQL query based on the error."),
            ("human", "Schema: {schema}\n\n{entity_info}{plan_info}Question: {question}\n\nPrevious error: {error}\n\nGenerate a valid SELECT query:")
        ])

        chain = retry_prompt | self.llm | StrOutputParser()
        raw_sql = chain.invoke({
            "schema": schema, 
            "question": question, 
            "error": error,
            "entity_info": entity_context,
            "plan_info": plan_context
        })

        sql = self._clean_sql(raw_sql)
        self._validate_sql(sql)
        print("\n🧾 Generated SQL (retry):")
        print(sql)
        return sql

    @staticmethod
    def _build_plan_context(analysis_plan: Optional[Dict[str, Any]]) -> str:
        if not analysis_plan:
            return ""
        summary_lines = []
        if analysis_plan.get("objective"):
            summary_lines.append(f"OBJECTIVE: {analysis_plan['objective']}")
        if analysis_plan.get("metrics"):
            summary_lines.append(f"TARGET METRICS: {', '.join(analysis_plan['metrics'])}")
        if analysis_plan.get("dimensions"):
            summary_lines.append(f"GROUP BY: {', '.join(analysis_plan['dimensions'])}")
        if analysis_plan.get("timeframe"):
            summary_lines.append(f"TIMEFRAME: {analysis_plan['timeframe']}")
        if analysis_plan.get("filters"):
            summary_lines.append(f"EXTRA FILTERS: {', '.join(analysis_plan['filters'])}")
        if analysis_plan.get("chart_type"):
            summary_lines.append(f"DESIRED CHART: {analysis_plan['chart_type']}")

        if not summary_lines:
            return ""

        return "\nANALYSIS PLAN:\n- " + "\n- ".join(summary_lines) + "\n\n"




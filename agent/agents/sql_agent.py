"""
SQLAgent: generates safe SQL queries from natural language questions.
"""

import re
from typing import Any, Dict, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent.agents.schema_agent import SchemaAgent
from agent.core.llm_provider import LLMProvider

try:
    from agent.utils.sql_query_logger import get_sql_logger
    SQL_LOGGER_AVAILABLE = True
except ImportError:
    SQL_LOGGER_AVAILABLE = False

try:
    from agent.utils.workflow_data_logger import get_workflow_logger
    WORKFLOW_LOGGER_AVAILABLE = True
except ImportError:
    WORKFLOW_LOGGER_AVAILABLE = False


class SQLAgent:
    """Generates SQL queries from natural language."""

    def __init__(self, llm_provider: LLMProvider, schema_agent: SchemaAgent, log_dir: str = "sql_logs"):
        self.llm = llm_provider.get_llm("openai", temperature=0.0)
        self.schema_agent = schema_agent
        
        # Initialize SQL logger
        if SQL_LOGGER_AVAILABLE:
            self.sql_logger = get_sql_logger(log_dir)
        else:
            self.sql_logger = None
        
        # Initialize workflow logger
        if WORKFLOW_LOGGER_AVAILABLE:
            self.workflow_logger = get_workflow_logger()
        else:
            self.workflow_logger = None

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
        
        # Check if this is a "top N" ranking query
        question_lower = question.lower()
        is_top_n_query = bool(re.search(r'\btop\s+\d+|top\s*10|top\s*5|top\s*20|hàng đầu|cao nhất', question_lower, re.IGNORECASE))
        
        # Build entity context for prompt
        entity_context = ""
        if entities:
            # For "top N" queries, do NOT add branch_code filter even if extracted
            # (EntityExtractor should have cleared them, but safety check)
            if entities.get('branch_codes') and not is_top_n_query:
                branch_codes = entities['branch_codes']
                entity_context += f"\nEXTRACTED BRANCH CODES: {branch_codes}\n"
                if len(branch_codes) > 1:
                    entity_context += f"⚠️ CRITICAL: There are {len(branch_codes)} branch codes. You MUST use ALL of them!\n"
                    entity_context += "→ USE branch.branch_code IN ({}) - include ALL codes!\n".format(
                        ', '.join(map(str, branch_codes))
                    )
                else:
                    entity_context += "→ USE branch.branch_code IN ({}) instead of LIKE for branch_name!\n".format(
                        ', '.join(map(str, branch_codes))
                    )
            elif is_top_n_query and entities.get('branch_codes'):
                # Safety: Clear branch_codes for "top N" queries
                entity_context += "\n⚠️ IMPORTANT: This is a 'top N' ranking query.\n"
                entity_context += "→ DO NOT add WHERE branch.branch_code IN (...) filter!\n"
                entity_context += "→ Instead, GROUP BY branch and ORDER BY metric DESC LIMIT N to rank ALL branches.\n"
            if entities.get('product_codes'):
                product_codes = entities['product_codes']
                entity_context += f"\nEXTRACTED PRODUCT CODES: {product_codes}\n"
                entity_context += "→ USE product.product_code IN ({}) instead of LIKE for product_name!\n".format(
                    ', '.join([f"'{code}'" for code in product_codes])
                )
            # Chỉ thêm region filter gợi ý nếu:
            # 1. KHÔNG có branch_codes (để tránh conflict)
            # 2. scope != "all" (nếu scope="all" thì không filter, lấy tất cả regions)
            # 3. Question KHÔNG có cụm từ "theo vùng miền" hoặc "tất cả vùng" (đây là phân tích tổng quan)
            scope = entities.get('scope', 'all')
            question_lower = question.lower()
            is_general_region_analysis = any(
                phrase in question_lower 
                for phrase in ['theo vùng miền', 'theo vung mien', 'tất cả vùng', 'tat ca vung', 'all regions']
            )
            
            if (entities.get('regions') 
                and not entities.get('branch_codes') 
                and scope != 'all'
                and not is_general_region_analysis):
                raw_regions = entities['regions']
                # Normalize regions to canonical uppercase names used in DB
                canonical_values = []
                for r in raw_regions:
                    r_norm = str(r).strip().lower()
                    if r_norm in ["miền trung", "mien trung"]:
                        # Miền Trung thường bao gồm MIỀN TRUNG + TÂY NGUYÊN trong schema
                        canonical_values.extend(["'MIỀN TRUNG'", "'TÂY NGUYÊN'"])
                    elif r_norm in ["miền nam", "mien nam"]:
                        # Miền Nam = ĐÔNG NAM BỘ + TÂY NAM BỘ + HỒ CHÍ MINH
                        canonical_values.extend(
                            ["'ĐÔNG NAM BỘ'", "'TÂY NAM BỘ'", "'HỒ CHÍ MINH'"]
                        )
                    else:
                        # Default: dùng đúng chữ hoa để match với DB
                        canonical_values.append(f"'{str(r).upper()}'")
                # Loại bỏ trùng lặp nếu có
                canonical_values = sorted(set(canonical_values))
                entity_context += f"\nEXTRACTED REGIONS: {raw_regions}\n"
                if canonical_values:
                    entity_context += "→ USE branch.region IN ({}) for exact match!\n".format(
                        ", ".join(canonical_values)
                    )
            elif is_general_region_analysis or (scope == 'all' and entities.get('regions')):
                # Nếu là phân tích tổng quan theo vùng, KHÔNG thêm filter region
                # Để query GROUP BY region và hiển thị tất cả regions
                entity_context += "\n⚠️ IMPORTANT: User is asking for analysis BY ALL REGIONS (theo vùng miền).\n"
                entity_context += "→ DO NOT add branch.region IN (...) filter!\n"
                entity_context += "→ Instead, GROUP BY branch.region to show ALL regions in results.\n"

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
            
            # Post-process: Fix branch_code IN clause if entities have multiple codes but query only has one
            sql = self._fix_branch_codes_in_clause(sql, entities)

            # For FORECAST, enforce that SQL exposes panel identifiers so the panel model can be used
            if intent == "FORECAST":
                self._validate_forecast_sql_identifiers(sql)
            
            # Không in SQL ra terminal; SQLQueryLogger sẽ log đầy đủ ra file
            
            # Log to file
            if self.sql_logger:
                self.sql_logger.log_generated_query(
                    query=sql,
                    source="SQLAgent",
                    question=question,
                    intent=intent,
                    entities=entities,
                    analysis_plan=analysis_plan
                )
            
            # Also log to workflow logger
            if self.workflow_logger:
                self.workflow_logger.log_sql_query(
                    "sql_generation",
                    "SQLAgent",
                    sql,
                    question=question,
                    intent=intent,
                    entities=entities
                )
            
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
- **MANDATORY**: If EXTRACTED BRANCH CODES contains MULTIPLE codes (e.g., [107, 41, 63, 17, 53]), you MUST use ALL of them in the IN clause: branch.branch_code IN (107, 41, 63, 17, 53)
- **DO NOT** use only the first code or a single code when multiple codes are provided!
- If no branch codes extracted → User says 'Đà Nẵng' but DB has 'Chi nhánh Đà Nẵng UN' → use LIKE '%Đà Nẵng%'
- When user mentions branch NAME without codes → use branch.branch_name LIKE '%name%' (NOT =!)
- When user mentions branch CODE → use branch.branch_code = code
- ALWAYS JOIN with branch table when filtering by branch!
- PRIORITY: branch_code IN (ALL codes from entities) > analysis_plan filters > branch_name LIKE '%...%'


"""

        if intent == "FORECAST":
            base += """
FOR FORECAST QUERIES:
- Include historical data (at least last 90 days)
- ALWAYS include product.product_code and product.product_name in SELECT when possible.
- When joining with branch and product tables, ALSO include:
  - branch.branch_code (for model identifiers)
  - branch.region
  - product.f_sku
- Group by date to get time series
- For general forecasts (toàn hệ thống): GROUP BY date
- For branch-specific forecasts: JOIN branch AND product, and in SELECT you MUST include:
  - product.product_code, product.product_name, product.f_sku
  - branch.branch_code, branch.branch_name, branch.region
  Then GROUP BY date, product.product_code, product.product_name, product.f_sku, branch.branch_code, branch.branch_name, branch.region
- For product-specific: JOIN product and GROUP BY date, product.product_code, product.product_name, product.f_sku
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
- **CRITICAL for "Top N" queries**:
  - "Top 10 chi nhánh" means rank ALL branches, NOT filter specific branches
  - DO NOT add WHERE branch.branch_code IN (...) filter for "top N" queries!
  - Use GROUP BY branch_name, ORDER BY metric DESC, LIMIT N
  - Example: "Top 10 chi nhánh" → GROUP BY branch.branch_name, ORDER BY SUM(quantity) DESC LIMIT 10
- LIMIT results if requesting "top N" (extract the number from question, e.g., "top 10" → LIMIT 10)

CRITICAL - DATE FILTERING:
- User says "tháng 10-2024" → use: date >= '2024-10-01' AND date < '2024-11-01'
- User says "tháng 10" → use current year or most recent year with data
- Always use DATE format: 'YYYY-MM-DD'
- For month ranges: date >= 'YYYY-MM-01' AND date < 'YYYY-MM+1-01'

CRITICAL - REGION FILTERING:
- region values are: 'MIỀN TRUNG', 'TÂY NAM BỘ', 'TÂY NGUYÊN', 'ĐÔNG NAM BỘ', 'HỒ CHÍ MINH' (uppercase with diacritics)
- **IMPORTANT**: "theo vùng miền" / "theo vung mien" / "tất cả vùng" means analysis BY ALL regions (grouping dimension)
  → DO NOT add WHERE branch.region IN (...) filter!
  → Instead, GROUP BY branch.region to show all regions
- User says "miền nam" (specific) → use: branch.region IN ('ĐÔNG NAM BỘ', 'TÂY NAM BỘ', 'HỒ CHÍ MINH')
- User says "miền bắc" → Dữ liệu không có miền bắc
- User says "miền trung" (specific) → use: branch.region IN ('MIỀN TRUNG', 'TÂY NGUYÊN')
- ALWAYS use exact match with uppercase: branch.region IN ('MIỀN TRUNG', 'TÂY NGUYÊN', 'ĐÔNG NAM BỘ', 'TÂY NAM BỘ', 'HỒ CHÍ MINH') (NOT LIKE!)
- If entity_context says "DO NOT add branch.region filter" → GROUP BY branch.region instead!

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

    def _validate_forecast_sql_identifiers(self, sql: str):
        """
        Ensure FORECAST queries expose panel identifiers for the XGBoost panel model.

        We require that forecast SQL includes the key identifier columns
        so that downstream agents can build per-branch, per-SKU time series
        instead of a single aggregated series.
        """
        sql_lower = sql.lower()

        has_branch_code = "branch.branch_code" in sql_lower or ".branch_code" in sql_lower
        has_region = "branch.region" in sql_lower or ".region" in sql_lower
        has_product_code = "product.product_code" in sql_lower or ".product_code" in sql_lower
        has_f_sku = "product.f_sku" in sql_lower or ".f_sku" in sql_lower

        # 1) Completely missing identifiers → chắc chắn sai, bắt buộc sinh lại panel-style query.
        if not (has_branch_code or has_region or has_product_code or has_f_sku):
            raise ValueError(
                "FORECAST PANEL ERROR: SQL is missing branch/product identifier columns. "
                "You MUST join both branch and product tables, SELECT at least: "
                "branch.branch_code, branch.region, product.product_code, product.product_name, product.f_sku, "
                "and GROUP BY date plus all these identifier columns so that the model can forecast per (branch, SKU)."
            )

        # 2) Nếu đã có branch (branch_code/region) nhưng chưa có bất kỳ identifier bên product
        #    thì vẫn chưa đủ cho panel model (không biết SKU nào), cũng bắt buộc sinh lại.
        if (has_branch_code or has_region) and not (has_product_code or has_f_sku):
            raise ValueError(
                "FORECAST PANEL ERROR: Branch identifiers present but product identifiers missing. "
                "For branch-level forecasts you MUST also JOIN product and SELECT: "
                "product.product_code, product.product_name, product.f_sku, and GROUP BY date plus all branch and product identifiers."
            )

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
                branch_codes = entities['branch_codes']
                entity_context += f"\nEXTRACTED BRANCH CODES: {branch_codes}\n"
                if len(branch_codes) > 1:
                    entity_context += f"⚠️ CRITICAL: There are {len(branch_codes)} branch codes. You MUST use ALL of them!\n"
                    entity_context += "→ USE branch.branch_code IN ({}) - include ALL codes!\n".format(
                        ', '.join(map(str, branch_codes))
                    )
                else:
                    entity_context += "→ USE branch.branch_code IN ({}) instead of LIKE!\n".format(
                        ', '.join(map(str, branch_codes))
                    )
        plan_context = self._build_plan_context(analysis_plan)
        
        retry_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a PostgreSQL expert. Fix the SQL query based on the error.\n"
                "If the error message contains 'FORECAST PANEL ERROR', then this is a time-series forecast query and you MUST:\n"
                "- JOIN both branch and product tables with sales.\n"
                "- SELECT at minimum: branch.branch_code, branch.region, product.product_code, product.product_name, product.f_sku, and date, plus any required aggregates (e.g., SUM(sales.quantity)).\n"
                "- GROUP BY date and ALL of these identifier columns so the caller can build per-(branch, SKU) series.\n"
                "- Do NOT return only (date, SUM(quantity)) without identifiers."
            ),
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
        
        # Post-process: Fix branch_code IN clause if entities have multiple codes but query only has one
        sql = self._fix_branch_codes_in_clause(sql, entities)
        
        # Không in SQL retry ra terminal; đã log qua SQLQueryLogger
        
        # Log retry query to file
        if self.sql_logger:
            self.sql_logger.log_generated_query(
                query=sql,
                source="SQLAgent (RETRY)",
                question=question,
                intent="RETRY",
                entities=entities,
                analysis_plan=analysis_plan
            )
        
        # Also log to workflow logger
        if self.workflow_logger:
            self.workflow_logger.log_sql_query(
                "sql_generation_retry",
                "SQLAgent",
                sql,
                question=question,
                intent="RETRY",
                entities=entities
            )
        
        return sql

    def _fix_branch_codes_in_clause(self, sql: str, entities: Optional[Dict]) -> str:
        """
        Post-process SQL to ensure ALL branch_codes from entities are included in IN clause.
        
        This fixes cases where LLM only uses the first branch_code instead of all.
        """
        if not entities or not entities.get('branch_codes'):
            return sql
        
        branch_codes = entities['branch_codes']
        if len(branch_codes) <= 1:
            return sql  # No fix needed if only one or zero codes
        
        # Find branch_code IN clauses in SQL
        # Pattern: branch_code IN (number) or sales.branch_code IN (number) or i.branch_code IN (number)
        # Match patterns like: branch_code IN (107) or sales.branch_code IN (107) or i.branch_code IN (107)
        pattern = r'(\w+\.)?branch_code\s+IN\s*\(([^)]+)\)'
        
        def replace_in_clause(match):
            table_alias = match.group(1) or ""
            existing_codes_str = match.group(2).strip()
            
            # Try to extract existing codes
            try:
                # Remove quotes and parse
                existing_codes = [int(c.strip().strip("'\"`")) for c in existing_codes_str.split(',')]
            except:
                # If parsing fails, use all codes from entities
                existing_codes = []
            
            # Check if query is missing some codes
            missing_codes = set(branch_codes) - set(existing_codes)
            
            if missing_codes:
                # Replace with ALL codes from entities
                all_codes_str = ', '.join(map(str, branch_codes))
                print(f"   ⚠️  Fixed branch_code IN clause: was {existing_codes_str}, now includes ALL: {all_codes_str}")
                return f"{table_alias}branch_code IN ({all_codes_str})"
            else:
                # Already has all codes or is correct
                return match.group(0)
        
        fixed_sql = re.sub(pattern, replace_in_clause, sql, flags=re.IGNORECASE)
        
        # Also check for single branch_code = pattern when we have multiple codes
        single_pattern = r'(\w+\.)?branch_code\s*=\s*(\d+)'
        def replace_single_with_in(match):
            table_alias = match.group(1) or ""
            single_code = int(match.group(2))
            
            if single_code in branch_codes and len(branch_codes) > 1:
                all_codes_str = ', '.join(map(str, branch_codes))
                print(f"   ⚠️  Fixed branch_code = {single_code} to IN clause with ALL codes: {all_codes_str}")
                return f"{table_alias}branch_code IN ({all_codes_str})"
            else:
                return match.group(0)
        
        fixed_sql = re.sub(single_pattern, replace_single_with_in, fixed_sql, flags=re.IGNORECASE)
        
        return fixed_sql
    
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




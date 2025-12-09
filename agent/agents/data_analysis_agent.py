"""
DataAnalysisAgent: interpret analytics questions before SQL generation.
UPDATED: Smart visualization logic and chart recommendations.
"""

import json
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent.core.llm_provider import LLMProvider


class DataAnalysisAgent:
    """
    Senior data analyst persona that converts business questions into
    structured analysis requirements (metrics, dimensions, charts, filters).
    """

    DEFAULT_PLAN = {
        "objective": "",
        "metrics": [],
        "dimensions": [],
        "timeframe": "",
        "filters": [],
        "chart_type": "bar", # Default fallback
        "chart_title": "",
        "narrative_guidance": ""
    }

    def __init__(self, llm_provider: LLMProvider):
        # Tăng temperature lên một chút (0.2) để nó sáng tạo hơn trong việc chọn biểu đồ
        self.llm = llm_provider.get_llm("openai", temperature=0.2)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", """
            Question: {question}
            Known entities (optional): {entity_context}
            Database schema snippet:
            {schema_context}
            """)
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _get_system_prompt(self) -> str:
        """
        Prompt nâng cao: Dạy Agent cách tư duy như một chuyên gia Data Viz.
        """
        return """You are a Senior Data Analyst & Visualization Expert with 10 years of experience.
Your goal is to interpret a user's business question and design the PERFECT analysis plan.

### 1. ANALYSIS LOGIC
- Break down the question into: Metrics (Y-axis) and Dimensions (X-axis/Legend).
- The sales table does NOT contain a price column; use `sales.quantity` as the proxy for revenue/demand.
- ALWAYS specify a timeframe if implied (e.g., "current", "trend" -> last 6 months).

### 2. VISUALIZATION STRATEGY (CRITICAL)
Choose the `chart_type` based on the intent:

* **TREND / EVOLUTION (Time Series)**
  - Use `line` chart.
  - Dimension must be a date/time column.
  
* **COMPARISON / RANKING**
  - Use `bar` chart (Vertical) for few items (< 10).
  - Use `barh` chart (Horizontal) for many items or long names (e.g., Top 20 Products).
  
* **COMPOSITION / PART-TO-WHOLE**
  - Use `pie` chart ONLY if categories < 5 (e.g., Region share).
  - Otherwise, use `bar` chart (e.g., Product share among 50 items).
  
* **DISTRIBUTION**
  - Use `histogram` for frequency.
  
* **RELATIONSHIP**
  - Use `scatter` for correlating two metrics.

* **SINGLE VALUE**
  - Use `kpi_card` if the user asks for a total sum/count without grouping.

### 3. OUTPUT FORMAT (JSON ONLY)
Return a valid JSON object with these fields:
{{
    "objective": "Brief explanation of what we are analyzing",
    "metrics": ["List of SQL aggregation functions, e.g., SUM(quantity)"],
    "dimensions": ["List of grouping columns, e.g., product_name, date"],
    "timeframe": "SQL logic for time, e.g., 'last 3 months' or specific dates",
    "filters": ["List of WHERE conditions"],
    "chart_type": "One of: line, bar, barh, pie, scatter, kpi_card",
    "chart_title": "A professional title for the chart (in Vietnamese)",
    "narrative_guidance": "Instructions for the SQL writer (e.g., 'Group by month', 'Limit to top 10')"
}}

### 4. LANGUAGE RULES
- The `chart_title` and `objective` MUST be in VIETNAMESE.
- Keep technical field names in English (from Schema).
"""

    def analyze(self, 
                question: str, 
                entities: Optional[Dict[str, Any]] = None, 
                schema_context: Optional[str] = None) -> Dict[str, Any]:
        """Generate analysis plan from question."""
        entity_context = self._format_entities(entities)
        schema_hint = self._prepare_schema_hint(schema_context)

        try:
            response = self.chain.invoke({
                "question": question, 
                "entity_context": entity_context,
                "schema_context": schema_hint
            })
            
            # Clean and parse JSON
            cleaned_response = self._extract_json(response)
            plan = json.loads(cleaned_response)
            
            # Merge with defaults to ensure safety
            final_plan = self.DEFAULT_PLAN.copy()
            final_plan.update(plan)
            
            return final_plan
            
        except Exception as e:
            print(f"⚠️ DataAnalysisAgent Error: {e}")
            return self.DEFAULT_PLAN

    # ... (Giữ nguyên các hàm static helper: _format_entities, _ensure_list, _extract_json, _prepare_schema_hint) ...
    # Bạn hãy copy lại các hàm static cũ vào đây
    
    @staticmethod
    def _format_entities(entities: Optional[Dict[str, Any]]) -> str:
        if not entities:
            return "Không có."
        parts = []
        if entities.get("branch_codes"):
            parts.append(f"Chi nhánh: {entities['branch_codes']}")
        if entities.get("product_codes"):
            parts.append(f"Mã sản phẩm: {entities['product_codes']}")
        if entities.get("regions"):
            parts.append(f"Vùng: {entities['regions']}")
        return "; ".join(parts) if parts else "Không có."

    @staticmethod
    def _extract_json(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]
        return text

    @staticmethod
    def _prepare_schema_hint(schema_context: str) -> str:
        if not schema_context:
            return "Không có thông tin schema."
        return schema_context[:2000] # Limit context size
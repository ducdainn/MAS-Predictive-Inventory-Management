"""
IntentAgent: classifies user questions into FORECAST / ANALYTICS / INVENTORY intents.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent.core.llm_provider import LLMProvider


class IntentAgent:
    """Classifies user intent."""

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
        """Classify question intent using improved context-aware approach."""
        question_lower = question.lower()

        forecast_keywords = ['dự báo', 'forecast', 'dự đoán', 'predict', 'tương lai', 'future',
                             'sẽ', 'will', 'next', 'tiếp theo', 'tới', 'coming']

        analytics_keywords = ['biểu đồ', 'chart', 'plot', 'phân tích', 'analysis', 'top',
                              'thống kê', 'distribution', 'so sánh', 'compare', 'ranking',
                              'xem', 'view', 'show', 'hiển thị', 'báo cáo', 'report']

        inventory_action_keywords = ['nhập hàng', 'restock', 'order', 'chuyển kho', 'transfer',
                                     'tối ưu', 'optimize', 'kế hoạch', 'plan', 'đề xuất', 'recommend',
                                     'rop', 'safety stock', 'replenish', 'need',
                                     'gấp', 'urgent', 'thiếu', 'shortage', 'thừa', 'surplus']

        inventory_viewing_keywords = ['tồn kho', 'inventory', 'stock', 'stock level']
        past_present_keywords = ['qua', 'past', 'trước', 'before', 'hiện tại', 'current',
                                 'đã', 'was', 'were', 'năm ngoái', 'last year', 'tháng trước', 'last month']

        forecast_score = sum(1 for kw in forecast_keywords if kw in question_lower)
        analytics_score = sum(1 for kw in analytics_keywords if kw in question_lower)
        inventory_action_score = sum(1 for kw in inventory_action_keywords if kw in question_lower)
        inventory_viewing_score = sum(1 for kw in inventory_viewing_keywords if kw in question_lower)
        past_present_score = sum(1 for kw in past_present_keywords if kw in question_lower)

        if inventory_action_score >= 1:
            return "INVENTORY_OPTIMIZATION"

        if inventory_viewing_score > 0 and inventory_action_score == 0:
            if forecast_score >= 2:
                return "FORECAST"
            return "ANALYTICS"

        if forecast_score >= 1:
            if past_present_score >= 1:
                return "ANALYTICS"
            return "FORECAST"

        if analytics_score >= 1:
            return "ANALYTICS"

        if 'nhu cầu' in question_lower or 'demand' in question_lower:
            if forecast_score > 0:
                return "FORECAST"
            return "ANALYTICS"

        if forecast_score == 0 and analytics_score == 0 and inventory_action_score == 0:
            try:
                result = self.chain.invoke({"question": question}).strip().upper()
                if result in ["FORECAST", "ANALYTICS", "INVENTORY_OPTIMIZATION"]:
                    return result
            except Exception:
                pass

        return "ANALYTICS"




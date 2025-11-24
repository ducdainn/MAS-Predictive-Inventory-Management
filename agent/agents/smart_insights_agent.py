"""
SmartInsightsGenerator: LLM-powered insights for inventory optimization output.
"""

from typing import Any, Dict, Optional

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent.core.llm_provider import LLMProvider
from agent.utils.dataframe_utils import format_dataframe_columns


class SmartInsightsGenerator:
    """LLM-powered insights generator for inventory optimization results."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider.get_llm("openai", temperature=0.3)

    def generate_insights(self,
                          recommendations: pd.DataFrame,
                          action_plan: Dict[str, Any],
                          entities: Optional[Dict[str, Any]] = None) -> str:
        """Generate intelligent business insights from optimization results."""
        print("🧠 Generating smart insights...")

        context = self._prepare_context(recommendations, action_plan, entities)

        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_insights_prompt()),
            ("human", """
Phân tích kết quả tối ưu hóa tồn kho này và đưa ra những hiểu biết chiến lược BẰNG TIẾNG VIỆT:

{context}

Hãy cung cấp:
1. PHÁT HIỆN CHÍNH (3-5 điểm)
2. CÁC VÙNG RỦI RO (vấn đề quan trọng)
3. CƠ HỘI (tiết kiệm chi phí, hiệu quả)
4. KHUYẾN NGHỊ CHIẾN LƯỢC (các bước có thể thực hiện)
5. HÀNH ĐỘNG ƯU TIÊN (làm gì trước tiên)

Hãy cụ thể, dựa trên dữ liệu và có thể hành động được.
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
        return """Bạn là chuyên gia tư vấn quản lý tồn kho với hơn 15 năm kinh nghiệm.

Vai trò của bạn: Phân tích kết quả tối ưu hóa tồn kho và đưa ra những hiểu biết chiến lược.

Hướng dẫn:
- Cụ thể và dựa trên dữ liệu
- Tập trung vào tác động kinh doanh (chi phí, mức dịch vụ, rủi ro)
- Xác định các mẫu và xu hướng
- Đưa ra khuyến nghị có thể thực hiện được
- Ưu tiên theo mức độ khẩn cấp và tác động
- Sử dụng ngôn ngữ chuyên nghiệp, rõ ràng

**QUAN TRỌNG: Trả lời HOÀN TOÀN BẰNG TIẾNG VIỆT**

Định dạng insights như sau:
📊 PHÁT HIỆN CHÍNH
- [Phát hiện 1 với dữ liệu cụ thể]
- [Phát hiện 2 với dữ liệu cụ thể]

⚠️ CÁC VÙNG RỦI RO
- [Rủi ro và tác động]
- [Chiến lược giảm thiểu]

💡 CƠ HỘI
- [Cơ hội với lợi ích cụ thể]

🎯 KHUYẾN NGHỊ CHIẾN LƯỢC
1. [Hành động cụ thể]
2. [Hành động cụ thể]

🔴 HÀNH ĐỘNG ƯU TIÊN (24-48 giờ tới)
1. [Hành động khẩn cấp]
2. [Hành động khẩn cấp]"""

    def _prepare_context(self,
                         recommendations: pd.DataFrame,
                         action_plan: Dict[str, Any],
                         entities: Optional[Dict[str, Any]]) -> str:
        """Prepare concise context for LLM."""
        context = []

        summary = action_plan['summary']
        context.append(f"SUMMARY:")
        context.append(f"- Total items analyzed: {len(recommendations)}")
        context.append(f"- Total actions needed: {summary['total_actions']}")
        context.append(f"- Restock orders: {summary['restock_actions']} (qty: {summary['total_restock_quantity']:.0f})")
        context.append(f"- Transfer opportunities: {summary['transfer_actions']} (qty: {summary['total_transfer_quantity']:.0f})")
        context.append(f"- High priority: {summary['high_priority_actions']}")

        if not recommendations.empty:
            action_dist = recommendations['action'].value_counts()
            context.append(f"\nACTION DISTRIBUTION:")
            for action, count in action_dist.items():
                pct = (count / len(recommendations)) * 100
                context.append(f"- {action}: {count} items ({pct:.1f}%)")

        if 'region' in recommendations.columns and not recommendations.empty:
            region_actions = recommendations[recommendations['action'] != 'OK'].groupby('region')['action'].count()
            if not region_actions.empty:
                context.append(f"\nREGIONAL BREAKDOWN:")
                for region, count in region_actions.items():
                    context.append(f"- {region}: {count} actions needed")

        urgent = recommendations[recommendations['action'] == 'URGENT_RESTOCK']
        if not urgent.empty:
            context.append(f"\nCRITICAL SHORTAGE: {len(urgent)} items below reorder point")
            top_urgent = urgent.nlargest(3, 'quantity_needed')
            for _, item in top_urgent.iterrows():
                context.append(f"  - {item['product_name'][:40]} at {item['branch_name']}: need {item['quantity_needed']:.0f}")

        transfers = action_plan.get('actions', [])
        transfer_actions = [t for t in transfers if t['action_type'] == 'TRANSFER']
        if transfer_actions:
            total_distance = sum(t.get('distance_km', 0) for t in transfer_actions)
            avg_distance = total_distance / len(transfer_actions)
            context.append(f"\nTRANSFER ANALYSIS:")
            context.append(f"- {len(transfer_actions)} transfer opportunities identified")
            context.append(f"- Average distance: {avg_distance:.1f} km")

        if entities and entities.get('scope') == 'specific':
            branches = entities.get('branch_names', [])
            if branches:
                context.append(f"\nSCOPE: Focused on {', '.join(branches[:3])}")

        return "\n".join(context)

    def _fallback_insights(self, recommendations: pd.DataFrame, action_plan: Dict[str, Any]) -> str:
        """Insights dự phòng dựa trên quy tắc nếu LLM thất bại."""
        insights = []
        summary = action_plan['summary']

        insights.append("📊 PHÁT HIỆN CHÍNH")
        insights.append(f"- Cần {summary['total_actions']} hành động tổng cộng cho tồn kho")
        insights.append(f"- {summary['high_priority_actions']} mặt hàng ưu tiên cao cần chú ý ngay lập tức")

        if summary['transfer_actions'] > 0:
            denominator = summary['total_restock_quantity'] + summary['total_transfer_quantity']
            savings_pct = (summary['total_transfer_quantity'] / denominator) * 100 if denominator else 0
            insights.append(f"- {savings_pct:.1f}% nhu cầu có thể đáp ứng qua chuyển kho nội bộ (tiết kiệm chi phí)")

        insights.append("\n⚠️ CÁC VÙNG RỦI RO")
        urgent = recommendations[recommendations['action'] == 'URGENT_RESTOCK']
        if not urgent.empty:
            insights.append(f"- {len(urgent)} mặt hàng ở mức tồn kho rất thấp (nguy cơ hết hàng)")

        insights.append("\n🎯 HÀNH ĐỘNG ƯU TIÊN")
        insights.append("1. Xử lý tất cả đơn nhập hàng ưu tiên CAO ngay lập tức")
        insights.append("2. Khởi động chuyển kho nội bộ để giảm đặt hàng bên ngoài")
        insights.append("3. Xem xét độ chính xác dự báo cho các mặt hàng có chênh lệch lớn")

        return "\n".join(insights)




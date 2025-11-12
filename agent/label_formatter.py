"""
Label Formatter Agent - Intelligent Chart Label Beautification
Converts raw database column names to beautiful, readable UI labels
"""

import re
from typing import Dict, Optional

# Use langchain_core for better stability (avoid deprecated imports)
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    # Fallback for older versions
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser


class LabelFormatterAgent:
    """
    Intelligent agent to format raw column names into beautiful chart labels.
    
    Features:
    - Rule-based formatting for common patterns
    - LLM-powered formatting for complex cases
    - Vietnamese translation support
    - Context-aware formatting (based on question)
    """
    
    def __init__(self, llm_provider=None):
        """
        Initialize Label Formatter.
        
        Args:
            llm_provider: Optional LLMProvider for intelligent formatting
        """
        self.llm = llm_provider.get_llm("openai", temperature=0.0) if llm_provider else None
        
        # Common column name mappings (fast path)
        self.column_mappings = {
            # Sales columns
            'total_sales_quantity': 'Tổng Số Lượng Bán',
            'total_quantity': 'Tổng Số Lượng',
            'quantity': 'Số Lượng',
            'total_revenue': 'Tổng Doanh Thu',
            'revenue': 'Doanh Thu',
            'selling_price': 'Giá Bán',
            'cost_price': 'Giá Vốn',
            'profit': 'Lợi Nhuận',
            
            # Product columns
            'product_name': 'Tên Sản Phẩm',
            'product_code': 'Mã Sản Phẩm',
            'category_name': 'Danh Mục',
            'category': 'Danh Mục',
            
            # Branch columns
            'branch_name': 'Chi Nhánh',
            'branch_code': 'Mã Chi Nhánh',
            'region': 'Khu Vực',
            
            # Inventory columns
            'quantity_on_hand': 'Tồn Kho Hiện Tại',
            'reorder_point': 'Điểm Đặt Hàng Lại',
            'safety_stock': 'Tồn Kho An Toàn',
            'current_stock': 'Tồn Kho Hiện Tại',
            
            # Time columns
            'date': 'Ngày',
            'month': 'Tháng',
            'year': 'Năm',
            'quarter': 'Quý',
            'week': 'Tuần',
            
            # Aggregations
            'count': 'Số Lượng',
            'avg': 'Trung Bình',
            'sum': 'Tổng',
            'min': 'Nhỏ Nhất',
            'max': 'Lớn Nhất',
            'total': 'Tổng',
            
            # Other common
            'transaction_count': 'Số Giao Dịch',
            'product_count': 'Số Sản Phẩm',
            'customer_count': 'Số Khách Hàng',
        }
        
        # Pattern-based rules
        self.patterns = [
            (r'total_(.+)', r'Tổng \1'),
            (r'avg_(.+)', r'Trung Bình \1'),
            (r'sum_(.+)', r'Tổng \1'),
            (r'count_(.+)', r'Số \1'),
            (r'(.+)_quantity', r'Số Lượng \1'),
            (r'(.+)_revenue', r'Doanh Thu \1'),
            (r'(.+)_count', r'Số \1'),
        ]
    
    def format_label(self, raw_label: str, context: Optional[str] = None) -> str:
        """
        Format a single raw label into beautiful UI label.
        
        Args:
            raw_label: Raw column name (e.g., 'total_sales_quantity')
            context: Optional context like user question for better formatting
            
        Returns:
            Formatted label (e.g., 'Tổng Số Lượng Bán')
        """
        if not raw_label or not isinstance(raw_label, str):
            return str(raw_label)
        
        # Step 1: Try exact mapping (fast path)
        label_lower = raw_label.lower().strip()
        if label_lower in self.column_mappings:
            return self.column_mappings[label_lower]
        
        # Step 2: Try pattern matching
        formatted = self._apply_patterns(raw_label)
        if formatted != raw_label:
            return formatted
        
        # Step 3: Use LLM for complex cases (if available)
        if self.llm and context:
            return self._format_with_llm(raw_label, context)
        
        # Step 4: Fallback to simple formatting
        return self._simple_format(raw_label)
    
    def format_chart_labels(self, x_label: str, y_label: str, 
                           title: Optional[str] = None,
                           context: Optional[str] = None) -> Dict[str, str]:
        """
        Format all chart labels at once.
        
        Args:
            x_label: X-axis raw label
            y_label: Y-axis raw label
            title: Optional chart title
            context: Optional context (user question)
            
        Returns:
            Dict with formatted labels: {'x', 'y', 'title'}
        """
        formatted_x = self.format_label(x_label, context)
        formatted_y = self.format_label(y_label, context)
        
        # Generate smart title if not provided
        if not title:
            title = self._generate_title(formatted_x, formatted_y, context)
        else:
            title = self.format_label(title, context)
        
        return {
            'x': formatted_x,
            'y': formatted_y,
            'title': title
        }
    
    def _apply_patterns(self, raw_label: str) -> str:
        """Apply pattern-based rules."""
        label = raw_label
        
        for pattern, replacement in self.patterns:
            match = re.match(pattern, label, re.IGNORECASE)
            if match:
                # Extract captured group and format
                if match.groups():
                    part = match.group(1)
                    part_formatted = self._format_word(part)
                    label = re.sub(pattern, replacement, label, flags=re.IGNORECASE)
                    label = label.replace(r'\1', part_formatted)
                    return label
        
        return label
    
    def _format_word(self, word: str) -> str:
        """Format a single word (snake_case to Title Case)."""
        # Remove underscores and capitalize
        words = word.replace('_', ' ').split()
        return ' '.join(w.capitalize() for w in words)
    
    def _simple_format(self, raw_label: str) -> str:
        """Simple fallback formatting."""
        # Replace underscores with spaces
        label = raw_label.replace('_', ' ')
        
        # Title case
        label = ' '.join(word.capitalize() for word in label.split())
        
        # Common abbreviations
        replacements = {
            'Qty': 'Số Lượng',
            'Rev': 'Doanh Thu',
            'Avg': 'Trung Bình',
            'Min': 'Nhỏ Nhất',
            'Max': 'Lớn Nhất',
            'Cnt': 'Số',
            'Prd': 'Sản Phẩm',
            'Br': 'Chi Nhánh',
        }
        
        for abbr, full in replacements.items():
            label = label.replace(abbr, full)
        
        return label
    
    def _format_with_llm(self, raw_label: str, context: str) -> str:
        """Use LLM for intelligent formatting."""
        if not self.llm:
            return self._simple_format(raw_label)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là chuyên gia về data visualization và UX.
Nhiệm vụ: Convert raw database column names thành beautiful chart labels.

Rules:
1. Output ONLY the formatted label (no explanations)
2. Use Vietnamese for common business terms
3. Keep it concise (2-4 words max)
4. Use Title Case for English words
5. Make it intuitive and readable

Examples:
- total_sales_quantity → Tổng Số Lượng Bán
- revenue_by_region → Doanh Thu Theo Khu Vực
- avg_daily_orders → Trung Bình Đơn Hàng/Ngày
- product_category_distribution → Phân Bố Danh Mục Sản Phẩm
"""),
            ("human", """Raw column name: {raw_label}

Context (user question): {context}

Formatted label:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            formatted = chain.invoke({
                "raw_label": raw_label,
                "context": context or "N/A"
            })
            return formatted.strip()
        except:
            return self._simple_format(raw_label)
    
    def _generate_title(self, x_label: str, y_label: str, 
                       context: Optional[str] = None) -> str:
        """Generate smart chart title from axis labels."""
        
        # Common patterns
        if 'Ngày' in x_label or 'Tháng' in x_label:
            # Time series
            return f"{y_label} Theo Thời Gian"
        elif 'Khu Vực' in x_label or 'Chi Nhánh' in x_label:
            # Geographic
            return f"{y_label} Theo {x_label}"
        elif 'Sản Phẩm' in x_label or 'Danh Mục' in x_label:
            # Product analysis
            return f"{y_label} Theo {x_label}"
        else:
            # Generic
            return f"{y_label} vs {x_label}"
    
    def format_legend(self, legend_label: str) -> str:
        """Format legend label."""
        return self.format_label(legend_label)
    
    def format_value_with_unit(self, value: float, column_name: str) -> str:
        """
        Format value with appropriate unit.
        
        Args:
            value: Numeric value
            column_name: Column name to determine unit
            
        Returns:
            Formatted value with unit (e.g., "1.5M VNĐ", "250 sản phẩm")
        """
        column_lower = column_name.lower()
        
        # Revenue/Money columns
        if any(kw in column_lower for kw in ['revenue', 'doanh_thu', 'price', 'gia', 'profit']):
            if value >= 1_000_000_000:
                return f"{value/1_000_000_000:.1f}B VNĐ"
            elif value >= 1_000_000:
                return f"{value/1_000_000:.1f}M VNĐ"
            elif value >= 1_000:
                return f"{value/1_000:.1f}K VNĐ"
            else:
                return f"{value:,.0f} VNĐ"
        
        # Quantity columns
        elif any(kw in column_lower for kw in ['quantity', 'so_luong', 'count', 'stock']):
            if value >= 1_000_000:
                return f"{value/1_000_000:.1f}M"
            elif value >= 1_000:
                return f"{value/1_000:.1f}K"
            else:
                return f"{value:,.0f}"
        
        # Percentage columns
        elif any(kw in column_lower for kw in ['percent', 'rate', 'ratio']):
            return f"{value:.1f}%"
        
        # Default
        else:
            if value >= 1_000_000:
                return f"{value/1_000_000:.1f}M"
            elif value >= 1_000:
                return f"{value/1_000:.1f}K"
            else:
                return f"{value:,.2f}"


# ============================================================================
# QUICK FORMATTER (Without LLM)
# ============================================================================

class QuickLabelFormatter:
    """
    Fast label formatter without LLM (for better performance).
    Use this if you don't need LLM-powered intelligent formatting.
    """
    
    def __init__(self):
        self.formatter = LabelFormatterAgent(llm_provider=None)
    
    def format_label(self, raw_label: str) -> str:
        """Format label without LLM."""
        return self.formatter.format_label(raw_label, context=None)
    
    def format_chart_labels(self, x_label: str, y_label: str, title: Optional[str] = None) -> Dict[str, str]:
        """Format chart labels without LLM."""
        return self.formatter.format_chart_labels(x_label, y_label, title, context=None)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_axis_label(raw_label: str) -> str:
    """
    Quick helper function to format a single axis label.
    
    Usage:
        from agent.label_formatter import format_axis_label
        plt.xlabel(format_axis_label('total_sales_quantity'))
    """
    formatter = QuickLabelFormatter()
    return formatter.format_label(raw_label)


def format_chart_title(x_label: str, y_label: str) -> str:
    """
    Quick helper to generate chart title.
    
    Usage:
        from agent.label_formatter import format_chart_title
        plt.title(format_chart_title('region', 'total_revenue'))
    """
    formatter = QuickLabelFormatter()
    labels = formatter.format_chart_labels(x_label, y_label)
    return labels['title']


def get_value_formatter(column_name: str):
    """
    Get a value formatter function for a specific column.
    
    Usage:
        from agent.label_formatter import get_value_formatter
        formatter = get_value_formatter('revenue')
        print(formatter(1500000))  # "1.5M VNĐ"
    """
    label_formatter = LabelFormatterAgent()
    
    def format_func(value):
        return label_formatter.format_value_with_unit(value, column_name)
    
    return format_func


"""
Q&A Agent - Answer General Questions about System and Data
"""

from typing import Dict, Any
import pandas as pd

class QAAgent:
    """
    Agent to handle general questions that don't require forecasting/analytics.
    
    Capabilities:
    - Answer questions about branches, products, inventory
    - Provide summaries and statistics
    - Handle "what is", "how many", "which", "list" questions
    """
    
    def __init__(self, db_manager, llm_provider=None):
        self.db = db_manager
        self.llm = llm_provider
    
    def answer(self, question: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Answer general questions based on question type.
        
        Question types:
        - COUNT: "Có bao nhiêu...", "How many..."
        - LIST: "Liệt kê...", "List..."
        - INFO: "Thông tin về...", "What is..."
        - SUMMARY: "Tổng quan...", "Overview..."
        """
        question_lower = question.lower()
        
        # Detect question type
        if any(word in question_lower for word in ['bao nhiêu', 'how many', 'số lượng', 'count']):
            return self._answer_count_question(question, entities)
        
        elif any(word in question_lower for word in ['liệt kê', 'danh sách', 'list', 'show me']):
            return self._answer_list_question(question, entities)
        
        elif any(word in question_lower for word in ['thông tin', 'what is', 'là gì', 'info']):
            return self._answer_info_question(question, entities)
        
        elif any(word in question_lower for word in ['tổng quan', 'overview', 'summary', 'tổng hợp']):
            return self._answer_summary_question(question, entities)
        
        else:
            # Fallback: try to answer with general info
            return self._answer_general(question, entities)
    
    def _answer_count_question(self, question: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Answer 'how many' type questions."""
        question_lower = question.lower()
        
        # Count branches
        if 'chi nhánh' in question_lower or 'branch' in question_lower:
            regions = entities.get('regions', [])
            
            if regions:
                sql = f"""
                SELECT region, COUNT(*) as count
                FROM branch
                WHERE region IN ({','.join([f"'{r}'" for r in regions])})
                GROUP BY region
                """
            else:
                sql = "SELECT region, COUNT(*) as count FROM branch GROUP BY region"
            
            df = self.db.execute_query(sql)
            
            summary = "📊 **Thống Kê Chi Nhánh**\n\n"
            for _, row in df.iterrows():
                summary += f"- **{row['region']}**: {row['count']} chi nhánh\n"
            
            total = df['count'].sum()
            summary += f"\n**Tổng cộng**: {total} chi nhánh"
            
            return {
                'success': True,
                'answer': summary,
                'data': df,
                'type': 'count_branches'
            }
        
        # Count products
        elif 'sản phẩm' in question_lower or 'product' in question_lower:
            sql = "SELECT COUNT(DISTINCT product_code) as total FROM inventory"
            result = self.db.execute_query(sql)
            total = result.iloc[0]['total']
            
            summary = f"📦 **Tổng số sản phẩm**: {total} sản phẩm"
            
            return {
                'success': True,
                'answer': summary,
                'total': total,
                'type': 'count_products'
            }
        
        # Count inventory items
        elif 'tồn kho' in question_lower or 'inventory' in question_lower:
            regions = entities.get('regions', [])
            
            if regions:
                sql = f"""
                SELECT b.region, SUM(i.quantity) as total_qty
                FROM inventory i
                JOIN branch b ON i.branch_code = b.branch_code
                WHERE b.region IN ({','.join([f"'{r}'" for r in regions])})
                GROUP BY b.region
                """
            else:
                sql = """
                SELECT b.region, SUM(i.quantity) as total_qty
                FROM inventory i
                JOIN branch b ON i.branch_code = b.branch_code
                GROUP BY b.region
                """
            
            df = self.db.execute_query(sql)
            
            summary = "📦 **Tổng Tồn Kho Theo Khu Vực**\n\n"
            for _, row in df.iterrows():
                summary += f"- **{row['region']}**: {row['total_qty']:,.0f} đơn vị\n"
            
            total = df['total_qty'].sum()
            summary += f"\n**Tổng cộng**: {total:,.0f} đơn vị"
            
            return {
                'success': True,
                'answer': summary,
                'data': df,
                'type': 'count_inventory'
            }
        
        return {'success': False, 'message': 'Không thể trả lời câu hỏi này'}
    
    def _answer_list_question(self, question: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Answer 'list' type questions."""
        question_lower = question.lower()
        regions = entities.get('regions', [])
        
        # List branches
        if 'chi nhánh' in question_lower or 'branch' in question_lower:
            if regions:
                sql = f"""
                SELECT branch_code, branch_name, region
                FROM branch
                WHERE region IN ({','.join([f"'{r}'" for r in regions])})
                ORDER BY region, branch_name
                """
            else:
                sql = "SELECT branch_code, branch_name, region FROM branch ORDER BY region, branch_name"
            
            df = self.db.execute_query(sql)
            
            summary = f"📋 **Danh Sách Chi Nhánh** ({len(df)} chi nhánh)\n\n"
            
            # Group by region
            for region in df['region'].unique():
                region_branches = df[df['region'] == region]
                summary += f"**{region}** ({len(region_branches)} chi nhánh):\n"
                for _, row in region_branches.iterrows():
                    summary += f"  - [{row['branch_code']}] {row['branch_name']}\n"
                summary += "\n"
            
            return {
                'success': True,
                'answer': summary,
                'data': df,
                'type': 'list_branches'
            }
        
        # List products
        elif 'sản phẩm' in question_lower or 'product' in question_lower:
            sql = """
            SELECT DISTINCT product_code, product_name, unit
            FROM inventory
            ORDER BY product_name
            LIMIT 50
            """
            
            df = self.db.execute_query(sql)
            
            summary = f"📦 **Danh Sách Sản Phẩm** (top 50)\n\n"
            for _, row in df.iterrows():
                summary += f"- **[{row['product_code']}]** {row['product_name']} ({row['unit']})\n"
            
            return {
                'success': True,
                'answer': summary,
                'data': df,
                'type': 'list_products'
            }
        
        return {'success': False, 'message': 'Không thể liệt kê dữ liệu này'}
    
    def _answer_info_question(self, question: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Answer 'what is' information questions."""
        branches = entities.get('branches', [])
        products = entities.get('products', [])
        regions = entities.get('regions', [])
        
        # Info about specific branch
        if branches:
            branch_code = branches[0]
            sql = f"""
            SELECT b.branch_code, b.branch_name, b.region,
                   COUNT(i.product_code) as product_count,
                   SUM(i.quantity) as total_inventory
            FROM branch b
            LEFT JOIN inventory i ON b.branch_code = i.branch_code
            WHERE b.branch_code = {branch_code}
            GROUP BY b.branch_code, b.branch_name, b.region
            """
            
            df = self.db.execute_query(sql)
            if not df.empty:
                row = df.iloc[0]
                summary = f"🏢 **Thông Tin Chi Nhánh [{row['branch_code']}]**\n\n"
                summary += f"- **Tên**: {row['branch_name']}\n"
                summary += f"- **Khu vực**: {row['region']}\n"
                summary += f"- **Số sản phẩm**: {row['product_count']}\n"
                summary += f"- **Tổng tồn kho**: {row['total_inventory']:,.0f} đơn vị\n"
                
                return {
                    'success': True,
                    'answer': summary,
                    'data': df,
                    'type': 'info_branch'
                }
        
        # Info about region
        if regions:
            region = regions[0]
            sql = f"""
            SELECT 
                COUNT(DISTINCT b.branch_code) as branch_count,
                COUNT(DISTINCT i.product_code) as product_count,
                SUM(i.quantity) as total_inventory
            FROM branch b
            LEFT JOIN inventory i ON b.branch_code = i.branch_code
            WHERE b.region = '{region}'
            """
            
            df = self.db.execute_query(sql)
            if not df.empty:
                row = df.iloc[0]
                summary = f"🌍 **Thông Tin Khu Vực: {region}**\n\n"
                summary += f"- **Số chi nhánh**: {row['branch_count']}\n"
                summary += f"- **Số sản phẩm**: {row['product_count']}\n"
                summary += f"- **Tổng tồn kho**: {row['total_inventory']:,.0f} đơn vị\n"
                
                return {
                    'success': True,
                    'answer': summary,
                    'data': df,
                    'type': 'info_region'
                }
        
        return {'success': False, 'message': 'Không tìm thấy thông tin'}
    
    def _answer_summary_question(self, question: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Answer summary/overview questions."""
        sql = """
        SELECT 
            (SELECT COUNT(*) FROM branch) as total_branches,
            (SELECT COUNT(DISTINCT region) FROM branch) as total_regions,
            (SELECT COUNT(DISTINCT product_code) FROM inventory) as total_products,
            (SELECT SUM(quantity) FROM inventory) as total_inventory,
            (SELECT COUNT(*) FROM sales WHERE date >= CURRENT_DATE - INTERVAL '30 days') as recent_sales
        """
        
        df = self.db.execute_query(sql)
        if not df.empty:
            row = df.iloc[0]
            
            summary = "📊 **Tổng Quan Hệ Thống**\n\n"
            summary += f"🏢 **Chi nhánh**: {row['total_branches']} chi nhánh tại {row['total_regions']} khu vực\n\n"
            summary += f"📦 **Sản phẩm**: {row['total_products']:,.0f} sản phẩm\n\n"
            summary += f"📦 **Tồn kho**: {row['total_inventory']:,.0f} đơn vị\n\n"
            summary += f"📈 **Giao dịch**: {row['recent_sales']:,.0f} giao dịch (30 ngày gần đây)\n"
            
            return {
                'success': True,
                'answer': summary,
                'data': df,
                'type': 'summary'
            }
        
        return {'success': False, 'message': 'Không thể tạo tổng quan'}
    
    def _answer_general(self, question: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback for general questions."""
        # Try to provide relevant info based on entities
        regions = entities.get('regions', [])
        
        if regions:
            return self._answer_info_question(question, entities)
        
        return {
            'success': False,
            'message': 'Xin lỗi, tôi chưa hiểu câu hỏi này. Bạn có thể hỏi về:\n' +
                      '- Số lượng chi nhánh/sản phẩm\n' +
                      '- Danh sách chi nhánh/sản phẩm\n' +
                      '- Thông tin về chi nhánh/khu vực\n' +
                      '- Dự báo nhu cầu\n' +
                      '- Tối ưu hóa tồn kho'
        }



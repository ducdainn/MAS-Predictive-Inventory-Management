"""
SchemaAgent: understands database schema and provides context to SQL generator.
UPDATED: Reads dynamic schema from file and enforces business rules.
"""

import os
from typing import Optional

from agent.manager.database_manager import DatabaseManager
from agent.manager.memory_manager import MemoryManager


class SchemaAgent:
    """Understands database schema and provides context."""

    def __init__(self, 
                 db_manager: DatabaseManager, 
                 memory: MemoryManager, 
                 schema_path: str = "init/01_schema.sql"):
        self.db = db_manager
        self.memory = memory
        self.schema_path = schema_path
        # Load schema ngay khi khởi tạo
        self.schema_summary = self._load_schema_source()

    def _load_schema_source(self) -> str:
        """
        Đọc cấu trúc DB từ file SQL gốc để giữ nguyên các comment nghiệp vụ.
        Đây là cách tốt nhất để Agent hiểu ngữ nghĩa cột.
        """
        if os.path.exists(self.schema_path):
            try:
                with open(self.schema_path, "r", encoding="utf-8") as f:
                    print(f"✅ SchemaAgent: Loaded schema definitions from {self.schema_path}")
                    return f.read()
            except Exception as e:
                print(f"⚠️ SchemaAgent: Error reading schema file: {e}")
        
        print("⚠️ SchemaAgent: Schema file not found, using fallback hardcoded schema.")
        return self._build_fallback_schema()

    def get_schema_context(self, question: str) -> str:
        """
        Tạo ngữ cảnh schema đầy đủ cho SQL Agent, bao gồm:
        1. Schema DDL (Create table + Comments)
        2. Luật nghiệp vụ (Business Rules)
        3. Các câu query mẫu từ quá khứ (Few-shot learning)
        """
        # 1. Base Schema
        context = f"""
DATABASE SCHEMA DEFINITION:
{self.schema_summary}

CRITICAL BUSINESS RULES (TUÂN THỦ TUYỆT ĐỐI):
1. PHÂN BIỆT BẢNG:
   - Muốn tính DOANH SỐ/BÁN HÀNG (Sold, Revenue) -> Dùng bảng 'sales'.
   - Muốn xem TỒN KHO (Stock, On-hand, Available) -> Dùng bảng 'inventory'.
   - KHÔNG được nhầm lẫn giữa sales và inventory.

2. ĐƠN VỊ TÍNH:
   - Nếu câu hỏi đề cập đến 'diện tích' hoặc 'm2' -> Dùng cột 'square_meters' trong bảng 'sales'.
   - Nếu không nói rõ -> Dùng cột 'quantity' (viên/thùng).

3. TÌM KIẾM:
   - Tên sản phẩm/kho hàng phải dùng ILIKE '%keyword%' để tìm kiếm không phân biệt hoa thường.
   - Ví dụ: product_name ILIKE '%gạch 60x60%'

4. LIÊN KẾT (JOIN):
   - inventory JOIN product ON inventory.product_code = product.product_code
   - sales JOIN product ON sales.product_code = product.product_code
   - sales JOIN branch ON sales.branch_code = branch.branch_code
"""

        # 2. Inject Similar Past Queries (Học từ quá khứ)
        # Phần này giữ nguyên logic hay của bạn
        similar = self.memory.search_similar(question, top_k=3, only_successful=True)
        if similar:
            context += "\n\nSUCCESSFUL PAST QUERIES (REFERENCE ONLY):\n"
            for i, s in enumerate(similar, 1):
                if isinstance(s, dict):
                    # Handle different memory formats safely
                    sql = s.get('solution', {}).get('sql', '') or s.get('sql', '')
                    if sql:
                        context += f"--- Example {i} ---\nQ: {s.get('question', '')}\nSQL: {sql}\n"

        return context

    def _build_fallback_schema(self) -> str:
        """Schema dự phòng nếu không đọc được file."""
        return """
        -- BẢNG 1: BRANCH (DANH MỤC KHO/CHI NHÁNH)
        -- Agent Note:
        --    - Khi user hỏi về "Kho Đà Nẵng", "Chi nhánh HCM"... hãy dùng ILIKE trên cột 'branch_name'.
        --    - 'branch_code' là khóa ngoại quan trọng để JOIN với bảng sales và inventory.
        --    - Dữ liệu mẫu: 1='Chi nhánh Đà Nẵng UN', 8='Chi nhánh Bình Chánh UN'.
        -- ================================================================
        DROP TABLE IF EXISTS branch CASCADE;
        CREATE TABLE branch (
        branch_code   INTEGER PRIMARY KEY,      -- ID duy nhất (VD: 1, 8, 10...)
        region        TEXT    NOT NULL,         -- Vùng miền (VD: 'MIỀN TRUNG', 'TÂY NGUYÊN', 'ĐÔNG NAM BỘ', 'TÂY NAM BỘ', 'HỒ CHÍ MINH'). Dùng để gom nhóm báo cáo vùng.
        branch_name   TEXT    NOT NULL          -- Tên hiển thị (VD: 'Chi nhánh Đồng Nai UN'). Dùng để tìm kiếm text.
        );

        -- ================================================================
        -- BẢNG 2: BRANCH_DISTANCE (KHOẢNG CÁCH)
        -- Agent Note:
        --    - Bảng này dùng cho bài toán "Tối ưu luân chuyển" (Stock Transfer).
        --    - Nếu kho A thiếu hàng, tìm kho B có 'distance_km' nhỏ nhất để điều chuyển.
        --    - distance_km = 0 nghĩa là chính nó.
        -- ================================================================
        DROP TABLE IF EXISTS branch_distance CASCADE;
        CREATE TABLE branch_distance (
        branch_code_1 INTEGER NOT NULL,         -- ID kho nguồn
        branch_code_2 INTEGER NOT NULL,         -- ID kho đích
        distance_km   NUMERIC(12,2) NOT NULL    -- Khoảng cách (Km).
        );

        -- ================================================================
        -- BẢNG 3: PRODUCT (DANH MỤC SẢN PHẨM - MASTER DATA)
        -- Agent Note:
        --    - Bảng chứa thông tin tĩnh của sản phẩm.
        --    - 'product_code' có dạng '14.L1.3060.A36410.7'.
        --    - 'spec_code_size' (VD: '3060', '6060') rất quan trọng khi user hỏi tìm gạch theo kích thước.
        --    - 'f_sku' (Family SKU) dùng để tìm các sản phẩm tương thay thế (substitutes) nếu mã chính hết hàng.
        -- ================================================================
        DROP TABLE IF EXISTS product CASCADE;
        CREATE TABLE product (
        product_code     VARCHAR(128) PRIMARY KEY, -- Mã SKU (Primary Key)
        product_name     TEXT NOT NULL,            -- Tên sản phẩm (VD: 'Gạch 30x60 MS A36410 Loại 1'). Dùng ILIKE để tìm.
        category         TEXT,                     -- Phân loại (Hiện tại dữ liệu mẫu thường để trống hoặc ít dùng).
        f_sku            TEXT,                     -- Mã nhóm hàng (Family SKU). Dùng để GROUP các sản phẩm cùng hoa văn/bộ.
        spec_code_size   TEXT,                     -- Kích thước/Quy cách (VD: '3060', '6060', '8080').
        unit             TEXT NOT NULL             -- Đơn vị tính (VD: 'Viên', 'Kg', 'Thùng').
        );

        -- ================================================================
        -- BẢNG 4: INVENTORY (TỒN KHO HIỆN TẠI - SNAPSHOT)
        -- Agent Note:
        --    - Đây là số lượng hàng ĐANG CÓ trong kho (On-hand).
        --    - Dùng bảng này khi user hỏi: "Còn bao nhiêu hàng?", "Tồn kho hiện tại".
        --    - KHÔNG dùng bảng này để tính doanh số bán.
        --    - Cột 'quantity' luôn >= 0.
        -- ================================================================
        DROP TABLE IF EXISTS inventory CASCADE;
        CREATE TABLE inventory (
        product_code   VARCHAR(128) NOT NULL
                        REFERENCES product(product_code)
                        ON UPDATE CASCADE ON DELETE CASCADE,
        branch_code    INTEGER NOT NULL
                        REFERENCES branch(branch_code)
                        ON UPDATE CASCADE ON DELETE CASCADE,
        product_name   TEXT,      -- (Denormalized) Tên SP lưu dư thừa để query nhanh.
        unit           TEXT,      -- (Denormalized) Đơn vị tính.
        quantity       INTEGER NOT NULL CHECK (quantity >= 0), -- Số lượng tồn kho thực tế.
        PRIMARY KEY (product_code, branch_code) -- Khóa chính kép: 1 Sản phẩm tại 1 Kho là duy nhất.
        );

        -- ================================================================
        -- BẢNG 5: SALES (LỊCH SỬ GIAO DỊCH BÁN HÀNG)
        -- Agent Note:
        --    - Đây là dữ liệu LỊCH SỬ (Historical Data) dùng để train AI Forecast.
        --    - Dùng bảng này khi user hỏi: "Bán được bao nhiêu?", "Doanh thu?", "Xu hướng?".
        --    - Lưu ý cột 'square_meters': Ngành gạch thường báo cáo theo m2 bên cạnh số lượng viên.
        --    - quantity: Số lượng bán ra (Demand).
        -- ================================================================
        DROP TABLE IF EXISTS sales CASCADE;
        CREATE TABLE sales (
        id             BIGSERIAL PRIMARY KEY,    -- Khóa chính tự tăng
        date           DATE NOT NULL,            -- Ngày bán (YYYY-MM-DD). Quan trọng để GROUP BY tháng/năm.
        branch_code    INTEGER NOT NULL
                        REFERENCES branch(branch_code)
                        ON UPDATE CASCADE ON DELETE RESTRICT,
        customer_code  VARCHAR(64) NOT NULL,     -- Mã khách hàng (VD: 'KH07781').
        product_code   VARCHAR(128) NOT NULL
                        REFERENCES product(product_code)
                        ON UPDATE CASCADE ON DELETE RESTRICT,
        quantity       INTEGER NOT NULL CHECK (quantity > 0), -- Số lượng bán (Viên/Thùng/Cái).
        square_meters  NUMERIC(12,4),            -- Diện tích bán được (m2). Quan trọng cho ngành gạch.
        unit           TEXT                      -- Đơn vị tính của quantity.
        );
        """
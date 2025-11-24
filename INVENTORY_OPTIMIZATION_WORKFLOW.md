# 🔄 Workflow Tối Ưu Hóa Tồn Kho (Inventory Optimization)

## 📋 Tổng Quan

Workflow này mô tả chi tiết quá trình xử lý khi người dùng yêu cầu tối ưu hóa tồn kho, từ lúc nhập câu hỏi đến khi nhận được kết quả cuối cùng.

---

## 🎯 Entry Point: Streamlit UI

**File:** `agent/ui/components/optimization.py`

### Bước 1: Người dùng nhập yêu cầu
```
User Input: "Tối ưu hóa tồn kho của chi nhánh đà nẵng"
```

### Bước 2: UI gọi Orchestrator
```python
result = orchestrator.process_query(
    question="Tối ưu hóa tồn kho của chi nhánh đà nẵng",
    forced_intent="INVENTORY_OPTIMIZATION"
)
```

---

## 🤖 OrchestratorAgent Processing

**File:** `agent/agents/orchestrator_agent.py`

### Step 1: Intent Classification
```python
# OrchestratorAgent.process_query()
intent = "INVENTORY_OPTIMIZATION"  # Forced by UI
```

**Output:**
- Intent được xác định: `INVENTORY_OPTIMIZATION`
- Không cần gọi `IntentAgent` vì UI đã force intent

---

### Step 2: Entity Extraction
```python
# OrchestratorAgent.process_query()
entities = self.entity_extractor.extract_entities(question)
```

**File:** `agent/agents/entity_extractor.py`

#### 2.1. LLM-based Extraction (Primary)
- **Input:** Câu hỏi + danh sách branch mẫu + regions
- **Process:**
  1. Gửi prompt đến LLM (OpenAI GPT-4o-mini)
  2. LLM trả về JSON với:
     - `branch_names`: ["đà nẵng"]
     - `product_names`: []
     - `regions`: []
     - `scope`: "specific"
- **Output:** 
  ```python
  {
      "branch_names": ["đà nẵng"],
      "branch_codes": [1, 66, 115],  # Matched via fuzzy matching
      "product_names": [],
      "product_codes": [],
      "regions": [],
      "scope": "specific"
  }
  ```

#### 2.2. Fallback Extraction (Nếu LLM fail)
- **Process:**
  1. Bỏ qua từ chung: "chi", "nhánh", "của", "tồn", "kho"
  2. Tạo cụm từ địa danh 2-3 từ: "đà nẵng"
  3. Match với tên branch (case-insensitive, bỏ dấu)
  4. Chỉ match branch có chứa cụm từ này
- **Output:** Tương tự LLM extraction

**⚠️ Fix mới:** Logic fallback đã được sửa để chỉ match branch cụ thể, không match tất cả branch.

---

### Step 3: Inventory Optimization Agent
```python
# OrchestratorAgent.process_query()
result = self.inventory_agent.optimize_inventory(
    question=question,
    entities=entities
)
```

**File:** `agent/agents/inventory_agent.py`

---

## 📦 InventoryOptimizationAgent Workflow

### Phase 1: Lấy Dữ Liệu Tồn Kho Hiện Tại

**Method:** `_get_current_inventory()`

#### 1.1. Build SQL Query với Filters
```sql
SELECT 
    i.product_code,
    i.branch_code,
    i.current_stock,
    p.product_name,
    b.branch_name,
    b.region,
    p.unit
FROM inventory i
JOIN product p ON i.product_code = p.product_code
JOIN branch b ON i.branch_code = b.branch_code
WHERE 1=1
    AND i.branch_code IN (:branch_1, :branch_2, ...)  -- Từ entities
    AND i.current_stock > 0
ORDER BY i.branch_code, i.product_code
```

#### 1.2. Execute Query
- **Database:** PostgreSQL
- **System Date:** Sử dụng `get_system_date()` (có thể là date trong quá khứ cho OOT testing)
- **Result:** DataFrame với tất cả inventory items matching criteria

**Output:**
```
✅ Found 45764 inventory items matching criteria
```

---

### Phase 2: Dự Báo Nhu Cầu Cho Từng Item

**Method:** `_get_forecast_data_per_item()` hoặc `_get_forecast_data_batch()`

#### 2.1. Build Timeseries Cache (Optimization)
```python
# Fetch historical data cho nhiều products cùng lúc
timeseries_cache = self._build_timeseries_cache(inventory_data)
```

**SQL Query (Bulk):**
```sql
SELECT 
    date,
    product_code,
    branch_code,
    SUM(quantity) as total_qty
FROM sales
WHERE date >= CURRENT_DATE - INTERVAL '90 days'
    AND (product_code, branch_code) IN (
        (:product_1, :branch_1),
        (:product_2, :branch_2),
        ...
    )
GROUP BY date, product_code, branch_code
ORDER BY date
```

**Lợi ích:** Giảm số lượng database queries từ N xuống 1 (N = số items)

#### 2.2. Parallel Forecast Generation

**Method:** `_forecast_single_item_worker()`

**Process cho mỗi (product_code, branch_code):**

1. **Lấy Historical Data:**
   ```sql
   SELECT date, SUM(quantity) as total_qty
   FROM sales
   WHERE date >= CURRENT_DATE - INTERVAL '90 days'
       AND product_code = :product_code
       AND branch_code = :branch_code
   GROUP BY date
   ORDER BY date
   ```

2. **Kiểm tra dữ liệu:**
   - **Nếu empty hoặc < 2 records:** → Fallback forecast (value = 0.0)
   - **Nếu < 7 records:** → Simple forecast (trung bình)
   - **Nếu >= 7 records:** → XGBoost forecast

3. **XGBoost Forecast (Primary):**
   - Load pre-trained model từ `agent/models/`
   - Feature engineering:
     - Date features (day_of_week, month, quarter)
     - Lag features (1, 2, 3, 7, 14, 30 days)
     - Rolling statistics (MA7, MA14, MA30, std)
     - Change features (day-over-day, week-over-week)
     - Trend & volatility
   - Predict 30 days ahead
   - Apply confidence intervals

4. **Post-processing:**
   - Cap forecast nếu quá cao so với recent demand:
     ```python
     if forecast_avg > recent_avg * max_forecast_vs_recent_ratio:
         forecast = recent_avg * max_forecast_vs_recent_ratio
     ```
   - Normalize dates (remove time components)

5. **Calculate Metrics:**
   ```python
   metrics = {
       'recent_avg_daily': recent_avg,
       'forecast_avg_daily': forecast_avg,
       'forecast_total': forecast_sum,
       'trend': 'increasing' | 'decreasing' | 'stable',
       'volatility': std_dev
   }
   ```

**Parallel Processing:**
- Sử dụng `ThreadPoolExecutor` với `max_workers=16`
- Mỗi worker xử lý 1 (product, branch) combination
- Progress tracking: log mỗi 100 items

**Output:**
```python
{
    (product_code, branch_code): {
        'forecast_df': DataFrame[date, forecast],
        'historical_df': DataFrame[date, quantity],
        'metrics': {
            'recent_avg_daily': float,
            'forecast_avg_daily': float,
            'forecast_total': float,
            'trend': str,
            'volatility': float
        }
    },
    ...
}
```

---

### Phase 3: Tính Toán Metrics Tồn Kho

**Method:** `_generate_recommendations()`

#### 3.1. Calculate ROP (Reorder Point)
```python
ROP = (Average Daily Demand × Lead Time) + Safety Stock
```
- **Lead Time:** 7 days (default)
- **Average Daily Demand:** Từ forecast metrics

#### 3.2. Calculate Safety Stock
```python
Safety Stock = Z × σ × √LT
```
- **Z:** Z-score cho service level (95% → Z ≈ 1.645)
- **σ:** Standard deviation của demand
- **LT:** Lead time (7 days)

#### 3.3. Calculate EOQ (Economic Order Quantity)
```python
EOQ = √((2 × D × S) / H)
```
- **D:** Annual demand
- **S:** Ordering cost (1000 VND default)
- **H:** Holding cost (50 VND default)

#### 3.4. Determine Actions
```python
if current_stock < ROP:
    action = "URGENT_RESTOCK"  # Priority: HIGH
elif current_stock < ROP + safety_stock:
    action = "RESTOCK"  # Priority: MEDIUM
elif current_stock > ROP + safety_stock + EOQ:
    action = "EXCESS_STOCK"  # Priority: LOW
else:
    action = "OK"  # No action needed
```

**Output:** DataFrame với columns:
- `product_code`, `branch_code`, `product_name`, `branch_name`
- `current_stock`, `unit`
- `recent_avg_daily_demand`, `forecast_avg_daily_demand`
- `forecast_total_30d`
- `rop`, `safety_stock`, `eoq`
- `action`, `priority`
- `restock_quantity` (nếu cần)

---

### Phase 4: Tìm Cơ Hội Chuyển Kho

**Method:** `_find_transfer_opportunities()`

#### 4.1. Identify Excess & Shortage
- **Excess:** Items với `action = "EXCESS_STOCK"`
- **Shortage:** Items với `action = "URGENT_RESTOCK"` hoặc `"RESTOCK"`

#### 4.2. Match Transfer Pairs
```python
for excess_item in excess_items:
    for shortage_item in shortage_items:
        if (excess_item.product_code == shortage_item.product_code and
            excess_item.branch_code != shortage_item.branch_code):
            
            # Calculate distance
            distance = calculate_distance(
                excess_item.branch_code,
                shortage_item.branch_code
            )
            
            if distance <= max_transfer_distance_km:  # 200 km default
                transfer_quantity = min(
                    excess_item.excess_amount,
                    shortage_item.shortage_amount
                )
                
                transfer_opportunities.append({
                    'product_code': ...,
                    'source_branch': ...,
                    'dest_branch': ...,
                    'quantity': transfer_quantity,
                    'distance_km': distance,
                    'savings': ...  # Cost savings vs external order
                })
```

**Output:** List of transfer opportunities

---

### Phase 5: Tạo Action Plan

**Method:** `_create_action_plan()`

#### 5.1. Build Actions List
```python
actions = []

# Restock actions
for item in recommendations[recommendations['action'].isin(['URGENT_RESTOCK', 'RESTOCK'])]:
    actions.append({
        'action_type': 'RESTOCK',
        'product_code': ...,
        'branch_code': ...,
        'quantity': item['restock_quantity'],
        'priority': item['priority'],
        'reason': f"Stock below ROP ({item['current_stock']:.0f} < {item['rop']:.0f})"
    })

# Transfer actions
for transfer in transfer_opportunities:
    actions.append({
        'action_type': 'TRANSFER',
        'product_code': ...,
        'source_branch': ...,
        'dest_branch': ...,
        'quantity': transfer['quantity'],
        'distance_km': transfer['distance_km'],
        'priority': 'MEDIUM',
        'reason': f"Transfer from excess stock (saves {transfer['savings']:.0f} VND)"
    })
```

#### 5.2. Generate Summary
```python
summary = {
    'total_actions': len(actions),
    'restock_actions': len([a for a in actions if a['action_type'] == 'RESTOCK']),
    'transfer_actions': len([a for a in actions if a['action_type'] == 'TRANSFER']),
    'high_priority_actions': len([a for a in actions if a['priority'] == 'HIGH']),
    'total_restock_quantity': sum([a['quantity'] for a in actions if a['action_type'] == 'RESTOCK']),
    'total_transfer_quantity': sum([a['quantity'] for a in actions if a['action_type'] == 'TRANSFER'])
}
```

**Output:**
```python
action_plan = {
    'summary': summary,
    'actions': actions
}
```

---

### Phase 6: Tạo Visualization

**Method:** `_plot_inventory_optimization()`

#### 6.1. Aggregate Forecasts
```python
# Aggregate tất cả forecasts theo date
total_forecast = pd.Series(0, index=forecast_dates)
for key, forecast_data in per_item_forecasts.items():
    forecast_df = forecast_data['forecast_df']
    total_forecast += forecast_df['forecast'].values
```

#### 6.2. Create Chart
- **Matplotlib chart** với:
  - Historical demand (90 days)
  - Forecast demand (30 days)
  - Current stock level
  - ROP line
  - Safety stock zone
- **Save to:** `charts/inventory_optimization_{timestamp}.png`

---

### Phase 7: Generate AI Insights

**Method:** `SmartInsightsGenerator.generate_insights()`

**File:** `agent/agents/smart_insights_agent.py`

#### 7.1. Prepare Context
```python
context = {
    'recommendations': recommendations DataFrame,
    'action_plan': action_plan dict,
    'entities': entities dict,
    'summary_stats': ...
}
```

#### 7.2. LLM Generation
- **Model:** OpenAI GPT-4o-mini
- **Prompt:** Structured prompt với context data
- **Output:** Natural language insights về:
  - Phát hiện chính
  - Các vùng rủi ro
  - Cơ hội tối ưu
  - Khuyến nghị chiến lược
  - Hành động ưu tiên

#### 7.3. Fallback (Nếu LLM fail)
- Rule-based insights từ summary statistics

---

## 📊 Final Result Structure

```python
{
    "success": True,
    "question": "Tối ưu hóa tồn kho của chi nhánh đà nẵng",
    "intent": "INVENTORY_OPTIMIZATION",
    "sql": "N/A - Inventory optimization uses multiple queries internally",
    "result": {
        "per_item_forecasts": {
            (product_code, branch_code): {
                'forecast_df': DataFrame,
                'historical_df': DataFrame,
                'metrics': {...}
            },
            ...
        },
        "inventory_data": DataFrame,  # Formatted for display
        "recommendations": DataFrame,  # Formatted for display
        "recommendations_raw": DataFrame,  # Raw data for filtering
        "transfer_opportunities": [...],
        "action_plan": {
            "summary": {...},
            "actions": [...]
        },
        "chart": "charts/inventory_optimization_20250106_123456.png",
        "summary": "...",
        "smart_insights": "📊 PHÁT HIỆN CHÍNH\n..."
    },
    "elapsed_seconds": 45.23
}
```

---

## 🎨 UI Display

**File:** `agent/ui/components/optimization.py`

### Tabs:
1. **📋 Kế Hoạch Hành Động:** Action plan grouped by priority
2. **📈 Biểu Đồ:** Matplotlib chart + Plotly interactive charts
3. **🧠 Phân Tích AI:** Smart insights text
4. **📦 Dữ Liệu Chi Tiết:** Filterable recommendations table
5. **💾 Xuất File:** Export to Excel/CSV

---

## ⚡ Performance Optimizations

1. **Bulk Timeseries Cache:** Fetch historical data cho nhiều items cùng lúc
2. **Parallel Forecasts:** ThreadPoolExecutor với 16 workers
3. **Batch Forecast Tool:** Option để dùng `BatchForecastTool` cho large inventories
4. **System Date:** Sử dụng configured system date thay vì real current date (cho OOT testing)

---

## 🔧 Configuration Parameters

**File:** `agent/agents/inventory_agent.py`

```python
self.service_level = 0.95  # 95% service level
self.lead_time_days = 7    # Default lead time
self.max_transfer_distance_km = 200  # Max distance for transfer
self.missing_data_forecast_value = 0.0  # Forecast for missing SKU history
self.max_forecast_vs_recent_ratio = 1.2  # Cap forecasts to avoid seasonal spikes
self.max_parallel_forecasts = 16  # Parallel workers
self.bulk_query_chunk_size = 200  # Items per bulk SQL query
```

---

## 📝 Notes

- **System Date:** Queries sử dụng `CURRENT_DATE` được replace bằng system date (có thể là date trong quá khứ)
- **Entity Extraction:** Fallback logic đã được fix để chỉ match branch cụ thể
- **Forecast Capping:** Forecasts được cap để tránh seasonal spikes không còn phù hợp
- **Missing Data:** SKUs không có lịch sử được forecast = 0.0 (không dùng heuristics)

---

## 🔄 Flow Diagram

```
User Input
    ↓
OrchestratorAgent.process_query()
    ↓
Intent Classification (forced: INVENTORY_OPTIMIZATION)
    ↓
EntityExtractor.extract_entities()
    ├─→ LLM Extraction (primary)
    └─→ Fallback Extraction (if LLM fails)
    ↓
InventoryOptimizationAgent.optimize_inventory()
    ├─→ _get_current_inventory() [SQL query với filters]
    ├─→ _get_forecast_data_per_item() [Parallel forecasts]
    │   ├─→ _build_timeseries_cache() [Bulk SQL]
    │   └─→ _forecast_single_item_worker() [XGBoost/Simple/Fallback]
    ├─→ _generate_recommendations() [Calculate ROP, Safety Stock, EOQ]
    ├─→ _find_transfer_opportunities() [Match excess ↔ shortage]
    ├─→ _create_action_plan() [Build actions + summary]
    ├─→ _plot_inventory_optimization() [Create chart]
    └─→ SmartInsightsGenerator.generate_insights() [LLM insights]
    ↓
Result Dict
    ↓
UI Display (5 tabs)
```

---

**Last Updated:** 2025-01-06



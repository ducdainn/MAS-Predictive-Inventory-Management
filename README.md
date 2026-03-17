# 🧱 BrickDemand - Predictive Inventory Management System

**Multi-Agent System for Intelligent Inventory Optimization**

Version: 3.1 | Last Updated: 2025-01-06

---

## 🎯 Overview

BrickDemand is an advanced multi-agent system (MAS) designed for predictive inventory management. It combines AI-powered demand forecasting, intelligent inventory optimization, and natural language analytics to help businesses make data-driven inventory decisions.

### Key Features

- 🤖 **Multi-Agent Architecture**: Specialized agents for intent classification, entity extraction, SQL generation, forecasting, and inventory optimization
- 📊 **Demand Forecasting**: Panel XGBoost models (multi-step and recursive) for accurate demand prediction
- 🎯 **Inventory Optimization**: AI-powered recommendations for restocking, transfers, and safety stock management
- 📈 **Interactive Analytics**: Natural language queries with automatic SQL generation
- 🎨 **Modern Web UI**: Beautiful Streamlit interface with real-time dashboards
- ⚡ **Performance Optimized**: Vectorized batch processing, caching, and smart routing

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd BrickDemand

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_streamlit.txt
```

### 2. Configuration

Create `.env` file:

```env
# Database
PG_USER=postgres
PG_PASSWORD=your_password
PG_HOST=localhost
PG_PORT=5433
PG_DB=brickdemand

# OpenAI (for LLM agents)
OPENAI_API_KEY=your_api_key

# Qdrant (vector database)
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

### 3. Run the Application

```bash
# Start Streamlit UI
streamlit run agent/ui/app.py

# Or use the launcher
python run_ui.py
```

The application will open at `http://localhost:8501`

---

## 📁 Project Structure

```
BrickDemand/
├── agent/
│   ├── agents/              # Core agent implementations
│   │   ├── orchestrator_agent.py    # Main coordinator
│   │   ├── intent_agent.py          # Intent classification
│   │   ├── entity_extractor.py      # Entity extraction (branches, products, regions)
│   │   ├── sql_agent.py             # SQL query generation
│   │   ├── forecast_agent.py        # Demand forecasting
│   │   ├── inventory_agent.py      # Inventory optimization ⭐
│   │   └── smart_insights_agent.py # AI insights generation
│   │
│   ├── ui/                   # Streamlit web interface
│   │   ├── app.py            # Main app entry point
│   │   └── components/       # UI components
│   │       ├── sidebar.py    # Navigation sidebar
│   │       ├── dashboard.py  # Dashboard page
│   │       ├── optimization.py  # Inventory optimization page
│   │       ├── analytics.py  # Analytics page
│   │       └── forecast_view.py  # Forecasting page
│   │
│   ├── panel_xgboost_model_loader.py  # Panel model loader (multi-step & recursive)
│   ├── core/                 # Core utilities
│   └── utils/                # Helper utilities
│
├── models_panel/             # Pre-trained Panel XGBoost models
│   ├── xgboost_panel_multistep_*.pkl  # Multi-step model (fast)
│   └── xgboost_panel_*.pkl           # Recursive model (fallback)
│
├── train_xgboost_panel_multistep.py  # Training script for multi-step model
├── run_ui.py                 # UI launcher
└── README.md                 # This file
```

---

## 🎯 Core Capabilities

### 1. Inventory Optimization

**Location**: `agent/agents/inventory_agent.py`

Intelligent inventory management with:

- **Per-Item Forecasting**: Each (product, branch) combination gets its own forecast
- **Smart Routing**:
  - **≥ 14 days history** → Panel XGBoost (multi-step) - Best accuracy
  - **7-13 days history** → Moving Average with trend - Balanced
  - **< 7 days history** → Simple Average or Cold Start - Conservative
- **Metrics Calculation**: ROP, Safety Stock, EOQ
- **Transfer Opportunities**: Find nearby branches for stock transfers
- **Action Plans**: Prioritized recommendations (HIGH/MEDIUM/LOW)

**Performance Optimizations** (2025-01-06):
- ✅ Caching for `_get_forecast_base_date()` to avoid repeated system calls
- ✅ Pre-computed date statistics for all items (vectorized)
- ✅ Early filtering and routing validation
- ✅ Vectorized batch processing (single model call for multiple series)
- ✅ Optimized database queries (bulk queries with IN clauses)
- ✅ Detailed timing logs for each step

**Example Query**:
```
"Tối ưu hóa tồn kho của chi nhánh đà nẵng"
```

### 2. Demand Forecasting

**Location**: `agent/agents/forecast_agent.py`

Multi-model forecasting system:

- **Panel XGBoost Models**:
  - **Multi-step Model** (Priority): Forecasts all 30 days in one prediction (faster)
  - **Recursive Model** (Fallback): Forecasts step-by-step
- **Features**: 30+ time-series features (lag, rolling, trend, volatility, date)
- **Batch Processing**: Multiple series in one model call

**Example Query**:
```
"Dự báo nhu cầu 30 ngày tới cho chi nhánh cà mau"
```

### 3. Entity Extraction

**Location**: `agent/agents/entity_extractor.py`

Smart entity extraction with fixes for:

- ✅ **"Top N" Queries**: Detects ranking queries (e.g., "top 10 chi nhánh") and returns empty branch_codes to allow general ranking
- ✅ **Region Analysis**: Distinguishes between:
  - "theo vùng miền" (general analysis) → scope="all", regions=[]
  - "miền trung" (specific region) → scope="specific", regions=["MIỀN TRUNG"]
- ✅ **Fuzzy Matching**: Matches branch/product names with diacritics handling

**Recent Fixes** (2025-01-06):
- Fixed "theo vùng miền" incorrectly extracting "miền trung"
- Fixed "top 10" queries returning only 5 branches
- Added detection for general analysis queries

### 4. SQL Generation

**Location**: `agent/agents/sql_agent.py`

Intelligent SQL query generation:

- **Entity-Aware**: Uses extracted branch_codes, product_codes, regions
- **Smart Filtering**:
  - "Top N" queries → GROUP BY + ORDER BY + LIMIT (no WHERE filter)
  - "theo vùng miền" → GROUP BY region (no WHERE filter)
- **Safety**: Validates SQL, prevents injection, uses parameterized queries

**Recent Fixes** (2025-01-06):
- Fixed region filtering for general analysis queries
- Fixed "top N" queries to use GROUP BY instead of WHERE filters
- Added instructions for LLM to handle grouping vs filtering correctly

### 5. Analytics

**Location**: `agent/ui/components/analytics.py`

Natural language analytics:

- Custom queries with automatic SQL generation
- Interactive visualizations (Plotly)
- Data tables with filtering
- CSV export

**Example Queries**:
- "Top 10 chi nhánh có doanh thu cao nhất"
- "Phân tích doanh số theo vùng miền"
- "Thống kê tồn kho theo chi nhánh"

---

## 🎨 UI Features

### Modern Design (2025-01-06)

- **Main Background**: Clean white (`#ffffff`) for better readability
- **Sidebar**: Dark theme with distinct menu items
- **Charts**: Transparent backgrounds with Plotly interactivity
- **Responsive**: Works on desktop, tablet, and mobile

### Pages

1. **📊 Dashboard**: Real-time metrics, trends, top products
2. **🎯 Inventory Optimization**: AI-powered recommendations
3. **📈 Analytics**: Custom queries and visualizations
4. **🔮 Forecast**: Demand prediction with configurable horizon

---

## ⚡ Performance

### Optimizations (2025-01-06)

1. **Vectorized Batch Processing**: Single model call for multiple series (faster than threading due to GIL)
2. **Caching**: 
   - Forecast base date caching
   - Pre-computed date statistics
   - Timeseries cache for historical data
3. **Smart Routing**: Routes to appropriate model based on data availability
4. **Bulk Queries**: Single SQL query for multiple items (IN clauses)
5. **Timing Logs**: Detailed breakdown for each step

### Timing Breakdown Example

```
✅ Inventory Optimization completed in 45.23s
📊 Timing Breakdown:
   • Step 1 (Fetch Inventory): 2.15s
   • Step 2 (Generate Forecasts): 38.42s  ← Main bottleneck
   • Step 3 (Calculate Metrics): 1.89s
   • Step 4 (Find Transfers): 0.67s
   • Step 5 (Action Plan): 0.12s
   • Step 6 (Visualization): 1.23s
   • Step 7 (AI Insights): 0.75s
```

---

## 📚 Documentation

- **Model Documentation**: `MODEL_DOCUMENTATION.md` - Detailed model architecture and features
- **Workflow**: `INVENTORY_OPTIMIZATION_WORKFLOW.md` - Complete optimization workflow
- **Data Processing**: `DATA_PROCESSING_FILES.md` - Data pipeline details
- **Forecast Pipeline**: `FORECAST_PIPELINE_FILES.md` - Forecasting system architecture
- **UI Guide**: `agent/ui/README.md` - Streamlit UI documentation

---

## 🔧 Configuration

### Inventory Optimization Parameters

**File**: `agent/agents/inventory_agent.py`

```python
self.service_level = 0.95  # 95% service level
self.lead_time_days = 7    # Default lead time
self.max_transfer_distance_km = 200  # Max distance for transfer
self.max_forecast_vs_recent_ratio = 1.2  # Cap forecasts to avoid spikes
self.bulk_query_chunk_size = 200  # Items per bulk SQL query
```

### Routing Logic

```python
# >= 14 days history → Panel XGBoost (multi-step) - Best accuracy
# 7-13 days history → Moving Average with trend - Balanced
# < 7 days history → Simple Average or Cold Start - Conservative
# Stale data (> 90 days) → Cold Start (dead stock)
# Sparse data (< 2 records) → Cold Start (insufficient data)
```

---

## 🐛 Recent Fixes (2025-01-06)

### 1. Region Filtering Fix
- **Issue**: "Phân tích doanh số theo vùng miền" only returned 2 regions instead of 5
- **Fix**: Updated `EntityExtractor` to recognize "theo vùng miền" as general analysis (scope="all", regions=[])
- **Fix**: Updated `SQLAgent` to use GROUP BY region instead of WHERE filter

### 2. Top N Query Fix
- **Issue**: "Top 10 chi nhánh có doanh thu cao nhất" only returned 5 branches
- **Fix**: Updated `EntityExtractor` to detect "top N" queries and return empty branch_codes
- **Fix**: Updated `SQLAgent` to use GROUP BY + ORDER BY + LIMIT instead of WHERE filter

### 3. UI Improvements
- **Background**: Changed main content to white (`#ffffff`)
- **Menu Bar**: Enhanced visual distinction between menu items
- **About Section**: Fixed HTML rendering in sidebar (changed to markdown)

### 4. Performance Optimizations
- Caching for forecast base date
- Pre-computed date statistics
- Vectorized batch processing
- Optimized database queries
- Detailed timing logs

---

## 🚀 Deployment

### Local Development

```bash
streamlit run agent/ui/app.py
```

### Production (Streamlit Cloud)

1. Push code to GitHub
2. Connect repository at [share.streamlit.io](https://share.streamlit.io)
3. Set environment variables
4. Deploy!

### Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements*.txt ./
RUN pip install -r requirements.txt -r requirements_streamlit.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "agent/ui/app.py", "--server.port=8501"]
```

---

## 📝 License

MIT License - See LICENSE file

---

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - Web framework
- [LangChain](https://www.langchain.com/) - LLM orchestration
- [XGBoost](https://xgboost.readthedocs.io/) - Machine learning
- [PostgreSQL](https://www.postgresql.org/) - Database
- [Plotly](https://plotly.com/) - Interactive charts
- [Qdrant](https://qdrant.tech/) - Vector database

---



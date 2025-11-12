# 🧱 BrickDemand Inventory AI - Streamlit UI

**Professional Web Interface for Predictive Inventory Management**

Version: 3.1 | Built with Streamlit + Multi-Agent System

---

## 🎯 OVERVIEW

Modern, responsive web interface for the BrickDemand Multi-Agent System featuring:

- 📊 **Real-time Dashboard** - Live metrics and insights
- 🎯 **Inventory Optimization** - AI-powered restock recommendations
- 📈 **Interactive Analytics** - Custom queries and visualizations  
- 🔮 **Demand Forecasting** - Predictive analytics for planning
- 💾 **Export Capabilities** - Excel, CSV, and images

---

## 🚀 QUICK START

### 1. Install Dependencies

```bash
pip install -r requirements_streamlit.txt
```

### 2. Set Environment Variables

Create `.env` file:

```env
# Database
PG_USER=postgres
PG_PASSWORD=your_password
PG_HOST=localhost
PG_PORT=5433
PG_DB=brickdemand

# OpenAI (optional)
OPENAI_API_KEY=your_api_key

# HuggingFace (optional)
HUGGINGFACEHUB_API_TOKEN=your_token
```

### 3. Run the App

```bash
streamlit run agent/ui/app.py
```

The app will open at `http://localhost:8501`

---

## 📁 PROJECT STRUCTURE

```
agent/
├── ui/
│   ├── app.py                    # Main Streamlit app
│   ├── components/
│   │   ├── __init__.py
│   │   ├── sidebar.py           # Navigation sidebar
│   │   ├── dashboard.py         # Dashboard page
│   │   ├── optimization.py      # Inventory optimization
│   │   ├── analytics.py         # Analytics page
│   │   └── forecast_view.py     # Forecasting page
│   └── README.md                # This file
│
├── core/
│   ├── __init__.py
│   ├── config.py                # Configuration
│   └── orchestrator_loader.py   # System initialization
│
└── improved_mas.py              # Core MAS logic
```

---

## 🎨 FEATURES

### 1. **Dashboard** 📊

Real-time overview with:
- Key metrics (branches, products, stock, sales)
- Sales trends (last 30 days)
- Top products analysis
- Regional distribution
- Low stock alerts
- Query history

**Perfect for:** Quick monitoring and situational awareness

---

### 2. **Inventory Optimization** 🎯

AI-powered recommendations with:

#### Input
- Quick templates or custom questions
- Entity extraction (branches, regions, products)
- Natural language processing

#### Analysis
- Current stock vs ROP (Reorder Point)
- Safety Stock calculations
- Demand forecasting per item
- Transfer opportunities (nearby branches)
- EOQ (Economic Order Quantity)

#### Output
- Priority-grouped action plan (HIGH/MEDIUM/LOW)
- Restock orders with quantities
- Internal transfer recommendations
- Distance-optimized logistics
- AI-generated strategic insights

#### Features
- 📋 **Action Plan Tab**: Priority-based recommendations
- 📈 **Charts Tab**: Visual analysis
- 🧠 **AI Insights Tab**: Strategic recommendations
- 📦 **Data Tab**: Filterable detailed tables
- 💾 **Export Tab**: Excel, CSV downloads

**Perfect for:** Daily inventory management decisions

---

### 3. **Analytics** 📊

Interactive data analysis with:
- Natural language queries
- SQL generation
- Custom visualizations
- Data tables
- CSV export

**Example Queries:**
- "Top 10 sản phẩm bán chạy nhất tháng này"
- "Phân tích doanh số theo vùng miền"
- "Thống kê tồn kho theo chi nhánh"

**Perfect for:** Ad-hoc analysis and exploration

---

### 4. **Forecast** 🔮

Demand prediction with:
- Configurable horizon (7-90 days)
- Historical vs forecast comparison
- Trend analysis
- Interactive Plotly charts
- Downloadable predictions

**Metrics Shown:**
- Recent average daily demand
- Forecast average daily demand
- Total forecast
- Trend direction (increasing/decreasing/stable)

**Perfect for:** Planning and budgeting

---

## 🎯 USE CASES

### Use Case 1: Daily Stock Check

1. Open **Dashboard**
2. Check "Low Stock Alerts"
3. If items < 10 units → go to Inventory Optimization

---

### Use Case 2: Branch Optimization

1. Open **Inventory Optimization**
2. Select template: "Tối ưu hóa tồn kho của chi nhánh đà nẵng"
3. Click "🚀 Run Analysis"
4. Review HIGH priority actions
5. Export Excel for procurement team

---

### Use Case 3: Sales Analysis

1. Open **Analytics**
2. Ask: "Top 10 sản phẩm bán chạy nhất tháng này"
3. View chart and table
4. Download CSV for reporting

---

### Use Case 4: Demand Planning

1. Open **Forecast**
2. Set horizon: 30 days
3. Ask: "Dự báo doanh số 30 ngày tới"
4. Review forecast metrics
5. Download forecast CSV for planning

---

## ⚙️ CONFIGURATION

### App Settings (sidebar)

- **Forecast Horizon**: 7-90 days (default: 30)
- **Max Transfer Distance**: 50-500 km (default: 200)
- **LLM Model**: OpenAI / HuggingFace

### Advanced Settings (`agent/core/config.py`)

```python
# Inventory optimization parameters
INVENTORY_CONFIG = {
    "service_level": 0.95,        # 95% service level
    "lead_time_days": 7,          # Delivery time
    "max_transfer_distance_km": 200,
    "min_forecast_days": 2,       # Min days for forecast
    "ordering_cost": 1000,        # Per order
    "holding_cost": 50            # Per unit per year
}
```

---

## 🎨 UI CUSTOMIZATION

### Theme

Edit custom CSS in `app.py`:

```python
st.markdown("""
<style>
    /* Your custom styles */
    .main {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)
```

### Logo

Replace placeholder in `sidebar.py`:

```python
st.image("path/to/your/logo.png", use_container_width=True)
```

---

## 📊 PERFORMANCE

### Caching

The app uses Streamlit caching for:

1. **System Initialization** - Cached via `@st.cache_resource`
2. **Database Queries** - Connection pooling
3. **Session State** - Persistent across reruns

### Tips for Better Performance

1. **Limit data size**: Filter by branch/region
2. **Use templates**: Pre-defined queries are faster
3. **Export large results**: Download CSV instead of viewing all

---

## 🐛 TROUBLESHOOTING

### Issue: "System Not Initialized"

**Solution:**
1. Check database connection in `.env`
2. Click "🚀 Initialize System" in sidebar
3. Wait for success message

---

### Issue: "No data returned"

**Solution:**
1. Check if database has data for selected branches
2. Try different branch (e.g., "Bình Chánh" has 24K+ sales)
3. Review query in backend logs

---

### Issue: "Export failed"

**Solution:**
1. Check write permissions in project directory
2. Close Excel file if already open
3. Check available disk space

---

### Issue: Charts not loading

**Solution:**
1. Check `charts/` directory exists
2. Verify matplotlib/PIL are installed
3. Clear browser cache

---

## 🚀 DEPLOYMENT

### Local Development

```bash
streamlit run agent/ui/app.py
```

### Production (Streamlit Cloud)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Set environment variables
5. Deploy!

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "agent/ui/app.py", "--server.port=8501"]
```

```bash
docker build -t brickdemand-ui .
docker run -p 8501:8501 brickdemand-ui
```

---

## 📱 MOBILE SUPPORT

The UI is responsive and works on:
- ✅ Desktop (recommended)
- ✅ Tablet
- ✅ Mobile (limited features)

---

## 🎓 TUTORIALS

### Tutorial 1: First Optimization

1. **Initialize**: Click "🚀 Initialize System"
2. **Navigate**: Go to "Inventory Optimization"
3. **Query**: Select "Tối ưu hóa tồn kho của chi nhánh bình chánh"
4. **Run**: Click "🚀 Run Analysis"
5. **Review**: Check HIGH priority actions
6. **Export**: Download Excel file
7. **Action**: Share with procurement team

### Tutorial 2: Custom Analysis

1. **Navigate**: Go to "Analytics"
2. **Query**: Type "Show sales last 7 days by branch"
3. **Run**: Click "🚀 Run Analysis"
4. **Explore**: View chart and table
5. **Filter**: Use interactive controls
6. **Export**: Download CSV

---

## 🔐 SECURITY

### Best Practices

1. **Environment Variables**: Never commit `.env`
2. **Database Access**: Use read-only user for production
3. **API Keys**: Rotate regularly
4. **HTTPS**: Use SSL in production
5. **Authentication**: Add auth layer (Streamlit supports this)

---

## 📈 ANALYTICS & MONITORING

### Track Usage

Streamlit provides analytics at:
- Total visits
- Active users
- Page views
- Error rates

### Custom Logging

Add logging to track queries:

```python
import logging

logging.info(f"User query: {question}")
logging.info(f"Result: {result['success']}")
```

---

## 🆘 SUPPORT

### Documentation
- **Main README**: `../README_USAGE.md`
- **Sparse Data Fix**: `../SPARSE_DATA_FIX.md`
- **Output Improvements**: `../DETAILED_OUTPUT_IMPROVEMENTS.md`

### Issues
Report bugs at: [GitHub Issues](https://github.com/yourusername/brickdemand/issues)

### Contact
Email: your.email@example.com

---

## 🎉 WHAT'S NEW

### Version 3.1

- ✅ Beautiful Streamlit UI
- ✅ Real-time dashboard
- ✅ Interactive charts (Plotly)
- ✅ Export capabilities
- ✅ Mobile responsive
- ✅ Quick templates
- ✅ Session state management
- ✅ Performance caching

---

## 📝 LICENSE

MIT License - See LICENSE file

---

## 🙏 ACKNOWLEDGMENTS

Built with:
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Plotly](https://plotly.com/)
- [PostgreSQL](https://www.postgresql.org/)

---

**🧱 Happy Optimizing!** 🚀


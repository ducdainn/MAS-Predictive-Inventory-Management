"""
🤖 BrickDemand Predictive Inventory Management System
Multi-Agent System with Streamlit UI

"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.core.config import APP_CONFIG
from agent.ui.components import sidebar, dashboard, optimization, analytics, forecast_view

# Page config
st.set_page_config(
    page_title="Predictive Inventory Management System",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/ducdainn/brickdemand',
        'Report a bug': 'https://github.com/ducdainn/brickdemand/issues',
        'About': '''
        # BrickDemand Inventory AI
        **Predictive Inventory Management System**
        
        Powered by Multi-Agent AI System
        - 🔮 Demand Forecasting
        - 📊 Analytics & Insights
        - 🎯 Inventory Optimization
        - 🚚 Transfer Recommendations
        
        Version 3.1 - Enhanced Output
        '''
    }
)

# Custom CSS
st.markdown("""
<style>
    /* Force light theme */
    .stApp {
        background-color: #ffffff;
        color: #262730;
    }
    
    /* Main theme */
    .main {
        padding: 0rem 1rem;
        background-color: #ffffff;
    }
    
    /* Header */
    h1 {
        color: #1f77b4 !important;
        font-weight: 700;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #1f77b4;
    }
    
    h2, h3, h4, h5, h6 {
        color: #262730 !important;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Cards */
    .css-1r6slb0 {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #145a8a;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #262730 !important;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        color: #262730 !important;
    }
    
    /* Success/Error alerts */
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
    }
    
    /* Warning boxes */
    .stWarning {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    
    /* Markdown text - Fix white text on white background */
    .stMarkdown {
        color: #262730;
    }
    
    /* Text elements */
    p, li, span, div {
        color: #262730 !important;
    }
    
    /* Code blocks */
    code {
        background-color: #f0f2f6;
        color: #262730;
        padding: 0.2rem 0.4rem;
        border-radius: 0.3rem;
    }
    
    /* Pre-formatted text */
    pre {
        background-color: #f0f2f6;
        color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        overflow-x: auto;
    }
    
    /* DataFrame */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #ffffff;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        color: #262730;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #1f77b4;
        border-bottom-color: #1f77b4;
    }
    
    /* Input fields */
    .stTextInput input, .stSelectbox select, .stTextArea textarea {
        background-color: #ffffff !important;
        color: #262730 !important;
        border: 1px solid #ddd !important;
    }
    
    /* Select dropdowns */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        color: #262730 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        color: #262730 !important;
    }
    
    /* All text elements force dark color */
    .stMarkdown, .stText, label {
        color: #262730 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state and auto-load system
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
    st.session_state.initialized = False
    st.session_state.last_result = None
    st.session_state.query_history = []
    st.session_state.init_error = None
    
    # Auto-initialize system on first load
    try:
        from agent.core.orchestrator_loader import initialize_system
        st.session_state.orchestrator = initialize_system()
        st.session_state.initialized = True
    except Exception as e:
        st.session_state.init_error = str(e)
        st.session_state.initialized = False

# Render sidebar
page = sidebar.render()

# Main content
if not st.session_state.initialized:
    # Show error if initialization failed
    if st.session_state.init_error:
        st.title("❌ System Initialization Failed")
        st.error(f"**Error:** {st.session_state.init_error}")
        
        st.markdown("---")
        st.markdown("""
        ### 🔧 Troubleshooting
        
        Please check:
        1. Database connection settings
        2. Environment variables (HUGGINGFACEHUB_API_TOKEN, OPENAI_API_KEY)
        3. Required packages installed
        
        """)
        
        if st.button("🔄 Retry Initialization", type="primary"):
            st.session_state.init_error = None
            st.rerun()
    else:
        # Show loading spinner (this should rarely be seen due to auto-init)
        with st.spinner("🚀 Initializing Multi-Agent System..."):
            st.info("System is loading, please wait...")

else:
    # Render selected page
    if page == "Dashboard":
        dashboard.render(st.session_state.orchestrator)
    
    elif page == "Inventory Optimization":
        optimization.render(st.session_state.orchestrator)
    
    elif page == "Analytics":
        analytics.render(st.session_state.orchestrator)
    
    elif page == "Forecast":
        forecast_view.render(st.session_state.orchestrator)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>BrickDemand Inventory AI</strong> | Version 3.1 | Powered by Multi-Agent System</p>
    <p>© 2025 | Built with ❤️ using Streamlit & LangChain</p>
</div>
""", unsafe_allow_html=True)


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

# Modern Custom CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* App Background - Clean White */
    .stApp {
        background: #f5f7fa;
        min-height: 100vh;
    }
    
    /* Main Content Area - Pure White */
    .main {
        padding: 2rem 2.5rem;
        background: #ffffff;
        border-radius: 0;
        margin: 0;
        box-shadow: none;
        backdrop-filter: none;
    }
    
    /* Header Styles - Modern Typography */
    h1 {
        color: #1a1a2e !important;
        font-weight: 800 !important;
        font-size: 2.5rem !important;
        padding-bottom: 0.75rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        border-bottom: 3px solid transparent;
        border-image: linear-gradient(90deg, #667eea, #764ba2) 1;
    }
    
    h2 {
        color: #2d3748 !important;
        font-weight: 700 !important;
        font-size: 1.75rem !important;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3, h4, h5, h6 {
        color: #4a5568 !important;
        font-weight: 600 !important;
    }
    
    /* Enhanced Metrics Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 600 !important;
        color: #718096 !important;
        font-size: 0.95rem !important;
    }
    
    /* Remove individual white boxes - use unified container */
    .css-1r6slb0, .element-container {
        background: transparent !important;
        backdrop-filter: none !important;
        border-radius: 0 !important;
        padding: 0 !important;
        box-shadow: none !important;
        border: none !important;
    }
    
    /* Enhanced Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        font-size: 0.95rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #5568d3 0%, #6b3f8f 100%);
    }
    
    /* Modern Sidebar - Enhanced Contrast */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
        box-shadow: 4px 0 25px rgba(0, 0, 0, 0.4);
        border-right: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    [data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }
    
    /* Sidebar Labels and Text */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #cbd5e0 !important;
    }
    
    /* Sidebar divs - transparent background unless explicitly styled */
    [data-testid="stSidebar"] div {
        color: #cbd5e0 !important;
    }
    
    /* Force all sidebar containers to have dark/transparent backgrounds */
    [data-testid="stSidebar"] .row-widget,
    [data-testid="stSidebar"] [class*="st"],
    [data-testid="stSidebar"] [class*="css"] {
        background-color: transparent !important;
    }
    
    /* Sidebar Button Styling */
    [data-testid="stSidebar"] .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    [data-testid="stSidebar"] .stButton>button:hover {
        background: linear-gradient(135deg, #5568d3 0%, #6b3f8f 100%) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Sidebar Radio Buttons - High Contrast */
    [data-testid="stSidebar"] .stRadio {
        background: rgba(30, 41, 59, 0.5);
        padding: 0.5rem;
        border-radius: 12px;
        margin: 0.25rem 0;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        color: #cbd5e1 !important;
        font-weight: 500;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        transition: all 0.3s ease;
        margin: 0.25rem 0;
        background: rgba(15, 23, 42, 0.4);
        border: 2px solid transparent;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(102, 126, 234, 0.2) !important;
        border-color: rgba(102, 126, 234, 0.4);
        color: #ffffff !important;
        transform: translateX(5px);
    }
    
    [data-testid="stSidebar"] .stRadio input[type="radio"]:checked + label {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        font-weight: 700;
        border-color: #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Enhanced Alerts */
    .stSuccess {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 1.25rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.3);
        font-weight: 500;
    }
    
    .stError {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        color: white;
        padding: 1.25rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(245, 101, 101, 0.3);
        font-weight: 500;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
        color: white;
        padding: 1.25rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(66, 153, 225, 0.3);
        font-weight: 500;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        color: white;
        padding: 1.25rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(237, 137, 54, 0.3);
        font-weight: 500;
    }
    
    /* Enhanced Input Fields */
    .stTextInput>div>div>input, 
    .stSelectbox>div>div>select, 
    .stTextArea>div>div>textarea {
        background-color: white !important;
        color: #2d3748 !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 10px !important;
        padding: 0.75rem !important;
        transition: all 0.3s ease !important;
        font-weight: 500 !important;
    }
    
    .stTextInput>div>div>input:focus, 
    .stSelectbox>div>div>select:focus, 
    .stTextArea>div>div>textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Enhanced Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.5);
        padding: 0.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: #718096;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Enhanced DataFrames */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar Expander - Dark Theme */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.6) !important;
        color: #ffffff !important;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.875rem 1rem;
        border: 2px solid rgba(102, 126, 234, 0.3);
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stSidebar"] .streamlit-expanderHeader:hover {
        background: rgba(30, 41, 59, 0.8) !important;
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    [data-testid="stSidebar"] .streamlit-expanderContent {
        background: rgba(15, 23, 42, 0.5) !important;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin-top: 0.5rem;
    }
    
    /* Sidebar Input Fields - Dark Theme - All Containers */
    [data-testid="stSidebar"] .stNumberInput,
    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] .stTextInput,
    [data-testid="stSidebar"] .stTextArea {
        background-color: transparent !important;
    }
    
    [data-testid="stSidebar"] .stNumberInput>div,
    [data-testid="stSidebar"] .stSelectbox>div,
    [data-testid="stSidebar"] .stTextInput>div,
    [data-testid="stSidebar"] .stTextArea>div {
        background-color: transparent !important;
    }
    
    [data-testid="stSidebar"] .stNumberInput>div>div,
    [data-testid="stSidebar"] .stSelectbox>div>div,
    [data-testid="stSidebar"] .stTextInput>div>div,
    [data-testid="stSidebar"] .stTextArea>div>div {
        background-color: rgba(30, 41, 59, 0.6) !important;
        border-radius: 10px;
    }
    
    [data-testid="stSidebar"] .stNumberInput>div>div>input,
    [data-testid="stSidebar"] .stSelectbox>div>div>select,
    [data-testid="stSidebar"] .stTextInput>div>div>input,
    [data-testid="stSidebar"] .stTextArea>div>div>textarea {
        background-color: rgba(30, 41, 59, 0.8) !important;
        color: #ffffff !important;
        border: 2px solid rgba(102, 126, 234, 0.4) !important;
        border-radius: 10px !important;
        padding: 0.75rem !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stSidebar"] .stNumberInput>div>div>input:focus,
    [data-testid="stSidebar"] .stSelectbox>div>div>select:focus,
    [data-testid="stSidebar"] .stTextInput>div>div>input:focus,
    [data-testid="stSidebar"] .stTextArea>div>div>textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
        background-color: rgba(30, 41, 59, 1) !important;
    }
    
    /* Number Input Stepper Buttons */
    [data-testid="stSidebar"] .stNumberInput>div>div>div {
        background-color: rgba(30, 41, 59, 0.8) !important;
        color: #ffffff !important;
        border-radius: 10px;
    }
    
    /* Selectbox Dropdown */
    [data-testid="stSidebar"] .stSelectbox>div>div>div {
        background-color: rgba(30, 41, 59, 0.8) !important;
        color: #ffffff !important;
    }
    
    /* Selectbox Dropdown Options */
    [data-testid="stSidebar"] .stSelectbox [role="listbox"],
    [data-testid="stSidebar"] .stSelectbox [role="option"] {
        background-color: rgba(30, 41, 59, 0.95) !important;
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox [role="option"]:hover {
        background-color: rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Remove all white backgrounds from sidebar containers - Force Override */
    [data-testid="stSidebar"] [style*="background: white"],
    [data-testid="stSidebar"] [style*="background-color: white"],
    [data-testid="stSidebar"] [style*="background:#fff"],
    [data-testid="stSidebar"] [style*="background-color:#fff"],
    [data-testid="stSidebar"] [style*="background: #fff"],
    [data-testid="stSidebar"] [style*="background-color: #fff"] {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Sidebar element containers - force dark/transparent */
    [data-testid="stSidebar"] .element-container,
    [data-testid="stSidebar"] .css-1r6slb0,
    [data-testid="stSidebar"] .block-container,
    [data-testid="stSidebar"] .stNumberInput,
    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] .stTextInput {
        background-color: transparent !important;
        background: transparent !important;
    }
    
    /* All sidebar form elements - dark theme */
    [data-testid="stSidebar"] .stNumberInput,
    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] .stTextInput,
    [data-testid="stSidebar"] .stTextArea {
        background-color: transparent !important;
    }
    
    /* Sidebar expander content - dark */
    [data-testid="stSidebar"] .streamlit-expanderContent {
        background-color: rgba(15, 23, 42, 0.5) !important;
    }
    
    /* Override Streamlit default white backgrounds in sidebar */
    [data-testid="stSidebar"] [data-baseweb="input"] {
        background-color: rgba(30, 41, 59, 0.8) !important;
    }
    
    [data-testid="stSidebar"] [data-baseweb="select"] {
        background-color: rgba(30, 41, 59, 0.8) !important;
    }
    
    /* Main Expander (not in sidebar) */
    .streamlit-expanderHeader:not([data-testid="stSidebar"] .streamlit-expanderHeader) {
        background: rgba(102, 126, 234, 0.1);
        color: #667eea !important;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 1rem;
    }
    
    /* Markdown and Text */
    .stMarkdown {
        color: #2d3748;
        line-height: 1.7;
    }
    
    p, li, span, div {
        color: #2d3748 !important;
    }
    
    /* Code blocks */
    code {
        background: rgba(102, 126, 234, 0.1);
        color: #667eea;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9em;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5568d3 0%, #6b3f8f 100%);
    }
    
    /* Footer */
    footer {
        background: rgba(26, 32, 44, 0.9) !important;
        color: #f7fafc !important;
        padding: 1.5rem !important;
        border-radius: 12px 12px 0 0;
        margin-top: 3rem;
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-color: #667eea transparent #764ba2 transparent !important;
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

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem 1rem; margin-top: 3rem;'>
    <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                padding: 1.5rem; border-radius: 16px; border: 1px solid rgba(102, 126, 234, 0.2);'>
        <p style='color: #4a5568; font-size: 0.95rem; margin: 0.5rem 0; font-weight: 600;'>
            🧱 <strong>BrickDemand Inventory AI</strong>
        </p>
        <p style='color: #718096; font-size: 0.85rem; margin: 0.5rem 0;'>
            Version 3.1 | Powered by Multi-Agent System
        </p>
        <p style='color: #a0aec0; font-size: 0.8rem; margin-top: 1rem;'>
            © 2025 | Built with ❤️ using Streamlit & LangChain
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


"""
Sidebar component for navigation and settings
"""

import streamlit as st

def render():
    """
    Render sidebar with navigation and settings
    
    Returns:
        str: Selected page name
    """
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=BrickDemand", 
                 use_container_width=True)
        
        st.title("🧱 BrickDemand AI")
        
        # System status
        if st.session_state.get('initialized', False):
            st.success("✅ System Online")
        else:
            st.warning("⚠️ System Not Initialized")
        
        st.markdown("---")
        
        # Navigation
        st.subheader("📋 Navigation")
        
        page = st.radio(
            "Select Page",
            ["Dashboard", "Inventory Optimization", "Analytics", "Forecast"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick Actions
        if st.session_state.get('initialized', False):
            st.subheader("⚡ Quick Actions")
            
            if st.button("🔄 Refresh System", use_container_width=True):
                st.cache_resource.clear()
                st.rerun()
            
            if st.button("🗑️ Clear History", use_container_width=True):
                if st.session_state.orchestrator:
                    st.session_state.orchestrator.clear_memory()
                    st.session_state.query_history = []
                    st.success("History cleared!")
            
            st.markdown("---")
            
            # Settings
            with st.expander("⚙️ Settings"):
                st.number_input(
                    "Forecast Horizon (days)",
                    min_value=7,
                    max_value=90,
                    value=30,
                    key="horizon_days",
                    help="Number of days to forecast"
                )
                
                st.number_input(
                    "Max Transfer Distance (km)",
                    min_value=50,
                    max_value=500,
                    value=200,
                    key="max_transfer_distance",
                    help="Maximum distance for inventory transfers"
                )
                
                st.selectbox(
                    "LLM Model",
                    ["OpenAI (gpt-4o-mini)", "HuggingFace (Qwen2.5)"],
                    key="llm_model",
                    help="Select language model"
                )
        
        st.markdown("---")
        
        # Info
        with st.expander("ℹ️ About"):
            st.markdown("""
            **BrickDemand Inventory AI**
            
            Version: 3.1
            
            **Features:**
            - 🔮 Demand Forecasting
            - 📊 Smart Analytics
            - 🎯 Inventory Optimization
            - 🚚 Transfer Recommendations
            
            **Powered by:**
            - LangChain
            - OpenAI / HuggingFace
            - PostgreSQL
            - Streamlit
            """)
        
        # Footer
        st.markdown("---")
        st.caption("© 2025 BrickDemand AI")
    
    return page


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
        # Logo Header with Gradient
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem 0; margin-bottom: 1rem;'>
            <h1 style='font-size: 2rem; margin: 0; background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #4facfe 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;'>
                🧱 BrickDemand AI
            </h1>
            <p style='color: #cbd5e0; margin-top: 0.5rem; font-size: 0.85rem;'>
                Predictive Inventory Management
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # System status with badge - Enhanced contrast
        status_container = st.container()
        with status_container:
            if st.session_state.get('initialized', False):
                st.markdown("""
                <div style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                            padding: 0.875rem 1.25rem; border-radius: 12px; text-align: center;
                            box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4); 
                            border: 2px solid rgba(16, 185, 129, 0.3);
                            margin-bottom: 1.5rem;'>
                    <p style='color: #ffffff; margin: 0; font-weight: 700; font-size: 0.95rem; 
                              text-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
                        ✅ System Online
                    </p>
                </div>
                """, unsafe_allow_html=True)
            elif st.session_state.get('init_error'):
                st.markdown("""
                <div style='background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); 
                            padding: 0.875rem 1.25rem; border-radius: 12px; text-align: center;
                            box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4); 
                            border: 2px solid rgba(239, 68, 68, 0.3);
                            margin-bottom: 1.5rem;'>
                    <p style='color: #ffffff; margin: 0; font-weight: 700; font-size: 0.95rem; 
                              text-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
                        ❌ Initialization Failed
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); 
                            padding: 0.875rem 1.25rem; border-radius: 12px; text-align: center;
                            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4); 
                            border: 2px solid rgba(59, 130, 246, 0.3);
                            margin-bottom: 1.5rem;'>
                    <p style='color: #ffffff; margin: 0; font-weight: 700; font-size: 0.95rem; 
                              text-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
                        🔄 Initializing...
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation with Icons - Enhanced styling
        st.markdown("""
        <div style='margin-bottom: 1.25rem; padding: 0.75rem 0; 
                    border-bottom: 2px solid rgba(102, 126, 234, 0.3);'>
            <h3 style='color: #ffffff; font-size: 1.15rem; font-weight: 800; 
                       margin-bottom: 0.5rem; text-transform: uppercase; 
                       letter-spacing: 0.5px;'>
                📋 Navigation
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.radio(
            "Select Page",
            ["Dashboard", "Inventory Optimization", "Analytics", "Forecast"],
            label_visibility="collapsed",
            format_func=lambda x: {
                "Dashboard": "📊 Dashboard",
                "Inventory Optimization": "🎯 Inventory Optimization",
                "Analytics": "📈 Analytics",
                "Forecast": "🔮 Forecast"
            }.get(x, x)
        )
        
        st.markdown("---")
        
        # Quick Actions - Enhanced styling
        if st.session_state.get('initialized', False):
            st.markdown("""
            <div style='margin: 1.5rem 0; padding: 0.75rem 0; 
                        border-bottom: 2px solid rgba(102, 126, 234, 0.3);'>
                <h3 style='color: #ffffff; font-size: 1.15rem; font-weight: 800; 
                           margin-bottom: 0.5rem; text-transform: uppercase; 
                           letter-spacing: 0.5px;'>
                    ⚡ Quick Actions
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄", help="Refresh System", use_container_width=True):
                    st.cache_resource.clear()
                    st.rerun()
            
            with col2:
                if st.button("🗑️", help="Clear History", use_container_width=True):
                    if st.session_state.orchestrator:
                        st.session_state.orchestrator.clear_memory()
                        st.session_state.query_history = []
                        st.success("History cleared!")
            
            st.markdown("---")
            
            # Settings with Modern Styling
            with st.expander("⚙️ **Settings**"):
                st.markdown("""
                <div style='padding: 0.5rem 0;'>
                    <p style='color: #cbd5e0; font-size: 0.9rem; margin-bottom: 0.5rem;'>
                        ⏱️ Forecast Horizon
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.number_input(
                    "Forecast Horizon (days)",
                    min_value=7,
                    max_value=90,
                    value=30,
                    key="horizon_days",
                    help="Number of days to forecast",
                    label_visibility="collapsed"
                )
                
                st.markdown("""
                <div style='padding: 0.5rem 0; margin-top: 1rem;'>
                    <p style='color: #cbd5e0; font-size: 0.9rem; margin-bottom: 0.5rem;'>
                        📍 Max Transfer Distance
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.number_input(
                    "Max Transfer Distance (km)",
                    min_value=50,
                    max_value=500,
                    value=200,
                    key="max_transfer_distance",
                    help="Maximum distance for inventory transfers",
                    label_visibility="collapsed"
                )
                
                st.markdown("""
                <div style='padding: 0.5rem 0; margin-top: 1rem;'>
                    <p style='color: #cbd5e0; font-size: 0.9rem; margin-bottom: 0.5rem;'>
                        🤖 LLM Model
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.selectbox(
                    "LLM Model",
                    ["OpenAI (gpt-4o-mini)", "HuggingFace (Qwen2.5)"],
                    key="llm_model",
                    help="Select language model",
                    label_visibility="collapsed"
                )
        
        st.markdown("""
        <div style='margin: 1.5rem 0; border-top: 2px solid rgba(102, 126, 234, 0.3);'></div>
        """, unsafe_allow_html=True)
        
        # About Section - Using markdown with proper formatting
        with st.expander("ℹ️ About"):
            st.markdown("**BrickDemand Inventory AI**")
            st.markdown("Version 3.1")
            st.markdown("---")
            
            st.markdown("**✨ Features:**")
            st.markdown("""
            - 🔮 Demand Forecasting
            - 📊 Smart Analytics
            - 🎯 Inventory Optimization
            - 🚚 Transfer Recommendations
            """)
            
            st.markdown("**🔧 Powered by:**")
            st.markdown("""
            LangChain • OpenAI / HuggingFace  
            PostgreSQL • Streamlit
            """)
        
        # Footer - Enhanced styling
        st.markdown("""
        <div style='margin-top: 2rem; padding-top: 1.5rem; 
                    border-top: 2px solid rgba(102, 126, 234, 0.3);'>
            <div style='text-align: center; padding: 1rem; 
                        background: rgba(15, 23, 42, 0.5); 
                        border-radius: 12px; 
                        border: 1px solid rgba(102, 126, 234, 0.2);'>
                <p style='color: #ffffff; font-size: 0.85rem; margin: 0; font-weight: 600;'>
                    © 2025 BrickDemand AI
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    return page


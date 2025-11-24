"""
Orchestrator loader for Streamlit UI
Wraps the initialization with caching

UPGRADED: Now uses Advanced Memory System with Vector DB
"""

import streamlit as st

from agent.agents.orchestrator_agent import OrchestratorAgent
from agent.core.llm_provider import LLMProvider
from agent.manager.database_manager import DatabaseManager
from agent.manager.memory_manager import MemoryManager  # Advanced memory support

@st.cache_resource
def initialize_system():
    """
    Initialize the Multi-Agent System with caching.
    This prevents re-initialization on every Streamlit rerun.
    """
    print("\n" + "="*80)
    print("🚀 Initializing Multi-Agent System...")
    print("="*80 + "\n")
    
    db_manager = DatabaseManager()
    memory = MemoryManager()
    llm_provider = LLMProvider()
    
    orchestrator = OrchestratorAgent(
        db_manager=db_manager,
        memory=memory,
        llm_provider=llm_provider
    )
    
    print("\n" + "="*80)
    print("🎉 Multi-Agent System Ready!")
    print("="*80)
    
    return orchestrator

def get_orchestrator():
    """Get cached orchestrator instance"""
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = initialize_system()
    return st.session_state.orchestrator


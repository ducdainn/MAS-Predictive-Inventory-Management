"""
Compatibility facade for the improved multi-agent system.

This module re-exports the main managers/agents for backwards compatibility
while delegating the implementation to the split modules under agent/agents and
agent/manager.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

# Re-export core classes
from agent.agents.orchestrator_agent import (
    OrchestratorAgent,
    display_conversation_history,
    export_inventory_plan_to_excel,
    export_forecasts_to_csv,
    export_recommendations_to_csv,
    display_action_plan,
    export_results_to_excel,
    initialize_system,
)
from agent.core.conversation import ConversationEntry
from agent.core.llm_provider import LLMProvider
from agent.manager.database_manager import DatabaseManager
from agent.manager.memory_manager import MemoryManager

__all__ = [
    # Core managers
    "DatabaseManager",
    "MemoryManager",
    "LLMProvider",
    "OrchestratorAgent",
    "ConversationEntry",
    # Functions
    "initialize_system",
    "display_conversation_history",
    "export_results_to_excel",
    "export_inventory_plan_to_excel",
    "export_forecasts_to_csv",
    "export_recommendations_to_csv",
    "display_action_plan",
]

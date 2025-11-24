"""
Shared conversation data structures for the multi-agent system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class ConversationEntry:
    """Single conversation entry in memory."""
    timestamp: datetime
    question: str
    intent: str
    sql_query: Optional[str] = None
    result_summary: Optional[str] = None
    charts: List[str] = field(default_factory=list)
    success: bool = True  # Track if query was successful
    error_message: Optional[str] = None  # Error message if failed




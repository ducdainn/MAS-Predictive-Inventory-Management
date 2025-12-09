"""
SQL Query Logger: Centralized logging for all SQL queries generated and executed.
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any


class SQLQueryLogger:
    """
    Centralized logger for SQL queries.
    Logs all queries to a file with timestamps and context.
    """
    
    def __init__(self, log_dir: str = "sql_logs"):
        """
        Initialize SQL query logger.
        
        Args:
            log_dir: Directory to store SQL log files
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "sql_queries.log")
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Ensure log file exists with header."""
        if not os.path.exists(self.log_file) or os.path.getsize(self.log_file) == 0:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write("SQL QUERIES LOG - All generated and executed queries\n")
                f.write("=" * 100 + "\n\n")
    
    def log_query(self,
                  query: str,
                  query_type: str = "EXECUTED",
                  source: str = "UNKNOWN",
                  params: Optional[Dict] = None,
                  context: Optional[Dict[str, Any]] = None):
        """
        Log a SQL query to file.
        
        Args:
            query: The SQL query string
            query_type: Type of query (GENERATED, EXECUTED, RETRY, etc.)
            source: Source of the query (SQLAgent, DatabaseManager, InventoryAgent, etc.)
            params: Query parameters (if any)
            context: Additional context (question, intent, entities, etc.)
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"[{timestamp}] {query_type} - Source: {source}\n")
            f.write("-" * 100 + "\n")
            
            # Write context if available
            if context:
                f.write("CONTEXT:\n")
                for key, value in context.items():
                    if value is not None:
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Write parameters if available
            if params:
                f.write("PARAMETERS:\n")
                for key, value in params.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Write the query
            f.write("QUERY:\n")
            f.write(query)
            f.write("\n")
            f.write("=" * 100 + "\n\n")
    
    def log_generated_query(self,
                           query: str,
                           source: str,
                           question: Optional[str] = None,
                           intent: Optional[str] = None,
                           entities: Optional[Dict] = None,
                           analysis_plan: Optional[Dict] = None):
        """
        Log a generated SQL query (from SQLAgent or similar).
        
        Args:
            query: The generated SQL query
            source: Source agent (e.g., "SQLAgent")
            question: Original question that generated the query
            intent: Query intent (FORECAST, ANALYTICS)
            entities: Extracted entities
            analysis_plan: Analysis plan (if any)
        """
        context = {}
        if question:
            context['question'] = question
        if intent:
            context['intent'] = intent
        if entities:
            context['entities'] = entities
        if analysis_plan:
            context['analysis_plan'] = analysis_plan
        
        self.log_query(
            query=query,
            query_type="GENERATED",
            source=source,
            context=context if context else None
        )
    
    def log_executed_query(self,
                          query: str,
                          source: str,
                          params: Optional[Dict] = None,
                          context: Optional[Dict[str, Any]] = None):
        """
        Log an executed SQL query (from DatabaseManager).
        
        Args:
            query: The executed SQL query
            source: Source agent/component (e.g., "DatabaseManager", "InventoryAgent")
            params: Query parameters
            context: Additional context
        """
        self.log_query(
            query=query,
            query_type="EXECUTED",
            source=source,
            params=params,
            context=context
        )
    
    def log_direct_query(self,
                        query: str,
                        source: str,
                        description: Optional[str] = None,
                        params: Optional[Dict] = None):
        """
        Log a directly constructed SQL query (not generated by LLM).
        
        Args:
            query: The SQL query
            source: Source component (e.g., "InventoryAgent._build_timeseries_cache")
            description: Description of what the query does
            params: Query parameters
        """
        context = {}
        if description:
            context['description'] = description
        
        self.log_query(
            query=query,
            query_type="DIRECT",
            source=source,
            params=params,
            context=context if context else None
        )


# Global instance
_sql_logger_instance: Optional[SQLQueryLogger] = None


def get_sql_logger(log_dir: str = "sql_logs") -> SQLQueryLogger:
    """
    Get or create the global SQL logger instance.
    
    Args:
        log_dir: Directory for log files
        
    Returns:
        SQLQueryLogger instance
    """
    global _sql_logger_instance
    if _sql_logger_instance is None:
        _sql_logger_instance = SQLQueryLogger(log_dir)
    return _sql_logger_instance



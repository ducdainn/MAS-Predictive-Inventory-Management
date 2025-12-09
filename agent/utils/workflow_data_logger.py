"""
WorkflowDataLogger: Centralized logger for saving all workflow data to files.

This logger captures data at each step of the agent workflow and saves it
to the 'workflowData' folder for analysis and debugging.
"""

import os
import json
from datetime import datetime
from typing import Any, Dict, Optional, List
import pandas as pd
import uuid


def _env_flag_enabled() -> bool:
    """Check env flag to optionally re-enable workflow logging."""
    value = os.environ.get("WORKFLOW_DATA_LOGGER_ENABLED", "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


class WorkflowDataLogger:
    """Logs all workflow data to files in workflowData folder."""
    
    def __init__(self, base_dir: str = "workflowData", enabled: Optional[bool] = None):
        self.base_dir = base_dir
        self.enabled = _env_flag_enabled() if enabled is None else enabled
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        self.session_dir = ""
        
        if self.enabled:
            os.makedirs(base_dir, exist_ok=True)
            self.session_dir = os.path.join(base_dir, self.session_id)
            os.makedirs(self.session_dir, exist_ok=True)
        
        # Track step counter for this session
        self.step_counter = 0
    
    def _get_step_filename(self, step_name: str, extension: str = "json") -> str:
        """Generate filename for a workflow step."""
        self.step_counter += 1
        return f"step_{self.step_counter:03d}_{step_name}.{extension}"
    
    def log_step(self, 
                 step_name: str,
                 agent_name: str,
                 data: Any,
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Log a workflow step with data.
        
        Args:
            step_name: Name of the step (e.g., "intent_classification", "sql_generation")
            agent_name: Name of the agent (e.g., "OrchestratorAgent", "ForecastAgent")
            data: Data to save (can be dict, DataFrame, str, etc.)
            metadata: Optional metadata about the step
            
        Returns:
            Path to the saved file
        """
        filename = self._get_step_filename(step_name, extension="json")
        
        if not self.enabled:
            return ""
        
        filepath = os.path.join(self.session_dir, filename)
        
        # Prepare log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "step_number": self.step_counter,
            "step_name": step_name,
            "agent_name": agent_name,
            "metadata": metadata or {},
        }
        
        # Handle different data types
        if isinstance(data, pd.DataFrame):
            # Save DataFrame separately as CSV, reference in JSON
            csv_filename = filename.replace(".json", ".csv")
            csv_filepath = os.path.join(self.session_dir, csv_filename)
            try:
                data.to_csv(csv_filepath, index=True, encoding='utf-8-sig')
                log_entry["data_type"] = "DataFrame"
                log_entry["data_file"] = csv_filename
                log_entry["data_shape"] = list(data.shape)
                log_entry["data_columns"] = list(data.columns)
                log_entry["data_preview"] = data.head(5).to_dict('records') if len(data) > 0 else []
            except Exception as e:
                log_entry["data_type"] = "DataFrame"
                log_entry["data_error"] = str(e)
                log_entry["data_shape"] = list(data.shape) if hasattr(data, 'shape') else None
        elif isinstance(data, dict):
            log_entry["data_type"] = "dict"
            log_entry["data"] = self._serialize_dict(data)
        elif isinstance(data, str):
            log_entry["data_type"] = "string"
            log_entry["data"] = data
        elif isinstance(data, (list, tuple)):
            log_entry["data_type"] = "list"
            log_entry["data"] = self._serialize_list(data)
        else:
            log_entry["data_type"] = "other"
            log_entry["data"] = str(data)
        
        # Save JSON log entry
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2, ensure_ascii=False, default=str)
            return filepath
        except Exception as e:
            print(f"⚠️  Failed to save workflow step {step_name}: {e}")
            return ""
    
    def log_dataframe(self,
                     step_name: str,
                     agent_name: str,
                     df: pd.DataFrame,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Log a DataFrame as CSV with metadata JSON.
        
        Args:
            step_name: Name of the step
            agent_name: Name of the agent
            df: DataFrame to save
            metadata: Optional metadata
            
        Returns:
            Path to the saved CSV file
        """
        filename = self._get_step_filename(step_name, extension="csv")
        
        if not self.enabled:
            return ""
        
        filepath = os.path.join(self.session_dir, filename)
        
        try:
            df.to_csv(filepath, index=True, encoding='utf-8-sig')
            
            # Also save metadata JSON
            json_filename = filename.replace(".csv", "_metadata.json")
            json_filepath = os.path.join(self.session_dir, json_filename)
            metadata_entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "step_name": step_name,
                "agent_name": agent_name,
                "data_type": "DataFrame",
                "shape": list(df.shape),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "metadata": metadata or {},
            }
            
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata_entry, f, indent=2, ensure_ascii=False, default=str)
            
            return filepath
        except Exception as e:
            print(f"⚠️  Failed to save DataFrame for step {step_name}: {e}")
            return ""
    
    def log_sql_query(self,
                     step_name: str,
                     agent_name: str,
                     sql: str,
                     question: Optional[str] = None,
                     intent: Optional[str] = None,
                     entities: Optional[Dict] = None,
                     result_df: Optional[pd.DataFrame] = None) -> str:
        """
        Log SQL query and optionally its result.
        
        Args:
            step_name: Name of the step
            agent_name: Name of the agent
            sql: SQL query string
            question: Original question
            intent: Intent (FORECAST, ANALYTICS, etc.)
            entities: Extracted entities
            result_df: Optional result DataFrame
            
        Returns:
            Path to the saved file
        """
        filename = self._get_step_filename(step_name, extension="json")
        
        if not self.enabled:
            return ""
        
        filepath = os.path.join(self.session_dir, filename)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "step_name": step_name,
            "agent_name": agent_name,
            "sql_query": sql,
            "question": question,
            "intent": intent,
            "entities": entities or {},
        }
        
        if result_df is not None:
            # Save result DataFrame separately
            csv_filename = filename.replace(".json", "_result.csv")
            csv_filepath = os.path.join(self.session_dir, csv_filename)
            try:
                result_df.to_csv(csv_filepath, index=True, encoding='utf-8-sig')
                log_entry["result_file"] = csv_filename
                log_entry["result_shape"] = list(result_df.shape)
                log_entry["result_columns"] = list(result_df.columns)
            except Exception as e:
                log_entry["result_error"] = str(e)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2, ensure_ascii=False, default=str)
            return filepath
        except Exception as e:
            print(f"⚠️  Failed to save SQL query for step {step_name}: {e}")
            return ""
    
    def _serialize_dict(self, d: Dict) -> Dict:
        """Recursively serialize dict, handling non-serializable types."""
        result = {}
        for k, v in d.items():
            if isinstance(v, pd.DataFrame):
                # For nested DataFrames, just store metadata
                result[k] = {
                    "_type": "DataFrame",
                    "shape": list(v.shape),
                    "columns": list(v.columns),
                }
            elif isinstance(v, dict):
                result[k] = self._serialize_dict(v)
            elif isinstance(v, (list, tuple)):
                result[k] = self._serialize_list(v)
            else:
                try:
                    json.dumps(v)  # Test if serializable
                    result[k] = v
                except (TypeError, ValueError):
                    result[k] = str(v)
        return result
    
    def _serialize_list(self, l: List) -> List:
        """Recursively serialize list, handling non-serializable types."""
        result = []
        for item in l:
            if isinstance(item, pd.DataFrame):
                result.append({
                    "_type": "DataFrame",
                    "shape": list(item.shape),
                    "columns": list(item.columns),
                })
            elif isinstance(item, dict):
                result.append(self._serialize_dict(item))
            elif isinstance(item, (list, tuple)):
                result.append(self._serialize_list(item))
            else:
                try:
                    json.dumps(item)
                    result.append(item)
                except (TypeError, ValueError):
                    result.append(str(item))
        return result
    
    def get_session_dir(self) -> str:
        """Get the current session directory."""
        return self.session_dir if self.enabled else ""
    
    def get_session_id(self) -> str:
        """Get the current session ID."""
        return self.session_id


# Singleton instance
_workflow_logger: Optional[WorkflowDataLogger] = None


def get_workflow_logger(base_dir: str = "workflowData", enabled: Optional[bool] = None) -> WorkflowDataLogger:
    """Get or create the global workflow logger instance."""
    global _workflow_logger
    if _workflow_logger is None:
        _workflow_logger = WorkflowDataLogger(base_dir=base_dir, enabled=enabled)
    return _workflow_logger


def reset_workflow_logger():
    """Reset the global workflow logger (create new session)."""
    global _workflow_logger
    _workflow_logger = None



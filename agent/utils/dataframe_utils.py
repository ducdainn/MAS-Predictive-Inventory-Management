"""
Utility helpers for formatting pandas DataFrames used across agents.
"""

import pandas as pd

from agent.label_formatter import QuickLabelFormatter


def format_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Format DataFrame column names to beautiful Vietnamese labels."""
    if df is None or df.empty:
        return df

    formatter = QuickLabelFormatter()

    new_columns = {}
    for col in df.columns:
        new_columns[col] = formatter.format_label(str(col))

    return df.rename(columns=new_columns)




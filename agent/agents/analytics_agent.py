"""
AnalyticsAgent: executes SQL queries and produces charts/analysis.
"""

import os
import uuid
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from agent.label_formatter import format_axis_label
from agent.manager.database_manager import DatabaseManager
from agent.utils.dataframe_utils import format_dataframe_columns


class AnalyticsAgent:
    """Creates visualizations and analytics."""

    def __init__(self, db_manager: DatabaseManager, output_dir: str = "charts"):
        self.db = db_manager
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def analyze(self, sql: str, question: str) -> Dict[str, Any]:
        """Execute query and create visualizations."""
        print(f"📊 Executing analytics query...")
        print(f"   SQL: {sql[:200]}...")

        try:
            df = self.db.execute_query(sql)
        except Exception as e:
            print(f"❌ Query execution failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False, 
                "message": f"Query execution failed: {str(e)}", 
                "data": pd.DataFrame(),
                "error": str(e)
            }

        if df.empty:
            print(f"⚠️  Query returned empty result")
            return {
                "success": False, 
                "message": "No data returned from query. Please check your filters (date range, region, branch, etc.)", 
                "data": df
            }

        print(f"✅ Retrieved {len(df)} rows with {len(df.columns)} columns")

        try:
            charts = self._create_charts(df, question)
            summary = self._generate_summary(df)
            df_display = format_dataframe_columns(df)
        except Exception as e:
            print(f"⚠️  Chart/summary generation failed: {e}")
            import traceback
            traceback.print_exc()
            # Still return data even if chart fails
            df_display = format_dataframe_columns(df)
            summary = f"Retrieved {len(df)} rows. Chart generation failed: {str(e)}"
            charts = []

        return {
            "success": True,
            "data": df_display,
            "summary": summary,
            "charts": charts,
            "row_count": len(df)
        }

    def _create_charts(self, df: pd.DataFrame, question: str) -> List[str]:
        """Create appropriate charts based on data."""
        charts = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        for col in categorical_cols[:]:
            if 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols.append(col)
                    categorical_cols.remove(col)
                except Exception:
                    pass

        if date_cols and numeric_cols:
            chart_path = self._plot_time_series(df, date_cols[0], numeric_cols[0])
            charts.append(chart_path)
        elif categorical_cols and numeric_cols and len(df) <= 50:
            chart_path = self._plot_bar_chart(df, categorical_cols[0], numeric_cols[0])
            charts.append(chart_path)
        elif len(numeric_cols) >= 1:
            chart_path = self._plot_distribution(df, numeric_cols[0])
            charts.append(chart_path)

        return charts

    def _plot_time_series(self, df: pd.DataFrame, date_col: str, value_col: str) -> str:
        """Plot time series with beautiful labels."""
        from agent.label_formatter import QuickLabelFormatter

        formatter = QuickLabelFormatter()
        labels = formatter.format_chart_labels(date_col, value_col)

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x=date_col, y=value_col, linewidth=2.5, color='steelblue')

        plt.xlabel(labels['x'], fontsize=12, fontweight='bold')
        plt.ylabel(labels['y'], fontsize=12, fontweight='bold')
        plt.title(labels['title'], fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f"timeseries_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        return filepath

    def _plot_bar_chart(self, df: pd.DataFrame, cat_col: str, value_col: str) -> str:
        """Create bar chart with Vietnamese labels."""
        df_plot = df.nlargest(20, value_col) if len(df) > 20 else df

        cat_label = format_axis_label(cat_col)
        value_label = format_axis_label(value_col)

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(df_plot)), df_plot[value_col], color='steelblue')
        plt.xticks(range(len(df_plot)), df_plot[cat_col], rotation=45, ha='right')
        plt.xlabel(cat_label, fontsize=12, fontweight='bold')
        plt.ylabel(value_label, fontsize=12, fontweight='bold')
        plt.title(f'{value_label} theo {cat_label}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        filename = f"bar_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📊 Created bar chart: {filepath}")
        return filepath

    def _plot_distribution(self, df: pd.DataFrame, col: str) -> str:
        """Create distribution plot with Vietnamese labels."""
        col_label = format_axis_label(col)

        plt.figure(figsize=(10, 6))
        plt.hist(df[col].dropna(), bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel(col_label, fontsize=12, fontweight='bold')
        plt.ylabel('Tần Suất', fontsize=12, fontweight='bold')
        plt.title(f'Phân Phối của {col_label}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        filename = f"dist_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📉 Created distribution chart: {filepath}")
        return filepath

    def _generate_summary(self, df: pd.DataFrame) -> str:
        """Generate text summary of results."""
        df_display = format_dataframe_columns(df)

        summary = f"Retrieved {len(df_display)} rows with {len(df_display.columns)} columns.\n\n"

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary += "Numeric summary:\n"
            for col in numeric_cols[:3]:
                formatted_col = format_dataframe_columns(df[[col]]).columns[0]
                summary += (
                    f"  - {formatted_col}: min={df[col].min():.2f}, "
                    f"max={df[col].max():.2f}, mean={df[col].mean():.2f}\n"
                )

        summary += f"\nFirst 5 rows:\n{df_display.head().to_string()}\n"
        return summary





"""
AnalyticsAgent: executes SQL queries and produces charts/analysis.
"""

import os
import uuid
from typing import Any, Dict, List, Optional

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

    def analyze(self, sql: str, question: str, analysis_plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            charts = self._create_charts(df, question, analysis_plan)
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

        result_payload = {
            "success": True,
            "data": df_display,
            "summary": summary,
            "charts": charts,
            "row_count": len(df)
        }
        if analysis_plan:
            result_payload["analysis_plan"] = analysis_plan
        return result_payload

    def _create_charts(self,
                       df: pd.DataFrame,
                       question: str,
                       analysis_plan: Optional[Dict[str, Any]] = None) -> List[str]:
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

        metric_col = self._pick_metric_column(numeric_cols, analysis_plan)
        category_col = self._pick_dimension_column(categorical_cols, date_cols, analysis_plan)

        preferred_chart = (analysis_plan or {}).get("chart_type", "").lower() if analysis_plan else ""

        if preferred_chart in {"line", "area"} and date_cols and metric_col:
            chart_path = self._plot_time_series(df, date_cols[0], metric_col)
            charts.append(chart_path)
            return charts

        if preferred_chart in {"bar", "column"} and category_col and metric_col:
            chart_path = self._plot_bar_chart(df, category_col, metric_col)
            charts.append(chart_path)
            return charts

        if preferred_chart == "barh" and category_col and metric_col:
            chart_path = self._plot_horizontal_bar_chart(df, category_col, metric_col)
            charts.append(chart_path)
            return charts

        if preferred_chart == "kpi_card" and metric_col:
            chart_path = self._render_kpi_card(df, metric_col, analysis_plan)
            charts.append(chart_path)
            return charts

        if date_cols and metric_col:
            chart_path = self._plot_time_series(df, date_cols[0], metric_col)
            charts.append(chart_path)
        elif category_col and metric_col and len(df) <= 50:
            # Wide labels -> horizontal bar improves readability
            if df[category_col].astype(str).str.len().max() > 25:
                chart_path = self._plot_horizontal_bar_chart(df, category_col, metric_col)
            else:
                chart_path = self._plot_bar_chart(df, category_col, metric_col)
            charts.append(chart_path)
        elif metric_col:
            chart_path = self._plot_distribution(df, metric_col)
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

    def _plot_horizontal_bar_chart(self, df: pd.DataFrame, cat_col: str, value_col: str) -> str:
        """Create horizontal bar chart for long labels."""
        df_plot = df.nlargest(20, value_col) if len(df) > 20 else df

        cat_label = format_axis_label(cat_col)
        value_label = format_axis_label(value_col)

        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_plot, y=cat_col, x=value_col, palette='Blues_r')
        plt.ylabel(cat_label, fontsize=12, fontweight='bold')
        plt.xlabel(value_label, fontsize=12, fontweight='bold')
        plt.title(f'{value_label} theo {cat_label}', fontsize=14, fontweight='bold')
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()

        filename = f"barh_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📊 Created horizontal bar chart: {filepath}")
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

    def _render_kpi_card(self,
                         df: pd.DataFrame,
                         value_col: str,
                         analysis_plan: Optional[Dict[str, Any]]) -> str:
        """Render KPI card by aggregating numeric column and showing highlight text."""
        kpi_config = (analysis_plan or {}).get("kpi_config", {}) if analysis_plan else {}
        agg_method = kpi_config.get("aggregation", "sum").lower()

        if agg_method == "mean":
            value = df[value_col].mean()
        elif agg_method == "max":
            value = df[value_col].max()
        elif agg_method == "min":
            value = df[value_col].min()
        else:
            value = df[value_col].sum()

        goal = kpi_config.get("target_value")
        label = kpi_config.get("label") or format_axis_label(value_col)

        plt.figure(figsize=(5, 3))
        plt.axis("off")
        plt.text(0.5, 0.65, label, ha="center", va="center", fontsize=16, fontweight="semibold", color="#4b5563")
        plt.text(0.5, 0.35, f"{value:,.0f}", ha="center", va="center", fontsize=34, fontweight="bold", color="#111827")
        if goal is not None:
            delta = value - goal
            delta_color = "#16a34a" if delta >= 0 else "#dc2626"
            plt.text(0.5, 0.1, f"Mục tiêu: {goal:,.0f} | Δ {delta:,.0f}", ha="center", va="center",
                     fontsize=11, color=delta_color)

        filename = f"kpi_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"🧮 Created KPI card: {filepath}")
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

    @staticmethod
    def _pick_metric_column(numeric_cols: List[str], analysis_plan: Optional[Dict[str, Any]]) -> Optional[str]:
        if not numeric_cols:
            return None
        if analysis_plan and analysis_plan.get("metrics"):
            for metric in analysis_plan["metrics"]:
                for col in numeric_cols:
                    if metric.lower() in col.lower():
                        return col
        return numeric_cols[0]

    @staticmethod
    def _pick_dimension_column(categorical_cols: List[str],
                               date_cols: List[str],
                               analysis_plan: Optional[Dict[str, Any]]) -> Optional[str]:
        columns = categorical_cols.copy()
        if analysis_plan and analysis_plan.get("dimensions"):
            for dimension in analysis_plan["dimensions"]:
                for col in date_cols + columns:
                    if dimension.lower() in col.lower():
                        return col
        if columns:
            return columns[0]
        if date_cols:
            return date_cols[0]
        return None





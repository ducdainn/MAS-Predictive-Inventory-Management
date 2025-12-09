"""
Data availability checker: compares inventory coverage vs. sales history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from agent.system_date import get_system_date
    SYSTEM_DATE_AVAILABLE = True
except ImportError:  # pragma: no cover
    SYSTEM_DATE_AVAILABLE = False


@dataclass
class DataCoverageSample:
    key: Tuple[str, int]
    detail: Dict[str, int]


class DataAvailabilityChecker:
    """Encapsulate logic for analyzing inventory vs sales coverage."""

    def __init__(self,
                 min_history_days: int = 60,
                 stale_days: int = 7):
        self.min_history_days = min_history_days
        self.stale_days = stale_days

    def _get_now(self) -> pd.Timestamp:
        if SYSTEM_DATE_AVAILABLE:
            return pd.to_datetime(get_system_date())
        return pd.Timestamp.now()

    def generate_report(self,
                        inventory_df: Optional[pd.DataFrame],
                        timeseries_cache: Dict[tuple, pd.DataFrame]) -> Dict[str, any]:
        """Produce coverage statistics comparing inventory vs historical sales."""
        if inventory_df is None or inventory_df.empty:
            return {"total_items": 0}

        unique_items = inventory_df[['product_code', 'branch_code']].drop_duplicates()
        total_items = len(unique_items)

        missing_history: List[Tuple[str, int]] = []
        short_history: List[DataCoverageSample] = []
        stale_history: List[DataCoverageSample] = []

        now = self._get_now()

        for _, row in unique_items.iterrows():
            key = (row['product_code'], row['branch_code'])
            cache_df = timeseries_cache.get(key)

            if cache_df is None or cache_df.empty:
                missing_history.append(key)
                continue

            history_df = cache_df.copy()
            history_df['date'] = pd.to_datetime(history_df['date'])
            first_sale = history_df['date'].min()
            last_sale = history_df['date'].max()
            history_span_days = int((last_sale - first_sale).days)
            days_since_last = int((now - last_sale).days)

            if history_span_days < self.min_history_days:
                short_history.append(
                    DataCoverageSample(key, {"history_span_days": max(history_span_days, 0)})
                )
            if days_since_last > self.stale_days:
                stale_history.append(
                    DataCoverageSample(key, {"days_since_last": days_since_last})
                )

        report = {
            "total_items": total_items,
            "items_with_history": total_items - len(missing_history),
            "missing_history_count": len(missing_history),
            "short_history_count": len(short_history),
            "stale_history_count": len(stale_history),
            "missing_history_samples": missing_history[:10],
            "short_history_samples": [
                {"key": sample.key, **sample.detail} for sample in short_history[:10]
            ],
            "stale_history_samples": [
                {"key": sample.key, **sample.detail} for sample in stale_history[:10]
            ],
        }
        return report

    def log_report(self, report: Dict[str, any]) -> None:
        if not report or report.get("total_items", 0) == 0:
            print("   ⚠️  Data availability report unavailable (no inventory rows).")
            return

        total = report["total_items"]
        def pct(count: int) -> str:
            return f"{(count / total):.1%}" if total else "0.0%"

        print("\n📊 DATA QUALITY SUMMARY")
        print("📊 DATA COVERAGE")
        print(f"   • Total product-branch combos: {total}")
        print(f"   • With history: {report.get('items_with_history', 0)}")
        print(f"   • Missing history: {report['missing_history_count']} ({pct(report['missing_history_count'])})")
        print(f"   • Short history (<{self.min_history_days} days): {report['short_history_count']} ({pct(report['short_history_count'])})")
        print(f"   • Stale history (>{self.stale_days} days since last sale): {report['stale_history_count']} ({pct(report['stale_history_count'])})\n")



"""
PanelXGBoostModelLoader
=======================

Loader cho PANEL XGBoost model (Global Model with Identifiers).

- Model được train bằng script: train_xgboost_panel.py
- Lưu ở: models_panel/xgboost_panel_*.pkl
- Metadata: models_panel/metadata_panel_*.json
  - Chứa: feature list, encoders cho branch_code, region, f_sku

API chính:
- predict(df_ts, branch_code, region, f_sku, horizon=30) -> forecast_df

Trong đó:
- df_ts: DataFrame với index = DatetimeIndex, cột 'value' = quantity theo ngày
- branch_code: int
- region: str (ví dụ 'MIỀN TRUNG')
- f_sku: str (ví dụ 'L1.5050.A5322.7')
"""

import os
import json
import pickle
import time
from datetime import datetime
from typing import Dict, Optional, List, Any

import numpy as np
import pandas as pd


class PanelXGBoostModelLoader:
    """Loader cho PANEL XGBoost model với ID features (1-step recursive)."""

    def __init__(self, models_dir: str = "models_panel"):
        self.models_dir = models_dir
        self.model = None
        self.metadata: Dict = {}
        self.feature_names = []
        self.encoders: Dict[str, Dict[str, int]] = {}
        self.loaded = False
        self.model_version = "unknown"

        os.makedirs(self.models_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # LOAD MODEL + METADATA
    # ------------------------------------------------------------------
    def load_latest_model(self) -> bool:
        """Load model panel mới nhất (xgboost_panel_*.pkl + metadata_panel_*.json)."""
        load_start = time.perf_counter()
        try:
            files = [
                f
                for f in os.listdir(self.models_dir)
                if f.startswith("xgboost_panel_") and f.endswith(".pkl")
            ]
            if not files:
                return False

            # Lấy file mới nhất theo mtime
            files.sort(
                key=lambda f: os.path.getmtime(os.path.join(self.models_dir, f)),
                reverse=True,
            )
            model_file = files[0]
            timestamp = (
                model_file.replace("xgboost_panel_", "").replace(".pkl", "")
            )

            model_path = os.path.join(self.models_dir, model_file)
            metadata_path = os.path.join(
                self.models_dir, f"metadata_panel_{timestamp}.json"
            )

            model_load_start = time.perf_counter()
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            model_load_elapsed = time.perf_counter() - model_load_start

            metadata_load_start = time.perf_counter()
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            metadata_load_elapsed = time.perf_counter() - metadata_load_start

            self.feature_names = self.metadata.get("features", [])
            self.encoders = self.metadata.get("encoders", {})
            self.model_version = model_file
            self.loaded = True
            
            total_elapsed = time.perf_counter() - load_start
            print(f"   ⏱️  Step 1 (Load Model): completed in {total_elapsed:.3f}s (model: {model_load_elapsed:.3f}s, metadata: {metadata_load_elapsed:.3f}s)")
            return True
        except Exception:
            self.loaded = False
            return False

    # ------------------------------------------------------------------
    # ENCODING HELPERS
    # ------------------------------------------------------------------
    def _encode_id(self, enc_name: str, raw_value: str) -> int:
        """
        Encode 1 ID (branch_code, region, f_sku) theo mapping trong metadata.
        Nếu không tìm thấy, trả về 0 (mặc định).
        """
        enc_map = self.encoders.get(enc_name, {})
        key = str(raw_value)
        if key in enc_map:
            return int(enc_map[key])
        # fallback: 0
        return 0

    # ------------------------------------------------------------------
    # FEATURE ENGINEERING (SINGLE SERIES)
    # ------------------------------------------------------------------
    def _trend(self, series: np.ndarray) -> float:
        if len(series) < 2:
            return 0.0
        x = np.arange(len(series))
        return float(np.polyfit(x, series, 1)[0])

    def _create_features_for_series(
        self,
        df_ts: pd.DataFrame,
        branch_code: int,
        region: str,
        f_sku: str,
    ) -> pd.DataFrame:
        """
        Tạo full feature vector cho 1 chuỗi (branch_code, f_sku).
        df_ts: index = date, cột 'value' = quantity.
        """
        df = df_ts.copy()
        df = df.sort_index()
        df["quantity"] = df["value"].astype(float)
        df["date"] = df.index

        # ID features (static) – encode theo metadata
        df["branch_code"] = int(branch_code)
        df["region"] = str(region).upper().strip()
        df["f_sku"] = str(f_sku).strip()

        df["branch_le"] = self._encode_id("branch_le", df["branch_code"].iloc[0])
        df["region_le"] = self._encode_id("region_le", df["region"].iloc[0])
        df["f_sku_le"] = self._encode_id("f_sku_le", df["f_sku"].iloc[0])

        # Scale features
        df["avg_sales_all_time"] = df["quantity"].mean()
        df["avg_sales_30d"] = (
            df["quantity"]
            .rolling(window=30, min_periods=1)
            .mean()
            .fillna(0)
        )

        qty = df["quantity"]

        # Lag features
        for lag in [1, 7, 14, 30]:
            df[f"lag_{lag}"] = qty.shift(lag)

        # Rolling stats
        for window in [7, 14, 30]:
            df[f"rolling_mean_{window}"] = (
                qty.rolling(window=window, min_periods=1).mean()
            )
            df[f"rolling_std_{window}"] = (
                qty.rolling(window=window, min_periods=1).std().fillna(0)
            )
            df[f"rolling_min_{window}"] = (
                qty.rolling(window=window, min_periods=1).min()
            )
            df[f"rolling_max_{window}"] = (
                qty.rolling(window=window, min_periods=1).max()
            )

        # Changes & pct changes
        df["change_1"] = qty.diff(1)
        df["change_7"] = qty.diff(7)
        pct_change_1 = qty.pct_change(1)
        pct_change_7 = qty.pct_change(7)
        df["pct_change_1"] = pct_change_1.replace([np.inf, -np.inf], 0).fillna(0)
        df["pct_change_7"] = pct_change_7.replace([np.inf, -np.inf], 0).fillna(0)

        # Trend & volatility
        df["trend_7"] = qty.rolling(7, min_periods=2).apply(
            self._trend, raw=True
        )
        df["trend_30"] = qty.rolling(30, min_periods=2).apply(
            self._trend, raw=True
        )

        df["volatility_7"] = (
            df["rolling_std_7"] / df["rolling_mean_7"]
        ).replace([np.inf, -np.inf], 0).fillna(0)
        df["volatility_30"] = (
            df["rolling_std_30"] / df["rolling_mean_30"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        # Date-based features
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_year"] = df["date"].dt.dayofyear
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["quarter"] = df["date"].dt.quarter
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
        df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

        # Drop NaN from early lags/rolling
        df = df.dropna()

        return df

    # ------------------------------------------------------------------
    # PREDICT (RECURSIVE) - SINGLE SERIES
    # ------------------------------------------------------------------
    def predict(
        self,
        df_ts: pd.DataFrame,
        branch_code: int,
        region: str,
        f_sku: str,
        horizon: int = 30,
    ) -> pd.DataFrame:
        """
        Dự báo cho 1 chuỗi (branch_code, f_sku) trong horizon ngày.

        df_ts: index = DatetimeIndex, cột 'value'.
        """
        if not self.loaded:
            raise RuntimeError("Panel model not loaded. Call load_latest_model() first.")

        if df_ts is None or df_ts.empty:
            raise ValueError("Empty time series for panel model.")

        df_ts = df_ts.copy()
        df_ts = df_ts.sort_index()

        last_date = df_ts.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

        temp_df = df_ts.copy()
        forecasts = []

        # Long-term mean for mild damping
        long_term_mean = df_ts["value"].mean() if len(df_ts) > 0 else 0.0
        damping_weight = 0.8

        for future_date in forecast_dates:
            # Add last known (or last forecast) value để tạo lag
            last_val = temp_df["value"].iloc[-1] if len(temp_df) > 0 else 0.0
            temp_df.loc[future_date] = last_val

            # FE cho toàn bộ chuỗi mở rộng, lấy hàng cuối
            fe_df = self._create_features_for_series(
                temp_df, branch_code=branch_code, region=region, f_sku=f_sku
            )
            if fe_df.empty:
                raw_pred = float(last_val)
            else:
                X_last = fe_df[self.feature_names].iloc[-1:].values
                # Đảm bảo không NaN/inf
                X_last = np.nan_to_num(X_last, nan=0.0, posinf=0.0, neginf=0.0)
                raw_pred = float(self.model.predict(X_last)[0])

            raw_pred = max(0.0, raw_pred)

            # Damping nhẹ để tránh bay xa
            if long_term_mean > 0:
                damped = (
                    raw_pred * damping_weight
                    + long_term_mean * (1.0 - damping_weight)
                )
            else:
                damped = raw_pred

            damped = max(0.0, damped)
            temp_df.loc[future_date, "value"] = damped
            forecasts.append(damped)

        forecast_df = pd.DataFrame(
            {"forecast": forecasts},
            index=forecast_dates,
        )
        return forecast_df

    # ------------------------------------------------------------------
    # PREDICT BATCH (RECURSIVE, MULTI-SERIES)
    # ------------------------------------------------------------------
    def _create_features_batch_vectorized_recursive(
        self,
        series_list_with_temp: List[Dict[str, Any]],
    ) -> Dict[Any, np.ndarray]:
        """
        VECTORIZED: Tạo features cho recursive prediction.
        Tương tự multistep nhưng cho recursive model.
        """
        if not series_list_with_temp:
            return {}
        
        # Concat tất cả temp_dfs
        all_dfs = []
        for item in series_list_with_temp:
            key = item["key"]
            temp_df = item["temp_df"]
            branch_code = item["branch_code"]
            region = item["region"]
            f_sku = item["f_sku"]
            
            if temp_df is None or temp_df.empty:
                continue
            
            df = temp_df.copy()
            df["_series_key"] = key
            df["_branch_code"] = int(branch_code)
            df["_region"] = str(region).upper().strip()
            df["_f_sku"] = str(f_sku).strip()
            df["quantity"] = df["value"].astype(float)
            df["date"] = df.index
            
            all_dfs.append(df)
        
        if not all_dfs:
            return {}
        
        # Concat và tạo features (tương tự multistep)
        combined = pd.concat(all_dfs, axis=0, ignore_index=False)
        combined = combined.sort_values(["_series_key", "date"])
        
        # Encode IDs
        combined["branch_le"] = combined["_branch_code"].apply(lambda x: self._encode_id("branch_le", x))
        combined["region_le"] = combined["_region"].apply(lambda x: self._encode_id("region_le", x))
        combined["f_sku_le"] = combined["_f_sku"].apply(lambda x: self._encode_id("f_sku_le", x))
        
        grouped = combined.groupby("_series_key", group_keys=False)
        
        # Scale features
        combined["avg_sales_all_time"] = grouped["quantity"].transform("mean")
        combined["avg_sales_30d"] = grouped["quantity"].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        ).fillna(0)
        
        # Lags
        for lag in [1, 7, 14, 30]:
            combined[f"lag_{lag}"] = grouped["quantity"].shift(lag)
        
        # Rolling stats
        for window in [7, 14, 30]:
            combined[f"rolling_mean_{window}"] = grouped["quantity"].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            combined[f"rolling_std_{window}"] = grouped["quantity"].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            ).fillna(0)
            combined[f"rolling_min_{window}"] = grouped["quantity"].transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
            combined[f"rolling_max_{window}"] = grouped["quantity"].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
        
        # Changes
        combined["change_1"] = grouped["quantity"].diff(1)
        combined["change_7"] = grouped["quantity"].diff(7)
        pct_1 = grouped["quantity"].pct_change(1)
        pct_7 = grouped["quantity"].pct_change(7)
        combined["pct_change_1"] = pct_1.replace([np.inf, -np.inf], 0).fillna(0)
        combined["pct_change_7"] = pct_7.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Trends
        combined["trend_7"] = grouped["quantity"].transform(
            lambda x: x.rolling(7, min_periods=2).apply(self._trend, raw=True)
        )
        combined["trend_30"] = grouped["quantity"].transform(
            lambda x: x.rolling(30, min_periods=2).apply(self._trend, raw=True)
        )
        
        # Volatility
        combined["volatility_7"] = (
            combined["rolling_std_7"] / combined["rolling_mean_7"]
        ).replace([np.inf, -np.inf], 0).fillna(0)
        combined["volatility_30"] = (
            combined["rolling_std_30"] / combined["rolling_mean_30"]
        ).replace([np.inf, -np.inf], 0).fillna(0)
        
        # Date features
        combined["year"] = combined["date"].dt.year
        combined["month"] = combined["date"].dt.month
        combined["day"] = combined["date"].dt.day
        combined["day_of_week"] = combined["date"].dt.dayofweek
        combined["day_of_year"] = combined["date"].dt.dayofyear
        combined["week_of_year"] = combined["date"].dt.isocalendar().week.astype(int)
        combined["quarter"] = combined["date"].dt.quarter
        combined["is_weekend"] = (combined["day_of_week"] >= 5).astype(int)
        combined["is_month_start"] = combined["date"].dt.is_month_start.astype(int)
        combined["is_month_end"] = combined["date"].dt.is_month_end.astype(int)
        
        # Extract last row features for each series
        results = {}
        for key, group in combined.groupby("_series_key"):
            group_clean = group.dropna()
            if not group_clean.empty:
                last_row = group_clean.iloc[-1]
                results[key] = last_row[self.feature_names].values
        
        return results

    def predict_batch(
        self,
        series_list,
        horizon: int = 30,
    ):
        """
        OPTIMIZED: Vectorized recursive prediction với batch feature engineering.
        """
        if not self.loaded:
            raise RuntimeError("Panel model not loaded. Call load_latest_model() first.")

        if not series_list:
            return {}

        # Chuẩn hóa input
        keys = []
        temp_dfs = []
        long_term_means = []

        for item in series_list:
            key = item["key"]
            df_ts = item["df_ts"]
            if df_ts is None or df_ts.empty:
                raise ValueError(f"Empty time series for panel model (key={key}).")

            df_sorted = df_ts.copy().sort_index()
            keys.append(key)
            temp_dfs.append(df_sorted)
            long_term_means.append(
                df_sorted["value"].mean() if len(df_sorted) > 0 else 0.0
            )

        n_series = len(series_list)
        forecasts_map = {key: [] for key in keys}
        dates_map = {key: [] for key in keys}
        damping_weight = 0.8

        for step in range(horizon):
            # VECTORIZED: Prepare batch for feature engineering
            batch_items = []
            for idx, item in enumerate(series_list):
                temp_df = temp_dfs[idx]
                if temp_df.empty:
                    last_val = 0.0
                    last_date = pd.Timestamp(datetime.utcnow().date())
                else:
                    last_val = float(temp_df["value"].iloc[-1])
                    last_date = temp_df.index[-1]

                next_date = last_date + pd.Timedelta(days=1)
                temp_df.loc[next_date] = last_val

                batch_items.append({
                    "key": item["key"],
                    "temp_df": temp_df,
                    "branch_code": item["branch_code"],
                    "region": item["region"],
                    "f_sku": item["f_sku"],
                })
            
            # VECTORIZED: Create features for all series at once
            features_dict = self._create_features_batch_vectorized_recursive(batch_items)
            
            # Build feature matrix
            feature_rows = []
            row_series_indices = []
            
            for idx, item in enumerate(series_list):
                key = item["key"]
                if key in features_dict:
                    feature_rows.append(features_dict[key])
                    row_series_indices.append(idx)

            # Predict for all series at once
            raw_preds = {}
            if feature_rows:
                X = np.vstack(feature_rows)
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                y_pred = self.model.predict(X)

                for local_idx, series_idx in enumerate(row_series_indices):
                    key = keys[series_idx]
                    raw_preds[key] = float(y_pred[local_idx])

            # Bước 3: áp dụng damping + cập nhật temp_dfs / forecasts
            for idx, item in enumerate(series_list):
                key = item["key"]
                temp_df = temp_dfs[idx]

                # next_date là index cuối cùng (vừa thêm ở trên)
                next_date = temp_df.index[-1]
                last_val = float(temp_df["value"].iloc[-2]) if len(temp_df) > 1 else float(
                    temp_df["value"].iloc[-1]
                )

                if key in raw_preds:
                    raw_pred = max(0.0, raw_preds[key])
                else:
                    # Không có feature (chuỗi quá ngắn, v.v.) → dùng last_val
                    raw_pred = max(0.0, last_val)

                long_term_mean = long_term_means[idx]
                if long_term_mean > 0:
                    damped = (
                        raw_pred * damping_weight
                        + long_term_mean * (1.0 - damping_weight)
                    )
                else:
                    damped = raw_pred

                damped = max(0.0, damped)
                # Ghi đè value tại next_date bằng giá trị đã damping
                temp_df.loc[next_date, "value"] = damped
                temp_dfs[idx] = temp_df

                forecasts_map[key].append(damped)
                dates_map[key].append(next_date)

        # Chuyển forecasts_map thành DataFrame
        result = {}
        for key in keys:
            if not dates_map[key]:
                # Không có forecast nào cho series này
                continue
            result[key] = pd.DataFrame(
                {"forecast": forecasts_map[key]},
                index=pd.DatetimeIndex(dates_map[key]),
            )

        return result


_panel_loader_instance: Optional[PanelXGBoostModelLoader] = None


def get_panel_model_loader(
    models_dir: str = "models_panel",
) -> PanelXGBoostModelLoader:
    """Singleton accessor cho PanelXGBoostModelLoader (recursive 1-step)."""
    global _panel_loader_instance
    if _panel_loader_instance is None:
        loader = PanelXGBoostModelLoader(models_dir=models_dir)
        loader.load_latest_model()
        _panel_loader_instance = loader
    return _panel_loader_instance


# ============================================================================
# MULTI-STEP PANEL LOADER (FAST INFERENCE)
# ============================================================================


class PanelXGBoostMultiStepLoader:
    """
    Loader cho PANEL XGBoost multi-step model.

    - Model được train bằng script: train_xgboost_panel_multistep.py
    - Đầu ra: H ngày forecast trong 1 lần predict (sử dụng feature 'horizon_offset').
    """

    def __init__(self, models_dir: str = "models_panel"):
        self.models_dir = models_dir
        self.model = None
        self.metadata: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.encoders: Dict[str, Dict[str, int]] = {}
        self.loaded = False
        self.model_version = "unknown"
        self.horizon = 30

        os.makedirs(self.models_dir, exist_ok=True)

    def load_latest_model(self) -> bool:
        """Load multi-step panel model mới nhất (xgboost_panel_multistep_*.pkl)."""
        load_start = time.perf_counter()
        try:
            files = [
                f
                for f in os.listdir(self.models_dir)
                if f.startswith("xgboost_panel_multistep_") and f.endswith(".pkl")
            ]
            if not files:
                return False

            files.sort(
                key=lambda f: os.path.getmtime(os.path.join(self.models_dir, f)),
                reverse=True,
            )
            model_file = files[0]
            timestamp = (
                model_file.replace("xgboost_panel_multistep_", "").replace(".pkl", "")
            )

            model_path = os.path.join(self.models_dir, model_file)
            metadata_path = os.path.join(
                self.models_dir, f"metadata_panel_multistep_{timestamp}.json"
            )

            model_load_start = time.perf_counter()
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            model_load_elapsed = time.perf_counter() - model_load_start

            metadata_load_start = time.perf_counter()
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            metadata_load_elapsed = time.perf_counter() - metadata_load_start

            self.feature_names = self.metadata.get("features", [])
            self.encoders = self.metadata.get("encoders", {})
            self.horizon = int(self.metadata.get("horizon", 30))
            self.model_version = model_file
            self.loaded = True
            
            total_elapsed = time.perf_counter() - load_start
            print(f"   ⏱️  Step 1 (Load Multi-step Model): completed in {total_elapsed:.3f}s (model: {model_load_elapsed:.3f}s, metadata: {metadata_load_elapsed:.3f}s)")
            return True
        except Exception:
            self.loaded = False
            return False

    def _encode_id(self, enc_name: str, raw_value: str) -> int:
        enc_map = self.encoders.get(enc_name, {})
        key = str(raw_value)
        if key in enc_map:
            return int(enc_map[key])
        return 0

    def _trend(self, series: np.ndarray) -> float:
        if len(series) < 2:
            return 0.0
        x = np.arange(len(series))
        return float(np.polyfit(x, series, 1)[0])

    def _create_features_for_series(
        self,
        df_ts: pd.DataFrame,
        branch_code: int,
        region: str,
        f_sku: str,
    ) -> pd.DataFrame:
        """
        Tạo full feature vector cho 1 chuỗi (branch_code, f_sku) – giống panel loader gốc,
        dùng để lấy state cuối cùng làm input multi-step.
        """
        df = df_ts.copy()
        df = df.sort_index()
        df["quantity"] = df["value"].astype(float)
        df["date"] = df.index

        # ID features
        df["branch_code"] = int(branch_code)
        df["region"] = str(region).upper().strip()
        df["f_sku"] = str(f_sku).strip()

        df["branch_le"] = self._encode_id("branch_le", df["branch_code"].iloc[0])
        df["region_le"] = self._encode_id("region_le", df["region"].iloc[0])
        df["f_sku_le"] = self._encode_id("f_sku_le", df["f_sku"].iloc[0])

        # Scale features
        df["avg_sales_all_time"] = df["quantity"].mean()
        df["avg_sales_30d"] = (
            df["quantity"]
            .rolling(window=30, min_periods=1)
            .mean()
            .fillna(0)
        )

        qty = df["quantity"]

        # Lag features
        for lag in [1, 7, 14, 30]:
            df[f"lag_{lag}"] = qty.shift(lag)

        # Rolling stats
        for window in [7, 14, 30]:
            df[f"rolling_mean_{window}"] = (
                qty.rolling(window=window, min_periods=1).mean()
            )
            df[f"rolling_std_{window}"] = (
                qty.rolling(window=window, min_periods=1).std().fillna(0)
            )
            df[f"rolling_min_{window}"] = (
                qty.rolling(window=window, min_periods=1).min()
            )
            df[f"rolling_max_{window}"] = (
                qty.rolling(window=window, min_periods=1).max()
            )

        # Changes & pct changes
        df["change_1"] = qty.diff(1)
        df["change_7"] = qty.diff(7)
        pct_change_1 = qty.pct_change(1)
        pct_change_7 = qty.pct_change(7)
        df["pct_change_1"] = pct_change_1.replace([np.inf, -np.inf], 0).fillna(0)
        df["pct_change_7"] = pct_change_7.replace([np.inf, -np.inf], 0).fillna(0)

        # Trend & volatility
        df["trend_7"] = qty.rolling(7, min_periods=2).apply(self._trend, raw=True)
        df["trend_30"] = qty.rolling(30, min_periods=2).apply(self._trend, raw=True)

        df["volatility_7"] = (
            df["rolling_std_7"] / df["rolling_mean_7"]
        ).replace([np.inf, -np.inf], 0).fillna(0)
        df["volatility_30"] = (
            df["rolling_std_30"] / df["rolling_mean_30"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        # Date-based features
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_year"] = df["date"].dt.dayofyear
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["quarter"] = df["date"].dt.quarter
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
        df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

        df = df.dropna()
        return df

    def _create_features_batch_vectorized(
        self,
        series_list: List[Dict[str, Any]],
    ) -> Dict[Any, pd.DataFrame]:
        """
        VECTORIZED: Tạo features cho nhiều series cùng lúc.
        
        Thay vì loop 205 lần, concat tất cả series → tính features 1 lần → split.
        Tăng tốc 4-5x so với loop.
        """
        if not series_list:
            return {}
        
        # Step 1: Concat tất cả series với series_key identifier
        all_dfs = []
        key_mapping = {}
        
        for item in series_list:
            key = item["key"]
            df_ts = item["df_ts"]
            branch_code = item["branch_code"]
            region = item["region"]
            f_sku = item["f_sku"]
            
            if df_ts is None or df_ts.empty:
                continue
            
            df = df_ts.copy().sort_index()
            df["_series_key"] = key
            df["_branch_code"] = int(branch_code)
            df["_region"] = str(region).upper().strip()
            df["_f_sku"] = str(f_sku).strip()
            df["quantity"] = df["value"].astype(float)
            df["date"] = df.index
            
            all_dfs.append(df)
            key_mapping[key] = (branch_code, region, f_sku)
        
        if not all_dfs:
            return {}
        
        # Step 2: Concat thành 1 DataFrame lớn
        combined = pd.concat(all_dfs, axis=0, ignore_index=False)
        combined = combined.sort_values(["_series_key", "date"])
        
        # Step 3: Encode ID features (vectorized)
        combined["branch_le"] = combined["_branch_code"].apply(lambda x: self._encode_id("branch_le", x))
        combined["region_le"] = combined["_region"].apply(lambda x: self._encode_id("region_le", x))
        combined["f_sku_le"] = combined["_f_sku"].apply(lambda x: self._encode_id("f_sku_le", x))
        
        # Step 4: Vectorized feature engineering per group
        grouped = combined.groupby("_series_key", group_keys=False)
        
        # Scale features
        combined["avg_sales_all_time"] = grouped["quantity"].transform("mean")
        combined["avg_sales_30d"] = grouped["quantity"].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        ).fillna(0)
        
        # Lag features (vectorized per group)
        for lag in [1, 7, 14, 30]:
            combined[f"lag_{lag}"] = grouped["quantity"].shift(lag)
        
        # Rolling stats (vectorized per group)
        for window in [7, 14, 30]:
            combined[f"rolling_mean_{window}"] = grouped["quantity"].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            combined[f"rolling_std_{window}"] = grouped["quantity"].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            ).fillna(0)
            combined[f"rolling_min_{window}"] = grouped["quantity"].transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
            combined[f"rolling_max_{window}"] = grouped["quantity"].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
        
        # Changes & pct changes (vectorized per group)
        combined["change_1"] = grouped["quantity"].diff(1)
        combined["change_7"] = grouped["quantity"].diff(7)
        pct_1 = grouped["quantity"].pct_change(1)
        pct_7 = grouped["quantity"].pct_change(7)
        combined["pct_change_1"] = pct_1.replace([np.inf, -np.inf], 0).fillna(0)
        combined["pct_change_7"] = pct_7.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Trend (vectorized per group)
        combined["trend_7"] = grouped["quantity"].transform(
            lambda x: x.rolling(7, min_periods=2).apply(self._trend, raw=True)
        )
        combined["trend_30"] = grouped["quantity"].transform(
            lambda x: x.rolling(30, min_periods=2).apply(self._trend, raw=True)
        )
        
        # Volatility (vectorized)
        combined["volatility_7"] = (
            combined["rolling_std_7"] / combined["rolling_mean_7"]
        ).replace([np.inf, -np.inf], 0).fillna(0)
        combined["volatility_30"] = (
            combined["rolling_std_30"] / combined["rolling_mean_30"]
        ).replace([np.inf, -np.inf], 0).fillna(0)
        
        # Date features (vectorized, no grouping needed)
        combined["year"] = combined["date"].dt.year
        combined["month"] = combined["date"].dt.month
        combined["day"] = combined["date"].dt.day
        combined["day_of_week"] = combined["date"].dt.dayofweek
        combined["day_of_year"] = combined["date"].dt.dayofyear
        combined["week_of_year"] = combined["date"].dt.isocalendar().week.astype(int)
        combined["quarter"] = combined["date"].dt.quarter
        combined["is_weekend"] = (combined["day_of_week"] >= 5).astype(int)
        combined["is_month_start"] = combined["date"].dt.is_month_start.astype(int)
        combined["is_month_end"] = combined["date"].dt.is_month_end.astype(int)
        
        # Step 5: Split back to individual series
        results = {}
        for key, group in combined.groupby("_series_key"):
            group_clean = group.dropna()
            if not group_clean.empty:
                results[key] = group_clean
        
        return results

    def predict_batch(
        self,
        series_list: List[Dict[str, Any]],
        horizon: int = 30,
    ):
        """
        Multi-step dự báo cho nhiều chuỗi (branch_code, f_sku) cùng lúc.
        
        OPTIMIZED: Sử dụng vectorized feature engineering thay vì loop.
        """
        if not self.loaded:
            raise RuntimeError(
                "Multi-step panel model not loaded. Call load_latest_model() first."
            )

        if not series_list:
            return {}

        effective_horizon = min(horizon, self.horizon)

        # VECTORIZED: Tạo features cho tất cả series cùng lúc
        feature_eng_start = time.perf_counter()
        features_dict = self._create_features_batch_vectorized(series_list)
        feature_eng_elapsed = time.perf_counter() - feature_eng_start
        
        # Extract last row features for each series
        keys: List[Any] = []
        last_dates: List[pd.Timestamp] = []
        feature_rows: List[np.ndarray] = []
        series_horizon: List[int] = []
        
        for item in series_list:
            key = item["key"]
            if key not in features_dict:
                continue
            
            fe_df = features_dict[key]
            if fe_df.empty:
                continue
            
            base_row = fe_df.iloc[-1]
            base_features = base_row[self.feature_names].values
            
            keys.append(key)
            last_dates.append(fe_df["date"].iloc[-1])
            feature_rows.append(base_features)
            series_horizon.append(effective_horizon)
        
        print(f"      ⚡ Vectorized feature engineering: {len(keys)} series in {feature_eng_elapsed:.3f}s ({len(keys)/feature_eng_elapsed:.1f} series/s)")

        if not feature_rows:
            return {}

        # Build full feature matrix cho tất cả series × horizon_offset
        X_all = []
        key_index = []
        horizon_index = []

        for i, base_vec in enumerate(feature_rows):
            for h in range(1, series_horizon[i] + 1):
                X_all.append(base_vec)
                key_index.append(keys[i])
                horizon_index.append(h)

        X_all = np.asarray(X_all, dtype=float)
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

        # Dự báo cho tất cả samples trong 1 lần
        model_pred_start = time.perf_counter()
        y_pred = self.model.predict(X_all)
        model_pred_elapsed = time.perf_counter() - model_pred_start

        # Map kết quả về từng series
        result: Dict[Any, pd.DataFrame] = {}
        for key, last_date, h, y in zip(key_index, last_dates, horizon_index, y_pred):
            if key not in result:
                result[key] = {"dates": [], "values": []}
            future_date = last_date + pd.Timedelta(days=int(h))
            result[key]["dates"].append(future_date)
            result[key]["values"].append(max(0.0, float(y)))

        # Chuyển sang DataFrame
        final_result: Dict[Any, pd.DataFrame] = {}
        for key, data in result.items():
            if not data["dates"]:
                continue
            df_fc = pd.DataFrame({"forecast": data["values"]}, index=data["dates"])
            df_fc = df_fc.sort_index()
            final_result[key] = df_fc

        return final_result


_panel_multistep_loader_instance: Optional[PanelXGBoostMultiStepLoader] = None


def get_panel_multistep_model_loader(
    models_dir: str = "models_panel",
) -> PanelXGBoostMultiStepLoader:
    """Singleton accessor cho PanelXGBoostMultiStepLoader (multi-step)."""
    global _panel_multistep_loader_instance
    if _panel_multistep_loader_instance is None:
        loader = PanelXGBoostMultiStepLoader(models_dir=models_dir)
        loader.load_latest_model()
        _panel_multistep_loader_instance = loader
    return _panel_multistep_loader_instance



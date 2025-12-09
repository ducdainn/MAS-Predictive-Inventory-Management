"""
Panel XGBoost Training (Multi-Step Horizon) for Predictive Inventory Management
===============================================================================

Mục tiêu:
- Train 1 GLOBAL panel XGBoost model cho (branch_code, region, f_sku)
- Model dự báo H ngày (mặc định 30 ngày) trong **một lần predict** bằng cách:
  - Giữ nguyên feature history (lag/rolling/trend/ID encoders, v.v.)
  - Thêm feature `horizon_offset` ∈ {1..H}
  - Target: quantity_{t + horizon_offset}

Kết quả:
- Model:   models_panel/xgboost_panel_multistep_*.pkl
- Metadata:models_panel/metadata_panel_multistep_*.json

Script này dựa trên `train_xgboost_panel.py` nhưng chuyển sang multi-step.
"""

import os
import json
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
import xgboost as xgb
import joblib


# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    "data_path": r"D:\Study\KLTN\BrickDemand\rawData\Data_FSKU_cleanedDate.csv",
    "branch_path": r"D:\Study\KLTN\BrickDemand\init\data\branch.csv",
    "model_dir": "models_panel",
    "random_seed": 42,
    # OOT split (panel-level) – dùng ngày **cơ sở** t để phân chia
    "train_end_date": "2025-05-31",
    "test_start_date": "2025-06-01",
    # Time-series features
    "lag_features": [1, 7, 14, 30],
    "rolling_windows": [7, 14, 30],
    # Horizon multi-step
    "horizon": 30,
    # XGBoost hyperparameters
    "xgb_params": {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "min_child_weight": 3,
        "gamma": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    },
}

os.makedirs(CONFIG["model_dir"], exist_ok=True)


# ============================================================================
# 1. DATA LOADING (PANEL)
# ============================================================================


def load_panel_data(
    data_path: str, branch_path: str
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    Load raw CSV và build base PANEL dataframe:
    (date, branch_code, region, f_sku, quantity).
    """
    print("\n📊 Step 1: Loading raw transactional data...")
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    print(f"   ✅ Loaded {len(df):,} rows from Data_FSKU_cleanedDate.csv")

    # Standardize columns
    col_date = "Ngày"
    col_branch = "Mã chi nhánh"
    col_region = "Khu vực"
    col_fsku = "F_SKU"
    col_qty = "Số lượng"

    df["date"] = pd.to_datetime(df[col_date])
    df["branch_code"] = df[col_branch].astype(int)
    df["region"] = df[col_region].astype(str).str.upper().str.strip()
    df["f_sku"] = df[col_fsku].astype(str).str.strip()
    df["quantity"] = df[col_qty].astype(float)

    print(
        f"   📅 Date range: {df['date'].min()} → {df['date'].max()} "
        f"({(df['date'].max() - df['date'].min()).days} days)"
    )

    # Load branch names (optional, chỉ để tham khảo)
    branch_df = pd.read_csv(branch_path, encoding="utf-8-sig")
    if "branch_code" not in branch_df.columns or "branch_name" not in branch_df.columns:
        # Thử map từ tên cột tiếng Việt
        if "Mã chi nhánh" in branch_df.columns and "Tên Chi Nhánh" in branch_df.columns:
            branch_df = branch_df.rename(
                columns={
                    "Mã chi nhánh": "branch_code",
                    "Tên Chi Nhánh": "branch_name",
                }
            )

    branch_df["branch_code"] = branch_df["branch_code"].astype(int)
    branch_name_map = dict(
        zip(branch_df["branch_code"].tolist(), branch_df["branch_name"].tolist())
    )

    print(f"   🏬 Unique branches: {df['branch_code'].nunique()}")
    print(f"   🎯 Unique F_SKU: {df['f_sku'].nunique()}")

    # PANEL aggregate: (date, branch_code, f_sku, region) → quantity
    print("\n📦 Step 2: Building PANEL aggregate (date, branch, F_SKU)...")
    panel = (
        df.groupby(["date", "branch_code", "region", "f_sku"], as_index=False)[
            "quantity"
        ]
        .sum()
        .sort_values(["branch_code", "f_sku", "date"])
        .reset_index(drop=True)
    )

    print(f"   ✅ PANEL rows: {len(panel):,}")
    print(
        f"   → {panel['branch_code'].nunique()} branches × "
        f"{panel['f_sku'].nunique()} F_SKU combinations"
    )

    return panel, branch_name_map


# ============================================================================
# 2. ENCODING IDENTIFIERS (STATIC FEATURES)
# ============================================================================


def encode_identifiers(
    panel: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Label-encode branch_code, region, f_sku và thêm:
    - branch_le, region_le, f_sku_le
    - avg_sales_all_time, avg_sales_30d
    """
    print("\n🧩 Step 3: Encoding identifiers & adding static features...")

    encoders: Dict[str, LabelEncoder] = {}
    maps: Dict[str, Dict[str, int]] = {}

    for col, name in [
        ("branch_code", "branch_le"),
        ("region", "region_le"),
        ("f_sku", "f_sku_le"),
    ]:
        le = LabelEncoder()
        panel[name] = le.fit_transform(panel[col].astype(str))
        encoders[name] = le
        maps[name] = {
            str(cls): int(code)
            for cls, code in zip(le.classes_.tolist(), le.transform(le.classes_).tolist())
        }
        print(f"   ✅ Encoded {col} → {name} (n={len(le.classes_):,})")

    # Per-series average sales (scale)
    print("   Calculating per-series scale feature avg_sales_all_time...")
    panel["series_key"] = (
        panel["branch_code"].astype(str) + "|" + panel["f_sku"].astype(str)
    )
    series_mean = panel.groupby("series_key")["quantity"].transform("mean")
    panel["avg_sales_all_time"] = series_mean

    print("   Calculating 30-day rolling mean per series (avg_sales_30d)...")
    panel = panel.sort_values(["branch_code", "f_sku", "date"])

    def add_scale_rolling(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("date")
        group["avg_sales_30d"] = (
            group["quantity"]
            .rolling(window=30, min_periods=1)
            .mean()
            .fillna(0)
        )
        return group

    panel = panel.groupby(["branch_code", "f_sku"], group_keys=False).apply(
        add_scale_rolling
    )

    return panel, maps


# ============================================================================
# 3. TIME-SERIES FEATURES PER SERIES (HISTORY STATE)
# ============================================================================


def create_panel_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo time-series features cho từng ngày t (state) của chuỗi (branch_code, f_sku).
    Đây là phần giống với script panel cũ – nhưng sẽ dùng làm "state" cho multi-step.
    """
    print("\n🔧 Step 4: Time-series feature engineering (panel)...")

    panel = panel.sort_values(["branch_code", "f_sku", "date"]).reset_index(drop=True)

    def trend(series: np.ndarray) -> float:
        if len(series) < 2:
            return 0.0
        x = np.arange(len(series))
        return float(np.polyfit(x, series, 1)[0])

    def fe_per_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        qty = g["quantity"]

        # Lag features
        for lag in CONFIG["lag_features"]:
            g[f"lag_{lag}"] = qty.shift(lag)

        # Rolling stats
        for window in CONFIG["rolling_windows"]:
            g[f"rolling_mean_{window}"] = (
                qty.rolling(window=window, min_periods=1).mean()
            )
            g[f"rolling_std_{window}"] = (
                qty.rolling(window=window, min_periods=1).std().fillna(0)
            )
            g[f"rolling_min_{window}"] = (
                qty.rolling(window=window, min_periods=1).min()
            )
            g[f"rolling_max_{window}"] = (
                qty.rolling(window=window, min_periods=1).max()
            )

        # Changes & pct changes
        g["change_1"] = qty.diff(1)
        g["change_7"] = qty.diff(7)
        pct_change_1 = qty.pct_change(1)
        pct_change_7 = qty.pct_change(7)
        g["pct_change_1"] = pct_change_1.replace([np.inf, -np.inf], 0).fillna(0)
        g["pct_change_7"] = pct_change_7.replace([np.inf, -np.inf], 0).fillna(0)

        # Trend & volatility
        g["trend_7"] = qty.rolling(7, min_periods=2).apply(trend, raw=True)
        g["trend_30"] = qty.rolling(30, min_periods=2).apply(trend, raw=True)

        g["volatility_7"] = (
            g["rolling_std_7"] / g["rolling_mean_7"]
        ).replace([np.inf, -np.inf], 0).fillna(0)
        g["volatility_30"] = (
            g["rolling_std_30"] / g["rolling_mean_30"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        # Date-based features
        g["year"] = g["date"].dt.year
        g["month"] = g["date"].dt.month
        g["day"] = g["date"].dt.day
        g["day_of_week"] = g["date"].dt.dayofweek
        g["day_of_year"] = g["date"].dt.dayofyear
        g["week_of_year"] = g["date"].dt.isocalendar().week.astype(int)
        g["quarter"] = g["date"].dt.quarter
        g["is_weekend"] = (g["day_of_week"] >= 5).astype(int)
        g["is_month_start"] = g["date"].dt.is_month_start.astype(int)
        g["is_month_end"] = g["date"].dt.is_month_end.astype(int)

        return g

    panel_fe = panel.groupby(["branch_code", "f_sku"], group_keys=False).apply(
        fe_per_group
    )

    # Drop initial rows with NaN từ lag/rolling
    print("   Dropping rows with NaN after feature engineering...")
    before = len(panel_fe)
    panel_fe = panel_fe.dropna()
    after = len(panel_fe)
    print(f"   ✅ Final panel rows (state rows): {after:,} (dropped {before - after:,})")

    return panel_fe


# ============================================================================
# 4. BUILD MULTI-STEP TRAINING DATA
# ============================================================================


def build_multistep_dataset(panel_fe: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Từ state features per day, tạo dataset multi-step:
    - Mỗi state tại ngày t → H bản ghi với horizon_offset ∈ {1..H}
    - Target = quantity_{t + horizon_offset}
    """
    print("\n📐 Step 5: Building multi-step training dataset...")

    groups = panel_fe.groupby(["branch_code", "f_sku"])
    rows = []
    total_groups = len(groups)
    print(f"   Creating multi-step samples for {total_groups} (branch, F_SKU) series...")

    for i, ((branch, fsku), g) in enumerate(groups, 1):
        if i % 100 == 0:
            print(f"      Progress: {i}/{total_groups} ({i/total_groups*100:.1f}%)")

        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)
        if n <= horizon:
            continue

        for idx in range(n):
            # base date t
            base_row = g.iloc[idx]
            max_h = min(horizon, n - idx - 1)
            if max_h <= 0:
                continue

            for h in range(1, max_h + 1):
                target_idx = idx + h
                target_qty = float(g.loc[target_idx, "quantity"])

                row = base_row.copy()
                row["horizon_offset"] = h
                row["target"] = target_qty
                # base_date dùng để split train/test
                row["base_date"] = base_row["date"]
                rows.append(row)

    dataset = pd.DataFrame(rows)
    print(f"   ✅ Multi-step samples: {len(dataset):,}")
    return dataset


def split_train_test_multistep(
    df: pd.DataFrame, train_end: str, test_start: str
):
    """
    Train/test split theo base_date t (không phải ngày target).
    """
    print("\n✂️  Step 6: Train/Test Split (multi-step, panel)...")

    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)

    train_df = df[df["base_date"] <= train_end_ts].copy()
    test_df = df[df["base_date"] >= test_start_ts].copy()

    print("   📅 Training period (base_date):")
    print(f"      Start: {train_df['base_date'].min()}  | End: {train_df['base_date'].max()}")
    print(f"      Rows:  {len(train_df):,}")

    print("\n   📅 Testing period (base_date):")
    print(f"      Start: {test_df['base_date'].min()}  | End: {test_df['base_date'].max()}")
    print(f"      Rows:  {len(test_df):,}")

    # Feature columns (exclude target & raw identifiers)
    exclude_cols = [
        "date",
        "quantity",
        "series_key",
        "branch_code",
        "region",
        "f_sku",
        "target",
        "base_date",
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    print(f"\n   ✅ Features: {len(feature_cols)}")
    print(f"   ✅ Train samples: {len(X_train):,}")
    print(f"   ✅ Test samples:  {len(X_test):,}")

    return X_train, X_test, y_train, y_test, feature_cols


# ============================================================================
# 5. TRAINING
# ============================================================================


def train_xgboost_panel_multistep(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
):
    """Train multi-step panel XGBoost model."""
    print("\n🚀 Step 7: Training MULTI-STEP PANEL XGBoost Model...")

    print("   Hyperparameters:")
    for k, v in CONFIG["xgb_params"].items():
        print(f"      {k:20s}: {v}")

    model = xgb.XGBRegressor(**CONFIG["xgb_params"])

    print("\n   Training in progress...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Train metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mape = mean_absolute_percentage_error(y_train, y_train_pred)

    print("\n   📊 Training Metrics (multi-step panel):")
    print(f"      RMSE:  {train_rmse:.2f}")
    print(f"      MAE:   {train_mae:.2f}")
    print(f"      R²:    {train_r2:.4f}")
    print(f"      MAPE:  {train_mape*100:.2f}%")

    # Test metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred)

    print("\n   📊 Testing Metrics (OOT, multi-step panel):")
    print(f"      RMSE:  {test_rmse:.2f}")
    print(f"      MAE:   {test_mae:.2f}")
    print(f"      R²:    {test_r2:.4f}")
    print(f"      MAPE:  {test_mape*100:.2f}%")

    metrics = {
        "train": {
            "rmse": float(train_rmse),
            "mae": float(train_mae),
            "r2": float(train_r2),
            "mape": float(train_mape),
        },
        "test": {
            "rmse": float(test_rmse),
            "mae": float(test_mae),
            "r2": float(test_r2),
            "mape": float(test_mape),
        },
    }

    return model, metrics


# ============================================================================
# 6. SAVE MODEL & METADATA
# ============================================================================


def save_panel_multistep_model_and_metadata(
    model,
    feature_cols,
    id_maps: Dict[str, Dict[str, int]],
    metrics: Dict,
) -> Tuple[str, str]:
    """Save multi-step panel model và metadata (bao gồm encoders)."""
    print("\n💾 Step 8: Saving MULTI-STEP PANEL model & metadata...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"xgboost_panel_multistep_{timestamp}.pkl"
    model_path = os.path.join(CONFIG["model_dir"], model_filename)

    joblib.dump(model, model_path)
    print(f"   ✅ Model saved: {model_path}")

    metadata = {
        "timestamp": timestamp,
        "model_filename": model_filename,
        "features": feature_cols,
        "n_features": len(feature_cols),
        "metrics": {
            "train_rmse": metrics["train"]["rmse"],
            "train_mae": metrics["train"]["mae"],
            "train_r2": metrics["train"]["r2"],
            "train_mape": metrics["train"]["mape"],
            "test_rmse": metrics["test"]["rmse"],
            "test_mae": metrics["test"]["mae"],
            "test_r2": metrics["test"]["r2"],
            "test_mape": metrics["test"]["mape"],
        },
        "hyperparameters": CONFIG["xgb_params"],
        "data_info": {
            "train_end": CONFIG["train_end_date"],
            "test_start": CONFIG["test_start_date"],
        },
        "encoders": id_maps,
        "horizon": CONFIG["horizon"],
        "multi_step": True,
    }

    metadata_path = os.path.join(
        CONFIG["model_dir"], f"metadata_panel_multistep_{timestamp}.json"
    )
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Metadata saved: {metadata_path}")

    return model_path, metadata_path


# ============================================================================
# 7. MAIN
# ============================================================================


def main():
    try:
        # 1. Load panel data
        panel, _ = load_panel_data(CONFIG["data_path"], CONFIG["branch_path"])

        # 2. Encode identifiers + static features
        panel_enc, id_maps = encode_identifiers(panel)

        # 3. Time-series features per series (state)
        panel_fe = create_panel_features(panel_enc)

        # 4. Build multi-step dataset
        multi_df = build_multistep_dataset(panel_fe, CONFIG["horizon"])

        # 5. Train/test split (theo base_date)
        X_train, X_test, y_train, y_test, feature_cols = split_train_test_multistep(
            multi_df,
            CONFIG["train_end_date"],
            CONFIG["test_start_date"],
        )

        # 6. Train model
        model, metrics = train_xgboost_panel_multistep(
            X_train, y_train, X_test, y_test
        )

        # 7. Save model & metadata
        model_path, metadata_path = save_panel_multistep_model_and_metadata(
            model, feature_cols, id_maps, metrics
        )

        print("\n" + "=" * 80)
        print("🎉 MULTI-STEP PANEL TRAINING COMPLETE!")
        print("=" * 80)
        print(
            f"\n📊 Final Metrics (Multi-step Panel): "
            f"Train RMSE={metrics['train']['rmse']:.2f}, R²={metrics['train']['r2']:.4f} | "
            f"Test RMSE={metrics['test']['rmse']:.2f}, R²={metrics['test']['r2']:.4f}"
        )
        print(f"\n💾 Saved Files:")
        print(f"   Model: {model_path}")
        print(f"   Metadata: {metadata_path}")

        return model, metrics

    except Exception as e:
        print(f"\n❌ ERROR in multi-step panel training: {e}")
        import traceback

        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    main()





"""
ModelLogger: centralized logging for forecasting pipelines.

Ghi log ra file `.log` thay vì in ra terminal, bao gồm:
- Thông tin tổng quan về DataFrame đầu vào (shape, cột)
- Thông tin về features đã tạo
- Kết quả dự đoán cho từng series / từng sản phẩm
"""

import json
import logging
import os
from typing import Any, Dict, Optional

import pandas as pd


class ModelLogger:
    """Light wrapper quanh logging.Logger với helper cho time-series & forecast."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)

    def log_dataframe_overview(
        self,
        df: Optional[pd.DataFrame],
        name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Ghi log thông tin tổng quan về DataFrame (shape, cột,...)."""
        base = {"name": name}

        if context:
            base["context"] = context

        if df is None:
            base["shape"] = None
            base["columns"] = []
        else:
            base["shape"] = list(df.shape)
            base["columns"] = list(df.columns[:20])

        self._logger.info("DATAFRAME_OVERVIEW " + json.dumps(base, ensure_ascii=False, default=str))

    def log_forecast_series(
        self,
        key: Dict[str, Any],
        historical_df: Optional[pd.DataFrame],
        forecast_df: Optional[pd.DataFrame],
        metrics: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Ghi log 1 dòng JSON cho mỗi series forecast (ví dụ từng sản phẩm / chi nhánh).
        Không dump full DataFrame để tránh file quá lớn, chỉ ghi kích thước & vài stats.
        """
        payload: Dict[str, Any] = {
            "key": key,
            "type": "FORECAST_SERIES",
        }

        if historical_df is not None and not historical_df.empty:
            payload["hist_days"] = int(len(historical_df))
            try:
                payload["hist_start"] = str(historical_df.index.min())
                payload["hist_end"] = str(historical_df.index.max())
                col = historical_df.columns[0]
                payload["hist_mean"] = float(historical_df[col].mean())
            except Exception:
                pass
        else:
            payload["hist_days"] = 0

        if forecast_df is not None and not forecast_df.empty:
            payload["horizon"] = int(len(forecast_df))
            try:
                payload["forecast_start"] = str(forecast_df.index.min())
                payload["forecast_end"] = str(forecast_df.index.max())
                col = forecast_df.columns[0]
                payload["forecast_mean"] = float(forecast_df[col].mean())
                payload["forecast_total"] = float(forecast_df[col].sum())
            except Exception:
                pass
        else:
            payload["horizon"] = 0

        if metrics:
            payload["metrics"] = metrics

        if extra:
            payload["extra"] = extra

        self._logger.info(json.dumps(payload, ensure_ascii=False, default=str))


_model_logger_instance: Optional[ModelLogger] = None


def get_model_logger(log_dir: str = "model_logs", filename: str = "model_debug.log") -> ModelLogger:
    """
    Lấy singleton ModelLogger, ghi log vào file .log.

    - log_dir: thư mục chứa file log
    - filename: tên file log (mặc định: model_debug.log)
    """
    global _model_logger_instance

    if _model_logger_instance is not None:
        return _model_logger_instance

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    logger = logging.getLogger("model_logger")
    logger.setLevel(logging.INFO)

    # Tránh add nhiều handler khi gọi nhiều lần
    if not logger.handlers:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    _model_logger_instance = ModelLogger(logger)
    logger.info("=== ModelLogger initialized ===")
    return _model_logger_instance








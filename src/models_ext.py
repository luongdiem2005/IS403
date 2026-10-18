"""
Module mở rộng (Giai đoạn 2 - Module 1): Cài đặt và tinh chỉnh các mô hình Deep Learning
cho dữ liệu bảng (FT-Transformer, NODE) tối ưu hóa theo e-Profits.
"""

from typing import Any, Dict
import pandas as pd
from lifelines import KaplanMeierFitter


def train_and_tune_dl_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    kmf: KaplanMeierFitter,
    seed: int = 42,
    n_trials: int = 20,
) -> Dict[str, Any]:
    """
    Huấn luyện và tinh chỉnh siêu tham số các mô hình Deep Learning trên dữ liệu bảng
    (FT-Transformer, NODE) bằng Optuna dưới các mục tiêu AUC và e-Profits.

    Args:
        X_train: Tập đặc trưng huấn luyện (bao gồm 'tenure' và 'customer_value').
        y_train: Nhãn rời bỏ tập huấn luyện.
        kmf: Mô hình KaplanMeierFitter đã fit trên tập huấn luyện.
        seed: Hạt giống ngẫu nhiên cố định.
        n_trials: Số lượng thử nghiệm Optuna.

    Returns:
        Dict ánh xạ tên mô hình mở rộng -> mô hình tốt nhất đã huấn luyện.
    """
    raise NotImplementedError("Sẽ được triển khai ở Giai đoạn 2 (Mở rộng mô hình Deep Learning).")
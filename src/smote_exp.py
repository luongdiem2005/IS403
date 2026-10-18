"""
Module thực nghiệm mất cân bằng lớp (Giai đoạn 2 - Module 2):
Đánh giá tác động của SMOTE / SMOTENC lên dịch chuyển xác suất (probability calibration)
và lợi nhuận thực tế e-Profits.
"""

from typing import Tuple
import pandas as pd
from imblearn.over_sampling import SMOTE


def apply_smote_training(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = 42,
    sampling_strategy: float = 1.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Áp dụng SMOTE chỉ trên tập huấn luyện (sau khi đã ước lượng đường cong Kaplan-Meier
    trên phân phối gốc để không làm méo xác suất sinh tồn).

    Args:
        X_train: Đặc trưng tập huấn luyện gốc.
        y_train: Nhãn tập huấn luyện gốc.
        seed: Hạt giống ngẫu nhiên cố định.
        sampling_strategy: Tỷ lệ lấy mẫu quá mức lớp thiểu số.

    Returns:
        Tuple (X_train_resampled, y_train_resampled)
    """
    smote = SMOTE(random_state=seed, sampling_strategy=sampling_strategy)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res, name=y_train.name)
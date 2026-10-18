"""
Phân tích sinh tồn Kaplan-Meier để ước lượng xác suất giữ chân khách hàng:
- TRR (Tenure-based Retention Probability): r_i^TRR(Delta) = S(T_i + Delta) / S(T_i)
- ARR (Average Retention Probability): trung bình của TRR trên tập dữ liệu.
Lưu ý: Chỉ fit đường cong S(.) trên tập huấn luyện để tránh rò rỉ dữ liệu.
"""

from typing import Tuple, Union
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


def fit_kaplan_meier(
    tenure_train: Union[pd.Series, np.ndarray],
    churn_train: Union[pd.Series, np.ndarray],
) -> KaplanMeierFitter:
    """
    Huấn luyện mô hình sinh tồn Kaplan-Meier S(t) chỉ trên tập train.

    Args:
        tenure_train: Thời gian gắn bó (tháng) của khách hàng tập train.
        churn_train: Nhãn rời bỏ thực tế (1 = churn event, 0 = right-censored).

    Returns:
        Đối tượng KaplanMeierFitter đã được fit.
    """
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=np.asarray(tenure_train, dtype=float),
        event_observed=np.asarray(churn_train, dtype=int),
    )
    return kmf


def compute_trr(
    kmf: KaplanMeierFitter,
    tenure: Union[pd.Series, np.ndarray],
    delta: float = 1.0,
) -> np.ndarray:
    """
    Tính xác suất duy trì có điều kiện theo thời gian gắn bó (TRR) cho từng khách hàng:
        r_i^TRR(delta) = S(T_i + delta) / max(S(T_i), 1e-12)

    Với T_i + delta vượt quá thời gian quan sát lớn nhất của tập train,
    lifelines giữ nguyên giá trị S cuối cùng (step function) đúng như code gốc.

    Args:
        kmf: Mô hình KaplanMeierFitter đã fit trên tập train.
        tenure: Mảng/Series thời gian gắn bó hiện tại T_i của khách hàng.
        delta: Bước thời gian dự phóng (mặc định = 1.0 tháng).

    Returns:
        Mảng numpy 1 chiều chứa giá trị TRR cho từng khách hàng.
    """
    t = np.asarray(tenure, dtype=float)
    s_t = kmf.survival_function_at_times(t).values.flatten()
    s_t_delta = kmf.survival_function_at_times(t + delta).values.flatten()

    trr = s_t_delta / np.maximum(s_t, 1e-12)
    return trr


def compute_arr(
    kmf: KaplanMeierFitter,
    tenure_train: Union[pd.Series, np.ndarray],
    delta: float = 1.0,
) -> float:
    """
    Tính xác suất duy trì trung bình toàn cục (ARR) từ phân phối tenure của tập train:
        r^ARR(delta) = (1 / N) * sum_{i=1}^N r_i^TRR(delta)

    Args:
        kmf: Mô hình KaplanMeierFitter đã fit trên tập train.
        tenure_train: Thời gian gắn bó T_i của tập train.
        delta: Bước thời gian (mặc định = 1.0 tháng).

    Returns:
        Giá trị số thực (float) biểu diễn ARR (trên IBM ~ 0.77, trên Maven ~ 0.75).
    """
    trr_train = compute_trr(kmf, tenure_train, delta=delta)
    return float(np.mean(trr_train))


def get_retention_estimates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    tenure_col: str = "tenure",
    delta: float = 1.0,
) -> Tuple[KaplanMeierFitter, float, np.ndarray, np.ndarray]:
    """
    Hàm tiện ích fit KM trên tập train và trả về (kmf, arr_train, trr_train, trr_eval).
    """
    kmf = fit_kaplan_meier(X_train[tenure_col], y_train)
    trr_train = compute_trr(kmf, X_train[tenure_col], delta=delta)
    arr_train = float(np.mean(trr_train))
    trr_eval = compute_trr(kmf, X_eval[tenure_col], delta=delta)
    return kmf, arr_train, trr_train, trr_eval
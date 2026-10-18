"""
Tính toán thước đo lợi nhuận kỳ vọng e-Profits, CLV, chi phí can thiệp,
cùng các độ đo đối sánh: EMP (Expected Maximum Profit), Top-Decile Lift (TDL) và Lift Index.
"""

from typing import Tuple, Union
import numpy as np
import pandas as pd
from scipy.special import beta as beta_func, betainc


def compute_clv(
    revenue: Union[np.ndarray, pd.Series, float],
    retention_rate: Union[np.ndarray, pd.Series, float],
    profit_margin: float = 0.3,
    cap: float = 0.995,
) -> np.ndarray:
    """
    Tính giá trị vòng đời khách hàng (CLV) theo Công thức (3) trong bài báo:
        CLV_i = (R_i * M) / (1 - min(r_i, cap))

    Args:
        revenue: Doanh thu khách hàng R_i (hoặc customer_value).
        retention_rate: Xác suất duy trì r_i (vô hướng ARR hoặc mảng TRR).
        profit_margin: Biên lợi nhuận M (mặc định = 0.3).
        cap: Trần chặn xác suất duy trì để tránh mẫu số tiến về 0 (mặc định = 0.995).

    Returns:
        Mảng numpy chứa CLV_i của từng khách hàng.
    """
    rev = np.asarray(revenue, dtype=float)
    r = np.asarray(retention_rate, dtype=float)
    r_capped = np.minimum(r, cap)
    clv = (rev * profit_margin) / (1.0 - r_capped)
    return clv


def compute_intervention_costs(
    clv: np.ndarray,
    cpo: float = 0.1,
    cost_of_contact: Tuple[float, float] = (0.0, 0.3),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính chi phí ưu đãi (C_offer) và chi phí liên hệ (C_contact) theo Công thức (4) & (5):
        C_offer,i = CPO * CLV_i
        C_contact,i = max(c_0, c_1 * C_offer,i)

    Lưu ý quan trọng từ code gốc:
    - Phụ lục A (Toy Example): cost_of_contact = (5.0, 0.3)
    - Khi tuning mô hình (2-Metrics-profitability.py): cost_of_contact = (0.0, 0.3)
    - Khi đánh giá tập test Bảng 3 & 4 (4-Evaluation.py): cost_of_contact = (1.0, 0.3)

    Returns:
        Tuple (c_offer, c_contact)
    """
    c0, c1 = cost_of_contact
    c_offer = cpo * np.asarray(clv, dtype=float)
    c_contact = np.maximum(c0, c1 * c_offer)
    return c_offer, c_contact


def calculate_per_customer_profit(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    revenue: Union[np.ndarray, pd.Series],
    retention_rate: Union[np.ndarray, pd.Series, float],
    profit_margin: float = 0.3,
    cpo: float = 0.1,
    cost_of_contact: Tuple[float, float] = (0.0, 0.3),
    cap: float = 0.995,
) -> np.ndarray:
    """
    Tính lợi nhuận tăng thêm Pi_i cho từng khách hàng theo Công thức (2):
    - True Positive (y_i = 1, y_hat_i = 1): CLV_i - C_offer,i - C_contact,i
    - False Positive (y_i = 0, y_hat_i = 1): -C_offer,i - C_contact,i
    - Không can thiệp (y_hat_i = 0): 0
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    clv = compute_clv(revenue, retention_rate, profit_margin=profit_margin, cap=cap)
    c_offer, c_contact = compute_intervention_costs(
        clv, cpo=cpo, cost_of_contact=cost_of_contact
    )

    profits = np.zeros_like(clv, dtype=float)
    tp_mask = (y_t == 1) & (y_p == 1)
    fp_mask = (y_t == 0) & (y_p == 1)

    profits[tp_mask] = clv[tp_mask] - c_offer[tp_mask] - c_contact[tp_mask]
    profits[fp_mask] = -c_offer[fp_mask] - c_contact[fp_mask]

    return profits


def calculate_eprofits(
    y_true: Union[np.ndarray, pd.Series],
    y_scores: Union[np.ndarray, pd.Series],
    customer_value: Union[np.ndarray, pd.Series],
    retention_rate: Union[np.ndarray, pd.Series, float],
    target_fraction: float = 1.0,
    profit_margin: float = 0.3,
    cpo: float = 0.1,
    cost_of_contact: Tuple[float, float] = (0.0, 0.3),
    threshold: float = 0.5,
    cap: float = 0.995,
    scores_are_binary: bool = False,
) -> float:
    """
    Tính tổng lợi nhuận e-Profits = sum(Pi_i) trên toàn bộ tập (target_fraction=1.0)
    hoặc phân khúc Top khách hàng (ví dụ target_fraction=0.2 cho Top 20%).

    Bám sát chính xác cách cắt Top 20% trong code gốc:
    - Sắp xếp giảm dần theo ['customer_value', 'scores']
    - Lấy int(len(df) * target_fraction) dòng đầu tiên
    - Chuyển xác suất dự đoán sang nhãn nhị phân với ngưỡng 0.5 rồi tính tổng Pi_i.
    """
    y_t = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_scores, dtype=float)
    val = np.asarray(customer_value, dtype=float)

    if np.ndim(retention_rate) == 0:
        ret = np.full(len(y_t), float(retention_rate), dtype=float)
    else:
        ret = np.asarray(retention_rate, dtype=float)

    if target_fraction < 1.0:
        df_temp = pd.DataFrame(
            {
                "y_true": y_t,
                "scores": scores,
                "customer_value": val,
                "retention": ret,
            }
        )
        df_temp = df_temp.sort_values(
            by=["customer_value", "scores"], ascending=False
        )
        top_n = int(len(df_temp) * target_fraction)
        df_temp = df_temp.head(top_n)

        y_t = df_temp["y_true"].values
        scores = df_temp["scores"].values
        val = df_temp["customer_value"].values
        ret = df_temp["retention"].values

    if len(y_t) == 0:
        return 0.0

    if scores_are_binary:
        y_pred = scores.astype(int)
    else:
        y_pred = (scores >= threshold).astype(int)

    profits = calculate_per_customer_profit(
        y_true=y_t,
        y_pred=y_pred,
        revenue=val,
        retention_rate=ret,
        profit_margin=profit_margin,
        cpo=cpo,
        cost_of_contact=cost_of_contact,
        cap=cap,
    )
    return float(np.sum(profits))


def calculate_top_decile_lift(
    y_true: Union[np.ndarray, pd.Series],
    y_scores: Union[np.ndarray, pd.Series],
    fraction: float = 0.1,
) -> float:
    """
    Tính Top-Decile Lift (TDL): tỷ lệ khách hàng thực sự rời bỏ trong top 10% có xác suất
    dự đoán cao nhất chia cho tỷ lệ rời bỏ trung bình của toàn tập.
    """
    y_t = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_scores, dtype=float)
    base_rate = np.mean(y_t)
    if base_rate == 0:
        return 0.0

    order = np.argsort(-scores, kind="mergesort")
    top_n = max(1, int(len(y_t) * fraction))
    top_rate = np.mean(y_t[order[:top_n]])
    return float(top_rate / base_rate)


def calculate_lift_index(
    y_true: Union[np.ndarray, pd.Series],
    y_scores: Union[np.ndarray, pd.Series],
    n_deciles: int = 10,
) -> float:
    """
    Tính Lift Index (weighted index of churners across ranked deciles).
    Giá trị tiến gần 1 thể hiện khả năng xếp hạng ưu tiên nhóm rời bỏ tốt.
    Công thức chuẩn: (1 * S_1 + 0.9 * S_2 + ... + 0.1 * S_10) / sum(y_true)
    """
    y_t = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_scores, dtype=float)
    total_churners = np.sum(y_t)
    if total_churners == 0:
        return 0.0

    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_t[order]
    deciles = np.array_split(y_sorted, n_deciles)

    weights = [1.0 - i / n_deciles for i in range(n_deciles)]
    weighted_sum = sum(w * np.sum(d) for w, d in zip(weights, deciles))
    return float(weighted_sum / total_churners)


def calculate_emp(
    y_true: Union[np.ndarray, pd.Series],
    y_scores: Union[np.ndarray, pd.Series],
    target_fraction: float = 1.0,
    alpha: float = 6.0,
    beta: float = 14.0,
    clv: float = 200.0,
    d: float = 10.0,
    f: float = 1.0,
) -> float:
    """
    Tính Expected Maximum Profit (EMP) theo Verbraken et al. (2013) với tham số bài báo:
    alpha=6, beta=14, clv=200, d=10 (C_offer), f=1 (C_contact).
    Hỗ trợ đánh giá trên toàn tập (target_fraction=1.0) hoặc top-segment (target_fraction=0.2)
    khi sắp xếp giảm dần theo xác suất dự đoán.
    """
    from sklearn.metrics import roc_curve

    y_t = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_scores, dtype=float)

    if target_fraction < 1.0:
        order = np.argsort(-scores, kind="mergesort")
        top_n = max(1, int(len(y_t) * target_fraction))
        idx = order[:top_n]
        y_t = y_t[idx]
        scores = scores[idx]

    if len(np.unique(y_t)) < 2:
        return 0.0

    pi_1 = np.mean(y_t == 1)
    pi_0 = 1.0 - pi_1

    fpr, tpr, _ = roc_curve(y_t, scores)
    # Xây dựng bao lồi ROC (ROC Convex Hull)
    from scipy.spatial import ConvexHull

    pts = np.column_stack([fpr, tpr])
    pts_aug = np.vstack([pts, [1.0, 0.0]])
    try:
        hull = ConvexHull(pts_aug)
        hull_eq = pts_aug[hull.vertices]
        hull_pts = hull_eq[
            ~((hull_eq[:, 0] == 1.0) & (hull_eq[:, 1] == 0.0))
        ]
        hull_pts = hull_pts[np.argsort(hull_pts[:, 0])]
    except Exception:
        hull_pts = pts

    F0 = hull_pts[:, 0]
    F1 = hull_pts[:, 1]

    delta = d / clv
    phi = f / clv

    # Tính các ngưỡng gamma trên bao lồi ROC
    dF0 = np.diff(F0)
    dF1 = np.diff(F1)
    valid = dF1 > 1e-12
    dF0 = dF0[valid]
    dF1 = dF1[valid]
    F0_seg = F0[:-1][valid]
    F1_seg = F1[:-1][valid]

    gammas = (pi_0 * (delta + phi) * dF0 + pi_1 * phi * dF1) / (pi_1 * dF1)
    gammas = np.clip(gammas, 0.0, 1.0)
    gamma_bounds = np.concatenate([[0.0], gammas, [1.0]])

    emp_val = 0.0
    mean_gamma_factor = beta_func(alpha + 1, beta) / beta_func(alpha, beta)

    for i in range(len(F0_seg)):
        g_low = np.clip(gamma_bounds[i], 0.0, 1.0)
        g_high = np.clip(gamma_bounds[i + 1], 0.0, 1.0)
        if g_high <= g_low:
            continue

        prob_interval = betainc(alpha, beta, g_high) - betainc(alpha, beta, g_low)
        exp_gamma_interval = mean_gamma_factor * (
            betainc(alpha + 1, beta, g_high) - betainc(alpha + 1, beta, g_low)
        )

        term1 = pi_1 * clv * (exp_gamma_interval - (delta + phi) * prob_interval) * F1_seg[i]
        term2 = -pi_0 * clv * (delta + phi) * prob_interval * F0_seg[i]
        emp_val += term1 + term2

    return float(max(0.0, emp_val))
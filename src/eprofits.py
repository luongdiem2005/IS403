"""
Module tính toán thước đo lợi nhuận kỳ vọng e-Profits, CLV, chi phí can thiệp,
cùng các độ đo đối sánh: EMP (dùng đúng gói EMP-PY của tác giả), Top-Decile Lift (TDL) và Lift Index
"""

from typing import Tuple, Union
import numpy as np
import pandas as pd

try:
    from EMP.metrics import empChurn
    HAS_EMP_PKG = True
except ImportError:
    HAS_EMP_PKG = False


def compute_clv(
    revenue: Union[np.ndarray, pd.Series, float],
    retention_rate: Union[np.ndarray, pd.Series, float],
    profit_margin: float = 0.3,
    cap: float = 0.995,
) -> np.ndarray:
    """Tính CLV_i = (R_i * M) / (1 - min(r_i, cap))"""
    rev = np.asarray(revenue, dtype=float)
    r = np.asarray(retention_rate, dtype=float)
    r_capped = np.minimum(r, cap)
    return (rev * profit_margin) / (1.0 - r_capped)


def compute_intervention_costs(
    clv: np.ndarray,
    cpo: float = 0.1,
    cost_of_contact: Tuple[float, float] = (0.0, 0.3),
) -> Tuple[np.ndarray, np.ndarray]:
    """Tính chi phí ưu đãi C_offer và chi phí liên hệ C_contact"""
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
    """Tính lợi nhuận Pi_i cho từng khách hàng theo Công thức (2)"""
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
    Tính tổng e-Profits bám sát hàm calculate_eprofits và _eprofits trong 4-Evaluation.py
    """
    y_t = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_scores, dtype=float)
    val = np.asarray(customer_value, dtype=float)

    if scores_are_binary:
        y_pred = scores.astype(int)
    else:
        y_pred = (scores >= threshold).astype(int)

    if np.ndim(retention_rate) == 0:
        ret = np.full(len(y_t), float(retention_rate), dtype=float)
    else:
        ret = np.asarray(retention_rate, dtype=float).flatten()

    df_eval = pd.DataFrame(
        {
            "customer_value": val,
            "true": y_t,
            "predict": y_pred,
            "scores": scores,
            "retention_rate": ret,
        }
    )
    df_eval = df_eval.sort_values(by=["customer_value", "scores"], ascending=False)

    if target_fraction < 1.0:
        top_n = int(len(df_eval) * target_fraction)
        df_eval = df_eval.head(top_n)

    if len(df_eval) == 0:
        return 0.0

    profits = calculate_per_customer_profit(
        y_true=df_eval["true"].values,
        y_pred=df_eval["predict"].values,
        revenue=df_eval["customer_value"].values,
        retention_rate=df_eval["retention_rate"].values,
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
    """Tính Top-Decile Lift bám sát hàm calculate_lift trong 4-Evaluation.py"""
    df = pd.DataFrame({"true": np.asarray(y_true, dtype=int), "scores": np.asarray(y_scores, dtype=float)})
    df = df.sort_values("scores", ascending=False)
    top_percentile = int(len(df) * fraction)
    top_df = df.head(top_percentile)
    overall_churn_rate = df["true"].mean()
    if overall_churn_rate == 0:
        return 0.0
    return float(top_df["true"].mean() / overall_churn_rate)


def calculate_lift_index(
    y_true: Union[np.ndarray, pd.Series],
    y_scores: Union[np.ndarray, pd.Series],
    n_deciles: int = 10,
) -> float:
    """Tính Lift Index bám sát hàm calculate_lift trong 4-Evaluation.py"""
    df = pd.DataFrame({"true": np.asarray(y_true, dtype=int), "scores": np.asarray(y_scores, dtype=float)})
    df = df.sort_values("scores", ascending=False)
    df["decile"] = pd.qcut(df["scores"].rank(method="first"), n_deciles, labels=False)
    deciles_churners = df.groupby("decile")["true"].sum().to_frame()
    deciles_churners["weight"] = np.arange(0.1, 1.1, 0.1)
    deciles_churners["w_churners"] = deciles_churners["weight"] * deciles_churners["true"]
    total = deciles_churners["true"].sum()
    if total == 0:
        return 0.0
    return float(deciles_churners["w_churners"].sum() / total)


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
    Tính EMP bám sát hàm calculate_EMP trong 4-Evaluation.py
    bằng cách sử dụng gói EMP-PY (empChurn)
    """
    df = pd.DataFrame({"true": np.asarray(y_true, dtype=int), "scores": np.asarray(y_scores, dtype=float)})
    df = df.sort_values("scores", ascending=False)

    if target_fraction < 1.0:
        top_percentile = int(len(df) * target_fraction)
        df = df.head(top_percentile)

    if HAS_EMP_PKG:
        try:
            res = empChurn(
                df["scores"].values,
                df["true"].values,
                alpha=alpha,
                beta=beta,
                clv=clv,
                d=d,
                f=f,
                print_output=False,
                return_output=True,
                rounding=None,
            )
            return float(res.EMP)
        except Exception:
            return 0.0
    else:
        raise ImportError("Vui lòng cài đặt gói EMP-PY bằng lệnh: pip install EMP-PY")
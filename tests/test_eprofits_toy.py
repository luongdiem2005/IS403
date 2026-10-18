"""
Unit test đối chiếu hàm tính e-Profits với Toy Example (5 khách hàng A-E)
tại Phụ lục A (Bảng 9, Trang 19-20) của bài báo gốc.
"""

import numpy as np
import pytest
from src.eprofits import (
    calculate_eprofits,
    calculate_per_customer_profit,
    compute_clv,
    compute_intervention_costs,
)


@pytest.fixture
def toy_data():
    """Bộ dữ liệu chuẩn 5 khách hàng A, B, C, D, E từ Bảng 9 của bài báo."""
    return {
        "customers": ["A", "B", "C", "D", "E"],
        "y_true": np.array([1, 0, 1, 0, 1]),
        "y_pred": np.array([1, 1, 0, 0, 1]),
        "revenue": np.array([100.0, 80.0, 120.0, 90.0, 150.0]),
        "trr": np.array([0.80, 0.90, 0.75, 0.85, 0.70]),
        "tenure": np.array([120, 50, 180, 90, 200]),
        "profit_margin": 0.3,
        "cpo": 0.1,
        "cost_of_contact": (5.0, 0.3),  # c0 = 5, c1 = 0.3 trong Phụ lục A
    }


def test_toy_example_trr_per_customer_and_total(toy_data):
    """
    Kiểm tra tính toán e-Profits theo TRR (Mục A.1 bài báo):
    - Khách hàng A (TP): CLV = 150, C_offer = 15, C_contact = 5 => Pi_A = 130.0
    - Khách hàng B (FP): CLV = 240, C_offer = 24, C_contact = 7.2 => Pi_B = -31.2
    - Khách hàng C (FN): y_hat = 0 => Pi_C = 0.0
    - Khách hàng D (TN): y_hat = 0 => Pi_D = 0.0
    - Khách hàng E (TP): CLV = 150, C_offer = 15, C_contact = 5 => Pi_E = 130.0
    - Tổng e-Profits (TRR) = 228.8
    """
    profits_trr = calculate_per_customer_profit(
        y_true=toy_data["y_true"],
        y_pred=toy_data["y_pred"],
        revenue=toy_data["revenue"],
        retention_rate=toy_data["trr"],
        profit_margin=toy_data["profit_margin"],
        cpo=toy_data["cpo"],
        cost_of_contact=toy_data["cost_of_contact"],
    )

    expected_profits_trr = np.array([130.0, -31.2, 0.0, 0.0, 130.0])
    np.testing.assert_allclose(profits_trr, expected_profits_trr, rtol=1e-6)

    total_trr = calculate_eprofits(
        y_true=toy_data["y_true"],
        y_scores=toy_data["y_pred"],
        customer_value=toy_data["revenue"],
        retention_rate=toy_data["trr"],
        target_fraction=1.0,
        profit_margin=toy_data["profit_margin"],
        cpo=toy_data["cpo"],
        cost_of_contact=toy_data["cost_of_contact"],
        scores_are_binary=True,
    )
    assert np.isclose(total_trr, 228.8, atol=1e-6)


def test_toy_example_arr_per_customer_and_total(toy_data):
    """
    Kiểm tra tính toán e-Profits theo ARR (Mục A.2 bài báo):
    - r^ARR(1) = (0.80 + 0.90 + 0.75 + 0.85 + 0.70) / 5 = 0.80
    - Khách hàng A (TP): CLV = 150, C_offer = 15, C_contact = 5 => Pi_A = 130.0
    - Khách hàng B (FP): CLV = 120, C_offer = 12, C_contact = 5 => Pi_B = -17.0
    - Khách hàng C & D: y_hat = 0 => Pi_C = Pi_D = 0.0
    - Khách hàng E (TP): CLV = 225, C_offer = 22.5, C_contact = 6.75 => Pi_E = 195.75
    - Tổng e-Profits (ARR) = 308.75
    """
    arr = float(np.mean(toy_data["trr"]))
    assert np.isclose(arr, 0.80, atol=1e-6)

    profits_arr = calculate_per_customer_profit(
        y_true=toy_data["y_true"],
        y_pred=toy_data["y_pred"],
        revenue=toy_data["revenue"],
        retention_rate=arr,
        profit_margin=toy_data["profit_margin"],
        cpo=toy_data["cpo"],
        cost_of_contact=toy_data["cost_of_contact"],
    )

    expected_profits_arr = np.array([130.0, -17.0, 0.0, 0.0, 195.75])
    np.testing.assert_allclose(profits_arr, expected_profits_arr, rtol=1e-6)

    total_arr = calculate_eprofits(
        y_true=toy_data["y_true"],
        y_scores=toy_data["y_pred"],
        customer_value=toy_data["revenue"],
        retention_rate=arr,
        target_fraction=1.0,
        profit_margin=toy_data["profit_margin"],
        cpo=toy_data["cpo"],
        cost_of_contact=toy_data["cost_of_contact"],
        scores_are_binary=True,
    )
    assert np.isclose(total_arr, 308.75, atol=1e-6)


def test_clv_retention_cap_at_0995():
    """Kiểm tra cơ chế chặn trần xác suất duy trì ở 0.995 khi TRR = 1.0."""
    clv = compute_clv(revenue=100.0, retention_rate=1.0, profit_margin=0.3, cap=0.995)
    # 100 * 0.3 / (1 - 0.995) = 30 / 0.005 = 6000.0
    assert np.isclose(clv, 6000.0, atol=1e-6)
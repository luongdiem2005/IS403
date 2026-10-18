"""
Đánh giá mô hình trên tập kiểm thử (held-out test set), tổng hợp Bảng 3, 4, 5
và thực hiện phân tích độ bất định Bootstrap (Bảng 6), kiểm định Wilcoxon (Bảng 7),
cùng tương quan thứ hạng Spearman (Bảng 8).
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from lifelines import KaplanMeierFitter

from src.eprofits import (
    calculate_emp,
    calculate_eprofits,
    calculate_lift_index,
    calculate_top_decile_lift,
)
from src.survival import compute_arr, compute_trr


RESULT_COLUMNS = [
    "Dataset",
    "Tuning_Policy",
    "Model",
    "F1",
    "Acc",
    "AUC",
    "TDL",
    "Lift_Index",
    "eProf_0.2_ARR",
    "eProf_0.2_TRR",
    "eProf_1.0_ARR",
    "eProf_1.0_TRR",
    "EMP_0.2",
    "EMP_1.0",
]


def evaluate_single_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    arr_train: float,
    trr_test: np.ndarray,
    dataset_name: str,
    policy_name: str,
    model_name: str,
    cost_of_contact_eval: Tuple[float, float] = (1.0, 0.3),
) -> Dict[str, Any]:
    """
    Đánh giá một mô hình đã huấn luyện trên tập test held-out.
    Lưu ý: Trong 4-Evaluation.py của tác giả, hàm get_business_report_dfs truyền
    Cost_of_Contact = (1, 0.3) để xuất ra Bảng 3 và Bảng 4.
    """
    y_true = np.asarray(y_test, dtype=int)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = y_pred.astype(float)

    cust_val = X_test["customer_value"].values

    f1_pct = round(f1_score(y_true, y_pred, zero_division=0) * 100.0, 2)
    acc_pct = round(accuracy_score(y_true, y_pred) * 100.0, 2)
    auc_pct = round(roc_auc_score(y_true, y_scores) * 100.0, 2)

    tdl = round(calculate_top_decile_lift(y_true, y_scores, fraction=0.1), 4)
    lift_idx = round(calculate_lift_index(y_true, y_scores, n_deciles=10), 4)

    eprof_02_arr = round(
        calculate_eprofits(
            y_true=y_true,
            y_scores=y_scores,
            customer_value=cust_val,
            retention_rate=arr_train,
            target_fraction=0.2,
            cost_of_contact=cost_of_contact_eval,
        )
    )
    eprof_02_trr = round(
        calculate_eprofits(
            y_true=y_true,
            y_scores=y_scores,
            customer_value=cust_val,
            retention_rate=trr_test,
            target_fraction=0.2,
            cost_of_contact=cost_of_contact_eval,
        )
    )
    eprof_10_arr = round(
        calculate_eprofits(
            y_true=y_true,
            y_scores=y_scores,
            customer_value=cust_val,
            retention_rate=arr_train,
            target_fraction=1.0,
            cost_of_contact=cost_of_contact_eval,
        )
    )
    eprof_10_trr = round(
        calculate_eprofits(
            y_true=y_true,
            y_scores=y_scores,
            customer_value=cust_val,
            retention_rate=trr_test,
            target_fraction=1.0,
            cost_of_contact=cost_of_contact_eval,
        )
    )

    emp_02 = round(calculate_emp(y_true, y_scores, target_fraction=0.2), 2)
    emp_10 = round(calculate_emp(y_true, y_scores, target_fraction=1.0), 2)

    return {
        "Dataset": dataset_name,
        "Tuning_Policy": policy_name,
        "Model": model_name,
        "F1": f1_pct,
        "Acc": acc_pct,
        "AUC": auc_pct,
        "TDL": tdl,
        "Lift_Index": lift_idx,
        "eProf_0.2_ARR": eprof_02_arr,
        "eProf_0.2_TRR": eprof_02_trr,
        "eProf_1.0_ARR": eprof_10_arr,
        "eProf_1.0_TRR": eprof_10_trr,
        "EMP_0.2": emp_02,
        "EMP_1.0": emp_10,
    }


def evaluate_all_models(
    trained_models: Dict[str, Dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    kmf: KaplanMeierFitter,
    dataset_name: str,
    cost_of_contact_eval: Tuple[float, float] = (1.0, 0.3),
) -> pd.DataFrame:
    """
    Đánh giá toàn bộ các mô hình đã huấn luyện trên tập test và trả về DataFrame
    khớp hoàn toàn với cấu trúc Bảng 3 và Bảng 4 của bài báo.
    """
    arr_train = compute_arr(kmf, X_train["tenure"], delta=1.0)
    trr_test = compute_trr(kmf, X_test["tenure"], delta=1.0)

    rows: List[Dict[str, Any]] = []
    for policy_name, models_dict in trained_models.items():
        for model_name, model in models_dict.items():
            row = evaluate_single_model(
                model=model,
                X_test=X_test,
                y_test=y_test,
                arr_train=arr_train,
                trr_test=trr_test,
                dataset_name=dataset_name,
                policy_name=policy_name,
                model_name=model_name,
                cost_of_contact_eval=cost_of_contact_eval,
            )
            rows.append(row)

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def get_best_models_summary_table5(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tái hiện Bảng 5: Với mỗi chiến lược tuning (Policy), chọn mô hình đạt
    e-Profits (Full TRR) cao nhất trên tập test.
    """
    summary_rows = []
    for (dataset, policy), group in results_df.groupby(
        ["Dataset", "Tuning_Policy"], sort=False
    ):
        best_idx = group["eProf_1.0_TRR"].idxmax()
        best_row = group.loc[best_idx]
        summary_rows.append(
            {
                "Dataset": dataset,
                "Policy": policy,
                "Model": best_row["Model"],
                "AUC (%)": best_row["AUC"],
                "F1 (%)": best_row["F1"],
                "Acc. (%)": best_row["Acc"],
                "e-Profits (Full TRR)": best_row["eProf_1.0_TRR"],
            }
        )
    return pd.DataFrame(summary_rows)


def bootstrap_confidence_intervals_table6(
    trained_models: Dict[str, Dict[str, Any]],
    results_df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    kmf: KaplanMeierFitter,
    dataset_name: str,
    n_bootstraps: int = 500,
    seed: int = 42,
    cost_of_contact_eval: Tuple[float, float] = (1.0, 0.3),
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Tái hiện Bảng 6: Bootstrap 500 lần lấy mẫu lại tập test để tính Mean và 95% CI
    cho e-Profits (Full TRR) của các mô hình tốt nhất dưới từng chính sách tuning.
    """
    rng = np.random.RandomState(seed)
    trr_test_full = compute_trr(kmf, X_test["tenure"], delta=1.0)
    y_true_full = np.asarray(y_test, dtype=int)
    cust_val_full = X_test["customer_value"].values
    n_samples = len(y_true_full)

    table6_rows = []
    bootstrap_distributions: Dict[str, np.ndarray] = {}

    for policy in ["AUC", "Accuracy", "e-Profits"]:
        if policy not in trained_models:
            continue
        sub_df = results_df[
            (results_df["Dataset"] == dataset_name)
            & (results_df["Tuning_Policy"] == policy)
        ]
        best_model_name = sub_df.loc[sub_df["eProf_1.0_TRR"].idxmax(), "Model"]
        best_model = trained_models[policy][best_model_name]

        if hasattr(best_model, "predict_proba"):
            scores_full = best_model.predict_proba(X_test)[:, 1]
        else:
            scores_full = best_model.predict(X_test).astype(float)

        boot_profits = np.zeros(n_bootstraps, dtype=float)
        for b in range(n_bootstraps):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            boot_profits[b] = calculate_eprofits(
                y_true=y_true_full[idx],
                y_scores=scores_full[idx],
                customer_value=cust_val_full[idx],
                retention_rate=trr_test_full[idx],
                target_fraction=1.0,
                cost_of_contact=cost_of_contact_eval,
            )

        bootstrap_distributions[policy] = boot_profits
        table6_rows.append(
            {
                "Dataset": dataset_name,
                "Policy": policy,
                "Mean": round(float(np.mean(boot_profits))),
                "CI low": round(float(np.percentile(boot_profits, 2.5))),
                "CI high": round(float(np.percentile(boot_profits, 97.5))),
            }
        )

    return pd.DataFrame(table6_rows), bootstrap_distributions


def wilcoxon_significance_table7(
    bootstrap_distributions: Dict[str, np.ndarray],
    dataset_name: str,
) -> pd.DataFrame:
    """
    Tái hiện Bảng 7: Kiểm định phi tham số cặp Wilcoxon signed-rank test so sánh
    chênh lệch e-Profits (Full TRR) giữa chiến lược tuning e-Profits với AUC và Accuracy.
    """
    rows = []
    if "e-Profits" in bootstrap_distributions:
        eprof_dist = bootstrap_distributions["e-Profits"]
        for baseline_policy in ["AUC", "Accuracy"]:
            if baseline_policy in bootstrap_distributions:
                base_dist = bootstrap_distributions[baseline_policy]
                diff = eprof_dist - base_dist
                _, p_val = wilcoxon(eprof_dist, base_dist, alternative="greater")
                rows.append(
                    {
                        "Dataset": dataset_name,
                        "Policy A": "e-Profits",
                        "Policy B": baseline_policy,
                        "Metric": "e-Profits (Full TRR)",
                        "Mean diff": f"{round(float(np.mean(diff))):+d}",
                        "p-value": p_val,
                    }
                )
    return pd.DataFrame(rows)


def spearman_reranking_table8(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tái hiện Bảng 8: Tính tương quan hạng Spearman (rho) giữa xếp hạng theo AUC
    và xếp hạng theo lợi nhuận e-Profits (Full TRR & Top 20% TRR).
    """
    rows = []
    for dataset, group in results_df.groupby("Dataset", sort=False):
        for col, label in [
            ("eProf_1.0_TRR", "e-Profits (Full TRR)"),
            ("eProf_0.2_TRR", "e-Profits (Top20 TRR)"),
        ]:
            rho, p_val = spearmanr(group["AUC"], group[col])
            rows.append(
                {
                    "Dataset": dataset,
                    "Profit metric": label,
                    "Spearman rho": round(float(rho), 4),
                    "p-value": p_val,
                }
            )
    return pd.DataFrame(rows)
"""
Script thực thi toàn bộ Giai đoạn 1: Tái hiện bài báo gốc e-Profits
trên hai bộ dữ liệu IBM Telco và Maven Telecom.
"""

import os
import pandas as pd
from utils.seed import set_seed
from src.preprocess import load_and_preprocess
from src.survival import compute_arr, fit_kaplan_meier
from src.models_base import train_and_tune_models
from src.evaluate import (
    bootstrap_confidence_intervals_table6,
    evaluate_all_models,
    get_best_models_summary_table5,
    spearman_reranking_table8,
    wilcoxon_significance_table7,
)

SEED = 42


def run_replication_for_dataset(dataset_name: str, output_dir: str = "results") -> pd.DataFrame:
    print("=" * 80)
    print(f"BẮT ĐẦU TÁI HIỆN BÀI BÁO GỐC TRÊN TẬP DỮ LIỆU: {dataset_name}")
    print("=" * 80)

    set_seed(SEED)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Tiền xử lý dữ liệu (test_size=0.3, random_state=42, không stratify đúng như code gốc)
    X_train, X_test, y_train, y_test = load_and_preprocess(
        dataset_name=dataset_name,
        data_dir="data",
        test_size=0.3,
        random_state=SEED,
        stratify=False,
    )
    print(f"[{dataset_name}] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 2. Ước lượng Kaplan-Meier chỉ trên tập train
    kmf = fit_kaplan_meier(X_train["tenure"], y_train)
    arr_train = compute_arr(kmf, X_train["tenure"], delta=1.0)
    print(f"[{dataset_name}] Estimated Training ARR (Delta=1): {arr_train:.4f}")

    # 3. Huấn luyện & Tinh chỉnh 6 mô hình theo 5 mục tiêu (lúc tuning dùng c0=0, c1=0.3)
    trained_models = train_and_tune_models(
        X_train=X_train,
        y_train=y_train,
        kmf=kmf,
        seed=SEED,
        cv=5,
        cost_of_contact_tuning=(0.0, 0.3),
    )

    # 4. Đánh giá trên tập test held-out (lúc test dùng c0=1, c1=0.3 như 4-Evaluation.py)
    results_df = evaluate_all_models(
        trained_models=trained_models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        kmf=kmf,
        dataset_name=dataset_name,
        cost_of_contact_eval=(1.0, 0.3),
    )

    out_csv = os.path.join(output_dir, f"phase1_{dataset_name.lower()}_table3_4.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\n[{dataset_name}] Đã lưu bảng kết quả chính tại: {out_csv}")
    print(results_df.to_string(index=False))

    # 5. Xuất Bảng 5, Bảng 6 (Bootstrap CI), Bảng 7 (Wilcoxon), Bảng 8 (Spearman)
    table5_df = get_best_models_summary_table5(results_df)
    table5_df.to_csv(os.path.join(output_dir, f"phase1_{dataset_name.lower()}_table5.csv"), index=False)

    table6_df, boot_dists = bootstrap_confidence_intervals_table6(
        trained_models=trained_models,
        results_df=results_df,
        X_test=X_test,
        y_test=y_test,
        kmf=kmf,
        dataset_name=dataset_name,
        n_bootstraps=500,
        seed=SEED,
        cost_of_contact_eval=(1.0, 0.3),
    )
    table6_df.to_csv(os.path.join(output_dir, f"phase1_{dataset_name.lower()}_table6_bootstrap.csv"), index=False)

    table7_df = wilcoxon_significance_table7(boot_dists, dataset_name=dataset_name)
    table7_df.to_csv(os.path.join(output_dir, f"phase1_{dataset_name.lower()}_table7_wilcoxon.csv"), index=False)

    table8_df = spearman_reranking_table8(results_df)
    table8_df.to_csv(os.path.join(output_dir, f"phase1_{dataset_name.lower()}_table8_spearman.csv"), index=False)

    return results_df


if __name__ == "__main__":
    ibm_results = run_replication_for_dataset("IBM")
    maven_results = run_replication_for_dataset("Maven")

    all_results = pd.concat([ibm_results, maven_results], ignore_index=True)
    all_results.to_csv("results/phase1_all_results.csv", index=False)
    print("\nHoàn tất Giai đoạn 1! Toàn bộ kết quả đã được lưu trong thư mục results/.")
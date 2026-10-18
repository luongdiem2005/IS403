"""
Module khởi tạo và huấn luyện (GridSearchCV 5-fold) 6 mô hình học máy gốc:
RF, GBC, XGB, CBC, LGBM, EBM dưới 5 mục tiêu tinh chỉnh (AUC, Accuracy, F1-score, EMP, e-Profits).
"""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from lifelines import KaplanMeierFitter

from src.eprofits import calculate_emp, calculate_eprofits
from src.survival import compute_trr


def get_base_models_and_grids(seed: int = 42) -> Dict[str, Tuple[Any, Dict[str, list]]]:
    """
    Trả về từ điển gồm 6 mô hình cơ sở và lưới siêu tham số đúng theo Bảng 2 của bài báo.
    """
    return {
        "RF": (
            RandomForestClassifier(random_state=seed, n_jobs=-1),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        "GBC": (
            GradientBoostingClassifier(random_state=seed),
            {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 4, 5],
            },
        ),
        "XGB": (
            XGBClassifier(
                random_state=seed,
                eval_metric="logloss",
                n_jobs=-1,
            ),
            {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 4, 5],
                "colsample_bytree": [0.3, 0.7],
            },
        ),
        "CBC": (
            CatBoostClassifier(
                random_state=seed,
                verbose=0,
                thread_count=-1,
            ),
            {
                "iterations": [100, 200],
                "learning_rate": [0.01, 0.1],
                "depth": [3, 5],
            },
        ),
        "LGBM": (
            LGBMClassifier(
                random_state=seed,
                verbose=-1,
                n_jobs=-1,
            ),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "num_leaves": [31, 50],
            },
        ),
        "EBM": (
            ExplainableBoostingClassifier(random_state=seed, n_jobs=-1),
            {
                "learning_rate": [0.01, 0.05, 0.1],
                "max_bins": [128, 256],
                "interactions": [10, 20],
                "max_interaction_bins": [16, 32],
            },
        ),
    }


def make_eprofits_cv_scorer(
    kmf: KaplanMeierFitter,
    tenure_col: str = "tenure",
    value_col: str = "customer_value",
    cost_of_contact: Tuple[float, float] = (0.0, 0.3),
):
    """
    Tạo custom scorer cho GridSearchCV tối ưu hóa e-Profits (TRR trên toàn tập validation fold).
    Lưu ý:
    - Đường cong Kaplan-Meier kmf được ước lượng 1 lần trên tập train và giữ cố định.
    - Trong code gốc (2-Metrics-profitability.py), lúc tuning dùng cost_of_contact = (0.0, 0.3).
    """

    def _scorer(estimator, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        if hasattr(estimator, "predict_proba"):
            y_prob = estimator.predict_proba(X_val)[:, 1]
        else:
            y_prob = estimator.predict(X_val)

        trr_val = compute_trr(kmf, X_val[tenure_col], delta=1.0)
        cust_val = X_val[value_col].values

        return calculate_eprofits(
            y_true=y_val,
            y_scores=y_prob,
            customer_value=cust_val,
            retention_rate=trr_val,
            target_fraction=1.0,
            profit_margin=0.3,
            cpo=0.1,
            cost_of_contact=cost_of_contact,
            threshold=0.5,
        )

    return _scorer


def make_emp_cv_scorer():
    """Tạo custom scorer cho GridSearchCV tối ưu hóa EMP trên validation fold."""

    def _scorer(estimator, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        if hasattr(estimator, "predict_proba"):
            y_prob = estimator.predict_proba(X_val)[:, 1]
        else:
            y_prob = estimator.predict(X_val)
        return calculate_emp(y_true=y_val, y_scores=y_prob, target_fraction=1.0)

    return _scorer


def get_tuning_scorers(
    kmf: KaplanMeierFitter,
    cost_of_contact_tuning: Tuple[float, float] = (0.0, 0.3),
) -> Dict[str, Any]:
    """
    Trả về 5 hàm mục tiêu tuning khớp với Bảng 3 và Bảng 4 của bài báo:
    AUC, Accuracy, F1-score, EMP, e-Profits.
    """
    return {
        "AUC": "roc_auc",
        "Accuracy": "accuracy",
        "F1-score": "f1",
        "EMP": make_emp_cv_scorer(),
        "e-Profits": make_eprofits_cv_scorer(
            kmf=kmf, cost_of_contact=cost_of_contact_tuning
        ),
    }


def train_and_tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    kmf: KaplanMeierFitter,
    seed: int = 42,
    cv: int = 5,
    cost_of_contact_tuning: Tuple[float, float] = (0.0, 0.3),
    selected_policies: list = None,
    selected_models: list = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Chạy GridSearchCV 5-fold cho tất cả các mô hình dưới từng chiến lược tuning.

    Returns:
        Từ điển lồng nhau: trained_models[policy_name][model_name] = best_estimator
    """
    scorers = get_tuning_scorers(kmf, cost_of_contact_tuning=cost_of_contact_tuning)
    models_and_grids = get_base_models_and_grids(seed=seed)

    if selected_policies is not None:
        scorers = {k: v for k, v in scorers.items() if k in selected_policies}
    if selected_models is not None:
        models_and_grids = {
            k: v for k, v in models_and_grids.items() if k in selected_models
        }

    trained_models: Dict[str, Dict[str, Any]] = {}

    for policy_name, scoring_fn in scorers.items():
        trained_models[policy_name] = {}
        print(f"\n--- Đang tinh chỉnh siêu tham số theo mục tiêu: {policy_name} ---")
        for model_name, (estimator, param_grid) in models_and_grids.items():
            # Tránh xung đột đa luồng khi chạy EBM hoặc CatBoost trong GridSearchCV
            n_jobs_gs = 1 if model_name in ["EBM", "CBC"] else -1
            grid = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring=scoring_fn,
                cv=cv,
                refit=True,
                n_jobs=n_jobs_gs,
            )
            grid.fit(X_train, y_train)
            trained_models[policy_name][model_name] = grid.best_estimator_
            print(
                f"  + [{model_name}] Best CV {policy_name}: {grid.best_score_:.4f} | Params: {grid.best_params_}"
            )

    return trained_models
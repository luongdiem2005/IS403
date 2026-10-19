"""
Module khởi tạo và huấn luyện (GridSearchCV 5-fold) 6 mô hình học máy gốc
bám sát với 3-Model_Training.py của tác giả
"""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from lifelines import KaplanMeierFitter

from src.eprofits import calculate_emp, calculate_eprofits
from src.survival import compute_trr


def get_base_models_and_grids(seed: int = 42) -> Dict[str, Tuple[Any, Dict[str, list]]]:
    """Khởi tạo 6 mô hình và lưới tham số bám sát với 3-Model_Training.py"""
    return {
        "RF": (
            RandomForestClassifier(random_state=seed),
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
                random_seed=seed,
                verbose=0,
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
            ),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "num_leaves": [31, 50],
            },
        ),
        "EBM": (
            ExplainableBoostingClassifier(random_state=seed),
            {
                "learning_rate": [0.01, 0.05, 0.1],
                "max_bins": [128, 256],
                "max_interaction_bins": [16, 32],
                "interactions": [10, 20],
            },
        ),
    }


def make_eprofits_cv_scorer(
    kmf: KaplanMeierFitter,
    tenure_col: str = "tenure",
    value_col: str = "customer_value",
    cost_of_contact: Tuple[float, float] = (0.0, 0.3),
):
    """Tạo custom scorer tối ưu hóa e-Profits (TRR) bám sát với custom_eprofits_scorer_tenure"""

    def _scorer(estimator, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        y_scores = estimator.predict_proba(X_val)[:, 1]
        y_pred = (y_scores >= 0.5).astype(int)
        trr_val = compute_trr(kmf, X_val[tenure_col], delta=1.0)
        cust_val = X_val[value_col].values

        return calculate_eprofits(
            y_true=y_val,
            y_scores=y_pred,
            customer_value=cust_val,
            retention_rate=trr_val,
            target_fraction=1.0,
            profit_margin=0.3,
            cpo=0.1,
            cost_of_contact=cost_of_contact,
            scores_are_binary=True,
        )

    return _scorer


def make_emp_cv_scorer():
    """Tạo custom scorer tối ưu hóa EMP bằng empChurn bám sát emp_scorer của tác giả"""

    def _scorer(estimator, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        y_proba = estimator.predict_proba(X_val)[:, 1]
        return calculate_emp(y_true=y_val, y_scores=y_proba, target_fraction=1.0)

    return _scorer


def get_tuning_scorers(
    kmf: KaplanMeierFitter,
    cost_of_contact_tuning: Tuple[float, float] = (0.0, 0.3),
    reproduce_author_auc_bug: bool = True,
) -> Dict[str, Any]:
    """
    Trả về 5 hàm mục tiêu tuning
    - Nếu reproduce_author_auc_bug=True: mô phỏng chính xác hành vi make_scorer(roc_auc_score, needs_proba=True)
      trong code gốc của tác giả (chọn cấu hình đầu tiên trong lưới tham số cho AUC) để ra kết quả như Bảng 3 & 4
    - Nếu reproduce_author_auc_bug=False: dùng chuẩn 'roc_auc' của scikit-learn
    """
    if reproduce_author_auc_bug:
        auc_scorer = lambda est, X, y: np.nan
    else:
        auc_scorer = "roc_auc"

    return {
        "AUC": auc_scorer,
        "Accuracy": metrics.make_scorer(metrics.accuracy_score),
        "F1-score": metrics.make_scorer(metrics.f1_score),
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
    reproduce_author_auc_bug: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Chạy GridSearchCV 5-fold bám sát với 3-Model_Training.py"""
    scorers = get_tuning_scorers(
        kmf,
        cost_of_contact_tuning=cost_of_contact_tuning,
        reproduce_author_auc_bug=reproduce_author_auc_bug,
    )
    models_and_grids = get_base_models_and_grids(seed=seed)
    trained_models: Dict[str, Dict[str, Any]] = {}

    for policy_name, scoring_fn in scorers.items():
        trained_models[policy_name] = {}
        n_jobs_gs = 1 if policy_name == "EMP" else -1
        print(f"\n--- Đang tinh chỉnh siêu tham số theo mục tiêu: {policy_name} ---")
        for model_name, (estimator, param_grid) in models_and_grids.items():
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
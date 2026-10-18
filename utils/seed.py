"""Cố định seed toàn dự án. Gọi set_seed() ở đầu mọi notebook/script"""
import os
import random
import numpy as np

SEED = 42


def set_seed(seed: int = SEED, deterministic_torch: bool = True) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
    return seed


def make_cv(n_splits: int = 5, seed: int = SEED, shuffle: bool = False):
    """shuffle=False (mặc định) tái hiện code gốc: cv=5 -> StratifiedKFold không shuffle.
    random_state chỉ có tác dụng khi shuffle=True, nên với False sẽ không truyền seed."""
    from sklearn.model_selection import StratifiedKFold
    if shuffle:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=n_splits)


def make_models(seed: int = SEED) -> dict:
    """6 mô hình gốc, tất cả dùng chung seed"""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from interpret.glassbox import ExplainableBoostingClassifier
    return {
        "rf":   RandomForestClassifier(random_state=seed),
        "gbc":  GradientBoostingClassifier(random_state=seed),
        "xgb":  XGBClassifier(random_state=seed, eval_metric="logloss"),
        "cbc":  CatBoostClassifier(verbose=0, random_seed=seed),   # CatBoost dùng random_seed
        "lgbm": LGBMClassifier(random_state=seed, verbose=-1),
        "ebm":  ExplainableBoostingClassifier(random_state=seed),
    }
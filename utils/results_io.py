"""Lưu / đọc kết quả theo đúng 14 cột Bảng 3-4 của bài báo."""
import json
import platform
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

RESULT_COLUMNS = [
    "Dataset", "Tuning_Policy", "Model", "F1", "Acc", "AUC", "TDL", "Lift_Index",
    "eProf_0.2_ARR", "eProf_0.2_TRR", "eProf_1.0_ARR", "eProf_1.0_TRR",
    "EMP_0.2", "EMP_1.0",
]
KEY = ["Dataset", "Tuning_Policy", "Model"]

# tên cột trả về từ business_metric_report() (4-Evaluation.py) -> tên chuẩn
RENAME_MAP = {
    "Name": "Model", "Accuracy": "Acc", "Top decile Lift": "TDL", "Lift Index": "Lift_Index",
    "eProfits (0.2,0.3,AVG_RR)": "eProf_0.2_ARR",
    "eProfits (0.2,0.3,Tenure_RR)": "eProf_0.2_TRR",
    "eProfits (1.0,0.3,AVG_RR)": "eProf_1.0_ARR",
    "eProfits (1.0,0.3,Tenure_RR)": "eProf_1.0_TRR",
    "EMP (0.2,200,10,1)": "EMP_0.2",
    "EMP (1.0,200,10,1)": "EMP_1.0",
}
# key của scoring dict trong code gốc -> nhãn gọn
POLICY_MAP = {"AUC": "AUC", "accuracy": "Accuracy", "f1": "F1", "emp": "EMP",
              "eprofits_avg": "eProfits_ARR", "eprofits_tenure": "eProfits_TRR"}


def standardize(df: pd.DataFrame, dataset: str, policy: str) -> pd.DataFrame:
    d = df.rename(columns=RENAME_MAP).copy()
    d["Dataset"] = dataset
    d["Tuning_Policy"] = POLICY_MAP.get(policy, policy)
    d["Model"] = d["Model"].astype(str).str.upper()
    missing = [c for c in RESULT_COLUMNS if c not in d.columns]
    if missing:
        raise ValueError(f"Thiếu cột: {missing}")
    return d[RESULT_COLUMNS]


def save_results(df, dataset, policy, results_dir=RESULTS_DIR, meta=None) -> Path:
    """Ghi/gộp vào results/<dataset>_results.csv. Chạy lại cùng (Dataset, Policy, Model) thì ghi đè dòng cũ."""
    results_dir = Path(results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{dataset.lower()}_results.csv"
    new = standardize(df, dataset, policy)
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
        new = new.drop_duplicates(subset=KEY, keep="last")
    new.sort_values(KEY).to_csv(path, index=False)
    _write_meta(results_dir / f"{dataset.lower()}_meta.json", meta)
    return path


def save_all(business_report_dfs_dict: dict, dataset: str, meta=None) -> None:
    """Truyền thẳng dict {policy: df} mà get_business_report_dfs() trả về."""
    for policy, df in business_report_dfs_dict.items():
        save_results(df, dataset, policy, meta=meta)


def load_results(dataset=None, results_dir=RESULTS_DIR) -> pd.DataFrame:
    files = sorted(Path(results_dir).glob(f"{dataset.lower() if dataset else '*'}_results.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def to_paper_table(df, dataset, policy) -> pd.DataFrame:
    """Dựng bảng giống Bảng 3/4: F1, Acc, AUC đổi sang %."""
    t = df[(df.Dataset == dataset) & (df.Tuning_Policy == policy)].drop(columns=["Dataset", "Tuning_Policy"]).copy()
    for c in ["F1", "Acc", "AUC"]:
        t[c] = (t[c] * 100).round(2)
    return t.round(4).reset_index(drop=True)


def _write_meta(path: Path, meta):
    import numpy as np, sklearn
    info = {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}
    info.update(meta or {})
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False))
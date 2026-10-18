"""
Tiền xử lý dữ liệu cho hai tập dữ liệu IBM Telco và Maven Telecom.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def preprocess_ibm(
    filepath: str = "data/IBM.csv",
    test_size: float = 0.3,
    random_state: int = 42,
    stratify: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Tiền xử lý tập dữ liệu IBM Telco Customer Churn theo code gốc.

    Quy trình:
    1. Bỏ cột 'customerID'.
    2. Đổi 'No internet service' -> 'No' ở 6 cột dịch vụ internet.
    3. Đổi 'No phone service' -> 'No' ở cột 'MultipleLines'.
    4. Thay khoảng trắng " " ở 'TotalCharges' bằng "0" và ép kiểu float.
    5. Áp dụng LabelEncoder cho mọi cột kiểu object, sau đó gọi pd.get_dummies(drop_first=True).
    6. Tạo cột 'customer_value' = MonthlyCharges * (tenure + 1) trước khi tách X, y.
    7. Chia train/test với test_size=0.3, random_state=42 (mặc định không stratify).

    Returns:
        X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(filepath)

    if "customerID" in df.columns:
        df.drop(["customerID"], axis=1, inplace=True)

    internet_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for col in internet_cols:
        if col in df.columns:
            df[col] = df[col].replace("No internet service", "No")

    if "MultipleLines" in df.columns:
        df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].replace(" ", "0")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # LabelEncoder cho tất cả các cột object
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Gọi get_dummies như code gốc (thực chất giữ nguyên vì các cột đã thành số)
    df = pd.get_dummies(df, drop_first=True)

    # Tính customer_value theo công thức IBM trong code gốc: MonthlyCharges * (tenure + 1)
    df["customer_value"] = df["MonthlyCharges"] * (df["tenure"] + 1)

    target_col = "Churn"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    stratify_param = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )

    return X_train, X_test, y_train, y_test


def preprocess_maven(
    filepath: str = "data/Maven.csv",
    test_size: float = 0.3,
    random_state: int = 42,
    stratify: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Tiền xử lý tập dữ liệu Maven Telecom Churn theo code gốc

    Quy trình:
    1. Chuẩn hóa tên cột: bỏ khoảng trắng và chuyển về chữ thường.
    2. Bỏ các cột định danh/rò rỉ: customerid, churncategory, churnreason, zipcode, longitude, latitude.
    3. Lọc bỏ nhóm khách hàng mới gia nhập ('Joined' / chứa chuỗi 'Join').
    4. Mã hóa nhãn customerstatus: Churned -> 1, Stayed -> 0.
    5. Xử lý giá trị khuyết (Missing values) cho các cột cước phí và dịch vụ.
    6. Chuẩn hóa tên cột thời gian gắn bó thành 'tenure' (nếu đang là 'tenureinmonths') để đồng nhất.
    7. Áp dụng LabelEncoder cho mọi cột object, sau đó gọi pd.get_dummies(drop_first=True).
    8. Tạo cột 'customer_value' = totalrevenue / (tenure + 1) trước khi tách X, y.
    9. Chia train/test với test_size=0.3, random_state=42 (mặc định không stratify).

    Returns:
        X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(filepath)

    # Chuẩn hóa tên cột: xóa khoảng trắng, chuyển chữ thường
    df.columns = df.columns.str.replace(" ", "", regex=False).str.lower()

    # Các cột cần loại bỏ
    drop_cols = [
        "customerid",
        "churncategory",
        "churnreason",
        "zipcode",
        "longitude",
        "latitude",
    ]
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    df.drop(columns=existing_drop_cols, inplace=True)

    # Lọc bỏ nhóm Joined (6% recent joiners)
    status_col = "customerstatus"
    df = df.loc[~df[status_col].astype(str).str.contains("Join", case=False, na=False)].copy()

    # Chuyển nhãn Churned -> 1, Stayed -> 0
    df[status_col] = df[status_col].astype(str).str.strip().map(
        {"Churned": 1, "Stayed": 0, "churned": 1, "stayed": 0}
    )

    # Xử lý missing values theo đúng code gốc Maven
    zero_fill_cols = ["avgmonthlylongdistancecharges", "avgmonthlygbdownload", "avgmonthlydownload"]
    for col in zero_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    if "multiplelines" in df.columns:
        df["multiplelines"] = df["multiplelines"].fillna("no phone service")

    internet_cols_maven = [
        "internettype",
        "onlinesecurity",
        "onlinebackup",
        "deviceprotectionplan",
        "premiumtechsupport",
        "streamingtv",
        "streamingmovies",
        "streamingmusic",
        "unlimiteddata",
    ]
    for col in internet_cols_maven:
        if col in df.columns:
            df[col] = df[col].fillna("no internet service")

    # Đảm bảo cột tenure đồng nhất tên để dùng chung cho phân tích sinh tồn
    if "tenureinmonths" in df.columns and "tenure" not in df.columns:
        df.rename(columns={"tenureinmonths": "tenure"}, inplace=True)

    #Điền các giá trị khuyết còn sót (nếu có)
    for col in df.columns:
        if col == status_col:
            continue
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(0)

    # Áp dụng LabelEncoder cho mọi cột object
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    df = pd.get_dummies(df, drop_first=True)

    # Tính customer_value cho Maven: totalrevenue / (tenure + 1)
    df["customer_value"] = df["totalrevenue"] / (df["tenure"] + 1)

    X = df.drop(columns=[status_col])
    y = df[status_col].astype(int)

    stratify_param = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )

    return X_train, X_test, y_train, y_test


def load_and_preprocess(
    dataset_name: str,
    data_dir: str = "data",
    test_size: float = 0.3,
    random_state: int = 42,
    stratify: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Hàm wrapper tải và tiền xử lý dữ liệu theo tên dataset ('IBM' hoặc 'Maven')."""
    name = dataset_name.strip().upper()
    if name == "IBM":
        return preprocess_ibm(
            filepath=f"{data_dir}/IBM.csv",
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    elif name == "MAVEN":
        return preprocess_maven(
            filepath=f"{data_dir}/Maven.csv",
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    else:
        raise ValueError(f"Dataset không hợp lệ: {dataset_name}. Chỉ hỗ trợ 'IBM' hoặc 'Maven'.")
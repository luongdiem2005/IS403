import numpy as np, pandas as pd
from utils.seed import set_seed
from utils.results_io import standardize, RESULT_COLUMNS, RENAME_MAP

def test_seed_reproducible():
    set_seed(42); a = np.random.rand(3)
    set_seed(42); b = np.random.rand(3)
    assert (a == b).all()

def test_columns():
    cols = ["Name", "F1", "Accuracy", "AUC", "Top decile Lift", "Lift Index"] + \
           [k for k in RENAME_MAP if k.startswith(("eProfits", "EMP"))]
    df = pd.DataFrame([[ "rf"] + [0.0] * (len(cols) - 1)], columns=cols)
    assert list(standardize(df, "IBM", "AUC").columns) == RESULT_COLUMNS
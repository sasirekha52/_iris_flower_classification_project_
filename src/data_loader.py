import pandas as pd
from .config import DATA_PATH, TARGET_COLUMN, FEATURE_COLUMNS

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def validate_data(df):
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    return True

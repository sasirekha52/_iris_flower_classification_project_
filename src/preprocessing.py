from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .config import FEATURE_COLUMNS

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), FEATURE_COLUMNS)
        ],
        remainder="drop",
    )

def build_pipeline(model):
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])

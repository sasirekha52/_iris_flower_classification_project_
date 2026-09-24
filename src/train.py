from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

from .config import *
from .data_loader import load_data, validate_data
from .preprocessing import build_pipeline
from .evaluate import save_evaluation

def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    validate_data(df)

    # Remove duplicate rows before modeling.
    df = df.drop_duplicates().reset_index(drop=True)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    fitted_models = {}

    for name, estimator in models.items():
        pipeline = build_pipeline(estimator)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
        pipeline.fit(X_train, y_train)
        test_pred = pipeline.predict(X_test)

        fitted_models[name] = pipeline
        results.append({
            "model": name,
            "cv_accuracy_mean": scores.mean(),
            "cv_accuracy_std": scores.std(),
            "test_accuracy": (test_pred == y_test).mean(),
        })

    comparison = pd.DataFrame(results).sort_values(
        "cv_accuracy_mean", ascending=False
    )
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    best_name = comparison.iloc[0]["model"]
    best_model = fitted_models[best_name]
    best_pred = best_model.predict(X_test)

    metrics = save_evaluation(
        y_test, best_pred, sorted(y.unique()), REPORT_DIR
    )

    joblib.dump(
        {
            "model": best_model,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "classes": sorted(y.unique()),
            "best_model_name": best_name,
        },
        MODEL_PATH
    )

    metadata = {
        "best_model": best_name,
        "n_rows": int(len(df)),
        "n_features": len(FEATURE_COLUMNS),
        "classes": sorted(y.unique()),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "test_metrics": metrics,
    }
    (REPORT_DIR / "training_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nModel comparison:")
    print(comparison.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print("Test metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()

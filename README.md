# Iris Flower Classification — Industry-Style ML Project

A complete machine-learning classification project using the provided Kaggle-style `Iris.csv` dataset.

## Objective
Predict one of three Iris species:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

from sepal and petal measurements.

## Project workflow
1. Load and validate the supplied CSV.
2. Perform exploratory data analysis (EDA).
3. Check missing values and duplicates.
4. Separate features and target.
5. Split into train/test sets using stratification.
6. Build preprocessing + model pipelines.
7. Compare Logistic Regression, KNN, Decision Tree, Random Forest, and SVM.
8. Evaluate accuracy, precision, recall, F1-score, and confusion matrices.
9. Select the best model using cross-validation.
10. Save the trained model with joblib.
11. Provide a reusable prediction script.
12. Provide a Streamlit web app for interactive prediction.

## Dataset
The project uses the exact `Iris.csv` supplied by the user. No synthetic data is generated.

## Quick start

### 1. Create environment
```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the complete training/evaluation pipeline
```bash
python -m src.train
```

This creates:
- `models/iris_best_model.joblib`
- `reports/model_comparison.csv`
- `reports/test_metrics.json`
- `reports/classification_report.txt`
- `reports/figures/confusion_matrix.png`
- `reports/figures/feature_importance.png` (when supported)

### 4. Make a command-line prediction
```bash
python -m src.predict --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2
```

### 5. Launch the web application
```bash
streamlit run app.py
```

## Expected project structure
```text
iris_flower_classification_project/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── Iris.csv
├── models/
├── reports/
│   └── figures/
├── notebooks/
│   └── iris_eda_and_modeling.ipynb
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── train.py
    ├── evaluate.py
    └── predict.py
```

## Notes
- The `Id` column is treated as an identifier and excluded from model features.
- Numeric measurements are standardized where appropriate through scikit-learn pipelines.
- The target column is detected from the supplied dataset (`Species`).
- The application expects the model to have been trained first.

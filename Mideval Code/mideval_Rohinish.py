import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# 1. Load data ----------------------------------------------------------
csv_path = "quantvision_financial_dataset_200.csv"   # adjust path if needed
df = pd.read_csv(csv_path)

target_col = "future_trend"      # label column in your CSV
X = df.drop(columns=[target_col])
y = df[target_col].values

# Identify numeric & categorical columns (from file header)
numeric_features = [
    "lookback_days",
    "technical_score",
    "edge_density",
    "slope_strength",
    "candlestick_variance",
    "pattern_symmetry",
]
categorical_features = [
    "asset_type",
    "market_regime",
    "high_volatility",
    "trend_continuation",
]

# 2. Preprocessor (no leakage) -----------------------------------------
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 3. Train–test split ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Pipelines for both models -----------------------------------------
nn_model = Pipeline([
    ("preprocess", preprocessor),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42
    ))
])

log_model = Pipeline([
    ("preprocess", preprocessor),
    ("clf", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# 5. Fit models ---------------------------------------------------------
nn_model.fit(X_train, y_train)
log_model.fit(X_train, y_train)

# 6. Evaluate NN --------------------------------------------------------
print("=== Neural Network (MLPClassifier) ===")
y_pred_nn = nn_model.predict(X_test)
y_proba_nn = nn_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_nn, digits=3))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred_nn))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_nn))
print()

# 7. Evaluate Logistic Regression --------------------------------------
print("=== Logistic Regression ===")
y_pred_log = log_model.predict(X_test)
y_proba_log = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_log, digits=3))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred_log))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_log))
print()

# 8. Simple comparison table -------------------------------------------
def get_main_metrics(y_true, y_pred, y_proba):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }

nn_metrics = get_main_metrics(y_test, y_pred_nn, y_proba_nn)
log_metrics = get_main_metrics(y_test, y_pred_log, y_proba_log)

comparison_df = pd.DataFrame.from_dict(
    {
        "nn_model": nn_metrics,
        "log_model": log_metrics,
    },
    orient="index"
)

print("=== Final Comparison Table ===")
print(comparison_df)

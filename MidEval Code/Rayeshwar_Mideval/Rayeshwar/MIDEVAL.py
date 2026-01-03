import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#  dataset
data = pd.read_csv("quantvision_financial_dataset_200.csv")

# Separate features and target
X = data.drop("future_trend", axis=1)
y = data["future_trend"]

#Convert Words to  Numbers
label_encoder = LabelEncoder()
X["asset_type"] = label_encoder.fit_transform(X["asset_type"])
X["market_regime"] = label_encoder.fit_transform(X["market_regime"])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

# Neural Network
nn_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=500,
    random_state=42
)
nn_model.fit(X_train, y_train)
nn_pred = nn_model.predict(X_test)

# Evaluation 
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Performance:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# Evaluate
evaluate_model("Logistic Regression", y_test, log_pred)
evaluate_model("Neural Network", y_test, nn_pred)

def get_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }
log_metrics = get_metrics(y_test, log_pred)
nn_metrics = get_metrics(y_test, nn_pred)


comparison = pd.DataFrame([
    {"model": "logmodel", **log_metrics},
    {"model": "nn_model", **nn_metrics}
])

print(comparison)
if nn_metrics["f1_score"] > log_metrics["f1_score"]:
    print("Neuralnetwork model wins")
else:
    print("Logistic regression model wins")




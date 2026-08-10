import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load and Encode
df = pd.read_csv('quantvision_financial_dataset_200.csv')
le = LabelEncoder()
for col in ['asset_type', 'market_regime']:
    df[col] = le.fit_transform(df[col])

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, Y_train)
y_pred_log = log_model.predict(X_test)

# Neural Network
nn_model = MLPClassifier(hidden_layer_sizes=(20,10), max_iter=1000)
nn_model.fit(X_train, Y_train)
y_pred_nn = nn_model.predict(X_test)

def print_metrics(name, y_true, y_pred):
    print(name,':')
    print('Accuracy:',  accuracy_score(y_true, y_pred))
    print('Precision:', precision_score(y_true, y_pred))
    print('Recall:',    recall_score(y_true, y_pred))
    print('F1 Score:',  f1_score(y_true, y_pred))
    print('Confusion Matrix:\n',confusion_matrix(y_true, y_pred),'\n')

print_metrics("Logistic Regression", Y_test, y_pred_log)
print_metrics("Neural Network", Y_test, y_pred_nn)


# Why Logistic Regression performs reasonably good or bad:
# Logistic Regression performs worse than Neural Network probably because it relies on linear weights, ignoring compound effects while training model. Since target variable has very low count of 0s, it might not have influenced the weights much. 

# Why Neural Network performs better or worse:
# It performs better here because its non-linear layers may have extracted hidden patterns that distinguish minority classes.

# The effect of volatility on predictions:
# High volatility causes linear models to misinterpret random price fluctuations as meaningful trend changes.

# The role of trend continuation:
# It acts as a primary momentum signal that both models rely on to achieve high accuracy during stable market conditions.

# Situations where the model fails and why:
# Due to high-volatility, entropy increases. As entropy increases, the distributions of these features begin to spread and overlap, making it mathematically impossible for a model to draw a clean boundary without misclassifying data points.
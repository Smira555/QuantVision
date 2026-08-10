import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import ComplementNB
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Dataset.csv')

# Exploratory Data Analysis

# Taking care of categorical variables by adding dummy collumns
df = pd.get_dummies(df,columns = ['market_regime','asset_type'], drop_first = True)
# Checking for multicolinearity
data = df[['lookback_days','technical_score','edge_density','slope_strength','candlestick_variance','pattern_symmetry']]
plt.figure(figsize=(10, 8))
corr_matrix = data.corr()
sns.heatmap(corr_matrix,annot = True, cmap='coolwarm')
plt.title("Heatmap: ")
plt.show()

# Comparing Future Trends with Volatility and Trend Continuation
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.countplot(data=df, x='high_volatility', hue='future_trend', ax=axes[0])
axes[0].set_title('high_volatility Distribution by Future Trend')

sns.countplot(data=df, x='trend_continuation', hue='future_trend', ax=axes[1])
axes[1].set_title('trend_continuation Distribution by Future Trend')
plt.tight_layout()
plt.show()

# Splitting the data
y = df['future_trend']
X = df.drop('future_trend', axis = 1)
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

# Scaling the data wherever required.
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



print(df.info(),'\n')
print(df.describe())
# Description of data tells us that mean of future trend is very close to 1 and all the quantiles are 1.
# This makes it evident that the data is heavily biased towards 1 in future_trends which would require adjustments to models
# We want the data to train on both 0 and 1 future trends as if it ignores a 0(negative future trend) which was of significance, it could lead to a huge loss.

# Model Training
f1 = [0,0,0]
accuracy = [0,0,0]
precision = [0,0,0]
recall = [0,0,0]
y_prob = [0,0,0]
rocauc = [0,0,0]

# Logistics Model

# Here I tried the model without any constraints and as expected, the confusion matrix shows that the model only predicts 1 for future trend due to biasing.
# regr = linear_model.LogisticRegression(random_state = 42)

# Here I have tried punishing model on not getting a 0 right 5 times as much as not getting a 1 right.
# regr = linear_model.LogisticRegression(class_weight={0:5,1:1}, random_state = 42)

# Here I have tried the Balanced Approach
regr = linear_model.LogisticRegression(class_weight='balanced', random_state = 42)
regr.fit(X_train,y_train)
# Predicting Future Trends
y_pred = regr.predict(X_test)
# Determining Scores and Printing them
cm = metrics.confusion_matrix(y_pred,y_test)
f1[0] = metrics.f1_score(y_pred,y_test)
accuracy[0] = metrics.accuracy_score(y_pred,y_test)
precision[0] = metrics.precision_score(y_pred,y_test)
recall[0] = metrics.recall_score(y_pred,y_test)
y_prob[0]= regr.predict_proba(X_test)[:,1]
rocauc[0] = metrics.roc_auc_score(y_test,y_prob[0])
print("ROC-AUC Score: ", rocauc[0])
print("Logistics Model:")
print('Confusion Matrix: ', cm)
print('F1 Score: ', f1[0])
print('Accuracy: ', accuracy[0])
print('Precision: ', precision[0])
print('recall: ', recall[0])


# Neural Network Model
# Neural Network is used for much larger data, and here it has the risk of overfitting with the data so that needs to be taken care of.
# I have used lbfgs as the solver because it is a very small dataset. I have ensured no overfitting by setting early_stopping to be True.
nn_model = MLPClassifier(hidden_layer_sizes=(16,8), activation = 'relu',max_iter = 1000, early_stopping = True, solver = 'lbfgs',alpha = 0.01, random_state = 42)
nn_model.fit(X_train,y_train)
# Predicting Future Trend
y_pred2 = nn_model.predict(X_test)
# Determining Scores and printing them
cm2 = metrics.confusion_matrix(y_pred2,y_test)
f1[1] = metrics.f1_score(y_pred2,y_test)
accuracy[1] = metrics.accuracy_score(y_pred2,y_test)
precision[1] = metrics.precision_score(y_pred2,y_test)
recall[1] = metrics.recall_score(y_pred2,y_test)
y_prob[1]= nn_model.predict_proba(X_test)[:,1]
rocauc[1] = metrics.roc_auc_score(y_test,y_prob[1])
print("ROC-AUC Score: ", rocauc[1])
print("Neural Network Model:")
print('Confusion Matrix: ', cm2)
print('F1 Score: ', f1[1])
print('Accuracy: ', accuracy[1])
print('Precision: ', precision[1])
print('recall: ', recall[1])

# using Complement of Baye's Naive since the data is heavily skewed. 
# It basically judges what is not correct (opposite of what models generally do) so it is a good choice during such biases.
cnb = ComplementNB()
cnb.fit(X_train,y_train)
# Predicting Future Trend
y_pred3 = cnb.predict(X_test)
# Determining Scores and printing them
cm3 = metrics.confusion_matrix(y_pred3,y_test)
f1[2] = metrics.f1_score(y_pred3,y_test)
accuracy[2] = metrics.accuracy_score(y_pred3,y_test)
precision[2] = metrics.precision_score(y_pred3,y_test)
recall[2] = metrics.recall_score(y_pred3,y_test)
y_prob[2]= cnb.predict_proba(X_test)[:,1]
rocauc[2] = metrics.roc_auc_score(y_test,y_prob[2])
print("ROC-AUC Score: ", rocauc[2])
print("Neural Network Model:")
print('Confusion Matrix: ', cm3)
print('F1 Score: ', f1[2])
print('Accuracy: ', accuracy[2])
print('Precision: ', precision[2])
print('recall: ', recall[2])

# Comparison of Models
compare = {
    'model':['Logistic Regression', 'Neural Network',"Complement Naive Bayes'"],
    'accuracy':accuracy,
    'precision':precision,
    'F1 Score': f1,
    'Recall': recall,
    'ROC-AUC score:': rocauc
}

print(pd.DataFrame(compare))

# The Logistics Regression is performing relatively slightly worse in comparison to Neural Network based on the scores.
# But, the scores are giving a false confidence to the models as the predicted values are heavily biased towards 1.
# Logistic Model has been adjusted so as to keep a balanced approach for 0 and 1 so it is infact learning 0's better as compared to NN
# The Scores here are merely an indication as to which model predicted the least amount of zeroes and it is evident from the confusion matrix as well.
# The So Called Scores are highly misleading, which brings in the need to use multiple models for testing data.

# Most models will fail miserabely for a testing data set biased towards 0 except for Complement Naive Bayes.

# Volatility and trend Continuation have a relation with future trend as can be seen from their frquency charts and do have some predictive power.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
from sklearn.metrics import confusion_matrix

#####-------------------------1.PRE-PROCESSING------------------######

## READING FROM THE CSV FILE
data=pd.read_csv('quantvision_financial_dataset_200.csv')

## TAKING X(input data) and Y(output/target values)
X=data.drop('future_trend',axis=1) ##Taking all rows and columns in the dataframe except the last column
Y=data['future_trend'] ##Taking the target column(0/1)

## ENCODING THE CATEGORICAL DATA & SCALING ALL THE NUMERICAL DATA in THE RANGE [-1,1]
data.columns=data.columns.str.strip()
categorical_columns=['asset_type','market_regime']
numerical_columns=['lookback_days','high_volatility','trend_continuation','technical_score','edge_density','slope_strength','candlestick_variance','pattern_symmetry']

preprocessor=ColumnTransformer(
    transformers=[('numerical',StandardScaler(),numerical_columns),
                  ('textdata',OneHotEncoder(),categorical_columns)]
)##The preprocessor object is created

X_processed=preprocessor.fit_transform(X) ##++++This processed data can be used to train models like Logistic Regression and neural network++++++

##Spliting training and testing dataset with 80-20
X_train,X_test,Y_train,Y_test=train_test_split(X_processed,Y,test_size=0.2,random_state=45,stratify=Y)
print(f'Training Data set: {X_train.shape}')
print(f'Testing Data set: {X_test.shape}')

######-------------END OF PRE-PROCESSING----------------######



####----------------2.MODEL TRAINING--------------######

##TRAINING LOGISTIC REGRESSION MODEL USING SCIKIT-LEARN
logistic_model=LogisticRegression(random_state=45)
logistic_model.fit(X_train,Y_train) ##FEEDING DATA INTO MODEL

##TRAINING NEURAL NETWORK MODEL
mlp_model=MLPClassifier(hidden_layer_sizes=(64,32),activation='relu',max_iter=700,random_state=45)
mlp_model.fit(X_train,Y_train)

#######-----------END OF MODEL TRAINING---------#######


#####----------------3.MODEL EVALUATION----------------#####

def eval_model(model,x,y,model_name):
    Y_predict=model.predict(x)

    acc=accuracy_score(y,Y_predict)
    prec=precision_score(y,Y_predict)
    recall=recall_score(y,Y_predict)
    f1=f1_score(y,Y_predict)

    print(f'###-------{model_name} EVALUATION---------###')
    print(f'Accuracy: {acc: 0.4f}')
    print(f'Precision: {prec: 0.4f}')
    print(f'Recall: {recall: 0.4f}')
    print(f'F1-Value: {f1: 0.4f}')
    print(classification_report(y,Y_predict)) ### TESTING HOW ACTUALLY THE MODEL PERFORMS ON 0s and 1s seperately

    print("CONFUSION MATRIX")
    print(confusion_matrix(y,Y_predict))###CONFUSION MATRIX
    return f1

f1_log=eval_model(logistic_model,X_test,Y_test,"LOGISTIC REGRESSION MODEL")
f1_mlp=eval_model(mlp_model,X_test,Y_test,"NEURAL NETWORK MODEL")

if(f1_log>f1_mlp):
    print("Logistic Regression model predicts better")
else:
    print("Neural Network(MLP) model predicts better")

#########--------------END OF MODEL EVALUATION---------#######




########-------------------------------4.FINANCIAL INTERPRETATION-----------------------###

### 1.Why Logistic Regression performs reasonably good or bad:
    ##Though the overall F1-value of the model is reasonably good(0.9474) still it has some major issues.
    ##One major reason is that the dataset by which it was trained was majorly bullish,and the model just analysed one particular column(trend_continuation),and assumed it has direct relationship with the output value.
    ##So it failed to analyse the relationship between multiple columns.
    ##From the confusion matrix and classification report it is clear that model is so bad at predicting downtrends as it didn't predict any of the 3 downtrends.
    ##As our testing dataset also contained mostly uptrends,the overall preicision,recall,accuracy was reportedly good(above 90%).


### 2.Why neural networks perform better or worse:
    ##From the evaluation comparison of both models we can clearly identify that neural network model performed better than logistic regression model(by comparing F1-value).
    ##This is so because the neural network model became too lazy to identify the reason for 0s in the future_trend columns,as it was trained on dataset with mostly 1s.
    ##This made the model to follow "Permanent Bullish" strategy,which means no matter what the trends and data we give,the predicted future_trend will be 1.
    ##As so the testing dataset also contained mostly 1s(37) and only 3 0s, the model's strategy was kind of rewarding as it predicted 37/40 correctly,with a overall F1-value of 0.9610.
    ##So overall this model is very bad at predicting crashes which clearly visible from the confusion matrix and classification report.


### 3.Effect of volatility on predictions:
    ##Usually volatility is used a precaution/danger signal for the prediction.
    ##But however since both our models was trained on majorly 1s , it just ignored the volatility data and didnt bother how it affects the future prediction,rather the model was heaviy weighted
    #to trend_continuation attribute.So if the momentum is high,the models just predicted the stock will rise in the future and vice-versa.


### 4.The role of trend_continuation:
    ##This was the major attribute that both models focused upon,and ignored the correlations and other nuances of other attributes.
    ##The model just memorized that if the stock is UP before , then it will go up in the future also and vice-versa for downtrend.


### 5.Situations where the model failed:
    ##The severe case where both the models failed is in predicting downtrends,which why both got 0s in true negative in confusion matrix.
    ##This is mainly because of the highly imbalanced data training set,as it contained mostly Uptrends.
    ##The model never truely learnt what a crash or severe downtrend looks like,as it only saw majorly uptrends(1s).

######--------------------------END OF FINANCIAL INTERPRETATION----------------------------#########

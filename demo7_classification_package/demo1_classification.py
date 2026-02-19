import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import seaborn as sns

# read csv using pandas and print df.head()

df=pd.read_csv("files/Titanic-Dataset.csv")
# print(df.head())
print(df["Age"].mean())

df=df[["Survived","Pclass","Age","Fare"]]

# print(df.head())
# check for missing values and update with mean
print(df.isnull().sum()) 
# fillna --> finds replace all nan/empty values to given value
df["Age"]=df["Age"].fillna(df["Age"].mean())
print(df.isnull().sum()) 

# categorical variable --> we need to convert into numerical variable 
df=pd.get_dummies(df,drop_first=True)

# print(df.head())

# load independent variable or feature X
X=np.array(df.drop("Survived",axis=1))
# print(X)
# load target column y
y=np.array(df["Survived"])
# print(y)
# split train and test
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=1)

# LogisticRegression object 
model=LogisticRegression()


# use obj ref and call fit - train the model
model.fit(X_train,y_train)


y_pred=model.predict(X_test)

print(y_pred)
print(y_test)
print(y_pred[0])
print(y_test[0])

# Evals
# Confusion matrix 
"""
--> true --> it means predicted=actual

[[TN FP 
  FN TP]]

"""
from sklearn.metrics import confusion_matrix,precision_score,recall_score
import matplotlib.pyplot as plt

cm=confusion_matrix(y_test,y_pred)
print(cm)

sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.show()

"""
1-> survived 
0 -> not survived
--> true --> it means predicted=actual
TP --> predicted = survived (1) , actual = survived (1) 
TN --> predicted = non survived (0) , actual = not survived (0) 

FP --> predicted = survived (1) ,  actual = not survived (0) 
FN --> predicted = non survived (0),  actual = survived (1) 

"""

"""
precision --> out of all predicted as survived, how many actually survived?
--> true positive out of predicted postivities 
--> quality of the model 
precision = TP/(TP+FP)

recall --> out of all actual survivor, how many the model correctely detected?
--> true positive out of real positive 
--> quanitity of the model 

--> out of all  all actual survivor --> only 41% were correctly identified by model 

recall = TP/(TP+FN)

"""

precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
print(precision)
print(recall)


# trade off prec-recall
y_prob=model.predict_proba(X_test)
print(y_prob)


# roc-auc
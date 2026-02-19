import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# read csv using pandas and print df.head()

df=pd.read_csv("files/Titanic-Dataset.csv")
# print(df.head())

df=df[["Survived","Pclass","Sex","Age","Fare"]]

# print(df.head())
# check for missing values and update with mean
print(df.isnull().sum()) 
df["Age"]=df["Age"].fillna(df["Age"].mean())
print(df.isnull().sum()) 

# categorical variable --> we need to convert into numerical variable 
df=pd.get_dummies(df,drop_first=True)

# print(df.head())

# load independent variable or feature X
X=np.array(df.drop("Survived",axis=1))

# load target column y
y=np.array(df["Survived"])

# split train and test
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=1)

# LogisticRegression object 
model=LogisticRegression()


# use obj ref and call fit - train the model
model.fit(X_train,y_train)


y_pred=model.predict(X_test)

print(y_pred)
print(y_pred[0])
print(y_test[0])

# Evals
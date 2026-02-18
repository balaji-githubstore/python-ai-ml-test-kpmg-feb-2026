import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# read csv using pandas and print df.head()

df=pd.read_csv("files/sale_prices_practice.csv")
print(df.head())

# (features /indpendent variable)
X=np.array(df[["OverallQual","GrLivArea","GarageCars","TotalBsmtSF",
               "YearBuilt","FullBath","BedroomAbvGr","LotArea"]])

# print(X)

# y(target column/dependent variable)
y=np.array(df["SalePrice"])
# print(y)


# 80% training (X_train, y_train)
# 20 % testing (X_test, y_test)
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=1)


# object creation to call fit() method
reg=LinearRegression()

# train the model
model=reg.fit(X_train,y_train)


# predicting y_pred on the test set 
y_pred= model.predict(X_test)

# compare y_pred (value predicted by model) vs y_test (actual value from dataset) 

print("Prediction \n")
print(y_pred[0])

print("Actual \n")
print(y_test[0])









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


print(X_test)
# predicting y_pred on the test set 
y_pred= model.predict(X_test)


# QA should know below details 
# compare y_pred (value predicted by model) vs y_test (actual value from dataset) 
# print(y_pred)

# print("Prediction")
# print(y_pred[0])

# print("Actual")
# print(y_test[0])

# Eval set 1
np.set_printoptions(suppress=True,precision=2)
# tell us how much each feature (column) affect the target 
# 1.20699441e+04  --> 
print(model.coef_)

# assuming feature column = 0 , what is the base value?  
# if postive, need to check why sales positive when all feature = 0. Rasie a question on model prediction. 
# y= intercept+coef(feature)*1560+coef(feature)*1560+... 
print(model.intercept_)


# Eval set 2 
"""
MAE - Mean absolute error 
- on average, the predicted house price differ from actual price by 9923. 
but it does not whether the model tends to overprecit or underpredict 
QA view: low value --> prediction are close to actuals 
QA View --> size of error for each house 

act1 = 2,00,000 
pred1 = 2,10,000

act2 = 2,00,000 
pred2 = 1,80,000

(10,000+20,000)/2=15,000

MSE - Mean squared error --> mse means high value prediction may have outliers 

act1 = 2,00,000 
pred1 = 2,10,000

act2 = 2,00,000 
pred2 = 1,80,000

((-10000)^2 + (20,000))^2/2

RMSE
QA view => 
RMSE = MAE ==> error are fairly uniform 
RMSE > MAE ==> model has some large errors 

R2
how much data variance the model explains? higher is better
QA view --> close to 1 -> good (higher is better)

"""

from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

mae=mean_absolute_error(y_true=y_test,y_pred=y_pred)
print(mae)

mse=mean_squared_error(y_true=y_test,y_pred=y_pred)
print(mse)

rmse=np.sqrt(mse)
print(rmse)

r2=r2_score(y_true=y_test,y_pred=y_pred)
print(r2)


"""
Residual = actual - predicted 

>0 or positive residual --> underpredicted 
<0 or negative residual --> overpredicted 

AQ View:
Good - random scatter around 0 
Wrong - pattern - model may be missing something

"""

residuals=y_test-y_pred

print(residuals)

print(y_test[0])
print(y_pred[0])
print(y_test[0]-y_pred[0])



# scatter plot -   predicted vs residual
import matplotlib.pyplot as plt

plt.scatter(y_pred,residuals)  
plt.axhline(y=0)
plt.xlabel("predicted")
plt.ylabel("residuals")

plt.show()

# pred - x axis (y_pred) vs actual - y axis (y_test)

plt.scatter(y_pred,y_test)  
plt.xlabel("predicted")
plt.ylabel("actual")

plt.show()

# will start at 11:35 AM IST
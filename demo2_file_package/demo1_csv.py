import pandas as pd


df=pd.read_csv(filepath_or_buffer="files/data.csv",delimiter=",")
print(df)
print(df.head())
print(df.tail())
print(df.describe())
print("*"*50)

# cleaning missing values
# column wise missing value
# fillna --> finds replace all nan values to given value
print(df.isnull().sum())
print(df.isnull().sum().sum())
df["Age"]=df["Age"].fillna(df["Age"].mean())
print(df.isnull().sum())

# removing duplicates 
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
print(df.duplicated().sum())

print(df)
print("*"*50)
print(df.describe())

print("*"*50)
# age --> find lower and upper bound using IQR 
# Quantile --> asc order and divide the age into 4 set (4 quantile - 25% each)
Q1=df["Age"].quantile(0.25)
Q3=df["Age"].quantile(0.75)

IQR=Q3-Q1
lower_bound_age = Q1- 1.5*IQR
upper_bound_age = Q3+1.5*IQR

print(lower_bound_age)
print(upper_bound_age)
print("-"*50)
# print outlier (those rows (age) less than lower bound and rows (age) which are more than upper bound  )
outliers_df=df[(df["Age"]<lower_bound_age) | (df["Age"]>upper_bound_age)]
print(outliers_df)
print("-"*50)

# salary --> find lower and upper bound using IQR 
# Q1=df["Salary"].quantile(0.25)
# Q3=df["Salary"].quantile(0.75)

# IQR=Q3-Q1
# lower_bound_salary = Q1- 1.5*IQR
# upper_bound_salary = Q3+1.5*IQR

# print(lower_bound_salary)
# print(upper_bound_salary)




# update df --> with age lower and upper bound (remove records outside this)
print(df)

df=df[(df["Age"]>=lower_bound_age) & (df["Age"]<=upper_bound_age)]
print(df)

print(df.describe())

# rows and column count 
print(df.shape)

print("-"*50)
print("-"*50)

print(df)
df_sorted=df.sort_values(by="Age")
print(df_sorted)


import matplotlib.pyplot as plt

# line plot - sort is required
plt.scatter(df_sorted["Age"],df_sorted["Salary"])  
plt.title("Line plot example")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.savefig("plot.png")
plt.show()



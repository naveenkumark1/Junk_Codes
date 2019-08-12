# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 05:55:42 2019

@author: Admin
"""

#importing important packages
import pandas as pd
import numpy as np

#reading the dataset
df=pd.read_csv(r"C:\Users\naveen.reddy\Desktop\Ensemble\Ensembles\train.csv")

#filling missing values
df['Gender'].fillna('Male', inplace=True)

nan_rows = df[df['Loan_ID'].isnull()]

df['Married'].fillna(df['Married'].mode()[0], inplace=True)

df['Married'].mode()[0]

#'Dependents',
df['Dependents']=df['Dependents'].replace('3+','4')

type(df['Dependents'])

df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)

df.describe()
df.info()

#'Education',
df['Education'].fillna(df['Education'].mode()[0], inplace=True)

df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)

df['ApplicantIncome'].fillna(df['ApplicantIncome'].mean(),inplace=True)

sum(df['ApplicantIncome'].isna())

df['CoapplicantIncome'].fillna(df['CoapplicantIncome'].mean(),inplace=True)

#'LoanAmount',

df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace = True)

#'Loan_Amount_Term',

df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(),inplace=True)

#'Credit_History',

df['Credit_History'].fillna(df['Credit_History'].mean(),inplace=True)

#'Property_Area',

df['Property_Area'].fillna(df['Property_Area'].mode()[0],inplace=True)

#'Loan_Status'

df['Loan_Status'].fillna(df['Loan_Status'].mode()[0],inplace=True)

df.isnull().sum()

df['Dependents'].isnull().sum()
 
#split dataset into train and test

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.3, random_state=0)

np.shape(df)

np.shape(test)
np.shape(train)

x_train=train.drop('Loan_Status',axis=1)
#x_train=x_train.drop('Property_Area',axis=1)
np.shape(x_train)
y_train=train['Loan_Status']
#y_train=y_train['Property_Area']

np.shape(y_train)

x_test=test.drop('Loan_Status',axis=1)
#x_train=train.drop('Property_Area',axis=1)
np.shape(x_test)
y_test=test['Loan_Status']
#y_test=test['Property_Area']

np.shape(y_test)

df.info()

df["Loan_Status"]

#catcol = df[["Loan_ID","Gender","Married","Dependents","Education","Self_Employed","Property_Area","Loan_Status"]]

catcol = df[["Gender","Married","Dependents","Education","Self_Employed","Property_Area"]]
numcol = df[["ApplicantIncome",'CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']]

# train 

catcol.columns.values
x_train.columns.values

list(df.columns.values)
catVars = list(catcol.columns.values)
numVars = list(numcol.columns.values)

catDf = x_train[catVars]
catDummies = pd.get_dummies(catDf)
colDummies =list(catDummies.columns)

totalTrain =pd.concat([x_train[numVars],catDummies],axis = 1)

##test
catDummiesTest = pd.get_dummies(x_test[catVars])
newCatDummiesDf=pd.DataFrame()
for col in colDummies:
    if col in list(catDummiesTest.columns):
        newCatDummiesDf[col] = catDummiesTest[col]
        
    else:
        newCatDummiesDf[col] = 0
    
totalTest =pd.concat([x_test[numVars],newCatDummiesDf],axis = 1)

#create dummies
#x_train=pd.get_dummies(x_train)

#x_test=pd.get_dummies(x_test)

## Random Forest 

from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(random_state=1)


model.fit(totalTrain, y_train)
model.score(totalTest,y_test)

### Variable importance 

for i, j in sorted(zip(totalTrain.columns, model.feature_importances_)):
    print(i, j)

#Regressor    
from sklearn.ensemble import RandomForestRegressor
model= RandomForestRegressor()
model.fit(totalTrain, y_train)
model.score(totalTest,y_test)    

### Boosting : Xgboost 

import xgboost as xgb
model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
model.fit(x_train, y_train)
model.score(x_test,y_test)

# Regression 

import xgboost as xgb
model=xgb.XGBRegressor()
model.fit(x_train, y_train)
model.score(x_test,y_test)


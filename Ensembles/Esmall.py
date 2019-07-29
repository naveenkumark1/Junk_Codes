# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 05:55:42 2019

@author: Admin
"""

#importing important packages
import pandas as pd
import numpy as np

#reading the dataset
df=pd.read_csv(r"C:\Users\Admin\Desktop\ET\Py_Blr\MLP\Ensembles\train_clean.csv")

#filling missing values
#df['Gender'].fillna('Male', inplace=True)

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
list(df.columns.values)

#create dummies
x_train=pd.get_dummies(x_train)
x_test=pd.get_dummies(x_test)

## Random Forest 

from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(random_state=1)

model.fit(x_train, y_train)
model.score(x_test,y_test)

### Variable importance 

for i, j in sorted(zip(x_train.columns, model.feature_importances_)):
    print(i, j)

#Regressor    
from sklearn.ensemble import RandomForestRegressor
model= RandomForestRegressor()
model.fit(x_train, y_train)
model.score(x_test,y_test)    

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


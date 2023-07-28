# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 13:57:16 2023

@author: tamer
"""

#1. libraries
import pandas as pd

#2. data loading
data = pd.read_csv("breast_cancer.csv")

#3. preprocessing
data = data.drop(["id", "Unnamed: 32"],axis=1)
x = data.iloc[:,1:]
y = data.iloc[:,0]

#3.1 label encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y) 

#4. dropping columns with low correlation
data = data.drop("diagnosis",axis=1)
data['diagnosis'] = y

correlations_with_target = abs(data.corr()['diagnosis'].drop('diagnosis'))
correlations_with_target = correlations_with_target.sort_values()

drop_list = ["symmetry_se","texture_se","fractal_dimension_mean","smoothness_se","fractal_dimension_se"]
x = data.drop(drop_list,axis=1)

#5. splitting data for training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#6. XGBoost
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(x_train, y_train)
y_pred = xgb.predict(x_test)

#7. Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)
print(cm)

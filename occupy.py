# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error

data = pd.read_csv(r"C:\Users\HP-PC\Desktop\data_science\ML_Exrcise\datasets\occupy.csv")
data.head()
print(data.sample(3))
print(data.info())
data = data.drop("date",axis = 1)
print(data.info())

y = data.Occupancy
X = data.drop("Occupancy",axis =1)

X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                    test_size=0.1, 
                                                    random_state=123, 
                                                    stratify=y)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, 
                                                    test_size=0.1, 
                                                    random_state=123, 
                                                    stratify=y_train)

print(X_train.sample(3))
print(X_test.sample(3))
print(y_train.sample(3))
print(y_test.sample(3))
print(X_train.info())
print(X_test.info())
print(X_val.info())

def light():
    occupy1 = data[data['Occupancy']==1]['Light'].value_counts()
    occupy0 = data[data['Occupancy']==0]['Light'].value_counts()
    DF = pd.DataFrame([occupy1,occupy0])
    DF.index = ['occupied','unoccupied']
    DF.plot(kind='bar',stacked=True)
    
#light()

def logistic():
    model_logistic = LogisticRegression()
    model_logistic.fit(X_train,y_train)
    train_acuracy = round(model_logistic.score(X_train,y_train),2)
    val_acc = round(model_logistic.score(X_val,y_val),2)
    Y_predL = model_logistic.predict(X_test)
    print(train_acuracy)
    print(val_acc)
    print(mean_squared_error(y_test,Y_predL))
    print(y_test[:10])
    print(Y_predL[:10])
    
def rf():
    RF = RandomForestClassifier()
    RF.fit(X_train,y_train)
    train_acuracy = round(RF.score(X_train,y_train),2)
    val_acc = round(RF.score(X_val,y_val),2)
    Y_predRF = RF.predict(X_test)
    print(train_acuracy)
    print(val_acc)
    print(mean_squared_error(y_test,Y_predRF))
    print(y_test[:10])
    print(Y_predRF[:10])
    
logistic()
rf()
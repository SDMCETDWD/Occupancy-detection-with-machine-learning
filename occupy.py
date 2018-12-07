
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

# load the data set using the pandas attribute read_csv we can read variety of data formates using pandas.
data = pd.read_csv(r"(path-to-the-folder-containing-the-datasetfile)\occupy.csv")
data.head()
# Take a look at the data by printing the sample 3 examples 
print(data.sample(3))
# Get the information about the data to check for any null enrty and to know the data type
print(data.info())

# Examine dataset for the depencies
# Drop the feature which does not contribute to the output
# As in our dataset the feature date has no contribution towards the output occupancy 
# Drop the date using .drop attribute in the parenthesis axis = 1 to indicate the drop function is for the column  
data = data.drop("date",axis = 1)
# check whether the column is removed or not
print(data.info())

# In our dataset Occupancy is the output so seperate the column and put it in y
y = data.Occupancy

# Put the features in X and drop the output column 
X = data.drop("Occupancy",axis =1)

# Split the data set into train,test and validation dataset
# We use the train_test_split fution to split the dataset into train and validation dataset for the 1st call
# In the 2nd call the remaining dataset is split into train and test dataset
# Our goal is to split the dataset train : val : test approximately in the ration 8:1:1

X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                    test_size=0.1, 
                                                    random_state=123, 
                                                    stratify=y)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, 
                                                    test_size=0.1, 
                                                    random_state=123, 
                                                    stratify=y_train)


# Take look at the training, test and validation dataset informations, counts of dataset examples in each set 
print(X_train.info())
print(X_test.info())
print(X_val.info())

# In case of doubt about any feature's contribution towards the output plot the feature and output  
def light():
    occupy1 = data[data['Occupancy']==1]['Light'].value_counts()
    occupy0 = data[data['Occupancy']==0]['Light'].value_counts()
    DF = pd.DataFrame([occupy1,occupy0])
    DF.index = ['occupied','unoccupied']
    DF.plot(kind='bar',stacked=True)
    
# Call the program only to clear the doubt and later we can delete the function    
light()

# Choose any classification algorithms to start with

def logistic():
    model_logistic = LogisticRegression()
    # Fit the training set to the model
    model_logistic.fit(X_train,y_train)
    # Evaluate the model on training set and validation set
    train_acuracy = round(model_logistic.score(X_train,y_train),2)
    val_acc = round(model_logistic.score(X_val,y_val),2)
    # Predict the output of the test dataset 
    Y_predL = model_logistic.predict(X_test)
    # Print the score of model on training and test dataset
    print(train_acuracy)
    print(val_acc)
    # Find the difference between the predicted and the actual value of Y test
    print(mean_squared_error(y_test,Y_predL))
    # Compare the sarting 10 values of both actual and predicted values of Y test
    print(y_test[:10])
    print(Y_predL[:10])
    
def rf():
    RF = RandomForestClassifier()
    # Fit the training set to the model
    RF.fit(X_train,y_train)
    # Evaluate the model on training set and validation set
    train_acuracy = round(RF.score(X_train,y_train),2)
    val_acc = round(RF.score(X_val,y_val),2)
    # Predict the output of the test dataset 
    Y_predRF = RF.predict(X_test)
    # Print the score of model on training and test dataset
    print(train_acuracy)
    print(val_acc)
    # Find the difference between the predicted and the actual value of Y test
    print(mean_squared_error(y_test,Y_predRF))
    # Compare the sarting 10 values of both actual and predicted values of Y test
    print(y_test[:10])
    print(Y_predRF[:10])
# Run both the classifiers
logistic()
rf()
# Select the one with best scores

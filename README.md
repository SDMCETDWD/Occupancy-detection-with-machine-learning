# Occupancy-detection-with-machine-learning
Machine learning code to decide whether the room is empty or not based on the features like Temperature, Humidity, Light, CO2, HumidityRatio

Download the dataset [hear](https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+#)

!!! CODE INSIGHT!!!

Line 2-9 : Import all the required libraries and modules 
       [More datails about the SKlearn library](https://scikit-learn.org/stable/user_guide.html)

Download the dataset [hear](https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+#)
Line 12  : Load the downloaded dataset with pandas read_csv() method
       [All about Pandas](https://pandas.pydata.org/pandas-docs/stable/)

Line 15,17,23: .sample() method is used to print the starting 3 rows of the dataset
               .info() method is used to print the information about the features in the dataset
               .drop() method is udes to drop a column from the dataset axis = 1 indicates a column should be dropped

Line 38-43 : train_test_split() is a method in sklearn.model_selection splits the dataset into training and test datasets, this method                splits the dataset in such a way that each section has uniform distribution of all classes [more about train test split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) 
             This method is called twice to devide the dataset as training, validation and test sets.
 
We can always visualise how the feature and the output are related and desirable changes can be made to the feature before feeding it to the training model, if there is no relation than that feature can be dropped,

Line 55 : light() plots the gragh indicating the relation between feature light and the output, call it once to understand the relation

Line 68 : logistic() uses LogisticRegression method from sklearn.linear_model [All about linear models](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
The output of the Logistic funstion:
* train_accuracy = 0.99
* val_accuracy = 0.99
* mean_squared_error between predicted and actual Y_test = 0.0075

Line 85 : rf() uses RandomForestClassifier method from sklearn.ensemble [All about Ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html) 
The output of the RandomForestClassifier funstion:
* train_accuracy = 1.0
* val_accuracy = 1.0
* mean_squared_error between predicted and actual Y_test = 0.0070


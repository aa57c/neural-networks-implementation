# Tutorial from:
# https://medium.com/analytics-vidhya/implementation-of-artificial-neural-network-in-python-step-by-step-guide-556d066f9f5b

# data set from:
# https://www.kaggle.com/datasets/shubh0799/churn-modelling?resource=download

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense
# loading dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# split dataset into independent (X) and dependent (Y) variables
# aka what are our inputs and what are we finding?
X = pd.DataFrame(dataset.iloc[:, 3:13])
Y = dataset.iloc[:, 13]

# output of split dataset
# X_OUTPUT = X.to_csv('X_OUTPUT.csv', header=True, index=False)
# Y_OUTPUT = pd.DataFrame(Y).to_csv('Y_OUTPUT.csv', header=True, index=False)


# encode categorical data (might not need this in assignment since it is already coded into integers
# but... let's do it anyway

# label encoding for gender variable
labelencoder_X_2 = LabelEncoder()
X["Gender"] = labelencoder_X_2.fit_transform(X["Gender"])

# X_OUTPUT = X.to_csv('X_OUTPUT.csv')


# one hot encoding for geography (might not need this for assignment)
# labelencoder_X_1 = LabelEncoder()
# X["Geography"] = labelencoder_X_1.fit_transform(X["Geography"])
# ct = make_column_transformer((X.loc[:, 1], OneHotEncoder()))
ct = ColumnTransformer([("Geography", OneHotEncoder(), [1])], remainder='passthrough')
X = ct.fit_transform(X)

# X_OUTPUT = pd.DataFrame(X).to_csv('X_OUTPUT.csv')

# split the X and Y datasets into training and test sets
# 80-90% of my data should be in training tests, hence test size is 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# feature scale balance and estimated salary
# (normalization - WILL DEFINITELY NEED THIS FOR ASSIGNMENT)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# building the artificial neural network
# activation function for hidden layers should be the Rectifier Activation function ('relu')
# input layer (input-dim) has 11 neurons (13 for assignment)
classifier = Sequential()
classifier.add(Dense(6, activation='relu', input_dim=12))
classifier.add(Dense(6, activation='relu'))
# activation layer - use tanh for assignment, this tutorial uses sigmoid
classifier.add(Dense(1, activation='sigmoid'))

# training the network

# compile network
# uses stochastic gradient descent (using adam from Tensorflow)
# updates weights during training and reduces loss (try some other ones for this assignment)
# loss function is binary, because we are doing binary prediction (should we use this for assignment?)
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fit the network to the training set
# comparing prediction results in batches of 10
# training for 100 epochs to improve accuracy
classifier.fit(X_train, Y_train, validation_split=0.3, batch_size=10, epochs=100)

# predict test set results
# Y_pred > 0.5 means if the value is in between 0 to 0.5, then this new Y_pred
# will become O (False). If larger than 0.5, the new prediction will become 1(True)
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

# validate predictions (using confusion matrix,
# might use this for assignment since we have a large dataset like this one)
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
print(accuracy_score(Y_test, Y_pred))








'''
For all versions, I used these python libraries, just to make the process of making layers easier

This is version 1 of my network.
I used a total of 4 layers, including input and output layers
Hidden layers have 6 neurons total
I normalized all the data
Loss function used was binary_crossentropy (which is the standard since I am only predicting two values
(artery disease present or not)
number of times I train it is 100 with a batch size of 10
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# load dataset

dataset = pd.read_csv('data/processed_cleveland.csv')

# split dataset into X and Y (independent vars or inputs, dependent var or output)

X = pd.DataFrame(dataset.iloc[:, 0:13].values)
Y = dataset.iloc[:, 13].values

# Testing to see if data saved correctly
# X_CSV = X.to_csv('data/X_CSV.csv', index=False, header=False)
# Y_CSV = pd.DataFrame(Y).to_csv('data/Y_CSV.csv', index=False, header=False)

# split the X and Y Dataset into Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# normalize values
# (for this version of the network, I normalized all columns)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Test to see if normalization works
# X_TRAIN_CSV = pd.DataFrame(X_train).to_csv('data/X_TRAIN_CSV.csv')
# X_TEST_CSV = pd.DataFrame(X_test).to_csv('data/X_TEST_CSV.csv')

# building the network

# initialize network
network = Sequential()

# adding input layer and first hidden layer
# fully completed layers have the Rectifier Activation Function
network.add(Dense(6, activation='relu', input_dim=13))
# adding second hidden layer
network.add(Dense(6, activation='relu'))
# adding output layer, Activation Function is Hyperbolic Tangent
network.add(Dense(1, activation='tanh'))

# training the network

# compile network
# optimizer = stochastic gradient descent
# loss = binary because we are testing to see if patient has a presence of artery disease or not
# metrics = accuracy because i want to display the accuracy of such prediction
network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fit the network to the training set
network.fit(X_train, Y_train, batch_size=10, epochs=100)

# predict the test set results
Y_pred = network.predict(X_test)
# if the accuracy score is from 0 to 0.5 for the epoch, then Y_pred will be set to False
# if accuracy score is above 0.5, then the Y_pred will be set to True
Y_pred = (Y_pred > 0.5)

Y_Prediction_CSV = pd.DataFrame(Y_pred).to_csv('data/Y_Prediction.csv')

cm = confusion_matrix(Y_test, Y_pred)
print(cm)
print("How accurate is the network at predicting artery disease?: %f" % accuracy_score(Y_test, Y_pred))

















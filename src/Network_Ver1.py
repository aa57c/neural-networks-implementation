"""
For all versions, I used these python libraries, just to make the process of making layers easier

This is version 1 of my network.
I used a total of 4 layers, including input and output layers
Hidden layers have 5 neurons total
I normalized all the data
Loss function used was binary_crossentropy (which is the standard since I am only predicting two values
(heart disease present or not)
number of times I train it is 100 with a batch size of 10
"""


import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

# load dataset

dataset = pd.read_csv('data/processed_cleveland.csv')

# split dataset into X and Y (independent vars or inputs, dependent var or output)

X = pd.DataFrame(dataset.iloc[:, 0:13])
Y = dataset.iloc[:, 13]

# Testing to see if data saved correctly
X_CSV = X.to_csv('data/Input_Output_Data/Ver1/X_CSV.csv', index=False)
Y_CSV = pd.DataFrame(Y).to_csv('data/Input_Output_Data/Ver1/Y_CSV.csv', index=False)

# one hot encode chest pain type column
ct = ColumnTransformer([("cp", OneHotEncoder(), [2])], remainder='passthrough')
X = ct.fit_transform(X)

# change dependent variable to match dictionary description
'''
Note for Prof Hare:

I don't know if this is allowed since I am messing with the dependent variable data, but I couldn't
get my network to classify in categorical data and then convert that to binary data depending on 
if there is presence of heart disease. So I changed the values a bit to make it easier. 0 means
no heart disease, and then if it is greater than 0, there is presence of heart disease
'''

Y = Y.replace(to_replace=[1, 2, 3, 4], value=1)

# split the X and Y Dataset into Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# printing out test and training data for X and Y (for testing)
pd.DataFrame(X_train).to_csv('data/Train_Data/Ver1/X_train_CSV.csv', index=False)
pd.DataFrame(X_test).to_csv('data/Test_Data/Ver1/X_test_CSV.csv', index=False)
pd.DataFrame(Y_train).to_csv('data/Train_Data/Ver1/Y_train_CSV.csv', index=False)
pd.DataFrame(Y_test).to_csv('data/Test_Data/Ver1/Y_test_CSV.csv', index=False)


# normalize values
# (for this version of the network, I normalized all columns)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Test to see if normalization worked
pd.DataFrame(X_train).to_csv('data/Train_Data/Ver1/X_TRAIN_NORMALIZED.csv', index=False)
pd.DataFrame(X_test).to_csv('data/Test_Data/Ver1/X_TEST_NORMALIZED.csv', index=False)

# building the network

# initialize network
network = Sequential()

# adding input layer and first hidden layer
# fully completed layers have the Rectifier Activation Function
network.add(Dense(5, activation='relu', input_dim=16))
# adding second hidden layer
network.add(Dense(5, activation='relu'))
# adding output layer, Activation Function is Hyperbolic Tangent
network.add(Dense(1, activation='tanh'))

# training the network

# compile network
'''
optimizer = stochastic gradient descent
loss = binary crossentropy (0 no heart disease, 1 presence of heart disease
metrics = accuracy because i want to display the accuracy of such prediction
'''
network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fit the network to the training set
network.fit(X_train, Y_train, validation_split=0.3, batch_size=10, epochs=100)


# predict the test set results
Y_pred = network.predict(X_test)
# if the accuracy score is from 0 to 0.5 for the epoch, then Y_pred will be set to False
# if accuracy score is above 0.5, then the Y_pred will be set to True
Y_pred = (Y_pred > 0.5)

Y_Prediction_CSV = pd.DataFrame(Y_pred).to_csv('data/Predict_TestData/Ver1/Y_Prediction.csv')

print("How accurate is the network at predicting artery disease?: %f" % accuracy_score(Y_test, Y_pred))

















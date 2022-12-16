import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

'''
for this version, I didn't use one hot to encode chest pain type and increased the neurons in the
hidden layer from 6 to 7
less input neurons this time, 13 instead of 16

'''

# load dataset

dataset = pd.read_csv('data/processed_cleveland.csv')
# split dataset into X and Y (independent vars or inputs, dependent var or output)

X = pd.DataFrame(dataset.iloc[:, 0:13])
Y = dataset.iloc[:, 13]

# Testing to see if data saved correctly
X_CSV = X.to_csv('data/Input_Output_Data/Ver2/X.csv', index=False)
Y_CSV = pd.DataFrame(Y).to_csv('data/Input_Output_Data/Ver2/Y.csv', index=False)

# change dependent (target) variable to match dictionary description
'''
Note for Prof Hare:

I don't know if this is allowed since I am messing with the dependent variable data, but I couldn't
get my network to classify in categorical data and then convert that to binary data depending on 
if there is presence of heart disease. So I changed the values a bit to make it easier. 0 means
no heart disease, and then if it is greater than 0, there is presence of heart disease
'''

Y = Y.replace(to_replace=[1, 2, 3, 4], value=1)

# split the X and Y Dataset into Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)

# printing out test, training, and validation data for X and Y (for testing)
pd.DataFrame(X_train).to_csv('data/Train_Data/Ver2/X_train.csv', index=False)
pd.DataFrame(X_test).to_csv('data/Test_Data/Ver2/X_test.csv', index=False)
pd.DataFrame(Y_train).to_csv('data/Train_Data/Ver2/Y_train.csv', index=False)
pd.DataFrame(Y_test).to_csv('data/Test_Data/Ver2/Y_test.csv', index=False)


# normalize values
# (for this version of the network, I normalized all columns)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Test to see if normalization worked
pd.DataFrame(X_train).to_csv('data/Train_Data/Ver2/X_TRAIN_NORMALIZED.csv', index=False)
pd.DataFrame(X_test).to_csv('data/Test_Data/Ver2/X_TEST_NORMALIZED.csv', index=False)


# building the network

# initialize network
network = Sequential()

# adding input layer and first hidden layer
# fully completed layers have the Rectifier Activation Function
network.add(Dense(7, activation='relu', input_dim=13))
# adding second hidden layer
network.add(Dense(7, activation='relu'))
# adding output layer, Activation Function is Hyperbolic Tangent
network.add(Dense(1, activation='tanh'))

# training the network

# compile network
# optimizer = stochastic gradient descent
# loss = binary because we are testing to see if patient has a presence of artery disease or not
# metrics = accuracy because i want to display the accuracy of such prediction
network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fit the network to the training set
network.fit(X_train, Y_train, validation_split=0.3, batch_size=10, epochs=100)

# predict the test set results
Y_pred = network.predict(X_test)
# if the accuracy score is from 0 to 0.5 for the epoch, then Y_pred will be set to False
# if accuracy score is above 0.5, then the Y_pred will be set to True
Y_pred = (Y_pred > 0.5)

pd.DataFrame(Y_pred).to_csv('data/Predict_TestData/Ver2/Y_Prediction.csv')

# prints confusion matrix (shows what the network classified as correct and which ones it did not)
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
print("How accurate is the network at predicting artery disease?: %f" % accuracy_score(Y_test, Y_pred))

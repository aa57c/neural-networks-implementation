import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from Network import Network
from FCLayer import FCLayer
from ActivationLayer import ActivationLayer
from ActivationFunction import tanh, tanh_prime
from Loss import mse, mse_prime





'''
Tutorial from : 
https://medium.com/analytics-vidhya/implementation-of-artificial-neural-network-in-python-step-by-step-guide-556d066f9f5b
dataset = pd.read_csv('data/processed_cleveland.csv')
independent_vars = pd.DataFrame(dataset.iloc[:, 0:12].values)
dependent_vars = pd.DataFrame(dataset.iloc[:, 13].values)

x_train, x_test, y_train, y_test = train_test_split(independent_vars, dependent_vars,
                                                    test_size=0.2, random_state=0)
classifier = Sequential()
classifier.add(Dense(6, activation='relu', input_dim=13))
classifier.add(Dense(6, activation='relu'))
classifier.add(Dense(1, activation='tanh'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


print(type(x_train))
print(type(y_train))

classifier.fit(x_train.to_numpy(), y_train.to_numpy(), batch_size=10, epochs=100)

y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


'''














# network
# net = Network()

# net.add(FCLayer())

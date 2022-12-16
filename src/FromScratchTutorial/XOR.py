"""
Tutorial from: https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
"""

import numpy as np
from src.FromScratchTutorial.Network import Network
from src.FromScratchTutorial.FCLayer import FCLayer
from src.FromScratchTutorial.ActivationLayer import ActivationLayer
from src.FromScratchTutorial.ActivationFunction import tanh, tanh_prime
from src.FromScratchTutorial.Loss import mse, mse_prime

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)

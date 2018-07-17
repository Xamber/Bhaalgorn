import numpy as np
from neuralnetwork import NeuralNetwork

np.random.seed(127)

training_nn_xor = np.array([
    [0.00, 0.00, 1.],
    [0.00, 1., 0.00],
    [1., 0.00, 0.00],
    [1., 1., 1.],
])

nn = NeuralNetwork(training_nn_xor, (2, 2, 1))
nn.train(10000)
print(nn.predicted)

training_set_rent_nn = np.array([
    [0.1, 0.500, 0.24000],
    [0.1, 0.1000, 0.22000],
    [0.1, 0.1000, 0.22000],
    [0.1, 0.1000, 0.22000],
    [0.2, 0.500, 0.32000],
    [0.2, 0.1000, 0.29000],
    [0.3, 0.500, 0.40000],
    [0.1, 0.500, 0.24000],
    [0.1, 0.1000, 0.22000],
    [0.2, 0.500, 0.32000],
    [0.2, 0.1000, 0.29000],
    [0.3, 0.500, 0.40000],
])

nn2 = NeuralNetwork(training_set_rent_nn, (2, 2, 1))
nn2.train(10000)
print(nn2.hypothesis([0.1, 0.1]))

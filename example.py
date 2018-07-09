import numpy as np
from regression import LinearRegression, LogisticRegression
from neuralnetwork import NeuralNetwork

np.random.seed(123)

# Fake training set for property rental with feature scaling
training_set_rent = np.array([
    [(1 - 2.5) / 5, (500 - 500) / 1000, (24000 - 50000) / 100000],
    [(1 - 2.5) / 5, (1000 - 500) / 1000, (22000 - 50000) / 100000],
    [(1 - 2.5) / 5, (1000 - 500) / 1000, (22000 - 50000) / 100000],
    [(1 - 2.5) / 5, (1000 - 500) / 1000, (22000 - 50000) / 100000],
    [(2 - 2.5) / 5, (500 - 500) / 1000, (32000 - 50000) / 100000],
    [(2 - 2.5) / 5, (1000 - 500) / 1000, (29000 - 50000) / 100000],
    [(3 - 2.5) / 5, (500 - 500) / 1000, (40000 - 50000) / 100000],
    [(1 - 2.5) / 5, (500 - 500) / 1000, (24000 - 50000) / 100000],
    [(1 - 2.5) / 5, (1000 - 500) / 1000, (22000 - 50000) / 100000],
    [(2 - 2.5) / 5, (500 - 500) / 1000, (32000 - 50000) / 100000],
    [(2 - 2.5) / 5, (1000 - 500) / 1000, (29000 - 50000) / 100000],
    [(3 - 2.5) / 5, (500 - 500) / 1000, (40000 - 50000) / 100000],
])

# Create and train LR with gradient
linear = LinearRegression(training_set_rent)
linear.train_gradient()

linear.show_info()

# Try to predict something
print(linear.hypothesis([(1 - 2.5) / 5, (1000 - 500) / 1000]) * 100000 + 50000)

# Create and train LR with normal equation
linear_with_normal = LinearRegression(training_set_rent)
linear_with_normal.normal_equation()

linear_with_normal.show_info()

training_set_logic = np.array([
    [0.5, 1.0, 1],
    [0.5, 0.6, 1],
    [0.6, 0.5, 1],
    [1.0, 1.0, 1],
    [0.1, 0.1, 0],
    [0.1, 0.3, 0],
    [0.2, 0.1, 0],
    [0.0, 0.0, 0],
    [0.4, 0.4, 0],
])

logical = LogisticRegression(training_set_logic)
logical.train_gradient(3000)

logical.show_info()

training_nn_xor = np.array([
    [0.00, 0.00, 1.],
    [0.00, 1., 0.00],
    [1., 0.00, 0.00],
    [1., 1., 1.],
])

nn = NeuralNetwork(training_nn_xor, (2, 3, 1))
nn.train(10000, speed=0.4)

print(nn.hypothesis([0.00, 0.00]))
print(nn.hypothesis([0.00, 1.00]))
print(nn.hypothesis([1.00, 0.00]))
print(nn.hypothesis([1.00, 1.00]))

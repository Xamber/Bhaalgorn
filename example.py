import numpy as np
from regression import LinearRegression

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

print("Epoch: ", linear._epoch)
print("Weights: ", linear.weights)
print("Accuracy: ", linear.accuracy)
print("Error: ", linear.error)

# Try to predict something
print(linear.hypothesis([(1 - 2.5) / 5, (1000 - 500) / 1000]) * 100000 + 50000)

# Create and train LR with normal equation
linear_with_normal = LinearRegression(training_set_rent)
linear_with_normal.normal_equation()
print("Weights: ", linear_with_normal.weights)
print("Accuracy: ", linear_with_normal.accuracy)
print("Error: ", linear_with_normal.error)

# Try to predict something
print(linear_with_normal.hypothesis([(1 - 2.5) / 5, (1000 - 500) / 1000]) * 100000 + 50000)

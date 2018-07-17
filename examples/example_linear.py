import numpy as np
np.random.seed(123)
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

linear.show_info()

# Try to predict something
print(linear.hypothesis([(1 - 2.5) / 5, (1000 - 500) / 1000]) * 100000 + 50000)

# Create and train LR with normal equation
linear_with_normal = LinearRegression(training_set_rent)
linear_with_normal.normal_equation()

linear_with_normal.show_info()
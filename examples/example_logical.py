import numpy as np
from regression import LogisticRegression

np.random.seed(123)

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

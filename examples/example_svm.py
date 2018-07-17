import numpy as np
from regression import SVM

np.random.seed(123)

training_set_logic = np.array([
    [0.5 * 10, 1.0 * 10, 1],
    [0.5 * 10, 0.6 * 10, 1],
    [0.6 * 10, 0.5 * 10, 1],
    [1.0 * 10, 1.0 * 10, 1],
    [0.1 * 10, 0.1 * 10, 0],
    [0.1 * 10, 0.3 * 10, 0],
    [0.2 * 10, 0.1 * 10, 0],
    [0.0 * 10, 0.0 * 10, 0],
    [0.4 * 10, 0.4 * 10, 0],
])

logical = SVM(training_set_logic)
logical.train_gradient(200)
print(logical.predicted)
print("Got:", list(map(lambda x: 1 if x > 1 else 0, logical.predicted)))
print("Need:", logical.outputs)

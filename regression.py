import numpy as np


class LinearRegression:
    squared_error = np.vectorize(lambda x, y: pow(x - y, 2))
    cost = np.vectorize(lambda h, y, x: (h - y) * x)

    def __init__(self, dataset):
        self.dataset = dataset

        # Use last column as a output. Others are input.
        self.inputs = self.dataset[..., :-1]
        self.outputs = self.dataset[..., -1:]

        # Store len of training sets and count of features.
        self.len_dataset, self.features_count = self.inputs.shape

        # Create randomized weight vector.
        self.weights = np.random.uniform(0, 1, self.features_count + 1)

        # Epoch of training.
        self._epoch = 1

        # Some prepared matrix/vectors for future usage.
        # - Matrix of input data with 1 as a first value: Use in prediction of training inputs and gradient.
        self.input_vectors = np.concatenate([np.ones((self.len_dataset, 1)), self.inputs], axis=1)

        # - Flatten output: use in gradient
        self.output_vector = self.outputs.flatten()

        # - Swapped input vectors: used in gradient
        self.swapped_input = np.swapaxes(self.input_vectors, 0, 1)

    @property
    def predicted(self):
        # Calculate training set with weight vector.
        return np.dot(self.input_vectors, self.weights)

    @property
    def error(self):
        # Calculate squared error of training set. Scalar value.
        error = (1 / (self.len_dataset * 2)) * self.squared_error(self.predicted, self.output_vector).sum()
        return np.asscalar(error)

    @property
    def accuracy(self):
        # Calculate accuracy based on error function
        return 100.00 - self.error * 100.00

    def hypothesis(self, vector):
        # Calculate value of input vector based on current weight
        return np.dot(self.weights, np.concatenate([np.ones(1), vector])).sum()

    def calculate(self, vector):
        # proxy method for hypothesis.
        # td: will be used as a calc with scalling variables
        return self.hypothesis(vector)

    def gradient(self, speed):
        new_weights = []
        for index, weight in enumerate(self.weights):
            cost = self.cost(self.predicted, self.output_vector, self.swapped_input[index]).sum()
            new_weights.append(weight - (speed * (1 / self.len_dataset) * cost))

        self.weights = new_weights
        self._epoch += 1

    def train_gradient(self, timeout=3000):
        speed = 0.1
        iteration = 0
        latest_error = 1
        while iteration <= timeout:
            error = self.error
            if error < 0.0001:
                break
            if error > latest_error:
                speed = speed * 0.5
            self.gradient(speed)

            latest_error = error
            iteration += 1


training_set = np.array([
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

l = LinearRegression(training_set)
l.train_gradient()

print("Epoch: ", l._epoch)
print("Weights: ", l.weights)
print("Accuracy: ", l.accuracy)
print("Error: ", l.error)

print(l.hypothesis([(1-2.5)/5, (1000-500)/1000]) * 100000 + 50000)

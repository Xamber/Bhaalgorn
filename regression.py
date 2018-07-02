import numpy as np


class LinearRegression:
    squared_error = np.vectorize(lambda h, y: pow(h - y, 2))
    gradient_form = np.vectorize(lambda h, y, x: (h - y) * x)

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
        # td: will be used as a calc with scaling variables
        return self.hypothesis(vector)

    def gradient(self, speed):
        # Gradient Descent iteration function
        new_weights = []
        for index, weight in enumerate(self.weights):
            gradient_form = self.gradient_form(self.predicted, self.output_vector, self.swapped_input[index]).sum()
            new_weights.append(weight - speed / self.len_dataset * gradient_form)

        self.weights = new_weights
        self._epoch += 1

    def normal_equation(self):
        # Normal Equation method
        transposed = self.input_vectors.T
        theta = np.linalg.inv(transposed.dot(self.input_vectors)).dot(transposed).dot(self.outputs)
        self.weights = theta.flatten()

    def train_gradient(self, timeout=3000):
        # Gradient Descent method with automation decreasing speed rate
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

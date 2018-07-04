import numpy as np


class LinearRegression:
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
        self.epoch = 1

        # Some prepared matrix/vectors for future usage.
        # - Matrix of input data with 1 as a first value: Use in prediction of training inputs and gradient.
        self.input_vectors = np.concatenate([np.ones((self.len_dataset, 1)), self.inputs], axis=1)
        # - Flatten output: use in error.
        self.output_vector = self.outputs.flatten()

        # - Transosed input vectors: used in gradient and normal equation
        self.trans_input = np.transpose(self.input_vectors)
        # - Clone output vector to matrix with len(weights) rows
        self.cloned_output = np.vstack([self.output_vector] * len(self.weights))

    @property
    def predicted(self):
        # Calculate training set with weight vector.
        return np.array([self.hypothesis(x) for x in self.inputs])

    @property
    def cloned_predicted(self):
        # Clone predicted vector to matrix with len(weights) rows
        return np.vstack([self.predicted] * len(self.weights))

    @property
    def error(self):
        # Calculate squared error of training set. Scalar value.
        return (1 / (self.len_dataset * 2)) * np.power(self.predicted - self.output_vector, 2).sum()

    @property
    def accuracy(self):
        # Calculate accuracy based on error function.
        return 100.00 - self.error * 100.00

    def hypothesis(self, vector):
        # Calculate value of input vector based on current weight
        return np.dot(self.weights, np.concatenate([np.ones(1), vector])).sum()

    def calculate(self, vector):
        # proxy method for hypothesis.
        return self.hypothesis(vector)

    def gradient(self, speed):
        # Gradient Descent iteration function.
        permit = speed / self.len_dataset
        self.weights -= ((self.cloned_predicted - self.cloned_output) * self.trans_input).sum(axis=1) * permit
        self.epoch += 1

    def normal_equation(self):
        # Normal Equation method
        theta = np.linalg.inv(self.trans_input.dot(self.input_vectors)).dot(self.trans_input).dot(self.outputs)
        self.weights = theta.flatten()

    def train_gradient(self, timeout=3000):
        # Gradient Descent method with automation decreasing speed rate.
        speed = 0.1
        iteration = 0
        latest_error = 1
        while iteration <= timeout:
            error = self.error
            if -0.0001 < error < 0.0001:
                break
            if error > latest_error:
                speed = speed * 0.5
            self.gradient(speed)

            latest_error = error
            iteration += 1


class LogisticRegression(LinearRegression):

    def normal_equation(self):
        raise NotImplemented

    def hypothesis(self, vector):
        # Calculate value of input vector based on current weight. Sigmoid activation.
        doted = np.dot(self.weights, np.concatenate([np.ones(1), vector]))
        return 1.0 / (1 + np.exp(-doted))

    @property
    def error(self):
        # Calculate error of training set. Scalar value.
        cost = -self.output_vector * np.log(self.predicted) - (1 - self.output_vector) * np.log(1 - self.predicted)
        return cost.sum() / self.len_dataset

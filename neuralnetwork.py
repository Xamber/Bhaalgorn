import numpy as np


class NeuralNetwork:

    def __init__(self, dataset, configuration):
        self.dataset = dataset
        self.layers = configuration

        # Use last column as a output. Others are input.
        self.inputs = self.dataset[..., :-1]
        self.outputs = self.dataset[..., -1:]

        # Create and fill weights and biases arrays.
        self.weights_arrays = [np.random.uniform(0, 1, (self.layers[x], self.layers[x + 1])) for x in
                               range(len(self.layers) - 1)]
        self.biases_array = [np.full((1, w.shape[1]), 0.1) for w in self.weights_arrays]

        # Store len of training sets and count of features.
        self.len_dataset, self.features_count = self.inputs.shape

        # Epoch of training.
        self.epoch = 1
        # Cache for training
        self.latest_calculated_vectors = []
        # Reversed indexes for training
        self.reversed_indexes = list(range(len(self.weights_arrays) - 1, -1, -1))

    @staticmethod
    def activate(x):
        # Activate layer with activation function
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative(s):
        # derivative of activation function
        return s * (1 - s)

    @property
    def predicted(self):
        return self.hypothesis(self.inputs)

    @property
    def error(self):
        return np.sum((self.predicted - self.outputs)**2) / self.len_dataset

    def hypothesis(self, vector):
        # Proxy method for forward_propagation
        return self.forward_propagation(vector)

    def forward_propagation(self, data):
        # Forward Propagation method
        self.latest_calculated_vectors = [data]

        for w, b in zip(self.weights_arrays, self.biases_array):
            data = self.activate(np.dot(data, w) + b)
            self.latest_calculated_vectors.append(data)

        return data

    def backward_propagation(self, speed=0.3):
        # Backward Propagation iteration
        self.forward_propagation(self.inputs)

        result = self.latest_calculated_vectors.pop()
        layer_error = (self.outputs - result)

        for i in self.reversed_indexes:
            delta = layer_error * self.derivative(result)
            result = self.latest_calculated_vectors.pop()
            layer_error = np.dot(layer_error, self.weights_arrays[i].T)
            self.weights_arrays[i] += speed * np.dot(result.T, delta)
            self.biases_array[i] += speed * delta.sum()

        self.epoch += 1

    def train(self, timeout=3000):
        speed = 1
        iteration = 0
        latest_error = 1

        while iteration <= timeout:
            error = self.error
            if -0.0001 < error < 0.0001:
                break
            if error > latest_error:
                speed = speed * 0.5
            self.backward_propagation(speed)

            latest_error = error
            iteration += 1

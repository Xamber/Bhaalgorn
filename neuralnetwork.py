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

    def hypothesis(self, vector):
        # Proxy method for forward_propagation
        return self.forward_propagation(vector)

    def forward_propagation(self, vector):
        # Forward Propagation method

        # Check is vector numpy array or not.
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)

        self.latest_calculated_vectors = [vector[np.newaxis, :]]
        for w, b in zip(self.weights_arrays, self.biases_array):
            vector = self.activate(np.dot(vector, w) + b)
            self.latest_calculated_vectors.append(vector)

        return vector

    def backward_propagation(self, speed=0.3):
        # Backward Propagation iteration
        for vector, expected_result in zip(self.inputs, self.outputs):
            self.forward_propagation(vector)

            result = self.latest_calculated_vectors.pop()
            layer_error = (expected_result - result)

            for i in self.reversed_indexes:
                delta = layer_error * self.derivative(result)
                result = self.latest_calculated_vectors.pop()
                layer_error = np.dot(layer_error, self.weights_arrays[i].T)
                self.weights_arrays[i] += speed * np.dot(result.T, delta)
                self.biases_array[i] += speed * delta

        self.epoch += 1

    def train(self, timeout=3000, speed=0.4):
        # Train neural network with backward_propagation
        iteration = 0
        while iteration <= timeout:
            self.backward_propagation(speed)
            iteration += 1

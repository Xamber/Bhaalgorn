import numpy as np


class LinearRegression:
    def __init__(self, features_count, dataset):

        self.dataset = dataset
        self.features_count = features_count

        self.inputs = self.dataset[..., :-1]
        self.outputs = self.dataset[..., -1:]

        self.weights = np.random.uniform(0, 1, self.features_count + 1)
        #self.weights = np.array([0, 1])
        self._epoch = 1

        self._prepared_input_cache = None

    @property
    def len_dataset(self):
        return len(self.inputs)

    @property
    def prepared_output(self):
        return np.swapaxes(self.outputs, 0, 1)

    @property
    def prepared_input(self):
        if self._prepared_input_cache is None:
            ones = np.ones(len(self.dataset)).reshape(-1, 1)
            self._prepared_input_cache = np.concatenate([ones, self.inputs], axis=1)
        return self._prepared_input_cache

    def hypothesis(self, vector):
        return np.dot(self.weights, np.concatenate([np.ones(1), vector])).sum()

    def error(self):
        squared_error = np.vectorize(lambda x, y: pow(x - y, 2))
        predicted = [self.hypothesis(x) for x in self.inputs]
        error = (1 / (self.len_dataset * 2)) * squared_error(predicted, self.prepared_output).sum()
        return np.asscalar(error)

    def accuracy(self):
        accuracy = 100.00 - self.error() * 100.00
        return accuracy if accuracy >= 0 else 0

    def gradient(self, speed):
        cost = np.vectorize(lambda h, y, x: (h - y) * x)

        new_weights = []
        predicted = np.array([self.hypothesis(x) for x in self.inputs])
        for index, weight in enumerate(self.weights):
            new_weights.append(weight - (speed * (1 / self.len_dataset) * cost(predicted, self.prepared_output, np.swapaxes(self.prepared_input, 0, 1)[index]).sum()))

        self.weights = new_weights
        self._epoch += 1

    def train_gradient(self, timeout=3000):
        speed = 0.1
        iteration = 0
        latest_error = 1

        while iteration <= timeout:
            error = self.error()
            if error < 0.0001:
                break

            if error > latest_error:
                speed = speed * 0.5
            self.gradient(speed)

            latest_error = error
            iteration += 1



dataset = np.array([
    [1, 1+10],
    [2, 2+10],
    [3, 3+10],
    [4, 4+10],
    [5, 5+10],
    [6, 6+10],
    [7, 7+10],
    [8, 8+10],
    [9, 9+10],
])

dataset = np.array([
    [(1 - 2.5) / 5, (500 - 500) / 1000, (24000 - 50000) / 100000],
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

l = LinearRegression(2, dataset)

print(l.accuracy())
print(l.prepared_input)
l.train_gradient(3000)

print("_epoch: ", l._epoch)
print("weights: ", l.weights)
print("accuracy: ", l.accuracy())
print("error: ", l.error())

tring = l.hypothesis([(1-2.5)/5, (1000-500)/1000])
print(tring * 100000 + 50000)

import random
import numpy as np

class TrainingSetsException(Exception):  pass

# y = np.array([10,10,10])
# y.reshape(-1, 1)
# x = np.array([(1,2,3), (4,5,6), (7,8,9)])


class LinearRegression:

    def __init__(self, features_count, dataset):

        self.dataset = dataset
        self.inputs = self.dataset[..., :-1]
        self.outputs = self.dataset[..., -1:]

        self.features_count = features_count
        self.weights = np.ones(self.features_count+1)
        self._epoch = 1

        self._prepared_input_cache = None

    @property
    def lenght_dataset(self):
        return len(self.inputs)

    @property
    def prepared_input(self):
        if self._prepared_input_cache is None:
            ones = np.ones(len(self.dataset)).reshape(-1, 1)
            self._prepared_input_cache = np.concatenate([ones, self.inputs], axis=1)
        return self._prepared_input_cache

    def hypothesis(self, vector):
        return np.dot(self.weights, np.concatenate([np.array([1]), vector])).sum()

    def error(self):
        predicted = [self.hypothesis(x) for x in self.inputs]
        error = (1 / (self.lenght_dataset * 2)) * sum([pow(h - y, 2) for h, y in zip(predicted, self.outputs)])
        return np.asscalar(error)

    def accuracy(self):
        accuracy = 100.00 - self.error() * 100.00
        return accuracy if accuracy >= 0 else 0

    def gradient(self, speed):
        new_weights = []
        predicted = [self.hypothesis(x) for x in self.inputs]
        for index, weight in enumerate(self.weights):
            new_weight = weight - speed * (1 / self.lenght_dataset) * sum([(h - y) * x for h, y, x in zip(predicted, self.outputs, self.prepared_input)])
            new_weights.append(new_weight)

        self.weights = new_weights
        self._epoch += 1

    def train_gradient(self, timeout=100000):
        speed = 0.01
        iteration = 0
        latest_error = 1

        while iteration <= timeout:
            error = self.error()
            if error < 0.0001:
                break

            if error > latest_error:
                speed = speed * 0.9

            #print(error, latest_error, error > latest_error, latest_error-error, speed)

            self.gradient(speed)

            latest_error = error
            iteration += 1


dataset = np.array([
    [(1 - 2.5) / 5, (500 - 500) / 1000, (24000-50000)/100000],
    [(1 - 2.5) / 5, (1000 - 500) / 1000, (22000-50000)/100000],
    [(2 - 2.5) / 5, (500 - 500) / 1000, (32000-50000)/100000],
    [(2 - 2.5) / 5, (1000 - 500) / 1000, (29000-50000)/100000],
    [(3 - 2.5) / 5, (500 - 500) / 1000, (40000-50000)/100000],
    [(1 - 2.5) / 5, (500 - 500) / 1000, (24000-50000)/100000],
    [(1 - 2.5) / 5, (1000 - 500) / 1000, (22000-50000)/100000],
    [(2 - 2.5) / 5, (500 - 500) / 1000, (32000-50000)/100000],
    [(2 - 2.5) / 5, (1000 - 500) / 1000, (29000-50000)/100000],
    [(3 - 2.5) / 5, (500 - 500) / 1000, (40000-50000)/100000],
])

l = LinearRegression(2, dataset)
print()
tring = l.hypothesis([(1-2.5)/5, (1000-500)/1000])
print(l.accuracy())
print(l.prepared_input)
l.train_gradient(10000)
print("_epoch: ", l._epoch)
print("weights: ", l.weights)
print("accuracy: ", l.accuracy())
print("error: ", l.error())

tring = l.hypothesis([(1-2.5)/5, (1000-500)/1000])
print(tring * 100000 + 50000)

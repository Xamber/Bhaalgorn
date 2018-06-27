import random


class TrainingSetsException(Exception):  pass


class LinearRegression:

    def __init__(self, features_count):

        self.inputs = []
        self.outputs = []

        self.features_count = features_count
        self.weights = [random.uniform(0, 1) for _ in range(self.features_count+1)]
        self._epoch = 1

    @property
    def lenght_dataset(self):
        return len(self.inputs)

    def add_example(self, input_vector, output):

        if not len(input_vector):
            raise TrainingSetsException("Cannot add zero len vector")

        if len(input_vector) != self.features_count:
            raise TrainingSetsException("Len of vector and len of features count are not a same")

        self.inputs.append(input_vector)
        self.outputs.append(output)

    def hypothesis(self, vector):
        return self.weights[0] + sum([x * w for x, w in zip(vector, self.weights[1:])])

    def error(self):
        predicted = [self.hypothesis(x) for x in self.inputs]
        return (1 / (self.lenght_dataset * 2)) * sum([pow(h - y, 2) for h, y in zip(predicted, self.outputs)])

    def gradient(self, speed):
        new_weights = []
        predicted = [self.hypothesis(x) for x in self.inputs]
        for index, w in enumerate(self.weights):
            new_weight = w - speed * (1 / self.lenght_dataset) * sum([(h - y) * (1 if index == 0 else x[index-1]) for h, y, x in zip(predicted, self.outputs, self.inputs)])
            new_weights.append(new_weight)
        self.weights = new_weights
        self._epoch += 1

l = LinearRegression(1)
l.add_example([1], 1)
l.add_example([2], 2)
l.add_example([3], 3)


print(l.error())

for i in range(0, 190):
    l.gradient(0.2)


print("=========")
print(l.weights)
print("=========")
print(l.error())

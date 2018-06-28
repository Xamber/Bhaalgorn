import random


class TrainingSetsException(Exception):  pass


class LinearRegression:

    def __init__(self, features_count):

        self.inputs = []
        self.outputs = []

        self.features_count = features_count
        self.weights = [random.uniform(0, 1) for _ in range(self.features_count+1)]
        self._epoch = 1

        self._prepared_input_cache = None

    @property
    def lenght_dataset(self):
        return len(self.inputs)

    @property
    def prepared_input(self):
        if self._prepared_input_cache == None:
            self._prepared_input_cache = [1] * self.lenght_dataset + [list(i) for i in zip(*self.inputs)]
        return self._prepared_input_cache

    def add_example(self, input_vector, output):

        if not len(input_vector):
            raise TrainingSetsException("Cannot add zero len vector")

        if len(input_vector) != self.features_count:
            raise TrainingSetsException("Len of vector and len of features count are not a same")

        self.inputs.append(input_vector)
        self.outputs.append(output)
        self._prepared_input_cache = None

    def hypothesis(self, vector):
        return self.weights[0] + sum([x * w for x, w in zip(vector, self.weights[1:])])

    def error(self):
        predicted = [self.hypothesis(x) for x in self.inputs]
        return (1 / (self.lenght_dataset * 2)) * sum([pow(h - y, 2) for h, y in zip(predicted, self.outputs)])

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

            print(error, latest_error, error > latest_error, latest_error-error, speed)

            self.gradient(speed)

            latest_error = error
            iteration += 1


l = LinearRegression(2)
l.add_example([(1-2.5)/5, (500-500)/1000], (24000-50000)/100000)
l.add_example([(1-2.5)/5, (1000-500)/1000], (22000-50000)/100000)
l.add_example([(2-2.5)/5, (500-500)/1000], (32000-50000)/100000)
l.add_example([(2-2.5)/5, (1000-500)/1000], (29000-50000)/100000)
l.add_example([(3-2.5)/5, (500-500)/1000], (40000-50000)/100000)
l.add_example([(1-2.5)/5, (500-500)/1000], (24000-50000)/100000)
l.add_example([(1-2.5)/5, (1000-500)/1000], (22000-50000)/100000)
l.add_example([(2-2.5)/5, (500-500)/1000], (32000-50000)/100000)
l.add_example([(2-2.5)/5, (1000-500)/1000], (29000-50000)/100000)
l.add_example([(3-2.5)/5, (500-500)/1000], (40000-50000)/100000)


print(l.accuracy())
print(l.prepared_input)
l.train_gradient()
print("_epoch: ", l._epoch)
print("weights: ", l.weights)
print("accuracy: ", l.accuracy())
print("error: ", l.error())

tring = l.hypothesis([(1-2.5)/5, (1000-500)/1000])
print(tring * 100000 + 50000)

from exc import TrainingSetsException


class Example:

    def __init__(self, input, output):
        if len(input) != len(output):
            raise TrainingSetsException("Len of inputs and outputs are not a same")

        self.input = input
        self.output = output


class LinearRegression:

    def __init__(self, dataset):
        self.weight_0 = 0
        self.weight_1 = 0.5
        self.dataset = dataset
        self.lenght_dataset = len(self.dataset.input)

        self._epoch = 1

    def hypothesis(self, x):
        return self.weight_0 + self.weight_1 * x

    def error(self):
        predicted_set = [self.hypothesis(x) for x in self.dataset.input]
        output = self.dataset.output
        return (1 / (self.lenght_dataset * 2)) * sum([pow(h - y, 2) for h, y in zip(predicted_set, output)])

    def gradient(self, speed):
        predicted_set = [self.hypothesis(x) for x in self.dataset.input]
        output = self.dataset.output
        input = self.dataset.input
        new_weight_0 = self.weight_0 - speed * (1 / (self.lenght_dataset)) * sum(
            [h - y for h, y in zip(predicted_set, output)])
        new_weight_1 = self.weight_1 - speed * (1 / (self.lenght_dataset)) * sum(
            [(h - y) * x for h, y, x in zip(predicted_set, output, input)])
        self.weight_0 = new_weight_0
        self.weight_1 = new_weight_1
        self._epoch += 1


dataset = Example([1, 2, 3, 4], [3, 4, 5, 6])

l = LinearRegression(dataset)
print(l.error())

for i in range(0, 190):
    l.gradient(0.1)

print(l.error())

print("=========")
print(l.weight_0)
print(l.weight_1)

print("=========")
print(l.error())


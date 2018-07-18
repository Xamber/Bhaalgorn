import numpy as np


class KMeans:

    def __init__(self, k, dataset):
        self.dataset = dataset
        self.k = k
        self.centroids = []

        # Randomly choose centroids
        self.randomize_centroids()

    @staticmethod
    def distance(point, centroid):
        return np.linalg.norm(point-centroid)

    @property
    def predicted(self):
        return np.array([self.predict(p) for p in self.dataset])

    def randomize_centroids(self):
        self.centroids = self.dataset[np.random.choice(len(self.dataset), self.k, replace=False), :]

    def get_prepared(self):
        return zip(self.dataset, self.predicted)

    def get_points_in_clusters(self):
        return [self.dataset[np.where(self.predicted == c)] for c in range(self.k)]

    def error(self):
        distances = [self.distance(point, self.centroids[int(cluster)]) for point, cluster in self.get_prepared()]
        return 1 / len(self.dataset) * sum(distances)

    def predict(self, point):
        return int(np.argmin([self.distance(point, centroid) for centroid in self.centroids]))

    def train(self, epochs=50):
        minimal_error = np.inf
        minimal_centroids = self.centroids

        for e in range(epochs):
            self.randomize_centroids()
            self.update_centroids()

            if self.error() < minimal_error:
                minimal_centroids = self.centroids
                minimal_error = self.error()

        self.centroids = minimal_centroids

    def update_centroids(self):
        predicted = None
        while (predicted is None) or (not all(predicted == self.predicted)):
            predicted = self.predicted
            self.centroids = list(map(lambda x: np.sum(x, axis=0) / (len(x) or 1), self.get_points_in_clusters()))

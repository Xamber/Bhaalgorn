import numpy as np
import matplotlib.pyplot as plt
from unsupervised import KMeans

training_set_logic = np.array([
    [300 / 20, 12],
    [300 / 20, 12],
    [250 / 20, 10],
    [350 / 20, 15],
    [320 / 20, 10],
    [360 / 20, 14],
    [370 / 20, 15],
    [399 / 20, 14],
    [890 / 20, 45],
    [950 / 20, 42],
    [950 / 20, 42],
    [970 / 20, 45],
    [999 / 20, 50],
    [950 / 20, 40],
    [390 / 20, 40],
    [350 / 20, 41],
    [350 / 20, 40],
    [370 / 20, 42],
    [399 / 20, 40],
    [350 / 20, 40],
    [900 / 20, 12],
    [900 / 20, 12],
    [900 / 20, 10],
    [900 / 20, 15],
    [900 / 20, 10],
    [900 / 20, 14],
    [900 / 20, 15],
    [900 / 20, 10],
])

k_means = KMeans(4, training_set_logic)
k_means.train()

fig, ax = plt.subplots()
for p in k_means.get_points_in_clusters():
    ax.plot(p[..., :-1], p[..., -1:], 'o')
for c in k_means.centroids:
    ax.plot(c[0], c[1], 'X', color="y")
plt.show()

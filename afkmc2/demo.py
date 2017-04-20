# -*- coding: utf-8 -*-
"""Demo/Benchmarking to show how this code finds good seedings for KMeans.
Example used is the Iris Dataset from sklearn.

Also contains some comparison between the different variants and graphing.

The code is not meant for beautiful/modular testing of the package,
instead it is a simple introduction to applying this and the result to expect.
"""
from sklearn.cluster import KMeans
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import afkmc2

np.random.seed(5)

# Runtime Measurements
X = np.random.rand(500, 80)

start = time.time()
for _ in range(50):
    afkmc2.kmpp(X, 20)
print "runtime kmpp:\t\t", (time.time() - start)/50

start = time.time()
for _ in range(50):
    afkmc2.kmc2(X, 20)
print "runtime kmc2:\t\t", (time.time() - start)/50

start = time.time()
for _ in range(50):
    afkmc2.afkmc2(X, 20)
print "runtime afkmc2:\t\t", (time.time() - start)/50

start = time.time()
for _ in range(50):
    afkmc2.afkmc2_c(X, 20)
print "runtime afkmc2_c:\t", (time.time() - start)/50

# Iris Visualization
iris = datasets.load_iris()
X = iris.data
y = iris.target


# AFKMC2 Seeding
seeding = afkmc2.afkmc2_c(X, 3)

fig = plt.figure(0, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

est = KMeans(n_clusters=3, init=seeding)
plt.cla()
est.fit(X)
labels = est.labels_

ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))
ax.scatter(seeding[:, 3], seeding[:, 0], seeding[:, 2], c='red', s=100)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_title('Clusters after AFKMC2C Seeding')
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')


# Random Seeding
fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

est = KMeans(n_clusters=3, init='random')
plt.cla()
est.fit(X)
labels = est.labels_

ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_title('Clusters with random seeding')
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')


# Plot the ground truth
fig = plt.figure(2, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

for name, label in [('Setosa', 0),
                    ('Versicolour', 1),
                    ('Virginica', 2)]:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y)
ax.scatter(seeding[:, 3], seeding[:, 0], seeding[:, 2], c='red', s=100)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_title('Ground Truth')
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
plt.show()

"""Python Implementation of Fast and Provably Good Seedings for k-Means

> Fast and Provably Good Seedings for k-Means
> Olivier Bachem, Mario Lucic, S. Hamed Hassani and Andreas Krause
> In Neural Information Processing Systems (NIPS), 2016.

Usage:
>>> import afkmc2
>>> X = np.array with d-dimensional data
>>> seeds = afkmc2.afkmc2(X, 3)

Use seeds in sklearn KMeans:
>>> from sklearn.custer import KMeans
>>> model = KMeans(3, init=seeds).fit(X)
>>> print model.cluster_centers_
"""
import numpy as np


def afkmc2(X, k, m=200, af=True):
    """Python Assumption Free KMC^2 Seeding in O(nd + mk^2d)

    Args:
        X: np.array with datapoints. shape: n, d
        k: Number cluster centers
        m: length of markov chain. Default: 200
        af: Use assumption free algorithm (true) or uniform assumption (false)

    Returns:
        np.array with cluster centers for seeding. shape: k, d
    """
    centers = np.zeros((k, X.shape[1]))

    # Sample Point uniformly
    centers[0, :] = X[np.random.choice(X.shape[0]), :]

    if af:
        # Create assumption free proposal distribution: O(n)
        d2 = [np.linalg.norm(X[i, :]-centers[0:1, :])**2
              for i in range(X.shape[0])]
        q = d2/(2*np.sum(d2)) + 1/(2.0*X.shape[0])
    else:
        # Uniform assumption
        q = np.ones(X.shape[0]) / X.shape[0]

    # k-1 iterations
    for i in range(1, k):
        # Sample initial point of Markov Chain
        x = np.random.choice(X.shape[0], p=q)
        # Get shortest distance from previous centers
        dx2 = min([np.linalg.norm(X[x, :]-centers[j, :])**2 for j in range(i)])

        # m-1 more candidates
        for j in range(1, m):
            # New Sample
            y = np.random.choice(X.shape[0], p=q)
            dy2 = min([np.linalg.norm(X[y, :]-centers[j, :])**2
                      for j in range(i)])
            # Move to candidate according to acceptance prob based on distances
            if (dy2*q[x])/(dx2*q[y]) > np.random.uniform():
                x = y
                dx2 = dy2

        # Store current choice after m samples
        centers[i] = X[x, :]

    return centers

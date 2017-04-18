.. afkmc2 documentation master file, created by
   sphinx-quickstart on Mon Apr 17 18:48:00 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Assumption Free K-Means++ Seedings
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

AFKMC2 is a sklearn compatible python implementation of the algorithm detailed in

    |   **Fast and Provably Good Seedings for k-Means**
    |   Olivier Bachem, Mario Lucic, S. Hamed Hassani and Andreas Krause
    |   In *Neural Information Processing Systems* (NIPS), 2016.
    |   https://las.inf.ethz.ch/files/bachem16fast.pdf

The algorithm uses Monte Carlo Markov Chain to quickly find good seedings for KMeans

Usage
_____

Using this package to get seedings for KMeans in sklearn is as simple as::

    import afkmc2
    X = np.array([[1, 2], [1, 4], [1, 0],
                 [4, 2], [4, 4], [4, 0]])
    seeds = afkmc2.afkmc2(X, 2)

    from sklearn.custer import KMeans
    model = KMeans(n_clusters=2, init=seeds).fit(X)
    print model.cluster_centers_

Installation
____________

Quickly install afkmc2 by running::

    pip install afkmc2

Contribute
__________

* Issue Tracker: https://github.com/adriangoe/afkmc2/issues
* Source Code: https://github.com/adriangoe/afkmc2

Support
_______

You can reach out to me through https://adriangoe.me.


License
_______

The project is licensed under the MIT License. The intellectual authors of the original algorithm are Olivier Bachem, Mario Lucic, S. Hamed Hassani and Andreas Krause.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

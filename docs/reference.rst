=================
Seeding Reference
=================

`View Code On Github <https://github.com/adriangoe/afkmc2/blob/master/afkmc2/afkmc2.py>`_.

K-Means++
---------

K-Means++ is the original seeding algorithm for K-Means as proposed by Arthur and Vassilvitskii in 2007 [kmpp]_.

It chooses one center uniformly, then computes distance of every datapoint to already chosen centers in order to use distance as weights when sampling next center. These steps are repeated until `k` centers are chosen.

Chosing good seedings speeds up convergence for K-Means, but extra time cost is occurred calculating all distances.

.. autofunction:: afkmc2.kmpp


K-Means Markov Chain Monte Carlo
--------------------------------

KMC^2 was proposed as an improvement over K-Means++ in 2016 [kmc2]_. While K-Means++ requires k full passes over the dataset, KMC^2 replaces the D^2 sampling step with Markov Chain Monte Carlo sampling. Runtime is no longer tied to number of datapoints while new centers will be chosen far from current centers.

.. autofunction:: afkmc2.kmc2


Assumption Free KMC^2
---------------------

AFKMC^2 is an improvement proposed by the same authors [afkmc2]_. While KMC^2 requires assumptions about the data generating distribution (in our implementation uniformity), this algorithm works without such assumptions. It the true D^2-sampling distribution with regards to the first center :code:`c_1` as a proposal distribution that can approximate nonuniform distributions.

This means an added runtime cost of :code:`O(nd)` to calculate the initial distribution, but performance improvements for nonuniform samples.

.. autofunction:: afkmc2.afkmc2

Cached AFKMC^2
--------------

The author of this package proposed this slight runtime improvement for AFKMC^2 (it could also be applied to KMC^2). Since the first :code:`O(nd)` pass already calculates all distances between :code:`X` and :code:`c_1` we can at minimum save ourselves k*m distance calculations by storing these results to be reused in the Markov Chain. This comes with an additional space cost of :code:`O(nk)`.

The savings are highest for small datasets but can yield significant runtime improvement for very large/high-dimensional ones as well.

.. autofunction:: afkmc2.afkmc2_c

References
----------

.. [kmpp] Arthur, D., & Vassilvitskii, S. (2007, January). k-means++: The advantages of careful seeding. In Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms (pp. 1027-1035). Society for Industrial and Applied Mathematics.
.. [kmc2] Bachem, O., Lucic, M., Hassani, S. H., & Krause, A. (2016, February). Approximate K-Means++ in Sublinear Time. In AAAI (pp. 1459-1467).
.. [afkmc2] Bachem, O., Lucic, M., Hassani, H., & Krause, A. (2016). Fast and Provably Good Seedings for k-Means. In Advances in Neural Information Processing Systems (pp. 55-63).

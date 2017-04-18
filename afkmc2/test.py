import afkmc2
import kmc2
import numpy as np
import time
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
import warnings
import matplotlib.pyplot as plt

test_data = np.zeros((40000, 2))
test_data[0:10000, :] = 30.0
test_data[10000:20000, :] = 60.0
test_data[20000:30000, :] = 90.0
test_data[30000:, :] = 120.0
X = test_data
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
# X = np.array([[0,1,5], [1,1,5], [1,0,4], [5,5,10], [5,6,9], [6,5,9], [0,5,0], [1,5,0], [0,6,0]])
seeding = afkmc2.afkmc2(X, 2, m = 100, af = True)
model = MiniBatchKMeans(2, init=seeding).fit(X)
new_centers = model.cluster_centers_
plt.scatter(X[:,0], X[:,1], color="blue")
plt.scatter(seeding[:,0], seeding[:,1], color="red")
plt.scatter(new_centers[:,0], new_centers[:,1], color="green")
plt.show()

import kmc2
print kmc2.kmc2(X, 3)

def scenarios():
    """A variety of small-scale problems"""
    rs = np.random.RandomState(0)
    a = rs.randn(500, 2)
    a_sparse = csr_matrix(a)
    lengths = [1, 2, 5, 10]
    for rs in [np.random.RandomState(0), 0, None]:
        for l in lengths:
            for afkmc2 in [True, False]:
                yield dict(X=a, k=5, m=l, af=True)


def test_scenarios():
    """Test that everything works"""
    for s in scenarios():
        seeding = afkmc2.afkmc2(**s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # disable sklearn warnings
            model = MiniBatchKMeans(s["k"], init=seeding).fit(s["X"])
        new_centers = model.cluster_centers_
        print seeding, new_centers
        plt.scatter(s["X"][:,0], s["X"][:,1], color="blue")
        plt.scatter(seeding[:,0], seeding[:,1], color="red")
        plt.scatter(new_centers[:,0], new_centers[:,1], color="green")
        plt.show()


def test_sparse_dense():
    """Test sparse / dense consistency"""
    for s in scenarios():
        validate_sparse_dense(**s)


def validate_sparse_dense(X, **kwargs):
   """Validate that sparse and dense input gives exactly the same result"""
   kwargs["random_state"] = 1  # important, set the same seed
   X_sparse = csr_matrix(X)
   res = kmc2.kmc2(X, **kwargs)
   res_sparse = kmc2.kmc2(X_sparse, **kwargs)
   np.testing.assert_array_equal(res, res_sparse)


def test_weights():
    """Test weight consistency"""
    for s in scenarios():
        validate_weights(**s)


def validate_weights(X, **kwargs):
   """Validate that sparse and dense input gives exactly the same result"""
   kwargs["random_state"] = 1  # important, set the same seed
   # Weights = None
   kwargs["weights"] = None
   res1 = kmc2.kmc2(X, **kwargs)
   # Weight = np.ones
   kwargs["weights"] = np.ones(X.shape[0])
   res2 = kmc2.kmc2(X, **kwargs)
   np.testing.assert_array_equal(res1, res2)
   # Weight = 1000
   kwargs["weights"] = np.ones(X.shape[0])*1000
   res3 = kmc2.kmc2(X, **kwargs)
   np.testing.assert_array_equal(res2, res3)

   X[0, :] *= 1000
   kwargs["k"] = 5
   # first element has
   kwargs["weights"] = np.ones(X.shape[0])
   kwargs["weights"][0] = 1001
   res4 = kmc2.kmc2(X, **kwargs)
   # one guy with
   X_new = np.vstack((X[[0]*1000,:], X))
   kwargs["weights"] = None
   res5 = kmc2.kmc2(X_new, **kwargs)
   np.testing.assert_array_equal(res4, res5)


def qe(X, centers):
    """Compute the quantization error"""
    a1 = np.sum(np.power(X, 2), axis=1)
    a2 = np.dot(X, centers.T)
    a3 = np.sum(np.power(centers, 2), axis=1)
    dist = - 2*a2 + a3[np.newaxis, :]
    argmin = np.argmin(dist, axis=1)
    mindist = np.min(dist, axis=1) + a1
    error = np.sum(mindist)
    return error

test_scenarios()
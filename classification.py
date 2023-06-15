import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import util

def SVM_classifier_nfold(X, Y, n=10):
    Y = np.array(Y)
    idx = np.arange(len(Y), dtype=np.int32)
    np.random.shuffle(idx)
    skf = StratifiedKFold(n_splits=n)
    cvscores = []
    error_labels_index=[]
    X=X[idx]
    Y=Y[idx]
    print ("Performing 10-fold cross validation...")
    for i, (train, test) in enumerate(skf.split(X, Y)):
    	 # Fit the SVM model
        clf2 = SVC(C=20, gamma=1.5e-04)
        clf2.fit(X[train], np.ravel(Y[train]))
        accuracy=clf2.score(X[test],np.ravel(Y[test]))
        predicted_labels=clf2.predict(X[test])
        test_labels=np.reshape(Y[test],(len(Y[test])))
        error=predicted_labels-test_labels
        # print ("Test Fold "+str(i))
        # print("Accuracy= %.2f%%" % (accuracy*100))
        cvscores.append(accuracy * 100)

    print("Average Accuracy={:.2f}-{:.2f}".format(np.mean(cvscores), 
                                        np.std(cvscores) / np.sqrt(n)))
    return np.mean(cvscores), np.std(cvscores)

def KNN_classifier_nfold(X, Y, n=10, k=1, p=2):
    # n-fold cross validation
    # k-NN classification

    Y = np.array(Y)
    idx = np.arange(len(Y), dtype=np.int32)
    np.random.shuffle(idx)
    skf = StratifiedKFold(n_splits=n)
    cvscores = []
    error_labels_index=[]
    X=X[idx]
    Y=Y[idx]
    print ("Performing {}-fold cross validation...".format(n))
    for i, (train, test) in enumerate(skf.split(X, Y)):
    	 # Fit the KNN model
        neigh = KNeighborsClassifier(n_neighbors=k, p=p)
        neigh.fit(X[train], np.ravel(Y[train]))
        accuracy=neigh.score(X[test],np.ravel(Y[test]))
        predicted_labels=neigh.predict(X[test])
        test_labels=np.reshape(Y[test],(len(Y[test])))
        error=predicted_labels-test_labels
        cvscores.append(accuracy * 100)
    # print(cvscores)
    print("Average Accuracy={:.2f}\\pm{:.2f}".format(np.mean(cvscores), 
                                        np.std(cvscores) / np.sqrt(n)))
    return np.mean(cvscores), np.std(cvscores)

from sklearn.linear_model import LogisticRegression

def Logistic_classifier_nfold(X, Y, n=10):
    # n-fold cross validation
    # k-NN classification

    Y = np.array(Y)
    idx = np.arange(len(Y), dtype=np.int32)
    np.random.shuffle(idx)
    skf = StratifiedKFold(n_splits=n)
    cvscores = []
    error_labels_index=[]
    X=X[idx]
    Y=Y[idx]
    print ("Performing {}-fold cross validation...".format(n))
    for i, (train, test) in enumerate(skf.split(X, Y)):
    	 # Fit the KNN model
        logis = LogisticRegression()
        logis.fit(X[train], np.ravel(Y[train]))
        accuracy=logis.score(X[test], np.ravel(Y[test]))
        predicted_labels=logis.predict(X[test])
        test_labels=np.reshape(Y[test], (len(Y[test])))
        error=predicted_labels-test_labels
        cvscores.append(accuracy * 100)
    # print(cvscores)
    print("Average Accuracy={}+-{}".format(np.mean(cvscores), 
                                        np.std(cvscores) / np.sqrt(n)))
    return np.mean(cvscores), np.std(cvscores)






from sklearn.metrics import accuracy_score, pairwise_distances 
# from sklearn.metrics.pairwise import _return_float_dtype
from sklearn.utils import gen_even_slices
from sklearn.utils.fixes import delayed
from sklearn.utils.validation import _num_samples
import ot
from ot.backend import get_backend, Backend
from joblib import Parallel, effective_n_jobs
from scipy.linalg import eigvalsh
import itertools
from functools import partial

from laplacian import normalizeLaplacian, sym_normalize_adj


def KNN_classifier_kfold_DistMtx(DistMtx, Y, n=10, k=1):
    Y = np.array(Y)
    idx = np.arange(len(Y), dtype=np.int32)
    np.random.shuffle(idx)
    skf = StratifiedKFold(n_splits=n)
    cvscores = []
    error_labels_index=[]
    X=DistMtx[idx][:, idx]
    # print(X.shape)
    Y=Y[idx]

    print ("Performing 10-fold cross validation...")
    for i, (train, test) in enumerate(skf.split(X, Y)):
    	 # Fit the KNN model
        neigh = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
        neigh.fit(X[train][:, train], np.ravel(Y[train]))

        # Though not specified in the doc, the code for self.score() is 
        # accuracy_score(y, self.predict(X), sample_weight=sample_weight)
        # which can also be applied to "precomputed"
        # accuracy=neigh.score(DistMtx[test][train],np.ravel(Y[test]))
        
        predicted_labels=neigh.predict(X[test][:, train])
        accuracy = accuracy_score(np.ravel(Y[test]), predicted_labels)

        test_labels=np.reshape(Y[test], len(Y[test]))
        error=predicted_labels-test_labels
        cvscores.append(accuracy * 100)
        
    print(cvscores)
    print ("Average Accuracy= %.2f%%" % (np.mean(cvscores)))
    return np.mean(cvscores), np.std(cvscores)

def _laplacian(x):
    r"""Compute Laplacian matrix"""
    nx = get_backend(x)
    L = nx.diag(nx.sum(x, axis=0)) + x
    # D-A can accelerate the computation of GW
    return L

def laplacian(x):
    r"""Compute Laplacian matrix"""
    nx = get_backend(x)
    L = nx.diag(nx.sum(x, axis=0)) - x
    return L

def _square_loss(a, b):
    return a*a + b*b - 2*a*b

def _normalizeLaplacian(G, beta=1):
    nx = get_backend(G)
    n = G.shape[0]
    # d = nx.sum(G, axis=0)
    # d_inv_sqrt = 1/np.sqrt(d)
    
    # return nx.eye(n) + nx.dot(nx.dot(d_inv_sqrt, G), d_inv_sqrt) * beta
    return nx.eye(n) + sym_normalize_adj(G) * beta


def _gw_dist(G1, G2, log=False, p=None, q=None, G0=None):
    if p is None: p = ot.unif(G1.shape[0])
    if q is None: q = ot.unif(G2.shape[0])
    # return ot.gromov.gromov_wasserstein2(G1, G2, p, q)
    return ot.gromov.gromov_wasserstein2(G1, G2, p, q, log=log, G0=G0)

    # return (ot.gromov.gromov_wasserstein2(G1, G2, p, q, log=True), 
        # ot.gromov.init_matrix(G1, G2, p, q))
    # (d, log), (constC, hC1, hC2) = _gw_dist(G1, G2)

def _egw_dist(G1, G2, epsilon, tol):
    p = ot.unif(G1.shape[0])
    q = ot.unif(G2.shape[0])
    # return ot.gromov.entropic_gromov_wasserstein2(_normalizeLaplacian(G1),
    #      _normalizeLaplacian(G2), p, q, loss_fun='square_loss', 
    #      epsilon=epsilon, tol=tol)
    return ot.gromov.entropic_gromov_wasserstein2(G1, G2, 
        p, q, loss_fun='square_loss', epsilon=epsilon, tol=tol)

def _pgw_dist(G1, G2):
    p = ot.unif(G1.shape[0])
    q = ot.unif(G2.shape[0])
    pgw, plog = ot.gromov.pointwise_gromov_wasserstein(G1, G2, 
        p, q, _square_loss, log=True)
    return plog["gw_dist_estimated"]

def _sgw_dist(G1, G2, epsilon):
    p = ot.unif(G1.shape[0])
    q = ot.unif(G2.shape[0])
    sgw, slog = ot.gromov.sampled_gromov_wasserstein(G1, G2, 
        p, q, _square_loss, epsilon=epsilon, log=True)
    return slog["gw_dist_estimated"]

class GW_Distance(object):

    def __init__(self, X, pre_process=_normalizeLaplacian) -> None:
        self.X = [pre_process(x) for x in X]
        self.TM = [[None for i in range(len(X))] for j in range(len(X))]
        self.CX = {}
        self.curr_key = None
        self.T_init = False

    def _dist_wrapper(self, dist_func, dist_matrix, slice_, *args, **kwargs):
        """Write in-place to a slice of a distance matrix."""
        dist_matrix[:, slice_] = dist_func(*args, **kwargs)

    def _parallel_pairwise(self, Y, func, n_jobs=2, **kwds):
        """Break the pairwise matrix in n_jobs even slices
        and compute them in parallel."""

        # TODO: Y is not None
        if Y is None:
            Y_indices = slice(0, len(self.X))
            lenY = len(self.X)

        if self.curr_key is None:
            X = self.X
        else:
            (X, prob, Cp) = self.CX[self.curr_key]

        dtype = np.float32

        if effective_n_jobs(n_jobs) == 1:
            return func(Y_indices, **kwds)

        # enforce a threading backend to prevent data communication overhead
        fd = delayed(self._dist_wrapper)
        ret = np.empty((len(X), lenY), dtype=dtype, order="F")
        Parallel(backend="threading", n_jobs=n_jobs)(
            fd(func, ret, s, s, **kwds)
            for s in gen_even_slices(lenY, effective_n_jobs(n_jobs))
        )

        if (self.X is Y or Y is None):
            np.fill_diagonal(ret, 0)

        return ret

    def _pairwise_callable(self, Y_indices, metric, **kwds):
        
        lenY = Y_indices.stop - Y_indices.start
        if self.curr_key is None:
            if lenY == len(self.X):
                # Only calculate metric for upper triangle
                out = np.zeros((len(self.X), len(self.X)), dtype="float")
                iterator = itertools.combinations(range(len(self.X)), 2)
                for i, j in iterator:
                    out[i, j], log = metric(self.X[i], self.X[j], log=True,
                                                                        **kwds)
                    self.TM[i][j] = log["T"]

                # Make symmetric
                # NB: out += out.T will produce incorrect results
                out = out + out.T

                # Calculate diagonal
                # NB: nonzero diagonals are allowed for both metrics and kernels
                # for i in range(X.shape[0]):
                #     x = X[i]
                #     out[i, i] = metric(x, x, **kwds)

            else:
                # Calculate all cells
                out = np.empty((len(self.X), lenY), dtype="float")
                iterator = itertools.product(range(len(self.X)), 
                    range(Y_indices.start, Y_indices.stop))
                for i, j in iterator:
                    out[i,j-Y_indices.start], log = metric(self.X[i], self.X[j],
                                                         log=True, **kwds)
                    self.TM[i][j] = log["T"]
        else:
            (X, prob, Cp) = self.CX[self.curr_key]
            # Calculate all cells
            out = np.empty((len(X), lenY), dtype="float")
            iterator = itertools.product(range(len(X)), 
                range(Y_indices.start, Y_indices.stop))
            for i, j in iterator:
                out[i, j-Y_indices.start] = metric(X[i], X[j], log=False,
                    p=prob[i], q=prob[j], G0 = Cp[i] @ self.TM[i][j] @ Cp[j].T, 
                    **kwds)

        return out

    def compute_DistMtx(self, Y=None, metric="gromov_wasserstein", 
        n_jobs=2, eps=1e-2, tol=1e-4):

        # X = [pre_process(x) for x in self.X]

        if metric == "gromov_wasserstein":
            func = partial(self._pairwise_callable, 
                metric=_gw_dist,
            )
        elif metric == "entropic_gromov_wasserstein":
            func = partial(self._pairwise_callable, 
                metric=_egw_dist, epsilon=eps, tol=tol
            )
        elif metric == "pointwise_gromov_wasserstein":
            func = partial(self._pairwise_callable, 
                metric=_pgw_dist,
            )
        elif metric == "sampled_gromov_wasserstein":
            func = partial(self._pairwise_callable, 
                metric=_sgw_dist, epsilon=eps,
            )

        res = self._parallel_pairwise(Y, func, n_jobs)
        # print(res)
        iterator = itertools.combinations(range(len(self.X)), 2)
        for i, j in iterator:
            if res[i, j] >= res[j, i]:
                res[i, j] = res[j, i]
                self.TM[i][j] = self.TM[j][i].T
            else:
                res[j, i] = res[i, j]
                self.TM[j][i] = self.TM[i][j].T
        
        res[np.abs(res) < 1e-9] = 0

        return res

    def compute_coarsenedDistMtx(self, X, X_key, prob, Cp,
        metric="gromov_wasserstein", n_jobs=2, eps=1e-2, tol=1e-4):

        self.CX[X_key] = (X, prob, Cp)
        self.curr_key = X_key

        if metric == "gromov_wasserstein":
            func = partial(self._pairwise_callable, 
                metric=_gw_dist,
            )
        elif metric == "entropic_gromov_wasserstein":
            func = partial(self._pairwise_callable, 
                metric=_egw_dist, epsilon=eps, tol=tol
            )
        elif metric == "pointwise_gromov_wasserstein":
            func = partial(self._pairwise_callable, 
                metric=_pgw_dist,
            )
        elif metric == "sampled_gromov_wasserstein":
            func = partial(self._pairwise_callable, 
                metric=_sgw_dist, epsilon=eps,
            )

        res = self._parallel_pairwise(None, func, n_jobs)
        res[np.abs(res) < 1e-9] = 0

        self.curr_key = None

        return res

    def compute_pairedDist(self, X, X_key, prob, Cp):
        
        # metric="gromov_wasserstein"

        # X_key is the name of the coarsening method
        self.CX[X_key] = (X, prob, Cp)
        self.curr_key = X_key

        res = np.empty((len(X), 3), dtype="float")
        for i in range(len(X)):
            n, N = Cp[i].shape
            res[i, 0] = _gw_dist(X[i], self.X[i], log=False, 
                                 p=prob[i], G0 = Cp[i]/N)

            Qw = util.orthoW_Q(Cp[i].T, None) # = Cw.T
            U = self.X[i]
            PiUPi = util.multiply_Q(U, Qw)

            eigvals = eigvalsh(U)[::-1]
            eigvals_c = eigvalsh(PiUPi)[::-1]

            res[i, 1] = (eigvals[:n] - eigvals_c).sum() / N / N

            res[i, 2] = (eigvals[N-n] * res[i, 1] + (eigvals * eigvals).sum() 
                         - np.dot(eigvals[:n], eigvals[-n:]))
            res[i, 2] = res[i, 2] / N / N

        res[np.abs(res) < 1e-9] = 0
        self.curr_key = None

        return res

import util
import numpy as np
import numpy.linalg as LA
from scipy.sparse.linalg import eigs

from numpy.linalg import eig

#def Laplacian(G):
def sym_normalize_adj(adj):
    deg = adj.sum(1)
    deg_inv = np.where(deg > 0, 1. / np.sqrt(deg), 0)
    return np.einsum('i,ij,j->ij', deg_inv, adj, deg_inv)

def GCN_GC(adj, Q):
    L = sym_normalize_adj(adj)
    return util.lift_Q(Q).T @ L @ Q

def normalizeLaplacian(G):
    n = G.shape[0]
    # d = np.sum(G, axis = 0)
    # d_inv_sqrt = 1/np.sqrt(d)
    # d_inv_sqrt = np.diag(d_inv_sqrt)
    return np.eye(n) - sym_normalize_adj(G)


def spectraLaplacian(G):
    L = normalizeLaplacian(G)
    L = (L + np.transpose(L)) / 2
    e, v = LA.eig(L)
    e = np.real(e)
    v = np.real(v)
    e_tmp = -e
    idx = e_tmp.argsort()[::-1]
    e = e[idx]
    v = v[:,idx]
    return e, v

def spectraLaplacian_two_end_n(G, n):
    N = G.shape[0]
    assert n <= N
    L = normalizeLaplacian(G)
    L = (L + np.transpose(L)) / 2
    e, v = eig(L)
    # e1, v1 = eigs(L, k = n, which = 'SM')
    # e1 = np.real(e1)
    # v1 = np.real(v1)
    # e1_tmp = -e1
    # idx = e1_tmp.argsort()[::-1]
    # e1 = e1[idx]
    # v1 = v1[:,idx]
    # e2, v2 = eigs(L, k = n, which = 'LM')
    # e2 = np.real(e2)
    # v2 = np.real(v2)
    # e2_tmp = -e2
    # idx = e2_tmp.argsort()[::-1]
    # e2 = e2[idx]
    # v2 = v2[:,idx]

    e = np.real(e)
    v = np.real(v)
    e_tmp = -e
    idx = e_tmp.argsort()[::-1]
    e = e[idx]
    v = v[:,idx]
    e1 = e[0:n]
    v1 = v[:, 0:n]
    e2 = e[N-n:N]
    v2 = v[:, N-n:N]
    return e1, v1, e2, v2

def spectraLaplacian_top_n(G, n):
    assert n <= G.shape[0]

    L = normalizeLaplacian(G)
    L = (L + np.transpose(L)) / 2

    e1, v1 = eig(L)
    # e1, v1 = eigs(L, k = n, which = 'SM')
    e1 = np.real(e1)
    v1 = np.real(v1)
    e1_tmp = -e1
    idx = e1_tmp.argsort()[::-1]
    e1 = e1[idx]
    e1 = e1[0:n]
    v1 = v1[:,idx]
    v1 = v1[:,0:n]
    return e1, v1

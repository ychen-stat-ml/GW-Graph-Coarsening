import random, sys, functools
import numpy as np
import numpy.linalg as LA
import networkx as nx

from sklearn.cluster import KMeans

import coarsening


def multiply_Q(G, Q):
    Gc = np.dot(np.dot(np.transpose(Q), G), Q)
    return Gc

def multiply_Q_lift(Gc, Q):
    G = np.dot(np.dot(Q, Gc), np.transpose(Gc))
    return G

def idx2Q(idx, n):
    N = idx.shape[0]
    Q = np.zeros((N, n))
    for i in range(N):
        Q[i, idx[i]] = 1
    return Q

def Q2idx(Q):
    N = Q.shape[0]
    n = Q.shape[1]
    idx = np.zeros(N, dtype=np.int32)
    for i in range(N):
        for j in range(n):
            if Q[i, j] > 0:
                idx[i] = j
    return idx

def random_two_nodes(n):
    perm = np.random.permutation(n)
    return perm[0], perm[1]

def merge_two_nodes(n, a, b):

    assert a != b and a < n and b < n
    Q = np.zeros((n, n-1))
    cur = 0
    for i in range(n):
        if i == a or i == b:
            Q[i, n-2] = 1
        else:
            Q[i, cur] = 1
            cur = cur + 1
    return Q

def lift_Q(Q):
    # Q = Cp.T
    # return Q0: \bar{C}.T
    N = Q.shape[0]
    n = Q.shape[1]
    idx = np.zeros(N, dtype=np.int16)
    for i in range(N):
        for j in range(n):
            if Q[i, j] == 1:
                idx[i] = j
    d = np.zeros((n, 1))
    for i in range(N):
        d[idx[i]] = d[idx[i]] + 1

    Q2 = np.zeros((N, n))
    for i in range(N):
        Q2[i, idx[i]] = 1/d[idx[i]]
    return Q2

def ortho_Q(Q):
    # Q = Cp.T
    # return Q1: C.T
    N = Q.shape[0]
    n = Q.shape[1]
    idx = np.zeros(N, dtype=np.int16)
    for i in range(N):
        for j in range(n):
            if Q[i, j] == 1:
                idx[i] = j
    d = np.zeros((n, 1))
    for i in range(N):
        d[idx[i]] = d[idx[i]] + 1

    Q2 = np.zeros((N, n))
    for i in range(N):
        Q2[i, idx[i]] = 1 / np.sqrt(d[idx[i]])
    return Q2

def orthoW_Q(Q, w):
    # Q = Cp.T
    # return Q1: C.T
    N = Q.shape[0]
    n = Q.shape[1]
    if w is None:
        w = np.ones(N)
    c = Q.T @ w

    Q2 = (1/np.sqrt(c) * Q).T * w**0.5

    return Q2.T

def getReducedN(N, k, ratio):
    n = int(np.ceil(ratio*N)) # ratio decides the # of clusters
    n = min(max([n, k]), N)

    return n

def get_h_init(Q, w=None):
    N = Q.shape[0]
    if w is None: 
        w = np.ones(N) / N
    else:
        w = w / np.sum(w)

    nn = Q.T @ w # n
    h = w * (Q / nn).T

    return h

METHODS = ['mgc', 'sgc', 'wgc', 'vgc', 'vegc',
                'sgwl', 'eig', 'none']
OTHER_METHODS = ['vgc', 'vegc', 'mgc', 'sgc', 'sgwl', 'eig', 'none']

def assign_parser(parser):
    # parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    parser.add_argument('--dataset', type=str, default="MUTAG",
                            help='name of dataset (default: MUTAG)')
    parser.add_argument('--method', type=str, default="mgc",
                            help='name of the coarsening method')
    parser.add_argument('--ratio', type=float, default=0.2,
                        help='the ratio between coarse and original graphs n/N')
    parser.add_argument('--NmaxRatio', action='store_true',
                help='in exp1, whether use the Nmax/logNmax ratio.')
    parser.add_argument('--runs', type=int, default=10,
                    help='number of the runs, since coarsening method is random')
    parser.add_argument('--alm', type=str, default="",
                help='specify the agglomerative hierarchical clustering for wgc')
    parser.add_argument('--weighted', type=int, default=0,
                help='whether use weighted kernel kmeans in wgc')
    parser.add_argument('--normalized', type=int, default=0,
                help='whether use normalized laplacian in wgc')
    parser.add_argument('--cscale', action='store_true',
                help='in exp1, whether use the correct scale for other methods.')
    parser.add_argument('--ninit', type=int, default=10,
                        help='n_init parameter for kmeans')
    parser.add_argument('--seed', type=int, default=42,
                        help='the ratio between coarse and original graphs n/N')
    parser.add_argument('--save', type=int, default=0,
                        help='if 1 then save the `res` variable')

def check_args(args):
    if args.dataset not in ["MUTAG", "PROTEINS", "IMDB-BINARY", "tumblr_ct1", 
                            "MSRC_9", "PTC_MR", "AQSOL", "ZINC"]:
        print("Incorrect input dataset")
        sys.exit()
    if args.method not in METHODS:
        print(args.method, "Incorrect input coarsening method", METHODS)
        sys.exit()
    if args.ratio < 0 or args.ratio > 1:
        print("Incorrect input ratio")
        sys.exit()

def non_coarse(G, n):
    return G, np.eye(G.shape[0]), None

def assign_method(args):
    if args.method == "vgc":
        coarse_method = functools.partial(coarsening.
                    template_graph_coarsening, method="variation_neighborhoods")
    elif args.method == "vegc":
        coarse_method = functools.partial(coarsening.
                    template_graph_coarsening, method="variation_edges")
    elif args.method == "mgc":
        coarse_method = coarsening.multilevel_graph_coarsening
    elif args.method == "sgc":
        coarse_method = coarsening.spectral_graph_coarsening
    elif args.method == "wgc":
        coarse_method = coarsening.weighted_graph_coarsening
    elif args.method == 'sgwl':
        coarse_method = coarsening.SGWL
    elif args.method == "none":
        coarse_method = non_coarse
    elif args.method == "eig":
        coarse_method = non_coarse # not used in the script
    
    return coarse_method
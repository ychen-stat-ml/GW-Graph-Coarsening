import random, sys
import numpy as np
import numpy.linalg as LA
import networkx as nx
import util
import measure
import laplacian
from sklearn.cluster import KMeans


def multilevel_graph_coarsening(G, n, **kawgs):
    N = G.shape[0]
    Gc = G
    cur_size = N
    Q = np.eye(N)
    while cur_size > n:
        stop_flag = 0
        max_dist = 10000
        max_dist_a = -1
        max_dist_b = -1
        for i in range(cur_size):
            for j in range(i+1, cur_size):
                dist = measure.normalized_L1(Gc[i], Gc[j])
                if dist < max_dist:
                    max_dist = dist
                    max_dist_a = i
                    max_dist_b = j
            #     if dist < 0.001:
            #         stop_flag = 1
            #         break
            # if stop_flag == 1:
            #     break
        if max_dist_a == -1:
            max_dist_a, max_dist_b = util.random_two_nodes(cur_size)
        cur_Q = util.merge_two_nodes(cur_size, max_dist_a, max_dist_b)
        Q = np.dot(Q, cur_Q)
        # cur_Q is C_p
        Gc = util.multiply_Q(Gc, cur_Q)
        # cur_size = cur_size - 1
        # ensure cur_size = Gc.shape[0], while is the same as the line above
        cur_size = Gc.shape[0]
    idx = util.Q2idx(Q)
    # Q = Cp.T

    if n != Q.shape[1]:
        print(n, Q.shape[1]-n)
        n = Q.shape[1]

    return Gc, Q, idx

def regular_partition(N, n):
    if N%n == 0:
        block_size = N//n + 1
    else:
        block_size = N//(n-1) + 1
    idx = np.zeros(N, dtype=np.int32)
    for i in range(N):
        idx[i] = i//block_size
    return idx

from graph_coarsening.coarsening_utils import *
import pygsp

def template_graph_coarsening(am, n, method, **kawgs):
    G = pygsp.graphs.Graph(am)
    N = G.N
    # r = 1 - n / N
    C, _, _, _ = coarsen(G, r=n, method=method)
    if n != C.shape[0]:
        print(n, C.shape[0]-n)
        n = C.shape[0]
    # C is orthogonal

    Q = np.zeros((N, n), dtype=np.int32)
    Q[C.toarray().T > 0] = 1
    idx = util.Q2idx(Q)

    # only Gc is used
    return util.multiply_Q(am, Q), Q, idx

def variation_nbhd_graph_coarsening(am, n, **kawgs):
    G = pygsp.graphs.Graph(am)
    N = G.N
    # r = 1 - n / N
    C, _, _, _ = coarsen(G, r=n, method="variation_neighborhood")
    if n != C.shape[0]:
        print(n, C.shape[0])
        n = C.shape[0]
    # C is orthogonal

    Q = np.zeros((N, n), dtype=np.int32)
    Q[C.toarray().T > 0] = 1
    idx = util.Q2idx(Q)

    # only Gc is used
    return util.multiply_Q(am, Q), Q, idx

def variation_edge_graph_coarsening(am, n, **kawgs):
    G = pygsp.graphs.Graph(am)
    N = G.N
    # r = 1 - n / N
    C, _, _, _ = coarsen(G, r=n, method="variation_edges")
    n = C.shape[0]

    # C is orthogonal

    Q = np.zeros((N, n), dtype=np.int32)
    Q[C.toarray().T > 0] = 1
    idx = util.Q2idx(Q)

    # only Gc is used
    return util.multiply_Q(am, Q), Q, idx

def spectral_graph_coarsening(G, n, **kawgs):
    N = G.shape[0]
    e1, v1, e2, v2 = laplacian.spectraLaplacian_two_end_n(G, n)
    min_dist = n+1

    if n >= N: 
        Q = np.eye(N)
        return G, Q, util.Q2idx(Q)

    for k in range(0, n):
        if e1[k] <= 1:
            if k+1 < n and e2[k+1] < 1:
                continue
            if k+1 < n and e2[k+1] >= 1:
                v_all = np.concatenate((v1[:, 0:(k+1)], v2[:, (k+1):n]), axis = 1)
            elif k == n-1:
                v_all = v1[:, 0:n]

            kmeans = KMeans(n_clusters=n).fit(v_all)
            idx = kmeans.labels_
            sumd = kmeans.inertia_
            Q = util.idx2Q(idx, n) # 不会出现cluster数目比预设的n更大的情况
            Gc = util.multiply_Q(G, Q)
            # return the eigenvalue/vectors of the normalized laplacian
            ec, vc = laplacian.spectraLaplacian(Gc)
            dist = measure.eig_partial_dist_k_two_end_n(e1, e2, ec, k)
            if dist < min_dist:
                min_dist = dist
                idx_min = idx
                min_sumd = sumd
                Gc_min = Gc
                Q_min = Q
    return Gc_min, Q_min, idx_min

from pygkernels.cluster import KKMeans, KKMeans_iterative

def weighted_graph_coarsening(S, n, scale=0, seed=42, n_init=10, 
        sample_weight=None, init='k-means++', h_init=None, tol_empty=False):
    # S: the similarity mtx
    # n: the targeted number of clusters
    # scale: 0 return \barC S \barC.T; 1 return C S C.T; 2 return Cp S Cp.T
    N = S.shape[0]
    if n >= N: 
        Q = np.eye(N)
        return S, Q, util.Q2idx(Q)

    kmeans = KKMeans(n_clusters=n, n_init=n_init, init=init, 
                                    init_measure='inertia', random_state=seed)
    idx = kmeans.predict(S, A=h_init, sample_weight=sample_weight, tol_empty=tol_empty)
    if idx is None:
        # sys.exit()
        print("specified initialization failed. turn to kmeans++")
        kmeans = KKMeans(n_clusters=n, n_init=n_init, init="k-means++", 
                                    init_measure='inertia', random_state=seed)
        idx = kmeans.predict(S, A=None, sample_weight=sample_weight, tol_empty=tol_empty)
    
    # if init == "specified" and n != h_init.shape[0]: 
    #     # print(idx, n, np.unique(idx), h_init.shape[0]-n)
    #     print(h_init.shape[0]-n)
    #     n = h_init.shape[0]
    Q = util.idx2Q(idx, n)

    if scale == 0: 
        Q2 = util.lift_Q(Q)
    elif scale == 1: 
        # Q2 = util.ortho_Q(Q)
        Q2 = util.orthoW_Q(Q, sample_weight)
    else:
        Q2 = Q

    Gc = util.multiply_Q(S, Q2)

    return Gc, Q, idx

def spectral_clustering(G, n):
    N = G.shape[0]
    e1, v1 = laplacian.spectraLaplacian_top_n(G, n)
    v_all = v1[:, 0:n]
    kmeans = KMeans(n_clusters=n).fit(v_all)
    idx = kmeans.labels_
    sumd = kmeans.inertia_
    Q = util.idx2Q(idx, n)
    Gc = util.multiply_Q(G, Q)
    return Gc, Q, idx

def get_random_partition(N, n):
    for i in range(500):
        flag = True
        a = np.zeros(N, dtype=np.int64)
        cnt = np.zeros(n, dtype=np.int64)
        for j in range(N):
            a[j] = random.randint(0, n-1)
            cnt[a[j]] += 1
        for j in range(n):
            if cnt[j] == 0:
                flag = False
                break
        if flag == False:
            continue
        else:
            break
    return a


# import methods.DataIO as DataIO
# import methods.GromovWassersteinGraphToolkit as GwGt

import importlib
DataIO = importlib.import_module("s-gwl.methods.DataIO")
GwGt = importlib.import_module("s-gwl.methods.GromovWassersteinGraphToolkit")

def SGWL(G, n, **kawgs):
    # G must be a networkX graph
    if not isinstance(G, nx.Graph):
        G = nx.from_numpy_matrix(G)
    am = nx.to_numpy_array(G)
    num_nodes = G.number_of_nodes()

    p_s, cost_s, idx2node = DataIO.extract_graph_info(G)
    p_s = (p_s + 1) ** 0.01
    p_s /= np.sum(p_s)

    ot_dict = {'loss_type': 'L2',  # the key hyperparameters of GW distance
                'ot_method': 'proximal',
                'beta': 0.15,
                'outer_iteration': 2 * num_nodes,  # outer, inner iterations and error bound of optimal transport
                'iter_bound': 1e-30,
                'inner_iteration': 5,
                'sk_bound': 1e-30,
                'node_prior': 0.0001,
                'max_iter': 1,  # iteration and error bound for calcuating barycenter
                'cost_bound': 1e-16,
                'update_p': False,  # optional updates of source distribution
                'lr': 0,
                'alpha': 0}

    _, _, sub_idx2nodes = GwGt.recursive_graph_partition_flex(
                                                0.5 * (cost_s + cost_s.T),
                                                p_s,
                                                idx2node,
                                                ot_dict,
                                                n,
                                                max_node_num=300)

    est_idx = np.zeros((num_nodes,), dtype=np.int)
    # n_cluster: cluster_idx
    for n_cluster in range(len(sub_idx2nodes)):
        for key in sub_idx2nodes[n_cluster].keys():
            idx = sub_idx2nodes[n_cluster][key]
            est_idx[idx] = n_cluster
    print(n, len(sub_idx2nodes), est_idx)
    Q = util.idx2Q(est_idx, n)
    Gc = util.multiply_Q(am, Q)

    return Gc, Q, est_idx
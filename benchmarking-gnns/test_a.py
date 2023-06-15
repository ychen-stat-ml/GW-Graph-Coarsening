import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import sys
sys.path.append("..")

import argparse
import numpy as np
from scipy.linalg import eigh
import coarsening
import util
import networkx as nx
import parse
from classification import _normalizeLaplacian
from laplacian import GCN_GC, sym_normalize_adj

import explog
import torch, dgl
# from data.superpixels import DGLFormDataset

from data.data import LoadData # import dataset

parser = argparse.ArgumentParser(description='Experiment for graph classification with coarse graphs')
util.assign_parser(parser)
args = parser.parse_args()
util.check_args(args)

explog.set_all_seeds(args.seed)

# am: adjacency matrix
# labels: +1/-1

coarse_method = util.assign_method(args)

print(args.dataset, args.method, args.alm, args.seed, args.normalized, args.weighted, args.ninit)

k = 5
# number of nodes to keep
# r = [getReducedN(x.shape[0], k, args.ratio) for x in am]
# weight for nodes
# if args.weighted:
#     d = [am[i].sum(1) for i in range(num_samples)]
# else:
#     d = [None] * num_samples


dataset = LoadData(args.dataset)
if args.method=="wgc" and args.alm and args.alm != "wgc":
    # m_dataset = explog.read_from("./data/molecules/AQSOL_vgc_0.pkl")
    m_dataset = explog.read_from("./data/molecules/{}_{}_{}.pkl".
                    format(args.dataset, args.alm, args.seed))
    m_ref = {"train": m_dataset.train, 'val': m_dataset.val, 'test': m_dataset.test}

'''
# count the total number of edges in the dataset for the table in the paper
edge_count = 0
for gset in [dataset.train, dataset.val, dataset.test]:
    for i, g in enumerate(gset.graph_lists):
        # g_nx = g.to_networkx()
        am_i = g.adjacency_matrix(scipy_fmt="csr")
        edge_count += am_i.count_nonzero() / 2

print(edge_count)
sys.exit()
'''

for gset in [dataset.train, dataset.val, dataset.test]:
    gset.Q = []
    for i, g in enumerate(gset.graph_lists):
        # g_nx = g.to_networkx()
        am_i = g.adjacency_matrix(scipy_fmt="csr").toarray()
        N = am_i.shape[0]
        n = util.getReducedN(N, k, args.ratio)
        
        try:
            if args.method in ["wgc"]:
                # the input and output of wgc are different than other coarsening methods
                ref = _normalizeLaplacian(am_i)
                if args.alm and args.alm != "wgc":
                    n = m_ref[gset.split].Q[i].shape[1]
                    h_init = util.get_h_init(m_ref[gset.split].Q[i])
                    # print(n, h_init.shape)
                    Gc, Q, _ = coarse_method(ref, n, scale=2, seed=args.seed, 
                                init='specified', h_init=h_init, tol_empty=False)
                else:
                    Gc, Q, _ = coarse_method(ref, n, scale=2, seed=args.seed,
                                            tol_empty=False)
                # not need to compute Gc=Cp A Cp.T                
                # Gc = util.multiply_Q(am_i, Q)

            else:
                Gc, Q, _ = coarse_method(am_i, n)
            
            if np.all(Gc < 1e-9):
                raise Exception("All-zero ")
        except Exception as e:
            print(e)
            print(i, "Use mgc instead for this graph")
            Gc, Q, _ = coarsening.multilevel_graph_coarsening(am_i, n)
        
        Gc = GCN_GC(am_i, Q)
        gset.Q.append(Q)

        Gc = nx.DiGraph(Gc)
        G = dgl.from_networkx(Gc, edge_attrs=['weight'])
        U = np.zeros((N, dataset.num_atom_type))
        U[range(N), g.ndata['feat']] = 1
        G.ndata['feat'] = torch.Tensor(util.lift_Q(Q).T @ U)
        # for the meanpool of the last layer in GCN
        # In DGL, it is implemented as
        # x = x * graph.nodes[ntype].data[weight]
        # e.g., x.shape = torch.Size([748, 145])
        # which requires the shape of G.ndata['w'] to be [748, 1]
        G.ndata['w'] = torch.Tensor(Q.sum(0, keepdims=True).T) * n/N
        
        # edge_type is not used in GCN
        G.edata['feat'] = torch.zeros(Gc.number_of_edges()) 

        gset.graph_lists[i] = G

        if i>0 and i % 400 == 0:
            print(gset.split, i)
            # sys.exit()

if args.method == args.alm:
    explog.save_to(dataset, "./data/molecules/{}_{}_{}.pkl".
        format(args.dataset, args.alm, args.seed))
elif args.alm:
    explog.save_to(dataset, "./data/molecules/{}_{}_{}_{}.pkl".
        format(args.dataset, args.method, args.alm, args.seed))
from cmath import exp
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import numpy as np
from scipy.linalg import eigh

import coarsening
import util
import networkx as nx
import parse
from classification import _normalizeLaplacian, _laplacian, laplacian
import classification
from laplacian import normalizeLaplacian

import sys

import netlsd
import explog

parser = argparse.ArgumentParser(description='Experiment for graph classification with coarse graphs')
util.assign_parser(parser)
args = parser.parse_args()
util.check_args(args)

explog.set_all_seeds(args.seed)

dir = 'dataset'
am, labels = parse.parse_dataset(dir, args.dataset)

# am: adjacency matrix
# labels: +1/-1

coarse_method = util.assign_method(args)

print(args.dataset, args.method, args.alm, args.seed, args.normalized, args.weighted, args.ninit)


num_samples = len(am) # graph的个数

h = 250
X = np.zeros((num_samples, h))
Y = labels
cam = []
if args.alm and args.method == 'wgc':

    if args.weighted:
        d = [am_i.sum(1) for am_i in am]
    else:
        d = [None] * num_samples

    cam = explog.read_from("./dataset/exp3/{}_{}_{}.pkl".
                            format(args.dataset, args.alm, args.seed))
    h_init = [util.get_h_init(cam[i][1], d[i]) for i in range(num_samples)]

for i in range(num_samples):
    N = am[i].shape[0]

    n = int(np.ceil(args.ratio*N)) # ratio decides the # of clusters
    
    if args.method in ['wgc']:
        # print("Iter: {}, {}".format(i, am[i].shape[0]))
        if args.weighted:
            d = am[i].sum(1)
        else:
            d = None
        
        if args.normalized:
            G = _normalizeLaplacian(am[i], beta=1) # I + DAD
        else:    
            G = _laplacian(am[i]) # D + A

        if args.alm:
            Gc, Q, idx = coarse_method(G, n=h_init[i].shape[0], scale=0, h_init=h_init[i], 
                seed=args.seed, n_init=1, sample_weight=d, init='specified')


        else:
            Gc, Q, idx = coarse_method(G, n, scale=0, seed=args.seed, 
                            n_init=args.ninit, sample_weight=d)

        Gc = util.multiply_Q(am[i], Q)
        Gc = normalizeLaplacian(Gc)
        G = eigh(Gc, eigvals_only=True)

    elif args.method in ['eig']:
        G = eigh(normalizeLaplacian(am[i]), eigvals_only=True)[:n]

    else:
        try:
            Gc, Q, idx = coarse_method(am[i], n)
        except Exception as e:
            print(e)
            print(i, "Use mgc instead for this graph")
            Gc, Q, idx = coarsening.multilevel_graph_coarsening(am[i], n)

        cam.append((Gc, Q, idx))
        G = nx.from_numpy_matrix(Gc)


        
    t = np.logspace(-2, 2, h)
    X[i] = netlsd.heat(G, t, normalization="empty")

acc, std = classification.KNN_classifier_nfold(X, Y, n=10, k=1)
if args.alm and args.method != "wgc":
    explog.save_to(cam, "./dataset/exp3/{}_{}_{}.pkl".
                            format(args.dataset, args.alm, args.seed))

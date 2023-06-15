import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import argparse
import numpy as np
from scipy.linalg import eigh
import coarsening
import util
import networkx as nx
import parse
from classification import _normalizeLaplacian, GW_Distance
from laplacian import normalizeLaplacian

import sys
import explog

parser = argparse.ArgumentParser(description=
                    'Experiment for graph classification with coarse graphs')
util.assign_parser(parser)
args = parser.parse_args()

explog.set_all_seeds(args.seed)

util.check_args(args)
coarse_method = util.assign_method(args)

dir = 'dataset'
am, labels = parse.parse_dataset(dir, args.dataset)


# sys.exit()

am = [x.astype(np.int32) for x in am]

# am: adjacency matrix
# labels: +1/-1
# dm: distance matrix

print(args.method, args.dataset, args.ratio)

@explog.get_time_output
def coarsen_wrapper(coarse_method, am, r, d, **kawgs):
    # only wgc will use sample_weight=d[i]
    clap = [coarse_method(am[i], r[i], sample_weight=d[i], **kawgs)
                 for i in range(len(am))]
    return clap

@explog.get_time_output
def alm_coarsen_wrapper(coarse_method, am, r, h_init, d, **kawgs):
    clap = [coarse_method(am[i], h_init[i].shape[0], h_init=h_init[i], sample_weight=d[i],
                **kawgs) for i in range(len(am))]
    return clap

def getTopKspectrum(normL, k=4, reverse=False):
    # print(normL)
    spectrum = eigh(normL, eigvals_only=True)
    n = len(spectrum)

    if reverse:
        spectrum = spectrum[::-1]

    if n >= k:
        return spectrum[:k]
    else:
        return np.concatenate([spectrum, np.zeros(k-n)])


num_samples = len(am)
k = 5
reverse = True
ref = [_normalizeLaplacian(am[i]) for i in range(num_samples)]
refX = np.array([getTopKspectrum(x, k, reverse) for x in ref])

# number of nodes to keep
r = [util.getReducedN(x.shape[0], k, args.ratio) for x in am]
# weight for nodes
if args.weighted:
    d = [am[i].sum(1) for i in range(num_samples)]
else:
    d = [None] * num_samples

res = np.zeros((args.runs, 2))
for j in range(args.runs):
    if args.method in util.OTHER_METHODS:
        cam, res[j, 1] = coarsen_wrapper(coarse_method, am, r, d)
        # X = [_normalizeLaplacian(c[0]) for c in cam] # Gc, Q, idx
        X = [util.multiply_Q(_normalizeLaplacian(am[i]), 
                util.ortho_Q(cam[i][1])) for i in range(num_samples)]

    elif args.method in ['wgc']:
        if args.alm:
            cam = explog.read_from("./dataset/exp2/{}_{}.pkl".format(args.alm, args.ratio))
            h_init = [util.get_h_init(cam[i][1], d[i]) for i in range(num_samples)]

            cam, res[j, 1] = alm_coarsen_wrapper(coarse_method, ref, r, h_init,
                d, seed=args.seed+j, n_init=10, init='specified')
        else:
            cam, res[j, 1] = coarsen_wrapper(coarse_method, ref, r, d,
                                                seed=args.seed+j, n_init=1)
        X = [util.multiply_Q(_normalizeLaplacian(am[i]), 
                util.orthoW_Q(cam[i][1], d[i])) for i in range(num_samples)]
    
    X = np.array([getTopKspectrum(x, k, reverse) for x in X])

    
    res[j, 0] = ((refX - X) / refX).mean()

    print(res)

if args.alm and args.method != "wgc":
    explog.save_to(cam, "./dataset/exp2/{}_{}.pkl".format(args.alm, args.ratio))

if args.save:
    explog.save_to(res, "./dataset/exp2/tumblr_{}_{}_{}_{}.pkl".format(
        args.method, args.ratio, args.alm, args.weighted))

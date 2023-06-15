import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import argparse, sys
import numpy as np
import util, coarsening
import networkx as nx
import parse, math
from classification import _normalizeLaplacian, GW_Distance
from laplacian import sym_normalize_adj

import explog

def catch(func, i, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(e); print(i, "Use mgc instead for this graph")
        return coarsening.multilevel_graph_coarsening(*args, **kwargs)

# modify n = 0.3 N
@explog.get_time_output
def coarsen_wrapper(coarse_method, am, r, **kawgs):
    clap = [coarse_method(am[i], r[i], **kawgs) for i in range(len(am))]
    return clap

@explog.get_time_output
def v_coarsen_wrapper(coarse_method, am, r, **kawgs):
    cam = [catch(coarse_method, i, am[i], r[i], **kawgs) for i in range(len(r))]
    return cam

@explog.get_time_output
def alm_coarsen_wrapper(coarse_method, am, h_init, **kawgs):
    clap = [coarse_method(am[i], h_init[i].shape[0], h_init=h_init[i], **kawgs)
                 for i in range(len(am))]
    return clap

@explog.get_time_output
def alm_coarsen_wrapper(coarse_method, am, r, h_init, **kawgs):
    clap = [coarse_method(am[i], h_init[i].shape[0], h_init=h_init[i], **kawgs) 
                for i in range(len(am))]
    return clap

parser = argparse.ArgumentParser(description=
                    'Experiment for graph classification with coarse graphs')
util.assign_parser(parser)
args = parser.parse_args()

util.check_args(args)
coarse_method = util.assign_method(args)

dir = 'dataset'
am, labels = parse.parse_dataset(dir, args.dataset)

# am: adjacency matrix
# labels: +1/-1
# dm: distance matrix

(mutag, mutag_squared_dm) = explog.read_from(
                        "./dataset/exp1/{}_gw_distMtx.pkl".format(args.dataset))

print(args.method, args.dataset, args.alm, args.ratio)

res = np.zeros((args.runs, 2))

if args.alm:
    res_save = []
    if args.method == "wgc":
        cams = explog.read_from("./dataset/exp1/alm_{}_{}_{}.pkl".
        format(args.dataset, args.alm, args.ratio))
    # am = [sym_normalize_adj(g) for g in am]

# number of nodes to keep
if args.NmaxRatio:
    print("Using Nmax setting")
    Nmax = max([x.shape[0] for x in am])
    k = math.floor(Nmax / np.log(Nmax))
    r = [util.getReducedN(x.shape[0], k, 0.0) for x in am]
else:
    r = [util.getReducedN(x.shape[0], 5, args.ratio) for x in am]

for j in range(args.runs):

    if args.method in util.OTHER_METHODS:
        if args.method[0] == 'v':
            print("test")
            cam, res[j, 1] = v_coarsen_wrapper(coarse_method, am, r)
        else:
            cam, res[j, 1] = coarsen_wrapper(coarse_method, am, r)
        # cam, res[j, 1] = coarsen_wrapper(coarse_method, mutag.X)

        if args.cscale:
            # correct scale
            X = [util.multiply_Q(mutag.X[i], util.lift_Q(cam[i][1])) 
                    for i in range(len(am))]
        else:
            # ortho scale
            X = [_normalizeLaplacian(c[0]) for c in cam]

        if args.alm: res_save.append(cam)
    else:
        if args.alm:
            cam = cams[j]
            h_init = [util.get_h_init(cam[i][1]) for i in range(len(am))]
            cam, res[j, 1] = alm_coarsen_wrapper(coarse_method, mutag.X, r, 
                h_init, seed=args.seed+j, n_init=5, init='specified')
        else:
            cam, res[j, 1] = coarsen_wrapper(coarse_method, mutag.X, r,
                                                # seed=args.seed+j, n_init=10)
                                                seed=args.seed+j, n_init=1)
        X = [c[0] for c in cam]
    
    prob = [c[1].sum(0) / c[1].sum() for c in cam]
    Cp = [c[1].T for c in cam]

    print("coarsening is done")

    temp_dm = mutag.compute_coarsenedDistMtx(X, args.method, prob, Cp)

    res[j, 0] = np.linalg.norm(mutag_squared_dm**0.5 - temp_dm**0.5)
    # res[j, 0] = np.linalg.norm(mutag_squared_dm**0.5 - temp_dm**0.5, ord=2)

    print(res[j])
    

print(res.mean(0), res.std(0, ddof=1) / np.sqrt(args.runs))
print(res)

if args.alm and args.method != "wgc":
    explog.save_to(res_save, "./dataset/exp1/alm_{}_{}_{}.pkl".
        format(args.dataset, args.method, args.ratio))


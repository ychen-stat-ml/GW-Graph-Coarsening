import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import parse
from classification import GW_Distance

import explog

# dataset = "PTC_MR"
dataset = "IMDB-BINARY"

dir = 'dataset'
am, labels = parse.parse_dataset(dir, dataset)

a = GW_Distance(am)
res = a.compute_DistMtx(n_jobs=10)

explog.save_to((a, res), "./dataset/exp1/{}_gw_distMtx.pkl".format(dataset))
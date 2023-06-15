code=test_distM.py
dataset=PTC_MR
log="scripts/exp1.txt"

# specify the dataset to process in "prepare_distM.py", to pre-compute the original GW distance matrix
python prepare_distM.py

# First run mgc method to save the coarsening results
python ${code} --dataset ${dataset} --method mgc --runs 10 --NmaxRatio --cscale --alm mgc >> ${log}
# run regular KGC method with option "wgc"
python ${code} --dataset ${dataset} --method wgc --runs 10 --NmaxRatio --cscale >> ${log}
# run KGC method with initialization saved from "mgc"
python ${code} --dataset ${dataset} --method wgc --runs 10 --NmaxRatio --cscale --alm mgc >> ${log}
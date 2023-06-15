code=test_clsf.py
dataset=tumblr_ct1
seed=12

echo "${seed}"
CUDA_VISIBLE_DEVICES= python $code --dataset ${dataset} --method wgc --alm vgc \
        --ratio 0.2 --seed $seed --normalized 0 --weighted 0
CUDA_VISIBLE_DEVICES= python $code --dataset ${dataset} --method wgc --alm vgc \
        --ratio 0.2 --seed $seed --normalized 0 --weighted 1
CUDA_VISIBLE_DEVICES= python $code --dataset ${dataset} --method wgc --alm vgc \
        --ratio 0.2 --seed $seed --normalized 1 --weighted 0
CUDA_VISIBLE_DEVICES= python $code --dataset ${dataset} --method wgc --alm vgc \
        --ratio 0.2 --seed $seed --normalized 1 --weighted 1

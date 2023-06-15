code=test_lap.py
dataset=tumblr_ct1

for r in 0.2 0.3 0.4 0.5 0.6 0.7 0.8
do
CUDA_VISIBLE_DEVICES= python $code --dataset $dataset --method vegc --ratio $r --save 1
CUDA_VISIBLE_DEVICES= python $code --dataset $dataset --method wgc --ratio 0.1 --alm 1 --weighted 0 --save 1
done

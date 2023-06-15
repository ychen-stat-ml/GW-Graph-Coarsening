# code=main_CSL_graph_classification.py 
# dataset=CSL
# seed=40
# dir_GIN='out/CSL/GIN/'

# CUDA_VISIBLE_DEVICES= python $code --dataset $dataset --gpu_id 0 --seed $((seed+2)) --out_dir $dir_GIN --config 'configs/CSL_graph_classification_GIN_CSL_100k.json'

# python test_csl.py --dataset CSL --gpu_id 0 --seed 42 --out_dir 'out/CSL/GIN/' --config 'configs/CSL_graph_classification_GIN_CSL_100k.json'

# 最后用的
# code=test_superpixel.py
# dataset=MNIST
# seed=40

# CUDA_VISIBLE_DEVICES= python $code --dataset $dataset --seed $seed --config 'configs/superpixels_graph_classification_GCN_MNIST_100k.json'

# 0123

python $code --dataset $dataset --method vgc --alm vgc
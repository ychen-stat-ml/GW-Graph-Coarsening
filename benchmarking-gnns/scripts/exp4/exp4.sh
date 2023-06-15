code=test_gcn.py
seed=27

python test_a.py --dataset ZINC --method vegc --ratio 0.3 --alm vegc --seed ${seed}
python test_a.py --dataset ZINC --method wgc --ratio 0.3 --alm wgc --seed ${seed}
python test_a.py --dataset ZINC --method wgc --ratio 0.3 --alm vegc --seed ${seed}

python $code --dataset ZINC --seed ${seed} --method vegc --config 'configs/GCGCN_ZINC.json' >> 'scripts/exp4/ZINC_res.txt'
python $code --dataset ZINC --seed ${seed} --method wgc --config 'configs/GCGCN_ZINC.json' >> 'scripts/exp4/ZINC_res.txt'
python test_gcn.py --dataset ZINC --seed ${seed} --method wgc --alm vegc --config 'configs/GCGCN_ZINC.json' >> 'scripts/exp4/ZINC_res.txt'

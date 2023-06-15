# GW-Graph-Coarsening

This is the official implementation for "A Gromov--Wasserstein Geometric View of Spectrum-Preserving Graph Coarsening" (ICML 2023).

## Main idea

Graph coarsening is a technique for solving large-scale graph problems by working on a smaller version of the original graph. 

This work studies graph coarsening from a different perspective, developing a theory for preserving graph distances and proposing a method to achieve this. 

The geometric approach is useful when working with a collection of graphs, such as in graph classification and regression. 

In this study, we consider a graph as an element on a metric space equipped with the Gromov--Wasserstein (GW) distance. We utilize the popular weighted kernel $K$-means method, which improves existing spectrum-preserving methods. 

Concrete details can be found in [our paper](Arxiv?).


## Installation

To prepare the conda environment for the code in this repo, the users can create the environment through
```sh
conda env create -f graph.yml
```

## Run Code

The initialization directory is the root directory `./`.

```
sh scripts/exp1.sh
sh scripts/exp2.sh
sh scripts/exp3.sh
```

The code for gcn tasks is adapted from [this repo](https://github.com/graphdeeplearning/benchmarking-gnns). We first enter the sub-directory and then run the following commands.

```
cd "benchmarking-gnns"
sh data/script_download_molecules.sh
sh scripts/exp4.sh
```

## Citation

If you find the repository helpful, please consider citing our papers:

```
@InProceedings{chen-etal-2023-gromov,
  title = 	 {A Gromov--Wasserstein Geometric View of Spectrum-Preserving Graph Coarsening},
  author =       {Chen, Yifan and Yao, Rentian and Yang, Yun and Chen, Jie},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  year = 	 {2023},
  publisher =    {PMLR},
}
```

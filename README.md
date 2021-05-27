# HoroPCA

This code is the official PyTorch implementation of the ICML 2021 paper: "HoroPCA: Hyperbolic Dimensionality Reduction via Horospherical Projections".
The code has an implementation of the HoroPCA method, as well as other methods for dimensionality reduction on manifolds, such as Principal Geodesic Analysis and tangent Principal Component Analysis.  

### Installation 

The code was tested on Python3.7. Start by installing the requirements: 
```bash
pip install -r requirements.txt
```

### Usage 

To run dimensionality reduction experiments use: 

```
python main.py --help

optional arguments:
  -h, --help            show this help message and exit
  --dataset {smalltree,phylo-tree,bio-diseasome,ca-CSphd}
                        Which datasets to use
  --model {pca,tpca,pga,bsa,hmds,horopca}
                        Which dimensionality reduction method to use
  --metrics METRICS [METRICS ...]
                        Which metrics to use
  --dim DIM             Embedding dimension to use
  --n-components N_COMPONENTS
                        Number of principal components
  --scale SCALE         Scale to use for embeddings computed with Sarkar's
                        construction
  --lr LR               Learning rate to use in optimization-based methods
  --n-runs N_RUNS       Number of runs for optimization-based methods
  --use-sarkar          Use sarkar to embed the graphs.
```
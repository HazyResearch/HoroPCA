"""Run dimensionality reduction experiment."""

import argparse
import logging

import networkx as nx
import numpy as np
import torch

import geom.hyperboloid as hyperboloid
import geom.poincare as poincare
from learning.frechet import Frechet
from learning.pca import TangentPCA, EucPCA, PGA, HoroPCA, BSA
from utils.data import load_graph, load_embeddings
from utils.metrics import avg_distortion_measures, compute_metrics, format_metrics, aggregate_metrics
from utils.sarkar import sarkar, pick_root

parser = argparse.ArgumentParser(
    description="Hyperbolic dimensionality reduction"
)
parser.add_argument('--dataset', type=str, help='which datasets to use', default="smalltree",
                    choices=["smalltree", "phylo-tree", "bio-diseasome", "ca-CSphd"])
parser.add_argument('--model', type=str, help='which dimensionality reduction method to use', default="horopca",
                    choices=["pca", "tpca", "pga", "bsa", "hmds", "horopca"])
parser.add_argument('--metrics', nargs='+', help='which metrics to use', default=["distortion", "frechet_var"])
parser.add_argument(
    "--dim", default=10, type=int, help="input embedding dimension to use"
)
parser.add_argument(
    "--n-components", default=2, type=int, help="number of principal components"
)

parser.add_argument(
    "--lr", default=5e-2, type=float, help="learning rate to use for optimization-based methods"
)
parser.add_argument(
    "--n-runs", default=5, type=int, help="number of runs for optimization-based methods"
)
parser.add_argument('--use-sarkar', default=False, action='store_true', help="use sarkar to embed the graphs")
parser.add_argument(
    "--sarkar-scale", default=3.5, type=float, help="scale to use for embeddings computed with Sarkar's construction"
)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)

    pca_models = {
        'pca': {'class': EucPCA, 'optim': False, 'iterative': False, "n_runs": 1},
        'tpca': {'class': TangentPCA, 'optim': False, 'iterative': False, "n_runs": 1},
        'pga': {'class': PGA, 'optim': True, 'iterative': True, "n_runs": args.n_runs},
        'bsa': {'class': BSA, 'optim': True, 'iterative': False, "n_runs": args.n_runs},
        'horopca': {'class': HoroPCA, 'optim': True, 'iterative': False, "n_runs": args.n_runs},
    }
    metrics = {}
    embeddings = {}
    logging.info(f"Running experiments for {args.dataset} dataset.")

    # load a graph args.dataset
    graph = load_graph(args.dataset)
    n_nodes = graph.number_of_nodes()
    nodelist = np.arange(n_nodes)
    graph_dist = torch.from_numpy(nx.floyd_warshall_numpy(graph, nodelist=nodelist))
    logging.info(f"Loaded {args.dataset} dataset with {n_nodes} nodes")

    # get hyperbolic embeddings
    if args.use_sarkar:
        # embed with Sarkar
        logging.info("Using sarkar embeddings")
        root = pick_root(graph)
        z = sarkar(graph, tau=args.sarkar_scale, root=root, dim=args.dim)
        z = torch.from_numpy(z)
        z_dist = poincare.pairwise_distance(z) / args.sarkar_scale
    else:
        # load pre-trained embeddings
        logging.info("Using optimization-based embeddings")
        assert args.dim in [2, 10, 50], "pretrained embeddings are only for 2, 10 and 50 dimensions"
        z = load_embeddings(args.dataset, dim=args.dim)
        z = torch.from_numpy(z)
        z_dist = poincare.pairwise_distance(z)
    if torch.cuda.is_available():
        z = z.cuda()
        z_dist = z_dist.cuda()
        graph_dist = graph_dist.cuda()

    # compute embeddings' distortion
    distortion = avg_distortion_measures(graph_dist, z_dist)[0]
    logging.info("Embedding distortion in {} dimensions: {:.4f}".format(args.dim, distortion))

    # Compute the mean and center the data
    logging.info("Computing the Frechet mean to center the embeddings")
    frechet = Frechet(lr=1e-2, eps=1e-5, max_steps=5000)
    mu_ref, has_converged = frechet.mean(z, return_converged=True)
    logging.info(f"Mean computation has converged: {has_converged}")
    x = poincare.reflect_at_zero(z, mu_ref)

    # Run dimensionality reduction methods
    logging.info(f"Running {args.model} for dimensionality reduction")
    metrics = []
    dist_orig = poincare.pairwise_distance(x)
    if args.model in pca_models.keys():
        model_params = pca_models[args.model]
        for _ in range(model_params["n_runs"]):
            model = model_params['class'](dim=args.dim, n_components=args.n_components, lr=args.lr, max_steps=500)
            if torch.cuda.is_available():
                model.cuda()
            model.fit(x, iterative=model_params['iterative'], optim=model_params['optim'])
            metrics.append(model.compute_metrics(x))
            embeddings = model.map_to_ball(x).detach().cpu().numpy()
        metrics = aggregate_metrics(metrics)
    else:
        # run hMDS baseline
        logging.info(f"Running hMDS")
        x_hyperboloid = hyperboloid.from_poincare(x)
        distances = hyperboloid.distance(x.unsqueeze(-2), x.unsqueeze(-3))
        D_p = poincare.pairwise_distance(x)
        x_h = hyperboloid.mds(D_p, d=args.n_components)
        x_proj = hyperboloid.to_poincare(x_h)
        embeddings["hMDS"] = x_proj.numpy()
        metrics = compute_metrics(x, x_proj)
    logging.info(f"Experiments for {args.dataset} dataset completed.")
    logging.info("Computing evaluation metrics")
    results = format_metrics(metrics, args.metrics)
    for line in results:
        logging.info(line)

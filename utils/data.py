"""Data utils."""

import networkx as nx
import numpy as np

import geom.poincare as poincare
from learning.frechet import Frechet


def load_graph(dataset):
    """Loads a graph dataset.

    Return: networkx graph object
    """
    G = nx.Graph()
    with open(f"data/edges/{dataset}.edges", "r") as f:
        for line in f:
            tokens = line.split()
            u = int(tokens[0])
            v = int(tokens[1])
            G.add_edge(u, v)
    return G


def load_embeddings(dataset, dim):
    embeddings_path = f"data/embeddings/{dataset}_{dim}_poincare.npy"
    return np.load(embeddings_path)


def center(x, lr):
    """Centers data so it has zero Frechet mean."""
    frechet = Frechet(lr=lr)
    mu = frechet.mean(x)
    return poincare.reflect_at_zero(x, mu)

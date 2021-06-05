"""Sarkar's combinatorial construction."""

import networkx as nx
import numpy as np
import scipy

MIN_NORM = 1e-15


# ################# CIRCLE INVERSIONS ########################

def reflect_at_zero(x, mu):  # Note: this differs from geom.poincare.reflect_at_zero because it's numpy instead of torch
    """
    Image of x by circle inversion that takes mu to the origin
    """
    mu_sqnorm = np.sum(mu ** 2)
    a = mu / mu_sqnorm.clip(min=1e-15)
    a_sqnorm = np.sum(a ** 2)
    r2 = a_sqnorm - np.longdouble([1.])
    xa_sqnorm = np.maximum(np.sum((x - a) ** 2, axis=-1, keepdims=True), MIN_NORM)
    return (r2 / xa_sqnorm) * (x - a) + a


def reflect_through_zero(p, q, x):
    """ Image of x under reflection that takes p (normalized) to q (normalized) and 0 to 0. """

    p_ = p / np.linalg.norm(p, axis=-1, keepdims=True).clip(min=1e-15)
    q_ = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-15)
    # print("norm p, q", np.linalg.norm(p_), np.linalg.norm(q_))
    r = q_ - p_
    # Magnitude of x in direction of r
    m = np.sum(r * x, axis=-1, keepdims=True) / np.sum(r * r, axis=-1, keepdims=True)
    return x - 2 * r * m


def test_reflect():
    pass


# ################# SARKAR CONSTRUCTION ########################

def pick_root(tree):
    graph_distances = np.array(nx.floyd_warshall_numpy(tree).astype(np.float32))
    j_ids = np.argmax(graph_distances, axis=1)
    i = np.argmax(graph_distances[np.arange(tree.number_of_nodes()), j_ids])
    j = j_ids[i]
    path = nx.shortest_path(tree, i, j)
    length = len(path)
    root = path[length // 2]
    return root


def place_children(z_parent, n_children, scaling, dim=2, coding=True):
    """Embeds children of node embedded at the origin.
    Assumes z is embedding of parent of node at the origin.
    children are at disrance scale/2 from their parent in hyperbolic metric.
    """
    if dim == 2:
        if z_parent is None:
            theta_parent = 0
            n_neighbors = n_children
        else:
            theta_parent = np.angle(z_parent[0] + z_parent[1] * 1j)
            n_neighbors = n_children + 1
        theta_children = [theta_parent + 2 * np.longdouble(np.pi) * (i + 1) / np.longdouble(n_neighbors) for i in
                          range(n_children)]
        z_children = []
        for theta_child in theta_children:
            z_children.append(scaling * np.array([np.cos(theta_child), np.sin(theta_child)]))
        return z_children
    else:
        normalize = lambda x: x / np.linalg.norm(x, keepdims=True)
        if coding:
            N = 2 ** int(np.ceil(np.log(dim) / np.log(2)))
            H = scipy.linalg.hadamard(N)
            if z_parent is not None:
                par_ = np.concatenate((z_parent, np.zeros(N - dim)))
                H = reflect_through_zero(H[0, :], par_, H)
                # print("reflecting H0 onto parent", scaling * normalize(H[0,:]) - par_)
            z_children = [H[i, :dim] for i in range(1, min(n_children + 1, N))]
            if n_children > N - 1:
                z_children += [np.random.randn(dim) for _ in range(n_children - N + 1)]
            z_children = [scaling * normalize(c) for c in z_children]
        else:
            z_children = [scaling * normalize(np.random.randn(dim)) for _ in range(n_children)]
        return z_children


def sarkar(tree, tau=1.0, root=None, dim=2, coding=False, seed=1234):
    """Embeds a tree in H_d using Sarkar's construction.

    Args:
        tree: nx.Graph object representing the tree structure to embed.
        root: index of the root node in the tree object.
        tau: scale of hyperbolic embeddings, parent-child will be placed at
             hyperbolic distance tau from each other.
    """
    np.random.seed(seed)
    if root is None:
        # pick root in Sarkar as node on longest path
        root = pick_root(tree)

    # Initialize embeddings array
    z = np.zeros((tree.number_of_nodes(), dim), dtype=np.float64)
    scaling = np.tanh(tau / 2)  # Euclidean distance corresponding to hyperbolic distance of tau
    # bfs traversal
    bfs_tree_rev = nx.reverse_view(nx.bfs_tree(tree, root))
    for current, children in nx.bfs_successors(tree, root):
        if current == root:
            z[root] = np.zeros(dim)
            z_children = place_children(None, len(children), scaling, dim=dim, coding=coding)
            for i, child_idx in enumerate(children):
                z[child_idx] = z_children[i]
        else:
            z_current = z[current]
            z_parent = z[list(bfs_tree_rev.neighbors(current))[0]]

            # inversion that maps current to the origin
            z_parent = reflect_at_zero(z_parent, z_current)
            z_children = place_children(z_parent, len(children), scaling, dim=dim, coding=coding)
            for i, child_idx in enumerate(children):
                z[child_idx] = reflect_at_zero(z_children[i], z_current)
    return z

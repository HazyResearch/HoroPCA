""" Geometric utility functions, mostly for standard Euclidean operations."""

import torch

MIN_NORM = 1e-15


def orthonormal(Q):
    """Return orthonormal basis spanned by the vectors in Q.

    Q: (..., k, d) k vectors of dimension d to orthonormalize
    """
    k = Q.size(-2)
    _, _, v = torch.svd(Q, some=False)  # Q = USV^T
    Q_ = v[:, :k]
    return Q_.transpose(-1, -2)  # (k, d) rows are orthonormal basis for rows of Q


def euc_reflection(x, a):
    """
    Euclidean reflection (also hyperbolic) of x
    Along the geodesic that goes through a and the origin
    (straight line)

    NOTE: this should be generalized by reflect()
    """
    xTa = torch.sum(x * a, dim=-1, keepdim=True)
    norm_a_sq = torch.sum(a ** 2, dim=-1, keepdim=True)
    proj = xTa * a / norm_a_sq.clamp_min(MIN_NORM)
    return 2 * proj - x


def reflect(x, Q):
    """Reflect points (euclidean) with respect to the space spanned by the rows of Q.

    Q: (k, d) set of k d-dimensional vectors (must be orthogonal)
    """
    ref = 2 * Q.transpose(0, 1) @ Q - torch.eye(x.shape[-1], device=x.device)
    return x @ ref

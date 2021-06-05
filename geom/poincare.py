"""Poincare utils functions."""

import torch

import geom.euclidean as euclidean

MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}


def expmap0(u):
    """Exponential map taken at the origin of the Poincare ball with curvature c.
    Args:
        u: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures
    Returns:
        torch.Tensor with tangent points shape (B, d)
    """
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = torch.tanh(u_norm) * u / u_norm
    return project(gamma_1)


def logmap0(y):
    """Logarithmic map taken at the origin of the Poincare ball with curvature c.
    Args:
        y: torch.Tensor of size B x d with tangent points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures
    Returns:
        torch.Tensor with hyperbolic points.
    """
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm / 1. * torch.atanh(y_norm.clamp(-1 + 1e-15, 1 - 1e-15))


def expmap(x, u):
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    second_term = torch.tanh(lambda_(x) * u_norm / 2) * u / u_norm
    gamma_1 = mobius_add(x, second_term)
    return gamma_1


def logmap(x, y):
    sub = mobius_add(-x, y)
    sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM).clamp_max(1 - 1e-15)
    return 2 / lambda_(x) * torch.atanh(sub_norm) * sub / sub_norm


def lambda_(x):
    """Computes the conformal factor."""
    x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
    return 2 / (1. - x_sqnorm).clamp_min(MIN_NORM)


def project(x):
    """Project points to Poincare ball with curvature c.
    Args:
        x: torch.Tensor of size B x d with hyperbolic points
    Returns:
        torch.Tensor with projected hyperbolic points.
    """
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def distance(x, y, keepdim=True):
    """Hyperbolic distance on the Poincare ball with curvature c.
    Args:
        x: torch.Tensor of size B x d with hyperbolic points
        y: torch.Tensor of size B x d with hyperbolic points
    Returns: torch,Tensor with hyperbolic distances, size B x 1
    """
    pairwise_norm = mobius_add(-x, y).norm(dim=-1, p=2, keepdim=True)
    dist = 2.0 * torch.atanh(pairwise_norm.clamp(-1 + MIN_NORM, 1 - MIN_NORM))
    if not keepdim:
        dist = dist.squeeze(-1)
    return dist


def pairwise_distance(x, keepdim=False):
    """All pairs of hyperbolic distances (NxN matrix)."""
    return distance(x.unsqueeze(-2), x.unsqueeze(-3), keepdim=keepdim)


def distance0(x, keepdim=True):
    """Computes hyperbolic distance between x and the origin."""
    x_norm = x.norm(dim=-1, p=2, keepdim=True)
    d = 2 * torch.atanh(x_norm.clamp(-1 + 1e-15, 1 - 1e-15))
    if not keepdim:
        d = d.squeeze(-1)
    return d


def mobius_add(x, y):
    """Mobius addition."""
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    denom = 1 + 2 * xy + x2 * y2
    return num / denom.clamp_min(MIN_NORM)


def mobius_mul(x, t):
    """Mobius multiplication."""
    normx = x.norm(dim=-1, p=2, keepdim=True).clamp(min=MIN_NORM, max=1. - 1e-5)
    return torch.tanh(t * torch.atanh(normx)) * x / normx


def midpoint(x, y):
    """Computes hyperbolic midpoint beween x and y."""
    t1 = mobius_add(-x, y)
    t2 = mobius_mul(t1, 0.5)
    return mobius_add(x, t2)


# Reflection (circle inversion of x through orthogonal circle centered at a)
def isometric_transform(x, a):
    r2 = torch.sum(a ** 2, dim=-1, keepdim=True) - 1.
    u = x - a
    return r2 / torch.sum(u ** 2, dim=-1, keepdim=True) * u + a


# center of inversion circle
def reflection_center(mu):
    return mu / torch.sum(mu ** 2, dim=-1, keepdim=True)


# Map x under the isometry (inversion) taking mu to origin
def reflect_at_zero(x, mu):
    a = reflection_center(mu)
    return isometric_transform(x, a)


def orthogonal_projection(x, Q, normalized=False):
    """ Orthogonally project x onto linear subspace (through the origin) spanned by rows of Q. """
    if not normalized:
        Q = euclidean.orthonormal(Q)
    x_ = euclidean.reflect(x, Q)
    return midpoint(x, x_)


def geodesic_between_ideals(ideals):
    """Return the center and radius of the Euclidean circle representing
    the geodesic joining two ideal points p = ideals[0] and q = ideals[1]

    Args:
        ideals: torch.tensor of shape (...,2,dim)
    Return:
        center: torch.tensor of shape (..., dim)
        radius: torch.tensor of shape (...)

    Note: raise an error if p = -q, i.e. if the geodesic between them is an Euclidean line
    """
    p = ideals[..., 0, :]
    q = ideals[..., 1, :]
    norm_sum = (p + q).norm(dim=-1, p=2)  # shape (...)
    assert torch.all(norm_sum != 0)
    center = (p + q) / (1 + (p * q).sum(dim=-1, keepdim=True))
    radius = (p - q).norm(dim=-1, p=2) / norm_sum
    return center, radius


def random_points(size, std=1.0):
    tangents = torch.randn(*size) * std
    x = expmap0(tangents)
    return x


def random_ideals(size):
    Q = torch.randn(*size)
    Q = Q / torch.norm(Q, dim=-1, keepdim=True)
    return Q

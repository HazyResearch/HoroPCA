"""Util functions for hyperboloid models
Convention: The ambient Minkowski space has signature -1, 1, 1, ...
                i.e. the squared norm of (t,x,y,z) is -t^2 + x^2 + y^2 + z^2,
            And we are using the positive sheet, i.e. every point on the hyperboloid
            has positive first coordinate.
"""
import torch

import geom.minkowski as minkowski
import geom.poincare as poincare

MIN_NORM = 1e-15


def distance(x, y):
    """
    Args:
        x, y: torch.tensor of the same shape (..., Minkowski_dim)

    Returns:
        torch.tensor of shape (..., )
    """
    # return torch.acosh(- minkowski.bilinear_pairing(x, y))
    return torch.acosh(torch.clamp(- minkowski.bilinear_pairing(x, y), min=1.0))


def exp_unit_tangents(base_points, unit_tangents, distances):
    """Batched exponential map using the given base points, unit tangent directions, and distances

    Args:
        base_points, unit_tangents: torch.tensor of shape (..., Minkowski_dim)
            Each unit_tangents[j..., :] must have (Minkowski) squared norm 1 and is orthogonal to base_points[j..., :]
        distances:      torch.tensor of shape (...) 

    Returns:
        torch.tensor of shape (..., Minkowski_dim)
    """
    distances = distances.unsqueeze(-1)
    return base_points * torch.cosh(distances) + unit_tangents * torch.sinh(distances)


# def exp(base_points, tangents):
#     """Batched exponential map using the given base points and tangent vectors
#
#     Args:
#         base_point, tangents: torch.tensor of shape (..., Minkowski_dim)
#             Each tangents[j..., :] must have squared norm > 0 and is orthogonal to base_points[j..., :]
#
#     Returns:
#         torch.tensor of shape (..., Minkowski_dim)
#     """
#     distances = torch.sqrt(minkowski.squared_norm(tangents))  # shape (...)
#     unit_tangets = tangents / distances.view(-1, 1)  # shape (..., Minkowski_dim)
#     return exp_unit_tangents(base_point, unit_tangents, distances)


def from_poincare(x, ideal=False):
    """Convert from Poincare ball model to hyperboloid model
    Args:
        x: torch.tensor of shape (..., dim)
        ideal: boolean. Should be True if the input vectors are ideal points, False otherwise

    Returns:
        torch.tensor of shape (..., dim+1)

    To do:
        Add some capping to make things numerically stable. This is only needed in the case ideal == False
    """
    if ideal:
        t = torch.ones(x.shape[:-1], device=x.device).unsqueeze(-1)
        return torch.cat((t, x), dim=-1)
    else:
        eucl_squared_norm = (x * x).sum(dim=-1, keepdim=True)
        return torch.cat((1 + eucl_squared_norm, 2 * x), dim=-1) / (1 - eucl_squared_norm).clamp_min(MIN_NORM)


def to_poincare(x, ideal=False):
    """Convert from hyperboloid model to Poincare ball model
    Args:
        x: torch.tensor of shape (..., Minkowski_dim), where Minkowski_dim >= 3
        ideal: boolean. Should be True if the input vectors are ideal points, False otherwise

    Returns:
        torch.tensor of shape (..., Minkowski_dim - 1)
    """
    if ideal:
        return x[..., 1:] / (x[..., 0].unsqueeze(-1)).clamp_min(MIN_NORM)
    else:
        return x[..., 1:] / (1 + x[..., 0].unsqueeze(-1)).clamp_min(MIN_NORM)


def decision_boundary_to_poincare(minkowski_normal_vec):
    """Convert the totally geodesic submanifold defined by the Minkowski normal vector to Poincare ball model
    (Here the Minkowski normal vector defines a linear subspace, which intersects the hyperboloid at our submanifold)

    Args:
        minkowski_normal_vec: torch.tensor of shape (Minkowski_dim, )

    Returns:
        center: torch.tensor of shape (Minkowski_dim -1, )
        radius: float

    Warning:
        minkowski_normal_vec must have positive squared norm
        minkowski_normal_vec[0] must be nonzero (otherwise the submanifold is a flat plane through the origin)
    """
    x = minkowski_normal_vec
    # poincare_origin = [1,0,0,0,...], # shape (Minkowski_dim, )
    poincare_origin = torch.zeros(minkowski_normal_vec.shape[0], device=minkowski_normal_vec.device)
    poincare_origin[0] = 1

    # shape (1, Minkowski_dim)
    poincare_origin_reflected = minkowski.reflection(minkowski_normal_vec, poincare_origin.unsqueeze(0))

    # shape (Minkowski_dim-1, )
    origin_reflected = to_poincare(poincare_origin_reflected).squeeze(0)
    center = poincare.reflection_center(origin_reflected)

    radius = torch.sqrt(torch.sum(center ** 2) - 1)

    return center, radius


def orthogonal_projection(basis, x):
    """Compute the orthogonal projection of x onto the geodesic submanifold
    spanned by the given basis vectors (i.e. the intersection of the hyperboloid with
    the Euclidean linear subspace spanned by the basis vectors).

    Args:
        basis: torch.tensor of shape(num_basis, Minkowski_dim)
        x:     torch.tensor of shape(batch_size, Minkowski_dim)

    Returns:
        torch.tensor of shape(batch_size, Minkowski_dim)

    Conditions:
        Each basis vector must have non-positive Minkowski squared norms.
        There must be at least 2 basis vectors.
        The basis vectors must be linearly independent.
    """
    minkowski_proj = minkowski.orthogonal_projection(basis, x)  # shape (batch_size, Minkowski_dim)
    squared_norms = minkowski.squared_norm(minkowski_proj)  # shape (batch_size, )
    return minkowski_proj / torch.sqrt(- squared_norms.unsqueeze(1))


def horo_projection(ideals, x):
    """Compute the projection based on horosphere intersections.
    The target submanifold has dimension num_ideals and is a geodesic submanifold passing through
    the ideal points and (1,0,0,0,...), i.e. the point corresponds to the origin in Poincare model.

    Args:
        ideals: torch.tensor of shape (num_ideals, Minkowski_dim)
            num_ideals must be STRICTLY between 1 and Minkowski_dim
            ideal vectors must be independent
            the geodesic submanifold spanned by ideals must not contain (1,0,0,...)

        x: torch.tensor of shape (batch_size, Minkowski_dim)


    Returns:
        torch.tensor of shape (batch_size, Minkowski_dim)
    """

    # Compute orthogonal (geodesic) projection from x to the geodesic submanifold spanned by ideals
    # We call this submanifold the "spine" because of the "open book" intuition
    spine_ortho_proj = orthogonal_projection(ideals, x)  # shape (batch_size, Minkowski_dim)
    spine_dist = distance(spine_ortho_proj, x)  # shape (batch_size, )

    # poincare_origin = [1,0,0,0,...], # shape (Minkowski_dim, )
    poincare_origin = torch.zeros(x.shape[1], device=x.device)
    poincare_origin[0] = 1

    # Find a tangent vector of the hyperboloid at spine_ortho_proj that is tangent to the target submanifold
    # and orthogonal to the spine.
    # This is done in a Gram-Schmidt way: Take the Euclidean vector pointing from spine_ortho_proj to poincare_origin,
    # then subtract a projection part so that it is orthogonal to the spine and tangent to the hyperboloid
    # Everything below has shape (batch_size, Minkowski_dim)
    chords = poincare_origin - spine_ortho_proj
    tangents = chords - minkowski.orthogonal_projection(ideals, chords)
    unit_tangents = tangents / torch.sqrt(minkowski.squared_norm(tangents)).view(-1, 1)

    proj_1 = exp_unit_tangents(spine_ortho_proj, unit_tangents, spine_dist)
    proj_2 = exp_unit_tangents(spine_ortho_proj, unit_tangents, -spine_dist)

    return proj_1, proj_2


def mds(D, d):
    """
    Args:
    D - (..., n, n) distance matrix

    Returns:
    X - (..., n, d) hyperbolic embeddings
    """
    Y = -torch.cosh(D)
    # print("Y:", Y)
    eigenvals, eigenvecs = torch.symeig(Y, eigenvectors=True)
    # print(Y.shape, eigenvals.shape, eigenvecs.shape)
    # print(eigenvals, eigenvecs)
    X = torch.sqrt(torch.clamp(eigenvals[-d:], min=0.)) * eigenvecs[..., -d:]
    # print("testing")
    # print(X)
    # print(Y @ X)
    u = torch.sqrt(1 + torch.sum(X * X, dim=-1, keepdim=True))
    M = torch.cat((u, X), dim=-1)
    # print(minkowski.pairwise_bilinear_pairing(M, M))
    return torch.cat((u, X), dim=-1)


def test():
    ideal = torch.tensor([[1.0, 0, 0, 0], [0.0, 1, 0, 0]])
    x = torch.tensor([[0.2, 0.3, 0.4, 0.5], [0.0, 0, 0, 0], [0.0, 0, 0, 0.7]])
    loid_ideal, loid_x = from_poincare(ideal, True), from_poincare(x)
    loid_p1, loid_p2 = horo_projection(loid_ideal, loid_x)
    pr1, pr2 = to_poincare(loid_p1), to_poincare(loid_p2)
    print(pr1)
    print(pr2)

    # ideals = torch.tensor([[3.0,3.0,0.0], [5.0,-5.0,0.0]])
    # x = torch.tensor([[5.0,0.0,math.sqrt(24)],[2.0,-math.sqrt(3), 0]])
    # print(orthogonal_projection(ideals, x))

    ideals = torch.tensor([[1.0, 1.0, 0.0], [5.0, 3, 4]])
    x = torch.tensor([[5.0, 0, 24 ** 0.5], [2.0, - 3 ** 0.5, 0]])
    print(horo_projection(ideals, x))


def test_mds(n=100, d=10):
    X = torch.randn(n, d)
    X = X / torch.norm(X, dim=-1, keepdim=True) * 0.9
    X = from_poincare(X)
    # print(X.shape)
    D = distance(X.unsqueeze(-2), X.unsqueeze(-3))
    # print(D.shape)
    # print(D-D.transpose(0,1))

    X_ = mds(D, d)
    # print(X_.shape)
    D_ = distance(X_.unsqueeze(-2), X_.unsqueeze(-3))
    print(D - D_)


def test_projection():
    """ Test that orthogonal projection agrees with the Poincare disk version. """
    d = 5
    # x = torch.randn(1, d) * 0.01
    x = poincare.random_points((1, d))
    # Q = torch.randn(2, d)
    # Q = Q / torch.norm(Q, dim=-1, keepdim=True)
    Q = poincare.random_ideals((2, d))

    # poincare projection
    import geom.poincare as P
    from geom.euclidean import orthonormal
    Q = orthonormal(Q)
    x_r = P.reflect(x, Q)
    p = P.midpoint(x, x_r)
    print(p)

    # hyperboloid projection
    Q = torch.cat([Q, torch.zeros(1, d)], dim=0)
    p_ = orthogonal_projection(from_poincare(Q, ideal=True), from_poincare(x))
    print(to_poincare(p_))


# Sanity checks
if __name__ == "__main__":
    # test()
    # test_mds(n=100, d=10)

    poincare_origin = torch.zeros(3)
    poincare_origin[0] = 1
    print(from_poincare(poincare_origin, ideal=True))
    print(to_poincare(from_poincare(poincare_origin, ideal=True), ideal=True))

    test_projection()

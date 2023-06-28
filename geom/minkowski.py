""" Util functions for the Minkowski metric.

Note that functions for the hyperboloid model itself are in geom.hyperboloid

Most functions in this file has a bilinear_form argument that can generally be ignored.
That argument is there just in case we need to use a non-standard norm/signature.
"""
import torch


def product(x, y):
    eucl_pairing = torch.sum(x * y, dim=-1, keepdim=False)
    return 2 * x[..., 0] * y[..., 0] - eucl_pairing


def bilinear_pairing(x, y, bilinear_form=None):
    """Compute the bilinear pairing (i.e. "dot product") of x and y using the given bilinear form.
    If bilinear_form is not provided, use the default Minkowski form,
        i.e. (x0, x1, x2) dot (y0, y1, y2) = -x0*y0 + x1*y1 + x2*y2

    Args:
        x, y: torch.tensor of the same shape (..., dim), where dim >= 2
        bilinear_form (optional): torch.tensor of shape (dim, dim)

    Returns:
        torch.tensor of shape (...)
    """
    if bilinear_form is None:
        eucl_pairing = torch.sum(x * y, dim=-1, keepdim=False)
        return eucl_pairing - 2 * x[..., 0] * y[..., 0]
    else:
        pairing = torch.matmul(x.unsqueeze(-2), (y @ bilinear_form).unsqueeze(-1))  # shape (..., 1, 1)
        return pairing.reshape(x.shape[:-1])


def squared_norm(x, bilinear_form=None):
    return bilinear_pairing(x, x, bilinear_form)


def pairwise_bilinear_pairing(x, y, bilinear_form=None):
    """Compute the pairwise bilinear pairings (i.e. "dot product") of two list of vectors
    with respect to the given bilinear form.
    If bilinear_form is not provided, use the default Minkowski form,
        i.e. (x0, x1, x2) dot (y0, y1, y2) = -x0*y0 + x1*y1 + x2*y2

    Args:
        x: torch.tensor of shape (..., M, dim), where dim >= 2
        y: torch.tensor of shape (..., N, dim), where dim >= 2
        bilinear_form (optional): torch.tensor of shape (dim, dim).
            
    Returns:
        torch.tensor of shape (..., M, N)
    """
    if bilinear_form is None:
        return x @ y.transpose(-1, -2) - 2 * torch.ger(x[:, 0], y[:, 0])
    else:
        return x @ bilinear_form @ y.transpose(-1, -2)


def orthogonal_projection(basis, x, bilinear_form=None):
    """Compute the orthogonal projection of x onto the vector subspace spanned by basis.
    Here orthogonality is defined using the given bilinear_form
    If bilinear_form is not provided, use the default Minkowski form,
        i.e. (x0, x1, x2) dot (y0, y1, y2) = -x0*y0 + x1*y1 + x2*y2

    Args:
        basis: torch.tensor of shape (subspace_dim, dim), where dim >= 2
        x:     torch.tensor of shape (batch_size, dim), where dim >= 2
        bilinear_form (optional): torch.tensor of shape (dim, dim).

    Returns:
        torch.tensor of shape (batch_size, dim)

    Warning: Will not work if the linear subspace spanned by basis is tangent to the light cone.
             (In that case, the orthogonal projection is not unique)
    """
    coefs = torch.linalg.solve(pairwise_bilinear_pairing(basis, basis, bilinear_form), 
                               pairwise_bilinear_pairing(basis, x, bilinear_form))

    return coefs.transpose(-1, -2) @ basis


def reflection(subspace, x, subspace_given_by_normal=True, bilinear_form=None):
    """Compute the reflection of x through a linear subspace (of dimension 1 less than the ambient space)
    Here reflection is defined using the notion of orthogonality coming from the given bilinear_form
    If bilinear_form is not provided, use the default Minkowski form,
        i.e. (x0, x1, x2) dot (y0, y1, y2) = -x0*y0 + x1*y1 + x2*y2
    
    Args:
        subspace: If subspace_given_by_normal:
                        torch.tensor of shape (dim, ), representing a normal vector to the subspace
                  Else:
                        torch.tensor of shape (dim-1, dim), representing a basis of the subspace
        x: torch.tensor of shape (batch_size, dim)
        bilinear_form (optional): torch.tensor of shape (dim, dim).

    Returns:
        torch.tensor of shape (batch_size, dim)

    Warning: Will not work if the linear subspace is tangent to the light cone.
             (In that case, the reflection is not unique)
    """
    if subspace_given_by_normal:
        return x - 2 * orthogonal_projection(subspace.unsqueeze(0), x, bilinear_form)
    else:
        return 2 * orthogonal_projection(subspace, x, bilinear_form) - x

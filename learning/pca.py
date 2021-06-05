"""Hyperbolic dimensionality reduction models."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import geom.euclidean as euclidean
import geom.hyperboloid as hyperboloid
import geom.minkowski as minkowski
import geom.poincare as poincare
from geom.horo import busemann, project_kd
from utils.metrics import compute_metrics


class PCA(ABC, nn.Module):
    """Dimensionality reduction model class."""

    def __init__(self, dim, n_components, lr=1e-3, max_steps=100, keep_orthogonal=False):
        super(PCA, self).__init__()
        self.dim = dim
        self.n_components = n_components
        self.components = nn.ParameterList(nn.Parameter(torch.randn(1, dim)) for _ in range(self.n_components))
        self.max_steps = max_steps
        self.lr = lr
        self.keep_orthogonal = keep_orthogonal

    def project(self, x):
        """Projects points onto the principal components."""
        Q = self.get_components()
        return self._project(x, Q)

    @abstractmethod
    def _project(self, x, Q):
        """Projects points onto the submanifold that goes through the origin and is spanned by different components.

        Args:
            x: torch.tensor of shape (batch_size, dim)
            Q: torch.tensor of shape (n_components, dim)

        Returns:
            x_p: torch.tensor of shape (batch_size, dim)
        """
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, x, Q):
        """Computes objective to minimize.

        Args:
            x: torch.tensor of shape (batch_size, dim), data before _projection
            Q: torch.tensor of shape (n_components, dim)

        Args:
            loss: torch.tensor of shape (1,)
        """
        raise NotImplementedError

    def gram_schmidt(self, ):
        """Applies Gram-Schmidt to the component vectors."""

        def inner(u, v):
            return torch.sum(u * v)

        Q = []
        for k in range(self.n_components):
            v_k = self.components[k][0]
            proj = 0.0
            for v_j in Q:
                v_j = v_j[0]
                coeff = inner(v_j, v_k) / inner(v_j, v_j).clamp_min(1e-15)
                proj += coeff * v_j
            v_k = v_k - proj
            v_k = v_k / torch.norm(v_k).clamp_min(1e-15)
            Q.append(torch.unsqueeze(v_k, 0))
        return torch.cat(Q, dim=0)

    def orthogonalize(self):
        Q = torch.cat([self.components[i] for i in range(self.n_components)])  # (k, d)
        # _, _, v = torch.svd(Q, some=False)  # Q = USV^T
        # Q_ = v[:, :self.n_components]
        # return Q_.transpose(-1, -2)# (k, d) rows are orthonormal basis for rows of Q
        return euclidean.orthonormal(Q)

    def normalize(self, ):
        """Makes the component vectors unit-norm (not orthogonal)."""
        Q = torch.cat([self.components[i] for i in range(self.n_components)])
        return Q / torch.norm(Q, dim=1, keepdim=True).clamp_min(1e-15)

    def get_components(self, ):
        if self.keep_orthogonal:
            Q = self.gram_schmidt()
            # Q = self.orthogonalize()
        else:
            Q = self.normalize()
        return Q  # shape (n_components, dim)

    def map_to_ball(self, x):
        """Returns coordinates of _projected points in a lower-dimensional Poincare ball model.
        Args:
            x: torch.tensor of shape (batch_size, dim)

        Returns:
            torch.tensor of shape (batch_size, n_components)
        """
        Q = self.get_components()
        x_p = self._project(x, Q)
        # Q_orthogonal = self.gram_schmidt()
        Q_orthogonal = self.orthogonalize()
        return x_p @ Q_orthogonal.transpose(0, 1)

    def fit_optim(self, x, iterative=False):
        """Finds component using gradient-descent-based optimization.

        Args:
            x: torch.tensor of size (batch_size x dim)
            iterative: boolean

        Note:
            If iterative = True returns optimizes components by components (nested subspace assumption).
        """
        loss_vals = []
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if not iterative:
            for i in range(self.max_steps):
                # Forward pass: compute _projected variance
                Q = self.get_components()
                loss = self.compute_loss(x, Q)
                loss_vals.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1e5)
                # if self.components[0].grad.sum().isnan().item():
                optimizer.step()
        else:
            for k in range(self.n_components):
                for i in range(self.max_steps):
                    # Forward pass: compute _projected variance
                    Q = self.get_components()
                    # Project on first k components
                    loss = self.compute_loss(x, Q[:k + 1, :])
                    loss_vals.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                self.components[k].data = self.get_components()[k].unsqueeze(0)
                self.components[k].requires_grad = False
        return loss_vals

    def fit_spectral(self, x):
        """Finds component using spectral decomposition (closed-form solution).

        Args:
            x: torch.tensor of size (batch_size x dim)
        """
        raise NotImplementedError

    def fit(self, x, iterative=False, optim=True):
        """Finds principal components using optimization or spectral decomposition approaches.

        Args:
            x: torch.tensor of size (batch_size x dim)
            iterative: boolean (true to do iterative optimization of nested subspaces)
            optim: boolean (true to find components via optimization, defaults to SVD otherwise)
        """
        if optim:
            self.fit_optim(x, iterative)
        else:
            self.fit_spectral(x)

    def compute_metrics(self, x):
        """Compute dimensionality reduction evaluation metrics."""
        Q = self.get_components()
        x_proj = self._project(x, Q).detach()
        return compute_metrics(x, x_proj)


class EucPCA(PCA):
    """Euclidean PCA (assumes data has Euclidean mean zero)."""

    def __init__(self, dim, n_components, lr=1e-3, max_steps=100):
        super(EucPCA, self).__init__(dim, n_components, lr, max_steps, keep_orthogonal=True)

    def compute_loss(self, x, Q):
        vals = x @ Q.transpose(0, 1)  # shape (batch_size, n_components)
        return - torch.sum(vals ** 2)

    def _project(self, x, Q):
        return (x @ Q.transpose(0, 1)) @ Q  # shape (batch_size, dim)

    def fit_spectral(self, x):
        """Euclidean PCA closed-form with SVD."""
        S = (x.T @ x)
        U, S, V = torch.svd(S)
        for k in range(self.n_components):
            self.components[k].data = U[k:k + 1]


class TangentPCA(PCA):
    """Euclidean PCA in the tangent space of the mean (assumes data has Frechet mean zero)."""

    def __init__(self, dim, n_components, lr=1e-3, max_steps=100):
        super(TangentPCA, self).__init__(dim, n_components, lr, max_steps, keep_orthogonal=True)

    def _project(self, x, Q):
        x_t = poincare.logmap0(x)  # shape (batch_size, dim)
        x_pt = (x_t @ Q.transpose(0, 1)) @ Q  # shape (batch_size, dim)
        x_p = poincare.expmap0(x_pt)  # shape (batch_size, dim)
        return x_p

    def compute_loss(self, x, Q):
        x_t = poincare.logmap0(x)  # shape (batch_size, dim)
        vals = x_t @ Q.transpose(0, 1)  # shape (batch_size, n_components)
        return - torch.sum(vals ** 2)

    def fit_spectral(self, x):
        """Geodesic PCA closed-form with SVD."""
        u = poincare.logmap0(x)
        S = (u.T @ u)
        U, S, V = torch.svd(S)
        for k in range(self.n_components):
            self.components[k].data = U[k:k + 1]


class PGA(PCA):
    """Exact Hyperbolic PGA using geodesic _projection (assuming data has Frechet mean zero).
    This assumption is necessary because otherwise its unclear how to geodesically _project on the submanifold spanned
    by tangent vectors. For general Frechet mean, the PGA paper approximates this using Tangent PCA.
    """

    def __init__(self, dim, n_components, lr=1e-3, max_steps=100):
        super(PGA, self).__init__(dim, n_components, lr, max_steps, keep_orthogonal=True)

    def _project(self, x, Q):
        """Geodesic projection."""
        proj = poincare.orthogonal_projection(x, Q, normalized=self.keep_orthogonal)
        return proj

    def compute_loss(self, x, Q):
        proj = self._project(x, Q)
        sq_distances = poincare.distance0(proj) ** 2
        var = torch.mean(sq_distances)
        return -var


class HoroPCA(PCA):
    """Hyperbolic PCA using horocycle _projections (assumes data has Frechet mean zero)."""

    def __init__(self, dim, n_components, lr=1e-3, max_steps=100, frechet_variance=False, auc=False, hyperboloid=True):
        """
        Currently auc=True and frechet_variance=True are not simultaneously supported (need to track mean parameter for each component).
        """

        super(HoroPCA, self).__init__(dim, n_components, lr, max_steps, keep_orthogonal=True)
        self.hyperboloid = hyperboloid
        self.frechet_variance = frechet_variance
        self.auc = auc
        if self.frechet_variance:
            self.mean_weights = nn.Parameter(torch.zeros(n_components))

    def _project(self, x, Q):
        if self.n_components == 1:
            proj = project_kd(Q, x)[0]
        else:
            if self.hyperboloid:
                hyperboloid_ideals = hyperboloid.from_poincare(Q, ideal=True)
                hyperboloid_x = hyperboloid.from_poincare(x)
                hyperboloid_proj = hyperboloid.horo_projection(hyperboloid_ideals, hyperboloid_x)[0]
                proj = hyperboloid.to_poincare(hyperboloid_proj)
            else:
                proj = project_kd(Q, x)[0]
        return proj

    def compute_variance(self, x):
        """ x are projected points. """
        if self.frechet_variance:
            # mean = self.mean_weights.unsqueeze(-1) * torch.stack(self.components, dim=0) # (k, d)
            Q = [self.mean_weights[i] * self.components[i] for i in range(self.n_components)]  # (k, d)
            mean = sum(Q).squeeze(0)
            distances = poincare.distance(mean, x)
            var = torch.mean(distances ** 2)
        else:
            distances = poincare.pairwise_distance(x)
            var = torch.mean(distances ** 2)
        return var

    def compute_loss(self, x, Q):
        if self.n_components == 1:
            # option 1
            bus = busemann(x, Q[0])  # shape (batch_size, n_components)
            return -torch.var(bus)
        else:
            auc = []
            if self.auc:
                for i in range(1, self.n_components):
                    Q_ = Q[:i, :]
                    proj = self._project(x, Q_)
                    var = self.compute_variance(proj)
                    auc.append(var)
                return -sum(auc)
            else:
                proj = self._project(x, Q)
                var = self.compute_variance(proj)
            return -var


class BSA(PCA):
    """ Stores k+1 reference points to define geodesic projections.

    If hyperboloid option is false, only stores k reference points and assumes the first reference point (mean) is the origin.
    """

    def __init__(self, dim, n_components, lr=1e-3, max_steps=100, hyperboloid=True, auc=True):
        """
        hyperboloid: Do computations in the hyperboloid model, allowing for subspaces that do not pass through the origin (stores k+1 reference points instead of k)
        auc: Use AUC objective to optimize the entire flag

        Note that if auc=False and iterative=True, this is equivalent to forward BSA. However, hyperboloid=True is not currently supported in this case
        """
        self.hyperboloid = hyperboloid
        self.auc = auc
        if self.hyperboloid:
            super(BSA, self).__init__(dim, n_components + 1, lr, max_steps, keep_orthogonal=True)
        else:
            super(BSA, self).__init__(dim, n_components, lr, max_steps, keep_orthogonal=True)

    def _project(self, x, Q):
        """Geodesic projection."""
        # return poincare.orthogonal_projection(x, Q, normalized=self.keep_orthogonal)
        proj = poincare.orthogonal_projection(x, Q, normalized=self.keep_orthogonal)
        return proj

    def compute_loss(self, x, Q):
        if self.auc:
            auc = []
            if self.hyperboloid:
                Q = hyperboloid.from_poincare(Q, ideal=True)
                x = hyperboloid.from_poincare(x)

                for i in range(1, self.n_components):
                    Q_ = Q[:i + 1, :]
                    proj = minkowski.orthogonal_projection(Q_, x)
                    residual_variance = torch.sum(hyperboloid.distance(x, proj) ** 2)
                    auc.append(residual_variance)
            else:
                for i in range(1, self.n_components):
                    Q_ = Q[:i, :]
                    proj = self._project(x, Q_)
                    residual_variance = torch.sum(poincare.distance(x, proj) ** 2)
                    auc.append(residual_variance)
            return sum(auc)
        else:
            if self.hyperboloid:
                Q = hyperboloid.from_poincare(Q, ideal=True)
                x = hyperboloid.from_poincare(x)
                proj = minkowski.orthogonal_projection(Q, x)
                residual_variance = torch.sum(hyperboloid.distance(x, proj) ** 2)
            else:
                proj = self._project(x, Q)
                residual_variance = torch.sum(poincare.distance(x, proj) ** 2)
            return residual_variance

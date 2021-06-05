"""Evaluation metrics."""
import numpy as np
import torch

import geom.poincare as poincare
from learning.frechet import Frechet


def avg_distortion_measures(distances1, distances2, tau=1.0):
    """Computes different measures of average distortion between two distance matrices.
    :param distances1: N x N torch tensor with pairwise distances. (ground truth)
    :param distances2: N x N torch tensor with pairwise distances. (scaled embeddings)
    :return: Average distortion (scalar).
    """
    n_nodes = distances1.shape[0]
    ids = torch.triu_indices(row=n_nodes, col=n_nodes, offset=1)
    distances1 = distances1[ids[0], ids[1]]
    distances2 = distances2[ids[0], ids[1]] / tau
    diff = torch.abs(distances2 - distances1)
    ratio = diff / distances1
    avg_distortion = torch.mean(ratio).item()
    avg_distortion_sq = torch.mean(ratio ** 2).item()
    avg_distortion_abs = torch.mean(diff ** 2).item()
    return avg_distortion, avg_distortion_sq, avg_distortion_abs


def worst_case_distortion(distances1, distances2):
    """Worst case distortion metric."""
    n_nodes = distances1.shape[0]
    ids = torch.triu_indices(row=n_nodes, col=n_nodes, offset=1)
    ratio = (distances2 / distances1)[ids[0], ids[1]]
    return (torch.max(ratio) / torch.min(ratio)).item()


def l2_error(dist_orig, dist_proj):
    """l2 error of distances."""
    return torch.mean((dist_orig - dist_proj) ** 2).item()


def unexplained_variance(x, x_proj):
    """Unexplained variance (see Pennec (2018))."""
    res = poincare.distance(x, x_proj) ** 2
    return torch.mean(res).item()


def frechet_var_approx(dist_proj):
    """Approximation of the Frechet variance with pairwise squared distances."""
    return torch.mean(dist_proj ** 2).item()


def compute_metrics(x, x_proj, frechet_lr=0.1):
    """Computes various evaluation metrics projections."""
    try:
        uv = unexplained_variance(x, x_proj)
    except RuntimeError:
        # exception for hMDS where unexplained variance cannot be computed
        uv = -1
    dist_orig = poincare.pairwise_distance(x)
    dist_proj = poincare.pairwise_distance(x_proj)
    avg_distortion, avg_distortion_sq, avg_distortion_abs = avg_distortion_measures(dist_orig, dist_proj)
    wc_distortion = worst_case_distortion(dist_orig, dist_proj)
    frechet_var_apx = frechet_var_approx(dist_proj)
    frechet_var_apx_orig = frechet_var_approx(dist_orig)
    l2 = l2_error(dist_orig, dist_proj)
    frechet = Frechet(lr=frechet_lr)
    frechet_var, has_converged = frechet.variance(x_proj, return_converged=True)
    return {
        'distortion': avg_distortion,
        'distortion_sq': avg_distortion_sq,
        'distortion_abs': avg_distortion_abs,
        'distortion_wc': wc_distortion,
        'unexplained_var': uv,
        'l2_error': l2,
        'frechet_var_apx': frechet_var_apx,
        'frechet_var': frechet_var.item(),
        'frechet_mean_has_converged': has_converged,
        "frechet_var_apx_orig": frechet_var_apx_orig,
    }


def format_metrics(metrics, metric_names):
    """Print metrics."""
    formatted_results = []
    for metric in metric_names:
        x = metrics[metric]
        if isinstance(x, list):
            mean, std = x
        else:
            mean, std = x, 0.0
        formatted_results.append("{}: \t{:.2f} +- {:.2f}".format(metric, mean, std))
    return formatted_results


def aggregate_metrics(metrics):
    """Compute average and standard deviation for metrics."""
    if len(metrics) == 1:
        return metrics[0]
    else:
        agg_metrics = metrics[0]
        for metric in agg_metrics.keys():
            vals = [x[metric] for x in metrics]
            agg_metrics[metric] = [np.mean(vals), np.std(vals)]
        return agg_metrics

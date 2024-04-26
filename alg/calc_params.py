from math import ceil, floor

import numpy as np

from configs.params import Params


def calculate_iters(p: Params):
    """Calculate the number of iterations based on the epsilon budget."""
    if p.dp == "none":
        return 7  # Maximum number of iterations
    if p.dp_release == "sumcount":  # sum-count method
        """From: https://github.com/IBM/differential-privacy-library/blob/main/diffprivlib/models/k_means.py"""
        epsilon_m = np.sqrt(500 * (p.k ** 3) / (p.data_size ** 2) * (p.dim + np.cbrt(4 * p.dim * (p.rho ** 2))) ** 3)
    else:  # centroid method
        epsilon_m = np.sqrt(500 * p.k * (p.dim ** 3) / ((p.min_cluster * p.num_clients + 1e-6) ** 2))

    iters = max(min(p.eps / epsilon_m, 7), 2)

    return int(iters)


def calculate_max_dist(p):
    """Calculate the maximum distance factor based on the dimension and boundary."""
    if p.dp != "none" and p.max_dist_mode == "hard":
        if p.dim == 2:
            # We compare with dist^2, so we need to square the max_dist
            return (1 / 2) * abs(p.bounds[1]) ** 2
        else:
            print("Hard max_dist only tested for 2D data")
    return p.dim * (p.bounds[1] - p.bounds[0]) ** 2  # This is the default value: maximum possible distance


def calculate_cluster_sizes(data_size, k, params_list):
    """Calculate the cluster sizes based on the data size and number of clients."""
    params_list["cluster_mins"] = [int(ceil(data_size / (alpha * k * params_list["num_clients"])))
                                   for alpha in params_list["cluster_min_alphas"]]
    params_list["cluster_maxs"] = [int(floor(data_size * alpha / (k * params_list["num_clients"])))
                                   for alpha in params_list["cluster_max_alphas"]]
    mins = list(zip(params_list["cluster_mins"], params_list["cluster_min_alphas"]))
    maxs = list(zip(params_list["cluster_maxs"], params_list["cluster_max_alphas"]))
    return mins, maxs

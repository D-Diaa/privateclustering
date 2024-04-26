from copy import copy

import numpy as np
from diffprivlib.mechanisms import Laplace

from configs.params import Params
from data_io.fixed import to_fixed, to_int


class DPApplier:

    def __init__(self, params: Params):
        self.count_mech = None
        self.count_eps = None
        self.count_sensitivity = None

        self.cent_mech = None
        self.cent_eps = None
        self.cent_sensitivity = None

        self.sum_mech = None
        self.sum_eps = None
        self.sum_sensitivity = None

        self.params = copy(params)
        self.dp_setup(self.params)

    def update_maxdist(self, iter):
        params = copy(self.params)
        params.max_dist = self.params.max_dist / (iter + 1)
        self.dp_setup(params)

    def dp_setup(self, params=None):
        if params is None:
            params = self.params
        if params.dp == "laplace":
            self.sum_sensitivity = max(abs(params.bounds[1]), abs(params.bounds[0]))
            if params.assignment == "constrained" and params.max_dist_mode == "hard":
                # If max_dist is enforced, then sensitivity could be bounded further
                self.sum_sensitivity = min(self.sum_sensitivity, 2 * params.max_dist)
        elif params.dp == "none":
            self.sum_sensitivity = 1
        else:
            raise NotImplementedError
        self.cent_sensitivity = self.sum_sensitivity / max(self.params.min_cluster, 1)
        self.count_sensitivity = 1
        self.sum_eps, self.count_eps = self.split_epsilon_sum_count()
        self.cent_eps = self.params.eps / self.params.iters
        if self.params.dp != "none":
            self.count_mech = Laplace(epsilon=self.count_eps, sensitivity=self.count_sensitivity,
                                      delta=0, random_state=params.seed)
            if self.params.dp == "laplace":
                self.sum_mech = Laplace(epsilon=self.sum_eps, sensitivity=self.sum_sensitivity,
                                        delta=0, random_state=params.seed)
                self.cent_mech = Laplace(epsilon=self.cent_eps, sensitivity=self.cent_sensitivity,
                                         delta=0, random_state=params.seed)
            else:
                raise NotImplementedError

    def split_epsilon_sum_count(self):
        epsilon_i = 1
        epsilon_0 = np.cbrt(4 * self.params.dim * self.params.rho ** 2)

        normaliser = self.params.eps / self.params.iters / (epsilon_i * self.params.dim + epsilon_0)

        return epsilon_i * normaliser, epsilon_0 * normaliser

    def randomize_masked_sumcount(self, sums, counts):
        if self.params.dp != "none":
            for x in np.nditer(sums, op_flags=['readwrite']):
                noise = self.sum_mech.randomise(0)
                x[...] += to_fixed(noise)
            for x in np.nditer(counts, op_flags=['readwrite']):
                x[...] += to_int(self.count_mech.randomise(0))
        return sums, counts

    def randomize_masked_centroid(self, cents):
        if self.params.dp != "none":
            for x in np.nditer(cents, op_flags=['readwrite']):
                noise = self.cent_mech.randomise(0)
                x[...] += to_fixed(noise)
        return cents

from copy import copy

from alg.mincostflow import mincostflow
from alg.postprocess import TruncationAndFolding
from configs.params import Params
from utils.utils import *

"""
This file contains the implementation of the clients in the Federated Clustering algorithm.
The clients are divided into two classes: UnmaskedClient and MaskedClient.
The UnmaskedClient class is used for the non-encrypted version of the Federated Clustering algorithm.
The MaskedClient class is used for the encrypted version of the Federated Clustering algorithm.

The UnmaskedClient class contains the following methods:
    - __init__: Initializes the UnmaskedClient object.
    - init_centroids_optimal: Initializes the centroids optimally (sphere-packing).
    - init_centroids_random: Initializes the centroids randomly.
    - postprocess_sumcount: Post-processes the centroids. (sum-count post-processing)
    - postprocess_centroid: Post-processes the centroids. (centroids post-processing)
    - update_centroids_sumcount: Updates the centroids in the SumCount setting.
    - update_centroids_centroid: Updates the centroids in the Centroid setting.
    - compute_assoc: Computes the associations between the data points and the centroids.
    - flow_step: Computes the flow step in the algorithm. (MinCostFlow)
    - basic_step: Computes the basic step in the algorithm. (Unconstrained)
    - step_sumcount: Performs a step in the SumCount setting. (Local Update: sum-count)
    - step_centroid: Performs a step in the Centroid setting. (Local Update: centroids)

The MaskedClient class inherits from the UnmaskedClient class and contains the following methods:
    - __init__: Initializes the MaskedClient object.
    - step_sumcount: Performs a step in the SumCount setting. (Local Update: sum-count)
    - step_centroid: Performs a step in the Centroid setting. (Local Update: centroids)
    - update_centroids_sumcount: Updates the centroids in the SumCount setting.
    - update_centroids_centroid: Updates the centroids in the Centroid setting.
"""


class UnmaskedClient:
    """
    UnmaskedClient class
    """

    def __init__(self, index: int, values, params: Params, centroids=None):
        """
        Initializes the UnmaskedClient object.
        :param index: client index
        :param values: data points
        :param params: parameters
        :param centroids: initial centroids (optional)
        """
        self.params = copy(params)
        self.associations = None
        self.distances = None
        self.params = params
        self.index = index
        self.common_generator = np.random.RandomState(params.seed)
        self.cent_post = TruncationAndFolding(params.bounds[0], params.bounds[1])
        self.sum_post = TruncationAndFolding(params.bounds[0] * params.max_cluster,
                                             params.bounds[1] * params.max_cluster)
        self.count_post = TruncationAndFolding(params.min_cluster, params.max_cluster)
        assert values.shape[1] == params.dim, "Error! values shape doesn't match"
        if params.fixed:
            values = unscale(values)
        self.values = values
        if centroids is not None:
            self.centroids = centroids
        else:
            if params.init == "random":
                self.centroids = self.init_centroids_random(params.k, params.dim, False, 1, self.common_generator)
            elif params.init == "optimal":
                self.centroids = self.init_centroids_optimal(self.common_generator)
            else:
                raise NotImplementedError

    def init_centroids_random(self, k, d, fixed=False, denom=1, generator=None):
        """
        Initializes the centroids randomly.
        :param k: Number of clusters
        :param d: Dimension of the data points
        :param fixed: Fixed precision flag
        :param denom: Denominator for fixed precision
        :param generator: Random number generator
        :return: Randomly initialized centroids
        """
        if generator is None:
            r = np.random.random((k, d))
        else:
            r = generator.random((k, d))
        c = r / denom
        return c if not fixed else to_fixed(c)

    def init_centroids_optimal(self, generator):
        """
        Initializes the centroids optimally (sphere-packing).
        From: https://github.com/IBM/differential-privacy-library/blob/main/diffprivlib/models/k_means.py#L167
        :param generator: Random number generator
        :return: Optimal centroids
        """
        dims = self.params.dim
        k = self.params.k
        bounds_processed = np.zeros(shape=(dims, 2))
        lower = self.params.bounds[0]
        upper = self.params.bounds[1]
        for dim in range(dims):
            bounds_processed[dim, :] = [upper - lower, lower]

        cluster_proximity = np.min(bounds_processed[:, 0]) / 2.0
        while cluster_proximity > 0:
            centers = np.zeros(shape=(k, dims))
            cluster, retry = 0, 0
            while retry < 100:
                if cluster >= k:
                    break
                temp_center = generator.random(dims) * (bounds_processed[:, 0] - 2 * cluster_proximity) + \
                              bounds_processed[:, 1] + cluster_proximity
                if cluster == 0:
                    centers[0, :] = temp_center
                    cluster += 1
                    continue
                min_distance = ((centers[:cluster, :] - temp_center) ** 2).sum(axis=1).min()
                if np.sqrt(min_distance) >= 2 * cluster_proximity:
                    centers[cluster, :] = temp_center
                    cluster += 1
                    retry = 0
                else:
                    retry += 1
            if cluster >= k:
                return centers
            cluster_proximity /= 2.0
        return None

    def postprocess_sumcount(self, totals, counts):
        """
        Post-processes the centroids.
        :param totals: Sum of the data points
        :param counts: Number of data points
        :return: Post-processed centroids
        """
        if self.params.sum_post == "fold":
            totals = self.sum_post.fold(totals)
        elif self.params.sum_post == "truncate":
            totals = self.sum_post.truncate(totals)

        if self.params.count_post == "fold":
            counts = self.count_post.fold(counts)
        elif self.params.count_post == "truncate":
            counts = self.count_post.truncate(counts)

        self.centroids = totals / (np.expand_dims(counts, axis=1) + 1e-6)

        if self.params.cent_post == "fold":
            self.centroids = self.cent_post.fold(self.centroids)
        elif self.params.cent_post == "truncate":
            self.centroids = self.cent_post.truncate(self.centroids)

    def postprocess_centroid(self, cents):
        """
        Post-processes the centroids.
        :param cents: Centroids
        :return: Post-processed centroids
        """
        self.centroids = cents
        if self.params.cent_post == "fold":
            self.centroids = self.cent_post.fold(self.centroids)
        elif self.params.cent_post == "truncate":
            self.centroids = self.cent_post.truncate(self.centroids)

    def update_centroids_sumcount(self, totals, counts):
        """
        Updates the centroids in the SumCount setting.
        :param totals: Sum of the data points
        :param counts: Number of data points
        :return: Updated centroids (post-processed if constrained)
        """
        if self.params.fixed:
            totals = unscale(totals)
        if self.params.assignment == "constrained":
            self.postprocess_sumcount(totals, counts)
        else:
            self.centroids = totals / (np.expand_dims(counts, axis=1) + 1e-6)

    def update_centroids_centroid(self, cents):
        """
        Updates the centroids in the Centroid setting.
        :param cents: Centroids
        :return: Updated centroids (post-processed if constrained)
        """
        if self.params.fixed:
            cents = unscale(cents)
        if self.params.assignment == "constrained":
            self.postprocess_centroid(cents)
        else:
            self.centroids = cents

    def compute_assoc(self):
        """
        Computes the associations between the data points and the centroids.
        :return: Associations
        """
        self.distances = distance_matrix(self.values, self.centroids, fixed=False)
        if self.params.assignment == "constrained":
            self.associations, _ = mincostflow(self.values.shape[0], self.params.k,
                                               self.distances,
                                               minCluster=self.params.min_cluster,
                                               maxCluster=self.params.max_cluster,
                                               maxDist=self.params.max_dist,
                                               fixed=False,
                                               max_dist_mode=self.params.max_dist_mode)
        else:
            self.associations = np.argmin(self.distances, axis=1)

    def flow_step(self, old_centroids):
        """
        Computes the centroids based on results from the min-cost flow step.
        :param old_centroids: Old centroids
        :return: Updated centroids and local counts
        """
        centroids, local_counts = update_centroids_multiassoc(self.values, self.associations, old_centroids)
        return centroids, local_counts.astype(np.int32)

    def basic_step(self, old_centroids):
        """
        Computes the centroids based on results from the basic (unconstrained) step.
        :param old_centroids: Old centroids
        :return: Updated centroids and local counts
        """
        centroids = []
        local_counts = []
        constraint_set = (np.argpartition(self.distances, self.params.min_cluster, axis=0)[:self.params.min_cluster]).T
        for cluster in range(self.params.k):
            indices = self.associations == cluster
            indices[constraint_set[cluster]] = True
            count = sum(indices)
            local_counts.append(count)
            if count > 0:
                centroid = self.values[indices].sum(axis=0) / count
            else:
                centroid = old_centroids[cluster]
            centroids.append(centroid)
        centroids = np.array(centroids)
        local_counts = np.array(local_counts, dtype=np.int32)
        return centroids, local_counts

    def step_sumcount(self):
        """
        Performs a step in the SumCount setting.
        :return: Updated centroids and local counts (fixed and divided by counts)
        """
        old_centroids = self.centroids
        if self.params.assignment == "constrained":
            centroids, local_counts = self.flow_step(old_centroids)
        else:
            centroids, local_counts = self.basic_step(old_centroids)

        totals = centroids * np.expand_dims(local_counts, axis=1)
        if self.params.fixed:
            totals = to_fixed(totals)
        return totals, local_counts

    def step_centroid(self):
        """
        Performs a step in the Centroid setting.
        :return: Updated centroids (fixed, divided by number of clients for assignmentregation)
        """
        old_centroids = self.centroids
        if self.params.assignment == "constrained":
            centroids, _ = self.flow_step(old_centroids)
        else:
            centroids, _ = self.basic_step(old_centroids)
        centroids = centroids / self.params.num_clients
        if self.params.fixed:
            centroids = to_fixed(centroids)
        return centroids


class MaskedClient(UnmaskedClient):
    """
    MaskedClient class
    """

    def __init__(self, index: int, values, params: Params):
        """
        Initializes the MaskedClient object.
        :param index: client index
        :param values: data points
        :param params: parameters
        """
        super().__init__(index, values, params)
        sum_masks = [
            np.array([self.common_generator.randint(0, MOD, (params.k, params.dim))
                      for _ in range(params.iters + 1)])
            for __ in range(params.num_clients)
        ]
        self.sum_dmasks = sum(sum_masks)
        self.sum_emasks = sum_masks[index]
        count_masks = [
            np.array([self.common_generator.randint(0, MOD, params.k)
                      for _ in range(params.iters + 1)])
            for __ in range(params.num_clients)
        ]
        self.count_dmasks = sum(count_masks)
        self.count_emasks = count_masks[index]
        self.curr_iter = 0

    def step_sumcount(self):
        """
        Performs a step in the SumCount setting.
        :return: Masked totals and masked counts
        """
        totals, counts = super().step_sumcount()
        masked_totals = self.sum_emasks[self.curr_iter] + totals
        masked_counts = self.count_emasks[self.curr_iter] + counts
        return masked_totals, masked_counts

    def step_centroid(self):
        """
        Performs a step in the Centroid setting.
        :return: Masked centroids (scaled down by number of clients)
        """
        cents = super().step_centroid()
        masked_cents = self.sum_emasks[self.curr_iter] + cents
        return masked_cents

    def update_centroids_sumcount(self, masked_totals, masked_counts):
        """
        Updates the centroids in the SumCount setting.
        :param masked_totals: Masked totals (assignmentregated from all clients)
        :param masked_counts: Masked counts (assignmentregated from all clients)
        :return: Updated centroids
        """
        totals = masked_totals - self.sum_dmasks[self.curr_iter]
        counts = masked_counts - self.count_dmasks[self.curr_iter]
        self.curr_iter += 1
        super().update_centroids_sumcount(totals, counts)

    def update_centroids_centroid(self, masked_cents):
        """
        Updates the centroids in the Centroid setting.
        :param masked_cents: Masked centroids (assignmentregated from all clients)
        :return: Updated centroids
        """
        cents = masked_cents - self.sum_dmasks[self.curr_iter]
        self.curr_iter += 1
        super().update_centroids_centroid(cents)

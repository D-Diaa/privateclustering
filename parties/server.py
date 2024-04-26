from alg.dp_applier import DPApplier
from configs.params import Params

"""
This file contains the server class for the distributed clustering algorithms.
The server class is implemented as a subclass of the DPApplier class.
The server class is responsible for applying the differential privacy noise to the cluster centroids.
"""


class Server(DPApplier):

    def __init__(self, params: Params):
        super().__init__(params)
        # When the server applies dp noise, it sees clusters that are from all clients hence the constraints are summed
        self.params.min_cluster *= self.params.num_clients
        self.params.max_cluster *= self.params.num_clients
        self.dp_setup(self.params)
        self.curr_iter = 0

    def step_sumcount(self, masked_sums, masked_counts):
        if self.params.assignment == "constrained" and self.params.max_dist_mode == "hard":
            self.update_maxdist(self.curr_iter)
        masked_sum = sum(masked_sums)
        masked_count = sum(masked_counts)
        masked_sum, masked_count = self.randomize_masked_sumcount(masked_sum, masked_count)
        self.curr_iter += 1
        return masked_sum, masked_count

    def step_centroid(self, masked_cents):
        if self.params.assignment == "constrained" and self.params.max_dist_mode == "hard":
            self.update_maxdist(self.curr_iter)
        masked_cent = sum(masked_cents)
        masked_cent = self.randomize_masked_centroid(masked_cent)
        self.curr_iter += 1
        return masked_cent

import itertools
import json
import os
from argparse import ArgumentParser
from datetime import datetime
from timeit import default_timer as timer

import pandas as pd

from alg.calc_params import calculate_cluster_sizes, calculate_max_dist, calculate_iters
from configs.defaults import *
from configs.params import Params
from data_io.data_handler import shuffle_and_split
from utils.evaluations import evaluate
from utils.utils import *


class Experiments:
    def __init__(self, protocol, k, dataset, values, params_list, exp_type, results_folder, plot, with_comm):
        self.results_folder = results_folder
        self.protocol = protocol
        self.k = k
        self.values = values
        self.dataset = dataset
        self.plot = plot
        self.with_comm = with_comm
        self.params_list = params_list
        self.results_df = None
        self.exp_type = exp_type
        self.proto_name = ("d" if params_list["max_dist_mode"] == "hard" else "") + params_list["dp_release"]
        self.failed = []

    def run_protocol(self, params):
        proportions = np.ones(params.num_clients)
        proportions /= proportions.sum()
        value_lists = shuffle_and_split(self.values, params.num_clients, proportions)
        start = timer()
        centroids = self.protocol(value_lists, params)
        end = timer()
        if params.fixed:
            values_unscaled = unscale(self.values.copy())  # create a copy before unscale operation
            centroids = unscale(centroids)
        else:
            values_unscaled = self.values
        # Evaluate the clustering on all the dataset
        variances = evaluate(centroids, values_unscaled)

        if self.plot:
            def generate_filename(params):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}_{params.assignment}_{params.dp}_{params.eps}_" \
                           f"[{params.min_cluster}-{params.max_cluster}]_{params.seed}"
                return filename

            plot_clusters(centroids, values_unscaled)
            filename = generate_filename(params)
            plt.title(filename)
            folder = f"results/{self.dataset}/{self.protocol.__name__}"
            os.makedirs(folder, exist_ok=True)
            plt.savefig(f"{folder}/{filename}.png")

        variances["elapsed"] = end - start

        return variances

    def run(self):
        # Define the order of keys for parameter combinations
        params_order = ["assignments", "init_methods", "sum_posts", "count_posts", "cent_posts", "delays", "dps"]

        cluster_mins, cluster_maxs = calculate_cluster_sizes(self.values.shape[0], self.k, self.params_list)

        # Calculate constant values once since they don't change
        dimension, data_size = self.values.shape[1], self.values.shape[0]

        # Generator function for creating params
        def all_params():
            for (assignment, init_method, sum_post, count_post, cent_post, delay, dp) in (
                    itertools.product(*[self.params_list[key] for key in params_order])):
                for eps_budget in self.get_eps_budgets(dp):
                    params = Params(
                        num_clients=self.params_list["num_clients"],
                        k=self.k,
                        dim=dimension,
                        data_size=data_size,
                        dp=dp,
                        eps=eps_budget,
                        assignment=assignment,
                        init=init_method,
                        dp_release=self.params_list["dp_release"],
                        sum_post=sum_post,
                        count_post=count_post,
                        cent_post=cent_post,
                        delay=delay,
                    )
                    if assignment == "unconstrained":
                        params.max_cluster = data_size
                        params.min_cluster = 0
                        params.min_cluster_alpha = 0
                        params.max_cluster_alpha = 0
                        params.max_dist_mode = "disabled"
                        yield params
                    else:
                        for min_cluster, min_alpha in cluster_mins:
                            for max_cluster, max_alpha in cluster_maxs:
                                params.min_cluster = min_cluster
                                params.max_cluster = max_cluster
                                params.min_cluster_alpha = min_alpha
                                params.max_cluster_alpha = max_alpha
                                params.max_dist_mode = self.params_list["max_dist_mode"]
                                params.max_dist = calculate_max_dist(params)
                                yield params

        # Run experiments for all parameter combinations
        for p in all_params():
            self.run_experiment(p)

    def get_eps_budgets(self, dp):
        if dp == "none":
            return [0]
        else:
            return self.params_list["eps_budgets"]

    def run_experiment(self, params):
        # Iters calculated optimally: clients negotiate data size with the server, which sets iterations.
        params.iters = calculate_iters(params)

        # Initialize metrics to collect
        total_metrics = {
            "Normalized Intra-cluster Variance (NICV)": [],
            "Between-Cluster Sum of Squares (BCSS)": [],
            "Empty Clusters": [],
            "elapsed": []
        }

        # Track successful experiments and the total denominator for averages
        succesful_experiments, experiment_count = 0, 0

        # Run protocol with different seeds
        for seed in self.params_list["seeds"]:
            params.seed = seed
            try:
                metrics = self.run_protocol(params)
                # Validate and collect metrics
                for metric, value in metrics.items():
                    if np.isnan(value):
                        raise ValueError("NAN Experiment")
                    total_metrics[metric].append(value)
                succesful_experiments += metrics["Empty Clusters"] == 0
                experiment_count += 1
            except Exception as e:
                print(e)
                self.failed.append(list(params))
                self.save()

        # Calculate mean and confidence intervals for collected metrics
        mm_metrics = {metric: mean_confidence_interval(values) for metric, values in total_metrics.items()}
        avg_metrics = {metric: values[0] for metric, values in mm_metrics.items()}
        conf_metrics = {f"{metric}_h": values[1] for metric, values in mm_metrics.items()}

        # Create a results dictionary from params
        result = {attr: getattr(params, attr) for attr in vars(params)}
        # Add calculated metrics and remove unnecessary keys
        result.pop("bounds", None)
        result.pop("attributes", None)
        result.update({
            "successes": succesful_experiments,
            "experiments": experiment_count,
            "post_method": f"{params.sum_post}_{params.count_post}_{params.cent_post}",
            **avg_metrics,
            **conf_metrics,
        })
        if self.with_comm:
            result.update(comm.get_comm_stats())
        # Append results to DataFrame and save
        new_results_df = pd.DataFrame([result])
        self.results_df = pd.concat([self.results_df, new_results_df],
                                    ignore_index=True) if self.results_df is not None else new_results_df
        self.save()

    def save(self):
        folder = f"{self.results_folder}/{self.exp_type}/{self.dataset}/{self.proto_name}"
        os.makedirs(folder, exist_ok=True)
        self.results_df = self.results_df.sort_values("Normalized Intra-cluster Variance (NICV)")
        file = f"{folder}/variances.csv"
        if self.with_comm:
            rank_str = f"{comm.rank}" if comm.world_size > 1 else ""
            file = f"{folder}/variances_{rank_str}.csv"
        self.results_df.to_csv(file)
        with open(f"{folder}/failed.json", "w") as f:
            json.dump(self.failed, f)


def parse_args():
    parser = ArgumentParser(description="Run experiments for multiparty DP clustering")
    parser.add_argument("--exp_type", default="accuracy", help="type of experiment")
    parser.add_argument("--dp_release", default="centroid", help="dp release",
                        choices=["sumcount", "centroid"])
    parser.add_argument("--plot", action="store_true", help="plot clusters")
    parser.add_argument("--num_runs", default=20, type=int, help="number of runs")
    parser.add_argument("--max_dist_mode", default="disabled", help="max distance mode",
                        choices=["disabled", "hard", "soft"])
    parser.add_argument("--init_method", default="optimal", help="initialization method",
                        choices=["optimal", "random"])
    parser.add_argument("--sum_post", default="none", help="sum post-processing method",
                        choices=["none", "truncate", "fold"])
    parser.add_argument("--count_post", default="none", help="count post-processing method",
                        choices=["none", "truncate", "fold"])
    parser.add_argument("--cent_post", default="fold", help="centroid post-processing method",
                        choices=["none", "truncate", "fold"])
    parser.add_argument("--cluster_min_alpha", default=1.25, type=float, help="min cluster alpha")
    parser.add_argument("--cluster_max_alpha", default=1.25, type=float, help="max cluster alpha")
    parser.add_argument("--results_folder", default="utility", help="folder for results  ")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fixed = True
    plot = args.plot
    exp_type = args.exp_type
    dp_release = args.dp_release
    # default (preferred) parameters
    params_list = {
        "num_runs": args.num_runs,
        "seeds": range(args.num_runs),
        "init_methods": [args.init_method],
        "sum_posts": [args.sum_post],
        "count_posts": [args.count_post],
        "cent_posts": [args.cent_post],
        "max_dist_mode": args.max_dist_mode,
        "cluster_min_alphas": [args.cluster_min_alpha],
        "cluster_max_alphas": [args.cluster_max_alpha],
        "fixed": fixed,
    }

    from data_io.data_handler import load_txt, normalize

    # could be overridden if ablation is used
    params_list.update(parameters[exp_type])
    if "timing" in exp_type:
        from data_io.comm import comm

        num_clients = comm.world_size - 1
        params_list["num_clients"] = num_clients
        exp_type = f"timing_{num_clients}"
    else:
        params_list["num_clients"] = 2

    for dataset in params_list["datasets"]:
        params_list["dp_release"] = dp_release
        if "synth" in dataset.lower():
            k = int(dataset.split("_")[1])
        else:
            k = num_clusters[dataset]
        dataset_file = f"data/{dataset}.txt"
        if not os.path.isfile(dataset_file):
            continue
        values = load_txt(dataset_file)

        values = normalize(values, fixed)

        if "timing" in exp_type:
            from protocols.server_mpi import mpi_server_sumcount, mpi_server_centroid

            proto = mpi_server_sumcount if dp_release == "sumcount" else mpi_server_centroid
            with_comm = True
        else:
            from protocols.server_local import local_server_sumcount, local_server_centroid

            proto = local_server_sumcount if dp_release == "sumcount" else local_server_centroid
            with_comm = False

        experiment = Experiments(proto, k, dataset, values, params_list, exp_type, args.results_folder, plot, with_comm)
        experiment.run()

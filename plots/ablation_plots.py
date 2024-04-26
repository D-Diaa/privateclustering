import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs.defaults import ablate_dataset
from utils.utils import mean_confidence_interval

plt.rcParams.update({'font.size': 16})


def load_data(file_path):
    return pd.read_csv(file_path)


def partition_data(data):
    partition_eps_1 = data[data['eps'] == 1.0]
    partition_eps_01 = data[data['eps'] == 0.1]
    partition_eps_0 = data[data['eps'] == 0]
    return partition_eps_1, partition_eps_01, partition_eps_0


def plot_nicv_vs_key(data, title, key="min_alpha"):
    plt.figure(figsize=(10, 6))
    ys = {}
    xs = {}
    y_min, y_max = 1e9, -1e9
    grouped_data = {}
    for cluster in data:
        _, k, d, _ = cluster.split("_")
        group_key = f"{k}"
        if group_key not in grouped_data:
            grouped_data[group_key] = []
        sorted_data = data[cluster].sort_values(key)
        xs[cluster] = sorted_data[key].values.tolist()
        ys[cluster] = sorted_data['Normalized Intra-cluster Variance (NICV)'].values
        if len(xs[cluster]) == 0:
            continue

        ys[cluster] = ys[cluster][1:]
        xs[cluster] = xs[cluster][1:]
        ys[cluster] = ys[cluster].tolist()
        grouped_data[group_key].append(ys[cluster])
        # plt.plot(xs[cluster], ys[cluster], label=cluster)
        plt.plot(xs[cluster], ys[cluster], linewidth=1, alpha=0.2, color='gray')
    all_xs = np.unique(np.concatenate(list(xs.values())))
    for group in grouped_data:
        avg_ys = []
        for i in range(len(grouped_data[group][0])):
            y_values = [group[i] for group in grouped_data[group] if i < len(group)]
            y_mean = np.mean(y_values)
            y_min = min(y_min, y_mean)
            y_max = max(y_max, y_mean)
            avg_ys.append(y_mean)

        plt.plot(all_xs, avg_ys, label=group)
    # Calculate average and standard deviation of y-values for each x-value
    avg_ys = []
    std_ys = []
    for x in all_xs:
        y_values = [ys[cluster][i] for cluster in xs if x in xs[cluster] for i, x_val in enumerate(xs[cluster]) if
                    x_val == x]
        avg_ys.append(np.mean(y_values))
        std_ys.append(mean_confidence_interval(y_values)[1])
    diff = (y_max - y_min) / 4
    # Plot average line with standard deviation shading
    plt.plot(all_xs, avg_ys, 'k--', linewidth=3)
    plt.fill_between(all_xs, np.subtract(avg_ys, std_ys), np.add(avg_ys, std_ys), color='gray', alpha=0.2)
    plt.gca().set_ylim([y_min - diff, y_max + diff])
    plt.xlabel(key)
    plt.ylabel('NICV')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"figs/{title}.png")


def bar_nicv_vs_key_grouped(data, title, key="init_method", defaults=None):
    plt.figure(figsize=(10, 6))
    ys = {}
    y_norm = {}
    y_std = {}
    width = 0.9

    # Determine unique methods
    methods = sorted(set(method for cluster_data in data.values() for method in cluster_data[key].unique()))

    # Define a color map for methods
    colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
    method_colors = {method: color for method, color in zip(methods, colors)}

    for cluster in data:
        if len(data[cluster]) == 0:
            continue
        unc_index = data[cluster]['assignment'] == "unconstrained"
        normalizing_index = unc_index
        for def_key in defaults:
            normalizing_index = normalizing_index & (data[cluster][def_key] == defaults[def_key])
        y_norm[cluster] = data[cluster]['Normalized Intra-cluster Variance (NICV)'][normalizing_index].values

    for method in methods:
        method_values = []
        for cluster in data:
            if len(data[cluster]) == 0:
                continue
            c_data = data[cluster]
            if key in c_data and method in c_data[key].values:
                method_index = c_data[key] == method
                method_data = c_data[method_index]
                if len(method_data) == 0:
                    continue
                y_values = method_data['Normalized Intra-cluster Variance (NICV)'].values
                # y_values /= y_norm[cluster]
                unc_index = method_data['assignment'] == "unconstrained"
                y_values = y_values[~unc_index]
                method_values.extend(y_values)

        # Calculate mean and standard deviation for each method
        ys[method] = np.mean(method_values)
        y_std[method] = mean_confidence_interval(method_values)[1]

    # Plot each method's bars with a unique position
    for i, method in enumerate(methods):
        plt.bar(i, ys[method], width, color=method_colors[method],
                yerr=y_std[method],  # label=method
                )

    # plt.title(title)
    plt.xlabel('Postprocessing Method')
    plt.ylabel('NICV')
    # rename methods from "none_none_x" to "x"
    methods = [method.split("_")[-1] for method in methods]
    # make first letter capital
    methods = [method[0].upper() + method[1:] for method in methods]
    plt.xticks(range(len(methods)), methods)  # Set the x-ticks to be the methods
    bounds = list(ys.values())
    diff = max(bounds) - min(bounds)
    diff = max(diff / 2, 0.1)
    plt.gca().set_ylim([min(bounds) - diff, max(bounds) + diff])
    # gridlines
    plt.gca().yaxis.grid(True)
    plt.grid(True)
    # plt.legend()
    plt.tight_layout()  # Adjust layout to make room for x-tick labels
    plt.savefig(f"figs/{title}.png")
    # sort ys by ys[method] and print
    sorted_ys = {k: v for k, v in sorted(ys.items(), key=lambda item: item[1])}
    print(title, sorted_ys)


def bar_nicv_vs_key(data, title, key="init_method", defaults=None):
    plt.figure(figsize=(10, 6))
    ys = {}
    xs = {}
    y_norm = {}
    width = 0.9

    # Determine unique clusters and methods
    unique_clusters = sorted(data.keys())
    methods = sorted(set(method for cluster_data in data.values() for method in cluster_data[key].unique()))
    num_clusters = len(unique_clusters)

    # Create a mapping from cluster to x position
    cluster_positions = {cluster: i for i, cluster in enumerate(unique_clusters)}
    bar_width = width / len(methods)  # Adjust bar width based on the number of methods

    # Define a color map for methods
    colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
    method_colors = {method: color for method, color in zip(methods, colors)}

    for cluster in data:
        if len(data[cluster]) == 0:
            continue
        unc_index = data[cluster]['assignment'] == "unconstrained"
        normalizing_index = unc_index
        for def_key in defaults:
            normalizing_index = normalizing_index & (data[cluster][def_key] == defaults[def_key])
        y_norm[cluster] = data[cluster]['Normalized Intra-cluster Variance (NICV)'][normalizing_index].values

    for method in methods:
        for cluster in unique_clusters:
            c_data = data[cluster]
            if key in c_data and method in c_data[key].values:
                method_data = c_data[c_data[key] == method]
                if len(method_data) == 0:
                    continue
                xs[cluster] = np.array([cluster_positions[cluster]])
                ys[method] = method_data['Normalized Intra-cluster Variance (NICV)'].values

                unc_index = method_data['assignment'] == "unconstrained"

                ys[method] /= y_norm[cluster]
                ys[method] = ys[method][~unc_index]

                # Adjust positions for each cluster to avoid bar overlap
                adjusted_positions = [pos + methods.index(method) * bar_width for pos in
                                      xs[cluster]]

                # Plot each method's bars with a unique position
                plt.bar(adjusted_positions, ys[method], bar_width, color=method_colors[method],
                        label=method if cluster == unique_clusters[0] else "")

    # plt.title(title)
    plt.xlabel('Cluster')
    plt.ylabel('NICV')
    plt.xticks(np.arange(num_clusters) + (len(methods) / 2 - 1 / 2) * bar_width,
               unique_clusters)  # Set the x-ticks to be the clusters
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # Adjust layout to make room for x-tick labels
    plt.savefig(f"../figs/{title}.png")


datasets = ablate_dataset


def main_lines(proto, folder="ablation"):
    keys = {
        "mincluster": "min_cluster_alpha",
        "maxcluster": "max_cluster_alpha",
    }
    for key in keys:
        partition_eps_1 = {}
        partition_eps_01 = {}
        partition_eps_0 = {}
        for dataset in datasets:
            file_path = f'{folder}/{key}_ablation/{dataset}/{proto}/variances.csv'
            if os.path.isfile(file_path):
                print(file_path)
                data = load_data(file_path)
                _eps_1, _eps_01, _eps_0 = partition_data(data)
                partition_eps_1[dataset] = _eps_1
                partition_eps_01[dataset] = _eps_01
                partition_eps_0[dataset] = _eps_0

        plot_nicv_vs_key(partition_eps_1, f"{proto} NICV vs {keys[key]} (eps=1.0)", key=keys[key])
        plot_nicv_vs_key(partition_eps_01, f"{proto} NICV vs {keys[key]} (eps=0.1)", key=keys[key])
        plot_nicv_vs_key(partition_eps_0, f"{proto} NICV vs {keys[key]} (Non Private)", key=keys[key])


def main_bars(proto, folder="ablation"):
    keys = {
        "post": "post_method",
    }
    defaults = {
        "post": {
            'init': "optimal",
            'post_method': "none_none_none",
        }
    }
    for key in keys:
        partition_eps_1 = {}
        partition_eps_0 = {}
        partition_eps_01 = {}
        for dataset in datasets:
            file_path = f'{folder}/{key}_ablation/{dataset}/{proto}/variances.csv'
            if os.path.isfile(file_path):
                print(file_path)
                data = load_data(file_path)
                _eps_1, _eps_01, _eps_0 = partition_data(data)
                partition_eps_1[dataset] = _eps_1
                partition_eps_01[dataset] = _eps_01
                partition_eps_0[dataset] = _eps_0

        bar_nicv_vs_key_grouped(partition_eps_1, f"{proto} NICV vs {keys[key]} (eps=1.0)", keys[key], defaults[key])
        bar_nicv_vs_key_grouped(partition_eps_01, f"{proto} NICV vs {keys[key]} (eps=0.1)", keys[key], defaults[key])
        bar_nicv_vs_key_grouped(partition_eps_0, f"{proto} NICV vs {keys[key]} (Non Private)", keys[key], defaults[key])


if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    argc = len(sys.argv)
    if argc > 1:
        proto = sys.argv[1]
        main_lines(proto)
        main_bars(proto)
    elif argc > 2:
        proto = sys.argv[1]
        folder = sys.argv[2]
        main_lines(proto, folder)
        main_bars(proto, folder)
    else:
        for proto in ["centroid", "sumcount"]:
            main_lines(proto)
            main_bars(proto)

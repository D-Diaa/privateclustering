import os
import sys
from os.path import isdir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

plt.rcParams.update({'font.size': 16})


# Configuration for legend handles and styling
def create_legend_handles():
    method_styles = [
        {'marker': 'o', 'color': 'red', 'label': 'SuLloyd'},
        {'marker': 'o', 'color': 'green', 'label': 'CSuLloyd'},
        {'marker': 'o', 'color': 'blue', 'label': 'FastLloyd'},
        {'marker': 'o', 'color': 'black', 'label': 'Lloyd'}
    ]
    method_handles = [Line2D([0], [0], marker=style['marker'], color='w', markerfacecolor=style['color'], markersize=10,
                             label=style['label']) for style in method_styles]

    return method_handles


def create_legend_image(config):
    # Initialize a figure with a specific size that might need adjustment
    fig, ax = plt.subplots(figsize=(6, 1))  # Adjust figsize to fit the legend as needed
    # Generate the legend handles using the previously defined function
    legend_handles = create_legend_handles()
    # Number of columns set to the number of legend handles to align them horizontally
    ncol = len(legend_handles)
    # Create the legend with the handles, specifying the number of columns
    ax.legend(handles=legend_handles, loc='center', ncol=ncol, frameon=False)
    # Hide the axes as they are not needed for the legend
    ax.axis('off')
    # Remove all the margins and paddings by setting bbox_inches to 'tight' and pad_inches to 0
    # The dpi (dots per inch) parameter might be adjusted for higher resolution
    fig.savefig(os.path.join(config['datasets_folders'][0], "legend.png"), bbox_inches='tight', pad_inches=0, dpi=300)
    # Clear the plot to free up memory
    plt.clf()


# Constants and configurations
CONFIG = {
    'eps_range': [0, 2],
    'method_names': {
        ("unconstrained", "sumcount", "laplace"): "SuLloyd",
        ("unconstrained", "sumcount", "none"): "Lloyd",
        ("constrained", "sumcount", "laplace"): "CSuLloyd",
        ("constrained", "dsumcount", "laplace"): "CSuLloyd",
        ("constrained", "centroid", "laplace"): "FastLloyd",
        ("constrained", "dcentroid", "laplace"): "FastLloyd",
    },
    "method_colors": {
        ("unconstrained", "sumcount", "laplace"): "red",
        ("unconstrained", "sumcount", "none"): "black",
        ("constrained", "sumcount", "laplace"): "green",
        ("constrained", "dsumcount", "laplace"): "green",
        ("constrained", "centroid", "laplace"): "blue",
        ("constrained", "dcentroid", "laplace"): "blue",
    },
    'datasets_folders': ["utility/accuracy"],
    'metrics': ["Normalized Intra-cluster Variance (NICV)", "Empty Clusters"]
}
metrics_dict = {
    "Normalized Intra-cluster Variance (NICV)": "NICV",
    "Between-Cluster Sum of Squares (BCSS)": "BCSS",
    "Empty Clusters": "Empty Clusters",
}


# Function to process datasets and generate plots
def process_datasets(config):
    for dataset_folder in config['datasets_folders']:
        for dataset in os.listdir(dataset_folder):
            all_data = None
            for dp_release in ['sumcount', 'centroid', 'dsumcount', 'dcentroid']:
                folder = os.path.join(dataset_folder, dataset, dp_release)
                if not isdir(folder):
                    continue
                filepath = os.path.join(folder, "variances.csv")
                if not os.path.exists(filepath):
                    print(f"FAILED: {folder}")
                    continue

                sample_data = pd.read_csv(filepath)
                sample_data['dp_release'] = dp_release
                if all_data is None:
                    all_data = sample_data
                else:
                    all_data = all_data._append(sample_data)
                config['dataset'] = dataset
                plot_data(sample_data, folder, config)
            if all_data is not None:
                plot_data(all_data, os.path.join(dataset_folder, dataset), config)


def plot_data(data, folder, config):
    for metric in config['metrics']:
        filtered_data = data[['assignment', 'dp', 'dp_release', 'eps', metric, f"{metric}_h"]].sort_values(by='eps')
        combinations = filtered_data[['assignment', 'dp', 'dp_release']].drop_duplicates().values
        for assignment, dp, dp_release in combinations:
            plot_metric(filtered_data, assignment, dp, dp_release, metric, config)

        finalize_plot(metric, folder, config["dataset"])


def plot_metric(data, assignment, dp, dp_release, metric, config):
    # rename metric to be more descriptive
    subset = data[(data['assignment'] == assignment) & (data['dp'] == dp) & (data['dp_release'] == dp_release)]
    method = (assignment, dp_release, dp)
    if method not in config['method_names']:
        return
    linestyle = 'solid' if dp_release in ['sumcount', 'centroid'] else 'dashed'
    color = config['method_colors'][method]
    label = config['method_names'][method]
    eps = subset['eps']

    if dp == 'none':
        eps = np.linspace(config['eps_range'][0], config['eps_range'][1])
        plt.hlines(y=subset[metric].mean(), xmin=config['eps_range'][0], xmax=config['eps_range'][1],
                   linestyle=linestyle, color=color, label=label)
    else:
        plt.scatter(eps, subset[metric], linestyle=linestyle, color=color, label=label)
        plt.plot(eps, subset[metric], linestyle=linestyle, color=color, label=label)

    plt.fill_between(eps, subset[metric] - subset[f"{metric}_h"], subset[metric] + subset[f"{metric}_h"], color=color,
                     alpha=0.2)


def finalize_plot(metric, folder, dataset=""):
    plt.xlabel('ε')
    plt.ylabel(metrics_dict[metric])
    plt.grid(True)
    plt.tight_layout()
    # plt.legend(handles=create_legend_handles(), loc='upper right')
    plt.savefig(os.path.join(folder, f"{dataset}_{metric}.png"))
    plt.clf()


if __name__ == "__main__":
    argc = len(sys.argv)
    if argc > 1:
        CONFIG['datasets_folders'] = [f"{sys.argv[1]}/accuracy"]
    process_datasets(CONFIG)
    create_legend_image(CONFIG)

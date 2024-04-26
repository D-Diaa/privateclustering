import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.transforms import Bbox

plt.rcParams.update({'font.size': 18})


# Refactored code
def extract_data(main_folder, assignments):
    data_dict = {assignment: {} for assignment in assignments}

    for folder in os.listdir(main_folder):
        if "synth" in folder:
            for assignment in assignments:
                clusters, dimensions = map(int, folder.split('_')[1:3])
                variances_file_path = os.path.join(main_folder, folder, f'{assignments[assignment]}',
                                                   'variances.csv')
                try:
                    df = pd.read_csv(variances_file_path)
                    # Using query method for cleaner filtering
                    query_str = f"dp == 'laplace' and assignment == '{assignment}'"
                    row = df.query(query_str)

                    if not row.empty and 'Normalized Intra-cluster Variance (NICV)' in row.columns:
                        nicv_value = row['Normalized Intra-cluster Variance (NICV)'].iloc[0]
                        if (clusters, dimensions) not in data_dict[assignment]:
                            data_dict[assignment][(clusters, dimensions)] = []
                        data_dict[assignment][(clusters, dimensions)].append(nicv_value)
                    else:
                        print(f"No 'laplace' data for {assignment} in file {variances_file_path}")
                except FileNotFoundError:
                    print(f"File not found: {variances_file_path}")
    for assignment in assignments:
        for key in data_dict[assignment]:
            vals = data_dict[assignment][key]
            data_dict[assignment][key] = sum(vals) / len(vals)

    return data_dict


def generate_heatmap_from_matrix(matrix, x_labels, y_labels, cmap, vmin, vmax, file_name, fmt=".2f", no_bar=False):
    plt.figure(figsize=(10, 8))
    if no_bar:
        sns.heatmap(matrix, annot=True, fmt=fmt, xticklabels=x_labels, yticklabels=y_labels, cmap=cmap, vmin=vmin,
                    vmax=vmax, cbar=False)
    else:
        sns.heatmap(matrix, annot=True, fmt=fmt, xticklabels=x_labels, yticklabels=y_labels, cmap=cmap, vmin=vmin,
                    vmax=vmax)
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Number of Clusters')
    plt.gca().invert_yaxis()  # Invert the y-axis
    scale_exp = "70" if "70" in main_folder else "21"
    if not no_bar:
        bbox = Bbox([[7.5, -1], [9, 7.5]])
        plt.savefig(f'{main_folder}/{scale_exp}_{file_name}.png', bbox_inches=bbox)
    else:
        plt.savefig(f'{main_folder}/{scale_exp}_{file_name}.png')
    plt.close()  # Close the plot to prevent it from displaying in interactive environments


def create_heatmap(data, assignment_name, vmin, vmax, no_bar=False):
    # Extract clusters and dimensions as separate lists
    clusters = sorted(set(key[0] for key in data.keys()))
    dimensions = sorted(set(key[1] for key in data.keys()))

    # Initialize a matrix to store NICV values
    nicv_matrix = np.zeros((len(clusters), len(dimensions)))

    # Populate the matrix with NICV values
    for i, cluster in enumerate(clusters):
        for j, dimension in enumerate(dimensions):
            nicv_matrix[i, j] = data.get((cluster, dimension), np.nan)  # Use NaN for missing values

    # Create a custom colormap: green (low), yellow (middle), red (high)
    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
    name = f'Heatmap' + ("_bar" if not no_bar else f'_{assignment_name}')
    generate_heatmap_from_matrix(nicv_matrix, dimensions, clusters, cmap, vmin, vmax, name, no_bar=no_bar)

    return nicv_matrix


if __name__ == "__main__":
    assignments = {"unconstrained": "sumcount", "constrained": "centroid"}
    argc = len(sys.argv)
    if argc > 1:
        main_folder = sys.argv[1]
    else:
        main_folder = "utility/scale/scale70"
    folder = f"{main_folder}"
    print(f"Processing data in {folder}")
    # Extract the data
    data_dict = extract_data(folder, assignments)

    # Find the global min and max NICV values for the shared scale
    all_nicv_values = [value for assignment_data in data_dict.values() for value in assignment_data.values()]
    vmin, vmax = min(all_nicv_values), max(all_nicv_values)

    # Generate heatmaps for each assignmentregation method and store NICV matrices
    nicv_matrices = {}
    for assignment, data in data_dict.items():
        nicv_matrices[assignment] = create_heatmap(data, assignment, vmin, vmax, no_bar=True)

    create_heatmap(data, assignment, vmin, vmax, no_bar=False)
    # Calculate the division matrix (ensure to handle division by zero appropriately)
    assignment_keys = list(assignments.keys())
    first_matrix = nicv_matrices[assignment_keys[0]]
    second_matrix = nicv_matrices[assignment_keys[1]]
    division_matrix = np.divide(first_matrix - second_matrix, first_matrix, out=np.zeros_like(first_matrix),
                                where=second_matrix != 0)
    # Assuming division_matrix, and the labels (dimensions and clusters) are already defined
    # cmap_division = sns.color_palette("YlGn", as_cmap=True)
    # cmao with shades of blue
    cmap_division = sns.color_palette("Blues", as_cmap=True)
    vmin_div, vmax_div = np.nanmin(division_matrix), np.nanmax(division_matrix)

    # Generate the heatmap for the division matrix
    generate_heatmap_from_matrix(division_matrix, sorted(set(key[1] for key in data.keys())),
                                 sorted(set(key[0] for key in data.keys())), cmap_division, vmin_div, vmax_div,
                                 'Heatmap_Division', fmt=".0%", no_bar=True)

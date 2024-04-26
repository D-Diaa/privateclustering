import os
import re
import sys

import pandas as pd

LAN = 0.000125
WAN = 0.025


def process_row(row):
    time = round((row['elapsed'] / row['iters']) * 1000, 2)
    ci = round((row['elapsed_h'] / row['iters']) * 1000, 2)
    return f"${time} \\pm {ci}$"


def process_all_datasets(extraction_dir, proto):
    def process_dataset(directory_path):
        variances_files = [f for f in os.listdir(directory_path) if f.startswith('variances_')]

        server_df = pd.read_csv(os.path.join(directory_path, 'variances_0.csv'))
        filtered_df = server_df[(server_df['assignment'] == 'constrained') & (server_df['dp'] == 'laplace')]
        assert len(filtered_df) == 2  # LAN, WAN

        lan_time_ci = process_row(filtered_df[filtered_df['delay'] == LAN].iloc[0])
        wan_time_ci = process_row(filtered_df[filtered_df['delay'] == WAN].iloc[0])

        comm_size_total = 0
        for file in variances_files:
            df = pd.read_csv(os.path.join(directory_path, file))
            df_filtered = df[(df['assignment'] == 'constrained') &
                             (df['dp'] == 'laplace') &
                             (df['delay'] == LAN)]
            comm_size_total += df_filtered['comm_size'].sum()

        return lan_time_ci, wan_time_ci, comm_size_total

    results = []
    for root, dirs, files in os.walk(extraction_dir):
        for name in dirs:
            dataset_path = os.path.join(root, name, proto)
            if os.path.isdir(dataset_path):
                lan_time_ci, wan_time_ci, comm_size = process_dataset(dataset_path)
                if lan_time_ci and wan_time_ci:  # Include only if matching the filtering criteria
                    # Extract d, k, n from the dataset name
                    match = re.match(r"timesynth_(\d+)_(\d+)_(\d+)", name)
                    if match:
                        k, d, n = match.groups()
                        n = int(n) // 1000
                        n = f"{n}K"
                        name = "TimeSynth"
                    else:
                        d = k = n = None  # Use None for non-matching dataset names
                    results.append([name, n, k, d, lan_time_ci, wan_time_ci, comm_size])

    columns = ['Dataset', 'n', 'k', 'd', 'LAN (ms)', 'WAN (ms)', 'Communication Size (bytes)']
    return pd.DataFrame(results, columns=columns)


if __name__ == "__main__":
    argc = len(sys.argv)
    if argc > 1:
        proto = sys.argv[1]
    else:
        proto = "centroid"
    for n_clients in [2, 4, 8]:
        extraction_dir = f'timing_clients/timing_{n_clients}'
        final_results_df = process_all_datasets(extraction_dir, proto)
        final_results_df = final_results_df.sort_values(by=["n", "k", "d"])
        final_results_df.to_csv(f"{extraction_dir}/table.csv", index=False)
        with open(f"timing_clients/table_{n_clients}.tex", 'w') as tex_file:
            final_results_df.to_latex(tex_file, index=False, escape=False)

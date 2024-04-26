# Differentially Private Federated $k$-means Clustering

To reproduce the experiments in the paper, follow the instructions below.
```bash
conda env create -f env.yml
conda activate privatekm
```
To run ablation experiments, use the following commands:
```bash
for param in "mincluster" "maxcluster" "post"; do
    python experiments.py --exp_type "${param}_ablation" --dp_release centroid --results_folder ablation
done
```
To produce ablation plots, use the following command:
```bash
python plots/ablation_plots.py centroid ablation
````
To run accuracy experiments, use the following commands:
```bash
  for dp_release in "centroid" "sumcount"; do
    python experiments.py --exp_type accuracy --dp_release $dp_release --results_folder utility
  done
```
For (optional) max_dist constraint, enable the `--max_dist_mode hard` flag.
Note: this is only tested on 2D datasets.
```bash
for dp_release in "centroid" "sumcount"; do
    python experiments.py --exp_type accuracy --dp_release $dp_release --max_dist_mode hard --results_folder utility
done
```
To produce accuracy plots (after all experiments are done running), use the following command:
```bash
python plots/results_plots.py utility
```
To run scale experiments, use the following commands:
```bash
for sep in 21 70; do
  for dp_release in "centroid" "sumcount"; do
    python experiments.py --exp_type scale${sep} --dp_release $dp_release --results_folder utility/scale
    done
done
```
To produce the scale plot, use the following command:
```bash
for sep in 21 70; do
    python plots/scale_heatmap.py utility/scale/scale${sep}
done
```

To run timing experiments, use the following commands:
```bash
for clients in 2 4 8; do
    mpirun -np $(($clients + 1)) python experiments.py --exp_type timing --dp_release centroid --results_folder timing_clients
done
```
To produce timing tables, use the following command:
```bash
python plots/timing_analysis.py centroid
```

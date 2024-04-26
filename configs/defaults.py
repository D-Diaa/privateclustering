# Datasets
ablate_dataset = [f"AblateSynth_{k}_{d}_{sep}"
                  for k in [2, 4, 8, 16] for d in [2, 4, 8, 16] for sep in [0.25, 0.5, 0.75]]
timing_datasets = ["s1", "lsun"] + [f"timesynth_{k}_{d}_{n}" for k in [2, 5] for d in [2, 5] for n in [10000, 100000]]
accuracy_datasets = ["iris", "s1", "house", "adult", "lsun", "birch2"]
scale21_datasets = [f"synth21_{k}_{d}_{t}" for k in range(2, 21, 2) for d in range(2, 21, 2) for t in [1, 2, 3]]
scale70_datasets = [f"synth70_{k}_{d}_{t}" for k in range(2, 21, 2) for d in range(2, 21, 2) for t in [1, 2, 3]]
# Timing parameters
timing_parameters = {
    "dps": ["laplace"],
    "assignments": ["constrained"],
    "eps_budgets": [1],
    "delays": [0.000125, 0.025],
    "datasets": timing_datasets,
}
# Accuracy parameters
acc_parameters = {
    "dps": ["none", "laplace"],
    "assignments": ["unconstrained", "constrained"],
    "eps_budgets": [0.1, 0.25, 0.5, 0.75, 1, 2.0],
    "delays": [0],
    "datasets": accuracy_datasets,
}
# Scale parameters
scale21_parameters = {
    "dps": ["none", "laplace"],
    "assignments": ["unconstrained", "constrained"],
    "eps_budgets": [1],
    "delays": [0],
    "datasets": scale21_datasets,
}

scale70_parameters = scale21_parameters.copy()
scale70_parameters["datasets"] = scale70_datasets

# Parameters for the parameter tuning
mincluster_ablation_parameters = {
    "dps": ["none", "laplace"],
    "assignments": ["unconstrained", "constrained"],
    "eps_budgets": [0.1, 1],
    "init_methods": ["optimal"],
    "sum_posts": ["none"],
    "count_posts": ["none"],
    "cent_posts": ["fold"],
    "delays": [0],
    "datasets": ablate_dataset,
    "cluster_min_alphas": [1, 1.25, 1.5, 1.75, 2, 2.5, 5],
    "cluster_max_alphas": [5],
}

maxcluster_ablation_parameters = {
    "dps": ["none", "laplace"],
    "assignments": ["unconstrained", "constrained"],
    "eps_budgets": [0.1, 1],
    "init_methods": ["optimal"],
    "sum_posts": ["none"],
    "count_posts": ["none"],
    "cent_posts": ["fold"],
    "delays": [0],
    "datasets": ablate_dataset,
    "cluster_min_alphas": [5.0],
    "cluster_max_alphas": [1, 1.25, 1.5, 1.75, 2, 2.5, 5],

}

post_ablation_parameters = {
    "dps": ["none", "laplace"],
    "assignments": ["unconstrained", "constrained"],
    "eps_budgets": [0.1, 1],
    "init_methods": ["optimal"],
    "sum_posts": ["none"],
    "count_posts": ["none"],
    "cent_posts": ["none", "fold", "truncate"],
    "delays": [0],
    "datasets": ablate_dataset,
    "cluster_min_alphas": [5.0],
    "cluster_max_alphas": [5.0]

}

num_clusters = {
    "iris": 3,
    "s1": 15,
    "birch2": 100,
    "house": 3,
    "adult": 3,
    "lsun": 3
}

parameters = {
    "timing": timing_parameters,
    "accuracy": acc_parameters,
    "mincluster_ablation": mincluster_ablation_parameters,
    "maxcluster_ablation": maxcluster_ablation_parameters,
    "post_ablation": post_ablation_parameters,
    "scale21": scale21_parameters,
    "scale70": scale70_parameters,
}

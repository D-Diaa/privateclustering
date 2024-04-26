import numpy as np

from utils.utils import distance_matrix


def evaluate(centroids, values):
    """
    Evaluate the clustering quality based on the centroids and the values.
    :param centroids: centroids of the clusters
    :param values: values to be clustered
    :return: dictionary with the evaluation metrics
    """
    distances = distance_matrix(values, centroids)
    associations = get_cluster_associations(distances)
    non_empty_clusters = np.unique(associations).size

    if non_empty_clusters == 0:
        raise ValueError("No non-empty clusters detected.")
    empty_clusters = centroids.shape[0] - non_empty_clusters
    return {
        "Normalized Intra-cluster Variance (NICV)": evaluate_NICV(associations, centroids, values),
        "Between-Cluster Sum of Squares (BCSS)": evaluate_BCSS(associations, centroids, values),
        "Empty Clusters": empty_clusters
    }


def get_cluster_associations(distances):
    associations = np.argmin(distances, axis=1)
    return associations


def evaluate_NICV(associations, centroids, values):
    return evaluate_WCSS(associations, centroids, values) / values.shape[0]


def evaluate_WCSS(associations, centroids, values):
    return sum([np.sum((values[associations == cluster] - centroids[cluster]) ** 2) for cluster in
                range(centroids.shape[0]) if np.sum(associations == cluster) > 0])


def evaluate_BCSS(associations, centroids, values):
    overall_centroid = np.mean(values, axis=0)
    return sum(
        [(np.linalg.norm(centroids[cluster] - overall_centroid) ** 2) * np.sum(associations == cluster) for cluster in
         range(centroids.shape[0])])

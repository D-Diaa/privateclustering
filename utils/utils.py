import scipy
from matplotlib import pyplot as plt

from data_io.fixed import *


def set_seed(seed):
    np.random.seed(seed)


"""
Mean and confidence interval computation
Computes the mean and confidence interval of a set of values
"""


def mean_confidence_interval(vals, confidence=0.95):
    a = 1.0 * np.array(vals)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def plot_clusters(centroids, values):
    plt.clf()
    distances = distance_matrix(values, centroids)
    associations = np.argmin(distances, axis=1)
    colors = ['r', 'g', 'b', 'y', 'darkgray', 'cyan', 'pink', 'orange', 'purple', 'olive', 'gray', 'brown', 'teal',
              'yellowgreen', 'lightcoral', 'lightpink', 'peru', 'tomato', 'gold', 'magenta']
    k = centroids.shape[0]
    for cluster in range(k):
        vals = values[associations == cluster]
        plt.scatter(vals[:, 0], vals[:, 1], color=colors[cluster % len(colors)])
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black')


"""
Distance matrix computation
Computes squared Euclidean distance between two sets of points
Adapted from:     https://jaykmody.com/blog/distance-matrices-with-numpy/
"""


def distance_matrix(X, Y, fixed=False):
    if fixed:
        XX = unscale(X)
        YY = unscale(Y)
    else:
        XX = X
        YY = Y

    x2 = np.sum(XX ** 2, axis=1)  # shape of (m)
    y2 = np.sum(YY ** 2, axis=1)  # shape of (n)

    # we can compute all x_i * y_j and store it in a matrix at xy[i][j] by
    # taking the matrix multiplication between X and X_train transpose
    # if you're stuggling to understand this, draw out the matrices and
    # do the matrix multiplication by hand
    # (m, d) x (d, n) -> (m, n)
    xy = np.matmul(XX, YY.T)

    # each row in xy needs to be added with x2[i]
    # each column of xy needs to be added with y2[j]
    # to get everything to play well, we'll need to reshape
    # x2 from (m) -> (m, 1), numpy will handle the rest of the broadcasting for us
    # see: https://numpy.org/doc/stable/user/basics.broadcasting.html
    x2 = x2.reshape(-1, 1)
    dists = x2 - 2 * xy + y2  # (m, 1) repeat columnwise + (m, n) + (n) repeat rowwise -> (m, n)
    if fixed:
        dists = to_fixed(dists)
    return dists


"""
Cluster counts computation
Points are allowed to be associated with multiple clusters
Counts the number of points associated with each cluster
"""


def get_counts_multiassoc(associations, k):
    counts = np.zeros(k)
    for cluster in range(k):
        indices = (associations == cluster).any(axis=-1)
        counts[cluster] = sum(indices)
    return counts


"""
Centroid update
Computes the new centroids based on the values and associations
Points are allowed to be associated with multiple clusters
"""


def update_centroids_multiassoc(values, associations, old_centroids, counts=None):
    new_centroids = []
    k = old_centroids.shape[0]
    new_counts = np.zeros(k)
    for cluster in range(k):
        indices = (associations == cluster).any(axis=-1)
        new_counts[cluster] = sum(indices)
        if counts is not None:
            count = counts[cluster]
        else:
            count = new_counts[cluster]
        if count > 0:
            total = values[indices].sum(axis=0)
            centroid = total / count
        else:
            centroid = old_centroids[cluster]
        new_centroids.append(centroid)
    new_centroids = np.array(new_centroids)
    return new_centroids, new_counts

import matplotlib.pyplot as plt
import numpy as np
import pylab


class KMeans:
    """ 
    Step 1: Initialize cluster centers randomly 
    Step 2: repeat until converged: {
        - update cluster labels : assign points to the nearest cluster centers 
        - update cluster centers: set center to the mean of the cluster 
        }
    """

    def __init__(self, n_clusters=3, max_iters=100, plot_steps=True):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.n_clusters)]
        # the centers (mean vector) of each cluster
        self.centroids = []

    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize centroids
        random_centroid_idx = np.random.choice(
            self.n_samples, self.n_clusters, replace=False)
        self.centroids = [self.X[idx] for idx in random_centroid_idx]

        for _ in range(self.max_iters):
            centroids_old = self.centroids
            # update clusters
            self._create_clusters(self.centroids)
            # update centroids
            self.centroids = self._get_centroids(self.clusters)
            # plot if asked
            if self.plot_steps:
                self._plot_steps()
            # check if converged
            if self._is_converged(centroids_old, self.centroids):
                # print("Convergence achieved")
                break

    def _create_clusters(self, centroids):
        for idx, sample in enumerate(self.X):
            centroid_idx = self._get_closest_centroid(sample)
            self.clusters[centroid_idx].append(idx)

    def _get_closest_centroid(self, sample):
        distances = [self._euclidian_distance(
            sample, centroid) for centroid in self.centroids]
        closest_centroid = np.argmin(distances)
        return closest_centroid

    def _get_centroids(self, clusters):
        centroids = [np.mean(self.X[cluster], axis=0) for cluster in clusters]
        return centroids

    def _is_converged(self, centroids_old, centroids_new):
        distances = [self._euclidian_distance(
            old, new) for old, new in zip(centroids_old, centroids_new)]
        return sum(distances) == 0.0

    def _euclidian_distance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2)**2))

    def _plot_steps(self):
        NUM_COLORS = len(self.centroids)
        cm = pylab.get_cmap('gist_rainbow')
        fig, ax = plt.subplots(figsize=(12, 8))

        for centroid_idx, centroid in enumerate(self.centroids):
            color = cm(centroid_idx/NUM_COLORS)
            cluster = self.clusters[centroid_idx]
            point = self.X[cluster].T
            ax.scatter(*point, color=color)
            ax.scatter(*centroid, marker="+", color=color, linewidth=5)

        plt.show()

    def predict(self, X):
        return [self._get_closest_centroid(x) for x in X]
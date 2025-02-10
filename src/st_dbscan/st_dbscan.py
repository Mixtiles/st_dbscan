# -*- coding: utf-8 -*-
"""
ST-DBSCAN - fast scalable implementation of ST DBSCAN
            scales also to memory by splitting into frames
            and merging the clusters together
"""

# Author: Eren Cakmak <eren.cakmak@uni-konstanz.de>
#         Manuel Plank <manuel.plank@uni-konstanz.de>
#
# License: MIT

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances
from sklearn.utils import check_array
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
import warnings

from tqdm import tqdm


class ST_DBSCAN:
    """
    A class to perform the ST_DBSCAN clustering
    Parameters
    ----------
    spatial_eps : float, default=0.5
        The spatial density threshold (maximum spatial distance) between
        two points to be considered related.
    temporal_eps : float, default=10
        The temporal threshold (maximum temporal distance) between two
        points to be considered related.
    min_samples : int, default=5
        The number of samples required for a core point.
    metric : string default='euclidean'
        The used distance metric - more options are
        ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
        ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’,
        ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘rogerstanimoto’, ‘sqeuclidean’,
        ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘yule’.
    n_jobs : int or None, default=-1
        The number of processes to start -1 means use all processors
    Attributes
    ----------
    labels : array, shape = [n_samples]
        Cluster labels for the data - noise is defined as -1
    References
    ----------
    Ester, M., H. P. Kriegel, J. Sander, and X. Xu, "A Density-Based
    Algorithm for Discovering Clusters in Large Spatial Databases with Noise".
    In: Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996

    Birant, Derya, and Alp Kut. "ST-DBSCAN: An algorithm for clustering spatial–temporal data." Data & Knowledge Engineering 60.1 (2007): 208-221.

    Peca, I., Fuchs, G., Vrotsou, K., Andrienko, N. V., & Andrienko, G. L. (2012). Scalable Cluster Analysis of Spatial Events. In EuroVA@ EuroVis.
    """

    def __init__(
        self,
        spatial_eps=0.5,
        temporal_eps=10,
        min_samples=5,
        metric="euclidean",
        spatial_metric=None,
        temporal_metric=None,
        n_jobs=-1,
    ):
        self.spatial_eps = spatial_eps
        self.temporal_eps = temporal_eps
        self.min_samples = min_samples
        self.spatial_metric = spatial_metric or metric
        self.temporal_metric = temporal_metric or metric
        self.n_jobs = n_jobs

    def fit(self, X):
        """
        Apply the ST DBSCAN algorithm
        ----------
        X : 2D numpy array with
            The first element of the array should be the time
            attribute as float. The following positions in the array are
            treated as spatial coordinates. The structure should look like this [[time_step1, x, y], [time_step2, x, y]..]
            For example 2D dataset:
            array([[0,0.45,0.43],
            [0,0.54,0.34],...])
        Returns
        -------
        self
        """
        # check if input is correct
        X = check_array(X)

        if (
            not self.spatial_eps > 0.0
            or not self.temporal_eps > 0.0
            or not self.min_samples > 0.0
        ):
            raise ValueError("spatial_eps, temporal_eps, min_samples must be positive")

        n, m = X.shape

        if len(X) < 20000:
            # compute with quadratic memory consumption

            # Compute sqaured form Euclidean Distance Matrix for 'time' attribute and the spatial attributes
            time_dist = pdist(X[:, 0].reshape(n, 1), metric=self.temporal_metric)
            if self.spatial_metric == "haversine":
                spatial_dist = haversine_distances(np.radians(X[:, 1:]))
            else:
                spatial_dist = pdist(X[:, 1:], metric=self.spatial_metric)
                spatial_dist = squareform(spatial_dist)

            # filter the spatial_dist matrix using the time_dist
            time_dist = squareform(time_dist)
            dist = np.where(
                time_dist <= self.temporal_eps, spatial_dist, 2 * self.spatial_eps
            )

            db = DBSCAN(
                eps=self.spatial_eps,
                min_samples=self.min_samples,
                metric="precomputed",
                n_jobs=self.n_jobs,
            )
            db.fit(dist)

            self.labels = db.labels_

        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # compute with sparse matrices
                # Compute sparse matrix für Euclidean distance
                nn_spatial = NearestNeighbors(
                    metric=self.spatial_metric,
                    radius=self.spatial_eps,
                    n_jobs=self.n_jobs,
                )
                nn_spatial.fit(X[:, 1:])
                euc_sp = nn_spatial.radius_neighbors_graph(X[:, 1:], mode="distance")

                # Compute sparse matrix für temporal distance
                nn_time = NearestNeighbors(
                    metric=self.temporal_metric,
                    radius=self.temporal_eps,
                    n_jobs=self.n_jobs,
                )
                nn_time.fit(X[:, 0].reshape(n, 1))
                time_sp = nn_time.radius_neighbors_graph(
                    X[:, 0].reshape(n, 1), mode="distance"
                )

                # combine both sparse matrixes and filter by time distance matrix
                row = time_sp.nonzero()[0]
                column = time_sp.nonzero()[1]
                v = np.array(euc_sp[row, column])[0]

                # create sparse distance matrix
                dist_sp = coo_matrix((v, (row, column)), shape=(n, n))
                dist_sp = dist_sp.tocsc()
                dist_sp.eliminate_zeros()

                db = DBSCAN(
                    eps=self.spatial_eps,
                    min_samples=self.min_samples,
                    metric="precomputed",
                    n_jobs=self.n_jobs,
                )
                db.fit(dist_sp)

                self.labels = db.labels_

        return self

    def fit_frame_split(
        self,
        X,
        frame_size,
        frame_overlap=None,
        *,
        progress: bool = False,
    ):
        """
        Apply the ST DBSCAN algorithm with splitting it into frames.
        ----------
        X : 2D numpy array with
            The first element of the array should be the time (sorted by time)
            attribute as float. The following positions in the array are
            treated as spatial coordinates. The structure should look like this [[time_step1, x, y], [time_step2, x, y]..]
            For example 2D dataset:
            array([[0,0.45,0.43],
            [0,0.54,0.34],...])
        frame_size : float, default= None
            If not none the dataset is split into frames and merged aferwards
        frame_overlap : float, default=temporal_eps
            If frame_size is set - there will be an overlap between the frames
            to merge the clusters afterwards
        Returns
        -------
        self
        """
        # check if input is correct
        X = check_array(X)

        # default values for overlap
        frame_overlap = frame_overlap or self.temporal_eps

        if (
            not self.spatial_eps > 0.0
            or not self.temporal_eps > 0.0
            or not self.min_samples > 0.0
        ):
            raise ValueError("eps1, eps2, minPts must be positive")

        if (
            not frame_size > 0.0
            or not frame_overlap > 0.0
            or frame_size < frame_overlap
        ):
            raise ValueError("frame_size, frame_overlap not correctly configured.")

        labels = None
        deficit = frame_size - len(X) % frame_size
        X = np.pad(X, ((0, deficit), (0, 0)), mode="constant", constant_values=np.nan)
        frames = sliding_window_view(X, frame_size, 0)[::frame_overlap].swapaxes(1, 2)
        for i, frame in enumerate(tqdm(frames, disable=not progress), 1):
            if i == len(frames):
                frame = frame[:-deficit]

            self.fit(frame)
            if not isinstance(labels, np.ndarray):
                labels = self.labels
            else:
                right_overlap = min(frame_overlap, len(frame))
                frame_one_overlap_labels = labels[-right_overlap:]
                frame_two_overlap_labels = self.labels[0:right_overlap]

                mapper = {}
                for i in list(zip(frame_one_overlap_labels, frame_two_overlap_labels)):
                    mapper[i[1]] = i[0]
                mapper[-1] = -1  # avoiding outliers being mapped to cluster

                # clusters without overlapping points are given new cluster
                ignore_clusters = set(self.labels) - set(frame_two_overlap_labels)
                # recode them to new cluster value
                if -1 in labels:
                    labels_counter = len(set(labels)) - 1
                else:
                    labels_counter = len(set(labels))
                for j in ignore_clusters:
                    mapper[j] = labels_counter
                    labels_counter += 1

                # objects in the second frame are relabeled to match the cluster id from the first frame
                # objects in clusters with no overlap are assigned to new clusters
                new_labels = np.array([mapper[j] for j in self.labels])

                # delete the right overlap
                labels = labels[0 : len(labels) - right_overlap]
                # change the labels of the new clustering and concat
                labels = np.concatenate((labels, new_labels))
                if len(labels) >= len(X):
                    break

        self.labels = labels[: len(X)]
        return self

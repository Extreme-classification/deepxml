import logging
import numpy as np
import multiprocessing as mp
import libs.ANN as ANN
import _pickle as pickle
from .dist_utils import Partitioner
import operator
from scipy.sparse import csr_matrix, diags
from .lookup import Table, PartitionedTable
from xclib.utils.sparse import topk
import os
import numba
from operator import itemgetter
from libs.clustering import Cluster


class Shortlist(object):
    """Get nearest neighbors using brute or HNSW algorithm
    Parameters
    ----------
    method: str
        brute or hnsw
    num_neighbours: int
        number of neighbors
    M: int
        HNSW M (Usually 100)
    efC: int
        construction parameter (Usually 300)
    efS: int
        search parameter (Usually 300)
    num_threads: int, optional, default=-1
        use multiple threads to cluster
    """

    def __init__(self, method, num_neighbours, M, efC, efS, num_threads=12):
        self.method = method
        self.num_neighbours = num_neighbours
        self.M = M
        self.efC = efC
        self.efS = efS
        self.num_threads = num_threads
        self.index = None
        self._construct()

    def _construct(self):
        if self.method == 'brute':
            self.index = ANN.NearestNeighbor(
                num_neighbours=self.num_neighbours,
                method='brute',
                num_threads=self.num_threads
            )
        elif self.method == 'hnsw':
            self.index = ANN.HNSW(
                M=self.M,
                efC=self.efC,
                efS=self.efS,
                num_neighbours=self.num_neighbours,
                num_threads=self.num_threads
            )
        else:
            print("Unknown NN method!")

    def fit(self, data):
        self.index.fit(data)

    def query(self, data, *args, **kwargs):
        indices, distances = self.index.predict(data, *args, **kwargs)
        return indices, distances

    def save(self, fname):
        self.index.save(fname)

    def load(self, fname):
        self.index.load(fname)

    def reset(self):
        # TODO Do we need to delete it!
        del self.index
        self._construct()


class ShortlistCentroids(Shortlist):
    def __init__(self, method, num_neighbours, M, efC, efS,
                 num_threads=12, space='cosine', verbose=False,
                 num_clusters=300):
        super().__init__(method, num_neighbours, M, efC, efS, num_threads)
        self.num_clusters = num_clusters
        self.space = space

    def _adjust_for_multiple_centroids(self, features, labels, label_centroids, multi_centroid_indices):
        embedding_dims = features.shape[1]
        _cluster_obj = Cluster(
            indices=multi_centroid_indices, embedding_dims=embedding_dims,
            num_clusters=self.num_clusters, max_iter=50, n_init=2, num_threads=-1)
        _cluster_obj.fit(features, labels)
        label_centroids = np.vstack(
                [label_centroids, _cluster_obj.predict()])
        return label_centroids

    def _compute_centroid(self, features, labels):
        label_centroids = labels.transpose().dot(features)
        freq = np.ravel(np.sum(labels, axis=0)).reshape(-1, 1)
        return label_centroids/freq

    def fit(self, features, labels, multi_centroid_indices=None):
        label_centroids = self._compute_centroid(features, labels)
        if multi_centroid_indices is not None:
            label_centroids = self._adjust_for_multiple_centroids(
                features, labels, label_centroids, multi_centroid_indices)
        super().fit(label_centroids)

    def query(self, data, *args, **kwargs):
        indices, distances = super().query(data, *args, **kwargs)
        return indices, 1-distances


class ParallelShortlist(object):
    """Multiple graphs; Supports parallel training
        Assumes that all parameters are same for each graph
    Parameters
    ----------
    method: str
        brute or hnsw
    num_neighbours: int
        number of neighbors
    M: int
        HNSW M (Usually 100)
    efC: int
        construction parameter (Usually 300)
    efS: int
        search parameter (Usually 300)
    num_threads: int, optional, default=-1
        use multiple threads to cluster
    num_graphs: int, optional, default=2
        #graphs to maintain
    """

    def __init__(self, method, num_neighbours, M, efC, efS,
                 num_threads=-1, num_graphs=2):
        self.num_graphs = num_graphs
        self.index = []
        for _ in range(num_graphs):
            self.index.append(
                Shortlist(method, num_neighbours, M, efC, efS, num_threads))

    def train(self, data):
        # Sequential for now; Shit happends in parallel
        for idx in range(self.num_graphs):
            self.index[idx].train(data[idx])

    def _query(self, idx, data):
        return self.index[idx].query(data)

    def query(self, data, idx=-1):
        # Sequential for now
        # Parallelize with return values?
        # Data is same for everyone
        if idx != -1:  # Query from particular graph only
            indices, distances = self._query(idx, data)
        else:
            indices, distances = [], []
            for idx in range(self.num_graphs):
                _indices, _distances = self._query(idx, data)
                indices.append(_indices)
                distances.append(_distances)
        return indices, distances

    def save(self, fname):
        pickle.dump({'num_graphs': self.num_graphs},
                    open(fname+".metadata", "wb"))
        for idx in range(self.num_graphs):
            self.index[idx].save(fname+".{}".format(idx))

    def load(self, fname):
        self.num_graphs = pickle.load(
            open(fname+".metadata", "rb"))['num_graphs']
        for idx in range(self.num_graphs):
            self.index[idx].load(fname+".{}".format(idx))

    def reset(self):
        for idx in range(self.num_graphs):
            self.index[idx].reset()

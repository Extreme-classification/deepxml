import logging
import numpy as np
import multiprocessing as mp
import xclib.utils.ann as ANN
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

    def __init__(self, method, num_neighbours, M, efC, efS, num_threads=24):
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
        return indices, 1-distances

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
                 num_threads=24, space='cosine', verbose=False,
                 num_clusters=1):
        super().__init__(method, num_neighbours, M, efC, efS, num_threads)
        self.num_clusters = num_clusters
        self.space = space
        self.padding_index = -1
        self.mapping = None
        self.ext_head = None

    def _cluster_multiple_rep(self, features, labels, label_centroids,
                              multi_centroid_indices):
        embedding_dims = features.shape[1]
        _cluster_obj = Cluster(
            indices=multi_centroid_indices,
            embedding_dims=embedding_dims,
            num_clusters=self.num_clusters,
            max_iter=50, n_init=2, num_threads=-1)
        _cluster_obj.fit(features, labels)
        label_centroids = np.vstack(
            [label_centroids, _cluster_obj.predict()])
        return label_centroids

    def _compute_centroid(self, features, labels):
        label_centroids = labels.transpose().dot(features)
        freq = np.ravel(np.sum(labels, axis=0)).reshape(-1, 1)
        return label_centroids/freq

    def process_multiple_rep(self, features, labels, label_centroids,
                             threshold=7500):
        freq = np.array(labels.sum(axis=0)).ravel()
        if np.max(freq) > threshold and self.num_clusters > 1:
            self.ext_head = np.where(freq >= threshold)[0]
            print("Found {} super-head labels".format(len(self.ext_head)))
            self.mapping = np.arange(label_centroids.shape[0])
            for idx in self.ext_head:
                self.mapping = np.append(
                    self.mapping, [idx]*self.num_clusters)
            return self._cluster_multiple_rep(
                features, labels, label_centroids, self.ext_head)
        else:
            return label_centroids

    def fit(self, features, labels, *args, **kwargs):
        self.padding_index = labels.shape[1]
        label_centroids = self._compute_centroid(features, labels)
        label_centroids = self.process_multiple_rep(
            features, labels, label_centroids)
        norms = np.sum(np.square(label_centroids), axis=1)
        super().fit(label_centroids)

    def query(self, data, *args, **kwargs):
        indices, sim = super().query(data, *args, **kwargs)
        return self._remap(indices, sim)

    def _remap(self, indices, sims):
        if self.mapping is None:
            return indices, sims
        print("Re-mapping code not optimized")
        mapped_indices = np.full_like(indices, self.padding_index)
        # minimum similarity for padding index
        mapped_sims = np.full_like(sims, -1000.0)
        for idx, (ind, sim) in enumerate(zip(indices, sims)):
            _ind, _sim = self._remap_one(ind, sim)
            mapped_indices[idx, :len(_ind)] = _ind
            mapped_sims[idx, :len(_sim)] = _sim
        return mapped_indices, mapped_sims

    def _remap_one(self, indices, vals,
                   _func=max, _limit=-1000):
        """
            Remap multiple centroids to original labels
        """
        indices = map(lambda x: self.mapping[x], indices)
        _dict = dict({})
        for idx, ind in enumerate(indices):
            _dict[ind] = _func(_dict.get(ind, _limit), vals[idx])
        indices, values = zip(*_dict.items())
        return np.fromiter(indices, dtype=np.int64), \
            np.fromiter(values, dtype=np.float32)

    def load(self, fname):
        temp = pickle.load(open(fname+".metadata", 'rb'))
        self.padding_index = temp['padding_index']
        self.mapping = temp['mapping']
        self.ext_head = temp['ext_head']
        super().load(fname)

    def save(self, fname):
        metadata = {
            'padding_index': self.padding_index,
            'mapping': self.mapping,
            'ext_head': self.ext_head
        }
        pickle.dump(metadata, open(fname+".metadata", 'wb'))
        super().save(fname)


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
            indices, similarities = self._query(idx, data)
        else:
            indices, sims = [], []
            for idx in range(self.num_graphs):
                _indices, _sims = self._query(idx, data)
                indices.append(_indices)
                sims.append(_sims)
        return indices, similarities

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

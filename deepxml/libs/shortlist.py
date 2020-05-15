import logging
import numpy as np
import multiprocessing as mp
import xclib.utils.ann as ANN
from xclib.utils.dense import compute_centroid
import _pickle as pickle
from .dist_utils import Partitioner
import operator
from scipy.sparse import csr_matrix, diags
from .lookup import Table, PartitionedTable
from xclib.utils.sparse import topk, csr_from_arrays
import os
import numba
import math
from operator import itemgetter
from libs.clustering import Cluster
from libs.utils import map_neighbors


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

    @property
    def model_size(self):
        # size on disk; see if there is a better solution
        import tempfile
        with tempfile.NamedTemporaryFile() as tmp:
            self.index.save(tmp.name)
            _size = os.path.getsize(tmp.name)/math.pow(2, 20)
        return _size


class ShortlistCentroids(Shortlist):
    """Get nearest labels using KCentroids
    * centroid(l) = mean_{i=1}^{N}{x_i*y_il}
    * brute or HNSW algorithm for search
    Parameters
    ----------
    method: str, optional, default='hnsw'
        brute or hnsw
    num_neighbours: int
        number of neighbors (same as efS)
        * may be useful if the NN search retrieve less number of labels
        * typically doesn't happen with HNSW etc.
    M: int, optional, default=100
        HNSW M (Usually 100)
    efC: int, optional, default=300
        construction parameter (Usually 300)
    efS: int, optional, default=300
        search parameter (Usually 300)
    num_threads: int, optional, default=18
        use multiple threads to cluster
    space: str, optional, default='cosine'
        metric to use while quering
    verbose: boolean, optional, default=True
        print progress
    num_clusters: int, optional, default=1
        cluster instances => multiple representatives for chosen labels
    threshold: int, optional, default=5000
        cluster instances if a label appear in more than 'threshold'
        training points
    """
    def __init__(self, method='hnsw', num_neighbours=300, M=100, efC=300,
                 efS=300, num_threads=24, space='cosine', verbose=True,
                 num_clusters=1, threshold=5000):
        super().__init__(method, num_neighbours, M, efC, efS, num_threads)
        self.num_clusters = num_clusters
        self.space = space
        self.pad_ind = -1
        self.mapping = None
        self.ext_head = None
        self.threshold = threshold

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

    def process_multiple_rep(self, features, labels, label_centroids):
        freq = np.array(labels.sum(axis=0)).ravel()
        if np.max(freq) > self.threshold and self.num_clusters > 1:
            self.ext_head = np.where(freq >= self.threshold)[0]
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
        self.pad_ind = labels.shape[1]
        label_centroids = compute_centroid(features, labels, reduction='mean')
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
        mapped_indices = np.full_like(indices, self.pad_ind)
        # minimum similarity for padding index
        mapped_sims = np.full_like(sims, -1000.0)
        for idx, (ind, sim) in enumerate(zip(indices, sims)):
            _ind, _sim = self._remap_one(ind, sim)
            mapped_indices[idx, :len(_ind)] = _ind
            mapped_sims[idx, :len(_sim)] = _sim
        return mapped_indices, mapped_sims

    def _remap_one(self, indices, vals, _func=max, _limit=-1000):
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
        self.pad_ind = temp['pad_ind']
        self.mapping = temp['mapping']
        self.ext_head = temp['ext_head']
        super().load(fname+".index")

    def save(self, fname):
        metadata = {
            'pad_ind': self.pad_ind,
            'mapping': self.mapping,
            'ext_head': self.ext_head
        }
        pickle.dump(metadata, open(fname+".metadata", 'wb'))
        super().save(fname+".index")

    def purge(self, fname):
        # purge files from disk
        if os.path.isfile(fname+".index"):
            os.remove(fname+".index")
        if os.path.isfile(fname+".metadata"):
            os.remove(fname+".metadata")


class ShortlistInstances(Shortlist):
    """Get nearest labels using KNN
    * brute or HNSW algorithm for search
    Parameters
    ----------
    method: str, optional, default='hnsw'
        brute or hnsw
    num_neighbours: int
        number of labels to keep per data point
        * labels may be shared across fetched instances
        * union of labels can be large when dataset is densly tagged
    M: int, optional, default=100
        HNSW M (Usually 100)
    efC: int, optional, default=300
        construction parameter (Usually 300)
    efS: int, optional, default=300
        search parameter (Usually 300)
    num_threads: int, optional, default=18
        use multiple threads to cluster
    space: str, optional, default='cosine'
        metric to use while quering
    verbose: boolean, optional, default=True
        print progress
    num_clusters: int, optional, default=1
        cluster instances => multiple representatives for chosen labels
    pad_val: int, optional, default=-10000
        value for padding indices
        - Useful as all documents may have different number of nearest
        labels after collasping them
    """
    def __init__(self, method='hnsw', num_neighbours=300, M=100, efC=300,
                 efS=300, num_threads=24, space='cosine', verbose=False,
                 pad_val=-10000):
        super().__init__(method, num_neighbours, M, efC, efS, num_threads)
        self.labels = None
        self.space = space
        self.pad_ind = None
        self.pad_val = pad_val

    def _remove_invalid(self, features, labels):
        # Keep data points with nnz features and atleast one label
        ind_ft = np.where(np.sum(np.square(features), axis=1) > 0)[0]
        ind_lb = np.where(np.sum(labels, axis=1) > 0)[0]
        ind = np.intersect1d(ind_ft, ind_lb)
        return features[ind], labels[ind]

    def _as_array(self, labels):
        n_pos_labels = list(map(len, labels))
        _labels = np.full(
            (len(labels), max(n_pos_labels)),
            self.pad_ind, np.int64)
        for ind, _lab in enumerate(labels):
            _labels[ind, :n_pos_labels[ind]] = labels[ind]
        return _labels

    def _remap(self, indices, distances):
        return map_neighbors(
            indices, 1-distances,
            self.labels, self.num_neighbours,
            self.pad_ind, self.pad_val)

    def fit(self, features, labels):
        features, labels = self._remove_invalid(features, labels)
        self.index.fit(features)
        self.pad_ind = labels.shape[1]
        self.labels = self._as_array(labels.tolil().rows)

    def query(self, data, *args, **kwargs):
        indices, distances = self.index.predict(data)
        indices, similarities = self._remap(indices, distances)
        return indices, similarities

    def save(self, fname):
        self.index.save(fname+".index")
        pickle.dump(
            {'labels': self.labels,
             'M': self.M, 'efC': self.efC,
             'efS': self.efS,
             'pad_ind': self.pad_ind,
             'pad_val': self.pad_val,
             'num_neighbours': self.num_neighbours,
             'space': self.space}, open(fname+".metadata", 'wb'))

    def load(self, fname):
        self.index.load(fname+".index")
        obj = pickle.load(
            open(fname+".metadata", 'rb'))
        self.num_neighbours = obj['num_neighbours']
        self.efS = obj['efS']
        self.space = obj['space']
        self.labels = obj['labels']
        self.pad_ind = obj['pad_ind']
        self.pad_val = obj['pad_val']

    def purge(self, fname):
        # purge files from disk
        if os.path.isfile(fname+".index"):
            os.remove(fname+".index")
        if os.path.isfile(fname+".metadata"):
            os.remove(fname+".metadata")


class ShortlistEnsemble(object):
    """Get nearest labels using KNN + Kcentroid
    * Give less weight to KNN (typically 0.1 or 0.075)
    * brute or HNSW algorithm for search
    Parameters
    ----------
    method: str, optional, default='hnsw'
        brute or hnsw
    num_neighbours: int, optional, default=500
        number of labels to keep for each instance
        * will pad using pad_ind and pad_val in case labels
          are less than num_neighbours
    M: int, optional, default=100
        HNSW M (Usually 100)
    efC: dict, optional, default={'kcentroid': 300, 'knn': 50}
        construction parameter for kcentroid and knn
        * Usually 300 for kcentroid and 50 for knn
    efS: dict, optional, default={'kcentroid': 300, 'knn': 500}
        search parameter for kcentroid and knn
        * Usually 300 for kcentroid and 500 for knn
    num_threads: int, optional, default=24
        use multiple threads to cluster
    space: str, optional, default='cosine'
        metric to use while quering
    verbose: boolean, optional, default=True
        print progress
    num_clusters: int, optional, default=1
        cluster instances => multiple representatives for chosen labels
    pad_val: int, optional, default=-10000
        value for padding indices
        - Useful as documents may have different number of nearest labels
    gamma: float, optional, default=0.075
        weight for KNN.
        * final shortlist => gamma * knn + (1-gamma) * kcentroid
    """
    def __init__(self, method='hnsw', num_neighbours={'ens': 500,
                 'kcentroid': 400, 'knn': 300},
                 M={'kcentroid': 100, 'knn': 50},
                 efC={'kcentroid': 300, 'knn': 50},
                 efS={'kcentroid': 400, 'knn': 100},
                 num_threads=24, space='cosine', verbose=True,
                 num_clusters=1, pad_val=-10000, gamma=0.075):
        self.kcentroid = ShortlistCentroids(
            method=method, num_neighbours=efS['kcentroid'],
            M=M['kcentroid'], efC=efC['kcentroid'], efS=efS['kcentroid'],
            num_threads=num_threads, space=space, verbose=True)
        self.knn = ShortlistInstances(
            method=method, num_neighbours=num_neighbours['knn'], M=M['knn'],
            efC=efC['knn'], efS=efS['knn'], num_threads=num_threads,
            space=space, verbose=True)
        self.num_labels = None
        self.num_neighbours = num_neighbours['ens']
        self.pad_val = pad_val
        self.pad_ind = -1
        self.gamma = gamma

    def fit(self, X, Y, *args, **kwargs):
        # Useful when number of neighbors are not same
        self.pad_ind = Y.shape[1]
        self.num_labels = Y.shape[1]
        self.kcentroid.fit(X, Y)
        self.knn.fit(X, Y)

    @property
    def model_size(self):
        return self.knn.model_size + self.kcentroid.model_size

    def merge(self, indices_kcentroid, indices_knn, sim_kcentroid, sim_knn):
        _shape = (len(indices_kcentroid), self.num_labels+1)
        short_knn = csr_from_arrays(
            indices_knn, sim_knn, _shape)
        short_kcentroid = csr_from_arrays(
            indices_kcentroid, sim_kcentroid, _shape)
        indices, sim = topk(
            (self.gamma*short_knn + (1-self.gamma)*short_kcentroid),
            k=self.num_neighbours, pad_ind=self.pad_ind,
            pad_val=self.pad_val, return_values=True)
        return indices, sim

    def query(self, data):
        indices_knn, sim_knn = self.knn.query(data)
        indices_kcentroid, sim_kcentroid = self.kcentroid.query(data)
        indices, similarities = self.merge(
            indices_kcentroid, indices_knn, sim_kcentroid, sim_knn)
        return indices, similarities

    def save(self, fname):
        # Returns the filename on disk; useful in purging checkpoints
        pickle.dump(
            {'num_labels': self.num_labels,
             'pad_ind': self.pad_ind}, open(fname+".metadata", 'wb'))
        self.kcentroid.save(fname+'.kcentroid')
        self.knn.save(fname+'.knn')

    def purge(self, fname):
        # purge files from disk
        self.knn.purge(fname)
        self.kcentroid.purge(fname)

    def load(self, fname):
        obj = pickle.load(
            open(fname+".metadata", 'rb'))
        self.num_labels = obj['num_labels']
        self.pad_ind = obj['pad_ind']
        self.kcentroid.load(fname+'.kcentroid')
        self.knn.load(fname+'.knn')

    def reset(self):
        self.kcentroid.reset()
        self.knn.reset()


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

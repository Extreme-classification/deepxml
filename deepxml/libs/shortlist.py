import pickle
from xclib.utils.sparse import topk, csr_from_arrays
from xclib.utils.shortlist import Shortlist
from xclib.utils.shortlist import ShortlistCentroids
from xclib.utils.shortlist import ShortlistInstances


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
            num_threads=num_threads, space=space, num_clusters=num_clusters,
            verbose=True)
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

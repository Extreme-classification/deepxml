# Approximate nearest neighbors with option to perform full k-nearest eighbour search
# Use CPUs for computations
# TODO: Add functionanlity to use GPUs

import nmslib
from sklearn.neighbors import NearestNeighbors
import pickle


class NearestNeighbor(object):
    def __init__(self, num_neighbours, method='brute', num_threads=-1):
        self.num_neighbours = num_neighbours
        self.index = NearestNeighbors(n_neighbors=num_neighbours, algorithm=method, metric='cosine', n_jobs=num_threads)

    def fit(self, data):
        self.index.fit(data)

    def predict(self, data):
        distances, indices = self.index.kneighbors(X=data, n_neighbors=self.num_neighbours, return_distance=True)
        return indices, distances

    def save(self, fname):
        with open(fname, 'wb') as fp:
            pickle.dump({'num_neighbours': self.num_neighbours, 
                         'index': self.index}, fp
                        )

    def load(self, fname):
        with open(fname, 'rb') as fp:
            temp = pickle.load(fp)
            self.index = temp['index']
            self.num_neighbours = temp['num_neighbours']


class HNSW(object):
    def __init__(self, M, efC, efS, num_neighbours, num_threads):
        self.index = nmslib.init(method='hnsw', space='cosinesimil')     
        self.M = M
        self.num_threads = num_threads
        self.efC = efC
        self.efS = efS
        self.num_neighbours = num_neighbours

    def fit(self, data, print_progress=True):
        self.index.addDataPointBatch(data)
        self.index.createIndex({'M': self.M, 
                                'indexThreadQty': self.num_threads, 
                                'efConstruction': self.efC},
                                print_progress=print_progress
                            )

    def _filter(self, output):
        indices = []
        distances = []
        for item in output:
            indices.append(item[0])
            distances.append(item[1])
        return indices, distances
        
    def predict(self, data):
        self.index.setQueryTimeParams({'efSearch': self.efS})
        output = self.index.knnQueryBatch(data, k=self.num_neighbours, num_threads=self.num_threads)
        indices, distances = self._filter(output)
        return indices, distances

    def save(self, fname):
        nmslib.saveIndex(self.index, fname)

    def load(self, fname):
        nmslib.loadIndex(self.index, fname)
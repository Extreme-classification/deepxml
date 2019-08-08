import logging
import numpy as np
import multiprocessing as mp
import libs.ANN as ANN
import _pickle as pickle
from .dist_utils import Partitioner
import operator
from .lookup import Table, PartitionedTable
import os


class Shortlist(object):
    def __init__(self, method, num_neighbours, M, efC, efS, num_threads=-1):
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
            self.index = ANN.NearestNeighbor(num_neighbours=self.num_neighbours, 
                                         method='brute', 
                                         num_threads=self.num_threads
                                        )
        elif self.method == 'hnsw':
            self.index = ANN.HNSW(M=self.M, 
                              efC=self.efC, 
                              efS=self.efS, 
                              num_neighbours=self.num_neighbours, 
                              num_threads=self.num_threads
                            )
        else:
            print("Unknown NN method!")

    def train(self, data):
        self.index.fit(data)

    def query(self, data, *args, **kwargs):
        indices, distances = self.index.predict(data)
        return indices, distances

    def save(self, fname):
        self.index.save(fname)

    def load(self, fname):
        self.index.load(fname)

    def reset(self):
        #TODO Do we need to delete it!
        del self.index
        self._construct()


class ParallelShortlist(object):
    """
        Multiple graphs; Supports parallel training
        Assumes that all parameters are same for each graph
    """
    def __init__(self, method, num_neighbours, M, efC, efS, num_threads=-1, num_graphs=2):
        self.num_graphs = num_graphs
        self.index = []
        for _ in range(num_graphs):
            self.index.append(Shortlist(method, num_neighbours, M, efC, efS, num_threads))

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
        if idx != -1: # Query from particular graph only
            indices, distances = self._query(idx, data)
        else:
            indices, distances = [], []
            for idx in range(self.num_graphs):
                _indices, _distances = self._query(idx, data)
                indices.append(_indices)
                distances.append(_distances)
        return indices, distances

    def save(self, fname):
        pickle.dump({'num_graphs': self.num_graphs}, open(fname+".metadata", "wb"))
        for idx in range(self.num_graphs):
            self.index[idx].save(fname+".{}".format(idx))

    def load(self, fname):
        self.num_graphs = pickle.load(open(fname+".metadata", "rb"))['num_graphs']
        for idx in range(self.num_graphs):
            self.index[idx].load(fname+".{}".format(idx))

    def reset(self):
        for idx in range(self.num_graphs):
            self.index[idx].reset()


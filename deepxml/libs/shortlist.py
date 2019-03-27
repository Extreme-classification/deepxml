import logging
import numpy as np

import libs.ANN as ANN

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
        
    def query(self, data):
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


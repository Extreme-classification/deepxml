import logging
import numpy as np
import multiprocessing as mp
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

    def train_one(self, data, idx):
        self.index[idx].train(data)

    def train(self, data):
        with mp.Pool(self.num_graphs) as p:
            p.map(self.train_one, data)
        # processes = []
        # for idx in range(0, self.num_graphs):
        #     p = mp.Process(target=self.index[idx].train, args=(data[idx],))
        #     processes.append(p)
        #     p.start()       
        # for process in processes:
        #     process.join()

        
    def query(self, data):
        # Sequential for now
        # Parallelize with return values?
        # Data is same for everyone 
        indices, distances = [], []
        for idx in range(self.num_graphs):
            _indices, _distances = self.index[idx].query(data)
            indices.append(_indices)
            distances.append(_distances)
        return indices, distances

    def save(self, fname):
        for idx in range(self.num_graphs):
            self.index[idx].save(fname+".{}".format(idx))

    def load(self, fname):
        for idx in range(self.num_graphs):
            self.index[idx].load(fname+".{}".format(idx))

    def reset(self):
        for idx in range(self.num_graphs):
            self.index[idx].reset()

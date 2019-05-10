import numpy as np
import _pickle as pickle

class Table(object):
    """
        Maintain a lookup table
        Supports in-memory and memmap file
    """
    def __init__(self, _type='memory', _dtype=np.float32):
        self._type = _type
        self._dtype = _dtype
        self._shape = None
        self.data = None

    def _get_fname(self, fname, mode='data'):
        if mode == 'data':
            return fname + ".dat.npy"
        else:
            return fname + ".metadata"

    def create(self, _data, _fname):
        """
            Create a file
            Will copy data
        """
        self._shape = _data.shape
        if self._type == 'memory':
            self.data = np.array(_data, dtype=self._dtype)
        elif self._type == 'memmap':
            data = np.memmap(self._get_fname(_fname), dtype=self._dtype, mode='w+', shape=self._shape)
            data[:] = np.array(_data, dtype=self._dtype)[:]
            del data  # Save to disk and delete object in write mode
            self.data = np.memmap(self._get_fname(_fname), dtype=self._dtype, shape=self._shape, mode='r')  # Open in read mode
        elif self._type == 'hdf5':
            pass
        elif self._type == 'pytables':
            pass
        else:
            raise NotImplementedError("Unknown type!")

    def query(self, indices):
        return self.data[indices]

    def save(self, _fname):
        obj = {'_type': self._type, '_dtype': self._dtype, '_shape': self._shape}
        pickle.dump(obj, open(self._get_fname(_fname, 'metadata'), 'wb'))
        if self._type == 'memory': # Save numpy array; others are already on disk
            np.save(self._get_fname(_fname), self.data)

    def load(self, _fname):
        obj = pickle.load(open(self._get_fname(_fname, 'metadata'), 'rb'))
        self._type = obj['_type']
        self._dtype = obj['_dtype']
        self._shape = obj['_shape']
        if self._type == 'memory':
            self.data = np.load(self._get_fname(_fname), allow_pickle=True)
        elif self._type == 'memmap':
            self.data = np.memmap(self._get_fname(_fname), mode='r', shape=self._shape, dtype=self._dtype)
        elif self._type == 'hdf5':
            pass
        elif self._type == 'pytables':
            pass
        else:
            raise NotImplementedError("Unknown type!")

    def __del__(self):
        del self.data


class PartitionedTable(object):
    """
        Maintain a lookup table
        Supports in-memory and memmap file
    """
    def __init__(self, num_partitions=1, _type='memory', _dtype=np.float32):
        self.num_partitions = num_partitions
        self.data = []
        for _ in range(self.num_partitions):
            self.data.append(Table(_type, _dtype))

    def create(self, _data, _fname):
        """
            Create a file
            Will copy data
        """
        for idx in range(self.num_partitions):
            self.data[idx].create(_data[idx], _fname + ".{}".format(idx))

    def query(self, indices):
        """
            Query indices will be fine as per each table
            No need to re-map here
        """
        out = []
        for idx in range(self.num_partitions):
            out.append(self.data[idx].query(indices))
        return out

    def save(self, _fname):
        for idx in range(self.num_partitions):
            self.data[idx].save(_fname + ".{}".format(idx))

    def load(self, _fname):
        for idx in range(self.num_partitions):
            self.data[idx].load(_fname + ".{}".format(idx))

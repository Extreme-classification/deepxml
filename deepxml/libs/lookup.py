import numpy as np
import _pickle as pickle
import h5py


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
        self.data_init = False
        self._file = None

    def _get_fname(self, fname, mode='data'):
        if mode == 'data':
            return fname + ".dat.npy"
        else:
            return fname + ".metadata"

    def create(self, _data, _fname, *args, **kwargs):
        """
            Create a file
            Will copy data
        """
        _data = np.array(_data, dtype=self._dtype)
        self._shape = _data.shape
        if self._type == 'memory':
            self.data = _data
        elif self._type == 'memmap':
            self.data = np.memmap(self._get_fname(_fname), dtype=self._dtype, mode='w+', shape=self._shape)
            self._file = self.data
            self.data[:] = _data[:]
            self.data.flush()
            del _data  # Save to disk and delete object in write mode
        elif self._type == 'hdf5':
            self._file = h5py.File(self._get_fname(_fname), 'w+')
            self._file.create_dataset('data', data=_data)
            self.data.flush()
            self.data = self._file.get('data')
            del _data
        elif self._type == 'pytables':
            pass
        else:
            raise NotImplementedError("Unknown type!")
        self.data_init = True

    def query(self, indices):
        return self.data[indices]

    def save(self, _fname):
        obj = {'_type': self._type, '_dtype': self._dtype, '_shape': self._shape}
        pickle.dump(obj, open(self._get_fname(_fname, 'metadata'), 'wb'))
        if self._type == 'memory': # Save numpy array; others are already on disk
            np.save(self._get_fname(_fname), self.data)
        elif self._type == 'hdf5': #Not expected to work when filenames are same
            _file = h5py.File(self._get_fname(_fname), 'w+')
            _file.create_dataset('data', data=self.data)
            _file.close()            
        elif self._type == 'memmap': #Not expected to work when filenames are same
            _file = np.memmap(self._get_fname(_fname), dtype=self._dtype, mode='w+', shape=self._shape)
            _file[:] = self.data[:]
            _file.flush()
        else:
            raise NotImplementedError("Unknown type!")

    def load(self, _fname):
        obj = pickle.load(open(self._get_fname(_fname, 'metadata'), 'rb'))
        self._type = obj['_type']
        self._dtype = obj['_dtype']
        self._shape = obj['_shape']
        if self._type == 'memory':
            self.data = np.load(self._get_fname(_fname), allow_pickle=True)
        elif self._type == 'memmap':
            self.data = np.memmap(self._get_fname(_fname), mode='r+', shape=self._shape, dtype=self._dtype)
        elif self._type == 'hdf5':
            fp = h5py.File(self._get_fname(_fname), 'r+')
            self.data = fp.get('data')
        elif self._type == 'pytables':
            pass
        else:
            raise NotImplementedError("Unknown type!")
        self.data_init = True

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
        self.data_init = False
        for _ in range(self.num_partitions):
            self.data.append(Table(_type, _dtype))

    def _create_one(self, _data, _fname, idx): # Create a specific graph only
        #TODO: Add condition to check for invalid idx
        #TODO: Good way to set data_init condition if individual graphs are set
        self.data[idx].create(_data, _fname + ".{}".format(idx))

    def create(self, _data, _fname, idx=-1):
        """
            Create a file
            Will copy data
        """
        if idx != -1:
            self._create_one(_data, _fname, idx)
        else:
            for idx in range(self.num_partitions):
                self._create_one(_data[idx], _fname, idx)
            self.data_init = True

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
        pickle.dump({'num_partitions': self.num_partitions}, open(_fname+".metadata", "wb"))
        for idx in range(self.num_partitions):
            self.data[idx].save(_fname + ".{}".format(idx))

    def load(self, _fname):
        self.num_partitions = pickle.load(open(_fname+".metadata", "rb"))['num_partitions']
        for idx in range(self.num_partitions):
            self.data[idx].load(_fname + ".{}".format(idx))
        self.data_init = True

    def set_status(self, _status):
        self.data_init = _status
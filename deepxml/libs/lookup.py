import numpy as np
import _pickle as pickle
import h5py


class Table(object):
    """Maintain a lookup table
    Supports in-memory and memmap file

    Arguments
    ----------
    _type: str, optional, default='memory'
        keep data in-memory or on-disk
    _dtype: str, optional, default=np.float32
        datatype of the incoming data
    """

    def __init__(self, _type='memory', _dtype=np.float32):
        self._type = _type
        self._dtype = _dtype
        self._shape = None
        self.data = None
        self._file = None

    def _get_fname(self, fname, mode='data'):
        if mode == 'data':
            return fname + ".dat.npy"
        else:
            return fname + ".metadata"

    def create(self, _data, _fname, *args, **kwargs):
        """
            Create a file
            Will not copy data if data-types are same
        """
        _data = np.asarray(_data, dtype=self._dtype)
        self._shape = _data.shape
        if self._type == 'memory':
            self.data = _data
        elif self._type == 'memmap':
            self.data = np.memmap(self._get_fname(
                _fname), dtype=self._dtype, mode='w+', shape=self._shape)
            self._file = self.data
            self.data[:] = _data[:]
            self.data.flush()
            del _data  # Save to disk and delete object in write mode
        else:
            raise NotImplementedError("Unknown type!")

    def query(self, indices):
        return self.data[indices]

    def save(self, _fname):
        obj = {'_type': self._type,
               '_dtype': self._dtype,
               '_shape': self._shape}
        pickle.dump(obj, open(self._get_fname(_fname, 'metadata'), 'wb'))
        # Save numpy array; others are already on disk
        if self._type == 'memory':
            np.save(self._get_fname(_fname), self.data)
        # Not expected to work when filenames are same
        elif self._type == 'memmap':
            _file = np.memmap(self._get_fname(_fname),
                              dtype=self._dtype, mode='w+', shape=self._shape)
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
            self.data = np.memmap(self._get_fname(_fname),
                                  mode='r+', shape=self._shape,
                                  dtype=self._dtype)
        else:
            raise NotImplementedError("Unknown type!")

    def __del__(self):
        del self.data

    @property
    def data_init(self):
        return True if self.data is not None else False


class PartitionedTable(object):
    """Maintain multiple lookup tables
        Supports in-memory and memmap file
    
    Arguments
    ---------
    num_partitions: int, optional, default=1
        #tables to maintain
    _type: str, optional, default='memory'
        keep data in-memory or on-disk
    _dtype: str, optional, default=np.float32
        datatype of the incoming data
    """

    def __init__(self, num_partitions=1, _type='memory', _dtype=np.float32):
        self.num_partitions = num_partitions
        self.data = []
        for _ in range(self.num_partitions):
            self.data.append(Table(_type, _dtype))

    def _create_one(self, _data, _fname, idx):  # Create a specific graph only
        # TODO: Add condition to check for invalid idx
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
        pickle.dump(
            {'num_partitions': self.num_partitions},
            open(_fname+".metadata", "wb"))
        for idx in range(self.num_partitions):
            self.data[idx].save(_fname + ".{}".format(idx))

    def load(self, _fname):
        self.num_partitions = pickle.load(
            open(_fname+".metadata", "rb"))['num_partitions']
        for idx in range(self.num_partitions):
            self.data[idx].load(_fname + ".{}".format(idx))

    @property
    def data_init(self):
        status = [item.data_init for item in self.data]
        return True if all(status) else False

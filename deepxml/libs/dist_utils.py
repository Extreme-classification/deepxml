import numpy as np
import _pickle as pickle


class Partitioner(object):
    """Utility to distribute an array
        Indices support: contiguous or otherwise (e.g. shortlist)
    * useful in distributed training of classifier

    Arguments:
    -----------
    size: int
        size of data
    num_patitions: int
        Divide data in these many parittions
    padding: boolean, optional, default=False
        Padding index (Not handeled #TODO)
    contiguous: boolean, optional, default=True
        whether data is contiguous or not
        (non-contiguous not supported as of now)
    """

    def __init__(self, size, num_patitions, padding=False, contiguous=True):
        # TODO: Handle padding
        self.num_patitions = num_patitions
        self.size = size
        self.contiguous = contiguous
        self._partitions = self._create_partitions()
        self.mapping_to_original, \
            self.mapping_to_partition = self._create_mapping()
        self.partition_boundaries = self._create_partition_boundaries()

    def get_padding_indices(self):
        return [item.size for item in self._partitions]

    def _create_partition_boundaries(self):
        """Split array at these points
        """
        _last = 0
        partition_boundaries = []
        for item in self._partitions[:-1]:
            partition_boundaries.append(_last+item.size)
            _last = partition_boundaries[-1]
        return partition_boundaries

    def _create_partitions(self):
        """Create partitions
        """
        return np.array_split(np.arange(self.size), self.num_patitions)

    def _create_mapping(self):
        """Mapping to map indices original<->partitioned
        """
        mapping_to_original = []
        mapping_to_partition = []
        for _, _partition in enumerate(self._partitions):
            mapping_to_original.append(
                dict(zip(np.arange(_partition.size), _partition)))
            mapping_to_partition.append(
                dict(zip(_partition, np.arange(_partition.size))))
        return mapping_to_original, mapping_to_partition

    def _map(self, fun, array):
        return np.fromiter(map(fun, array), dtype=array.dtype)

    def map_to_original(self, array, idx=None):
        return self._map(self.mapping_to_original[idx].get, array)

    # def map_to_partition(self, array):
    #     return self.map(array, self.fun_map_to_partition)

    def get_partition_index(self, index):
        """
            In which partition this index falls into?
        """
        _last = 0
        for idx, _current in enumerate(self.partition_boundaries):
            if index >= _last and index < _current:
                return idx
        return self.num_patitions-1

    def split_indices_with_data(self, indices, data):
        """Split given indices and data (Shortlist)
            i.e. get the partition and map them accordingly
        """
        out_ind = [[] for _ in range(self.num_patitions)]
        out_vals = [[] for _ in range(self.num_patitions)]
        for key, val in zip(indices, data):
            part = self.get_partition_index(key)
            ind = self.mapping_to_partition[part][key]
            out_ind[part].append(ind)
            out_vals[part].append(val)
        return out_ind, out_vals

    def split_indices(self, indices):
        """Split given indices (Shortlist)
            i.e. get the partition and map them accordingly
        """
        out_ind = [[] for _ in range(self.num_patitions)]
        for key in indices:
            part = self.get_partition_index(key)
            ind = self.mapping_to_partition[part][key]
            out_ind[part].append(ind)
        return out_ind

    def split(self, array):  # Split as per bondaries
        """Split given array in partitions (For contiguous indices)
        """
        if self.contiguous:
            return np.hsplit(array, self.partition_boundaries)
        else:
            pass

    def merge(self, arrays):
        if self.contiguous:
            return np.hstack(arrays)
        else:
            pass

    def save(self, fname):
        pickle.dump(self.__dict__, open(fname, 'wb'))

    def load(self, fname):
        self.__dict__ = pickle.load(open(fname, 'rb'))

    def get_indices(self, part):
        # Indices for given partition
        return self._partitions[part]

    def __repr__(self):
        return "({}) #Size: {}, #partitions: {}".format(
            self.__class__.__name__, self.size, self.num_patitions)

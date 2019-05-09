import numpy as np

class Partitionar(object):
    """
        Utility to distribute an array
        Supports: sparse or dense indices i.e. contiguous or otherwise (e.g. shortlist)
    """
    def __init__(self, size, num_patitions, padding=False, contiguous=False):
        #TODO: Handle padding
        #TODO: Test non-contiguous
        self.num_patitions = num_patitions
        self.size = size
        self.contiguous = contiguous
        self._partitions = self._create_partitions()
        self.mapping_to_original, self.mapping_to_partition = self._create_mapping()
        self.fun_map_to_original = lambda x: self.mapping_to_original[x]
        self.fun_map_to_partition = lambda x: self.mapping_to_partition[x]
        self.partition_boundaries = self._create_partition_boundaries()

    def _create_partition_boundaries(self):
        _last = 0
        partition_boundaries = []
        for item in self._partitions[:-1]:
            partition_boundaries.append(_last+item.size)
            _last = partition_boundaries[-1]
        return partition_boundaries

    def _create_partitions(self):
        return np.array_split(np.arange(self.size), self.num_patitions)

    def _create_mapping(self):
        mapping_to_original = []
        mapping_to_partition = []
        for _, _partition in enumerate(self._partitions):
            mapping_to_original.append(dict(zip(np.arange(_partition.size), _partition)))
            mapping_to_partition.append(dict(zip(_partition, np.arange(_partition.size))))
        return mapping_to_original, mapping_to_partition

    def map(self, array, fun_map):
        return np.array(map(fun_map, array))

    def map_to_original(self, array):
        return self.map(array, self.fun_map_to_original)

    def map_to_partition(self, array):
        return self.map(array, self.fun_map_to_partition)

    def split(self, array):
        if self.contiguous:
            return np.hsplit(array, self.partition_boundaries)
        else:
            pass

    def merge(self, arrays):
        if self.contiguous:
            return np.hstack(arrays)
        else:
            pass

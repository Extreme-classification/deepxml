import sklearn.preprocessing.normalize as scale
import numpy as np
import _pickle as pickle
from xclib.data import data_utils


class LabelsBase(object):
    """
        Base class for Labels
        Args:
            data_dir: str: data directory
            fname: str: load data from this file
            Y: np.ndarray or csr_matrix: data is already provided
    """
    def __init__(self, data_dir, fname, Y=None):
        self.Y = self.load(data_dir, fname, Y)
        self.num_instances, self.num_labels = self.Y.shape

    def _select_instances(self, indices):
        self.Y = self.Y[indices]

    def _select_labels(self, indices):
        self.Y = self.Y[:, indices]

    def load(self, data_dir, fname, Y):
        if Y is not None:
            return Y
        else:
            if fname.lower().endswith('.pkl'):
                return pickle.load(open(fname, 'rb'))['Y']
            elif fname.lower().endswith('.txt'):
                return data_utils.read_sparse_file(fname, dtype=np.float32)
            else:
                raise NotImplementedError("Unknown file extension")

    def get_invalid(self, axis=0):
        return np.where(self.frequency(axis)==0)[1]

    def get_valid(self, axis=0):
        return np.where(self.frequency(axis)>0)[1]

    def remove_invalid(self, axis=0):
        indices = self.get_valid(axis)
        self.index_select(indices)
        return indices

    def index_select(self, indices, axis=1, fname=None):
        """
            Choose only selected labels or instances
        """
        #TODO: Load and select from file
        if axis == 0:
            self._select_instances(indices)
        elif axis == 1:
            self._select_labels(indices)
        else:
            NotImplementedError("Unknown Axis.")
        self.num_instances, self.num_labels = self.Y.shape

    def frequency(self, axis=0):
        return np.array(self.Y.astype(np.bool).sum(axis=axis)).ravel()

    def __getitem__(self, index):
        return self.Y[index]


class DenseLabels(LabelsBase):
    """
        Class for dense labels
        Assumes it's for XC i.e. labels are stored as csr_matrix
        Args:
            data_dir: str: data directory
            fname: str: load data from this file
            X: np.ndarray: data is already provided
    """
    def __init__(self, data_dir, fname, Y=None, normalize=False):
        super().__init__(data_dir, fname, Y)

    def normalize(self, norm='l2', copy=False):
        self.Y = scale(self.Y, copy=copy, norm=norm)

    def __getitem__(self, index):
        return np.array(super().__getitem__(index).todense(),
                      dtype=np.float32).reshape(self.num_labels)


class SparseLabels(LabelsBase):
    """
        Class for sparse labels
        Args:
            data_dir: str: data directory
            fname: str: load data from this file
            X: csr_matrix: data is already provided
    """
    def __init__(self, data_dir, fname, Y=None, normalize=False):
        super().__init__(data_dir, fname, Y)

    def normalize(self, norm='l2', copy=False):
        self.Y = scale(self.Y, copy=copy, norm=norm)

    def __getitem__(self, index):
        y = self.Y[index].indices
        w = self.Y[index].data
        return y, w


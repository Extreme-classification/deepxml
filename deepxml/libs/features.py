from sklearn.preprocessing import normalize as scale
import numpy as np
import _pickle as pickle
from xclib.data import data_utils
import os


def construct(data_dir, fname, X=None, normalize=False, _type='sparse'):
    if _type == 'sparse':
        return SparseFeatures(data_dir, fname, X, normalize)
    elif _type == 'dense':
        return DenseFeatures(data_dir, fname, X, normalize)
    elif _type == 'sequential':
        return SequentialFeatures(data_dir, fname, X)
    else:
        raise NotImplementedError("Unknown feature type")


class FeaturesBase(object):
    """
        Base class for features
        Args:
            data_dir: str: data directory
            fname: str: load data from this file
            X: np.ndarray or csr_matrix: data is already provided
    """
    def __init__(self, data_dir, fname, X=None):
        self.X = self.load(data_dir, fname, X)
        self.num_instances, self.num_features = self.X.shape

    def frequency(self, axis=0):
        return np.array(self.X.sum(axis=axis)).ravel()

    def get_invalid(self, axis=0):
        return np.where(self.frequency(axis)==0)[0]

    def get_valid(self, axis=0):
        return np.where(self.frequency(axis)>0)[0]

    def remove_invalid(self, axis=0):
        indices = self.get_valid(axis)
        self.index_select(indices)
        return indices

    def _select_instances(self, indices):
        self.X = self.X[indices]

    def _select_features(self, indices):
        # Not valid in general case
        pass

    def index_select(self, indices, axis=1, fname=None):
        """
            Choose only selected labels or instances
        """
        #TODO: Load and select from file
        if axis == 0:
            self._select_instances(indices)
        elif axis == 1:
            self._select_features(indices)
        else:
            raise NotImplementedError("Unknown Axis.")
        self.num_instances, self.num_features = self.X.shape

    def load(self, data_dir, fname, X):
        fname = os.path.join(data_dir, fname)
        if X is not None:
            return X
        else:
            if fname.lower().endswith('.pkl'):
                return pickle.load(open(fname, 'rb'))['X']
            elif fname.lower().endswith('.txt'):
                return data_utils.read_sparse_file(fname, dtype=np.float32, force_header=True)
            else:
                raise NotImplementedError("Unknown file extension")

    def __getitem__(self, index):
        return self.X[index]


class DenseFeatures(FeaturesBase):
    """
        Class for dense features
        Args:
            data_dir: str: data directory
            fname: str: load data from this file
            X: np.ndarray: data is already provided
    """
    def __init__(self, data_dir, fname, X=None, normalize=False):
        super().__init__(data_dir, fname, X)

    def _select_features(self, indices):
        self.X = self.X[:, indices]

    def normalize(self, norm='l2', copy=False):
        self.X = scale(self.X, copy=copy, norm=norm)

    def frequency(self, axis=0):
        return np.array(self.X.astype(np.bool).sum(axis=axis)).ravel()


class SparseFeatures(FeaturesBase):
    """
        Class for sparse features
        Args:
            data_dir: str: data directory
            fname: str: load data from this file
            X: csr_matrix: data is already provided
    """
    def __init__(self, data_dir, fname, X=None, normalize=False):
        super().__init__(data_dir, fname, X)
        if normalize:
            self.normalize()

    def normalize(self, norm='l2', copy=False):
        self.X = scale(self.X, copy=copy, norm=norm)

    def _select_features(self, indices):
        self.X = self.X[:, indices]

    def frequency(self, axis=0):
        return np.array(self.X.astype(np.bool).sum(axis=axis)).ravel()

    def __getitem__(self, index):
        x = list(map(lambda item: item+1, self.X[index, :].indices)) # Treat idx:0 as Padding
        w = self.X[index, :].data.tolist()
        return x, w


class SequentialFeatures(FeaturesBase):
    """
        Class for Sequential features; useful for sequential models
        Args:
            data_dir: str: data directory
            fname: str: load data from this file
            X: ?: ?
    """
    pass
from sklearn.preprocessing import normalize as scale
import numpy as np
import _pickle as pickle
from xclib.data import data_utils
import os
from scipy.sparse import load_npz
from xclib.utils.sparse import binarize


def construct(data_dir, fname, Y=None, normalize=False, _type='sparse'):
    """Construct label class based on given parameters
    Arguments
    ----------
    data_dir: str
        data directory
    fname: str
        load data from this file
    Y: csr_matrix or None, optional, default=None
        data is already provided
    normalize: boolean, optional, default=False
        Normalize the labels or not
        Useful in case of non binary labels
    _type: str, optional, default=sparse
        -sparse or dense
    """
    if fname is None and Y is None:  # No labels are provided
        return LabelsBase(data_dir, fname, Y)
    else:
        if _type == 'sparse':
            return SparseLabels(data_dir, fname, Y, normalize)
        elif _type == 'dense':
            return DenseLabels(data_dir, fname, Y, normalize)
        else:
            raise NotImplementedError("Unknown label type")


class LabelsBase(object):
    """Base class for Labels
    Parameters
    ----------
    data_dir: str
        data directory
    fname: str
        load data from this file
    Y: csr_matrix or np.ndarray or None, optional, default=None
        data is already provided
    """

    def __init__(self, data_dir, fname, Y=None):
        self.Y = self.load(data_dir, fname, Y)

    def _select_instances(self, indices):
        self.Y = self.Y[indices] if self._valid else None

    def _select_labels(self, indices):
        self.Y = self.Y[:, indices] if self._valid else None

    def normalize(self, norm='max', copy=False):
        self.Y = scale(self.Y, copy=copy, norm=norm) if self._valid else None

    def load(self, data_dir, fname, Y):
        if Y is not None:
            return Y
        elif fname is None:
            return None
        else:
            fname = os.path.join(data_dir, fname)
            if fname.lower().endswith('.pkl'):
                return pickle.load(open(fname, 'rb'))['Y']
            elif fname.lower().endswith('.txt'):
                return data_utils.read_sparse_file(
                    fname, dtype=np.float32)
            elif fname.lower().endswith('.npz'):
                return load_npz(fname)
            else:
                raise NotImplementedError("Unknown file extension")

    def get_invalid(self, axis=0):
        return np.where(self.frequency(axis) == 0)[0] if self._valid else None

    def get_valid(self, axis=0):
        return np.where(self.frequency(axis) > 0)[0] if self._valid else None

    def remove_invalid(self, axis=0):
        indices = self.get_valid(axis)
        self.index_select(indices)
        return indices

    def binarize(self):
        if self._valid:
            self.Y = binarize(self.Y, copy=False)

    def index_select(self, indices, axis=1, fname=None):
        """
            Choose only selected labels or instances
        """
        # TODO: Load and select from file
        if axis == 0:
            self._select_instances(indices)
        elif axis == 1:
            self._select_labels(indices)
        else:
            NotImplementedError("Unknown Axis.")

    def frequency(self, axis=0):
        return np.array(self.Y.astype(np.bool).sum(axis=axis)).ravel() \
            if self._valid else None

    def transpose(self):
        return self.Y.transpose() if self._valid else None

    @property
    def _valid(self):
        return self.Y is not None

    @property
    def num_instances(self):
        return self.Y.shape[0] if self._valid else -1

    @property
    def num_labels(self):
        return self.Y.shape[1] if self._valid else -1

    @property
    def shape(self):
        return (self.num_instances, self.num_labels)

    def __getitem__(self, index):
        return self.Y[index] if self._valid else None


class DenseLabels(LabelsBase):
    """Class for dense labels
    Parameters:
    data_dir: str
        data directory
    fname: str
        load data from this file
    Y: np.ndarray or None, optional, default=None
        data is already provided
    normalize: boolean, optional, default=False
        Normalize the labels or not
        Useful in case of non binary labels
    """

    def __init__(self, data_dir, fname, Y=None, normalize=False):
        super().__init__(data_dir, fname, Y)

    def __getitem__(self, index):
        return np.array(
            super().__getitem__(index).todense(),
            dtype=np.float32).reshape(self.num_labels)


class SparseLabels(LabelsBase):
    """Class for sparse labels
    data_dir: str
        data directory
    fname: str
        load data from this file
    Y: csr_matrix or None, optional, default=None
        data is already provided
    normalize: boolean, optional, default=False
        Normalize the labels or not
        Useful in case of non binary labels
    """

    def __init__(self, data_dir, fname, Y=None, normalize=False):
        super().__init__(data_dir, fname, Y)

    def __getitem__(self, index):
        y = self.Y[index].indices.tolist()
        w = self.Y[index].data.tolist()
        return y, w

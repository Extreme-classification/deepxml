import numpy as np
from scipy.sparse import csr_matrix
import warnings
from xclib.utils.sparse import csr_from_arrays, retain_topk


def topk(values, indices=None, k=10, sorted=False):
    """
    Return topk values from a np.ndarray with support for optional
    second array

    Arguments:
    ---------
    values: np.ndarray
        select topk values based on this array
    indices: np.ndarray or None, optional, default=None
        second array; return corresponding entries for this array
        as well; useful for key, value pairs
    k: int, optional, default=10
        k in top-k
    sorted: boolean, optional, default=False
        Sort the topk values or not
    """
    assert values.shape[1] >= k, f"value has less than {k} values per row"
    if indices is not None:
        assert values.shape == indices.shape, \
            f"Shape of values {values.shape} != indices {indices.shape}"
        # Don't do anything if n_cols = k or k = -1
        if k == indices.shape[1] or k == -1:
            return values, indices
    if not sorted:
        ind = np.argpartition(values, -k)[:, -k:]
    else:
        ind = np.argpartition(
            values, list(range(-k, 0)))[:, -k:][:, ::-1]
    val = np.take_along_axis(values, ind, axis=-1)
    if indices is not None:
        out = (val, np.take_along_axis(indices, ind, axis=-1))
    else:
        out = (val, ind)
    return out


class Prediction(object):
    """
    Class to store and manipulate predictions
    * This can be more suitable as:
    - We already know num_instances & top_k
    - space can be allocated in advance
    - updation is faster

    Support for:
    * OVA predictions
    * Predictions with a label shortlist

    Uses num_labels as pad_ind; will remove the pad_ind with as_sparse()
    Predictions may have:
    * (batch_size, num_labels+1) shape for dense predictions
    * num_labels as entry in ind array

    Arguments:
    ----------
    num_instances: int
        lenght of 0th dimension
    k: int
        store k values per instance
    num_labels: int
        lenght of 1st dimension
        pad indices with this value as well
    k: int
        the k in top-k
    pad_val: float, optional, default=-1e5
        default value of predictions
    fname: float or None, optional, default=None
        Use memmap files and store on disk if filename is provides
    """
    def __init__(self, num_instances, num_labels, k, pad_val=-1e5, fname=None):
        self.num_instances = num_instances
        self.k = k
        self.num_labels = num_labels
        self.pad_ind = num_labels
        self.pad_val = pad_val
        self.indices = self._array(
            fname + ".ind" if fname is not None else None,
            fill_value=self.pad_ind,
            dtype='int64')
        self.values = self._array(
            fname + ".val" if fname is not None else None,
            fill_value=self.pad_val,
            dtype='float32')

    def _array(self, fname, fill_value, dtype):
        if fname is None:
            arr = np.full(
                (self.num_instances, self.k),
                fill_value=fill_value, dtype=dtype)
        else:
            arr = np.memmap(
                fname, shape=(self.num_instances, self.k),
                dtype=dtype, mode='w+')
            arr[:] = fill_value
        return arr

    def data(self, format='sparse'):
        """Returns the predictions as a csr_matrix or indices & values arrays
        """
        self.flush()
        if format == 'sparse':
            if not self.in_memory:
                warnings.warn("Files on disk; will create copy in memory.")
            return csr_from_arrays(
                self.indices, self.values,
                shape=(self.num_instances, self.num_labels+1))[:, :-1]
        else:
            return self.indices, self.values

    def update_values(self, start_idx, vals, ind=None):
        """Update the entries as per given indices and values
        """
        top_val, top_ind = self.topk(vals, ind)
        _size = vals.shape[0]
        self.values[start_idx: start_idx+_size, :] = top_val
        self.indices[start_idx: start_idx+_size, :] = top_ind

    def topk(self, vals, ind):
        """Assumes inputs are np.ndarrays/ Implement your own method
        for some other type.
        Output must be np.ndarrays

        * if ind is None: will return corresponding indices of vals
            typically used with OVA predictions
        * otherwise: will use corresponding entries from ind
            typically used with predictions with a label shortlist
        """
        return topk(vals, ind, k=self.k)

    @property
    def in_memory(self):
        return not isinstance(self.indices, np.memmap)

    def flush(self):
        if not self.in_memory:
            self.indices.flush()
            self.values.flush()

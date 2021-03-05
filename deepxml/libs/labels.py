from xclib.data.labels import DenseLabels, SparseLabels, LabelsBase


def construct(data_dir, fname, Y=None, normalize=False, _type='sparse'):
    """Construct label class based on given parameters

    Support for:
    * pkl file: Key 'Y' is used to access the labels
    * txt file: sparse libsvm format with header
    * npz file: numpy's sparse format

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


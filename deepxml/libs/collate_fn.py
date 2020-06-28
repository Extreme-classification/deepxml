import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def pad_and_collate(x, pad_val=0, dtype=torch.FloatTensor):
    """
    A generalized function for padding batch using utils.rnn.pad_sequence
    * pad as per the maximum length in the batch
    * returns a collated tensor

    Arguments:
    ---------
    x: iterator
        iterator over np.ndarray that needs to be converted to
        tensors and padded
    pad_val: float
        pad tensor with this value
        will cast the value as per the data type
    dtype: datatype, optional (default=torch.FloatTensor)
        tensor should be of this type
    """
    return pad_sequence([torch.from_numpy(z) for z in x],
                        batch_first=True, padding_value=pad_val).type(dtype)


def collate_dense(x, dtype=torch.FloatTensor):
    """
    Collate dense documents/labels and returns

    Arguments:
    ---------
    x: iterator
        iterator over np.ndarray that needs to be converted to
        tensors and padded
    dtype: datatype, optional (default=torch.FloatTensor)
        features should be of this type
    """
    return torch.stack([torch.from_numpy(z) for z in x], 0).type(dtype)


def collate_sparse(x, pad_val=0.0, has_weight=False, dtype=torch.FloatTensor):
    """
    Collate sparse documents
    * Can handle with or without weights
    * Expects an iterator over tuples if has_weight=True

    Arguments:
    ---------
    x: iterator
        iterator over data points which can be
        np.array or tuple of np.ndarray depending on has_weight
    pad_val: list or float, optional, default=(0.0)
        padding value for indices and weights
        * expects a list when has_weight=True
    has_weight: bool, optional, default=False
        If entries have weights
        * True: objects are tuples of np.ndarrays
            0: indices, 1: weights
        * False: objects are np.ndarrays
    dtypes: list or dtype, optional (default=torch.FloatTensor)
        dtypes of indices and values
        * expects a list when has_weight=True
    """
    weights = None
    if has_weight:
        x = list(x)
        indices = pad_and_collate(map(lambda z: z[0], x), pad_val[0], dtype[0])
        weights = pad_and_collate(map(lambda z: z[1], x), pad_val[1], dtype[1])
    else:
        indices = pad_and_collate(x, pad_val, dtype)
    return indices, weights


def get_iterator(x, ind=None):
    if ind is None:
        return map(lambda z: z, x)
    else:
        return map(lambda z: z[ind], x)


def construct_collate_fn(feature_type, classifier_type, num_partitions=1):
    def _collate_fn_dense_full(batch):
        return collate_fn_dense_full(batch, num_partitions)

    def _collate_fn_dense(batch):
        return collate_fn_dense(batch)

    def _collate_fn_sparse(batch):
        return collate_fn_sparse(batch)

    def _collate_fn_dense_sl(batch):
        return collate_fn_dense_sl(batch, num_partitions)

    def _collate_fn_sparse_full(batch):
        return collate_fn_sparse_full(batch, num_partitions)

    def _collate_fn_sparse_sl(batch):
        return collate_fn_sparse_sl(batch, num_partitions)

    if feature_type == 'dense':
        if classifier_type == 'None':
            return _collate_fn_dense
        elif classifier_type == 'shortlist':
            return _collate_fn_dense_sl
        else:
            return _collate_fn_dense_full
    else:
        if classifier_type == 'None':
            return _collate_fn_sparse
        elif classifier_type == 'shortlist':
            return _collate_fn_sparse_sl
        else:
            return _collate_fn_sparse_full


def collate_fn_sparse_sl(batch, num_partitions):
    """
        Combine each sample in a batch with shortlist
        For sparse features
    """
    _is_partitioned = True if num_partitions > 1 else False
    batch_data = {'batch_size': len(batch), 'X_ind': None}
    batch_data['batch_size'] = len(batch)
    batch_data['X_ind'], batch_data['X'] = collate_sparse(
        get_iterator(batch, 0), pad_val=[0, 0.0], has_weight=True,
        dtype=[torch.LongTensor, torch.FloatTensor])

    z = list(get_iterator(batch, 1))
    if _is_partitioned:
        batch_data['Y_s'] = [collate_dense(
            get_iterator(get_iterator(z, 0), idx), dtype=torch.LongTensor)
            for idx in range(num_partitions)]
        batch_data['Y'] = [collate_dense(
            get_iterator(get_iterator(z, 1), idx), dtype=torch.FloatTensor)
            for idx in range(num_partitions)]
        batch_data['Y_sim'] = [collate_dense(
            get_iterator(get_iterator(z, 2), idx), dtype=torch.FloatTensor)
            for idx in range(num_partitions)]
        batch_data['Y_mask'] = [collate_dense(
            get_iterator(get_iterator(z, 3), idx), dtype=torch.BoolTensor)
            for idx in range(num_partitions)]
        batch_data['Y_map'] = collate_dense(
            get_iterator(z, 4), dtype=torch.LongTensor)
    else:
        batch_data['Y_s'] = collate_dense(
            get_iterator(z, 0), dtype=torch.LongTensor)
        batch_data['Y'] = collate_dense(
            get_iterator(z, 1), dtype=torch.FloatTensor)
        batch_data['Y_sim'] = collate_dense(
            get_iterator(z, 2), dtype=torch.FloatTensor)
        batch_data['Y_mask'] = collate_dense(
            get_iterator(z, 3), dtype=torch.BoolTensor)
    return batch_data


def collate_fn_dense_sl(batch, num_partitions):
    """
        Combine each sample in a batch with shortlist
        For dense features
    """
    _is_partitioned = True if num_partitions > 1 else False
    batch_data = {'batch_size': len(batch), 'X_ind': None}
    batch_data['X'] = collate_dense(get_iterator(batch, 0))

    z = list(get_iterator(batch, 1))
    if _is_partitioned:
        batch_data['Y_s'] = [collate_dense(
            get_iterator(get_iterator(z, 0), idx), dtype=torch.LongTensor)
            for idx in range(num_partitions)]
        batch_data['Y'] = [collate_dense(
            get_iterator(get_iterator(z, 1), idx), dtype=torch.FloatTensor)
            for idx in range(num_partitions)]
        batch_data['Y_sim'] = [collate_dense(
            get_iterator(get_iterator(z, 2), idx), dtype=torch.FloatTensor)
            for idx in range(num_partitions)]
        batch_data['Y_mask'] = [collate_dense(
            get_iterator(get_iterator(z, 3), idx), dtype=torch.BoolTensor)
            for idx in range(num_partitions)]
        batch_data['Y_map'] = collate_dense(
            get_iterator(z, 4), dtype=torch.LongTensor)
    else:
        batch_data['Y_s'] = collate_dense(
            get_iterator(z, 0), dtype=torch.LongTensor)
        batch_data['Y'] = collate_dense(
            get_iterator(z, 1), dtype=torch.FloatTensor)
        batch_data['Y_sim'] = collate_dense(
            get_iterator(z, 2), dtype=torch.FloatTensor)
        batch_data['Y_mask'] = collate_dense(
            get_iterator(z, 3), dtype=torch.BoolTensor)
    return batch_data


def collate_fn_dense_full(batch, num_partitions):
    """
        Combine each sample in a batch
        For dense features
    """
    _is_partitioned = True if num_partitions > 1 else False
    batch_data = {'batch_size': len(batch), 'X_ind': None}
    batch_data['X'] = collate_dense(get_iterator(batch, 0))
    if _is_partitioned:
        batch_data['Y'] = [collate_dense(
            get_iterator(get_iterator(batch, 1), idx))
                for idx in range(self.num_partitions)]
    else:
        batch_data['Y'] = collate_dense(get_iterator(batch, 1))
    return batch_data


def collate_fn_sparse_full(batch, num_partitions):
    """
        Combine each sample in a batch
        For sparse features
    """
    _is_partitioned = True if num_partitions > 1 else False
    batch_data = {'batch_size': len(batch), 'X_ind': None}
    batch_data['X_ind'], batch_data['X'] = collate_sparse(
        get_iterator(batch, 0), pad_val=[0, 0.0], has_weight=True,
        dtype=[torch.LongTensor, torch.FloatTensor])
    if _is_partitioned:
        batch_data['Y'] = [collate_dense(
            get_iterator(get_iterator(batch, 1), idx))
                for idx in range(self.num_partitions)]
    else:
        batch_data['Y'] = collate_dense(get_iterator(batch, 1))
    return batch_data


def collate_fn_sparse(batch):
    """
        Combine each sample in a batch
        For sparse features
    """
    batch_data = {'batch_size': len(batch), 'X_ind': None}
    batch_data['X_ind'], batch_data['X'] = collate_sparse(
        get_iterator(batch), pad_val=[0, 0.0], has_weight=True,
        dtype=[torch.LongTensor, torch.FloatTensor])
    return batch_data


def collate_fn_dense(batch):
    """
        Combine each sample in a batch
        For sparse features
    """
    batch_data = {'batch_size': len(batch), 'X_ind': None}
    batch_data['X'] = collate_dense(get_iterator(batch))
    return batch_data

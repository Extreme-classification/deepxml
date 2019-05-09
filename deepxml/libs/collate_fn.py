import torch
import numpy as np


def construct_collate_fn(feature_type, use_shortlist=False, use_seq_features=False, num_partitions=1):
    def _collate_fn_dense_full(batch):
        return collate_fn_dense_full(batch, num_partitions)
    def _collate_fn_dense_sl(batch):
        return collate_fn_dense_sl(batch)
    def _collate_fn_sparse_full(batch):
        return collate_fn_sparse_full(batch, num_partitions)
    def _collate_fn_sparse_sl(batch):
        return collate_fn_sparse_sl(batch)
    if feature_type=='dense':
        if use_shortlist:
            return _collate_fn_dense_sl
        else:
            return _collate_fn_dense_full
    else:
        if use_shortlist:
            return _collate_fn_sparse_sl
        else:
            return _collate_fn_sparse_full


def collate_fn_sparse_sl(batch):
    """
        Combine each sample in a batch with shortlist
        For sparse features
    """
    batch_data = {}
    batch_size = len(batch)
    seq_lengths = [len(item[0]) for item in batch]
    batch_data['X'] = torch.zeros(batch_size, max(seq_lengths)).long()
    batch_data['X_w'] = torch.zeros(batch_size, max(seq_lengths))
    sequences = [item[0] for item in batch]
    for idx, (seq, seqlen) in enumerate(zip(sequences, seq_lengths)):
        batch_data['X'][idx, :seqlen] = torch.LongTensor(seq)
        batch_data['X_w'][idx, :seqlen] = torch.FloatTensor(batch[idx][1])

    shortlist_size = len(batch[0][2])
    batch_data['Y_s'] = torch.zeros(batch_size, shortlist_size).long()
    batch_data['Y'] = torch.zeros(batch_size, shortlist_size)
    batch_data['Y_d'] = torch.zeros(batch_size, shortlist_size)
    sequences = [item[2] for item in batch]
    for idx, seq in enumerate(sequences):
        batch_data['Y_s'][idx, :] = torch.LongTensor(seq)
        batch_data['Y'][idx, :] = torch.FloatTensor(batch[idx][3])
        batch_data['Y_d'][idx, :] = torch.FloatTensor(batch[idx][4])
    return batch_data

def collate_fn_dense_sl(batch):
    """
        Combine each sample in a batch with shortlist
        For dense features
    """
    batch_data = {}
    batch_size = len(batch)
    emb_dims = batch[0][0].size
    batch_data['X'] = np.zeros((batch_size, emb_dims))
    for idx, _batch in enumerate(batch):
        batch_data['X'][idx, :] = _batch[0]
    batch_data['X'] = torch.from_numpy(batch_data['X']).type(torch.FloatTensor)
    shortlist_size = len(batch[0][2])
    batch_data['Y_s'] = torch.zeros(batch_size, shortlist_size).long()
    batch_data['Y'] = torch.zeros(batch_size, shortlist_size)
    batch_data['Y_d'] = torch.zeros(batch_size, shortlist_size)
    sequences = [item[1] for item in batch]
    for idx, seq in enumerate(sequences):
        batch_data['Y_s'][idx, :] = torch.LongTensor(seq)
        batch_data['Y'][idx, :] = torch.FloatTensor(batch[idx][2])
        batch_data['Y_d'][idx, :] = torch.FloatTensor(batch[idx][3])
    return batch_data


def collate_fn_dense_full(batch, num_partitions):
    """
        Combine each sample in a batch
        For dense features
    """
    _is_partitioned = True if num_partitions > 1 else False
    batch_data = {}
    batch_size = len(batch)
    emb_dims = batch[0][0].size
    batch_data['X'] = np.zeros((batch_size, emb_dims))
    for idx, _batch in enumerate(batch):
        batch_data['X'][idx, :] = _batch[0]
    batch_data['X'] = torch.from_numpy(batch_data['X']).type(torch.FloatTensor)
    if _is_partitioned:
        batch_data['Y'] = []
        for idx in range(num_partitions):
            batch_data['Y'].append(torch.stack([torch.from_numpy(x[1][idx]) for x in batch], 0))
    else:
        batch_data['Y'] = torch.stack([torch.from_numpy(x[1]) for x in batch], 0)
    return batch_data


def collate_fn_sparse_full(batch, num_partitions):
    """
        Combine each sample in a batch
        For sparse features
    """
    _is_partitioned = True if num_partitions > 1 else False
    batch_data = {}
    batch_size = len(batch)
    seq_lengths = [len(item[0]) for item in batch]
    batch_data['X'] = torch.zeros(batch_size, max(seq_lengths)).long()
    batch_data['X_w'] = torch.zeros(batch_size, max(seq_lengths))
    sequences = [item[0] for item in batch]
    for idx, (seq, seqlen) in enumerate(zip(sequences, seq_lengths)):
        batch_data['X'][idx, :seqlen] = torch.LongTensor(seq)
        batch_data['X_w'][idx, :seqlen] = torch.FloatTensor(batch[idx][1])
    if _is_partitioned:
        batch_data['Y'] = []
        for idx in range(num_partitions):
            batch_data['Y'].append(torch.stack([torch.from_numpy(x[2][idx]) for x in batch], 0))
    else:
        batch_data['Y'] = torch.stack([torch.from_numpy(x[2]) for x in batch], 0)
    return batch_data

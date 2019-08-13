import torch
import numpy as np


def construct_collate_fn(feature_type, use_shortlist=False, num_partitions=1):
    def _collate_fn_dense_full(batch):
        return collate_fn_dense_full(batch, num_partitions)
    def _collate_fn_dense_sl(batch):
        return collate_fn_dense_sl(batch, num_partitions)
    def _collate_fn_sparse_full(batch):
        return collate_fn_sparse_full(batch, num_partitions)
    def _collate_fn_sparse_sl(batch):
        return collate_fn_sparse_sl(batch, num_partitions)
    def _collate_fn_seq_full(batch):
        return collate_fn_seq_full(batch, num_partitions)
    def _collate_fn_seq_sl(batch):
        raise NotImplementedError

    if feature_type=='dense':
        if use_shortlist:
            return _collate_fn_dense_sl
        else:
            return _collate_fn_dense_full
    if feature_type=='sequential':
        if use_shortlist:
            return _collate_fn_seq_sl
        else:
            return _collate_fn_seq_full
    else:
        if use_shortlist:
            return _collate_fn_sparse_sl
        else:
            return _collate_fn_sparse_full


def collate_fn_sparse_sl(batch, num_partitions):
    """
        Combine each sample in a batch with shortlist
        For sparse features
    """
    _is_partitioned = True if num_partitions > 1 else False
    batch_data = {}
    batch_size = len(batch)
    seq_lengths = [len(item[0][0]) for item in batch]
    batch_data['X'] = torch.zeros(batch_size, max(seq_lengths)).long()
    batch_data['X_w'] = torch.zeros(batch_size, max(seq_lengths))
    sequences = [item[0] for item in batch]
    for idx, (seq, seqlen) in enumerate(zip(sequences, seq_lengths)):
        batch_data['X'][idx, :seqlen] = torch.LongTensor(seq[0])
        batch_data['X_w'][idx, :seqlen] = torch.FloatTensor(seq[1])

    if _is_partitioned:
        batch_data['Y'], batch_data['Y_s'], batch_data['Y_d'] = [None]*num_partitions, [None]*num_partitions, [None]*num_partitions
        for part_idx in range(num_partitions):
            shortlist_size = len(batch[0][1][0][part_idx])
            batch_data['Y_s'][part_idx] = torch.zeros(batch_size, shortlist_size).long()
            batch_data['Y'][part_idx] = torch.zeros(batch_size, shortlist_size)
            batch_data['Y_d'][part_idx] = torch.zeros(batch_size, shortlist_size)
            sequences = [item[1] for item in batch]
            for idx, seq in enumerate(sequences):
                batch_data['Y_s'][part_idx][idx, :] = torch.LongTensor(seq[0][part_idx])
                batch_data['Y'][part_idx][idx, :] = torch.FloatTensor(seq[1][part_idx])
                batch_data['Y_d'][part_idx][idx, :] = torch.FloatTensor(seq[2][part_idx])
        batch_data['Y_m'] = torch.stack([torch.LongTensor(x[1][3]) for x in batch], 0)
    else:
        shortlist_size = len(batch[0][1][0])
        batch_data['Y_s'] = torch.zeros(batch_size, shortlist_size).long()
        batch_data['Y'] = torch.zeros(batch_size, shortlist_size)
        batch_data['Y_d'] = torch.zeros(batch_size, shortlist_size)
        sequences = [item[1] for item in batch]
        for idx, seq in enumerate(sequences):
            batch_data['Y_s'][idx, :] = torch.LongTensor(seq[0])
            batch_data['Y'][idx, :] = torch.FloatTensor(seq[1])
            batch_data['Y_d'][idx, :] = torch.FloatTensor(seq[2])
    batch_data['batch_size'] = batch_size
    return batch_data

def collate_fn_dense_sl(batch, num_partitions):
    """
        Combine each sample in a batch with shortlist
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
        batch_data['Y'], batch_data['Y_s'], batch_data['Y_d'] = [None]*num_partitions, [None]*num_partitions, [None]*num_partitions
        for part_idx in range(num_partitions):
            shortlist_size = len(batch[0][1][0][part_idx])
            batch_data['Y_s'][part_idx] = torch.zeros(batch_size, shortlist_size).long()
            batch_data['Y'][part_idx] = torch.zeros(batch_size, shortlist_size)
            batch_data['Y_d'][part_idx] = torch.zeros(batch_size, shortlist_size)
            sequences = [item[1] for item in batch]
            for idx, seq in enumerate(sequences):
                batch_data['Y_s'][part_idx][idx, :] = torch.LongTensor(seq[0][part_idx])
                batch_data['Y'][part_idx][idx, :] = torch.FloatTensor(seq[1][part_idx])
                batch_data['Y_d'][part_idx][idx, :] = torch.FloatTensor(seq[2][part_idx])
        batch_data['Y_m'] = torch.stack([torch.LongTensor(x[1][3]) for x in batch], 0)
    else:
        shortlist_size = len(batch[0][1][0])
        batch_data['Y_s'] = torch.zeros(batch_size, shortlist_size).long()
        batch_data['Y'] = torch.zeros(batch_size, shortlist_size)
        batch_data['Y_d'] = torch.zeros(batch_size, shortlist_size)
        sequences = [item[1] for item in batch]
        for idx, seq in enumerate(sequences):
            batch_data['Y_s'][idx, :] = torch.LongTensor(seq[0])
            batch_data['Y'][idx, :] = torch.FloatTensor(seq[1])
            batch_data['Y_d'][idx, :] = torch.FloatTensor(seq[2])
    batch_data['batch_size'] = batch_size
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
    batch_data['batch_size'] = batch_size
    return batch_data


def collate_fn_sparse_full(batch, num_partitions):
    """
        Combine each sample in a batch
        For sparse features
    """
    _is_partitioned = True if num_partitions > 1 else False
    batch_data = {}
    batch_size = len(batch)
    seq_lengths = [len(item[0][0]) for item in batch]
    batch_data['X'] = torch.zeros(batch_size, max(seq_lengths)).long()
    batch_data['X_w'] = torch.zeros(batch_size, max(seq_lengths))
    sequences = [item[0] for item in batch]
    for idx, (seq, seqlen) in enumerate(zip(sequences, seq_lengths)):
        batch_data['X'][idx, :seqlen] = torch.LongTensor(seq[0])
        batch_data['X_w'][idx, :seqlen] = torch.FloatTensor(seq[1])
    if _is_partitioned:
        batch_data['Y'] = []
        for idx in range(num_partitions):
            batch_data['Y'].append(torch.stack([torch.from_numpy(x[1][idx]) for x in batch], 0))
    else:
        batch_data['Y'] = torch.stack([torch.from_numpy(x[1]) for x in batch], 0)
    batch_data['batch_size'] = batch_size
    return batch_data


def collate_fn_seq_full(batch, num_partitions):
    """
        Combine each sample in a batch
        For sparse features
    """
    _is_partitioned = True if num_partitions > 1 else False
    batch_data = {}
    batch_size = len(batch)
    seq_lengths = [len(item[0]) for item in batch]
    batch_data['X'] = torch.zeros(batch_size, max(seq_lengths)).long()
    batch_data['X_l'] = torch.zeros(batch_size).long()
    sequences = [item[0] for item in batch]
    for idx, (seq, seqlen) in enumerate(zip(sequences, seq_lengths)):
        batch_data['X'][idx, :seqlen] = torch.LongTensor(seq)
        batch_data['X_l'][idx] = len(seq)

    if _is_partitioned:
        batch_data['Y'] = []
        for idx in range(num_partitions):
            batch_data['Y'].append(torch.stack([torch.from_numpy(x[1][idx]) for x in batch], 0))
    else:
        batch_data['Y'] = torch.stack([torch.from_numpy(x[1]) for x in batch], 0)
    batch_data['batch_size'] = batch_size
    return batch_data

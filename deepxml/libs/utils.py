import numpy as np
import torch
import json
import os
from scipy.sparse import csr_matrix, save_npz
import numba as nb
from xclib.utils.sparse import _map


def save_predictions(preds, result_dir, valid_labels, num_samples,
                     num_labels, get_fnames=['knn', 'clf', 'combined'],
                     prefix='predictions'):
    if isinstance(preds, dict):
        for _fname, _pred in preds.items():
            if _fname in get_fnames:
                if valid_labels is not None:
                    predicted_labels = _map(
                        _pred, valid_labels,
                        shape=(num_samples, num_labels),
                        axis=1)
                else:
                    predicted_labels = _pred
                save_npz(os.path.join(
                    result_dir, '{}_{}.npz'.format(prefix, _fname)),
                    predicted_labels, compressed=False)
    else:
        if valid_labels is not None:
            predicted_labels = _map(
                preds, valid_labels,
                shape=(num_samples, num_labels),
                axis=1)
        else:
            predicted_labels = preds
        save_npz(os.path.join(result_dir, '{}.npz'.format(prefix)),
                 predicted_labels, compressed=False)


def append_padding_classifier_one(classifier, num_labels,
                                  key_w='classifier.weight',
                                  key_b='classifier.bias'):
    _num_labels, dims = classifier[key_w].size()
    if _num_labels != num_labels:
        status = "Appended padding classifier."
        _device = classifier[key_w].device
        classifier[key_w] = torch.cat(
            [classifier[key_w], torch.zeros(1, dims).to(_device)], 0)
        classifier[key_b] = torch.cat(
            [classifier[key_b], -1e5*torch.ones(1, 1).to(_device)], 0)
    else:
        status = "Shapes are fine, Not padding again."
    return status


def append_padding_classifier(net, num_labels):
    if isinstance(num_labels, list):
        status = []
        for idx, item in enumerate(num_labels):
            status.append(append_padding_classifier_one(
                net, item, 'classifier.classifier.{}.weight'.format(
                    idx), 'classifier.classifier.{}.bias'.format(idx)))
        print("Padding not implemented for distributed classifier for now!")
    else:
        return append_padding_classifier_one(net, num_labels)


def append_padding_embedding(embeddings):
    """
        Append a row of zeros as embedding for <PAD>
        Args:
            embeddings: numpy.ndarray: embedding matrix
        Returns:
            embeddings: numpy.ndarray: transformed embedding matrix
    """
    embedding_dim = embeddings.shape[1]
    app = np.zeros((1, embedding_dim))
    return np.vstack([app, embeddings])


def get_header(fname):
    with open(fname, 'r') as fp:
        line = fp.readline()
    return list(map(int, line.split(" ")))


def get_data_stats(fname, key):
    def get(fname, key):
        with open(fname, 'r') as fp:
            val = json.load(fp)[key]
        return val
    if isinstance(key, tuple):
        out = []
        for _key in key:
            out.append(get(fname, _key))
        return tuple(out)
    else:
        return get(fname, key)


def save_parameters(fname, params):
    json.dump({'num_labels': params.num_labels,
               'vocabulary_dims': params.vocabulary_dims,
               'use_shortlist': params.use_shortlist,
               'ann_method': params.ann_method,
               'num_nbrs': params.num_nbrs,
               'trans_method': params.trans_method,
               'embedding_dims': params.embedding_dims,
               'num_clf_partitions': params.num_clf_partitions,
               'label_padding_index': params.label_padding_index,
               'keep_invalid': params.keep_invalid},
              open(fname, 'w'),
              sort_keys=True,
              indent=4)


def load_parameters(fname, params):
    temp = json.load(open(fname, 'r'))
    params.num_labels = temp['num_labels']
    params.vocabulary_dims = temp['vocabulary_dims']
    params.num_nbrs = temp['num_nbrs']
    params.trans_method = temp['trans_method']
    params.num_clf_partitions = temp['num_clf_partitions']
    params.label_padding_index = temp['label_padding_index']
    params.ann_method = temp['ann_method']
    params.embedding_dims = temp['embedding_dims']
    params.keep_invalid = temp['keep_invalid']


def _select_2d(src, indices):
    n_rows, n_cols = indices.shape
    ind = np.zeros((n_rows*n_cols, 2), dtype=np.int)
    ind[:, 0] = np.repeat(np.arange(n_rows), [n_cols]*n_rows)
    ind[:, 1] = indices.flatten('C')
    return src[ind[:, 0], ind[:, 1]].flatten('C')


def update_predicted_shortlist(start_idx, batch_size, predicted_batch_labels,
                               predicted_labels, shortlist, top_k=10):
    """
        Update the predicted answers for the batch
        Args:
            predicted_batch_labels
            predicted_labels
    """
    top_values, top_indices = predicted_batch_labels.topk(
        k=top_k, dim=1, sorted=False)
    batch_size, shortlist_size = shortlist.shape
    ind = np.zeros((top_k*batch_size, 2), dtype=np.int)
    ind[:, 0] = np.repeat(
        np.arange(start_idx, start_idx+batch_size, 1), [top_k]*batch_size)
    ind[:, 1] = _select_2d(shortlist, top_indices.cpu().numpy())
    vals = top_values.cpu().numpy().flatten('C')
    predicted_labels[ind[:, 0], ind[:, 1]] = vals


def update_predicted(start_idx, batch_size, predicted_batch_labels,
                     predicted_labels, top_k=10):
    """
        Update the predicted answers for the batch
        Args:
            predicted_batch_labels
            predicted_labels
    """
    top_values, top_indices = predicted_batch_labels.topk(
        k=top_k, dim=1, sorted=False)
    ind = np.zeros((top_k*batch_size, 2), dtype=np.int)
    ind[:, 0] = np.repeat(
        np.arange(start_idx, start_idx+batch_size, 1), [top_k]*batch_size)
    ind[:, 1] = top_indices.cpu().numpy().flatten('C')
    vals = top_values.cpu().numpy().flatten('C')
    predicted_labels[ind[:, 0], ind[:, 1]] = vals


@nb.njit(cache=True)
def bin_index(array, item): # Binary search
    first, last = 0, len(array) - 1

    while first <= last:
        mid = (first + last) // 2
        if array[mid] == item:
            return mid

        if item < array[mid]:
            last = mid - 1
        else:
            first = mid + 1

    return -1


@nb.njit(cache=True)
def safe_normalize(array):
    _max = np.max(array)
    if _max != 0:
        return array/_max
    else:
        return array


@nb.njit(cache=True)
def safe_normalize(array):
    _max = np.max(array)
    if _max != 0:
        return array/_max
    else:
        return array


@nb.njit(nb.types.Tuple((nb.int64[:], nb.float32[:]))(nb.int64[:, :], nb.float32[:], nb.int64))
def map_one(indices_labels, similarity, padding_ind):
    unique_point_labels = np.unique(indices_labels)
    unique_point_labels = unique_point_labels[unique_point_labels != padding_ind]
    point_label_similarity = np.zeros((len(unique_point_labels), ), dtype=np.float32)
    for j in range(len(indices_labels)):
        for lbl in indices_labels[j]:
            if(lbl != padding_ind):
                _ind = bin_index(unique_point_labels, lbl)
                point_label_similarity[_ind] += similarity[j]
    point_label_similarity = safe_normalize(point_label_similarity)
    return unique_point_labels, point_label_similarity



@nb.njit(nb.types.Tuple((nb.int64[:, :], nb.float32[:, :]))(nb.int64[:, :], nb.float32[:, :], nb.int64[:, :], nb.int64, nb.int64, nb.float32), parallel=True)
def map_neighbors(indices, similarity, labels, top_k, padding_ind, padding_val):
    m = indices.shape[0]
    point_labels = np.full(
        (m, top_k), padding_ind, dtype=np.int64)
    point_label_similarities = np.full(
        (m, top_k), padding_val, dtype=np.float32)
    for i in nb.prange(m):
        unique_point_labels, point_label_similarity = map_one(labels[indices[i]], similarity[i], padding_ind)
        if top_k < len(unique_point_labels):
            top_indices = np.argsort(
                point_label_similarity)[-1 * top_k:][::-1]
            point_labels[i] = unique_point_labels[top_indices]
            point_label_similarities[i] = point_label_similarity[top_indices]
        else:
            point_labels[i, :len(unique_point_labels)] = unique_point_labels
            point_label_similarities[i, :len(unique_point_labels)] = point_label_similarity
    return point_labels, point_label_similarities


@nb.njit(cache=True)
def _remap_centroid_one(indices, sims, mapping):
    mapped_indices = mapping[indices]
    unique_mapped_indices = np.unique(mapped_indices)
    unique_mapped_sims = np.zeros(
        (len(unique_mapped_indices), ), dtype=np.float32)
    for i in range(len(unique_mapped_indices)):
        ind = unique_mapped_indices[i]
        unique_mapped_sims[i] = np.max(sims[mapped_indices == ind])
    return unique_mapped_indices, unique_mapped_sims


@nb.njit()
def map_centroids(indices, sims, mapping, pad_ind, pad_val):
    mapped_indices = np.full(
        indices.shape, fill_value=pad_ind, dtype=np.int64)
    mapped_sims = np.full(
        indices.shape, fill_value=pad_val, dtype=np.float32)

    for i in nb.prange(indices.shape[0]):
        _ind, _sim = _remap_centroid_one(indices[i], sims[i], mapping)
        mapped_indices[i, :len(_ind)] = _ind
        mapped_sims[i, :len(_sim)] = _sim
    return mapped_indices, mapped_sims

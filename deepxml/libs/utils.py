import numpy as np
import torch
import json
import os
from scipy.sparse import csr_matrix, save_npz
from xctools.data import data_utils
import pdb

def compute_svd(X, num_components):
    """
        Perform low rank factorization
        Args:
            X: torch.Tensor: (num_labels, input_dims)
            num_components: int: choose top singular vectors
        Returns:
            U: torch.Tensor: (num_labels, num_components)
            V: torch.Tensor: (num_components, input_dims)
    """
    U, s, Vh = torch.svd(X)
    U, s, Vh = U[:, :num_components], torch.diag(s[:num_components]), Vh[:, :num_components]
    return U@s, Vh


def write_predictions(predictions, result_dir, fname, label_mapping, num_samples, 
                      num_labels):
    ext = ['.npz', '_knn.npz', '_clf.npz']
    if isinstance(predictions, tuple):
        for _pred, _ext in zip(predictions, ext):
            _fname = os.path.join(result_dir, fname.replace(".txt", _ext))
            if label_mapping is not None:
                _pred = map_to_original(
                    _pred, label_mapping, (num_samples, num_labels))
                #print(": ", _fname, _pred.nnz)
            save_npz(_fname, _pred)
            #data_utils.write_sparse_file(_pred, _fname, header=True)
    else:
        fname = os.path.join(result_dir, fname)
        if label_mapping is not None:
            predictions = map_to_original(
                predictions, label_mapping, (num_samples, num_labels))
        _fname = os.path.join(result_dir, fname.replace(".txt", ext[0]))
        save_npz(_fname, predictions)


def adjust_for_low_rank(state_dict, rank):
    clf_wts = state_dict['classifier.weight']
    U, V = compute_svd(clf_wts.t(), rank)
    state_dict['classifier.weight'] = V
    state_dict['low_rank_layer.weight'] = U.t()


def append_padding_classifier(net, num_labels):
    _num_labels, dims = net['classifier.weight'].size()
    if _num_labels != num_labels:
        status = "Appended padding classifier."
        _device = net['classifier.weight'].device
        net['classifier.weight'] = torch.cat(
            [net['classifier.weight'], torch.zeros(1, dims).to(_device)], 0)
        net['classifier.bias'] = torch.cat(
            [net['classifier.bias'], -1e5*torch.ones(1, 1).to(_device)], 0)
    else:
        status = "Shapes are fine, Not padding again."
    return status


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


def get_scores(out_ans, batch_dist, beta):
    """
        Combine predicted scores and Approx knn distances
    """
    return beta*torch.sigmoid(out_ans) + (1-beta)*torch.sigmoid(1-batch_dist)


def get_data_header(fname):
    with open(fname, 'r') as fp:
        header = fp.readline()
        try:
            num_samples, num_features, num_labels = header.split(" ")
        except Exception as e:
            print("Error: FTF")
            exit()
    return int(num_samples), int(num_features), int(num_labels)


def map_to_original(mat, mapping, _shape, axis=1):
    mat = mat.tocsr()
    row_idx, col_idx = mat.nonzero()
    vals = np.array(mat[row_idx, col_idx]).squeeze()
    col_indices = list(map(lambda x: mapping[x], col_idx))
    return csr_matrix((vals, (np.array(row_idx), np.array(col_indices))), shape=_shape)


def save_parameters(fname, params):
    json.dump({'num_labels': params.num_labels,
               'vocabulary_dims': params.vocabulary_dims,
               'use_shortlist': params.use_shortlist,
               'use_residual': params.use_residual,
               'ann_method': params.ann_method,
               'num_nbrs': params.num_nbrs,
               'embedding_dims': params.embedding_dims,
               'label_padding_index': params.label_padding_index,
               'hidden_dims': params.hidden_dims,
               'use_hash_embeddings': params.use_hash_embeddings,
               'num_buckets': params.num_buckets,
               'num_hashes': params.num_hashes,
               'trans_method': params.trans_method,
               'keep_invalid': params.keep_invalid},
              open(fname, 'w'),
              sort_keys=True,
              indent=4)


def load_parameters(fname, params):
    temp = json.load(open(fname, 'r'))
    params.num_labels = temp['num_labels']
    params.vocabulary_dims = temp['vocabulary_dims']
    params.use_residual = temp['use_residual']
    params.num_nbrs = temp['num_nbrs']
    params.ann_method = temp['ann_method']
    params.num_hashes = temp['num_hashes']
    params.num_buckets = temp['num_buckets']
    params.label_padding_index = temp['label_padding_index']
    params.use_hash_embeddings = temp['use_hash_embeddings'] 
    params.ann_method = temp['ann_method']
    params.embedding_dims = temp['embedding_dims']
    params.trans_method = temp['trans_method']
    params.hidden_dims = temp['hidden_dims']
    params.keep_invalid = temp['keep_invalid']

def get_label_embeddings(document_embeddings, labels):
    label_embeddings = labels.transpose().dot(document_embeddings)
    return label_embeddings


def _select_2d(src, indices):
    n_rows, n_cols = indices.shape
    ind = np.zeros((n_rows*n_cols, 2), dtype=np.int)
    ind[:, 0] = np.repeat(np.arange(n_rows), [n_cols]*n_rows)
    ind[:, 1] = indices.flatten('C')
    return src[ind[:, 0], ind[:, 1]].flatten('C')


def update_predicted_shortlist(start_idx, batch_size, predicted_batch_labels, predicted_labels, shortlist, top_k=10):
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


def update_predicted(start_idx, batch_size, predicted_batch_labels, predicted_labels, top_k=10):
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


def _recall(true_labels, predicted_labels):
    total_labels = np.sum(true_labels, axis=1)
    _r = true_labels.multiply(predicted_labels).astype(
        np.bool).astype(np.int32)
    total_predicted = np.sum(_r, axis=1)/(total_labels+1e-5)
    recall_shortlist = np.mean(total_predicted)
    return recall_shortlist

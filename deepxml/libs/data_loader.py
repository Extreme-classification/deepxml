import torch
import pickle
import os
import pickle
import sys
from scipy.sparse import lil_matrix
import numpy as np
from sklearn.preprocessing import normalize

import xctools.data.data_utils as data_utils

def collate_fn_sl(batch):
    """
        Combine each sample in a batch with shortlist
    """
    batch_size = len(batch)
    seq_lengths = [len(item[0]) for item in batch]
    seq_tensor = torch.zeros(batch_size, max(seq_lengths)).long()
    wt_tensor = torch.zeros(batch_size, max(seq_lengths))
    sequences = [item[0] for item in batch]
    for idx, (seq, seqlen) in enumerate(zip(sequences, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        wt_tensor[idx, :seqlen] = torch.FloatTensor(batch[idx][1])

    shortlist_size = len(batch[0][2])
    batch_shortlist = torch.zeros(batch_size, shortlist_size).long()
    batch_labels_mask = torch.zeros(batch_size, shortlist_size)
    batch_dist = torch.zeros(batch_size, shortlist_size)
    sequences = [item[2] for item in batch]
    for idx, seq in enumerate(sequences):
        batch_shortlist[idx, :] = torch.LongTensor(seq)
        batch_labels_mask[idx, :] = torch.FloatTensor(batch[idx][3])
        batch_dist[idx, :] = torch.FloatTensor(batch[idx][4])
    return seq_tensor, wt_tensor, batch_shortlist, batch_labels_mask, batch_dist


def collate_fn_fx_sl(batch):
    """
        Combine each sample in a batch with shortlist
    """
    batch_size = len(batch)
    emb_dims = batch[0][0].size
    seq_tensor = np.zeros((batch_size, emb_dims))
    for idx,bat in enumerate(batch):
        seq_tensor[idx, :] = bat[0]
    seq_tensor = torch.from_numpy(seq_tensor).type(torch.FloatTensor)
    shortlist_size = len(batch[0][2])
    batch_shortlist = torch.zeros(batch_size, shortlist_size).long()
    batch_labels_mask = torch.zeros(batch_size, shortlist_size)
    batch_dist = torch.zeros(batch_size, shortlist_size)
    sequences = [item[2] for item in batch]
    for idx, seq in enumerate(sequences):
        batch_shortlist[idx, :] = torch.LongTensor(seq)
        batch_labels_mask[idx, :] = torch.FloatTensor(batch[idx][3])
        batch_dist[idx, :] = torch.FloatTensor(batch[idx][4])
    return seq_tensor, torch.FloatTensor([1.0]), batch_shortlist, batch_labels_mask, batch_dist



def collate_fn_fx_full(batch):
    """
        Combine each sample in a batch
    """
    batch_size = len(batch)
    emb_dims = batch[0][0].size
    seq_tensor = np.zeros((batch_size, emb_dims))
    for idx, bat in enumerate(batch):
        seq_tensor[idx, :] = bat[0]
    seq_tensor = torch.from_numpy(seq_tensor).type(torch.FloatTensor)
    lb = torch.stack([torch.from_numpy(x[2]) for x in batch], 0)
    return seq_tensor, torch.FloatTensor([1.0]), lb




def collate_fn_full(batch):
    """
        Combine each sample in a batch
    """
    batch_size = len(batch)
    seq_lengths = [len(item[0]) for item in batch]
    seq_tensor = torch.zeros(batch_size, max(seq_lengths)).long()
    wt_tensor = torch.zeros(batch_size, max(seq_lengths))
    sequences = [item[0] for item in batch]
    for idx, (seq, seqlen) in enumerate(zip(sequences, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        wt_tensor[idx, :seqlen] = torch.FloatTensor(batch[idx][1])
    lb = torch.stack([torch.from_numpy(x[2]) for x in batch], 0)
    return seq_tensor, wt_tensor, lb


def _get_split_id(fname):
    idx = fname.split("_")[-1].split(".")[0]
    return idx


class XMLDataset(torch.utils.data.Dataset):
    """
        Dataset to load and use XML-Datasets
    """

    def __init__(self, data_dir, fname, model_dir='', mode='train', use_shortlist=False, size_shortlist=-1,
                feature_indices=None, label_indices=None, normalize_features=True, keep_invalid=False, 
                num_centroids=1, embeddings=None):
        """
            Expects 'libsvm' format with header
            Args:
                data_file: str: File name for the set
        """
        self._split = None
        self._ext_head = None
        self.data_dir = data_dir
        fname = os.path.join(data_dir, fname)
        self.num_centroids = num_centroids #Use multiple centroids for ext head labels
        self.multiple_cent_mapping = None
        self.features, self.labels, self.num_samples, \
            self.num_features, self.num_labels = data_utils.read_data(fname)
        if feature_indices is not None:
            feature_indices = np.loadtxt(feature_indices, dtype=np.int32)
            self.features = self.features[:, feature_indices]
            self.num_features = self.features.shape[1]
        if normalize_features:
            self.features = normalize(self.features, copy=False)
        self.labels = data_utils.binarize_labels(self.labels, self.num_labels)
        if label_indices is not None:
            self._split = _get_split_id(label_indices) #Label wala split
            label_indices = np.loadtxt(label_indices, dtype=np.int32)
            self.labels = self.labels[:, label_indices]
            self.num_labels = self.labels.shape[1]
        self.mode = mode
        self.shortlist = None
        self.dist = None
        self.use_shortlist = use_shortlist
        self.size_shortlist = size_shortlist
        self.embeddings = embeddings
        if self.mode == 'train':
            self._remove_document_wo_features_and_labels()
        if not keep_invalid:
            self._process_labels(model_dir)
        self.label_padding_index = self.num_labels
        self.use_fixed = False
        if self.embeddings is not None:
            self.features = self.features.dot(embeddings)
            self.use_fixed = True

    def _remove_document_wo_features_and_labels(self):
        # Remove instances if they don't have any feature or label
        def _compute_freq(data):
            return np.array(data.sum(axis=1)).ravel()
        freq = _compute_freq(self.features)
        indices_feat = np.where(freq > 0)[0]
        freq = _compute_freq(self.labels)
        indices_labels = np.where(freq > 0)[0]
        indices = np.intersect1d(indices_feat, indices_labels)
        self.features = self.features[indices]
        self.labels = self.labels[indices]
        self.num_samples = indices.size

    def _convert_shortlist_to_sparse(self):
        assert self.shortlist is not None
        shortlist_sp = lil_matrix(
            (self.num_samples, self.num_labels), dtype=np.float32)
        for idx in range(self.num_samples):
            pos_labels = self.labels[idx, :].indices.tolist()
            short = self.shortlist[idx].tolist()
            dist = self.dist[idx].tolist()
            short, labels_mask, _ = self._adjust_shortlist(
                pos_labels, short, dist)
            labels_mask[labels_mask == 0] = -1
            shortlist_sp[idx, short] = labels_mask
        return shortlist_sp

    def _pad_seq(self, indices, dist):
        _pad_length = self.size_shortlist - len(indices)
        indices.extend([self.label_padding_index]*_pad_length)
        dist.extend([100]*_pad_length)


    def _remap_multiple_centroids(self, indices, vals, _func=min, _limit=1e5):
        indices = np.asarray(list(map(lambda x: self.multiple_cent_mapping[x], indices)))
        _dict = dict({})
        for id, ind in enumerate(indices):
            _dict[ind] = _func(_dict.get(ind, _limit), vals[id])
        indices, values = zip(*_dict.items())
        indices, values = list(indices), list(values)
        if len(indices) < self.size_shortlist:
            self._pad_seq(indices, values)
        return indices, values

    def _process_labels(self, data_dir, _ext_head_threshold=10000):
        def _get_ext_head(freq, threshold):
            return np.where(freq>=threshold)[0]
        obj = {}
        fname = os.path.join(data_dir, 'labels_params.pkl' if self._split is None else "labels_params_split_{}.pkl".format(self._split))
        if self.mode == 'train':
            valid_labels = np.where(np.array(self.labels.sum(axis=0)) != 0)[1]
            obj['valid_labels'] = valid_labels
            obj['num_labels'] = self.num_labels
            obj['ext_head'] = None
            obj['multiple_cent_mapping'] = None
            self.labels = self.labels[:, valid_labels]
            self.num_labels = valid_labels.size
            print("Valid labels after processing: ", self.num_labels)
            if self.num_centroids != 1:
                freq = np.array(self.labels.sum(axis=0)).ravel()
                self._ext_head = _get_ext_head(freq, _ext_head_threshold)
                self.multiple_cent_mapping = np.arange(self.num_labels)
                for idx in self._ext_head:
                    self.multiple_cent_mapping = np.append(self.multiple_cent_mapping, [idx]*self.num_centroids)
                obj['ext_head'] = self._ext_head
                obj['multiple_cent_mapping'] = self.multiple_cent_mapping
            pickle.dump(obj, open(fname, 'wb'))
        # Retrain with ANNS
        elif self.mode == 'retrain_w_shorty':
            obj = pickle.load(open(fname, 'rb'))
            valid_labels = obj['valid_labels']
            self.labels = self.labels[:, valid_labels]
            self.num_labels = valid_labels.size
            if self.num_centroids != 1:
                freq = np.array(self.labels.sum(axis=0)).ravel()
                self._ext_head = _get_ext_head(freq, _ext_head_threshold)
                self.multiple_cent_mapping = np.arange(self.num_labels)
                for idx in self._ext_head:
                    self.multiple_cent_mapping = np.append(
                        self.multiple_cent_mapping, [idx]*self.num_centroids)
                obj['ext_head'] = self._ext_head
                obj['multiple_cent_mapping'] = self.multiple_cent_mapping
                print("Ext labels: ", self._ext_head)
            pickle.dump(obj, open(fname, 'wb'))
        else:
            obj = pickle.load(open(fname, 'rb'))
            self._ext_head = obj['ext_head']
            self.multiple_cent_mapping = obj['multiple_cent_mapping']
            valid_labels = obj['valid_labels']
            self.labels = self.labels[:, valid_labels]
            self.num_labels = valid_labels.size

    def update_shortlist(self, shortlist, dist):
        self.shortlist = shortlist
        self.dist = dist
    
    def save_shortlist(self, fname):
        pickle.dump({"shortlist": self.shortlist,
                     "dist": self.dist}, open(fname, 'wb'))

    def load_shortlist(self, fname):
        _temp = pickle.load(open(fname, 'rb'))
        self.shortlist = _temp['shortlist']
        self.dist = _temp['dist']
        
    def get_stats(self):
        """
            Get dataset statistics
            Returns:
                Num Samples, Num features and Num labels
        """
        return self.num_samples, self.num_features, self.num_labels

    def _adjust_shortlist(self, pos_labels, shortlist, dist):
        if self.mode == 'train':
            # TODO: Adjust dist as well
            neg_labels = list(
                filter(lambda x: x not in set(pos_labels), shortlist))
            diff = self.size_shortlist - len(pos_labels)
            labels_mask = [1]*len(pos_labels)
            dist = [2]*len(pos_labels) + dist[:diff]
            shortlist = pos_labels + neg_labels[:diff]
            labels_mask = labels_mask + [0]*diff
        else:
            labels_mask = [0]*self.size_shortlist
            pos_labels = set(pos_labels)
            for idx, item in enumerate(shortlist):
                if item in pos_labels:
                    labels_mask[idx] = 1
        return shortlist, labels_mask, dist

    def _get_sl(self, index):
        feat = self.features[index, :].nonzero()[1].tolist()
        wt = self.features[index, feat].todense().tolist()[0]
        feat = [item+1 for item in feat]  # Treat idx:0 as Padding
        pos_labels = self.labels[index, :].indices.tolist()
        if self.shortlist is not None:
            shortlist = self.shortlist[index].tolist()
            dist = self.dist[index].tolist()
            # Remap to original labels if multiple centroids are used
            if self.num_centroids != 1:
                shortlist, dist = self._remap_multiple_centroids(shortlist, dist)
            shortlist, labels_mask, dist = self._adjust_shortlist(
                pos_labels, shortlist, dist)
        else:
            shortlist = [0]*self.size_shortlist
            labels_mask = [0]*self.size_shortlist
            dist = [0]*self.size_shortlist
        return feat, wt, shortlist, labels_mask, dist

    def _get_fx_sl(self, index):
        pos_labels = self.labels[index, :].indices.tolist()
        if self.shortlist is not None:
            shortlist = self.shortlist[index].tolist()
            dist = self.dist[index].tolist()
            # Remap to original labels if multiple centroids are used
            if self.num_centroids != 1:
                shortlist, dist = self._remap_multiple_centroids(shortlist, dist)
            shortlist, labels_mask, dist = self._adjust_shortlist(
                pos_labels, shortlist, dist)
        else:
            shortlist = [0]*self.size_shortlist
            labels_mask = [0]*self.size_shortlist
            dist = [0]*self.size_shortlist
        return self.features[index], [1], shortlist, labels_mask, dist
    
    def _get_full(self, index):
        feat = self.features[index, :].nonzero()[1].tolist()
        wt = self.features[index, feat].todense().tolist()[0]
        feat = [item+1 for item in feat]  # Treat idx:0 as Padding
        lb = np.array(self.labels[index, :].todense(),
                      dtype=np.float32).reshape(self.num_labels)
        return feat, wt, lb

    def _get_fx_full(self, index):
        lb = np.array(self.labels[index, :].todense(),
                      dtype=np.float32).reshape(self.num_labels)
        return self.features[index], [1], lb

    def __getitem__(self, index):
        """
            Get features and labels for index
            Args:
                index: for this sample
            Returns:
                features: : non zero entries
                labels: : numpy array

        """
        if self.use_shortlist:
            if self.use_fixed:
                return self._get_fx_sl(index)
            return self._get_sl(index)
        else:
            if self.use_fixed:
                return self._get_fx_full(index)
            return self._get_full(index)

    def __len__(self):
        return self.num_samples

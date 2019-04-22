import torch
import _pickle as pickle
import os
import sys
from scipy.sparse import lil_matrix
import numpy as np
from sklearn.preprocessing import normalize
from .dataset_base import DatasetBase
import xclib.data.data_utils as data_utils


def construct_dataset(data_dir, fname, data=None, model_dir='', mode='train', size_shortlist=-1,
                      normalize_features=True, keep_invalid=False, num_centroids=1, 
                      feature_type='sparse', **kwargs):
    feature_indices, label_indices = None, None 
    # FIXME: See if this can be done efficently
    if 'feature_indices' in kwargs:
        feature_indices = kwargs['feature_indices']
    if 'label_indices' in kwargs:
        label_indices = kwargs['label_indices']
     # Construct dataset for dense data
    if feature_type == 'dense':
        return DatasetDense(data_dir, fname, data, model_dir, mode, size_shortlist,
                            feature_indices, label_indices, normalize_features, keep_invalid, num_centroids)
    elif feature_type == 'sparse':
       # Construct dataset for sparse data
        return DatasetSparse(data_dir, fname, data, model_dir, mode, size_shortlist,
                             feature_indices, label_indices, normalize_features, keep_invalid, num_centroids)
    else:
        raise NotImplementedError(
            "Feature type: {}, not yet supported.".format(feature_type))


class DatasetDense(DatasetBase):
    """
        Dataset to load and use dense XML-Datasets
    """

    def __init__(self, data_dir, fname, data=None, model_dir='', mode='train', 
                 size_shortlist=-1, feature_indices=None, label_indices=None, 
                 normalize_features=True, keep_invalid=False, num_centroids=1):
        """
            Expects 'libsvm' format with header
            Args:
                data_file: str: File name for the set
        """
        super().__init__(data_dir, fname, data, model_dir, mode, size_shortlist,
                         label_indices, keep_invalid, num_centroids)
        self._split = None
        self.feature_type = 'dense'
        self._ext_head = None
        if not keep_invalid:
            self._process_labels(model_dir)
        if self.mode == 'train':
            self._remove_samples_wo_features_and_labels()
        self.label_padding_index = self.num_labels
        self._sel_features(feature_indices)
        if normalize_features:
            self.features = self._normalize(self.features)

    def _sel_features(self, feature_indices):
        """
            Train only for given features
        """
        def _get_split_id(fname):
            idx = fname.split("_")[-1].split(".")[0]
            return idx
        if feature_indices is not None:
            self._split = _get_split_id(feature_indices)
            feature_indices = np.loadtxt(feature_indices, dtype=np.int32)
            self.features = self.features[:, feature_indices]
            self.num_samples, self.num_features = self.features.shape

    def _normalize(self, data):
        return normalize(data, copy=False)

    def _remove_samples_wo_features_and_labels(self):
        """
            Remove instances if they don't have any feature or label
        """
        def _compute_freq(data):
            return np.array(data.sum(axis=1)).ravel()
        freq = _compute_freq(self.features)
        indices_feat = np.where(freq > 0)[0]
        freq = _compute_freq(self.labels.astype(np.bool))
        indices_labels = np.where(freq > 0)[0]
        indices = np.intersect1d(indices_feat, indices_labels)
        self.features = self.features[indices]
        self.labels = self.labels[indices]
        self.num_samples = indices.size

    def _get_sl(self, index):
        """
            Get data for given index with shortlist
        """
        pos_labels = self.labels[index, :].indices.tolist()
        if self.shortlist is not None:
            shortlist = self.shortlist[index].tolist()
            dist = self.dist[index].tolist()
            # Remap to original labels if multiple centroids are used
            if self.num_centroids != 1:
                shortlist, dist = self._remap_multiple_centroids(
                    shortlist, dist)
            shortlist, labels_mask, dist = self._adjust_shortlist(
                pos_labels, shortlist, dist)
        else:
            shortlist = [0]*self.size_shortlist
            labels_mask = [0]*self.size_shortlist
            dist = [0]*self.size_shortlist
        return self.features[index], shortlist, labels_mask, dist

    def _get_full(self, index):
        """
            Get data for given index
        """
        lb = np.array(self.labels[index, :].todense(),
                      dtype=np.float32).reshape(self.num_labels)
        return self.features[index], lb

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
            return self._get_sl(index)
        else:
            return self._get_full(index)


class DatasetSparse(DatasetBase):
    """
        Dataset to load and use XML-Dataset with sparse features
    """

    def __init__(self, data_dir, fname, data=None, model_dir='', mode='train',
                 size_shortlist=-1, feature_indices=None, label_indices=None,
                 normalize_features=True, keep_invalid=False, num_centroids=1):
        """
            Expects 'libsvm' format with header
            Args:
                data_file: str: File name for the set
        """
        super().__init__(data_dir, fname, data, model_dir, mode, size_shortlist,
                         label_indices, keep_invalid, num_centroids)
        self._split = None
        self._ext_head = None
        if self.mode == 'train':
            self._remove_samples_wo_features_and_labels()
        self.label_padding_index = self.num_labels
        self._sel_features(feature_indices)
        if normalize_features:
            self.features = self._normalize(self.features)

    def _sel_features(self, feature_indices):
        """
            Train only for given features
        """
        def _get_split_id(fname):
            idx = fname.split("_")[-1].split(".")[0]
            return idx
        if feature_indices is not None:
            self._split = _get_split_id(feature_indices)
            feature_indices = np.loadtxt(feature_indices, dtype=np.int32)
            self.features = self.features[:, feature_indices]
            self.num_samples, self.num_features = self.features.shape

    def _normalize(self, data):
        return normalize(data, copy=False)

    def _remove_samples_wo_features_and_labels(self):
        """
            Remove instances if they don't have any feature or label
        """
        def _compute_freq(data):
            return np.array(data.sum(axis=1)).ravel()
        freq = _compute_freq(self.features)
        indices_feat = np.where(freq > 0)[0]
        freq = _compute_freq(self.labels.astype(np.bool))
        indices_labels = np.where(freq > 0)[0]
        indices = np.intersect1d(indices_feat, indices_labels)
        self.features = self.features[indices]
        self.labels = self.labels[indices]
        self.num_samples = indices.size

    def _get_sl(self, index):
        """
            Get data with shortlist for given data index
        """
        feat = self.features[index, :].nonzero()[1].tolist()
        wt = self.features[index, feat].todense().tolist()[0]
        feat = [item+1 for item in feat]  # Treat idx:0 as Padding
        pos_labels = self.labels[index, :].indices.tolist()
        if self.shortlist is not None:
            shortlist = self.shortlist[index].tolist()
            dist = self.dist[index].tolist()
            # Remap to original labels if multiple centroids are used
            if self.num_centroids != 1:
                shortlist, dist = self._remap_multiple_centroids(
                    shortlist, dist)
            shortlist, labels_mask, dist = self._adjust_shortlist(
                pos_labels, shortlist, dist)
        else:
            shortlist = [0]*self.size_shortlist
            labels_mask = [0]*self.size_shortlist
            dist = [0]*self.size_shortlist
        return feat, wt, shortlist, labels_mask, dist

    def _get_full(self, index):
        """
            Get data for given data index
        """
        feat = self.features[index, :].nonzero()[1].tolist()
        wt = self.features[index, feat].todense().tolist()[0]
        feat = [item+1 for item in feat]  # Treat idx:0 as Padding
        lb = np.array(self.labels[index, :].todense(),
                      dtype=np.float32).reshape(self.num_labels)
        return feat, wt, lb

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
            return self._get_sl(index)
        else:
            return self._get_full(index)

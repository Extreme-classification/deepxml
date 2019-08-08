import torch
import _pickle as pickle
import os
import sys
from scipy.sparse import lil_matrix
import numpy as np
from sklearn.preprocessing import normalize
from .dataset_base import DatasetBase
import xclib.data.data_utils as data_utils
from .dist_utils import Partitioner
import operator
from .lookup import Table, PartitionedTable
from .shortlist_handler import ShortlistHandlerStatic, ShortlistHandlerDynamic


def construct_dataset(data_dir, fname_features, fname_labels, data=None, model_dir='',
                      mode='train', size_shortlist=-1, normalize_features=True,
                      normalize_labels=True, keep_invalid=False, num_centroids=1,
                      feature_type='sparse', num_clf_partitions=1, feature_indices=None,
                      label_indices=None, shortlist_type='static', shorty=None):
    if size_shortlist == -1:
        return DatasetDense(data_dir, fname_features, fname_labels, data, model_dir, mode,
                            feature_indices, label_indices, keep_invalid, normalize_features,
                            normalize_labels, num_clf_partitions, feature_type)
    else:
       # Construct dataset for sparse data
        return DatasetSparse(data_dir, fname_features, fname_labels, data, model_dir,
                             mode, feature_indices, label_indices, keep_invalid,
                             normalize_features, normalize_labels, num_clf_partitions,
                             size_shortlist, num_centroids, feature_type, shortlist_type, 
                             shorty)


class DatasetDense(DatasetBase):
    """
        Dataset to load and use XML-Datasets with full output space only
    """

    def __init__(self, data_dir, fname_features, fname_labels, data=None, model_dir='',
                 mode='train', feature_indices=None, label_indices=None, keep_invalid=False,
                 normalize_features=True, normalize_labels=False, num_clf_partitions=1,
                 feature_type='sparse', label_type='dense'):
        """
            Expects 'libsvm' format with header
            Args:
                data_file: str: File name for the set
            Can Support datasets w/o any label
        """
        super().__init__(data_dir, fname_features, fname_labels, data, model_dir,
                         mode, feature_indices, label_indices, keep_invalid,
                         normalize_features, normalize_labels, feature_type, label_type)
        if not keep_invalid and self.labels._valid:
            # Remove labels w/o any positive instance
            self._process_labels(model_dir)
        if self.mode == 'train':
            # Remove samples w/o any feature or label
            self._remove_samples_wo_features_and_labels()
        self.feature_type = feature_type
        self.partitioner = None
        self.num_clf_partitions = num_clf_partitions if self.labels._valid else 1
        if self.mode == 'train':
            assert self.labels._valid, "Labels can not be None while training.."
            if self.num_clf_partitions > 1:
                self.partitioner = Partitioner(
                    self.num_labels, self.num_clf_partitions, padding=False, contiguous=True)
                self.partitioner.save(os.path.join(
                    self.model_dir, 'partitionar.pkl'))
        else:
            if self.num_clf_partitions > 1:
                self.partitioner = Partitioner(
                    self.num_labels, self.num_clf_partitions, padding=False, contiguous=True)
                self.partitioner.load(os.path.join(
                    self.model_dir, 'partitionar.pkl'))

        # TODO Take care of this select and padding index
        self.label_padding_index = self.num_labels

    def __getitem__(self, index):
        """
            Get features and labels for index
            Args:
                index: for this sample
            Returns:
                features: : non zero entries
                labels: : numpy array

        """
        x = self.features[index]
        y = self.labels[index]
        if self.partitioner is not None:  # Split if required
            y = self.partitioner.split(y)
        return x, y


class DatasetSparse(DatasetBase):
    """
        Dataset to load and use XML-Dataset with shortlist
    """

    def __init__(self, data_dir, fname_features, fname_labels, data=None, model_dir='',
                 mode='train', feature_indices=None, label_indices=None, keep_invalid=False,
                 normalize_features=True, normalize_labels=False, num_clf_partitions=1,
                 size_shortlist=-1, num_centroids=1, feature_type='sparse', shortlist_type='static',
                 shorty=None, label_type='sparse', shortlist_in_memory=True):
        """
            Expects 'libsvm' format with header
            Args:
                data_file: str: File name for the set
        """
        super().__init__(data_dir, fname_features, fname_labels, data, model_dir, mode,
                         feature_indices, label_indices, keep_invalid, normalize_features,
                         normalize_labels, feature_type, label_type)
        if self.labels is None:
            NotImplementedError(
                "No support for shortlist w/o any label, consider using dense dataset.")
        self.feature_type = feature_type
        self.num_centroids = num_centroids
        self.num_clf_partitions = num_clf_partitions
        self.shortlist_in_memory = shortlist_in_memory
        self.size_shortlist = size_shortlist
        self.multiple_cent_mapping = None
        self.shortlist_type = shortlist_type
        if not keep_invalid:
            # Remove labels w/o any positive instance
            self._process_labels(model_dir)
        if self.mode == 'train':
            # Remove samples w/o any feature or label
            self._remove_samples_wo_features_and_labels()
        if shortlist_type == 'static':
            self.shortlist = ShortlistHandlerStatic(self.num_labels, model_dir, num_clf_partitions,
                                                    mode, size_shortlist, num_centroids,
                                                    shortlist_in_memory, self.multiple_cent_mapping)
        else:
            self.shortlist = ShortlistHandlerDynamic(self.num_labels, shorty, model_dir, num_clf_partitions,
                                                    mode, size_shortlist, num_centroids,
                                                    self.multiple_cent_mapping)
        self.use_shortlist = True if self.size_shortlist > 0 else False
        self.label_padding_index = self.num_labels

    def update_shortlist(self, shortlist, dist, fname='tmp', idx=-1):
        """
            Update label shortlist for each instance
        """
        self.shortlist.update_shortlist(shortlist, dist, fname, idx)

    def save_shortlist(self, fname):
        """
            Save label shortlist and distance for each instance
        """
        self.shortlist.save_shortlist(fname)

    def load_shortlist(self, fname):
        """
            Load label shortlist and distance for each instance
        """
        self.shortlist.load_shortlist(fname)

    def _get_ext_head(self, freq, threshold):
        return np.where(freq >= threshold)[0]

    def _process_labels_train(self, data_obj, _ext_head_threshold):
        """
            Process labels for train data
            - Remove labels without any training instance
            - Handle multiple centroids
        """
        super()._process_labels_train(data_obj)
        data_obj['ext_head'] = None
        data_obj['multiple_cent_mapping'] = None
        print("Valid labels after processing: ", self.num_labels)
        if self.num_centroids != 1:
            freq = self.labels.frequency()
            self._ext_head = self._get_ext_head(freq, _ext_head_threshold)
            self.multiple_cent_mapping = np.arange(self.num_labels)
            for idx in self._ext_head:
                self.multiple_cent_mapping = np.append(
                    self.multiple_cent_mapping, [idx]*self.num_centroids)
            data_obj['ext_head'] = self._ext_head
            data_obj['multiple_cent_mapping'] = self.multiple_cent_mapping

    def _process_labels_predict(self, data_obj):
        """
            Process labels for predict data
            - Load stats from train set
        """
        super()._process_labels_predict(data_obj)
        try:
            self._ext_head = data_obj['ext_head']
            self.multiple_cent_mapping = data_obj['multiple_cent_mapping']
        except KeyError:
            self._ext_head = None
            self.multiple_cent_mapping = None

    def _process_labels(self, model_dir, _ext_head_threshold=10000):
        """
            Process labels to handle labels without any training instance;
            Handle multiple centroids if required
        """
        data_obj = {}
        fname = os.path.join(
            model_dir, 'labels_params.pkl' if self._split is None else
            "labels_params_split_{}.pkl".format(self._split))
        if self.mode == 'train':
            self._process_labels_train(data_obj, _ext_head_threshold)
            pickle.dump(data_obj, open(fname, 'wb'))
        else:
            data_obj = pickle.load(open(fname, 'rb'))
            self._process_labels_predict(data_obj)

    def get_shortlist(self, index):
        """
            Get data with shortlist for given data index
        """
        pos_labels, _ = self.labels[index]
        return self.shortlist.get_shortlist(index, pos_labels)

    def __getitem__(self, index):
        """
            Get features and labels for index
            Args:
                index: for this sample
            Returns:
                x: 
                y: 
        """
        x = self.features[index]
        y = self.get_shortlist(index)
        return x, y

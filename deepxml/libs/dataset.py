import os
import numpy as np
from .dataset_base import DatasetBase, DatasetTensor
from .dist_utils import Partitioner
from xclib.utils.sparse import _map
from .shortlist_handler import construct_handler


def construct_dataset(data_dir, fname_features, fname_labels, data=None,
                      model_dir='', mode='train', size_shortlist=-1,
                      normalize_features=True, normalize_labels=True,
                      keep_invalid=False, feature_type='sparse',
                      num_clf_partitions=1, feature_indices=None,
                      label_indices=None, shortlist_method='static',
                      shorty=None, surrogate_mapping=None, _type='full',
                      pretrained_shortlist=None):
    if _type == 'full':  # with OVA classifier
        return DatasetFull(
            data_dir, fname_features, fname_labels, data, model_dir, mode,
            feature_indices, label_indices, keep_invalid, normalize_features,
            normalize_labels, num_clf_partitions, feature_type, surrogate_mapping)
    elif _type == 'shortlist':  # with a shortlist
        #  Construct dataset for sparse data
        return DatasetShortlist(
            data_dir, fname_features, fname_labels, data, model_dir, mode,
            feature_indices, label_indices, keep_invalid, normalize_features,
            normalize_labels, num_clf_partitions, size_shortlist,
            feature_type, shortlist_method, shorty, surrogate_mapping,
            pretrained_shortlist=pretrained_shortlist)
    elif _type == 'tensor':
        return DatasetTensor(
            data_dir, fname_features, data, feature_indices,
            normalize_features, feature_type)
    else:
        raise NotImplementedError("Unknown dataset type")


class DatasetFull(DatasetBase):
    """Dataset to load and use XML-Datasets with full output space only

    Arguments
    ---------
    data_dir: str
        data files are stored in this directory
    fname_features: str
        feature file (libsvm or pickle)
    fname_labels: str
        labels file (libsvm or pickle)    
    data: dict, optional, default=None
        Read data directly from this obj rather than files
        Files are ignored if this is not None
        Keys: 'X', 'Y'
    model_dir: str, optional, default=''
        Dump data like valid labels here
    mode: str, optional, default='train'
        Mode of the dataset
    feature_indices: str or None, optional, default=None
        Train with selected features only (read from file)
    label_indices: str or None, optional, default=None
        Train for selected labels only (read from file)
    keep_invalid: bool, optional, default=False
        Don't touch data points or labels
    normalize_features: bool, optional, default=True
        Normalize data points to unit norm
    normalize_lables: bool, optional, default=False
        Normalize labels to convert in probabilities
        Useful in-case on non-binary labels
    num_clf_partitions: int, optional, default=1
        Partition classifier in multiple parts
        Support for multiple GPUs
    feature_type: str, optional, default='sparse'
        sparse or dense features
    label_type: str, optional, default='dense'
        sparse (i.e. with shortlist) or dense (OVA) labels
    surrogate_mapping: str, optional, default=None
        Re-map clusters as per given mapping
        e.g. when labels are clustered
    """

    def __init__(self, data_dir, fname_features, fname_labels, data=None,
                 model_dir='', mode='train', feature_indices=None,
                 label_indices=None, keep_invalid=False,
                 normalize_features=True, normalize_labels=False,
                 num_clf_partitions=1, feature_type='sparse',
                 surrogate_mapping=None, label_type='dense'):
        super().__init__(data_dir, fname_features, fname_labels, data,
                         model_dir, mode, feature_indices, label_indices,
                         keep_invalid, normalize_features, normalize_labels,
                         feature_type, label_type)
        if self.mode == 'train':
            # Remove samples w/o any feature or label
            self._remove_samples_wo_features_and_labels()
        if not keep_invalid and self.labels._valid:
            # Remove labels w/o any positive instance
            self._process_labels(model_dir, surrogate_mapping)
        self.feature_type = feature_type
        self.partitioner = None
        self.num_clf_partitions = 1
        if self.labels._valid:  # If no labels are provided
            self.num_clf_partitions = num_clf_partitions
        if self.mode == 'train':
            assert self.labels._valid, "Labels can not be None while training."
            if self.num_clf_partitions > 1:
                self.partitioner = Partitioner(
                    self.num_labels, self.num_clf_partitions,
                    padding=False, contiguous=True)
                self.partitioner.save(os.path.join(
                    self.model_dir, 'partitionar.pkl'))
        else:
            if self.num_clf_partitions > 1:
                self.partitioner = Partitioner(
                    self.num_labels, self.num_clf_partitions,
                    padding=False, contiguous=True)
                self.partitioner.load(os.path.join(
                    self.model_dir, 'partitionar.pkl'))

        # TODO Take care of this select and padding index
        self.label_padding_index = self.num_labels

    def _process_labels(self, model_dir, surrogate_mapping):
        super()._process_labels(model_dir)
        # if surrogate task is clustered labels
        if surrogate_mapping is not None:
            print("Surrogate mapping is not None, mapping labels")
            surrogate_mapping = np.loadtxt(surrogate_mapping, dtype=np.int)
            _num_labels = len(np.unique(surrogate_mapping))
            mapping = dict(
                zip(range(len(surrogate_mapping)), surrogate_mapping))
            self.labels.Y = _map(self.labels.Y, mapping=mapping,
                                 shape=(self.num_instances, _num_labels),
                                 axis=1)
            self.labels.binarize()

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


class DatasetShortlist(DatasetBase):
    """Dataset to load and use XML-Datasets with shortlist

    Arguments
    ---------
    data_dir: str
        data files are stored in this directory
    fname_features: str
        feature file (libsvm or pickle)
    fname_labels: str
        labels file (libsvm or pickle)    
    data: dict, optional, default=None
        Read data directly from this obj rather than files
        Files are ignored if this is not None
        Keys: 'X', 'Y'
    model_dir: str, optional, default=''
        Dump data like valid labels here
    mode: str, optional, default='train'
        Mode of the dataset
    feature_indices: str or None, optional, default=None
        Train with selected features only (read from file)
    label_indices: str or None, optional, default=None
        Train for selected labels only (read from file)
    keep_invalid: bool, optional, default=False
        Don't touch data points or labels
    normalize_features: bool, optional, default=True
        Normalize data points to unit norm
    normalize_lables: bool, optional, default=False
        Normalize labels to convert in probabilities
        Useful in-case on non-binary labels
    num_clf_partitions: int, optional, default=1
        Partition classifier in multiple
        Support for multiple GPUs
    feature_type: str, optional, default='sparse'
        sparse or dense features
    shortlist_type: str, optional, default='static'
        type of shortlist (static or dynamic)
    shorty: obj, optional, default=None
        Useful in-case of dynamic shortlist
    surrogate_mapping: str, optional, default=None
        Re-map clusters as per given mapping
        e.g. when labels are clustered
    label_type: str, optional, default='dense'
        sparse (i.e. with shortlist) or dense (OVA) labels
    shortlist_in_memory: boolean, optional, default=True
        Keep shortlist in memory if True otherwise keep on disk
    pretrained_shortlist: None or str, optional, default=None
        Pre-trained shortlist (useful in a re-ranker)
    """

    def __init__(self, data_dir, fname_features, fname_labels, data=None,
                 model_dir='', mode='train', feature_indices=None,
                 label_indices=None, keep_invalid=False,
                 normalize_features=True, normalize_labels=False,
                 num_clf_partitions=1, size_shortlist=-1,
                 feature_type='sparse', shortlist_method='static',
                 shorty=None, surrogate_mapping=None, label_type='sparse',
                 shortlist_in_memory=True, pretrained_shortlist=None):
        super().__init__(data_dir, fname_features, fname_labels, data,
                         model_dir, mode, feature_indices, label_indices,
                         keep_invalid, normalize_features, normalize_labels,
                         feature_type, label_type)
        if self.labels is None:
            NotImplementedError(
                "No support for shortlist w/o any label, \
                    consider using dense dataset.")
        self.feature_type = feature_type
        self.num_clf_partitions = num_clf_partitions
        self.shortlist_in_memory = shortlist_in_memory
        self.size_shortlist = size_shortlist
        self.shortlist_method = shortlist_method
        if self.mode == 'train':
            # Remove samples w/o any feature or label
            if pretrained_shortlist is None:
                self._remove_samples_wo_features_and_labels()

        if not keep_invalid:
            # Remove labels w/o any positive instance
            self._process_labels(model_dir, surrogate_mapping)

        self.shortlist = construct_handler(
            shortlist_type=shortlist_method,
            num_instances=self.num_instances,
            num_labels=self.num_labels,
            model_dir=model_dir,
            shorty=shorty,
            mode=mode,
            size_shortlist=size_shortlist,
            label_mapping=None,
            in_memory=shortlist_in_memory,
            corruption=150,
            fname=pretrained_shortlist)
        self.use_shortlist = True if self.size_shortlist > 0 else False
        self.label_padding_index = self.num_labels

    def _process_labels(self, model_dir, surrogate_mapping):
        super()._process_labels(model_dir)
        # if surrogate task is clustered labels
        if surrogate_mapping is not None:
            surrogate_mapping = np.loadtxt(surrogate_mapping, dtype=np.int)
            _num_labels = len(np.unique(surrogate_mapping))
            mapping = dict(
                zip(range(len(surrogate_mapping)), surrogate_mapping))
            self.labels.Y = _map(self.labels.Y, mapping=mapping,
                                 shape=(self.num_instances, _num_labels),
                                 axis=1)
            self.labels.binarize()

    def update_shortlist(self, ind, sim, fname='tmp', idx=-1):
        """Update label shortlist for each instance
        """
        self.shortlist.update_shortlist(ind, sim, fname)

    def save_shortlist(self, fname):
        """Save label shortlist and distance for each instance
        """
        self.shortlist.save_shortlist(fname)

    def load_shortlist(self, fname):
        """Load label shortlist and distance for each instance
        """
        self.shortlist.load_shortlist(fname)

    def get_shortlist(self, index):
        """
            Get data with shortlist for given data index
        """
        pos_labels, _ = self.labels[index]
        return self.shortlist.get_shortlist(index, pos_labels)

    def __getitem__(self, index):
        """Get features and labels for index
        Arguments
        ---------
        index: int
            data for this index
        Returns
        -------
        features: np.ndarray or tuple
            for dense: np.ndarray
            for sparse: feature indices and their weights
        labels: tuple
            shortlist: label indices in the shortlist
            labels_mask: 1 for relevant; 0 otherwise
            dist: distance (used during prediction only)
        """
        x = self.features[index]
        y = self.get_shortlist(index)
        return x, y

import torch
import pickle
import os
import numpy as np
from .features import construct as construct_f
from .labels import construct as construct_l


class DatasetTensor(torch.utils.data.Dataset):
    """Dataset to load and use sparse/dense matrix
    * Support npz, pickle, npy or libsvm file format
    * Useful when iterating over features or labels

    Arguments
    ---------
    data_dir: str
        data files are stored in this directory
    fname: str
        file name file (libsvm or npy or npz or pkl)
        will use 'X' key in case of pickle
    data: scipy.sparse or np.ndarray, optional, default=None
        Read data directly from this obj rather than files
        Files are ignored if this is not None
    indices: None or str, optional, default=None
        Use only these indices in the given list
    normalize: bool, optional, default=True
        Normalize the rows to unit norm
    _type: str, optional, default='sparse'
        Type of data (sparse/dense)
    """

    def __init__(self, data_dir, fname, data=None, indices=None,
                 normalize=True, _type='sparse'):
        self.data = self.construct(
            data_dir, fname, data, indices, normalize, _type)

    def construct(self, data_dir, fname, data, indices, normalize, _type):
        data = construct_f(data_dir, fname, data, normalize, _type)
        if indices is not None:
            indices = np.loadtxt(indices, dtype=np.int64)
            data._index_select(indices)
        return data

    def __len__(self):
        return self.num_instances

    @property
    def num_instances(self):
        return self.data.num_instances

    def __getitem__(self, index):
        """Get data for a given index
        Arguments
        ---------
        index: int
            data for this index
        Returns
        -------
        features: tuple
            feature indices and their weights
        """
        return self.data[index]


class DatasetBase(torch.utils.data.Dataset):
    """Dataset to load and use XML-Datasets
    Support pickle or libsvm file format

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
    feature_type: str, optional, default='sparse'
        sparse or dense features
    label_type: str, optional, default='dense'
        sparse (i.e. with shortlist) or dense (OVA) labels
    """

    def __init__(self, data_dir, fname_features, fname_labels,
                 data=None, model_dir='', mode='train',
                 feature_indices=None, label_indices=None,
                 keep_invalid=False, normalize_features=True,
                 normalize_lables=False, feature_type='sparse',
                 label_type='dense'):
        self.data_dir = data_dir
        self.mode = mode
        self.features, self.labels = self.load_data(
            data_dir, fname_features, fname_labels, data,
            normalize_features, normalize_lables,
            feature_type, label_type)
        self._split = None
        self.index_select(feature_indices, label_indices)
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.label_padding_index = self.num_labels

    def _remove_samples_wo_features_and_labels(self):
        """Remove instances if they don't have any feature or label
        """
        indices = self.features.get_valid_indices(axis=1)
        if self.labels is not None:
            indices_labels = self.labels.get_valid_indices(axis=1)
            indices = np.intersect1d(indices, indices_labels)
            self.labels._index_select(indices, axis=0)
        self.features._index_select(indices, axis=0)

    def index_select(self, feature_indices, label_indices):
        """Transform feature and label matrix to specified
        features/labels only
        """
        def _get_split_id(fname):
            """Split ID (or quantile) from file name
            """
            idx = fname.split("_")[-1].split(".")[0]
            return idx
        if label_indices is not None:
            self._split = _get_split_id(label_indices)
            label_indices = np.loadtxt(label_indices, dtype=np.int32)
            self.labels._index_select(label_indices, axis=1)
        if feature_indices is not None:
            self._split = _get_split_id(feature_indices)
            feature_indices = np.loadtxt(feature_indices, dtype=np.int32)
            self.features._index_select(feature_indices, axis=1)

    def load_features(self, data_dir, fname, X,
                      normalize_features, feature_type):
        """Load features from given file
        Features can also be supplied directly
        """
        return construct_f(data_dir, fname, X,
                           normalize_features, feature_type)

    def load_labels(self, data_dir, fname, Y, normalize_labels, label_type):
        """Load labels from given file
        Labels can also be supplied directly
        """
        labels = construct_l(data_dir, fname, Y, normalize_labels,
                             label_type)  # Pass dummy labels if required
        if normalize_labels:
            if self.mode == 'train':  # Handle non-binary labels
                print("Non-binary labels encountered in train; Normalizing.")
                labels.normalize(norm='max', copy=False)
            else:
                print("Non-binary labels encountered in test/val; Binarizing.")
                labels.binarize()
        return labels

    def load_data(self, data_dir, fname_f, fname_l, data,
                  normalize_features=True, normalize_labels=False,
                  feature_type='sparse', label_type='dense'):
        """Load features and labels from file in libsvm format or pickle
        """
        features = self.load_features(
            data_dir, fname_f, data['X'], normalize_features, feature_type)
        labels = self.load_labels(
            data_dir, fname_l, data['Y'], normalize_labels, label_type)
        return features, labels

    @property
    def num_instances(self):
        return self.features.num_instances

    @property
    def num_features(self):
        return self.features.num_features

    @property
    def num_labels(self):
        return self.labels.num_labels

    def get_stats(self):
        """Get dataset statistics
        """
        return self.num_instances, self.num_features, self.num_labels

    def _process_labels_train(self, data_obj):
        """Process labels for train data
            - Remove labels without any training instance
        """
        data_obj['num_labels'] = self.num_labels
        valid_labels = self.labels.remove_invalid()
        data_obj['valid_labels'] = valid_labels

    def _process_labels_predict(self, data_obj):
        """Process labels for test data
           Only use valid labels i.e. which had atleast one training
           example
        """
        valid_labels = data_obj['valid_labels']
        self.labels._index_select(valid_labels, axis=1)

    def _process_labels(self, model_dir):
        """Process labels to handle labels without any training instance;
        """
        data_obj = {}
        fname = os.path.join(
            model_dir, 'labels_params.pkl' if self._split is None else
            "labels_params_split_{}.pkl".format(self._split))
        if self.mode == 'train':
            self._process_labels_train(data_obj)
            pickle.dump(data_obj, open(fname, 'wb'))
        else:
            data_obj = pickle.load(open(fname, 'rb'))
            self._process_labels_predict(data_obj)

    def __len__(self):
        return self.num_instances

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
        labels: np.ndarray
            1 when relevant; 0 otherwise
        """
        x = self.features[index]
        y = self.labels[index]
        return x, y

from sklearn.preprocessing import normalize as scale
import numpy as np
import _pickle as pickle
from xclib.data import data_utils
import os
import re
import json


def construct(data_dir, fname, X=None, normalize=False, _type='sparse'):
    if _type == 'sparse':
        return SparseFeatures(data_dir, fname, X, normalize)
    elif _type == 'dense':
        return DenseFeatures(data_dir, fname, X, normalize)
    elif _type == 'sequential':
        return SequentialFeatures(data_dir, fname, X)
    else:
        raise NotImplementedError("Unknown feature type")


class FeaturesBase(object):
    """
        Base class for features
        Args:
            data_dir: str: data directory
            fname: str: load data from this file
            X: np.ndarray or csr_matrix: data is already provided
    """
    def __init__(self, data_dir, fname, X=None):
        self.X = self.load(data_dir, fname, X)

    def frequency(self, axis=0):
        return np.array(self.X.sum(axis=axis)).ravel()

    def get_invalid(self, axis=0):
        return np.where(self.frequency(axis)==0)[0]

    def get_valid(self, axis=0):
        return np.where(self.frequency(axis)>0)[0]

    def remove_invalid(self, axis=0):
        indices = self.get_valid(axis)
        self.index_select(indices)
        return indices

    def _select_instances(self, indices):
        self.X = self.X[indices]

    def _select_features(self, indices):
        # Not valid in general case
        pass

    def index_select(self, indices, axis=1, fname=None):
        """
            Choose only selected labels or instances
        """
        #TODO: Load and select from file
        if axis == 0:
            self._select_instances(indices)
        elif axis == 1:
            self._select_features(indices)
        else:
            raise NotImplementedError("Unknown Axis.")

    def load(self, data_dir, fname, X):
        if X is not None:
            return X
        else:
            assert fname is not None, "Filename can not be None."
            fname = os.path.join(data_dir, fname)
            if fname.lower().endswith('.pkl'):
                return pickle.load(open(fname, 'rb'))['X']
            elif fname.lower().endswith('.txt'):
                return data_utils.read_sparse_file(fname, dtype=np.float32, force_header=True)
            else:
                raise NotImplementedError("Unknown file extension")

    @property
    def num_instances(self):
        return self.X.shape[0]

    @property
    def num_features(self):
        return self.X.shape[1]

    @property
    def shape(self):
        return (self.num_instances, self.num_features)

    def __getitem__(self, index):
        return self.X[index]


class DenseFeatures(FeaturesBase):
    """
        Class for dense features
        Args:
            data_dir: str: data directory
            fname: str: load data from this file
            X: np.ndarray: data is already provided
    """
    def __init__(self, data_dir, fname, X=None, normalize=False):
        super().__init__(data_dir, fname, X)

    def _select_features(self, indices):
        self.X = self.X[:, indices]

    def normalize(self, norm='l2', copy=False):
        self.X = scale(self.X, copy=copy, norm=norm)

    def frequency(self, axis=0):
        return np.array(self.X.astype(np.bool).sum(axis=axis)).ravel()


class SparseFeatures(FeaturesBase):
    """
        Class for sparse features
        Args:
            data_dir: str: data directory
            fname: str: load data from this file
            X: csr_matrix: data is already provided
    """
    def __init__(self, data_dir, fname, X=None, normalize=False):
        super().__init__(data_dir, fname, X)
        if normalize:
            self.normalize()

    def normalize(self, norm='l2', copy=False):
        self.X = scale(self.X, copy=copy, norm=norm)

    def _select_features(self, indices):
        self.X = self.X[:, indices]

    def frequency(self, axis=0):
        return np.array(self.X.astype(np.bool).sum(axis=axis)).ravel()

    def __getitem__(self, index):
        x = list(map(lambda item: item+1, self.X[index, :].indices)) # Treat idx:0 as Padding
        w = self.X[index, :].data.tolist()
        return x, w


class SequentialFeatures(FeaturesBase):
    """
        Class for Sequential features; useful for sequential models
        Args:
            data_dir: str: data directory
            fname: str: load data from this file
            X: ?: ?
        * 0: Reserved for padding index; 1: UNK, 2: Start token and 3: end token
    """
    def __init__(self, data_dir, fname, X=None, vocabulary_file=None, cutoff_len=300):
        self.load(data_dir, fname)
        self.cutoff_len = cutoff_len
        if vocabulary_file is not None:
            self.token_to_index = json.load(open(os.path.join(data_dir, vocabulary_file)))
            self._check_pad_index()
            self.process_text() #Assumes text is already vectorized if no vocab file is supplied

    def _check_pad_index(self):
        if '<PAD>' not in self.token_to_index:
            self.token_to_index = {k:v+1 for k,v in self.token_to_index.items()}
            self.token_to_index['<PAD>'] = 0

    @property
    def num_instances(self):
        return len(self.X)

    @property
    def num_features(self):
        return len(self.token_to_index)+1

    def process_text(self):
        self.X = [self.convert(item) for item in self.X]

    def load(self, data_dir, fname):
        with open(os.path.join(data_dir, fname), 'r', encoding='latin') as fp:
            self.X = fp.readlines()

    def _clean_text(self, sentence):
        sentence = sentence.lower().strip()
        sentence = re.sub(r"([.!?])", r" \1", sentence)
        sentence = re.sub(r"[^a-zA-Z0-9.!?]+", r" ", sentence)
        return sentence

    def convert(self, sentence):
        # Assuming there is no need to truncate as of now
        sentence = ['<S>'] + self._clean_text(sentence).split(" ")[:self.cutoff_len] + ["</S>"]
        return self.map_to_indices(sentence)

    def map_to_indices(self, sentence):
        # +1 for Padding
        return list(map(lambda x: self.token_to_index[x] if x in self.token_to_index else self.token_to_index['<UNK>'], sentence))

    def __getitem__(self, index):
        return self.X[index]
 
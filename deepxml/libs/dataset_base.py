import torch
import _pickle as pickle
import os
import sys
from scipy.sparse import lil_matrix
import numpy as np
from sklearn.preprocessing import normalize
import xclib.data.data_utils as data_utils


class DatasetBase(torch.utils.data.Dataset):
    """
        Dataset to load and use XML-Datasets
    """

    def __init__(self, data_dir, fname, data=None, model_dir='', mode='train', size_shortlist=-1,
                label_indices=None, keep_invalid=False, num_centroids=1):
        """
            Support pickle or libsvm file format
            Args:
                data_file: str: File name for the set
        """
        self.data_dir = data_dir
        fname = os.path.join(data_dir, fname)
        self.features, self.labels, self.num_samples, \
            self.num_features, self.num_labels = self.load_data(fname, data)
        self._split = None
        self._sel_labels(label_indices)
        self.mode = mode
        self._ext_head = None
        self.data_dir = data_dir
        self.num_centroids = num_centroids  # Use multiple centroids for ext head labels
        self.multiple_cent_mapping = None
        self.shortlist = None
        self.dist = None
        self.size_shortlist = size_shortlist
        self.use_shortlist = True if self.size_shortlist > 0 else False
        if not keep_invalid:
            self._process_labels(model_dir)
        self.label_padding_index = self.num_labels

    def _sel_labels(self, label_indices):
        """
            Train only for given labels
        """
        def _get_split_id(fname):
            idx = fname.split("_")[-1].split(".")[0]
            return idx
        if label_indices is not None:
            self._split = _get_split_id(label_indices)
            label_indices = np.loadtxt(label_indices, dtype=np.int32)
            self.labels = self.labels[:, label_indices]
            self.num_labels = self.labels.shape[1]

    def load_data(self, fname, data):
        """
            Load features and labels from file in libsvm format
        """
        if data is not None:
            features = data['features']
            labels = data['labels']
            num_samples, num_features = self.features.shape
            num_labels = labels.shape[1]
        else:
            # TODO: load data from pickle file
            features, labels, num_samples, num_features, num_labels = data_utils.read_data(
                fname)
            labels = data_utils.binarize_labels(labels, num_labels)
        return features, labels, num_samples, num_features, num_labels

    def get_stats(self):
        """
            Get dataset statistics
        """
        return self.num_samples, self.num_features, self.num_labels

    def _pad_seq(self, indices, dist):
        _pad_length = self.size_shortlist - len(indices)
        indices.extend([self.label_padding_index]*_pad_length)
        dist.extend([100]*_pad_length)

    def _remap_multiple_centroids(self, indices, vals, _func=min, _limit=1e5):
        """
            Remap multiple centroids to original labels
        """
        indices = np.asarray(
            list(map(lambda x: self.multiple_cent_mapping[x], indices)))
        _dict = dict({})
        for id, ind in enumerate(indices):
            _dict[ind] = _func(_dict.get(ind, _limit), vals[id])
        indices, values = zip(*_dict.items())
        indices, values = list(indices), list(values)
        if len(indices) < self.size_shortlist:
            self._pad_seq(indices, values)
        return indices, values

    def _get_label_freq(self):
        # Can handle non-binary labels
        return np.array(self.labels.astype(np.bool).sum(axis=0)).ravel()

    def _get_ext_head(self, freq, threshold):
        return np.where(freq >= threshold)[0]

    def _process_labels_train(self, data_obj, _ext_head_threshold):
        """
            Process labels for train data
            - Remove labels without any training instance
            - Handle multiple centroids
        """
        valid_labels = np.where(np.array(self.labels.sum(axis=0)) != 0)[1]
        data_obj['valid_labels'] = valid_labels
        data_obj['num_labels'] = self.num_labels
        data_obj['ext_head'] = None
        data_obj['multiple_cent_mapping'] = None
        self.labels = self.labels[:, valid_labels]
        self.num_labels = valid_labels.size
        print("Valid labels after processing: ", self.num_labels)
        if self.num_centroids != 1:
            freq = self._get_label_freq()
            self._ext_head = self._get_ext_head(freq, _ext_head_threshold)
            self.multiple_cent_mapping = np.arange(self.num_labels)
            for idx in self._ext_head:
                self.multiple_cent_mapping = np.append(
                    self.multiple_cent_mapping, [idx]*self.num_centroids)
            data_obj['ext_head'] = self._ext_head
            data_obj['multiple_cent_mapping'] = self.multiple_cent_mapping

    def _process_labels_retrain(self, data_obj, _ext_head_threshold):
        """
            Process labels for re-train data
            - Remove labels without any training instance
            - Handle multiple centroids
        """
        valid_labels = data_obj['valid_labels']
        self.labels = self.labels[:, valid_labels]
        self.num_labels = valid_labels.size
        if self.num_centroids != 1:
            freq = self._get_label_freq()
            self._ext_head = self._get_ext_head(freq, _ext_head_threshold)
            self.multiple_cent_mapping = np.arange(self.num_labels)
            for idx in self._ext_head:
                self.multiple_cent_mapping = np.append(
                    self.multiple_cent_mapping, [idx]*self.num_centroids)
            data_obj['ext_head'] = self._ext_head
            data_obj['multiple_cent_mapping'] = self.multiple_cent_mapping
            print("Ext labels: ", self._ext_head)

    def _process_labels_predict(self, data_obj):
        """
            Process labels for predict data
            - Load stats from train set
        """
        self._ext_head = data_obj['ext_head']
        self.multiple_cent_mapping = data_obj['multiple_cent_mapping']
        valid_labels = data_obj['valid_labels']
        self.labels = self.labels[:, valid_labels]
        self.num_labels = valid_labels.size

    def _process_labels(self, model_dir, _ext_head_threshold=10000):
        """
            Process labels to handle labels without any training instance;
            Handle multiple centroids if required
        """
        data_obj = {}
        fname = os.path.join(
            model_dir, 'labels_params.pkl' if self._split is None else "labels_params_split_{}.pkl".format(self._split))
        if self.mode == 'train':
            self._process_labels_train(data_obj, _ext_head_threshold)
            pickle.dump(data_obj, open(fname, 'wb'))
        elif self.mode == 'retrain_w_shorty':
            data_obj = pickle.load(open(fname, 'rb'))
            self._process_labels_retrain(data_obj, _ext_head_threshold)
            pickle.dump(data_obj, open(fname, 'wb'))
        elif self.mode == 'predict':
            data_obj = pickle.load(open(fname, 'rb'))
            self._process_labels_predict(data_obj)
        else:
            raise NotImplementedError("Unknown mode!")

    def update_shortlist(self, shortlist, dist):
        """
            Update label shortlist for each instance
        """
        self.shortlist = shortlist
        self.dist = dist

    def save_shortlist(self, fname):
        """
            Save label shortlist and distance for each instance
        """
        pickle.dump({"shortlist": self.shortlist,
                     "dist": self.dist}, open(fname, 'wb'))

    def load_shortlist(self, fname):
        """
            Load label shortlist and distance for each instance
        """
        _temp = pickle.load(open(fname, 'rb'))
        self.shortlist = _temp['shortlist']
        self.dist = _temp['dist']

    def _adjust_shortlist(self, pos_labels, shortlist, dist):
        """
            Adjust shortlist for a instance
            Training: Add positive labels to the shortlist
            Inference: Return shortlist with label mask
        """
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

    def __len__(self):
        return self.num_samples


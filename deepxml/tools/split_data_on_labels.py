import numpy as np
from xclib.utils.sparse import binarize


class splitData(object):
    def __init__(self, split_method=0, num_splits=2, threshold=[2]):
        """
            Create data splits based on label splits
            Args:
                split_method: int: 0(based on frequency), 1(based on percentage), 2(based on absolute numbers)
                num_splits: int: split labels in these many parts
                threshold: list of int/float: Should contain num_split-1 entries
        """
        self.split_method = split_method
        self.num_splits = num_splits
        self.threshold = threshold
        self.labels_split = []
        self.num_valid_labels = []
        self.features_split = [] # Active features in each split

    def remove_documents_wo_features(self, features, labels):
        def _compute_freq(data):
            return np.array(data.sum(axis=1)).ravel()
        freq = _compute_freq(features)
        indices = np.where(freq > 0)[0]
        features = features[indices]
        labels = labels[indices]
        return features, labels


    def get_valid(self, freq):
        return np.where(freq != 0)[0]

    def split_based_on_topk(self, labels):
        """
            Split labels based on frequency
        """
        freq = self.get_frequency(labels)
        num_labels = freq.size
        _sorted_indices = np.argsort(freq)
        low = 0
        for idx in range(self.num_splits):
            if idx != self.num_splits-1:
                high = num_labels - self.threshold[idx]
                current_split = _sorted_indices[low:high] 
                low = high
            else:
                current_split = _sorted_indices[low:]
            print("#: ", current_split.size, np.sum(freq[current_split]>0))
            self.num_valid_labels.append(np.sum(freq[current_split]>0))
            current_split.sort() # Sort indices within the split
            self.labels_split.append(current_split)

    def split_based_on_frequency(self, labels):
        """
            Split labels based on frequency
        """
        freq = self.get_frequency(labels)
        print("# Freq: ", np.sum(freq==0))
        low = 0
        for idx in range(self.num_splits):
            if idx != self.num_splits-1:
                high = self.threshold[idx]
                current_split = np.where(np.logical_and(freq>=low, freq<high))[0]
                low = high
            else:
                current_split = np.where(freq>=low)[0]
            print("#: ", current_split.size, np.sum(freq[current_split]>0))
            self.num_valid_labels.append(np.sum(freq[current_split]>0))
            self.labels_split.append(current_split)

    def split_labels(self, labels):
        # Assumes invalid labels are already removed
        if self.split_method == 0:
            self.split_based_on_frequency(labels)
        elif self.split_method == 1:
            self.split_based_on_topk(labels)
        else:
            pass

    def split_features(self, features, labels):
        for idx in range(self.num_splits):
            temp_features = features.copy()
            temp_labels = labels[:, self.labels_split[idx]]
            temp_features = temp_features.T.dot(temp_labels)
            temp = np.array(temp_features.sum(axis=1)).ravel() 
            temp = np.where(temp>0)[0]
            self.features_split.append(temp)       

    def split(self, features, labels):
        self.split_labels(labels)
        self.split_features(features, labels)

    def get_frequency(self, data):
        """
            Get frequency of data such as labels
        """
        data = binarize(data, copy=True) #Useful in case of non-binary labels
        freq = np.array(data.sum(axis=0)).ravel()
        return freq

    def fit(self, features, labels, remove_invalid_on_features=True, remove_invalid_on_labels=False):
        if remove_invalid_on_features:
            features, labels = self.remove_documents_wo_features(features, labels)
        if remove_invalid_on_labels:
            #Process data to remove invalid labels
            features, labels = self._process(features.copy(), labels.copy(), _set='train')
        #Split labels in different partitions based on defined strategy
        self.split(features, labels)
    
    def remove_invalid_samples(self, features, labels):
        num_samples, num_feat = features.shape
        valid_based_on_feat = np.where(np.array(features.sum(axis=1)).reshape(num_samples)>0)[0]
        valid_based_on_labels = np.where(np.array(labels.sum(axis=1)).reshape(num_samples)>0)[0]
        return np.intersect1d(valid_based_on_feat, valid_based_on_labels)

    def transform(self, features, labels):
        features, labels = self._process(features.copy(), labels.copy(), _set='test')
        data_split = []
        valid_indices = []
        for idx in range(self.num_splits):
            feat = features[:, self.features_split[idx]]
            lab = labels[:, self.labels_split[idx]]
            v_indices = self.remove_invalid_samples(feat, lab)
            valid_indices.append(v_indices)
            data_split.append((feat[v_indices, :], lab[v_indices, :]))
        return data_split, valid_indices

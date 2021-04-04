import numpy as np
from xclib.utils.sparse import binarize, normalize, compute_centroid
import functools
import xclib.data.data_utils as data_utils
from xclib.utils.graph import RandomWalk
from xclib.utils.clustering import cluster_balance
from xclib.utils.clustering import b_kmeans_sparse, b_kmeans_dense
import os
import json


def compute_correlation(Y, walk_to=50, p_reset=0.2, k=10):
    rw = RandomWalk(Y)
    return rw.simulate(walk_to=walk_to, p_reset=p_reset, k=k)


class SurrogateMapping(object):
    """
    Generate mapping of labels for surrogate task

    Arguments:
    ----------
    method: int, optional (default: 0)
        - 0 none (use extreme task)
        - 1 cluster labels & treat clusters as new labels
        - 2 pick topk labels based on given label frequency
        - 3 pick topk labels

    threshold: int, optional (default: 65536)
        - method 0: none
        - method 1: number of clusters
        - method 2: label frequency; pick labels more with
                    frequency more than given value
        - method 3: #labels to pick
    """
    def __init__(self, method=0, threshold=65536, feature_type='sparse'):
        self.feature_type = feature_type
        self.method = method
        self.threshold = threshold

    def map_on_cluster(self, features, labels):
        label_centroids = compute_centroid(features, labels)
        cooc = normalize(compute_correlation(labels), norm="l1")
        if self.feature_type == 'sparse':
            freq = labels.getnnz(axis = 0)
            if freq.max() > 5000:
                print("Correlation matrix is too dense. Skipping..")
            else:
                label_centroids = cooc.dot(label_centroids)    
            splitter=functools.partial(b_kmeans_sparse)
        elif self.feature_type == 'dense':
            label_centroids = cooc.dot(label_centroids)
            splitter=functools.partial(b_kmeans_dense)
        else:
            raise NotImplementedError("Unknown feature type!")
        _, self.mapping = cluster_balance(
            X=label_centroids,
            clusters=[np.asarray(np.arange(labels.shape[1]), dtype=np.int64)],
            num_clusters=self.threshold,
            splitter=splitter)
        self.num_surrogate_labels = self.threshold

    def map_on_frequency(self, labels):
        raise NotImplementedError("")

    def map_on_topk(self, labels):
        raise NotImplementedError("")

    def remove_documents_wo_features(self, features, labels):
        if isinstance(features, np.ndarray):
            features = np.power(features, 2)
        else:
            features = features.power(2)
        freq = np.array(features.sum(axis=1)).ravel()
        indices = np.where(freq > 0)[0]
        features = features[indices]
        labels = labels[indices]
        return features, labels

    def map_none(self):
        self.num_surrogate_labels = len(self.valid_labels)
        self.mapping = self.valid_labels

    def gen_mapping(self, features, labels):
        # Assumes invalid labels are already removed
        if self.method == 0:
            self.map_none()    
        elif self.method == 1:
            self.map_on_cluster(features, labels)    
        elif self.method == 1:
            self.map_on_frequency(labels)
        elif self.method == 2:
            self.map_on_topk(labels)
        else:
            pass

    def get_valid_labels(self, labels):
        freq = np.array(labels.sum(axis=0)).ravel()
        ind = np.where(freq > 0)[0]
        return labels[:, ind], ind

    def fit(self, features, labels):
        self.num_labels = labels.shape[1]
        # Remove documents w/o any feature
        # these may impact the count, if not removed
        features, labels = self.remove_documents_wo_features(features, labels)
        # keep only valid labels; main code will also remove invalid labels
        labels, self.valid_labels = self.get_valid_labels(labels)
        self.gen_mapping(features, labels)


def run(feat_fname, lbl_fname, feature_type, method, threshold, seed, tmp_dir):
    np.random.seed(seed)
    if feature_type == 'dense':
        features = data_utils.read_gen_dense(feat_fname)
    elif feature_type == 'sparse':
        features = data_utils.read_gen_sparse(feat_fname)
    else:
        raise NotImplementedError()
    labels = data_utils.read_sparse_file(lbl_fname)
    assert features.shape[0] == labels.shape[0], \
        "Number of instances must be same in features and labels"
    num_features = features.shape[1]
    stats_obj = {}
    stats_obj['threshold'] = threshold
    stats_obj['method'] = method

    sd = SurrogateMapping(
        method=method, threshold=threshold, feature_type=feature_type)
    sd.fit(features, labels)
    stats_obj['surrogate'] = "{},{},{}".format(
        num_features, sd.num_surrogate_labels, sd.num_surrogate_labels)
    stats_obj['extreme'] = "{},{},{}".format(
        num_features, sd.num_labels, len(sd.valid_labels))

    json.dump(stats_obj, open(
        os.path.join(tmp_dir, "data_stats.json"), 'w'), indent=4)

    np.savetxt(os.path.join(tmp_dir, "valid_labels.txt"),
               sd.valid_labels, fmt='%d')
    np.savetxt(os.path.join(tmp_dir, "surrogate_mapping.txt"),
               sd.mapping, fmt='%d')

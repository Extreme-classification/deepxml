import numpy as np
from xclib.utils.sparse import binarize, normalize
import functools
import operator
from multiprocessing import Pool
from sklearn.cluster import KMeans
import time
import xclib.data.data_utils as data_utils
import os
import json


def compute_centroid(X, Y):
    return Y.T.dot(X).tocsr()


def balanced_kmeans(labels_features, index, metric='cosine', tol=1e-4,
                    leakage=None):
    if labels_features.shape[0] == 1:
        return [index]
    cluster = np.random.randint(low=0, high=labels_features.shape[0], size=(2))
    while cluster[0] == cluster[1]:
        cluster = np.random.randint(
            low=0, high=labels_features.shape[0], size=(2))
    _centeroids = labels_features[cluster].todense()
    _similarity = _sdist(labels_features, _centeroids,
                         metric=metric, norm='l2')
    old_sim, new_sim = -1000000, -2
    while new_sim - old_sim >= tol:
        clustered_lbs = np.array_split(
            np.argsort(_similarity[:, 1]-_similarity[:, 0]), 2)
        _centeroids = np.vstack([
            labels_features[x, :].mean(
                axis=0) for x in clustered_lbs
        ])
        _similarity = _sdist(labels_features, _centeroids,
                             metric=metric, norm='l2')
        old_sim, new_sim = new_sim, np.sum(
            [np.sum(
                _similarity[indx, i]
            ) for i, indx in enumerate(clustered_lbs)])

    if leakage is not None:
        _distance = 1-_similarity
        # Upper boundary under which labels will co-exists
        ex_r = [(1+leakage)*np.max(_distance[indx, i])
                for i, indx in enumerate(clustered_lbs)]
        """
        Check for labels in 2nd cluster who are ex_r_0 closer to
        1st Cluster and append them in first cluster
        """
        clustered_lbs = list(
            map(lambda x: np.concatenate(
                [clustered_lbs[x[0]],
                 x[1][_distance[x[1], x[0]] <= ex_r[x[0]]]
                 ]),
                enumerate(clustered_lbs[::-1])
                )
        )
    return list(map(lambda x: index[x], clustered_lbs))


def _sdist(XA, XB, metric, norm=None):
    if norm is not None:
        XA = normalize(XA, norm)
        XB = normalize(XB, norm)
    if metric == 'cosine':
        score = XA.dot(XB.transpose())

    if metric == 'sigmoid':
        score = 2/(1 + np.exp(-XA.dot(XB.transpose())))-1
    return score


def cluster_labels(labels, clusters, num_nodes, splitter, num_threads=10):
    with Pool(num_threads) as p:
        while len(clusters) != num_nodes:
            start_time = time.time()
            temp_cluster_list = functools.reduce(
                operator.iconcat,
                p.starmap(
                    splitter,
                    map(lambda cluster: (labels[cluster], cluster),
                        clusters)
                ), [])
            end_time = time.time()
            print("Total clusters {}; Avg cluster size: {}; "
                  "Time taken (sec): {}".format(
                    len(temp_cluster_list),
                    np.mean(list(map(len, temp_cluster_list))),
                    end_time-start_time))
            clusters = temp_cluster_list
            del temp_cluster_list
    mapping = {}
    for idx, item in enumerate(clusters):
        for _item in item:
            mapping[_item] = idx
    return clusters, mapping


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
    def __init__(self, method=0, threshold=65536):
        self.method = method
        self.threshold = threshold

    def map_on_cluster(self, features, labels):
        label_centroids = compute_centroid(features, labels)
        cooc = normalize(labels.T.dot(labels).tocsr(), norm='l1')
        label_centroids = cooc.dot(label_centroids)
        _, mapping = cluster_labels(
            labels=label_centroids,
            clusters=[np.asarray(np.arange(labels.shape[1]), dtype=np.int64)],
            num_nodes=self.threshold,
            splitter=functools.partial(balanced_kmeans))
        self.mapping = [None]*len(mapping)
        self.num_surrogate_labels = self.threshold
        for key, val in mapping.items():
            self.mapping[key] = val

    def map_on_frequency(self, labels):
        raise NotImplementedError("")

    def map_on_topk(self, labels):
        raise NotImplementedError("")

    def remove_documents_wo_features(self, features, labels):
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


def run(feat_fname, lbl_fname, method, threshold, seed, tmp_dir):
    np.random.seed(seed)
    features = data_utils.read_sparse_file(feat_fname)
    labels = data_utils.read_sparse_file(lbl_fname)
    assert features.shape[0] == labels.shape[0], \
        "Number of instances must be same in features and labels"
    num_features = features.shape[1]
    stats_obj = {}
    stats_obj['threshold'] = threshold
    stats_obj['method'] = method

    sd = SurrogateMapping(method=method, threshold=threshold)
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

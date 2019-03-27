from sklearn.preprocessing import normalize
import numpy as np
import libs.utils as utils
from libs.clustering import Cluster


def get_multiple_centroids(_ext_indices, num_centroids, features, labels):
    embedding_dims = features.shape[1]
    _cluster_obj = Cluster(indices=_ext_indices, embedding_dims=embedding_dims,
                           num_clusters=num_centroids, max_iter=50, n_init=2, num_threads=-1)
    _cluster_obj.fit(features, labels)
    return _cluster_obj.predict()


def get_shortlist(document_embeddings, shorty):
    short, distances = shorty.query(document_embeddings)
    return short, distances


def update(data_loader, model, embedding_dim, shorty, flag=0):
    # 0: train and update, 1: train, 2: update
    num_centroids = data_loader.dataset.num_centroids
    doc_embeddings = model.get_document_embeddings(data_loader)
    # Do not normalize if kmeans clustering needs to be done!
    # doc_embeddings = normalize(doc_embeddings, copy=False)
    if flag == 0:
        # train and update shortlist
        label_embeddings = utils.get_label_embeddings(
            doc_embeddings, data_loader.dataset.labels)
        if num_centroids != 1:
            extra_label_embeddings = get_multiple_centroids(
                data_loader.dataset._ext_head, num_centroids, doc_embeddings, data_loader.dataset.labels)
            label_embeddings = np.vstack([label_embeddings, extra_label_embeddings])
        print("Label embedding shape: ", label_embeddings.shape)
        # label_embeddings = normalize(label_embeddings, copy=False)
        shorty.train(label_embeddings)
        short, dist = get_shortlist(doc_embeddings, shorty)
        data_loader.dataset.update_shortlist(short, dist)
    elif flag == 1:
        # train and don't get shortlist
        label_embeddings = utils.get_label_embeddings(
            doc_embeddings, data_loader.dataset.labels)
        if num_centroids != 1:
            print("Clustering labels!")
            extra_label_embeddings = get_multiple_centroids(
                data_loader.dataset._ext_head, num_centroids, doc_embeddings, data_loader.dataset.labels)
            label_embeddings = np.vstack([label_embeddings, extra_label_embeddings])
        print("Label embedding shape: ", label_embeddings.shape)
        # label_embeddings = normalize(label_embeddings, copy=False)
        shorty.train(label_embeddings)
    else:
        # get shortlist
        short, dist = get_shortlist(doc_embeddings, shorty)
        data_loader.dataset.update_shortlist(short, dist)
    return doc_embeddings

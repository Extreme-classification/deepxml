from sklearn.preprocessing import normalize
import numpy as np
import libs.utils as utils


def get_and_update_shortlist(document_embeddings, shorty,
                             data_loader, _save_mem=True):
    # FIXME: Figure out a way to delete document embeddings
    if not hasattr(shorty, 'num_graphs'):
        _save_mem = False
    if _save_mem:  # Fetch one-by-one; save to disk and delete
        for idx in range(shorty.num_graphs):
            short, distances = shorty.query(document_embeddings, idx)
            data_loader.dataset.update_shortlist(short, distances, idx=idx)
    else:  # Fetch shortlist at once
        short, distances = shorty.query(document_embeddings)
        data_loader.dataset.update_shortlist(short, distances)


def compute_label_embeddings(doc_embeddings, data_loader, num_graphs):
    # Compute label embeddings for single or multiple graphs
    if num_graphs == 1:
        return utils.get_label_embeddings(
            doc_embeddings, data_loader.dataset.labels)
    else:
        out = []
        for idx in range(num_graphs):
            _l_indices = data_loader.dataset.shortlist.get_partition_indices(
                idx)
            # TODO: See if there is some better way to handle this
            out.append(utils.get_label_embeddings(
                doc_embeddings, data_loader.dataset.labels.data[:, _l_indices]))
        return out


def update(data_loader, model, embedding_dim, shorty, flag=0,
           num_graphs=1, use_coarse=False):
    # 0: train and update, 1: train, 2: update
    doc_embeddings = model._document_embeddings(
        data_loader, return_coarse=use_coarse)
    # Do not normalize if kmeans clustering needs to be done!
    # doc_embeddings = normalize(doc_embeddings, copy=False)
    if flag == 0:
        shorty.fit(doc_embeddings, data_loader.dataset.labels.data,
                   data_loader.dataset._ext_head)
        get_and_update_shortlist(doc_embeddings, shorty, data_loader)
    elif flag == 1:
        # train and don't get shortlist
        shorty.fit(doc_embeddings, data_loader.dataset.labels.data,
                   data_loader.dataset._ext_head)
    else:
        # get shortlist
        get_and_update_shortlist(doc_embeddings, shorty, data_loader)
    return None

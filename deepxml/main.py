import sys
import argparse
import os
import numpy as np
import json
import _pickle as pickle
import torch
import torch.utils.data
from pathlib import Path
import libs.utils as utils
import models.network as network
import libs.shortlist as shortlist
import libs.shortlist_utils as shortlist_utils
import libs.model as model_utils
import libs.optimizer_utils as optimizer_utils
import libs.parameters as parameters
import libs.negative_sampling as negative_sampling


__author__ = 'KD'


torch.manual_seed(22)
torch.cuda.manual_seed_all(22)
np.random.seed(22)


def load_emeddings(params):
    """Load word embeddings from numpy file
    * Support for:
        - loading pre-trained embeddings
        - loading head embeddings
    * vocabulary_dims must match #rows in embeddings
    """
    if params.use_head_embeddings:
        embeddings = np.load(
            os.path.join(os.path.dirname(params.model_dir),
                         params.embeddings))

    else:
        fname = os.path.join(
            params.data_dir, params.dataset, params.embeddings)
        if Path(fname).is_file():
            print("Loading embeddings from file: {}".format(fname))
            embeddings = np.load(fname)
        else:
            print("Loading random embeddings")
            embeddings = np.random.rand(
                params.vocabulary_dims, params.embedding_dims)
    if params.feature_indices is not None:
        indices = np.genfromtxt(params.feature_indices, dtype=np.int32)
        embeddings = embeddings[indices, :]
        del indices
    assert params.vocabulary_dims == embeddings.shape[0]
    return embeddings


def pp_with_shorty(model, params, shorty):
    """Post-process with shortlist
    Train a shortlist for a already trained model (typically from OVA) 
    """
    model._pp_with_shortlist(
        model_dir=params.model_dir,
        model_fname=params.model_fname,
        shorty=shorty,
        data_dir=params.data_dir,
        dataset=params.dataset,
        tr_feat_fname=params.tr_feat_fname,
        tr_label_fname=params.tr_label_fname,
        keep_invalid=False,
        normalize_features=params.normalize,
        normalize_labels=params.nbn_rel,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        label_indices=params.label_indices,
        feature_indices=params.feature_indices)


def train(model, params):
    """Train the model with given data
    Parameters
    ----------
    model: DeepXML
        train this model (typically DeepXML model)
    params: NameSpace
        parameter of the model
    """
    model.fit(
        data_dir=params.data_dir,
        model_dir=params.model_dir,
        result_dir=params.result_dir,
        dataset=params.dataset,
        data={'X': None, 'Y': None},
        learning_rate=params.learning_rate,
        num_epochs=params.num_epochs,
        tr_feat_fname=params.tr_feat_fname,
        val_feat_fname=params.val_feat_fname,
        tr_label_fname=params.tr_label_fname,
        val_label_fname=params.val_label_fname,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        normalize_features=params.normalize,
        normalize_labels=params.nbn_rel,
        shuffle=params.shuffle,
        validate=params.validate,
        beta=params.beta,
        init_epoch=params.last_epoch,
        keep_invalid=params.keep_invalid,
        shortlist_method=params.shortlist_method,
        validate_after=params.validate_after,
        feature_indices=params.feature_indices,
        label_indices=params.label_indices)
    model.save(params.model_dir, params.model_fname)


def get_document_embeddings(model, params, _save=True):
    """Get document embedding for given test file
    Parameters
    ----------
    model: DeepXML
        train this model (typically DeepXML model)
    params: NameSpace
        parameter of the model
    _save: boolean, optional, default=True
        Save embeddings as well (fname=params.out_fname)
    """
    fname_temp = None
    if params.huge_dataset:
        fname_temp = os.path.join(
            params.result_dir, params.out_fname + ".memmap.npy")
    doc_embeddings = model.get_document_embeddings(
        data_dir=params.data_dir,
        dataset=params.dataset,
        fname_features=params.ts_feat_fname,
        fname_labels=params.ts_label_fname,
        data={'X': None, 'Y': None},
        fname_out=fname_temp,
        return_coarse=params.use_coarse_for_shorty,
        keep_invalid=params.keep_invalid,
        batch_size=params.batch_size,
        normalize_features=params.normalize,
        num_workers=params.num_workers,
        feature_indices=params.feature_indices)
    fname = os.path.join(params.result_dir, params.out_fname)
    if _save:  # Save
        np.save(fname, doc_embeddings)
    if fname_temp is not None and os.path.exists(fname_temp):
        os.remove(fname_temp)
        del doc_embeddings
        doc_embeddings = None
    return doc_embeddings


def get_word_embeddings(model, params):
    """Extract word embeddings for the given model
    Parameters
    ----------
    model: DeepXML
        train this model (typically DeepXML model)
    params: NameSpace
        parameter of the model
    """
    _embeddings = model.net.embeddings.get_weights()
    fname = os.path.join(params.result_dir, params.out_fname)
    np.save(fname, _embeddings)


def get_classifier_wts(model, params):
    """Get classifier weights and biases for given model
    * -inf bias for untrained classifiers i.e. labels without any data
    * default path: params.result_dir/export/classifier.npy
    * Bias is appended in the end
    Parameters
    ----------
    model: DeepXML
        train this model (typically DeepXML model)
    params: NameSpace
        parameter of the model
    """
    _split = None
    if params.label_indices is not None:
        _split = params.label_indices.split("_")[-1].split(".")[0]

    fname = os.path.join(params.model_dir,
                         'labels_params.pkl' if _split is None
                         else "labels_params_split_{}.pkl".format(_split))
    temp = pickle.load(open(fname, 'rb'))
    label_mapping = temp['valid_labels']
    num_labels = temp['num_labels']
    clf_wts = np.zeros((num_labels, params.embedding_dims+1),
                       dtype=np.float32)  # +1 for bias
    clf_wts[:, -1] = -1e5  # -inf bias for untrained classifiers
    clf_wts[label_mapping, :] = model.net.get_clf_weights()
    fname = os.path.join(params.result_dir, 'export/classifier.npy')
    np.save(fname, clf_wts)


def inference(model, params):
    """Predict the top-k labels for given test data
    Parameters
    ----------
    model: DeepXML
        train this model (typically DeepXML model)
    params: NameSpace
        parameter of the model
    """
    predicted_labels = model.predict(
        data_dir=params.data_dir,
        dataset=params.dataset,
        ts_label_fname=params.ts_label_fname,
        ts_feat_fname=params.ts_feat_fname,
        normalize_features=params.normalize,
        normalize_labels=params.nbn_rel,
        beta=params.beta,
        num_workers=params.num_workers,
        top_k=params.top_k,
        data={'X': None, 'Y': None},
        keep_invalid=params.keep_invalid,
        feature_indices=params.feature_indices,
        label_indices=params.label_indices,
        use_coarse=params.use_coarse_for_shorty,
        shortlist_method=params.shortlist_method
    )
    # Real number of labels
    num_samples, num_labels = utils.get_header(
        os.path.join(params.data_dir, params.dataset, params.ts_label_fname))
    label_mapping = None
    if not params.keep_invalid:
        _split = None
        if params.label_indices is not None:
            _split = params.label_indices.split("_")[-1].split(".")[0]
        temp = "labels_params_split_{}.pkl".format(_split)
        if _split is None:
            temp = 'labels_params.pkl'
        fname = os.path.join(params.model_dir, temp)
        temp = pickle.load(open(fname, 'rb'))
        label_mapping = temp['valid_labels']
        num_labels = temp['num_labels']
    utils.save_predictions(
        predicted_labels, params.result_dir,
        label_mapping, num_samples, num_labels,
        prefix=params.pred_fname, get_fnames=params.get_only)


def construct_network(params):
    """Construct DeepXML network
    """
    if params.use_shortlist:
        return network.DeepXMLt(params)
    else:
        return network.DeepXMLh(params)


def construct_shortlist(params):
    """Construct shorty
    * Support for:
        - negative sampling (ns)
        - hnsw
        - parallel shortlist
    """
    if params.shortlist_method == 'reranker':
        return None

    if params.use_shortlist == -1:
        return None

    if params.ns_method == 'ns':  # Negative Sampling
        if params.num_clf_partitions > 1:
            raise NotImplementedError("Not tested yet!")
        else:
            shorty = negative_sampling.NegativeSampler(
                params.num_labels, params.num_nbrs, None, False)
    elif params.ns_method == 'kcentroid':
        if params.num_clf_partitions > 1:
            shorty = shortlist.ParallelShortlist(
                params.ann_method, params.num_nbrs, params.M, params.efC,
                params.efS, params.ann_threads, params.num_clf_partitions)
        else:
            shorty = shortlist.ShortlistCentroids(
                params.ann_method, params.num_nbrs, params.M,
                params.efC, params.efS, params.ann_threads)
    else:
        raise NotImplementedError("Not yet implemented!")
    return shorty


def construct_model(params, net, criterion, optimizer, shorty):
    """Construct shorty
    * Support for:
        - negative sampling (ns)
        - OVA (full)
        - hnsw (shortlist)
    """
    if params.model_method == 'ns':  # Random negative Sampling
        model = model_utils.ModelNS(
            params, net, criterion, optimizer, shorty)
    elif params.model_method == 'shortlist':  # Approximate Nearest Neighbor
        model = model_utils.ModelShortlist(
            params, net, criterion, optimizer, shorty)
    elif params.model_method == 'full':
        model = model_utils.ModelFull(params, net, criterion, optimizer)
    elif params.model_method == 'reranker':
        model = model_utils.ModelReRanker(
            params, net, criterion, optimizer, shorty)
    else:
        raise NotImplementedError("Unknown model_method.")
    return model


def main(params):
    """
        Main function
    """
    if params.mode == 'train':
        # Use last index as padding label
        if params.num_centroids != 1:
            params.label_padding_index = params.num_labels
        net = construct_network(params)
        embeddings = load_emeddings(params)
        net.initialize_embeddings(
            utils.append_padding_embedding(embeddings))
        del embeddings
        print("Initialized embeddings!")
        criterion = torch.nn.BCEWithLogitsLoss(
           reduction='sum' if params.use_shortlist else 'mean')
        print("Model parameters: ", params)
        print("\nModel configuration: ", net)
        optimizer = optimizer_utils.Optimizer(
            opt_type=params.optim,
            learning_rate=params.learning_rate,
            momentum=params.momentum,
            freeze_embeddings=params.freeze_embeddings,
            weight_decay=params.weight_decay)
        params.lrs = {"embeddings": params.learning_rate*1.0}
        optimizer.construct(net, params)
        shorty = construct_shortlist(params)
        model = construct_model(params, net, criterion, optimizer, shorty)
        model.transfer_to_devices()
        train(model, params)
        fname = os.path.join(params.result_dir, 'params.json')
        utils.save_parameters(fname, params)

    elif params.mode == 'retrain':
        fname = os.path.join(params.result_dir, 'params.json')
        utils.load_parameters(fname, params)
        net = construct_network(params)
        optimizer = optimizer_utils.Optimizer(
            opt_type=params.optim,
            learning_rate=params.learning_rate,
            momentum=params.momentum,
            freeze_embeddings=params.freeze_embeddings)
        criterion = torch.nn.BCEWithLogitsLoss(
            size_average=False if params.use_shortlist else True)
        shorty = construct_shortlist(params)
        model = construct_model(params, net, criterion, optimizer, shorty)

        model.load_checkpoint(
            params.model_dir, params.model_fname, params.last_epoch)
        model.transfer_to_devices()

        model.optimizer = optimizer
        model.optimizer.construct(model.net)

        # fname = os.path.join(
        #   params.model_dir, 'checkpoint_net_{}.pkl'.format(
        #   params.last_epoch))
        # checkpoint = torch.load(open(fname, 'rb'))
        # model.optimizer.load_state_dict(checkpoint['optimizer'])
        print("Model configuration is: ", params)
        train(model, params)

    elif params.mode == 'predict':
        fname = os.path.join(params.result_dir, 'params.json')
        utils.load_parameters(fname, params)
        net = construct_network(params)
        print("Model parameters: ", params)
        print("\nModel configuration: ", net)
        shorty = None
        shorty = construct_shortlist(params)
        model = construct_model(params, net, None, None, shorty)
        model.transfer_to_devices()
        model.load(params.model_dir, params.model_fname)
        inference(model, params)

    elif params.mode == 'retrain_w_shorty':
        #  Train ANNS for 1-vs-all classifier
        fname = os.path.join(params.result_dir, 'params.json')
        utils.load_parameters(fname, params)
        #  Pad label in case of multiple-centroids
        if params.num_centroids != 1:
            params.label_padding_index = params.num_labels
        net = construct_network(params)
        print("Model parameters: ", params)
        print("\nModel configuration: ", net)
        shorty = construct_shortlist(params)
        model = construct_model(
            params, net, criterion=None, optimizer=None, shorty=shorty)
        model.load(params.model_dir, params.model_fname)
        model.transfer_to_devices()
        pp_with_shorty(model, params, shorty)
        utils.save_parameters(fname, params)

    elif params.mode == 'extract':
        fname = os.path.join(params.result_dir, 'params.json')
        utils.load_parameters(fname, params)
        net = construct_network(params)
        print("Model parameters: ", params)
        print("\nModel configuration: ", net)
        shorty = construct_shortlist(params)
        model = construct_model(
            params, net, criterion=None, optimizer=None, shorty=shorty)
        model.load(params.model_dir, params.model_fname)
        model.transfer_to_devices()
        if params.ts_feat_fname == "0":
            get_word_embeddings(model, params)
            get_classifier_wts(model, params)
        else:
            get_document_embeddings(model, params)

    else:
        raise NotImplementedError("Unknown mode!")


if __name__ == '__main__':
    args = parameters.Parameters("Parameters")
    args.parse_args()
    main(args.params)

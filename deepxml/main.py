import sys
import argparse
import os
import numpy as np
import time
import json
import logging
import math
from scipy.io import savemat
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
import pdb


__author__ = 'KD'

torch.manual_seed(22)
torch.cuda.manual_seed_all(22)
np.random.seed(22)


def load_emeddings(params):
    if params.use_head_embeddings:
        embeddings = np.load(
            os.path.join(os.path.dirname(params.model_dir),
                        params.embeddings))
    else:
        fname = os.path.join(params.data_dir, params.dataset, params.embeddings)
        if Path(fname).is_file():
            embeddings = np.load(fname)
        else:
            print("Loading random embeddings")
            embeddings = np.random.rand(params.vocabulary_dims, params.embedding_dims)
    if params.feature_indices is not None:
        indices = np.genfromtxt(params.feature_indices, dtype=np.int32)
        embeddings = embeddings[indices, :]
        del indices
    assert params.vocabulary_dims == embeddings.shape[0]
    return embeddings


def pp_with_shorty(model, params, shorty):
    model._pp_with_shortlist(shorty=shorty,
                         data_dir=params.data_dir,
                         dataset=params.dataset,
                         fname=params.tr_fname,
                         keep_invalid=False,
                         batch_size=params.batch_size, 
                         num_workers=4,
                         label_indices=params.label_indices,
                         feature_indices=params.feature_indices)
    shorty.save(os.path.join(params.model_dir, params.model_fname+'_ANN.pkl'))


def train(model, params):
    """
        Train the model with given data
        Args:
            model: model_utils
            params: : parameters
    """
    model.fit(data_dir=params.data_dir,
              model_dir=params.model_dir,
              result_dir=params.result_dir,
              dataset=params.dataset,
              learning_rate=params.learning_rate,
              num_epochs=params.num_epochs,
              tr_fname=params.tr_fname,
              val_fname=params.val_fname,
              batch_size=params.batch_size,
              num_workers=4,
              shuffle=params.shuffle,
              validate=params.validate,
              beta=params.beta,
              init_epoch=params.last_epoch,
              keep_invalid=params.keep_invalid,
              feature_indices=params.feature_indices,
              label_indices=params.label_indices)
    #TODO: Accomodate low rank
    model.save(params.model_dir, params.model_fname, params.low_rank)


def get_document_embeddings(model, params, _save=True):
    """
        Get document embedding for given test file
        Args:
            model: model_utils
            params: parameters
    """
    doc_embeddings = model.get_document_embeddings(data_dir=params.data_dir,
                                                   dataset=params.dataset,
                                                   fname=params.ts_fname,
                                                   data=None,
                                                   keep_invalid=params.keep_invalid,
                                                   batch_size=params.batch_size,
                                                   num_workers=4,
                                                   feature_indices=params.feature_indices,
                                                   label_indices=params.label_indices)
    fname = os.path.join(params.result_dir, params.out_fname)
    if _save:
        np.save(fname, doc_embeddings)
    return doc_embeddings

def get_word_embeddings(model, params):
    """
        Get word embeddings
        Embedding corresponding to padding index is removed
        Args:
            model: model_utils
            params: parameters
    """
    if params.use_hash_embeddings:
        _embeddings, importance_wts = model.net.embeddings.get_weights()
        fname = os.path.join(params.result_dir, params.out_fname)
        np.save(fname, _embeddings)
        fname = os.path.join(params.result_dir, params.out_fname+"_imp_wts")
        np.save(fname, importance_wts)
    else:
        _embeddings = model.net.embeddings.get_weights()
        fname = os.path.join(params.result_dir, params.out_fname)
        np.save(fname, _embeddings)


def get_classifier_wts(model, params):
    """
        Get classifier weights and biases
        -inf bias for untrained classifiers i.e. labels without any data
        Args:
            model: model_utils
            params: parameters
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
    clf_wts = np.zeros((num_labels, params.embedding_dims+1), dtype=np.float32) # +1 for bias
    clf_wts[:, -1] = -1e5  # -inf bias for untrained classifiers
    clf_wts[label_mapping, :] = model.net.get_clf_weights()
    fname = os.path.join(params.result_dir, 'export/classifier.npy')
    np.save(fname, clf_wts)


def inference(model, params):
    """
        Predict the top-k labels for given test data
        Args:
            model: model_utils
            params: : parameters
    """
    predicted_labels = model.predict(data_dir=params.data_dir,
                                     dataset=params.dataset,
                                     ts_fname=params.ts_fname,
                                     beta=params.beta,
                                     keep_invalid=params.keep_invalid,
                                     feature_indices=params.feature_indices,
                                     label_indices=params.label_indices)
    # Real number of labels
    num_samples, _, num_labels = utils.get_data_header(
        os.path.join(params.data_dir, params.dataset, params.ts_fname))
    label_mapping = None
    if not params.keep_invalid:
        _split = None
        if params.label_indices is not None:
            _split = params.label_indices.split("_")[-1].split(".")[0]
        fname = os.path.join(params.model_dir,
            'labels_params.pkl' if _split is None else "labels_params_split_{}.pkl".format(_split))
        temp = pickle.load(open(fname, 'rb'))
        label_mapping = temp['valid_labels']
        num_labels = temp['num_labels']
    utils.save_predictions(predicted_labels, params.result_dir, label_mapping,
                           num_samples, num_labels, _fnames=['knn', 'clf'])



def main(params):
    """
        Main function
    """
    if params.mode == 'train':
        # Use last index as padding label
        if params.num_centroids != 1:
            params.label_padding_index = params.num_labels
        net = network.DeepXML(params)
        if not params.use_hash_embeddings:
            embeddings = load_emeddings(params)
            net.initialize_embeddings(
                utils.append_padding_embedding(embeddings))
            del embeddings
            print("Initialized embeddings!")
        criterion = torch.nn.BCEWithLogitsLoss(
            size_average=False if params.use_shortlist else True)
        print("Model parameters: ", params)
        print("\nModel configuration: ", net)
        optimizer = optimizer_utils.Optimizer(opt_type=params.optim,
                                              learning_rate=params.learning_rate,
                                              momentum=params.momentum,
                                              freeze_embeddings=params.freeze_embeddings,
                                              weight_decay=params.weight_decay)
        params.lrs = {"embeddings": params.learning_rate*1.0}
        optimizer.construct(net, params)
        if params.use_shortlist:
            shorty = shortlist.Shortlist(
                params.ann_method, params.num_nbrs, params.M, params.efC, params.efS, params.ann_threads)
            model = model_utils.ModelShortlist(params, net, criterion, optimizer, shorty)
        else:
            model = model_utils.ModelFull(params, net, criterion, optimizer)
        model.transfer_to_devices()
        train(model, params)
        fname = os.path.join(params.result_dir, 'params.json')
        utils.save_parameters(fname, params)

    elif params.mode == 'retrain':
        fname = os.path.join(params.result_dir, 'params.json')
        utils.load_parameters(fname, params)
        net = network.DeepXML(params)
        if params.use_shortlist:
            shorty = shortlist.Shortlist(
                params.ann_method, params.num_nbrs, params.M, params.efC, params.efS, params.ann_threads)
        optimizer = optimizer_utils.Optimizer(opt_type=params.optim,
                                              learning_rate=params.learning_rate,
                                              momentum=params.momentum,
                                              freeze_embeddings=params.freeze_embeddings)
        criterion = torch.nn.BCEWithLogitsLoss(
            size_average=False if params.use_shortlist else True)
        model = model.Model(
            params, net, criterion=criterion, optimizer=None, shorty=shorty)
        
        model.load_checkpoint(
            params.model_dir, params.model_fname, params.last_epoch)
        model.transfer_to_devices()

        model.optimizer = optimizer
        model.optimizer.construct(model.net)

        # fname = os.path.join(params.model_dir, 'checkpoint_net_{}.pkl'.format(params.last_epoch))
        # checkpoint = torch.load(open(fname, 'rb'))
        # model.optimizer.load_state_dict(checkpoint['optimizer'])
        print("Model configuration is: ", params)
        train(model, params)

    elif params.mode == 'predict':
        print("Model parameters: ", params)
        fname = os.path.join(params.result_dir, 'params.json')
        utils.load_parameters(fname, params)
        net = network.DeepXML(params)
        shorty = None
        if params.use_shortlist:
            shorty = shortlist.Shortlist(
                params.ann_method, params.num_nbrs, params.M, params.efC, params.efS, params.ann_threads)
            model = model_utils.ModelShortlist(params=params, net=net, criterion=None, optimizer=None, shorty=shorty)
        else:
            model = model_utils.ModelFull(params=params, net=net, criterion=None, optimizer=None)
        model.transfer_to_devices()
        model.load(params.model_dir, params.model_fname, params.use_low_rank)
        print("\nModel configuration: ", net)
        inference(model, params)

    elif params.mode == 'retrain_w_shorty':
        # Train ANNS for 1-vs-all classifier
        fname = os.path.join(params.result_dir, 'params.json')
        utils.load_parameters(fname, params)
        if params.num_centroids != 1:  # Pad label in case of multiple-centroids
            params.label_padding_index = params.num_labels
        net = network.DeepXML(params)
        print("Model parameters: ", params)
        print("\nModel configuration: ", net)
        shorty = None
        if params.use_shortlist:
            shorty = shortlist.Shortlist(
                params.ann_method, params.num_nbrs, params.M, params.efC, params.efS, params.ann_threads)
        model = model_utils.ModelFull(params, net, criterion=None, optimizer=None)
        model.load(params.model_dir, params.model_fname, params.use_low_rank)
        model.transfer_to_devices()
        pp_with_shorty(model, params, shorty)
        utils.save_parameters(fname, params)

    elif params.mode == 'extract':
        fname = os.path.join(params.result_dir, 'params.json')
        utils.load_parameters(fname, params)
        net = network.DeepXML(params)
        print("Model parameters: ", params)
        print("\nModel configuration: ", net)
        if params.use_shortlist:
            shorty = shortlist.Shortlist(
                params.ann_method, params.num_nbrs, params.M, params.efC, params.efS, params.ann_threads)
            model = model_utils.ModelShortlist(params=params, net=net, criterion=None, optimizer=None, shorty=shorty)
        else:
            model = model_utils.ModelFull(params=params, net=net, criterion=None, optimizer=None)
        model.load(params.model_dir, params.model_fname)
        model.transfer_to_devices()
        if params.ts_fname == "0":
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

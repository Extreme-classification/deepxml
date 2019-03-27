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
import libs.model_utils as model_utils
import libs.optimizer_utils as optimizer_utils

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
              num_workers=16,
              shuffle=params.shuffle,
              validate=params.validate,
              beta=params.beta,
              init_epoch=params.last_epoch,
              keep_invalid=params.keep_invalid)
    model.save(params.model_dir, params.model_fname, params.low_rank)


def get_document_embeddings(model, params):
    """
        Get document embedding for given test file
        Args:
            model: model_utils
            params: parameters
    """
    dataset = model._create_dataset(os.path.join(params.data_dir, params.dataset),
                                    fname=params.ts_fname,
                                    mode='predict',
                                    # Implemented only for shortlist as of now.
                                    use_shortlist=True,
                                    keep_invalid=params.keep_invalid)
    _data_loader = model._create_data_loader(dataset,
                                             batch_size=params.batch_size,
                                             num_workers=4)
    document_embeddings = model.get_document_embeddings(_data_loader)
    fname = os.path.join(params.result_dir, params.out_fname)
    np.save(fname, document_embeddings)


def get_word_embeddings(model, params):
    """
        Get document embedding for given test file
        Args:
            model: model_utils
            params: parameters
            0th index is the padding index
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
    _split = None
    if params.label_indices is not None:
        _split = params.label_indices.split("_")[-1].split(".")[0]
   
    fname = os.path.join(params.model_dir,
                         'labels_params.pkl' if _split is None
                         else "labels_params_split_{}.pkl".format(_split))
    _l_map = pickle.load(open(fname, 'rb'))
    label_mapping = _l_map['valid_labels']
    num_labels = _l_map['num_labels']
    clf_wts = np.zeros((num_labels, params.embedding_dims+1), dtype=np.float32) # +1 for bias
    clf_wts[:, -1] = -1e5  # -inf bias for untrained classifiers
    clf_wts[label_mapping, :] = model.net._get_clf_wts()
    fname = os.path.join(params.result_dir, 'export/classifier.npy')
    np.save(fname, clf_wts)


def inference(model, params):
    predicted_labels = model.predict(data_dir=params.data_dir,
                                     dataset=params.dataset,
                                     ts_fname=params.ts_fname,
                                     beta=params.beta,
                                     keep_invalid=params.keep_invalid)
    num_samples, _, num_labels = utils.get_data_header(
        os.path.join(params.data_dir, params.dataset, params.ts_fname))
    label_mapping = None
    if not params.keep_invalid:
        _split = None
        if params.label_indices is not None:
            _split = params.label_indices.split("_")[-1].split(".")[0]
        fname = os.path.join(params.model_dir,
            'labels_params.pkl' if _split is None else "labels_params_split_{}.pkl".format(_split))
        _l_map = pickle.load(open(fname, 'rb'))
        label_mapping = _l_map['valid_labels']
        num_labels = _l_map['num_labels']
    utils.write_predictions(
        predicted_labels, params.result_dir, params.out_fname, label_mapping, num_samples, num_labels)


def post_process(model, params):
    model.post_process_full(params.data_dir, params.model_dir, params.result_dir, params.dataset,
                            params.tr_fname, params.batch_size, keep_invalid=params.keep_invalid, num_workers=4)
    model.save(params.model_dir, params.model_fname, low_rank=-1)


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
                                              freeze_embeddings=params.freeze_embeddings)
        params.lrs = {"embeddings": params.learning_rate*params.emb_lrf}
        optimizer.construct(net, params)
        shorty = None
        if params.use_shortlist:
            shorty = shortlist.Shortlist(
                params.ann_method, params.num_nbrs, params.M, params.efC, params.efS, params.ann_threads)
        model = model_utils.Model(params, net, criterion, optimizer, shorty)
        model.transfer_to_devices()
        #model.net = torch.nn.DataParallel(model.net, device_ids=[0, 1])
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
        model = model_utils.Model(
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
        model = model_utils.Model(
            params, net, criterion=None, optimizer=None, shorty=shorty)
        model.load(params.model_dir, params.model_fname, params.use_low_rank)
        print(model.get_size())
        model.transfer_to_devices()
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
        model = model_utils.Model(
            params, net, criterion=None, optimizer=None, shorty=None)
        print(params.model_fname)
        model.load(params.model_dir, params.model_fname, params.use_low_rank)
        model.shorty = shorty
        model.transfer_to_devices()
        post_process(model, params)
        utils.save_parameters(fname, params)

    elif params.mode == 'extract':
        fname = os.path.join(params.result_dir, 'params.json')
        utils.load_parameters(fname, params)
        net = network.DeepXML(params)
        print("Model parameters: ", params)
        print("\nModel configuration: ", net)
        model = model_utils.Model(
            params, net, criterion=None, optimizer=None, shorty=None)
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
    parser = argparse.ArgumentParser(description='DeepXML')
    parser.add_argument(
        '--dataset',
        dest='dataset',
        action='store',
        type=str,
        help='dataset')
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        action='store',
        type=str,
        help='path to dataset')
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        action='store',
        type=str,
        help='directory to store models')
    parser.add_argument(
        '--result_dir',
        dest='result_dir',
        action='store',
        type=str,
        help='directory to store results')
    parser.add_argument(
        '--trans_method',
        dest='trans_method',
        default='non_linear',
        type=str,
        action='store',
        help='which network to use')
    parser.add_argument(
        '--lr',
        dest='learning_rate',
        default=0.1,
        action='store',
        type=float,
        help='learning rate')
    parser.add_argument(
        '--emb_lrf',
        dest='emb_lrf',
        default=1.0,
        action='store',
        type=dict,
        help='learning rate factor for embedding layer')
    parser.add_argument(
        '--dlr_step',
        dest='dlr_step',
        default=7,
        action='store',
        type=int,
        help='dlr_step')
    parser.add_argument(
        '--last_saved_epoch',
        dest='last_epoch',
        default=0,
        action='store',
        type=int,
        help='Last saved model at this epoch!')
    parser.add_argument(
        '--last_epoch',
        dest='last_epoch',
        default=0,
        action='store',
        type=int,
        help='Start training from here')
    parser.add_argument(
        '--ann_method',
        dest='ann_method',
        default='hnsw',
        action='store',
        type=str,
        help='Approximate nearest neighbor method')
    parser.add_argument(
        '--ann_threads',
        dest='ann_threads',
        default=4,
        action='store',
        type=int,
        help='HSNW params')
    parser.add_argument(
        '--num_hashes',
        dest='num_hashes',
        default=-1,
        action='store',
        type=int,
        help='#Hash functions to use')
    parser.add_argument(
        '--num_buckets',
        dest='num_buckets',
        default=-1,
        action='store',
        type=int,
        help='#buckets to hash vocabulary')
    parser.add_argument(
        '--M',
        dest='M',
        default=100,
        action='store',
        type=int,
        help='HSNW params')
    parser.add_argument(
        '--num_nbrs',
        dest='num_nbrs',
        default=300,
        action='store',
        type=int,
        help='HSNW params')
    parser.add_argument(
        '--label_indices',
        dest='label_indices',
        default=None,
        action='store',
        type=str,
        help='Use these labels only')
    parser.add_argument(
        '--feature_indices',
        dest='feature_indices',
        default=None,
        action='store',
        type=str,
        help='Use these features only')
    parser.add_argument(
        '--efC',
        dest='efC',
        default=300,
        action='store',
        type=int,
        help='HSNW params')
    parser.add_argument(
        '--efS',
        dest='efS',
        default=300,
        action='store',
        type=int,
        help='HSNW params')
    parser.add_argument(
        '--num_labels',
        dest='num_labels',
        default=-1,
        action='store',
        type=int,
        help='#labels')
    parser.add_argument(
        '--vocabulary_dims',
        dest='vocabulary_dims',
        default=-1,
        action='store',
        type=int,
        help='#features')
    parser.add_argument(
        '--padding_idx',
        dest='padding_idx',
        default=0,
        action='store',
        type=int,
        help='padding_idx')
    parser.add_argument(
        '--model_fname',
        dest='model_fname',
        default='dxc_model',
        action='store',
        type=str,
        help='model file name')
    parser.add_argument(
        '--out_fname',
        dest='out_fname',
        default='out',
        action='store',
        type=str,
        help='prediction file name')
    parser.add_argument(
        '--dlr_factor',
        dest='dlr_factor',
        default=0.5,
        action='store',
        type=float,
        help='dlr_factor')
    parser.add_argument(
        '--m',
        dest='momentum',
        default=0.9,
        action='store',
        type=float,
        help='momentum')
    parser.add_argument(
        '--w',
        dest='weight_decay',
        default=0.0,
        action='store',
        type=float,
        help='weight decay parameter')
    parser.add_argument(
        '--dropout',
        dest='dropout',
        default=0.5,
        action='store',
        type=float,
        help='Dropout')
    parser.add_argument(
        '--optim',
        dest='optim',
        default='SGD',
        action='store',
        type=str,
        help='Optimizer')
    parser.add_argument(
        '--embedding_dims',
        dest='embedding_dims',
        default=300,
        action='store',
        type=int,
        help='embedding dimensions')
    parser.add_argument(
        '--embeddings',
        dest='embeddings',
        default='fasttextB_embeddings_300d.npy',
        action='store',
        type=str,
        help='embedding file name')
    parser.add_argument(
        '--tr_fname',
        dest='tr_fname',
        default='train.txt',
        action='store',
        type=str,
        help='training file name')
    parser.add_argument(
        '--val_fname',
        dest='val_fname',
        default='val.txt',
        action='store',
        type=str,
        help='validation file name')
    parser.add_argument(
        '--ts_fname',
        dest='ts_fname',
        default='test.txt',
        action='store',
        type=str,
        help='test file name')
    parser.add_argument(
        '--hidden_dims',
        dest='hidden_dims',
        default=300,
        action='store',
        type=int,
        help='units in penultimate layer')
    parser.add_argument(
        '--num_epochs',
        dest='num_epochs',
        default=20,
        action='store',
        type=int,
        help='num epochs')
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        default=64,
        action='store',
        type=int,
        help='batch size')
    parser.add_argument(
        '--num_centroids',
        dest='num_centroids',
        default=1,
        type=int,
        action='store',
        help='#Centroids (Use multiple for ext head if more than 1)')
    parser.add_argument(
        '--low_rank',
        dest='low_rank',
        default=-1,
        type=int,
        action='store',
        help='#dim of low dimensional space')
    parser.add_argument(
        '--beta',
        dest='beta',
        default=0.2,
        type=float,
        action='store',
        help='weight of classifier')
    parser.add_argument(
        '--res_init',
        dest='res_init',
        default='eye',
        type=str,
        action='store',
        help='eye or random')
    parser.add_argument(
        '--label_padding_index',
        dest='label_padding_index',
        default=None,
        type=int,
        action='store',
        help='Pad with this')
    parser.add_argument(
        '--mode',
        dest='mode',
        default='train',
        type=str,
        action='store',
        help='train or predict')
    parser.add_argument(
        '--keep_invalid',
        action='store_true',
        help='Keep labels which do not have any training instance!.')
    parser.add_argument(
        '--freeze_embeddings',
        action='store_true',
        help='Do not train word embeddings.')
    parser.add_argument(
        '--use_residual',
        action='store_true',
        help='Use residual connection')
    parser.add_argument(
        '--use_low_rank',
        action='store_true',
        help='Use low rank on classifier')
    parser.add_argument(
        '--use_shortlist',
        action='store_true',
        help='Use shortlist or full')
    parser.add_argument(
        '--use_head_embeddings',
        action='store_true',
        help='Use embeddings from head or default')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate or just train')
    parser.add_argument(
        '--shuffle',
        action='store',
        default=True,
        type=bool,
        help='Shuffle data during training!')
    parser.add_argument(
        '--device_embeddings',
        action='store',
        default='cuda:0',
        help='Device for embeddings'
    )
    parser.add_argument(
        '--use_hash_embeddings',
        action='store_true',
        help='Use embeddings'
    )
    parser.add_argument(
        '--update_shortlist',
        action='store_true',
        help='Update shortlist while predicting'
    )
    parser.add_argument(
        '--device_classifier',
        action='store',
        default='cuda:0',
        help='Device for classifier'
    )
    parser.add_argument(
        '--logit_type',
        action='store',
        default=-1,
        type=int,
        help='Logit Type'
    )
    parser.add_argument(
        '--retrain_hnsw_after',
        action='store',
        default=1,
        type=int,
        help='Logit Type'
    )
    
    params = parser.parse_args()
    main(params)

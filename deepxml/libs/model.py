import logging
import math
import os
import time
from scipy.sparse import lil_matrix
import _pickle as pickle
from .model_base import ModelBase
import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np
import sys
import libs.shortlist_utils as shortlist_utils
import libs.utils as utils


class ModelFull(ModelBase):
    """
        Models with fully connected output layer
    """

    def __init__(self, params, net, criterion, optimizer):
        super().__init__(params, net, criterion, optimizer)
        self.feature_indices = params.feature_indices

    def _pp_with_shortlist(self, shorty, data_dir, dataset, fname='train.txt', 
                           data=None, keep_invalid=False, batch_size=128, 
                           num_workers=4, data_loader=None, **kwargs):
        if data_loader is None:
            dataset = self._create_dataset(os.path.join(data_dir, dataset),
                                        fname=fname,
                                        data=data,
                                        mode='predict',
                                        keep_invalid=keep_invalid,
                                        **kwargs)
            data_loader = self._create_data_loader(dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers)

        self.logger.info("Post-processing with shortlist!")
        shorty.reset()
        shortlist_utils.update(
            data_loader, self, self.embedding_dims, shorty, flag=1)
        return shorty

class ModelShortlist(ModelBase):
    """
        Models with label shortlist
    """

    def __init__(self, params, net, criterion, optimizer, shorty):
        super().__init__(params, net, criterion, optimizer)
        self.shorty = shorty
        self.num_centroids = params.num_centroids
        self.feature_indices = params.feature_indices
        self.label_indices = params.label_indices
        self.retrain_hnsw_after = params.retrain_hnsw_after
        self.update_shortlist = params.update_shortlist

    def _combine_scores(self, out_logits, batch_dist, beta):
        return beta*torch.sigmoid(out_logits) + (1-beta)*torch.sigmoid(1-batch_dist)

    def _strip_padding_label(self, mat, num_labels):
        stripped_vals = {}
        for key, val in mat.items():
            stripped_vals[key] = val[:, :num_labels].tocsr()
            del val
        return stripped_vals

    def validate(self, data_loader, beta=0.2):
        self.net.eval()
        torch.set_grad_enabled(False)
        num_labels = data_loader.dataset.num_labels
        offset = 1 if self.label_padding_index is not None else 0
        _num_labels = data_loader.dataset.num_labels + offset
        num_batches = data_loader.dataset.num_samples//data_loader.batch_size
        mean_loss = 0
        predicted_labels = {}
        predicted_labels['combined'] = lil_matrix((data_loader.dataset.num_samples,
                                                   _num_labels))
        predicted_labels['knn'] = lil_matrix((data_loader.dataset.num_samples,
                                              _num_labels))
        predicted_labels['clf'] = lil_matrix((data_loader.dataset.num_samples,
                                              _num_labels))
        count = 0
        for batch_idx, batch_data in enumerate(data_loader):
            batch_size = batch_data['X'].size(0)
            out_ans = self.net.forward(batch_data)
            loss = self._compute_loss(out_ans, batch_data)
            mean_loss += loss.item()*batch_size
            # loss = self.criterion(
            #     out_ans, self._to_device(batch_data['Y']))/batch_size
            out_ans = out_ans.cpu()
            # mean_loss += loss.item()*batch_size
            scores = self._combine_scores(
                out_ans.data, batch_data['Y_d'].data, beta)
            batch_shortlist = batch_data['Y_s'].numpy()
            utils.update_predicted_shortlist(
                count, batch_size, out_ans.data, predicted_labels['clf'], batch_shortlist)
            utils.update_predicted_shortlist(
                count, batch_size, 1 -
                batch_data['Y_d'], predicted_labels['knn'],
                batch_shortlist, top_k=self.shortlist_size)
            utils.update_predicted_shortlist(
                count, batch_size, scores, predicted_labels['combined'], batch_shortlist)
            count += batch_size
            if batch_idx % self.progress_step == 0:
                self.logger.info(
                    "Validation progress: [{}/{}]".format(batch_idx, num_batches))
        return self._strip_padding_label(predicted_labels, num_labels), mean_loss / \
            data_loader.dataset.num_samples

    def fit(self, data_dir, model_dir, result_dir, dataset, learning_rate, num_epochs, data=None, tr_fname='train.txt',
            val_fname='test.txt', batch_size=128, num_workers=4, shuffle=False, beta=0.2,
            init_epoch=0, keep_invalid=False, **kwargs):
        self.logger.info("Loading training data.")
        train_dataset = self._create_dataset(os.path.join(data_dir, dataset),
                                             fname=tr_fname,
                                             data=data,
                                             mode='train',
                                             keep_invalid=keep_invalid,
                                             **kwargs)
        train_loader_shuffle = self._create_data_loader(train_dataset,
                                                        batch_size=batch_size,
                                                        num_workers=num_workers,
                                                        shuffle=shuffle)
        train_loader = self._create_data_loader(train_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers)
        self.logger.info("Loading validation data.")
        validation_loader = None
        if self._validate:
            validation_dataset = self._create_dataset(os.path.join(data_dir, dataset),
                                                    fname=val_fname,
                                                    data=data,
                                                    mode='predict',
                                                    keep_invalid=keep_invalid,
                                                    **kwargs)
            validation_loader = self._create_data_loader(validation_dataset,
                                                        batch_size=batch_size,
                                                        num_workers=num_workers)
        for epoch in range(init_epoch, init_epoch+num_epochs):
            if epoch != 0 and self.dlr_step != -1 and epoch % self.dlr_step == 0:
                self._adjust_parameters()
            batch_train_start_time = time.time()
            if epoch % self.retrain_hnsw_after == 0:
                self.logger.info(
                    "Updating shortlist at epoch: {}".format(epoch))
                shorty_start_t = time.time()
                self.shorty.reset()
                shortlist_utils.update(
                    train_loader, self, self.embedding_dims, self.shorty, flag=0, num_graphs=self.num_clf_partitions)
                if self._validate:
                    shortlist_utils.update(
                        validation_loader, self, self.embedding_dims, self.shorty, flag=2, num_graphs=self.num_clf_partitions)
                shorty_end_t = time.time()
                self.logger.info("ANN train time: {} sec".format(
                    shorty_end_t - shorty_start_t))
                self.tracking.shortlist_time = self.tracking.shortlist_time + \
                    shorty_end_t - shorty_start_t
                batch_train_start_time = time.time()
                if self._validate:
                    try:
                        _fname = kwargs['shorty_fname']
                    except:
                        _fname = 'validation_shortlist.pkl'
                    validation_loader.dataset.save_shortlist(
                        os.path.join(model_dir, _fname))
            tr_avg_loss = self._step(train_loader_shuffle, batch_div=True)
            self.tracking.mean_train_loss.append(tr_avg_loss)
            batch_train_end_time = time.time()
            self.tracking.train_time = self.tracking.train_time + \
                batch_train_end_time - batch_train_start_time

            self.logger.info("Epoch: {}, loss: {}, time: {} sec".format(
                epoch, tr_avg_loss, batch_train_end_time - batch_train_start_time))
            if self._validate and epoch % 2 == 0:
                val_start_t = time.time()
                predicted_labels, val_avg_loss = self.validate(
                    validation_loader)
                _acc = self.evaluate(
                    validation_loader.dataset.labels, predicted_labels)
                self.logger.info("clf: {}, knn: {}".format(
                    _acc['clf'][0]*100, _acc['knn'][0]*100))
                val_end_t = time.time()
                self.tracking.validation_time = self.tracking.validation_time + val_end_t - val_start_t
                _acc = self.evaluate(
                    validation_loader.dataset.labels, predicted_labels)
                self.tracking.val_precision.append(_acc['combined'][0])
                self.tracking.val_ndcg.append(_acc['combined'][1])
                self.logger.info("Model saved after epoch: {}".format(epoch))
                self.save_checkpoint(model_dir, epoch+1)
                self.tracking.last_saved_epoch = epoch
                self.logger.info("P@1: {}, loss: {}, time: {} sec".format(
                    _acc['combined'][0][0]*100, val_avg_loss, val_end_t-val_start_t))
            self.tracking.last_epoch += 1

        self.save_checkpoint(model_dir, epoch+1)
        self.tracking.save(os.path.join(result_dir, 'training_statistics.pkl'))
        self.logger.info("Training time: {} sec, Validation time: {} sec, Shortlist time: {} sec".format(
            self.tracking.train_time, self.tracking.validation_time, self.tracking.shortlist_time))

    def _predict(self, data_loader, **kwargs):
        beta = kwargs['beta'] if 'beta' in kwargs else 0.5
        self.logger.info("Loading test data.")
        self.net.eval()
        num_labels = data_loader.dataset.num_labels
        offset = 1 if self.label_padding_index is not None else 0
        _num_labels = data_loader.dataset.num_labels + offset
        torch.set_grad_enabled(False)
        # TODO Add flag for loading or training
        if self.update_shortlist:
            shortlist_utils.update(
                data_loader, self, self.embedding_dims, self.shorty, flag=2)
        else:
            try:
                _fname = kwargs['shorty_fname']
            except:
                _fname = 'validation_shortlist.pkl'
            print("Loading Pre-computer shortlist from file: ", _fname)
            data_loader.dataset.load_shortlist(
                os.path.join(self.model_dir, _fname))

        num_batches = data_loader.dataset.num_samples//data_loader.batch_size

        predicted_labels = {}
        predicted_labels['combined'] = lil_matrix((data_loader.dataset.num_samples,
                                                   _num_labels))
        predicted_labels['knn'] = lil_matrix((data_loader.dataset.num_samples,
                                              _num_labels))
        predicted_labels['clf'] = lil_matrix((data_loader.dataset.num_samples,
                                              _num_labels))

        count = 0
        for batch_idx, batch_data in enumerate(data_loader):
            batch_size = batch_data['X'].size(0)
            out_ans = self.net.forward(batch_data).cpu()
            scores = self._combine_scores(
                out_ans.data, batch_data['Y_d'].data, beta)
            batch_shortlist = batch_data['Y_s'].numpy()

            utils.update_predicted_shortlist(
                count, batch_size, scores, predicted_labels['combined'], batch_shortlist)
            utils.update_predicted_shortlist(
                count, batch_size, out_ans.data, predicted_labels[
                    'clf'], batch_shortlist, top_k=self.shortlist_size)
            utils.update_predicted_shortlist(
                count, batch_size, 1 -
                batch_data['Y_d'], predicted_labels['knn'],
                batch_shortlist, top_k=self.shortlist_size)

            count += batch_size
            if batch_idx % self.progress_step == 0:
                self.logger.info(
                    "Prediction progress: [{}/{}]".format(batch_idx, num_batches))
        return self._strip_padding_label(predicted_labels, num_labels)

    def save_checkpoint(self, model_dir, epoch):
        super().save_checkpoint(model_dir, epoch)
        self.tracking.saved_checkpoints[-1]['ANN'] = 'checkpoint_ANN_{}.pkl'.format(epoch) 
        self.shorty.save(os.path.join(model_dir, self.tracking.saved_checkpoints[-1]['ANN']))
        self.purge(model_dir)

    def load_checkpoint(self, model_dir, fname, epoch):
        super().load_checkpoint(model_dir, fname, epoch)
        fname = os.path.join(model_dir, 'checkpoint_ANN_{}.pkl'.format(epoch))
        self.shorty.load(fname)

    def save(self, model_dir, fname, low_rank=-1):
        super().save(model_dir, fname)
        self.shorty.save(os.path.join(model_dir, fname+'_ANN.pkl'))
        #TODO: Handle low rank
        # if low_rank != -1:
        #     utils.adjust_for_low_rank(state_dict, low_rank)
        #     torch.save(state_dict, os.path.join(
        #         model_dir, fname+'_network_low_rank.pkl'))

    def load(self, model_dir, fname, use_low_rank=False):
        super().load(model_dir, fname)
        self.shorty.load(os.path.join(model_dir, fname+'_ANN.pkl'))

    def purge(self, model_dir):
        # if len(self.tracking.saved_checkpoints) > self.tracking.checkpoint_history:
        #     fname = self.tracking.saved_checkpoints[-1]['ANN']
        #     os.remove(os.path.join(model_dir, fname[1]))
        super().purge(model_dir)

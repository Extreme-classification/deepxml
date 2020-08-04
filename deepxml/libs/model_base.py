import logging
import math
import os
import time
from scipy.sparse import lil_matrix, issparse
import _pickle as pickle
import sys
import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np
import xclib.evaluation.xc_metrics as xc_metrics
import sys
import libs.utils as utils
from .dataset import construct_dataset
from .collate_fn import construct_collate_fn
from .tracking import Tracking
import torch.utils.data
from torch.utils.data import DataLoader


class ModelBase(object):
    """
        Base class for Deep extreme multi-label learning
    """

    def __init__(self, params, net, criterion, optimizer, *args, **kwargs):
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.learning_rate = params.learning_rate
        self.current_epoch = 0
        self.nbn_rel = params.nbn_rel
        self.last_saved_epoch = -1
        self.num_clf_partitions = params.num_clf_partitions
        self.model_dir = params.model_dir
        self.label_padding_index = params.label_padding_index
        self.last_epoch = 0
        self.feature_type = params.feature_type
        self.shortlist_size = params.num_nbrs if params.use_shortlist else -1
        self.dlr_step = params.dlr_step
        self.dlr_factor = params.dlr_factor
        self.progress_step = 500
        self.freeze_intermediate = params.freeze_intermediate
        self.model_fname = params.model_fname
        self.logger = self.get_logger(name=self.model_fname)
        self.devices = self._create_devices(params.devices)
        self.embedding_dims = params.embedding_dims
        self.tracking = Tracking()

    def transfer_to_devices(self):
        self.net.to()

    def _create_devices(self, _devices):
        if len(_devices) < 2:  # Repeat devices if required
            _devices = _devices*2
        # Allows model distributed training
        devices = []
        for item in _devices:
            devices.append(torch.device(
                item if torch.cuda.is_available() else "cpu"))
        return devices

    def _create_dataset(self, data_dir, fname_features, fname_labels=None,
                        data=None, mode='predict', normalize_features=True,
                        normalize_labels=False, feature_type='sparse',
                        keep_invalid=False, feature_indices=None,
                        label_indices=None, size_shortlist=-1,
                        shortlist_method='static', shorty=None,
                        aux_mapping=None, _type='full',
                        pretrained_shortlist=None):
        """
            Create dataset as per given parameters
        """
        _dataset = construct_dataset(
            data_dir=data_dir,
            fname_features=fname_features,
            fname_labels=fname_labels,
            data=data,
            model_dir=self.model_dir,
            mode=mode,
            size_shortlist=size_shortlist,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            keep_invalid=keep_invalid,
            feature_type=feature_type,
            num_clf_partitions=self.num_clf_partitions,
            feature_indices=feature_indices,
            label_indices=label_indices,
            shortlist_method=shortlist_method,
            shorty=shorty,
            aux_mapping=aux_mapping,
            pretrained_shortlist=pretrained_shortlist,
            _type=_type)
        return _dataset

    def _create_data_loader(self, dataset, batch_size=128,
                            num_workers=4, shuffle=False,
                            mode='predict', feature_type='sparse',
                            classifier_type='full'):
        """
            Create data loader for given dataset
        """
        dt_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=construct_collate_fn(
                feature_type, classifier_type, self.num_clf_partitions),
            shuffle=shuffle)
        return dt_loader

    def get_logger(self, name='DeepXML', level=logging.INFO):
        """
            Return logging object!
        """
        logging.basicConfig(level=level, stream=sys.stdout)
        logger = logging.getLogger(name)
        return logger

    def _to_device(self, tensor, index=-1):
        """
            Transfer model to respective devices
        """
        # FIXME: For now it assumes classifier is on last device
        return tensor.to(self.devices[index])

    def _compute_loss_one(self, _pred, _true):
        # Compute loss for one classifier
        _true = _true.to(_pred.get_device())
        return self.criterion(_pred, _true).to(self.devices[-1])

    def _compute_loss(self, out_ans, batch_data, weightage=1.0):
        # Support loss for parallel classifier as well
        # TODO: Integrate weightage
        if self.num_clf_partitions > 1:
            out = []
            for _, _out in enumerate(zip(out_ans, batch_data['Y'])):
                out.append(self._compute_loss_one(*_out))
            return torch.stack(out).mean()
        else:
            return self._compute_loss_one(out_ans, batch_data['Y'])

    def _step(self, data_loader, batch_div=False, bypass_coarse=False):
        """
            Training step
        """
        self.net.train()
        torch.set_grad_enabled(True)
        num_batches = data_loader.dataset.num_instances//data_loader.batch_size
        mean_loss = 0
        for batch_idx, batch_data in enumerate(data_loader):
            self.net.zero_grad()
            batch_size = batch_data['batch_size']
            out_ans = self.net.forward(batch_data, bypass_coarse)
            loss = self._compute_loss(out_ans, batch_data)
            # If loss is sum and average over samples is required
            if batch_div:
                loss = loss/batch_size
            mean_loss += loss.item()*batch_size
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.progress_step == 0:
                self.logger.info(
                    "Training progress: [{}/{}]".format(
                        batch_idx, num_batches))
            # TODO Delete items from tuple
            del batch_data
        return mean_loss / data_loader.dataset.num_instances

    def _merge_part_predictions(self, out_ans):
        return torch.stack(out_ans, axis=1)

    def _validate(self, data_loader, top_k=10):
        self.net.eval()
        top_k = min(top_k, data_loader.dataset.num_labels)
        torch.set_grad_enabled(False)
        num_batches = data_loader.dataset.num_instances//data_loader.batch_size
        mean_loss = 0
        predicted_labels = lil_matrix((data_loader.dataset.num_instances,
                                       data_loader.dataset.num_labels))
        count = 0
        for batch_idx, batch_data in enumerate(data_loader):
            batch_size = batch_data['batch_size']
            out_ans = self.net.forward(batch_data)
            loss = self._compute_loss(out_ans, batch_data)
            mean_loss += loss.item()*batch_size
            if self.num_clf_partitions > 1:
                out_ans = torch.cat(out_ans, dim=1)
            utils.update_predicted(
                count, batch_size, out_ans.data, predicted_labels, top_k)
            count += batch_size
            if batch_idx % self.progress_step == 0:
                self.logger.info(
                    "Validation progress: [{}/{}]".format(
                        batch_idx, num_batches))
            del batch_data
        return predicted_labels, mean_loss / data_loader.dataset.num_instances

    def _fit(self, train_loader, validation_loader, model_dir, result_dir,
             init_epoch, num_epochs, validate_after=5, bypass_coarse=False):
        for epoch in range(init_epoch, init_epoch+num_epochs):
            cond = self.dlr_step != -1 and epoch % self.dlr_step == 0
            if epoch != 0 and cond:
                self._adjust_parameters()
            batch_train_start_time = time.time()
            tr_avg_loss = self._step(train_loader, bypass_coarse=bypass_coarse)
            self.tracking.mean_train_loss.append(tr_avg_loss)
            batch_train_end_time = time.time()
            self.tracking.train_time = self.tracking.train_time + \
                batch_train_end_time - batch_train_start_time

            self.logger.info(
                "Epoch: {:d}, loss: {:.6f}, time: {:.2f} sec".format(
                    epoch, tr_avg_loss,
                    batch_train_end_time - batch_train_start_time))
            if validation_loader is not None and epoch % validate_after == 0:
                val_start_t = time.time()
                predicted_labels, val_avg_loss = self._validate(
                    validation_loader)
                val_end_t = time.time()
                self.tracking.validation_time = self.tracking.validation_time \
                    + val_end_t \
                    - val_start_t
                _prec, _ndcg = self.evaluate(
                    validation_loader.dataset.labels.Y, predicted_labels)
                self.tracking.mean_val_loss.append(val_avg_loss)
                self.tracking.val_precision.append(_prec)
                self.tracking.val_ndcg.append(_ndcg)
                self.logger.info("Model saved after epoch: {}".format(epoch))
                self.save_checkpoint(model_dir, epoch+1)
                self.tracking.last_saved_epoch = epoch
                self.logger.info(
                    "P@1: {:.2f}, loss: {:.6f}, time: {:.2f} sec".format(
                        _prec[0]*100, val_avg_loss, val_end_t-val_start_t))
            self.tracking.last_epoch += 1
        self.save_checkpoint(model_dir, epoch+1)
        self.tracking.save(os.path.join(result_dir, 'training_statistics.pkl'))
        self.logger.info(
            "Training time: {:.2f} sec, Validation time: {:.2f} sec"
            ", Shortlist time: {:.2f} sec, Model size: {:.2f} MB".format(
                self.tracking.train_time, self.tracking.validation_time,
                self.tracking.shortlist_time, self.model_size))

    def fit(self, data_dir, model_dir, result_dir, dataset, learning_rate,
            num_epochs, data=None, tr_feat_fname='trn_X_Xf.txt',
            tr_label_fname='trn_X_Y.txt', val_feat_fname='tst_X_Xf.txt',
            val_label_fname='tst_X_Y.txt', batch_size=128, num_workers=4,
            shuffle=False, init_epoch=0, keep_invalid=False,
            feature_indices=None, label_indices=None, normalize_features=True,
            normalize_labels=False, validate=False, feature_type='sparse',
            validate_after=5, aux_mapping=None, **kwargs):
        self.logger.info("Loading training data.")
        train_dataset = self._create_dataset(
            os.path.join(data_dir, dataset),
            fname_features=tr_feat_fname,
            fname_labels=tr_label_fname,
            data=data,
            mode='train',
            feature_type=feature_type,
            keep_invalid=keep_invalid,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            feature_indices=feature_indices,
            label_indices=label_indices,
            aux_mapping=aux_mapping)
        train_loader = self._create_data_loader(
            train_dataset,
            feature_type=feature_type,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle)
        bypass_coarse = False
        # Compute and store representation if embeddings are fixed
        if self.freeze_intermediate:
            bypass_coarse = True
            self.logger.info(
                "Computing and reusing coarse document embeddings"
                " to save computations.")
            data = {'X': None, 'Y': None}
            data['X'] = self.get_embeddings(
                data_dir=None,
                fname=None,
                data=train_dataset.features.data,
                return_coarse=True)
            data['Y'] = train_dataset.labels.data
            train_dataset = self._create_dataset(
                os.path.join(data_dir, dataset),
                data=data,
                fname_features=None,
                feature_type='dense',
                mode='train',
                keep_invalid=True)  # Invalid labels already removed
            train_loader = self._create_data_loader(
                train_dataset,
                feature_type='dense',
                batch_size=batch_size,
                num_workers=num_workers,
                classifier_type='full',
                shuffle=shuffle)
        self.logger.info("Loading validation data.")
        validation_loader = None
        if validate:
            validation_dataset = self._create_dataset(
                os.path.join(data_dir, dataset),
                fname_features=val_feat_fname,
                fname_labels=val_label_fname,
                data={'X': None, 'Y': None},
                mode='predict',
                feature_type=feature_type,
                keep_invalid=keep_invalid,
                normalize_features=normalize_features,
                normalize_labels=normalize_labels,
                feature_indices=feature_indices,
                label_indices=label_indices,
                aux_mapping=aux_mapping)
            validation_loader = self._create_data_loader(
                validation_dataset,
                feature_type=feature_type,
                batch_size=batch_size,
                num_workers=num_workers)
        self._fit(
            train_loader, validation_loader, model_dir, result_dir,
            init_epoch, num_epochs, validate_after, bypass_coarse)

    def _format_acc(self, acc):
        _res = ""
        if isinstance(acc, dict):
            for key, val in acc.items():
                _val = ','.join(map(lambda x: '%0.2f' % (x*100), val[0]))
                _res += "({}): {} ".format(key, _val)
        else:
            _val = ','.join(map(lambda x: '%0.2f' % (x*100), acc[0]))
            _res = "(clf): {}".format(_val)
        return _res

    def predict(self, data_dir, dataset, data=None,
                ts_feat_fname='tst_X_Xf.txt', ts_label_fname='tst_X_Y.txt',
                batch_size=256, num_workers=6, keep_invalid=False,
                feature_indices=None, label_indices=None, top_k=50,
                normalize_features=True, normalize_labels=False,
                aux_mapping=None, feature_type='sparse',
                classifier_type='full', **kwargs):
        dataset = self._create_dataset(
            os.path.join(data_dir, dataset),
            fname_features=ts_feat_fname,
            fname_labels=ts_label_fname,
            data=data,
            mode='predict',
            feature_type=feature_type,
            size_shortlist=self.shortlist_size,
            _type=classifier_type,
            keep_invalid=keep_invalid,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            feature_indices=feature_indices,
            label_indices=label_indices,
            aux_mapping=aux_mapping)
        data_loader = self._create_data_loader(
            feature_type=feature_type,
            classifier_type=classifier_type,
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers)
        time_begin = time.time()
        predicted_labels = self._predict(data_loader, top_k, **kwargs)
        time_end = time.time()
        prediction_time = time_end - time_begin
        acc = self.evaluate(dataset.labels.data, predicted_labels)
        _res = self._format_acc(acc)
        self.logger.info(
            "Prediction time (total): {:.2f} sec.,"
            "Prediction time (per sample): {:.2f} msec., P@k(%): {:s}".format(
                prediction_time,
                prediction_time*1000/data_loader.dataset.num_instances, _res))
        return predicted_labels

    def _predict(self, data_loader, top_k, **kwargs):
        self.net.eval()
        torch.set_grad_enabled(False)
        num_batches = data_loader.dataset.num_instances//data_loader.batch_size
        predicted_labels = lil_matrix((data_loader.dataset.num_instances,
                                       data_loader.dataset.num_labels))
        count = 0
        for batch_idx, batch_data in enumerate(data_loader):
            batch_size = batch_data['batch_size']
            out_ans = self.net.forward(batch_data)
            if self.num_clf_partitions > 1:
                out_ans = torch.cat(out_ans, dim=1)
            utils.update_predicted(
                count, batch_size, out_ans.data, predicted_labels, top_k)
            count += batch_size
            if batch_idx % self.progress_step == 0:
                self.logger.info(
                    "Prediction progress: [{}/{}]".format(
                        batch_idx, num_batches))
        return predicted_labels

    def _embeddings(self, data_loader, encoder=None,
                    bypass_fine=False, fname_out=None,
                    _dtype='float32'):
        if encoder is None:
            self.logger.info("Using the default encoder.")
            encoder = self.net.encode
        self.net.eval()
        torch.set_grad_enabled(False)
        if fname_out is not None:  # Save to disk
            embeddings = np.memmap(
                fname_out, dtype=_dtype, mode='w+',
                shape=(data_loader.dataset.num_instances,
                       self.net.representation_dims))
        else:  # Keep in memory
            embeddings = np.zeros((
                data_loader.dataset.num_instances,
                self.net.representation_dims),
                dtype=_dtype)
        count = 0
        for _, batch_data in enumerate(data_loader):
            batch_size = batch_data['batch_size']
            out_ans = encoder(
                batch_data['X'], batch_data['X_ind'], bypass_fine)
            embeddings[count:count+batch_size,
                       :] = out_ans.detach().cpu().numpy()
            count += batch_size
        torch.cuda.empty_cache()
        if fname_out is not None:  # Flush all changes to disk
            embeddings.flush()
        return embeddings

    def get_embeddings(self, encoder=None, data_dir=None, fname=None,
                       data=None, batch_size=1024, num_workers=6,
                       normalize=False, indices=None, fname_out=None,
                       bypass_fine=False, feature_type='sparse'):
        """
            Encode given data points
        """
        if data is None:
            assert data_dir is not None and fname is not None, \
                "valid file path is required when data is not passed"
        dataset = self._create_dataset(
            data_dir, fname_features=fname,
            data=data, normalize_features=normalize,
            feature_type=feature_type,
            feature_indices=indices, _type='tensor')
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=construct_collate_fn(
                feature_type=feature_type, classifier_type='None'),
            shuffle=False)
        return self._embeddings(data_loader, encoder, bypass_fine, fname_out)

    def _adjust_parameters(self):
        self.optimizer.adjust_lr(self.dlr_factor)
        self.learning_rate *= self.dlr_factor
        self.dlr_step = max(5, self.dlr_step//2)
        self.logger.info(
            "Adjusted learning rate to: {}".format(self.learning_rate))

    def save_checkpoint(self, model_dir, epoch, do_purge=True):
        checkpoint = {
            'epoch': epoch,
            'criterion': self.criterion.state_dict(),
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Useful if there are multiple parts of a model
        fname = {'net': 'checkpoint_net_{}.pkl'.format(epoch)}
        torch.save(checkpoint, os.path.join(model_dir, fname['net']))
        self.tracking.saved_checkpoints.append(fname)
        if do_purge:
            self.purge(model_dir)

    def load_checkpoint(self, model_dir, fname, epoch):
        fname = os.path.join(model_dir, 'checkpoint_net_{}.pkl'.format(epoch))
        checkpoint = torch.load(open(fname, 'rb'))
        self.net.load_state_dict(checkpoint['net'])
        self.criterion.load_state_dict(checkpoint['criterion'])
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save(self, model_dir, fname, *args):
        fname = os.path.join(
            model_dir, fname+'_network.pkl')
        self.logger.info("Saving model at: {}".format(fname))
        state_dict = self.net.state_dict()
        torch.save(state_dict, fname)

    def load(self, model_dir, fname, *args):
        fname_net = fname+'_network.pkl'
        state_dict = torch.load(
            os.path.join(model_dir, model_dir, fname_net))
        # Append Padding classifier if shapes do not match.
        # Distributed classifier not tested for now
        _output_size = self.net.classifier.output_size
        if self.num_clf_partitions > 1:
            _output_size = self.net.classifier._output_sizes
        self.logger.info(utils.append_padding_classifier(
            state_dict, _output_size))
        self.net.load_state_dict(state_dict)

    def purge(self, model_dir):
        if len(self.tracking.saved_checkpoints) \
                > self.tracking.checkpoint_history:
            fname = self.tracking.saved_checkpoints.pop(0)
            self.logger.info(
                "Purging network checkpoint: {}".format(fname['net']))
            self.net.purge(os.path.join(model_dir, fname['net']))

    def _evaluate(self, true_labels, predicted_labels):
        acc = xc_metrics.Metrics(true_labels)
        acc = acc.eval(predicted_labels.tocsr(), 5)
        return acc

    def evaluate(self, true_labels, predicted_labels):
        if issparse(predicted_labels):
            return self._evaluate(true_labels, predicted_labels)
        else:  # Multiple set of predictions
            acc = {}
            for key, val in predicted_labels.items():
                acc[key] = self._evaluate(true_labels, val)
            return acc

    @property
    def model_size(self):
        return self.net.model_size

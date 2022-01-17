import logging
import os
import time
from scipy.sparse import issparse
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
from xclib.utils.matrix import SMatrix
from tqdm import tqdm


class ModelBase(object):
    """
    Base class for Deep extreme multi-label learning

    Arguments
    ---------
    params: NameSpace
        object containing parameters like learning rate etc.
    net: models.network.DeepXMLBase
        * DeepXMLs: network with a label shortlist
        * DeepXMLf: network with fully-connected classifier
    criterion: libs.loss._Loss
        to compute loss given y and y_hat
    optimizer: libs.optimizer.Optimizer
        to back-propagate and updating the parameters
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
        # TODO
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
                        surrogate_mapping=None, _type='full',
                        pretrained_shortlist=None):
        """
        Create dataset as per given data and parameters

        Arguments
        ---------
        data_dir: str or None, optional, default=None
            load data from this directory when data is None
        fname_features: str
            load features from this file when data is None
        fname_labels: str or None, optional, default=None
            load labels from this file when data is None
        data: dict or None, optional, default=None
            directly use this this data when available
            * X: feature; Y: label (can be empty)
        mode: str, optional, default='predict'
            train or predict
        normalize_features: bool, optional, default=True
            Normalize data points to unit norm
        normalize_lables: bool, optional, default=False
            Normalize labels to convert in probabilities
            Useful in-case on non-binary labels
        feature_type: str, optional, default='sparse'
            sparse or dense features
        keep_invalid: bool, optional, default=False
            Don't touch data points or labels
        feature_indices: str or None, optional, default=None
            Train with selected features only (read from file)
        label_indices: str or None, optional, default=None
            Train for selected labels only (read from file)
        size_shortlist: int, optional, default=-1
            Size of shortlist (useful for datasets with a shortlist)
        shortlist_method: str, optional, default='static'
            static: fixed shortlist
            dynamic: dynamically generate shortlist
            hybrid: mixture of static and dynamic
        shorty: libs.shortlist.Shortlist or None, optional, default=None
            to generate a shortlist of labels
        surrogate_mapping: str, optional, default=None
            Re-map clusters as per given mapping
            e.g. when labels are clustered
        pretrained_shortlist: csr_matrix or None, default=None
            Shortlist for the dataset
        _type: str, optional, default='full'
            full: with full ground truth
            shortlist: with a shortlist
            tensor: with only features

        Returns
        -------
        dataset: Dataset
            return dataset created using given data and parameters
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
            surrogate_mapping=surrogate_mapping,
            pretrained_shortlist=pretrained_shortlist,
            _type=_type)
        return _dataset

    def _create_data_loader(self, dataset, batch_size=128,
                            num_workers=4, shuffle=False,
                            mode='predict', feature_type='sparse',
                            classifier_type='full'):
        """
        Create data loader for given dataset

        Arguments
        ---------
        dataset: Dataset
            Dataset object
        batch_size: int, optional, default=128
            batch size
        num_workers: int, optional, default=4
            #workers in data loader
        shuffle: boolean, optional, default=False
            shuffle train data in each epoch
        mode: str, optional, default='predict'
            train or predict
        feature_type: str, optional, default='sparse'
            sparse or dense features
        classifier_type: str, optional, default='full'
            OVA or a classifier with shortlist
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
        logger = logging.getLogger(name)
        if (logger.hasHandlers()):
            logger.handlers.clear()
        logger.propagate = False
        logging.Formatter(fmt='%(levelname)s:%(message)s')
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.setLevel(level=level)
        return logger

    def _to_device(self, tensor, index=-1):
        """
            Transfer model to respective devices
        """
        # FIXME: For now it assumes classifier is on last device
        return tensor.to(self.devices[index])

    def _compute_loss_one(self, _pred, _true):
        """
        Compute loss for one classifier
        """
        _true = _true.to(_pred.get_device())
        return self.criterion(_pred, _true).to(self.devices[-1])

    def _compute_loss(self, out_ans, batch_data, weightage=1.0):
        """
        Compute loss for given pair of ground truth and logits
        * Support for distributed classifier as well
        #TODO: Integrate weightage
        """
        if self.num_clf_partitions > 1:
            out = []
            for _, _out in enumerate(zip(out_ans, batch_data['Y'])):
                out.append(self._compute_loss_one(*_out))
            return torch.stack(out).mean()
        else:
            return self._compute_loss_one(out_ans, batch_data['Y'])

    def _step(self, data_loader, batch_div=False,
              precomputed_intermediate=False):
        """
        Training step (one pass over dataset)

        Arguments
        ---------
        data_loader: DataLoader
            data loader over train dataset
        batch_div: boolean, optional, default=False
            divide the loss with batch size?
            * useful when loss is sum over instances and labels
        precomputed_intermediate: boolean, optional, default=False
            if precomputed intermediate features are already available
            * avoid recomputation of intermediate features

        Returns
        -------
        loss: float
            mean loss over the train set
        """
        self.net.train()
        torch.set_grad_enabled(True)
        mean_loss = 0
        pbar = tqdm(data_loader)
        for batch_data in pbar:
            self.net.zero_grad()
            batch_size = batch_data['batch_size']
            out_ans = self.net.forward(batch_data, precomputed_intermediate)
            loss = self._compute_loss(out_ans, batch_data)
            # If loss is sum and average over samples is required
            if batch_div:
                loss = loss/batch_size
            mean_loss += loss.item()*batch_size
            loss.backward()
            self.optimizer.step()
            pbar.set_description(
                f"loss: {loss.item():.5f}")
            del batch_data
        return mean_loss / data_loader.dataset.num_instances

    def _merge_part_predictions(self, out_ans):
        """
        Merge prediction in case of distributed classifier
        """
        return torch.stack(out_ans, axis=1)

    def _validate(self, data_loader, top_k=10):
        """
        predict for the given data loader
        * retruns loss and predicted labels

        Arguments
        ---------
        data_loader: DataLoader
            data loader over validation dataset
        top_k: int, optional, default=10
            Maintain top_k predictions per data point

        Returns
        -------
        predicted_labels: csr_matrix
            predictions for the given dataset
        loss: float
            mean loss over the validation dataset
        """
        self.net.eval()
        top_k = min(top_k, data_loader.dataset.num_labels)
        torch.set_grad_enabled(False)
        mean_loss = 0
        predicted_labels = SMatrix(
            n_rows=data_loader.dataset.num_instances,
            n_cols=data_loader.dataset.num_labels,
            nnz=top_k)
        count = 0
        for batch_data in tqdm(data_loader):
            batch_size = batch_data['batch_size']
            out_ans = self.net.forward(batch_data)
            loss = self._compute_loss(out_ans, batch_data)
            mean_loss += loss.item()*batch_size
            if self.num_clf_partitions > 1:
                out_ans = torch.cat(out_ans, dim=1)
            vals, ind = torch.topk(out_ans, k=top_k, dim=-1, sorted=False)
            predicted_labels.update_block(
                count, ind.cpu().numpy(), vals.cpu().numpy())
            count += batch_size
            del batch_data
        return predicted_labels.data(), \
            mean_loss / data_loader.dataset.num_instances

    def _fit(self, train_loader, validation_loader, model_dir,
             result_dir, init_epoch, num_epochs, validate_after=5,
             precomputed_intermediate=False):
        """
        Train for the given data loader

        Arguments
        ---------
        train_loader: DataLoader
            data loader over train dataset
        validation_loader: DataLoader or None
            data loader over validation dataset
        model_dir: str
            save checkpoints etc. in this directory
        result_dir: str
            save logs etc in this directory
        init_epoch: int, optional, default=0
            start training from this epoch
            (useful when fine-tuning from a checkpoint)
        num_epochs: int
            #passes over the dataset
        validate_after: int, optional, default=5
            validate after a gap of these many epochs
        precomputed_intermediate: boolean, optional, default=False
            if precomputed intermediate features are already available
            * avoid recomputation of intermediate features
        """
        for epoch in range(init_epoch, init_epoch+num_epochs):
            cond = self.dlr_step != -1 and epoch % self.dlr_step == 0
            if epoch != 0 and cond:
                self._adjust_parameters()
            batch_train_start_time = time.time()
            trn_avg_loss = self._step(
                train_loader,
                precomputed_intermediate=precomputed_intermediate)
            self.tracking.mean_train_loss.append(trn_avg_loss)
            batch_train_end_time = time.time()
            self.tracking.train_time = self.tracking.train_time + \
                batch_train_end_time - batch_train_start_time

            self.logger.info(
                "Epoch: {:d}, loss: {:.6f}, time: {:.2f} sec".format(
                    epoch, trn_avg_loss,
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
            num_epochs, data=None, trn_feat_fname='trn_X_Xf.txt',
            trn_label_fname='trn_X_Y.txt', val_feat_fname='tst_X_Xf.txt',
            val_label_fname='tst_X_Y.txt', batch_size=128, num_workers=4,
            shuffle=False, init_epoch=0, keep_invalid=False,
            feature_indices=None, label_indices=None, normalize_features=True,
            normalize_labels=False, validate=False, validate_after=5,
            feature_type='sparse', surrogate_mapping=None, **kwargs):
        """
        Train for the given data
        * Also prints train time and model size

        Arguments
        ---------
        data_dir: str or None, optional, default=None
            load data from this directory when data is None
        model_dir: str
            save checkpoints etc. in this directory
        result_dir: str
            save logs etc in this directory
        dataset: str
            Name of the dataset
        learning_rate: float
            initial learning rate
        num_epochs: int
            #passes over the dataset
        data: dict or None, optional, default=None
            directly use this this data to train when available
            * X: feature; Y: label
        trn_feat_fname: str, optional, default='trn_X_Xf.txt'
            train features
        trn_label_fname: str, optional, default='trn_X_Y.txt'
            train labels
        val_feat_fname: str, optional, default='tst_X_Xf.txt'
            validation features (used only when validate is True)
        val_label_fname: str, optional, default='tst_X_Y.txt'
            validation labels (used only when validate is True)
        batch_size: int, optional, default=1024
            batch size in data loader
        num_workers: int, optional, default=6
            #workers in data loader
        shuffle: boolean, optional, default=True
            shuffle train data in each epoch
        init_epoch: int, optional, default=0
            start training from this epoch
            (useful when fine-tuning from a checkpoint)
        keep_invalid: bool, optional, default=False
            Don't touch data points or labels
        feature_indices: str or None, optional, default=None
            Train with selected features only (read from file)
        label_indices: str or None, optional, default=None
            Train for selected labels only (read from file)
        normalize_features: bool, optional, default=True
            Normalize data points to unit norm
        normalize_lables: bool, optional, default=False
            Normalize labels to convert in probabilities
            Useful in-case on non-binary labels
        validate: bool, optional, default=True
            validate using the given data if flag is True
        validate_after: int, optional, default=5
            validate after a gap of these many epochs
        feature_type: str, optional, default='sparse'
            sparse or dense features
        surrogate_mapping: str, optional, default=None
            Re-map clusters as per given mapping
            e.g. when labels are clustered
        """
        # Reset the logger to dump in train log file
        self.logger.addHandler(
            logging.FileHandler(os.path.join(result_dir, 'log_train.txt'))) 
        self.logger.info("Loading training data.")
        train_dataset = self._create_dataset(
            os.path.join(data_dir, dataset),
            fname_features=trn_feat_fname,
            fname_labels=trn_label_fname,
            data=data,
            mode='train',
            feature_type=feature_type,
            keep_invalid=keep_invalid,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            feature_indices=feature_indices,
            label_indices=label_indices,
            surrogate_mapping=surrogate_mapping)
        train_loader = self._create_data_loader(
            train_dataset,
            feature_type=feature_type,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle)
        precomputed_intermediate = False
        # Compute and store representation if embeddings are fixed
        if self.freeze_intermediate:
            precomputed_intermediate = True
            self.logger.info(
                "Computing and reusing coarse document embeddings"
                " to save computations.")
            data = {'X': None, 'Y': None}
            data['X'] = self.get_embeddings(
                data_dir=None,
                fname=None,
                data=train_dataset.features.data,
                use_intermediate=True)
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
                surrogate_mapping=surrogate_mapping)
            validation_loader = self._create_data_loader(
                validation_dataset,
                feature_type=feature_type,
                batch_size=batch_size,
                num_workers=num_workers)
        self._fit(
            train_loader, validation_loader, model_dir, result_dir,
            init_epoch, num_epochs, validate_after, precomputed_intermediate)
        train_time = self.tracking.train_time + self.tracking.shortlist_time
        return train_time, self.model_size

    def _format_acc(self, acc):
        """
        Format accuracies (precision, ndcg) as string
        Useful in case of multiple
        """
        _res = ""
        if isinstance(acc, dict):
            for key, val in acc.items():
                _val = ','.join(map(lambda x: '%0.2f' % (x*100), val[0]))
                _res += "({}): {} ".format(key, _val)
        else:
            _val = ','.join(map(lambda x: '%0.2f' % (x*100), acc[0]))
            _res = "(clf): {}".format(_val)
        return _res.strip()

    def predict(self, data_dir, result_dir, dataset, data=None,
                tst_feat_fname='tst_X_Xf.txt', tst_label_fname='tst_X_Y.txt',
                batch_size=256, num_workers=6, keep_invalid=False,
                feature_indices=None, label_indices=None, top_k=50,
                normalize_features=True, normalize_labels=False,
                surrogate_mapping=None, feature_type='sparse',
                classifier_type='full', **kwargs):
        """
        Predict for the given data
        * Also prints prediction time, precision and ndcg

        Arguments
        ---------
        data_dir: str or None, optional, default=None
            load data from this directory when data is None
        dataset: str
            Name of the dataset
        data: dict or None, optional, default=None
            directly use this this data when available
            * X: feature; Y: label (can be empty)
        tst_feat_fname: str, optional, default='tst_X_Xf.txt'
            load features from this file when data is None
        tst_label_fname: str, optional, default='tst_X_Y.txt'
            load labels from this file when data is None
            * can be dummy
        batch_size: int, optional, default=1024
            batch size in data loader
        num_workers: int, optional, default=6
            #workers in data loader
        keep_invalid: bool, optional, default=False
            Don't touch data points or labels
        feature_indices: str or None, optional, default=None
            Train with selected features only (read from file)
        label_indices: str or None, optional, default=None
            Train for selected labels only (read from file)
        top_k: int
            Maintain top_k predictions per data point
        normalize_features: bool, optional, default=True
            Normalize data points to unit norm
        normalize_lables: bool, optional, default=False
            Normalize labels to convert in probabilities
            Useful in-case on non-binary labels
        surrogate_mapping: str, optional, default=None
            Re-map clusters as per given mapping
            e.g. when labels are clustered
        feature_type: str, optional, default='sparse'
            sparse or dense features
        classifier_type: str, optional, default='full'
            OVA or a classifier with shortlist

        Returns
        -------
        predicted_labels: csr_matrix
            predictions for the given dataset
        """
        # Reset the logger to dump in predict log file
        self.logger.addHandler(
            logging.FileHandler(os.path.join(result_dir, 'log_predict.txt')))
        dataset = self._create_dataset(
            os.path.join(data_dir, dataset),
            fname_features=tst_feat_fname,
            fname_labels=tst_label_fname,
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
            surrogate_mapping=surrogate_mapping)
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
        avg_prediction_time = prediction_time*1000/len(data_loader.dataset)
        acc = self.evaluate(dataset.labels.data, predicted_labels)
        _res = self._format_acc(acc)
        self.logger.info(
            "Prediction time (total): {:.2f} sec.,"
            "Prediction time (per sample): {:.2f} msec., P@k(%): {:s}".format(
                prediction_time,
                avg_prediction_time, _res))
        return predicted_labels, prediction_time, avg_prediction_time

    def _predict(self, data_loader, top_k, **kwargs):
        """
        Predict for the given data_loader

        Arguments
        ---------
        data_loader: DataLoader
            DataLoader object to create batches and iterate over it
        top_k: int
            Maintain top_k predictions per data point

        Returns
        -------
        predicted_labels: csr_matrix
            predictions for the given dataset
        """
        self.net.eval()
        torch.set_grad_enabled(False)
        predicted_labels = SMatrix(
            n_rows=data_loader.dataset.num_instances,
            n_cols=data_loader.dataset.num_labels,
            nnz=top_k)
        count = 0
        for batch_data in tqdm(data_loader):
            batch_size = batch_data['batch_size']
            out_ans = self.net.forward(batch_data)
            if self.num_clf_partitions > 1:
                out_ans = torch.cat(out_ans, dim=1)
            vals, ind = torch.topk(out_ans, k=top_k, dim=-1, sorted=False)
            predicted_labels.update_block(
                count, ind.cpu().numpy(), vals.cpu().numpy())
            count += batch_size
        return predicted_labels.data()

    def _embeddings(self, data_loader, encoder=None,
                    use_intermediate=False, fname_out=None,
                    _dtype='float32'):
        """
        Encode given data points
        * support for objects or files on disk

        Arguments
        ---------
        data_loader: DataLoader
            DataLoader object to create batches and iterate over it
        encoder: callable or None, optional, default=None
            use this function to encode given dataset
            * net.encode is used when None
        use_intermediate: boolean, optional, default=False
            return intermediate representation if True
        fname_out: str or None, optional, default=None
            load data from this file when data is None
        _dtype: str, optional, default='float32'
            data type of the encoded data
        """
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
        for batch_data in tqdm(data_loader):
            batch_size = batch_data['batch_size']
            out_ans = encoder(
                batch_data['X'], batch_data['X_ind'], use_intermediate)
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
                       use_intermediate=False, feature_type='sparse'):
        """
        Encode given data points
        * support for objects or files on disk

        Arguments
        ---------
        encoder: callable or None, optional, default=None
            use this function to encode given dataset
            * net.encode is used when None
        data_dir: str or None, optional, default=None
            load data from this directory when data is None
        fname: str or None, optional, default=None
            load data from this file when data is None
        data: csr_matrix or ndarray or None, optional, default=None
            directly use this this data when available
        batch_size: int, optional, default=1024
            batch size in data loader
        num_workers: int, optional, default=6
            #workers in data loader
        normalize: boolean, optioanl, default=False
            Normalize instances to unit l2-norm if True
        indices: list or None, optional or None
            Use only these feature indices; use all when None
        fname_out: str or None, optioanl, default=None
            save as memmap if filename is given
        use_intermediate: boolean, optional, default=False
            return intermediate representation if True
        feature_type: str, optional, default='sparse'
            feature type such as sparse/dense
        """
        if data is None:
            assert data_dir is not None and fname is not None, \
                "valid file path is required when data is not passed"
        dataset = self._create_dataset(
            data_dir, fname_features=fname,
            data=data, normalize_features=normalize,
            feature_type=feature_type,
            feature_indices=indices,
            _type='tensor')
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=construct_collate_fn(
                feature_type=feature_type, classifier_type='None'),
            shuffle=False)
        return self._embeddings(
            data_loader, encoder, use_intermediate, fname_out)

    def _adjust_parameters(self):
        """
        Adjust learning rate

        This strategy seems to work well in practise
        * lr = lr * dlr_factor
        * dlr_step = max(dlr_step//2, 5)
        """
        self.optimizer.adjust_lr(self.dlr_factor)
        self.learning_rate *= self.dlr_factor
        self.dlr_step = max(5, self.dlr_step//2)
        self.logger.info(
            "Adjusted learning rate to: {}".format(self.learning_rate))

    def save_checkpoint(self, model_dir, epoch, do_purge=True):
        """
        Save checkpoint on disk
        * save network, optimizer and loss
        * filename: checkpoint_net_epoch.pkl for network

        Arguments:
        ---------
        model_dir: str
            save checkpoint into this directory
        epoch: int
            checkpoint after this epoch (used in file name)
        do_purge: boolean, optional, default=True
            delete old checkpoints beyond a point
        """
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

    def load_checkpoint(self, model_dir, epoch):
        """
        Load checkpoint from disk
        * load network, optimizer and loss
        * filename: checkpoint_net_epoch.pkl for network

        Arguments:
        ---------
        model_dir: str
            load checkpoint into this directory
        epoch: int
            checkpoint after this epoch (used in file name)
        """
        fname = os.path.join(model_dir, 'checkpoint_net_{}.pkl'.format(epoch))
        checkpoint = torch.load(open(fname, 'rb'))
        self.net.load_state_dict(checkpoint['net'])
        self.criterion.load_state_dict(checkpoint['criterion'])
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save(self, model_dir, fname, *args):
        """
        Save model on disk
        * uses prefix: _network.pkl for network

        Arguments:
        ---------
        model_dir: str
            save model into this directory
        fname: str
            save model with this file name
        """
        fname = os.path.join(
            model_dir, fname+'_network.pkl')
        self.logger.info("Saving model at: {}".format(fname))
        state_dict = self.net.state_dict()
        torch.save(state_dict, fname)

    def load(self, model_dir, fname, *args):
        """
        Load model from disk
        * uses prefix: _network.pkl for network

        Arguments:
        ---------
        model_dir: str
            load model from this directory
        fname: str
            load model with this file name
        """
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
        """
        Remove checkpoints from disk
        * uses checkpoint_history to decide which checkpoint to delete
        * delete if #saved_checkpoints is more than a threshold; otherwise skip
        """
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
        """
        Compute precision and ndcg for given prediction matrix

        Arguments
        ---------
        true_labels: csr_matrix
            ground truth matrix
        predicted_labels: csr_matrix or dict
            predictions matrix (expect dictionary in case of multiple)

        Returns
        --------
        acc: list or dict of list
            return precision and ndcg
            * output dictionary uses same keys as input
        """
        if issparse(predicted_labels):
            acc = self._evaluate(true_labels, predicted_labels)
        else:  # Multiple set of predictions
            acc = {}
            for key, val in predicted_labels.items():
                acc[key] = self._evaluate(true_labels, val)
        return acc

    @property
    def model_size(self):
        """
        Return model size (in MB)
        """
        return self.net.model_size

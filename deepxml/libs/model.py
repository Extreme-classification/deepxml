import logging
import os
import time
from .model_base import ModelBase
import torch.utils.data
from .features import DenseFeatures
from xclib.utils.matrix import SMatrix
from xclib.utils.sparse import sigmoid
from tqdm import tqdm


class ModelFull(ModelBase):
    """
    Models with fully connected output layer

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

    def __init__(self, params, net, criterion, optimizer):
        super().__init__(params, net, criterion, optimizer)
        self.feature_indices = params.feature_indices


class ModelShortlist(ModelBase):
    """
    Models with label shortlist

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
    shorty: libs.shortlist.Shortlist
        to generate a shortlist of labels (typically an ANN structure)
        * same shortlist method is used during training and prediction
    """

    def __init__(self, params, net, criterion, optimizer, shorty):
        super().__init__(params, net, criterion, optimizer)
        self.shorty = shorty
        self.feature_indices = params.feature_indices
        self.label_indices = params.label_indices
        self.retrain_hnsw_after = params.retrain_hnsw_after
        self.update_shortlist = params.update_shortlist

    def _compute_loss_one(self, _pred, _true, _mask):
        """
        Compute loss for one classifier
        """
        _true = _true.to(_pred.get_device())
        if _mask is not None:
            _mask = _mask.to(_true.get_device())
        return self.criterion(_pred, _true, _mask).to(self.devices[-1])

    def _compute_loss(self, out_ans, batch_data, weightage=1.0):
        """
        Compute loss for given pair of ground truth and logits
        * Support for distributed classifier as well
        #TODO: Integrate weightage
        """
        if self.num_clf_partitions > 1:
            out = []
            temp = zip(out_ans, batch_data['Y'], batch_data['Y_mask'])
            for _, _out in enumerate(temp):
                out.append(self._compute_loss_one(*_out))
            return torch.stack(out).mean()
        else:
            return self._compute_loss_one(
                out_ans, batch_data['Y'], batch_data['Y_mask'])

    def _combine_scores(self, logit, sim, beta):
        """
        Combine scores of label classifier and shortlist
        score = beta*sigmoid(logit) + (1-beta)*sigmoid(sim)
        """
        return beta*sigmoid(logit, copy=True) \
            + (1-beta)*sigmoid(sim, copy=True)

    def _fit_shorty(self, features, labels, doc_embeddings=None,
                    use_intermediate=True, feature_type='sparse'):
        """
        Train the ANN Structure with given data

        * Support for pre-computed features
        * Features are computed when pre-computed features are not available

        Arguments
        ---------
        features: np.ndarray or csr_matrix or None
            features for given data (used when doc_embeddings is None)
        labels: csr_matrix
            ground truth matrix for given data
        doc_embeddings: np.ndarray or None, optional, default=None
            pre-computed features; features are computed when None
        use_intermediate: boolean, optional, default=True
            use intermediate representation if True
        feature_type: str, optional, default='sparse'
            sparse or dense features
        """
        if doc_embeddings is None:
            doc_embeddings = self.get_embeddings(
                data=features,
                feature_type=feature_type,
                use_intermediate=use_intermediate)
        self.shorty.fit(doc_embeddings, labels)

    def _update_shortlist(self, dataset, use_intermediate=True, mode='train',
                          flag=True):
        """
        Get nearest neighbors for the given data and
         update the shortlist in dataset

        * Will train ANN structure for train set
        * flag can be used to ignore (useful in case of precomputed shortlist)

        Arguments
        ---------
        dataset: Dataset
            Dataset object
            * features and labels are used when required
            * update_shortlist method is used to update neighbors and sim
            * will directly use features for DenseFeatures
        use_intermediate: boolean, optional, default=True
            use intermediate representation if True
        mode: str, optional, default='train'
            train the ANN structure when mode is 'train'
        flag: boolean, optional, default=True
            use function only when flag is True
        """
        if flag:
            if isinstance(dataset.features, DenseFeatures) and use_intermediate:
                self.logger.info("Using pre-trained embeddings for shortlist.")
                doc_embeddings = dataset.features.data
            else:
                doc_embeddings = self.get_embeddings(
                    data=dataset.features.data,
                    use_intermediate=use_intermediate)
            if mode == 'train':
                self.shorty.reset()
                self._fit_shorty(
                    features=None,
                    labels=dataset.labels.data,
                    doc_embeddings=doc_embeddings)
            dataset.update_shortlist(
                *self._predict_shorty(doc_embeddings))

    def _predict_shorty(self, doc_embeddings):
        """
        Get nearest neighbors (and sim) for given document embeddings

        Arguments
        ---------
        doc_embeddings: np.ndarray
            embeddings/encoding for the data points

        Returns
        -------
        neighbors: np.ndarray
            indices of nearest neighbors
        sim: np.ndarray
            similarity with nearest neighbors
        """
        return self.shorty.query(doc_embeddings)

    def _update_predicted_shortlist(self, count, batch_size, predicted_labels,
                                    batch_out, batch_data):
        # IF rev mapping exist; case of distributed classifier
        if 'Y_map' in batch_data:
            _indices = batch_data['Y_map']
            # Send this as merged?
            _knn_score = torch.cat(batch_data['Y_sim'], 1).data.cpu().numpy()
            _clf_score = torch.cat(batch_out, 1).data.cpu().numpy()
        else:
            _indices = batch_data['Y_s'].numpy()
            _knn_score = batch_data['Y_sim'].numpy()
            _clf_score = batch_out.data.cpu().numpy()
        predicted_labels['clf'].update_block(count, _indices, _clf_score)
        predicted_labels['knn'].update_block(count, _indices, _knn_score)


    def _validate(self, data_loader, beta=0.2, top_k=20):
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
        torch.set_grad_enabled(False)
        num_labels = data_loader.dataset.num_labels
        num_instances = data_loader.dataset.num_instances
        mean_loss = 0
        predicted_labels = {}
        predicted_labels['knn'] = SMatrix(
            n_rows=num_instances,
            n_cols=num_labels,
            nnz=top_k)

        predicted_labels['clf'] = SMatrix(
            n_rows=num_instances,
            n_cols=num_labels,
            nnz=top_k)

        count = 0
        for batch_data in tqdm(data_loader):
            batch_size = batch_data['batch_size']
            out_ans = self.net.forward(batch_data)
            loss = self._compute_loss(out_ans, batch_data)/batch_size
            mean_loss += loss.item()*batch_size
            self._update_predicted_shortlist(
                count, batch_size, predicted_labels, out_ans, batch_data)
            count += batch_size
        for k, v in predicted_labels.items():
            predicted_labels[k] = v.data()
        predicted_labels['ens'] = self._combine_scores(
            predicted_labels['clf'], predicted_labels['knn'], beta)
        return predicted_labels, mean_loss / num_instances


    def _fit(self, train_loader, validation_loader, model_dir, result_dir,
             init_epoch, num_epochs, validate_after, beta,
             use_intermediate_for_shorty, precomputed_intermediate):
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
        beta: float
            weightage of classifier when combining with shortlist scores
        use_intermediate_for_shorty: boolean
            use intermediate representation for negative sampling/ ANN search
        precomputed_intermediate: boolean, optional, default=False
            if precomputed intermediate features are already available
            * avoid recomputation of intermediate features
        """
        for epoch in range(init_epoch, init_epoch+num_epochs):
            cond = self.dlr_step != -1 and epoch % self.dlr_step == 0
            if epoch != 0 and cond:
                self._adjust_parameters()
            batch_train_start_time = time.time()
            if epoch % self.retrain_hnsw_after == 0:
                self.logger.info(
                    "Updating shortlist at epoch: {}".format(epoch))
                shorty_start_t = time.time()
                self._update_shortlist(
                    dataset=train_loader.dataset,
                    use_intermediate=use_intermediate_for_shorty,
                    mode='train',
                    flag=self.shorty is not None)
                if validation_loader is not None:
                    self._update_shortlist(
                        dataset=validation_loader.dataset,
                        use_intermediate=use_intermediate_for_shorty,
                        mode='predict',
                        flag=self.shorty is not None)
                shorty_end_t = time.time()
                self.logger.info("ANN train time: {0:.2f} sec".format(
                    shorty_end_t - shorty_start_t))
                self.tracking.shortlist_time = self.tracking.shortlist_time \
                    + shorty_end_t - shorty_start_t
                batch_train_start_time = time.time()
            tr_avg_loss = self._step(
                train_loader, batch_div=True,
                precomputed_intermediate=precomputed_intermediate)
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
                    validation_loader, beta, self.shortlist_size)
                val_end_t = time.time()
                _acc = self.evaluate(
                    validation_loader.dataset.labels.data, predicted_labels)
                self.tracking.validation_time = self.tracking.validation_time \
                    + val_end_t - val_start_t
                self.tracking.mean_val_loss.append(val_avg_loss)
                self.tracking.val_precision.append(_acc['ens'][0])
                self.tracking.val_ndcg.append(_acc['ens'][1])
                self.logger.info("Model saved after epoch: {}".format(epoch))
                self.save_checkpoint(model_dir, epoch+1)
                self.tracking.last_saved_epoch = epoch
                _res = self._format_acc(_acc)
                self.logger.info(
                    "P@k {:s}, loss: {:.6f}, time: {:.2f} sec".format(
                        _res, val_avg_loss, val_end_t-val_start_t))
            self.tracking.last_epoch += 1

        self.save_checkpoint(model_dir, epoch+1)
        self.tracking.save(os.path.join(result_dir, 'training_statistics.pkl'))
        self.logger.info(
            "Training time: {:.2f} sec, Validation time: {:.2f} sec, "
            "Shortlist time: {:.2f} sec, Model size: {:.2f} MB".format(
                self.tracking.train_time,
                self.tracking.validation_time,
                self.tracking.shortlist_time,
                self.model_size))

    def fit(self, data_dir, model_dir, result_dir, dataset, learning_rate,
            num_epochs, data=None, trn_feat_fname='trn_X_Xf.txt',
            trn_label_fname='trn_X_Y.txt', val_feat_fname='tst_X_Xf.txt',
            val_label_fname='tst_X_Y.txt', batch_size=128, num_workers=4,
            shuffle=False, init_epoch=0, keep_invalid=False,
            feature_indices=None, label_indices=None, normalize_features=True,
            normalize_labels=False, validate=False, beta=0.2,
            use_intermediate_for_shorty=True, shortlist_method='static',
            validate_after=5, surrogate_mapping=None, feature_type='sparse',
            trn_pretrained_shortlist=None, val_pretrained_shortlist=None):
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
        beta: float, optional, default=0.5
            weightage of classifier when combining with shortlist scores
        use_intermediate_for_shorty: boolean, optional, default=True
            use intermediate representation for negative sampling/ANN
        shortlist_method: str, optional, default='static'
            static: fixed shortlist
            dynamic: dynamically generate shortlist
            hybrid: mixture of static and dynamic
        validate_after: int, optional, default=5
            validate after a gap of these many epochs
        surrogate_mapping: str, optional, default=None
            Re-map clusters as per given mapping
            e.g. when labels are clustered
        feature_type: str, optional, default='sparse'
            sparse or dense features
        trn_pretrained_shortlist: csr_matrix or None, default=None
            Shortlist for train dataset
        val_pretrained_shortlist: csr_matrix or None, default=None
            Shortlist for validation dataset
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
            keep_invalid=keep_invalid,
            normalize_features=normalize_features,
            size_shortlist=self.shortlist_size,
            normalize_labels=normalize_labels,
            feature_indices=feature_indices,
            shortlist_method=shortlist_method,
            feature_type=feature_type,
            label_indices=label_indices,
            surrogate_mapping=surrogate_mapping,
            pretrained_shortlist=trn_pretrained_shortlist,
            _type='shortlist')
        train_loader = self._create_data_loader(
            train_dataset,
            feature_type=train_dataset.feature_type,
            classifier_type='shortlist',
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle)
        precomputed_intermediate = False
        if self.freeze_intermediate or not self.update_shortlist:
            self.retrain_hnsw_after = 10000

        # No need to update embeddings
        if self.freeze_intermediate and feature_type != 'dense':
            precomputed_intermediate = True
            self.logger.info(
                "Computing and reusing intermediate document embeddings "
                "to save computations.")
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
                mode='train',
                normalize_features=False,  # do not normalize dense features
                shortlist_method=shortlist_method,
                size_shortlist=self.shortlist_size,
                feature_type='dense',
                pretrained_shortlist=trn_pretrained_shortlist,
                keep_invalid=True,   # Invalid labels already removed
                _type='shortlist')
            train_loader = self._create_data_loader(
                train_dataset,
                feature_type='dense',
                classifier_type='shortlist',
                batch_size=batch_size,
                num_workers=num_workers,
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
                size_shortlist=self.shortlist_size,
                keep_invalid=keep_invalid,
                normalize_features=normalize_features,
                normalize_labels=normalize_labels,
                feature_type=feature_type,
                feature_indices=feature_indices,
                pretrained_shortlist=val_pretrained_shortlist,
                label_indices=label_indices,
                surrogate_mapping=surrogate_mapping,
                _type='shortlist')
            validation_loader = self._create_data_loader(
                validation_dataset,
                feature_type=validation_dataset.feature_type,
                classifier_type='shortlist',
                batch_size=batch_size,
                num_workers=num_workers)
        self._fit(
            train_loader, validation_loader, model_dir, result_dir,
            init_epoch, num_epochs, validate_after, beta,
            use_intermediate_for_shorty, precomputed_intermediate)
        train_time = self.tracking.train_time + self.tracking.shortlist_time
        return train_time, self.model_size

    def _predict(self, data_loader, top_k, beta,
                use_intermediate_for_shorty):
        """
        Predict for the given data_loader

        Arguments
        ---------
        data_loader: DataLoader
            DataLoader object to create batches and iterate over it
        top_k: int
            Maintain top_k predictions per data point
        use_intermediate_for_shorty: bool
            use intermediate representation for negative sampling/ANN

        Returns
        -------
        predicted_labels: csr_matrix
            predictions for the given dataset
        """
        self.logger.info("Loading test data.")
        self.net.eval()
        num_labels = data_loader.dataset.num_labels
        torch.set_grad_enabled(False)
        self.logger.info("Fetching shortlist.")
        self._update_shortlist(
            dataset=data_loader.dataset,
            use_intermediate=use_intermediate_for_shorty,
            mode='predict',
            flag=self.shorty is not None)
        num_instances = data_loader.dataset.num_instances
        predicted_labels = {}
        predicted_labels['knn'] = SMatrix(
            n_rows=num_instances,
            n_cols=num_labels,
            nnz=top_k)

        predicted_labels['clf'] = SMatrix(
            n_rows=num_instances,
            n_cols=num_labels,
            nnz=top_k)

        count = 0
        for batch_data in tqdm(data_loader):
            batch_size = batch_data['batch_size']
            out_ans = self.net.forward(batch_data)
            self._update_predicted_shortlist(
                count, batch_size, predicted_labels,
                out_ans, batch_data)
            count += batch_size
            del batch_data
        for k, v in predicted_labels.items():
            predicted_labels[k] = v.data()
        predicted_labels['ens'] = self._combine_scores(
            predicted_labels['clf'], predicted_labels['knn'], beta)
        return predicted_labels


    def predict(self, data_dir, result_dir, dataset, data=None,
                tst_feat_fname='tst_X_Xf.txt', tst_label_fname='tst_X_Y.txt',
                batch_size=256, num_workers=6, keep_invalid=False,
                feature_indices=None, label_indices=None, top_k=50,
                normalize_features=True, normalize_labels=False,
                surrogate_mapping=None, feature_type='sparse',
                pretrained_shortlist=None, beta=0.5,
                use_intermediate_for_shorty=True, **kwargs):
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
        trn_pretrained_shortlist: csr_matrix or None, default=None
            Shortlist for test dataset
            * will directly use this this shortlist when available
        use_intermediate_for_shorty: bool
            use intermediate representation for negative sampling/ANN

        Returns
        -------
        predicted_labels: csr_matrix
            predictions for the given dataset
        """
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
            _type='shortlist',
            pretrained_shortlist=pretrained_shortlist,
            keep_invalid=keep_invalid,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            feature_indices=feature_indices,
            label_indices=label_indices,
            surrogate_mapping=surrogate_mapping)
        data_loader = self._create_data_loader(
            feature_type=feature_type,
            classifier_type='shortlist',
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers)
        time_begin = time.time()
        predicted_labels = self._predict(
            data_loader, top_k, beta, use_intermediate_for_shorty)
        time_end = time.time()
        prediction_time = time_end - time_begin
        avg_prediction_time = prediction_time*1000/len(data_loader.dataset)
        acc = self.evaluate(dataset.labels.data, predicted_labels)
        _res = self._format_acc(acc)
        self.logger.info(
            "Prediction time (total): {:.2f} sec., "
            "Prediction time (per sample): {:.2f} msec., "
            "P@k(%): {:s}".format(
                prediction_time,
                avg_prediction_time, _res))
        return predicted_labels, prediction_time, avg_prediction_time

    def save_checkpoint(self, model_dir, epoch):
        # Avoid purge call from base class
        super().save_checkpoint(model_dir, epoch, False)
        if self.shorty is not None:
            self.tracking.saved_checkpoints[-1]['ANN'] \
                = 'checkpoint_ANN_{}.pkl'.format(epoch)
            self.shorty.save(os.path.join(
                model_dir, self.tracking.saved_checkpoints[-1]['ANN']))
        self.purge(model_dir)

    def load_checkpoint(self, model_dir, fname, epoch):
        super().load_checkpoint(model_dir, fname, epoch)
        if self.shorty is not None:
            fname = os.path.join(model_dir, 'checkpoint_ANN_{}'.format(epoch))
            self.shorty.load(fname)

    def save(self, model_dir, fname):
        super().save(model_dir, fname)
        if self.shorty is not None:
            self.shorty.save(os.path.join(model_dir, fname+'_ANN'))

    def load(self, model_dir, fname):
        super().load(model_dir, fname)
        if self.shorty is not None:
            self.shorty.load(os.path.join(model_dir, fname+'_ANN'))

    def purge(self, model_dir):
        if self.shorty is not None:
            if len(self.tracking.saved_checkpoints) \
                    > self.tracking.checkpoint_history:
                fname = self.tracking.saved_checkpoints[0]['ANN']
                self.shorty.purge(fname)  # let the class handle the deletion
        super().purge(model_dir)

    @property
    def model_size(self):
        s = self.net.model_size
        if self.shorty is not None:
            return s + self.shorty.model_size
        return s


class ModelNS(ModelBase):
    """
    Models with negative sampling
    * When the negative sampling is done on the fly

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
    shorty: libs.shortlist.Shortlist
        to generate a shortlist of labels
    """

    def __init__(self, params, net, criterion, optimizer, shorty):
        super().__init__(params, net, criterion, optimizer)
        self.shorty = shorty
        self.feature_indices = params.feature_indices
        self.label_indices = params.label_indices

    def _strip_padding_label(self, mat, num_labels):
        """
        Strip padding label from a matrix
        * Useful when a padding label is used in a classifier/shortlist
        * Support for multiple matrics (expects dictionary) 
        """
        stripped_vals = {}
        for key, val in mat.items():
            stripped_vals[key] = val[:, :num_labels].tocsr()
            del val
        return stripped_vals

    def fit(self, data_dir, model_dir, result_dir, dataset,
            learning_rate, num_epochs, data=None,
            trn_feat_fname='trn_X_Xf.txt', trn_label_fname='trn_X_Y.txt',
            val_feat_fname='tst_X_Xf.txt', val_label_fname='tst_X_Y.txt',
            batch_size=128, num_workers=4, shuffle=False, init_epoch=0,
            keep_invalid=False, feature_indices=None, label_indices=None,
            normalize_features=True, normalize_labels=False, validate=False,
            validate_after=5, *args, **kwargs):
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
        """
        self.logger.info("Loading training data.")

        train_dataset = self._create_dataset(
            os.path.join(data_dir, dataset),
            fname_features=trn_feat_fname,
            fname_labels=trn_label_fname,
            data=data,
            mode='train',
            keep_invalid=keep_invalid,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            feature_indices=feature_indices,
            label_indices=label_indices,
            shortlist_method='dynamic',
            shorty=self.shorty)
        train_loader = self._create_data_loader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle)
        # No need to update embeddings
        if self.freeze_embeddings:
            self.logger.info(
                "Computing and reusing document embeddings"
                "to save computations.")
            data = {'X': None, 'Y': None}
            data['X'] = self._document_embeddings(train_loader)
            data['Y'] = train_dataset.labels.data
            #  Invalid labels already removed
            train_dataset = self._create_dataset(
                os.path.join(data_dir, dataset),
                data=data,
                fname_features=None,
                mode='train',
                shortlist_method='dynamic',
                feature_type='dense',
                normalize_features=False,
                keep_invalid=True)
            train_loader = self._create_data_loader(
                train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
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
                keep_invalid=keep_invalid,
                normalize_features=normalize_features,
                normalize_labels=normalize_labels,
                feature_indices=feature_indices,
                label_indices=label_indices,
                size_shortlist=-1)  # No shortlist during prediction
            validation_loader = self._create_data_loader(
                validation_dataset,
                batch_size=batch_size,
                num_workers=num_workers)
        self._fit(train_loader, validation_loader,
                  model_dir, result_dir, init_epoch,
                  num_epochs, validate_after)
        train_time = self.tracking.train_time + self.tracking.shortlist_time
        return train_time, self.model_size


class ModelReRanker(ModelShortlist):
    """
    Models with a pre-computed label shortlist

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
    shorty: libs.shortlist.Shortlist
        to generate a shortlist of labels
    """

    def __init__(self, params, net, criterion, optimizer, shorty):
        super().__init__(params, net, criterion, optimizer, shorty)

    def _combine_scores(self, logit, sim, beta):
        return beta*logit + (1-beta)*sim

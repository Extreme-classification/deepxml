import logging
import math
import os
import time
from scipy.sparse import lil_matrix
import _pickle as pickle
import sys
import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np
import xctools.evaluation.xc_metrics as xc_metrics
import sys
import libs.data_loader as data_loader
import libs.shortlist_utils as shortlist_utils
import libs.utils as utils
from libs.tracking import Tracking


class Model(object):
    def __init__(self, params, net, criterion, optimizer, shorty=None):
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.shorty = shorty
        self.num_centroids = params.num_centroids
        self.learning_rate = params.learning_rate
        self.current_epoch = 0
        self.last_saved_epoch = -1
        self.model_dir = params.model_dir
        self.label_padding_index = params.label_padding_index
        self.validate = params.validate
        self.last_epoch = 0
        self.shortlist_size = params.num_nbrs
        self.dlr_step = params.dlr_step
        self.dlr_factor = params.dlr_factor
        self.use_shortlist = params.use_shortlist
        self.progress_step = 500
        self.feature_indices = params.feature_indices
        self.label_indices = params.label_indices
        self.freeze_embeddings = params.freeze_embeddings
        self.logger = self.get_logger()
        self.device_embeddings = torch.device(
            params.device_embeddings if torch.cuda.is_available() else "cpu")
        self.device_classifier = torch.device(
            params.device_classifier if torch.cuda.is_available() else "cpu")
        self.embedding_dims = params.embedding_dims
        self.tracking = Tracking()
        self.retrain_hnsw_after = params.retrain_hnsw_after
        self.update_shortlist = params.update_shortlist
        self.model_fname = params.model_fname


    def _create_dataset(self, data_dir, fname, mode='predict', use_shortlist=False, normalize_features=True, keep_invalid=False):
        embeddings=None
        if self.freeze_embeddings:
            embeddings = self.net.embeddings.weight.cpu().detach().data[1:].numpy()
        _dataset = data_loader.XMLDataset(data_dir, fname, mode=mode, use_shortlist=use_shortlist,
                                                     size_shortlist=self.shortlist_size,
                                                     feature_indices=self.feature_indices,
                                                     label_indices=self.label_indices,
                                                     normalize_features=normalize_features,
                                                     keep_invalid=keep_invalid,
                                                     num_centroids=self.num_centroids,
                                                     model_dir=self.model_dir,
                                                     embeddings=embeddings)
        return _dataset


    def _create_data_loader(self, dataset, batch_size=128, num_workers=4, shuffle=False, mode='predict'):
        if dataset.use_shortlist:
            collate_fn = data_loader.collate_fn_sl
            if dataset.use_fixed:
                collate_fn = data_loader.collate_fn_fx_sl
        else:
            collate_fn = data_loader.collate_fn_full
            if dataset.use_fixed:
                collate_fn = data_loader.collate_fn_fx_full
        
        dt_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              collate_fn= collate_fn,
                              shuffle=shuffle)
        return dt_loader

    def get_logger(self, name='DeepXML', level=logging.INFO):
        """
            Return logging object!
        """
        logging.basicConfig(level=level, stream=sys.stdout)
        logger= logging.getLogger(name)
        return logger

    def _step_full(self, data_loader):
        self.net.train()
        torch.set_grad_enabled(True)
        num_batches = data_loader.dataset.num_samples//data_loader.batch_size
        mean_loss = 0
        for batch_idx, (batch_features, batch_weights, batch_labels) in enumerate(data_loader):
            self.net.zero_grad()
            batch_size = batch_labels.size(0)
            out_ans = self.net.forward(batch_features.to(
                self.device_embeddings), batch_weights.to(self.device_embeddings))
            loss = self.criterion(
                out_ans, batch_labels.to(self.device_classifier))
            mean_loss += loss.item()*batch_size
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.progress_step == 0:
                self.logger.info(
                    "Training progress: [{}/{}]".format(batch_idx, num_batches))
        del batch_features, batch_weights, batch_labels
        return mean_loss / data_loader.dataset.num_samples

    def _step_sl(self, data_loader):
        self.net.train()
        torch.set_grad_enabled(True)
        num_batches = data_loader.dataset.num_samples//data_loader.batch_size
        mean_loss = 0
        for batch_idx, (batch_features, batch_weights, batch_shortlist, batch_labels_mask, batch_dist) in enumerate(data_loader):
            self.net.zero_grad()
            out_ans = self.net.forward(batch_features.to(self.device_embeddings), batch_weights.to(
                self.device_embeddings), batch_shortlist.to(self.device_classifier))
            # Note: Loss is summed over labels but averaged over batch!
            _out_ans = self.net._get_logits_train(out_ans, batch_dist.to(
                self.device_classifier), batch_labels_mask.to(self.device_classifier))
            loss = self.criterion(
                _out_ans, batch_labels_mask.to(self.device_classifier))/batch_features.size(0)
            mean_loss += loss.item()*batch_features.size(0)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.progress_step == 0:
                self.logger.info(
                    "Training progress: [{}/{}]".format(batch_idx, num_batches))
        del batch_features, batch_weights, batch_labels_mask, batch_shortlist
        return mean_loss / data_loader.dataset.num_samples

    def _adjust_parameters(self):
        self.optimizer.adjust_lr(self.dlr_factor)
        self.learning_rate *= self.dlr_factor
        self.dlr_step = max(5, self.dlr_step//2)
        self.logger.info(
            "Adjusted learning rate to: {}".format(self.learning_rate))

    def validate_full(self, data_loader):
        self.net.eval()
        torch.set_grad_enabled(False)
        num_batches = data_loader.dataset.num_samples//data_loader.batch_size
        mean_loss = 0
        predicted_labels = lil_matrix((data_loader.dataset.num_samples,
                                       data_loader.dataset.num_labels))
        count = 0
        for batch_idx, (batch_features, batch_weights, batch_labels) in enumerate(data_loader):
            batch_size = batch_labels.size(0)
            out_ans = self.net.forward(batch_features.to(
                self.device_embeddings), batch_weights.to(self.device_embeddings))
            loss = self.criterion(
                out_ans, batch_labels.to(self.device_classifier))
            mean_loss += loss.item()*batch_size
            # TODO Check if batch_dist and batch_shortlist need to be transferred to CPU again.
            utils.update_predicted(
                count, batch_size, out_ans.data, predicted_labels)
            count += batch_size
            if batch_idx % self.progress_step == 0:
                self.logger.info(
                    "Validation progress: [{}/{}]".format(batch_idx, num_batches))
        return predicted_labels, mean_loss / data_loader.dataset.num_samples

    def validate_sl(self, data_loader, beta=0.2):
        self.net.eval()
        torch.set_grad_enabled(False)
        num_labels = data_loader.dataset.num_labels
        offset = 1 if self.label_padding_index is not None else 0
        _num_labels = data_loader.dataset.num_labels + offset
        num_batches = data_loader.dataset.num_samples//data_loader.batch_size
        mean_loss = 0
        predicted_labels = lil_matrix((data_loader.dataset.num_samples,
                                       _num_labels))
        predicted_labels_knn = lil_matrix((data_loader.dataset.num_samples,
                                           _num_labels))
        predicted_labels_clf = lil_matrix((data_loader.dataset.num_samples,
                                           _num_labels))
        count = 0
        for batch_idx, (batch_features, batch_weights, batch_shortlist, batch_labels_mask, batch_dist) in enumerate(data_loader):
            batch_size = batch_shortlist.size(0)
            out_ans = self.net.forward(batch_features.to(self.device_embeddings), batch_weights.to(
                self.device_embeddings), batch_shortlist.to(self.device_classifier))
            _out_ans = self.net._get_logits_train(out_ans, batch_dist.to(
                self.device_classifier), batch_labels_mask.to(self.device_classifier))
            loss = self.criterion(
                _out_ans, batch_labels_mask.to(self.device_classifier))/batch_size
            mean_loss += loss.item()*batch_size
            # TODO Check if batch_dist and batch_shortlist need to be transferred to CPU again.

            new_knn, new_clf = self.net.rescale_logits(
                batch_dist.to(self.device_classifier), out_ans.data)

            scores = self.net.rescale_logits._get_score(
                new_knn, new_clf, beta)
            utils.update_predicted_shortlist(
                count, batch_size, new_clf.sigmoid(), predicted_labels_clf, batch_shortlist.numpy(), top_k=self.shortlist_size)
            utils.update_predicted_shortlist(
                count, batch_size, new_knn.sigmoid(), predicted_labels_knn, batch_shortlist.numpy(), top_k=self.shortlist_size)
            utils.update_predicted_shortlist(
                count, batch_size, scores, predicted_labels, batch_shortlist.numpy())
            count += batch_size
            if batch_idx % self.progress_step == 0:
                self.logger.info(
                    "Validation progress: [{}/{}]".format(batch_idx, num_batches))
        return predicted_labels[:, :num_labels], predicted_labels_knn[:, :num_labels], \
            predicted_labels_clf[:, :num_labels], mean_loss / \
            data_loader.dataset.num_samples

    def fit(self, data_dir, model_dir, result_dir, dataset, learning_rate, num_epochs, tr_fname='train.txt', val_fname='test.txt',
            batch_size=128, num_workers=4, shuffle=False, validate=False, beta=0.2, init_epoch=0, keep_invalid=False):
        self.logger.info("Loading training data.")
        train_dataset = self._create_dataset(os.path.join(data_dir, dataset),
                                                fname=tr_fname,
                                                mode='train',
                                                use_shortlist=self.use_shortlist,
                                                keep_invalid=keep_invalid)
        train_loader_shuffle = self._create_data_loader(train_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                shuffle=shuffle)
        train_loader = self._create_data_loader(train_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers)
        self.logger.info("Loading validation data.")
        validation_dataset = self._create_dataset(os.path.join(data_dir, dataset),
                                                fname=val_fname,
                                                mode='predict',
                                                use_shortlist=self.use_shortlist,
                                                keep_invalid=keep_invalid)
        validation_loader = self._create_data_loader(validation_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers)
        for epoch in range(init_epoch, init_epoch+num_epochs):
            if epoch != 0 and self.dlr_step != -1 and epoch % self.dlr_step == 0:
                self._adjust_parameters()
            batch_train_start_time = time.time()
            if self.use_shortlist:
                if epoch % self.retrain_hnsw_after == 0:
                    self.logger.info(
                        "Updating shortlist at epoch: {}".format(epoch))
                    shorty_start_t = time.time()
                    self.shorty.reset()
                    shortlist_utils.update(
                        train_loader, self, self.embedding_dims, self.shorty, flag=0)
                    shortlist_utils.update(
                        validation_loader, self, self.embedding_dims, self.shorty, flag=2)
                    shorty_end_t = time.time()
                    self.logger.info("ANN train time: {} sec".format(
                        shorty_end_t - shorty_start_t))
                    self.tracking.shortlist_time = self.tracking.shortlist_time + \
                        shorty_end_t - shorty_start_t
                    batch_train_start_time = time.time()
                    self.net.rescale_logits._reset()
                    self.net.rescale_logits.train()
                    validation_loader.dataset.save_shortlist(self.model_dir+'/test_shorty.pkl')
                tr_avg_loss = self._step_sl(train_loader_shuffle)
            else:
                tr_avg_loss = self._step_full(train_loader_shuffle)
            self.tracking.mean_train_loss.append(tr_avg_loss)
            batch_train_end_time = time.time()
            self.tracking.train_time = self.tracking.train_time + \
                batch_train_end_time - batch_train_start_time

            self.logger.info("Epoch: {}, loss: {}, time: {} sec".format(
                epoch, tr_avg_loss, batch_train_end_time - batch_train_start_time))
            if self.validate and epoch % 2 == 0:
                val_start_t = time.time()
                if self.use_shortlist:
                    print(self.net.rescale_logits)
                    predicted_labels, predicted_labels_knn, predicted_labels_clf, val_avg_loss = self.validate_sl(
                        validation_loader)
                    _prec_clf, _ = self.evaluate(
                        validation_loader.dataset.labels, predicted_labels_clf, mode=1, size=10)
                    _prec_knn, _ = self.evaluate(
                        validation_loader.dataset.labels, predicted_labels_knn, mode=1, size=self.shortlist_size)
                    self.logger.info("clf: {}, knn: {}".format(
                        _prec_clf[0]*100, _prec_knn[0]*100))
                else:
                    predicted_labels, val_avg_loss = self.validate_full(
                        validation_loader)
                val_end_t = time.time()
                self.tracking.validation_time = self.tracking.validation_time + val_end_t - val_start_t
                _prec, _ndcg = self.evaluate(
                    validation_loader.dataset.labels, predicted_labels)
                self.tracking.val_precision.append(_prec)
                self.tracking.val_precision.append(_ndcg)
                self.logger.info("Model saved after epoch: {}".format(epoch))
                self.save_checkpoint(model_dir, epoch+1)
                self.tracking.last_saved_epoch = epoch
                self.logger.info("P@1: {}, loss: {}, time: {} sec".format(
                    _prec[0]*100, val_avg_loss, val_end_t-val_start_t))
            self.tracking.last_epoch += 1
        
        self.save_checkpoint(model_dir, epoch+1)
        self.tracking.save(os.path.join(result_dir, 'training_statistics.pkl'))
        self.logger.info("Training time: {} sec, Validation time: {} sec, Shortlist time: {} sec".format(
            self.tracking.train_time, self.tracking.validation_time, self.tracking.shortlist_time))

    def post_process_full(self, data_dir, model_dir, result_dir, dataset,
                          tr_fname, batch_size, keep_invalid=False,
                          num_workers=6, mode='retrain_w_shorty'):
        dataset = self._create_dataset(os.path.join(data_dir, dataset),
                                       fname=tr_fname,
                                       mode=mode,
                                       use_shortlist=self.use_shortlist,
                                       keep_invalid=keep_invalid)
        data_loader = self._create_data_loader(dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers)
        self.logger.info("Post-processing!")
        self.shorty.reset()
        shortlist_utils.update(
            data_loader, self, self.embedding_dims, self.shorty, flag=1)

    def get_document_embeddings(self, data_loader):
        """
            Get document embeddings
        """
        self.net.eval()
        torch.set_grad_enabled(False)
        embeddings = torch.zeros(
            data_loader.dataset.num_samples, self.net.repr_dims)
        count = 0

        for _, (batch_features, batch_weights, _, _, _) in enumerate(data_loader):
            batch_size = batch_features.size(0)
            out_ans = self.net.forward(batch_features.to(self.device_embeddings), batch_weights.to(
                self.device_embeddings), return_embeddings=True)
            embeddings[count:count+batch_size, :] = out_ans.cpu()
            count += batch_size
        torch.cuda.empty_cache()
        return embeddings.numpy()

    def transfer_to_devices(self):
        self.net.embeddings.to(self.device_embeddings)
        self.net.transform.to(self.device_embeddings)
        self.net.rescale_logits.to(self.device_classifier)
        self.net.classifier.to(self.device_classifier)
        if self.net.low_rank != -1:
            self.net.low_rank_layer.to(self.device_classifier)
        if self.criterion:
            self.criterion.to(self.device_classifier)
        self.net.device_embeddings = self.device_embeddings
        self.net.device_classifier = self.device_classifier

    def predict(self, data_dir, dataset, ts_fname='test.txt', batch_size=256, num_workers=6, beta=0.2, keep_invalid=False):
        def eval(predictions, true_labels):
            _res = ""
            if isinstance(predictions, tuple):
                acc_knn = self.evaluate(
                    true_labels, predictions[1], mode=1)
                acc_clf = self.evaluate(
                    true_labels, predictions[2], mode=1)
                _res = "clf: {}, knn: {}".format(acc_clf[0]*100, acc_knn[0]*100)
            else:
                acc = self.evaluate(
                    true_labels, predictions, mode=1)
                _res = "clf: {}".format(acc[0]*100)
            return _res

        dataset = self._create_dataset(os.path.join(data_dir, dataset),
                                                fname=ts_fname,
                                                mode='predict',
                                                use_shortlist=self.use_shortlist,
                                                keep_invalid=keep_invalid)
        data_loader = self._create_data_loader(dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers)
        time_begin = time.time()
        if self.use_shortlist:
            predicted_labels = self.predict_sl(data_loader, beta)
        else:
            predicted_labels = self.predict_full(data_loader)
        time_end = time.time()
        prediction_time = time_end - time_begin
        _res = eval(predicted_labels, dataset.labels)
        self.logger.info("Prediction time (total): {} sec., Prediction time (per sample): {} msec., P@k(%): {}".format(
            prediction_time, prediction_time*1000/data_loader.dataset.num_samples, _res))
        return predicted_labels

    def predict_full(self, data_loader):
        self.logger.info("Loading test data.")
        self.net.eval()
        torch.set_grad_enabled(False)
        num_batches = data_loader.dataset.num_samples//data_loader.batch_size
        predicted_labels = lil_matrix((data_loader.dataset.num_samples,
                                       data_loader.dataset.num_labels))
        count = 0
        for batch_idx, (batch_features, batch_weights, _) in enumerate(data_loader):
            batch_size = batch_features.size(0)
            out_ans = self.net.forward(batch_features.to(
                self.device_embeddings), batch_weights.to(self.device_embeddings))
            # TODO Check if batch_dist and batch_shortlist need to be transferred to CPU again.
            utils.update_predicted(
                count, batch_size, out_ans.data, predicted_labels)
            count += batch_size
            if batch_idx % self.progress_step == 0:
                self.logger.info(
                    "Prediction progress: [{}/{}]".format(batch_idx, num_batches))
        return predicted_labels

    def predict_sl(self, data_loader, beta):
        self.logger.info("Loading test data.")
        self.net.eval()
        num_labels = data_loader.dataset.num_labels
        offset = 1 if self.label_padding_index is not None else 0
        _num_labels = data_loader.dataset.num_labels + offset
        torch.set_grad_enabled(False)
        #TODO Add flag for loading or training
        if self.update_shortlist:
            shortlist_utils.update(
                data_loader, self, self.embedding_dims, self.shorty, flag=2)
        else:
            print("Loading Pre-computer shortlist")
            data_loader.dataset.load_shortlist(
                self.model_dir+'/test_shorty.pkl')

        num_batches = data_loader.dataset.num_samples//data_loader.batch_size

        predicted_labels = lil_matrix((data_loader.dataset.num_samples,
                                       _num_labels))
        predicted_labels_knn = lil_matrix((data_loader.dataset.num_samples,
                                           _num_labels))
        predicted_labels_clf = lil_matrix((data_loader.dataset.num_samples,
                                           _num_labels))

        count = 0
        for batch_idx, (batch_features, batch_weights, batch_shortlist, _, batch_dist) in enumerate(data_loader):
            batch_size = batch_shortlist.size(0)
            out_ans = self.net.forward(batch_features.to(self.device_embeddings), batch_weights.to(
                self.device_embeddings), batch_shortlist.to(self.device_classifier))
            # TODO Check if batch_dist and batch_shortlist need to be transferred to CPU again.

            new_knn, new_clf = self.net.rescale_logits(
                batch_dist.to(self.device_classifier), out_ans.data)
            scores = self.net.rescale_logits._get_score(
                new_knn, new_clf, beta)

            utils.update_predicted_shortlist(
                count, batch_size, scores, predicted_labels, batch_shortlist.numpy(), top_k=self.shortlist_size)

            utils.update_predicted_shortlist(
                count, batch_size, new_knn.exp(), predicted_labels_knn, batch_shortlist.numpy(), top_k=self.shortlist_size//10
            )
            utils.update_predicted_shortlist(
                count, batch_size, new_clf.exp(), predicted_labels_clf, batch_shortlist.numpy(), top_k=self.shortlist_size//10
            )

            count += batch_size
            if batch_idx % self.progress_step == 0:
                self.logger.info(
                    "Prediction progress: [{}/{}]".format(batch_idx, num_batches))        
        return (predicted_labels[:, :num_labels].tocsr(), predicted_labels_knn[:, :num_labels].tocsr(), predicted_labels_clf[:, :num_labels].tocsr())

    def save_checkpoint(self, model_dir, epoch):
        checkpoint = {
            'epoch': epoch,
            'criterion': self.criterion.state_dict(),
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        fname = ('checkpoint_net_{}.pkl'.format(epoch), 'checkpoint_ANN_{}.pkl'.format(
            epoch) if self.use_shortlist else None)
        torch.save(checkpoint, os.path.join(model_dir, fname[0]))
        if self.use_shortlist:
            self.shorty.save(os.path.join(model_dir, fname[1]))
        self.tracking.saved_checkpoints.append(fname)
        self.purge(model_dir)

    def load_checkpoint(self, model_dir, fname, epoch):
        fname = os.path.join(model_dir, 'checkpoint_net_{}.pkl'.format(epoch))
        checkpoint = torch.load(open(fname, 'rb'))
        self.net.load_state_dict(checkpoint['net'])
        self.criterion.load_state_dict(checkpoint['criterion'])
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        fname = os.path.join(model_dir, 'checkpoint_ANN_{}.pkl'.format(epoch))
        if self.shorty:
            self.shorty.load(fname)

    def save(self, model_dir, fname, low_rank=-1):
        state_dict = self.net.state_dict()
        torch.save(state_dict, os.path.join(
            model_dir, fname+'_network.pkl'))
        if self.use_shortlist:
            self.shorty.save(os.path.join(model_dir, fname+'_ANN.pkl'))
        if low_rank != -1:
            utils.adjust_for_low_rank(state_dict, low_rank)
            torch.save(state_dict, os.path.join(
                model_dir, fname+'_network_low_rank.pkl'))

    def load(self, model_dir, fname, use_low_rank=False):
        fname_net = fname+'_network.pkl' if not use_low_rank else fname+'_network_low_rank.pkl'
        state_dict = torch.load(
            os.path.join(model_dir, model_dir, fname_net))
        # Append Padding classifier if shapes do not match.
        self.logger.info(utils.append_padding_classifier(
            state_dict, self.net.classifier.output_size))
        self.net.load_state_dict(state_dict)
        if self.use_shortlist and self.shorty is not None:
            self.shorty.load(os.path.join(model_dir, fname+'_ANN.pkl'))

    def purge(self, model_dir):
        if len(self.tracking.saved_checkpoints) > self.tracking.checkpoint_history:
            fname = self.tracking.saved_checkpoints.pop(0)
            self.logger.info("Purging checkpoint: {}".format(fname))
            os.remove(os.path.join(model_dir, fname[0]))
            if fname[1]:
                os.remove(os.path.join(model_dir, fname[1]))

    def evaluate(self, true_labels, predicted_labels, mode=0, size=300):
        acc = xc_metrics.Metrices(true_labels)
        acc = acc.eval(predicted_labels.tocsr(), 5)
        if mode:
            recall = utils._recall(true_labels, predicted_labels)
            self.logger.info("Recall@%d: %0.2f" % (size, 100.0*recall))
        return acc

    def get_size(self):
        total = 0
        for key,val in self.net.__dict__['_modules'].items():
            pytorch_total_params = np.sum([p.numel() for p in val.parameters() if p.requires_grad])
            print(key,pytorch_total_params*4/(1024*1024*1024))
            total += pytorch_total_params*4/(1024*1024*1024)
        if self.use_shortlist:
            size = os.path.getsize(self.model_dir+'/'+self.model_fname+'_ANN.pkl')
            print("ANN",int(size)/(1024*1024*1024))
            total += int(size)/(1024*1024*1024)
        return total



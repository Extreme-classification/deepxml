import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import os
import numpy as np
import models.custom_embeddings as custom_embeddings
import models.hash_embeddings as hash_embeddings
import models.transform_layer as transform_layer
import models.sparse_linear as sparse_linear
import models.attention_layer as attention_layer
import models.rnn_layer as rnn_layer


__author__ = 'KD'


class DeepXML(nn.Module):
    """
        DeepXML: A Scalable Deep learning approach for eXtreme Multi-label Learning
    """

    def __init__(self, params):
        super(DeepXML, self).__init__()
        self.vocabulary_dims = params.vocabulary_dims+1
        self.hidden_dims = params.embedding_dims
        self.embedding_dims = params.embedding_dims
        self.trans_method = params.trans_method
        self.dropout = params.dropout
        self.num_clf_partitions = params.num_clf_partitions
        self.num_labels = params.num_labels
        self.use_hash_embeddings = params.use_hash_embeddings
        self.num_buckets = params.num_buckets
        self.num_hashes = params.num_hashes
        self.label_padding_index = params.label_padding_index
        self.use_shortlist = params.use_shortlist
        self.append_weight = True
        self.low_rank = params.low_rank if params.use_low_rank else -1
        assert params.use_low_rank == (
            self.low_rank != -1), "Sorry, can't train with negative rank! Go read about positive numbers."
        self.use_residual = params.use_residual
        # Hash embeddings append weights
        # TODO: will not work for aggregation_mode 'concat'
        self.pt_repr_dims, self.repr_dims = self._compute_rep_dims()

        if self.use_hash_embeddings:
            assert self.num_buckets != -1, "#buckets must be positive"
            assert self.num_hashes != -1, "#hashes must be positive"
            self.embeddings = hash_embeddings.HashEmbedding(vocabulary_dims=self.vocabulary_dims,
                                                            embedding_dims=self.embedding_dims,
                                                            num_hashes=self.num_hashes,
                                                            num_buckets=self.num_buckets,
                                                            aggregation_mode='sum',
                                                            mask_zero=False,
                                                            append_weight=self.append_weight,
                                                            seed=None,
                                                            sparse=True)
        else:
            self.embeddings = custom_embeddings.CustomEmbedding(num_embeddings=self.vocabulary_dims,
                                                                embedding_dim=self.embedding_dims,
                                                                padding_idx=params.padding_idx,
                                                                scale_grad_by_freq=False,
                                                                sparse=True)

        self.transform = transform_layer.Transform(hidden_dims=params.hidden_dims,
                                                   embedding_dims=self.pt_repr_dims,
                                                   trans_method=params.trans_method,
                                                   dropout=params.dropout,
                                                   use_residual=params.use_residual,
                                                   res_init=params.res_init,
                                                   use_shortlist=self.use_shortlist
                                                   )
        if self.low_rank != -1:
            self.low_rank_layer = sparse_linear.SparseLinear(
                self.repr_dims, self.low_rank, sparse=False, bias=False)
        offset = self.num_clf_partitions if self.label_padding_index is not None else 0
        if self.num_clf_partitions > 1:  # Run the distributed version
            # TODO: Label padding index
            _sparse = True if self.use_shortlist else False
            _low_rank = [self.low_rank for _ in range(self.num_clf_partitions)]
            # last one is padding index for each partition
            _num_labels = self.num_labels + offset
            _padding_idx = [None for _ in range(self.num_clf_partitions)]
            _bias = [True for _ in range(self.num_clf_partitions)]
            _clf_devices = ["cuda:{}".format(
                idx) for idx in range(self.num_clf_partitions)]
            self.classifier = sparse_linear.ParallelSparseLinear(input_size=self.repr_dims if self.low_rank == -1 else self.low_rank,
                                                                 output_size=_num_labels,
                                                                 sparse=_sparse,
                                                                 low_rank=_low_rank,
                                                                 bias=_bias,
                                                                 padding_idx=_padding_idx,
                                                                 num_partitions=self.num_clf_partitions,
                                                                 devices=_clf_devices)
        else:
            self.classifier = sparse_linear.SparseLinear(self.repr_dims if self.low_rank == -1 else self.low_rank,
                                                         self.num_labels + offset,  # last one is padding index
                                                         sparse=True if self.use_shortlist else False,
                                                         low_rank=self.low_rank,
                                                         padding_idx=self.label_padding_index)
        self.device_embeddings = torch.device(
            "cuda:0")  # Keep embeddings on first device

    def _compute_rep_dims(self):
        pt_repr_dims = self.embedding_dims
        pt_repr_dims += self.num_hashes if (
            self.use_hash_embeddings and self.append_weight == True) else 0
        rep_dims = pt_repr_dims
        if self.trans_method == 'deep_non_linear' or self.use_residual:
            rep_dims = self.hidden_dims
        return pt_repr_dims, rep_dims

    def forward(self, batch_data, return_embeddings=False):
        """
            Forward pass
            Args:
                batch_data['X']: torch.LongTensor: feature indices
                batch_data['X_w]: torch.Tensor: feature weights in case of sparse features
                batch_data['Y_s]: torch.LongTensor: Relevant labels for each sample
                return_embeddings: boolean: Return embeddings or classify
            Returns:
                out: logits for each label
        """
        if 'X_w' in batch_data:  # Sparse features
            embed = self.embeddings(batch_data['X'].to(
                self.device_embeddings),
                batch_data['X_w'].to(self.device_embeddings))
        else:  # Dense features
            embed = batch_data['X'].to(self.device_embeddings)
        embed = self.transform(embed)
        if return_embeddings:
            out = embed
        else:
            if self.low_rank != -1:
                embed = self.low_rank_layer(embed)
            batch_shortlist = None if self.num_clf_partitions == 1 else [
                None]*self.num_clf_partitions
            if 'Y_s' in batch_data:  # Use shortlist
                batch_shortlist = batch_data['Y_s']
            out = self.classifier(embed, batch_shortlist)
        return out

    def initialize_embeddings(self, word_embeddings):
        """
            Initialize embeddings from existing ones
            Args:
                word_embeddings: numpy array: existing embeddings
        """
        self.embeddings.weight.data.copy_(torch.from_numpy(word_embeddings))

    def initialize_classifier(self, clf_weights):
        """
            Initialize classifier from existing weights
            Args:
                clf_weights: numpy.ndarray: (num_labels, repr_dims+1) last dimension is bias
        """
        self.classifier.weight.data.copy_(torch.from_numpy(clf_weights[:, -1]))
        self.classifier.bias.data.copy_(
            torch.from_numpy(clf_weights[:, -1]).view(-1, 1))

    def get_clf_weights(self):
        self.classifier.get_weights()

    def to_device(self):
        self.embeddings.to_device()
        self.transform.to_device()
        self.classifier.to_device()


class DeepSeqXML(nn.Module):
    """
        DeepSeqXML: A Scalable Sequential Deep learning approach for eXtreme Multi-label Learning
    """

    def __init__(self, params):
        super(DeepSeqXML, self).__init__()
        self.vocabulary_dims = params.vocabulary_dims+1
        self.hidden_dims = params.embedding_dims
        self.embedding_dims = params.embedding_dims
        self.trans_method = params.trans_method
        self.dropout = params.dropout
        self.bidirectional = params.bidirectional
        self.num_clf_partitions = params.num_clf_partitions
        self.num_labels = params.num_labels
        self.label_padding_index = params.label_padding_index
        self.use_shortlist = params.use_shortlist
        self.append_weight = True
        self.low_rank = params.low_rank if params.use_low_rank else -1
        assert params.use_low_rank == (
            self.low_rank != -1), "Sorry, can't train with negative rank! Go read about positive numbers."
        self.use_residual = params.use_residual

        self.pt_repr_dims, self.repr_dims = self._compute_rep_dims()

        self.embeddings = nn.Embedding(num_embeddings=self.vocabulary_dims,
                                       embedding_dim=self.embedding_dims,
                                       padding_idx=params.padding_idx,
                                       scale_grad_by_freq=False,
                                       sparse=True)
        self.rnn = rnn_layer.RNN(input_size=self.pt_repr_dims,
                                 cell_type='LSTM',
                                 num_layers=1,
                                 hidden_size=self.hidden_dims,
                                 bidirectional=self.bidirectional,
                                 dropout=0.2,
                                 batch_first=True)
        self.attention = attention_layer.Attention(
            in_dimensions=self.repr_dims, hidden_dimensions=self.hidden_dimensions)
        if self.low_rank != -1:
            self.low_rank_layer = sparse_linear.SparseLinear(
                self.repr_dims, self.low_rank, sparse=False, bias=False)
        offset = self.num_clf_partitions if self.label_padding_index is not None else 0
        if self.num_clf_partitions > 1:  # Run the distributed version
            # TODO: Label padding index
            _sparse = True if self.use_shortlist else False
            _low_rank = [self.low_rank for _ in range(self.num_clf_partitions)]
            # last one is padding index for each partition
            _num_labels = self.num_labels + offset
            _padding_idx = [None for _ in range(self.num_clf_partitions)]
            _bias = [True for _ in range(self.num_clf_partitions)]
            _clf_devices = ["cuda:{}".format(
                idx) for idx in range(self.num_clf_partitions)]
            self.classifier = sparse_linear.ParallelSparseLinear(input_size=self.repr_dims if self.low_rank == -1 else self.low_rank,
                                                                 output_size=_num_labels,
                                                                 sparse=_sparse,
                                                                 low_rank=_low_rank,
                                                                 bias=_bias,
                                                                 padding_idx=_padding_idx,
                                                                 num_partitions=self.num_clf_partitions,
                                                                 devices=_clf_devices)
        else:
            self.classifier = sparse_linear.SparseLinear(self.repr_dims if self.low_rank == -1 else self.low_rank,
                                                         self.num_labels + offset,  # last one is padding index
                                                         sparse=True if self.use_shortlist else False,
                                                         low_rank=self.low_rank,
                                                         padding_idx=self.label_padding_index)
        self.device_embeddings = torch.device(
            "cuda:0")  # Keep embeddings on first device

    def _compute_rep_dims(self):
        pt_repr_dims = self.embedding_dims
        rep_dims = self.hidden_dims*2 if self.bidirectional else self.hidden_dims
        return pt_repr_dims, rep_dims

    def forward(self, batch_data, return_embeddings=False):
        """
            Forward pass
            Args:
                batch_data['X']: torch.LongTensor: feature indices
                batch_data['X_l']: torch.LongTensor: length of each document
                batch_data['Y_s]: torch.LongTensor: Relevant labels for each sample
                return_embeddings: boolean: Return embeddings or classify
            Returns:
                out: logits for each label
        """
        embed = self.embeddings(batch_data['X'])
        embed, _ = self.rnn(embed, batch_data['X_l'])
        embed = self.attention(embed)
        if return_embeddings:
            out = embed
        else:
            if self.low_rank != -1:
                embed = self.low_rank_layer(embed)
            batch_shortlist = None if self.num_clf_partitions == 1 else [
                None]*self.num_clf_partitions
            if 'Y_s' in batch_data:  # Use shortlist
                batch_shortlist = batch_data['Y_s']
            out = self.classifier(embed, batch_shortlist)
        return out

    def initialize_embeddings(self, word_embeddings):
        """
            Initialize embeddings from existing ones
            Args:
                word_embeddings: numpy array: existing embeddings
        """
        self.embeddings.weight.data.copy_(torch.from_numpy(word_embeddings))

    def initialize_classifier(self, clf_weights):
        """
            Initialize classifier from existing weights
            Args:
                clf_weights: numpy.ndarray: (num_labels, repr_dims+1) last dimension is bias
        """
        self.classifier.weight.data.copy_(torch.from_numpy(clf_weights[:, -1]))
        self.classifier.bias.data.copy_(
            torch.from_numpy(clf_weights[:, -1]).view(-1, 1))

    def get_clf_weights(self):
        self.classifier.get_weights()

    def to_device(self):
        self.embeddings.to_device()
        self.transform.to_device()
        self.classifier.to_device()

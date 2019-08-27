import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import os
import numpy as np
import models.custom_embeddings as custom_embeddings
import models.hash_embeddings as hash_embeddings
import models.elemental_transform as transform_layer
import models.linear_layer as linear_layer
import models.attention_layer as attention_layer
import models.rnn_layer as rnn_layer


__author__ = 'KD'


class DeepXMLBase(nn.Module):
    """DeepXMLBase: Base class for DeepXML architecture
    Parameters:
    ----------
    vocabulary_dims: int
        number of tokens in the vocabulary
    num_labels: int
        number of labels
    embedding_dims: int
        size of word/token representations
    trans_config: list of strings
        configuration of the transformation layer
    dropout: float, default=0.5
        value of dropout
    num_clf_partitions: int, default=1
        partition classifier in these many parts
    padding_idx: int, default=0
        padding index in words embedding layer
    """

    def __init__(self, vocabulary_dims, num_labels, embedding_dims, 
                 trans_config, dropout=0.5, num_clf_partitions=1, 
                 padding_idx=0):
        super(DeepXMLBase, self).__init__()
        self.vocabulary_dims = vocabulary_dims+1
        self.embedding_dims = embedding_dims
        self.trans_config = trans_config
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.num_clf_partitions = num_clf_partitions
        self.num_labels = num_labels
        self.pt_repr_dims, self.repr_dims = self._compute_rep_dims()
        self.embeddings = self._construct_embedding()
        self.transform = self._construct_transform(trans_config)
        self.classifier = self._construct_classifier()
        # Keep embeddings on first device
        self.device_embeddings = torch.device("cuda:0")

    def _construct_embedding(self):
        return custom_embeddings.CustomEmbedding(
            num_embeddings=self.vocabulary_dims,
            embedding_dim=self.embedding_dims,
            padding_idx=self.padding_idx,
            scale_grad_by_freq=False,
            sparse=True)

    def _construct_classifier(self):
        if self.num_clf_partitions > 1:  # Run the distributed version
            _bias = [True for _ in range(self.num_clf_partitions)]
            _clf_devices = ["cuda:{}".format(
                idx) for idx in range(self.num_clf_partitions)]
            return linear_layer.ParallelLinear(
                input_size=self.repr_dims,
                output_size=self.num_labels,
                bias=_bias,
                num_partitions=self.num_clf_partitions,
                devices=_clf_devices)
        else:
            return linear_layer.Linear(
                input_size=self.repr_dims,
                output_size=self.num_labels,  # last one is padding index
                bias=True
            )

    def _construct_transform(self, trans_config):
        return transform_layer.Transform(transform_layer.get_functions(trans_config))

    def _compute_rep_dims(self):
        return 300, 300

    def encode(self, batch_data):
        """encode documents
        Parameters:
        -----------
        batch_data: dict
            batch_data['X']: torch.Tensor
                dense feature vector or
                feature indices in case of sparse features
            batch_data['X_w]: torch.Tensor
                feature weights in case of sparse features

        Returns
        -------
        out: torch.Tensor
            encoding of a document
        """
        if 'X_w' in batch_data:  # Sparse features
            embed = self.embeddings(
                batch_data['X'].to(self.device_embeddings),
                batch_data['X_w'].to(self.device_embeddings))
        else:  # Dense features
            embed = batch_data['X'].to(self.device_embeddings)
        return self.transform(embed)

    def forward(self, batch_data):
        """Forward pass
        Parameters:
        -----------
        batch_data: dict
            batch_data['X']: torch.Tensor
                dense feature vector or
                feature indices in case of sparse features
            batch_data['X_w]: torch.Tensor
                feature weights in case of sparse features
        Returns
        -------
        out: logits for each label
        """
        return self.classifier(self.encode(batch_data))

    def initialize_embeddings(self, word_embeddings):
        """Initialize embeddings from existing ones
        Parameters:
        -----------
        word_embeddings: numpy array
            existing embeddings
        """
        self.embeddings.weight.data.copy_(torch.from_numpy(word_embeddings))

    def initialize_classifier(self, clf_weights):
        """Initialize classifier from existing weights
        Parameters:
        -----------
        clf_weights: numpy.ndarray
            (num_labels, repr_dims+1) last dimension is bias
        """
        self.classifier.weight.data.copy_(torch.from_numpy(clf_weights[:, -1]))
        self.classifier.bias.data.copy_(
            torch.from_numpy(clf_weights[:, -1]).view(-1, 1))

    def get_clf_weights(self):
        """Get classifier weights
        """
        return self.classifier.get_weights()

    def to(self):
        """Send layers to respective devices
        """
        self.embeddings.to()
        self.transform.to()
        self.transform_fine.to()
        self.classifier.to()


class DeepXMLh(DeepXMLBase):
    """DeepXMLh: Head network for DeepXML architecture
    Allows additional transform layer
    """

    def __init__(self, params):
        transform_config_dict = transform_layer.fetch_json(params.trans_method, params)
        print(transform_config_dict)
        params.trans_config_coarse = transform_config_dict['transform_coarse']
        params.trans_config_fine = transform_config_dict['transform_fine']
        super(DeepXMLh, self).__init__(
            vocabulary_dims=params.vocabulary_dims,
            num_labels=params.num_labels,
            embedding_dims=params.embedding_dims,
            trans_config=params.trans_config_coarse,
            dropout=params.dropout,
            num_clf_partitions=params.num_clf_partitions,
            padding_idx=params.padding_idx)
        print("Hardcoding configs for now.......")
        self.transform_fine = self._construct_transform(params.trans_config_fine)

    def encode(self, batch_data, return_coarse=False):
        """encode documents
        Parameters:
        -----------
        batch_data: dict
            batch_data['X']: torch.Tensor
                dense feature vector or
                feature indices in case of sparse features
            batch_data['X_w]: torch.Tensor
                feature weights in case of sparse features
        return_coarse: boolean, default=False
            whether to return intermediate representation

        Returns
        -------
        out: torch.Tensor
            encoding of a document
        """
        encoding = super().encode(batch_data)
        return encoding if return_coarse else self.transform_fine(encoding)

    def to(self):
        """Send layers to respective devices
        """
        super().to()


class DeepXMLt(DeepXMLBase):
    """DeepXMLt: Tail network for DeepXML architecture
    Allows additional transform layer and shortlist
    """

    def __init__(self, params):
        transform_config_dict = transform_layer.fetch_json(params.trans_method, params)
        params.trans_config_coarse = transform_config_dict['transform_coarse']
        params.trans_config_fine = transform_config_dict['transform_fine']
        self.label_padding_index = params.label_padding_index
        super(DeepXMLt, self).__init__(
            vocabulary_dims=params.vocabulary_dims,
            num_labels=params.num_labels,
            embedding_dims=params.embedding_dims,
            trans_config=params.trans_config_coarse,
            dropout=params.dropout,
            num_clf_partitions=params.num_clf_partitions,
            padding_idx=params.padding_idx)
        self.transform_fine = self._construct_transform(params.trans_config_fine)

    def encode(self, batch_data, return_coarse=False):
        """encode documents
        Parameters:
        -----------
        batch_data: dict
            batch_data['X']: torch.Tensor
                dense feature vector or
                feature indices in case of sparse features
            batch_data['X_w]: torch.Tensor
                feature weights in case of sparse features
        return_coarse: boolean, default=False
            whether to return intermediate representation

        Returns
        -------
        out: torch.Tensor
            encoding of a document
        """
        encoding = super().encode(batch_data)
        return encoding if return_coarse else self.transform_fine(encoding)

    def forward(self, batch_data):
        """Forward pass
        Parameters:
        -----------
        batch_data: dict
            batch_data['X']: torch.Tensor
                dense feature vector or
                feature indices in case of sparse features
            batch_data['X_w]: torch.Tensor
                feature weights in case of sparse features
            batch_data['Y_s]: torch.LongTensor
                indices of label shortlist
        Returns
        -------
        out: logits for each label in the shortlist
        """
        return self.classifier(
            self.encode(batch_data),
            batch_data['Y_s'])

    def _construct_classifier(self):
        offset = 0
        if self.label_padding_index:
            offset = self.num_clf_partitions
        if self.num_clf_partitions > 1:  # Run the distributed version
            # TODO: Label padding index
            # last one is padding index for each partition
            _num_labels = self.num_labels + offset
            _padding_idx = [None for _ in range(self.num_clf_partitions)]
            _bias = [True for _ in range(self.num_clf_partitions)]
            _clf_devices = ["cuda:{}".format(
                idx) for idx in range(self.num_clf_partitions)]
            return linear_layer.ParallelSparseLinear(
                input_size=self.repr_dims,
                output_size=_num_labels,
                bias=_bias,
                padding_idx=_padding_idx,
                num_partitions=self.num_clf_partitions,
                devices=_clf_devices)
        else:
            return linear_layer.SparseLinear(
                input_size=self.repr_dims,
                output_size=self.num_labels + offset,  # last one is padding index
                padding_idx=self.label_padding_index,
                bias=True)

    def to(self):
        """Send layers to respective devices
        """
        super().to()

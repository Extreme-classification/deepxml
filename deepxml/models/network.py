import torch
import torch.nn as nn
import numpy as np
import math
import os
import models.custom_embeddings as custom_embeddings
import models.transform_layer as transform_layer
import models.linear_layer as linear_layer


__author__ = 'KD'


class DeepXMLBase(nn.Module):
    """DeepXMLBase: Base class for DeepXML architecture

    * Identity op as classifier by default
    (derived class should implement it's own classifier)
    * embedding and classifier shall automatically transfer
    the vector to the appropriate device

    Arguments:
    ----------
    vocabulary_dims: int
        number of tokens in the vocabulary
    embedding_dims: int
        size of word/token representations
    trans_config: list of strings
        configuration of the transformation layer
    padding_idx: int, default=0
        padding index in words embedding layer
    """

    def __init__(self, vocabulary_dims, embedding_dims,
                 trans_config, padding_idx=0):
        super(DeepXMLBase, self).__init__()
        self.vocabulary_dims = vocabulary_dims+1
        self.embedding_dims = embedding_dims
        self.trans_config = trans_config
        self.padding_idx = padding_idx
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
        return nn.Identity()

    def _construct_transform(self, trans_config):
        return transform_layer.Transform(
            transform_layer.get_functions(trans_config))

    @property
    def representation_dims(self):
        return self.transform.representation_dims

    def encode(self, x, x_ind=None):
        """Forward pass
        * Assumes features are dense if x_ind is None

        Arguments:
        -----------
        x: torch.FloatTensor
            (sparse features) contains weights of features as per x_ind or
            (dense features) contains the dense representation of a point
        x_ind: torch.LongTensor or None, optional (default=None)
            contains indices of features (sparse features)

        Returns
        -------
        out: logits for each label
        """
        if x_ind is None:
            embed = x.to(self.device_embeddings)
        else:
            embed = self.embeddings(
                x_ind.to(self.device_embeddings),
                x.to(self.device_embeddings))
        return self.transform(embed)

    def forward(self, batch_data):
        """Forward pass
        * Assumes features are dense if X_w is None
        * By default classifier is identity op

        Arguments:
        -----------
        batch_data: dict
            * 'X': torch.FloatTensor
                feature weights for given indices or dense rep.
            * 'X_ind': torch.LongTensor
                feature indices (LongTensor) or None

        Returns
        -------
        out: logits for each label
        """
        return self.classifier(
            self.encode(batch_data['X'], batch_data['X_ind']))

    def initialize_embeddings(self, word_embeddings):
        """Initialize embeddings from existing ones
        Parameters:
        -----------
        word_embeddings: numpy array
            existing embeddings
        """
        self.embeddings.from_pretrained(word_embeddings)

    def to(self):
        """Send layers to respective devices
        """
        self.embeddings.to()
        self.transform.to()
        self.classifier.to()

    def purge(self, fname):
        if os.path.isfile(fname):
            os.remove(fname)

    @property
    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def model_size(self):  # Assumptions: 32bit floats
        return self.num_trainable_params * 4 / math.pow(2, 20)

    def __repr__(self):
        return f"{self.embeddings}\n(Transform): {self.transform}"


class DeepXMLf(DeepXMLBase):
    """DeepXMLf: Network for DeepXML's architecture
    with fully-connected o/p layer (a.k.a 1-vs.-all in literature)

    Allows additional transform layer to transform features from the
    base class. e.g. base class can handle intermediate rep. and transform
    could be used to the intermediate rep. from base class
    """

    def __init__(self, params):
        self.num_labels = params.num_labels
        self.num_clf_partitions = params.num_clf_partitions
        transform_config_dict = transform_layer.fetch_json(
            params.trans_method, params)
        trans_config_coarse = transform_config_dict['transform_coarse']
        super(DeepXMLf, self).__init__(
            vocabulary_dims=params.vocabulary_dims,
            embedding_dims=params.embedding_dims,
            trans_config=trans_config_coarse,
            padding_idx=params.padding_idx)
        trans_config_fine = transform_config_dict['transform_fine']
        self.transform_fine = self._construct_transform(
            trans_config_fine)

    def encode(self, x, x_ind=None, return_coarse=False):
        """Forward pass
        * Assumes features are dense if x_ind is None

        Arguments:
        -----------
        x: torch.FloatTensor
            (sparse features) contains weights of features as per x_ind or
            (dense features) contains the dense representation of a point
        x_ind: torch.LongTensor or None, optional (default=None)
            contains indices of features (sparse features)
        return_coarse: boolean, optional (default=False)
            Return coarse features or not

        Returns
        -------
        out: logits for each label
        """
        encoding = super().encode(x, x_ind)
        return encoding if return_coarse else self.transform_fine(encoding)

    def _construct_classifier(self):
        if self.num_clf_partitions > 1:  # Run the distributed version
            _bias = [True for _ in range(self.num_clf_partitions)]
            _clf_devices = ["cuda:{}".format(
                idx) for idx in range(self.num_clf_partitions)]
            return linear_layer.ParallelLinear(
                input_size=self.representation_dims,
                output_size=self.num_labels,
                bias=_bias,
                num_partitions=self.num_clf_partitions,
                devices=_clf_devices)
        else:
            return linear_layer.Linear(
                input_size=self.representation_dims,
                output_size=self.num_labels,  # last one is padding index
                bias=True
            )

    def to(self):
        """Send layers to respective devices
        """
        self.transform_fine.to()
        super().to()

    def initialize_classifier(self, weight, bias=None):
        """Initialize classifier from existing weights

        Arguments:
        -----------
        weight: numpy.ndarray
        bias: numpy.ndarray or None, optional (default=None)
        """
        self.classifier.weight.data.copy_(torch.from_numpy(weight))
        if bias is not None:
            self.classifier.bias.data.copy_(
                torch.from_numpy(bias).view(-1, 1))

    def get_clf_weights(self):
        """Get classifier weights
        """
        return self.classifier.get_weights()

    def __repr__(self):
        s = f"{self.embeddings}\n{self.transform}\n"
        s += f"(Transform fine): {self.transform_fine}"
        s += f"\n(Classifier): {self.classifier}\n"
        return s


class DeepXMLs(DeepXMLBase):
    """DeepXMLt: DeepXML architecture to be trained with
                 a label shortlist
    * Allows additional transform layer for features
    """

    def __init__(self, params):
        transform_config_dict = transform_layer.fetch_json(
            params.trans_method, params)
        trans_config_coarse = transform_config_dict['transform_coarse']
        self.num_labels = params.num_labels
        self.num_clf_partitions = params.num_clf_partitions
        self.label_padding_index = params.label_padding_index
        super(DeepXMLs, self).__init__(
            vocabulary_dims=params.vocabulary_dims,
            embedding_dims=params.embedding_dims,
            trans_config=trans_config_coarse,
            padding_idx=params.padding_idx)
        trans_config_fine = transform_config_dict['transform_fine']
        self.transform_fine = self._construct_transform(
            trans_config_fine)

    def encode(self, x, x_ind=None, return_coarse=False):
        """Forward pass
        * Assumes features are dense if x_ind is None

        Arguments:
        -----------
        x: torch.FloatTensor
            (sparse features) contains weights of features as per x_ind or
            (dense features) contains the dense representation of a point
        x_ind: torch.LongTensor or None, optional (default=None)
            contains indices of features (sparse features)
        return_coarse: boolean, optional (default=False)
            Return coarse features or not

        Returns
        -------
        out: logits for each label
        """
        encoding = super().encode(x, x_ind)
        return encoding if return_coarse else self.transform_fine(encoding)

    def forward(self, batch_data):
        """Forward pass

        Arguments:
        -----------
        batch_data: dict
            * 'X': torch.FloatTensor
                feature weights for given indices or dense rep.
            * 'X_ind': torch.LongTensor
                feature indices (LongTensor) or None
            * 'Y_s': torch.LongTensor
                indices of labels to pick for each document

        Returns
        -------
        out: logits for each label in the shortlist
        """
        return self.classifier(
            self.encode(batch_data['X'], batch_data['X_ind']),
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
                input_size=self.representation_dims,
                output_size=_num_labels,
                bias=_bias,
                padding_idx=_padding_idx,
                num_partitions=self.num_clf_partitions,
                devices=_clf_devices)
        else:
            # last one is padding index
            return linear_layer.SparseLinear(
                input_size=self.representation_dims,
                output_size=self.num_labels + offset,
                padding_idx=self.label_padding_index,
                bias=True)

    def to(self):
        """Send layers to respective devices
        """
        self.transform_fine.to()
        super().to()

    def initialize_classifier(self, weight, bias=None):
        """Initialize classifier from existing weights

        Arguments:
        -----------
        weight: numpy.ndarray
        bias: numpy.ndarray or None, optional (default=None)
        """
        self.classifier.weight.data.copy_(torch.from_numpy(weight))
        if bias is not None:
            self.classifier.bias.data.copy_(
                torch.from_numpy(bias).view(-1, 1))

    def get_clf_weights(self):
        """Get classifier weights
        """
        return self.classifier.get_weights()

    def __repr__(self):
        s = f"{self.embeddings}\n{self.transform}\n"
        s += f"(Transform fine): {self.transform_fine}"
        s += f"\n(Classifier): {self.classifier}\n"
        return s

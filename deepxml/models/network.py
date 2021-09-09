import torch
import torch.nn as nn
import math
import os
import models.transform_layer as transform_layer
import models.linear_layer as linear_layer


__author__ = 'KD'


def _to_device(x, device):
    if x is None:
        return None
    elif isinstance(x, (tuple, list)):
        out = []
        for item in x:
            out.append(_to_device(item, device))
        return out
    else:
        return x.to(device)


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

    def __init__(self, config, device="cuda:0"):
        super(DeepXMLBase, self).__init__()
        self.transform = self._construct_transform(config)
        self.classifier = self._construct_classifier()
        self.device = torch.device(device)

    def _construct_classifier(self):
        return nn.Identity()

    def _construct_transform(self, trans_config):
        return transform_layer.Transform(
            transform_layer.get_functions(trans_config))

    @property
    def representation_dims(self):
        return self._repr_dims

    @representation_dims.setter
    def representation_dims(self, dims):
        self._repr_dims = dims

    def encode(self, x):
        """Forward pass
        * Assumes features are dense if x_ind is None

        Arguments:
        -----------
        x: tuple
            torch.FloatTensor or None
                (sparse features) contains weights of features as per x_ind or
                (dense features) contains the dense representation of a point
            torch.LongTensor or None
                contains indices of features (sparse or seqential features)

        Returns
        -------
        out: logits for each label
        """
        return self.transform(
            _to_device(x, self.device))

    def forward(self, batch_data, *args):
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

    def initialize(self, x):
        """Initialize embeddings from existing ones
        Parameters:
        -----------
        word_embeddings: numpy array
            existing embeddings
        """
        self.transform.initialize(x)

    def to(self):
        """Send layers to respective devices
        """
        self.transform.to()
        self.classifier.to()

    def purge(self, fname):
        if os.path.isfile(fname):
            os.remove(fname)

    @property
    def num_params(self, ignore_fixed=False):
        if ignore_fixed:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    @property
    def model_size(self):  # Assumptions: 32bit floats
        return self.num_params * 4 / math.pow(2, 20)

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
            params.arch, params)
        trans_config_coarse = transform_config_dict['transform_coarse']
        self.representation_dims = int(
            transform_config_dict['representation_dims'])
        self._bias = params.bias
        super(DeepXMLf, self).__init__(trans_config_coarse)
        if params.freeze_intermediate:
            print("Freezing intermediate model parameters!")
            for params in self.transform.parameters():
                params.requires_grad = False
        trans_config_fine = transform_config_dict['transform_fine']
        self.transform_fine = self._construct_transform(
            trans_config_fine)

    def encode_fine(self, x):
        """Forward pass (assumes input is coarse computation)

        Arguments:
        -----------
        x: torch.FloatTensor
            (sparse features) contains weights of features as per x_ind or
            (dense features) contains the dense representation of a point

        Returns
        -------
        out: torch.FloatTensor
            encoded x with fine encoder
        """
        return self.transform_fine(_to_device(x, self.device))

    def encode(self, x, x_ind=None, bypass_fine=False):
        """Forward pass
        * Assumes features are dense if x_ind is None

        Arguments:
        -----------
        x: torch.FloatTensor
            (sparse features) contains weights of features as per x_ind or
            (dense features) contains the dense representation of a point
        x_ind: torch.LongTensor or None, optional (default=None)
            contains indices of features (sparse features)
        bypass_fine: boolean, optional (default=False)
            Return coarse features or not

        Returns
        -------
        out: logits for each label
        """
        encoding = super().encode((x, x_ind))
        return encoding if bypass_fine else self.transform_fine(encoding)

    def forward(self, batch_data, bypass_coarse=False):
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
        if bypass_coarse:
            return self.classifier(
                self.encode_fine(batch_data['X']))
        else:
            return self.classifier(
                self.encode(batch_data['X'], batch_data['X_ind']))

    def _construct_classifier(self):
        if self.num_clf_partitions > 1:  # Run the distributed version
            _bias = [self._bias for _ in range(self.num_clf_partitions)]
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
                bias=self._bias
            )

    def get_token_embeddings(self):
        return self.transform.get_token_embeddings()

    def save_intermediate_model(self, fname):
        torch.save(self.transform.state_dict(), fname)

    def load_intermediate_model(self, fname):
        self.transform.load_state_dict(torch.load(fname))

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
        s = f"{self.transform}\n"
        s += f"(Transform fine): {self.transform_fine}"
        s += f"\n(Classifier): {self.classifier}\n"
        return s


class DeepXMLs(DeepXMLBase):
    """DeepXMLt: DeepXML architecture to be trained with
                 a label shortlist
    * Allows additional transform layer for features
    """

    def __init__(self, params):
        self.num_labels = params.num_labels
        self.num_clf_partitions = params.num_clf_partitions
        self.label_padding_index = params.label_padding_index
        transform_config_dict = transform_layer.fetch_json(
            params.arch, params)
        trans_config_coarse = transform_config_dict['transform_coarse']
        self.representation_dims = int(
            transform_config_dict['representation_dims'])
        self._bias = params.bias
        super(DeepXMLs, self).__init__(trans_config_coarse)
        if params.freeze_intermediate:
            print("Freezing intermediate model parameters!")
            for params in self.transform.parameters():
                params.requires_grad = False
        trans_config_fine = transform_config_dict['transform_fine']
        self.transform_fine = self._construct_transform(
            trans_config_fine)

    def save_intermediate_model(self, fname):
        torch.save(self.transform.state_dict(), fname)

    def load_intermediate_model(self, fname):
        self.transform.load_state_dict(torch.load(fname))

    def encode_fine(self, x):
        """Forward pass (assumes input is coarse computation)

        Arguments:
        -----------
        x: torch.FloatTensor
            (sparse features) contains weights of features as per x_ind or
            (dense features) contains the dense representation of a point

        Returns
        -------
        out: torch.FloatTensor
            encoded x with fine encoder
        """
        return self.transform_fine(_to_device(x, self.device))

    def encode(self, x, x_ind=None, bypass_fine=False):
        """Forward pass
        * Assumes features are dense if x_ind is None

        Arguments:
        -----------
        x: torch.FloatTensor
            (sparse features) contains weights of features as per x_ind or
            (dense features) contains the dense representation of a point
        x_ind: torch.LongTensor or None, optional (default=None)
            contains indices of features (sparse features)
        bypass_fine: boolean, optional (default=False)
            Return coarse features or not

        Returns
        -------
        out: logits for each label
        """
        encoding = super().encode((x, x_ind))
        return encoding if bypass_fine else self.transform_fine(encoding)

    def forward(self, batch_data, bypass_coarse=False):
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
        if bypass_coarse:
            return self.classifier(
                self.encode_fine(batch_data['X']), batch_data['Y_s'])
        else:
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
            _bias = [self._bias for _ in range(self.num_clf_partitions)]
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
                bias=self._bias)

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
        s = f"{self.transform}\n"
        s += f"(Transform fine): {self.transform_fine}"
        s += f"\n(Classifier): {self.classifier}\n"
        return s

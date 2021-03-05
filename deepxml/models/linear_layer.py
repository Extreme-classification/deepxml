import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

__author__ = 'KD'


class Linear(nn.Module):
    """Linear layer
    Parameters:
    -----------
    input_size: int
        input size of transformation
    output_size: int
        output size of transformation
    bias: boolean, default=True
        whether to use bias or not
    device: str, default="cuda:0"
        keep on this device
    """

    def __init__(self, input_size, output_size,
                 bias=True, device="cuda:0"):
        super(Linear, self).__init__()
        self.device = device  # Useful in case of multiple GPUs
        self.input_size = input_size
        self.output_size = output_size
        self.weight = Parameter(
            torch.Tensor(self.output_size, self.input_size))
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        if self.bias is not None:
            return F.linear(
                input.to(self.device), self.weight, self.bias.view(-1))
        else:
            return F.linear(
                input.to(self.device), self.weight)

    def to(self):
        """Transfer to device
        """
        super().to(self.device)

    def reset_parameters(self):
        """Initialize vectors
        """
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv, stdv)

    def get_weights(self):
        """Get weights as numpy array
        Bias is appended in the end
        """
        _wts = self.weight.detach().cpu().numpy()
        if self.bias is not None:
            _bias = self.bias.detach().cpu().numpy()
            _wts = np.hstack([_wts, _bias])
        return _wts

    def __repr__(self):
        s = '{name}({input_size}, {output_size}, {device}'
        if self.bias is not None:
            s += ', bias=True'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    @property
    def sparse(self):
        return False


class SparseLinear(Linear):
    """Sparse Linear linear with sparse gradients
    Parameters:
    -----------
    input_size: int
        input size of transformation
    output_size: int
        output size of transformation
    padding_idx: int
        index for dummy label; embedding is not updated
    bias: boolean, default=True
        whether to use bias or not
    device: str, default="cuda:0"
        keep on this device
    """

    def __init__(self, input_size, output_size, padding_idx=None,
                 bias=True, device="cuda:0"):
        self.padding_idx = padding_idx
        super(SparseLinear, self).__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            device=device)

    def forward(self, embed, shortlist):
        """Forward pass for Linear sparse layer
        Parameters:
        ----------
        embed: torch.Tensor
            input to the layer
        shortlist: torch.LongTensor
            evaluate these labels only

        Returns
        -------
        out: torch.Tensor
            logits for each label in provided shortlist
        """
        embed = embed.to(self.device)
        shortlist = shortlist.to(self.device)
        short_weights = F.embedding(shortlist,
                                    self.weight,
                                    sparse=self.sparse,
                                    padding_idx=self.padding_idx)
        out = torch.matmul(embed.unsqueeze(1), short_weights.permute(0, 2, 1))
        if self.bias is not None:
            short_bias = F.embedding(shortlist,
                                     self.bias,
                                     sparse=self.sparse,
                                     padding_idx=self.padding_idx)
            out = out + short_bias.permute(0, 2, 1)
        return out.squeeze()

    def reset_parameters(self):
        """Initialize weights vectors
        """
        super().reset_parameters()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def __repr__(self):
        s = '{name}({input_size}, {output_size}, {device}'
        if self.bias is not None:
            s += ', bias=True'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        s += ', sparse=True)'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def get_weights(self):
        """Get weights as numpy array
        Bias is appended in the end
        """
        _wts = self.weight.detach().cpu().numpy()
        if self.padding_idx is not None:
            _wts = _wts[:-1, :]
        if (self.bias is not None):
            _bias = self.bias.detach().cpu().numpy()
            if self.padding_idx is not None:
                _bias = _bias[:-1, :]
            _wts = np.hstack([_wts, _bias])
        return _wts

    @property
    def sparse(self):
        return True


class ParallelLinear(nn.Module):
    """Distributed Linear layer with support for multiple devices
    Parameters:
    -----------
    input_size: int
        input size of transformation
    output_size: int
        output size of transformation
    bias: boolean, default=True
        whether to use bias or not
    num_partitions: int, default=2
        partition classifier in these many partitions
    device: list or None, default=None
        devices for each partition; keep "cuda:0" for everyone if None
    """

    def __init__(self, input_size, output_size, bias=True,
                 num_partitions=2, devices=None):
        super(ParallelLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.devices = devices
        self.bias = bias
        if devices is None:  # Keep everything on cuda:0
            self.devices = ["cuda:{}".format(idx) for idx in num_partitions]
        self.num_partitions = num_partitions
        self.classifier = self._construct()

    def _construct(self):
        self._output_sizes = [item.size for item in np.array_split(
            np.arange(self.output_size), self.num_partitions)]
        clf = nn.ModuleList()

        # Input size is same for everyone
        for out in zip(self._output_sizes, self.bias, self.devices):
            clf.append(Linear(self.input_size, *out))
        return clf

    def forward(self, embed):
        """Forward pass
        Arguments:
        -----------
        embed: torch.Tensor
            input to the layer

        Returns:
        --------
        out: list
            logits for each partition
        """
        out = []  # Sequential for now
        for idx in range(self.num_partitions):
            out.append(self.classifier[idx](embed))
        return out

    def to(self):
        """ Transfer to device
        """
        for item in self.classifier:
            item.to()

    def get_weights(self):
        """Get weights as numpy array
        Bias is appended in the end
        """
        out = [item.get_weights() for item in self.classifier]
        return np.vstack(out)


class ParallelSparseLinear(ParallelLinear):
    """Distributed Linear layer with support for multiple devices
    Parameters:
    -----------
    input_size: int
        input size of transformation
    output_size: int
        output size of transformation
    padding_idx: int or None, default=None
        padding index in classifier
    bias: boolean, default=True
        whether to use bias or not
    num_partitions: int, default=2
        partition classifier in these many partitions
    device: list or None, default=None
        devices for each partition; keep "cuda:0" for everyone if None
    """

    def __init__(self, input_size, output_size, padding_idx=None,
                 bias=True, num_partitions=2, devices=None):
        self.padding_idx = padding_idx
        super(ParallelSparseLinear, self).__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            num_partitions=num_partitions,
            devices=devices)

    def _construct(self):
        self._output_sizes = [item.size for item in np.array_split(
            np.arange(self.output_size), self.num_partitions)]
        clf = nn.ModuleList()
        for out in zip(self._output_sizes, self.padding_idx, self.bias, self.devices):
            clf.append(SparseLinear(self.input_size, *out))
        return clf

    def forward(self, embed, shortlist):
        """Forward pass
        Arguments:
        -----------
        embed: torch.Tensor
            input to the layer
        shortlist: [torch.LongTensor]
            Shortlist for each partition

        Returns:
        --------
        out: list
            logits for each partition
        """
        out = []
        for idx in range(self.num_partitions):
            out.append(self.classifier[idx](embed, shortlist[idx]))
        return out

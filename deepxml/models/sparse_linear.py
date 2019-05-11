import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

__author__ = 'KD'


class SparseLinear(nn.Module):
    """
        Sparse Linear linear with support for sparse gradients
    """
    def __init__(self, input_size, output_size, padding_idx=None,
                 sparse=False, low_rank=-1, bias=True, device="cuda:0"):
        """
            Args:
                num_embeddings: int: vocalubary size
                embedding_dim: int: dimension for embeddings
                padding_idx: int: index for dummy label; embedding is not 
                                  updated
                sparse: boolean: sparse or dense gradients
        """
        super(SparseLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bias = None
        # TODO Pytorch gives weird error in case of sparse
        self.weight = Parameter(torch.Tensor(self.output_size,
                                             self.input_size))
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size, 1))
        self.padding_idx = padding_idx
        self.sparse = sparse
        self.device = torch.device(device)
        self.reset_parameters()

    def forward(self, embed, shortlist=None):
        """
            Forward pass for Linear sparse layer
            Args:
                embed
                shortlist
            Returns:
                out: logits for each label
        """
        embed = embed.to(self.device)
        if shortlist is not None:
            shortlist = shortlist.to(self.device)
            short_weights = F.embedding(shortlist,
                               self.weight,
                               sparse=self.sparse)
            out = torch.matmul(embed.unsqueeze(1), short_weights.permute(0, 2, 1))
            if self.bias is not None:
                short_bias = F.embedding(shortlist,
                                self.bias,
                                sparse=self.sparse)
                out = out + short_bias.permute(0, 2, 1)
        else:
            out = F.linear(embed, self.weight, self.bias.squeeze() if self.bias is not None else None)
        return out.squeeze()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


    def __repr__(self):
        return 'SparseLinear(in_features={}, out_features={}, bias={}, sparse={}, padding_index={}, device={})'.format(
            self.input_size, self.output_size, True, self.sparse, self.padding_idx, self.device
        )

    def to_device(self):
        self.to(self.device)

    def get_weights(self):
        _wts = self.weight.detach().cpu().numpy()
        _bias = self.bias.detach().cpu().numpy()
        if self.padding_idx is not None:
            _wts = _wts[:-1, :]
            _bias = _bias[:-1, :]
        return np.hstack([_wts, _bias])


class ParallelSparseLinear(nn.Module):
    """
        Distributed version of Sparse Linear linear with support for sparse gradients
    """
    def __init__(self, input_size, output_size, padding_idx=None,
                 sparse=False, low_rank=-1, bias=True, num_partitions=2, devices=None):
        """
            Args:
                num_embeddings: int: vocalubary size
                embedding_dim: int: dimension for embeddings
                padding_idx: int: index for dummy label; embedding is not 
                                  updated
                sparse: boolean: sparse or dense gradients
        """
        super(ParallelSparseLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.sparse = sparse
        self.devices = devices
        if devices is None:
            self.devices = ["cuda:{}".format(idx) for idx in num_partitions]
        self.num_partitions = num_partitions
        self._num_labels_per_split = [item.size for item in np.array_split(np.arange(self.output_size), self.num_partitions)]
        self.classifier = nn.ModuleList()
        for _, (_output_size, _padding_index, _sparse, _low_rank, _bias, _dev) in enumerate(zip(self._num_labels_per_split, padding_idx, sparse, low_rank, bias, self.devices)):
            self.classifier.append(SparseLinear(
                self.input_size, _output_size, _padding_index, _sparse, _low_rank, _bias, _dev))

    def forward(self, embed, shortlist=None):
        """
            Forward pass for Linear sparse layer
            Args:
                embed: torch.Tensor
                shortlist: [torch.LongTensor]: Shortlist for each partition
            Returns:
                out: []: logits for each label
        """
        out = []
        for idx in range(self.num_partitions):
            out.append(self.classifier[idx](embed, shortlist[idx]))
        return out

    def to_device(self):
        for item in self.classifier:
            item.to_device()

    def get_weights(self):
        out = [item.get_weights() for item in self.classifier]
        return np.vstack(out)
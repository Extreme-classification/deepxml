import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

__author__ = 'KD'


class SparseLinear(nn.Module):
    """
        Sparse Linear linear with support for sparse gradients
    """
    def __init__(self, input_size, output_size, padding_idx=None,
                 sparse=False, low_rank=-1, bias=True):
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
        if shortlist is not None:
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
        return 'SparseLinear(in_features={}, out_features={}, bias={}, sparse={}, padding_index={})'.format(
            self.input_size, self.output_size, True, self.sparse, self.padding_idx
        )

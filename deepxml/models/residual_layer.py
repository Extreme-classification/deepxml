import torch
import torch.nn as nn
import torch.nn.functional as F


__author__ = 'KD'


class Residual(nn.Module):
    """
        Implementation of a Residual block
    """
    def __init__(self, input_size, output_size, dropout, use_shortlist=False,init='eye'):
        """
            Args:
                input_size: int: input dimension
                output_size: int: output dimension
                dropout: float: dropout probability
                init: str: initialization strategy
        """
        super(Residual, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.init = init
        self.dropout = dropout
        self.padding_size = self.output_size - self.input_size
        self.hidden_layer = nn.Sequential(nn.Linear(self.input_size,
                                          self.output_size),
                                          nn.BatchNorm1d(self.output_size),
                                          nn.ReLU(),
                                          nn.Dropout(self.dropout))
        self.use_shortlist = use_shortlist
        self.initialize(self.init)
    def forward(self, embed):
        """
            Forward pass for Residual
            Args:
                embed: torch.Tensor: dense document embeddings
            Returns:
                out: torch.Tensor: dense document embeddings transformed via residual block
        """
        embed = self.hidden_layer(embed) + F.pad(embed, (0, self.padding_size), 'constant', 0)
        return embed

    def initialize(self, init_type):
        """
            Initialize units
            Args:
                init_type: str: random or eye
        """
        if init_type == 'random':
            nn.init.xavier_uniform_(self.hidden_layer[0].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(self.hidden_layer[0].bias, 0.0)
        else:
            print("Using eye to initialize!")
            nn.init.eye_(self.hidden_layer[0].weight)
            nn.init.constant_(self.hidden_layer[0].bias, 0.0)
            if self.use_shortlist:
                self.hidden_layer[1].weight.data.fill_(1.0)
                self.hidden_layer[1].bias.data.fill_(0.0)

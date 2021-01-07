import torch
import torch.nn as nn
import torch.nn.functional as F


__author__ = 'KD'


class Residual(nn.Module):
    """Implementation of a Residual block
    Parameters:
    ----------
    input_size: int
        input dimension
    output_size: int
        output dimension
    dropout: float
        dropout probability
    init: str, default='eye'
        initialization strategy
    """

    def __init__(self, input_size, output_size, dropout, init='eye'):
        super(Residual, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.init = init
        self.dropout = dropout
        self.padding_size = self.output_size - self.input_size
        self.hidden_layer = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.input_size, self.output_size)),
            nn.ReLU(),
            nn.Dropout(self.dropout))
        self.initialize(self.init)

    def forward(self, embed):
        """Forward pass for Residual
        Parameters:
        ----------
        embed: torch.Tensor
            dense document embedding

        Returns
        -------
        out: torch.Tensor
            dense document embeddings transformed via residual block
        """
        temp = F.pad(embed, (0, self.padding_size), 'constant', 0)
        embed = self.hidden_layer(embed) + temp
        return embed

    def initialize(self, init_type):
        """Initialize units
        Parameters:
        -----------
        init_type: str
            Initialize hidden layer with 'random' or 'eye'
        """
        if init_type == 'random':
            nn.init.xavier_uniform_(
                self.hidden_layer[0].weight,
                gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(self.hidden_layer[0].bias, 0.0)
        else:
            nn.init.eye_(self.hidden_layer[0].weight)
            nn.init.constant_(self.hidden_layer[0].bias, 0.0)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


__author__ = 'KD'


class RNN(nn.Module):
    """
        RNN layer
    """

    def __init__(self,
                 input_size,
                 cell_type='RNN',
                 batch_first=True,
                 num_layers=1,
                 dropout=0.0,
                 hidden_size=256,
                 bidirectional=False):
        super(RNN, self).__init__()
        self.cell_type = cell_type
        self.input_size = input_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.rnn = None
        if self.cell_type == 'RNN':
            self.rnn = torch.nn.RNN(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=self.bidirectional)
        elif self.cell_type == 'LSTM':
            self.rnn = torch.nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                batch_first=True,
                bidirectional=self.bidirectional)
        elif self.cell_type == 'GRU':
            self.rnn = torch.nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                batch_first=True,
                bidirectional=self.bidirectional)
        else:
            raise NotImplementedError("Unknown RNN cell type!")
        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

    def forward(self, features, lengths):
        """
            Forward pass for the RNN
            Args:
                features: torch.FloatTensor: embeddings for words
            Returns:
                out: document embedding
        """
        features = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
        output, h_n = self.rnn(features)
        output = pad_packed_sequence(output, batch_first=True)
        #FIXME: return h_n with appropriate sizes and return
        # h_n = h_n.permute(1, 0, 2).contiguous().view(-1, self.num_layers*self.num_directions*self.hidden_size)
        return output, None

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class Attention(nn.Module):
    """ Applies attention mechanism
    """

    def __init__(self, in_dimensions, hidden_dimensions):
        super(Attention, self).__init__()
        self.project = nn.Linear(in_dimensions, hidden_dimensions, bias=False)
        self.context = Parameter(torch.Tensor(hidden_dimensions, 1))
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.initialize()

    def forward(self, query):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        trans_query = self.tanh(self.project(query))
        attention_scores = torch.matmul(trans_query, self.context)
        attention_scores = self.softmax(attention_scores)
        return torch.sum(trans_query * attention_scores, dim=1) 

    def initialize(self):
        """
            Initialize modules
        """
        nn.init.xavier_uniform_(self.context, gain=nn.init.calculate_gain('tanh'))

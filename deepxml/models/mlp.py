import torch
import torch.nn as nn


__author__ = 'KD'


class MLP(nn.Module):
    """
    A multi-layer perceptron with flexibility for non-liearity
    * no non-linearity after last layer
    * support for 2D or 3D inputs

    Parameters:
    -----------
    input_size: int
        input size of embeddings
    hidden_size: int or list of ints or str (comma separated)
        e.g., 512: a single hidden layer with 512 neurons
              "512": a single hidden layer with 512 neurons
              "512,300": 512 -> nnl -> 300
              [512, 300]: 512 -> nnl -> 300
        dimensionality of layers in MLP
    nnl: str, optional, default='relu'
        which non-linearity to use
    device: str, default="cuda:0"
        keep on this device
    """
    def __init__(self, input_size, hidden_size, nnl='relu', device="cuda:0"):
        super(MLP, self).__init__()
        hidden_size = self.parse_hidden_size(hidden_size)
        assert len(hidden_size) >= 1, "Should contain atleast 1 hidden layer"
        hidden_size = [input_size] + hidden_size
        self.device = torch.device(device)
        layers = []
        for i, (i_s, o_s) in enumerate(zip(hidden_size[:-1], hidden_size[1:])):
            layers.append(nn.Linear(i_s, o_s, bias=True))
            if i < len(hidden_size) - 2:
                layers.append(self._get_nnl(nnl))
        self.transform = torch.nn.Sequential(*layers)

    def parse_hidden_size(self, hidden_size):
        if isinstance(hidden_size, int):
            return [hidden_size]
        elif isinstance(hidden_size, str):
            _hidden_size = []
            for item in hidden_size.split(","):
                _hidden_size.append(int(item))
            return _hidden_size
        elif isinstance(hidden_size, list):
            return hidden_size
        else:
            raise NotImplementedError("hidden_size must be a int, str or list")

    def _get_nnl(self, nnl):
        if nnl == 'sigmoid':
            return torch.nn.Sigmoid()
        elif nnl == 'relu':
            return torch.nn.ReLU()
        elif nnl == 'gelu':
            return torch.nn.GELU()
        elif nnl == 'tanh':
            return torch.nn.Tanh()
        else:
            raise NotImplementedError(f"{nnl} not implemented!")

    def forward(self, x):
        return self.transform(x)

    def to(self):
        """Transfer to device
        """
        super().to(self.device)

    @property
    def sparse(self):
        return False

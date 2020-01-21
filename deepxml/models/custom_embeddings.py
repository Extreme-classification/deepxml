import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class CustomEmbedding(torch.nn.Module):
    """
        Memory efficient way to compute weighted EmbeddingBag
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, device="cuda:0"):
        """
            Args:
                num_embeddings: int: vocalubary size
                embedding_dim: int: dimension for embeddings
                padding_idx: int: index for <PAD>; embedding is not updated
                max_norm: 
                norm_type: int: default: 2
                scale_grad_by_freq: boolean: True/False
                sparse: boolean: sparse or dense gradients
        """
        super(CustomEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.sparse = sparse
        self.device = torch.device(device)
        self.reset_parameters()

    def reset_parameters(self):
        """
            Reset weights
        """
        self.weight.data.normal_(0, 1)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def to(self):
        super().to(self.device)

    def forward(self, features, weights, _div=False):
        """
        Forward pass for embedding layer
        Args:
            features: torch.LongTensor: (batch_size, max_features_in_a_batch)
            weights: torch.Tensor: (batch_size, max_features_in_a_batch)
            _div: boolean: weighted sum or weighted average.
        Returns:
            out: torch.Tensor: embedding for each sample (batch_size, embedding_dims)
        """
        out = []
        batch_size = features.size()[0]
        local_batch_size = 32
        for batch_idx in range(0, batch_size, local_batch_size):
            begin_idx = batch_idx
            end_idx = min(batch_idx+local_batch_size, batch_size)
            _input = features[begin_idx:end_idx, :]
            _weight = weights[begin_idx:end_idx, :].unsqueeze(2)
            temp = F.embedding(
                _input, self.weight,
                self.padding_idx, self.max_norm, self.norm_type,
                self.scale_grad_by_freq, self.sparse)
            temp = temp * _weight
            temp = torch.sum(temp, dim=1)
            if _div:
                temp = temp/torch.sum(_weight, dim=1)
            out.append(temp)

        out = torch.cat(out, dim=0)
        return out

    def get_weights(self):
        return self.weight.detach().cpu().numpy()[1:, :]

    def __repr__(self):
        s = '{name}({num_embeddings}, {embedding_dim}, {device}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

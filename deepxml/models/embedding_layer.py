import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class Embedding(torch.nn.Module):
    """
    General way to handle embeddings

    * Support for sequential models
    * Memory efficient way to compute weighted EmbeddingBag

    Arguments:
    ----------
    num_embeddings: int
        vocalubary size
    embedding_dim: int
        dimension for embeddings
    padding_idx: 0 or None, optional (default=None)
        index for <PAD>; embedding is not updated
    max_norm: None or float, optional (default=None)
        maintain norm of embeddings
    norm_type: int, optional (default=2)
        norm for max_norm
    scale_grad_by_freq: boolean, optional (default=False)
        Scale gradients by token frequency
    sparse: boolean, optional (default=False)
        sparse or dense gradients
        * the optimizer will infer from this parameters
    reduction: str or None, optional (default=None)
        * None: don't reduce
        * sum: sum over tokens
        * mean: mean over tokens
    pretrained_weights: torch.Tensor or None, optional (default=None)
        Initialize with these weights
        * first token is treated as a padding index
        * dim=1 should be one less than the num_embeddings
    device: str, optional (default="cuda:0")
        Keep embeddings on this device
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, reduction=True, pretrained_weights=None,
                 device="cuda:0"):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.sparse = sparse
        self.reduce = self._construct_reduce(reduction)
        self.reduction = reduction
        self.device = torch.device(device)
        self.reset_parameters()
        if pretrained_weights is not None:
            self.from_pretrained(pretrained_weights)

    def _construct_reduce(self, reduction):
        if reduction is None:
            return self._reduce
        elif reduction == 'sum':
            return self._reduce_sum
        elif reduction == 'mean':
            return self._reduce_mean
        else:
            return NotImplementedError(f"Unknown reduction: {reduction}")

    def reset_parameters(self):
        """
            Reset weights
        """
        torch.nn.init.xavier_uniform_(
            self.weight.data, gain=torch.nn.init.calculate_gain('relu'))
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def to(self):
        super().to(self.device)

    def _reduce_sum(self, x, w):
        if w is None:
            return torch.sum(x, dim=1)
        else:
            return torch.sum(x * w.unsqueeze(2), dim=1)

    def _reduce_mean(self, x, w):
        if w is None:
            return torch.mean(x, dim=1)
        else:
            return torch.mean(x * w.unsqueeze(2), dim=1)

    def _reduce(self, x, *args):
        return x

    def forward(self, x, w=None):
        """
        Forward pass for embedding layer

        Arguments:
        ---------
        x: torch.LongTensor
            indices of tokens in a batch
            (batch_size, max_features_in_a_batch)
        w: torch.Tensor or None, optional (default=None)
            weights of tokens in a batch
            (batch_size, max_features_in_a_batch)

        Returns:
        --------
        out: torch.Tensor
            embedding for each sample
            Shape: (batch_size, seq_len, embedding_dims), if reduction is None
            Shape: (batch_size, embedding_dims), otherwise
        """
        x = F.embedding(
            x, self.weight,
            self.padding_idx, self.max_norm, self.norm_type,
            self.scale_grad_by_freq, self.sparse)
        return self.reduce(x, w)

    def from_pretrained(self, embeddings):
        # first index is treated as padding index
        assert embeddings.shape[0] == self.num_embeddings-1, \
            "Shapes doesn't match for pre-trained embeddings"
        self.weight.data[1:, :] = torch.from_numpy(embeddings)

    def get_weights(self):
        return self.weight.detach().cpu().numpy()[1:, :]

    def __repr__(self):
        s = '{name}({num_embeddings}, {embedding_dim}, {device}'
        s += ', reduction={reduction}'
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

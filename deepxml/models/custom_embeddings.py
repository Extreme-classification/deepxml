import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class CustomEmbedding(torch.nn.Module):
    """
    Memory efficient way to compute weighted EmbeddingBag

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
    scale_grad_by_freq: boolean
        Scale gradients by token frequency
    sparse: boolean
        sparse or dense gradients
        * the optimizer will infer from this parameters
    device: str, optional (default="cuda:0")
        Keep embeddings on this device
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, device="cuda:0"):
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
            out: torch.Tensor: embedding for each sample
            (batch_size, embedding_dims)
        """
        batch_size = features.size()[0]
        out = F.embedding(
            features, self.weight,
            self.padding_idx, self.max_norm, self.norm_type,
            self.scale_grad_by_freq, self.sparse)
        out = torch.sum(out * weights.unsqueeze(2), dim=1)
        if _div:
            out = out/torch.sum(weight, dim=1)
        return out

    def from_pretrained(self, embeddings):
        # first index is treated as padding index
        if self.padding_index is not None:
            self.weight.data[1:, :] = torch.from_numpy(embeddings)
        else:
            self.weight.data.copy_(torch.from_numpy(embeddings))

    def get_weights(self):
        if self.padding_index is not None:
            return self.weight.detach().cpu().numpy()[1:, :]
        else:
            return self.weight.detach().cpu().numpy()

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

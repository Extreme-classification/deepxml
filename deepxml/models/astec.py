import torch.nn as nn
import models.embedding_layer as embedding_layer


class Astec(nn.Module):
    """
    Encode a document using the feature representaion as per Astec

    Arguments:
    ----------
    num_embeddings: int
        vocalubary size
    embedding_dim: int, optional (default=300)
        dimension for embeddings
    dropout: float, optional (default=0.5)
        drop probability
    padding_idx: int, optional (default=0)
        index for <PAD>; embedding is not updated
        Values other than 0 are not yet tested
    reduction: str or None, optional (default=None)
        * None: don't reduce
        * sum: sum over tokens
        * mean: mean over tokens
    sparse: boolean, optional (default=False)
        sparse or dense gradients
        * the optimizer will infer from this parameters
    freeze_embeddings: boolean, optional (default=False)
        * freeze the gradient of token embeddings
    device: str, optional (default="cuda:0")
        Keep embeddings on this device
    """
    def __init__(self, vocabulary_dims, embedding_dims=300,
                 dropout=0.5, padding_idx=0, reduction='sum',
                 sparse=True, freeze=False, device="cuda:0"):
        super(Astec, self).__init__()
        self.vocabulary_dims = vocabulary_dims + 1
        self.embedding_dims = embedding_dims
        self.padding_idx = padding_idx
        self.device = device
        self.sparse = sparse
        self.reduction = reduction
        self.embeddings = self._construct_embedding()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.freeze = freeze
        if self.freeze:
            for params in self.embeddings.parameters():
                params.requires_grad = False

    def _construct_embedding(self):
        return embedding_layer.Embedding(
            num_embeddings=self.vocabulary_dims,
            embedding_dim=self.embedding_dims,
            padding_idx=self.padding_idx,
            scale_grad_by_freq=False,
            device=self.device,
            reduction=self.reduction,
            sparse=self.sparse)

    def encoder(self, x, x_ind):
        if x_ind is None:  # Assume embedding is pre-computed
            return x
        else:
            return self.embeddings(x_ind, x)

    def forward(self, x):
        """
        Arguments:
        ----------
        x: (torch.Tensor or None, torch.LongTensor)
            token weights and indices
            weights can be None

        Returns:
        --------
        embed: torch.Tensor
            transformed document representation
            Dimension depends on reduction
        """
        return self.dropout(self.relu(self.encoder(*x)))

    def to(self):
        super().to(self.device)

    def initialize(self, x):
        self.embeddings.from_pretrained(x)

    def initialize_token_embeddings(self, x):
        return self.initialize(x)

    def get_token_embeddings(self):
        return self.embeddings.get_weights()

    @property
    def representation_dims(self):
        return self.embedding_dims

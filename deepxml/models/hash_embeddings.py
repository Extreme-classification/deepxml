from __future__ import unicode_literals, division

import numpy as np

import torch
import torch.nn as nn


class HashFamily():
    r"""Universal hash family as proposed by Carter and Wegman.

    .. math::

            \begin{array}{ll}
            h_{{a,b}}(x)=((ax+b)~{\bmod  ~}p)~{\bmod  ~}m \ \mid p > m\\
            \end{array}

    Args:
        bins (int): Number of bins to hash to. Better if a prime number.
        mask_zero (bool, optional): Whether the 0 input is a special "padding" value to mask out.
        moduler (int,optional): Temporary hashing. Has to be a prime number.
    """

    def __init__(self, bins, mask_zero=False, moduler=None):
        if moduler and moduler <= bins:
            raise ValueError("p (moduler) should be >> m (buckets)")

        self.bins = bins
        self.moduler = moduler if moduler else self._next_prime(
            np.random.randint(self.bins + 1, 2**32))
        self.mask_zero = mask_zero

        # do not allow same a and b, as it could mean shifted hashes
        self.sampled_a = set()
        self.sampled_b = set()

    def _is_prime(self, x):
        """Naive is prime test."""
        for i in range(2, int(np.sqrt(x))):
            if x % i == 0:
                return False
        return True

    def _next_prime(self, n):
        """Naively gets the next prime larger than n."""
        while not self._is_prime(n):
            n += 1

        return n

    def draw_hash(self, a=None, b=None):
        """Draws a single hash function from the family."""
        if a is None:
            while a is None or a in self.sampled_a:
                a = np.random.randint(1, self.moduler - 1)
                assert len(self.sampled_a
                           ) < self.moduler - 2, "please give a bigger moduler"

            self.sampled_a.add(a)
        if b is None:
            while b is None or b in self.sampled_b:
                b = np.random.randint(0, self.moduler - 1)
                assert len(self.sampled_b
                           ) < self.moduler - 1, "please give a bigger moduler"

            self.sampled_b.add(b)

        if self.mask_zero:
            # The return doesn't set 0 to 0 because that's taken into account in the hash embedding
            # if want to use for an integer then should uncomment second line !!!!!!!!!!!!!!!!!!!!!
            return lambda x: ((a * x + b) % self.moduler) % (self.bins - 1) + 1
            # return lambda x: 0 if x == 0 else ((a*x + b) % self.moduler) % (self.bins-1) + 1
        else:
            return lambda x: ((a * x + b) % self.moduler) % self.bins

    def draw_hashes(self, n, **kwargs):
        """Draws n hash function from the family."""
        return [self.draw_hash() for i in range(n)]


class HashEmbedding(nn.Module):
    """
        Implementation of hash embeddings as per "Hash Embeddings for Efficient Word Representations"
    """

    def __init__(self,
                 vocabulary_dims,
                 embedding_dims,
                 num_hashes,
                 num_buckets,
                 aggregation_mode='sum',
                 mask_zero=False,
                 append_weight=True,
                 seed=None,
                 sparse=True):
        super(HashEmbedding, self).__init__()

        self.num_embeddings = vocabulary_dims + 1
        self.embedding_dims = embedding_dims
        self.num_hashes = num_hashes
        self.append_weight = append_weight
        self.sparse = sparse
        # self.num_buckets = params.num_buckets - 1
        self.num_buckets = num_buckets
        self.padding_idx = 0 if mask_zero else None
        self.seed = seed
        self.importance_weights = nn.Embedding(
            self.num_embeddings,
            self.num_hashes,
            self.padding_idx,
            sparse=True)
        self.shared_embeddings = nn.Embedding(
            self.num_buckets + 1,
            self.embedding_dims,
            padding_idx=self.padding_idx,
            sparse=True)

        hash_family = HashFamily(self.num_buckets, mask_zero=mask_zero)
        self.hash_functions = hash_family.draw_hashes(self.num_hashes)

        if aggregation_mode == 'sum':
            self.aggregate = lambda x: torch.sum(x, dim=-1)
        elif aggregation_mode == 'mean':
            self.aggregate = lambda x: torch.mean(x, dim=-1)
        elif aggregation_mode == 'concatenate':
            self.aggregate = lambda x: torch.cat(
                [x[:, :, :, i] for i in range(self.num_hashes)], dim=-1)
        else:
            raise ValueError(
                'unknown aggregation function {}'.format(aggregation_mode))

        self.reset_parameters()

    def get_weights(self):
        return self.shared_embeddings.weight.detach().cpu().numpy()[1:, :], self.importance_weights.weight.detach().cpu().numpy()[1:, :]

    def reset_parameters(self):
        """
            Reset model parameters
        """
        nn.init.normal_(self.shared_embeddings.weight)
        nn.init.constant_(self.importance_weights.weight, 1)
        if self.padding_idx is not None:
            self.shared_embeddings.weight.data[self.padding_idx].fill_(0)
            self.importance_weights.weight.data[self.padding_idx].fill_(0)

    def _combine(self, feat, wts):
        return None

    def forward(self, features, weights):
        """
            Forward pass of Hash Embeddings
            Args:
                features: Variable: (batch_size, max_len)
            returns:
                word_embeddings: torch.Tensor: (batch_size, size, max_len)
        """
        idx_importance_weights = features % self.num_embeddings
        # Ensure that hash(0) = 0
        idx_shared_embeddings = torch.stack(
            [h(features).masked_fill_(features == 0, 0)
             for h in self.hash_functions],
            dim=-1)

        shared_embedding = torch.stack(
            [
                self.shared_embeddings(idx_shared_embeddings[:, :, iHash])
                for iHash in range(self.num_hashes)
            ],
            dim=-1)
        importance_weight = torch.sigmoid(
            self.importance_weights(idx_importance_weights))
        importance_weight = importance_weight.unsqueeze(-2)
        word_embedding = self.aggregate(importance_weight * shared_embedding)
        if self.append_weight:
            word_embedding = torch.cat(
                [word_embedding, importance_weight.squeeze(-2)], dim=-1)
        out = torch.sum(word_embedding * weights.unsqueeze(2), dim=1)
        return out

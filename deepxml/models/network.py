import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import os
import numpy as np
import models.custom_embeddings as custom_embeddings
import models.hash_embeddings as hash_embeddings
import models.transform_layer as transform_layer
import models.sparse_linear as sparse_linear

__author__ = 'KD'


class Scale1(nn.Module):
    def __init__(self, ):
        super(Scale1, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([1.0]).type(torch.FloatTensor))
        self.beta = nn.Parameter(torch.Tensor([1.0]).type(torch.FloatTensor))
        self.gamma = nn.Parameter(torch.Tensor([0.0]).type(torch.FloatTensor))

    def forward(self, knn_out, clf_out):
        return self.gamma.sigmoid()*(self.beta-self.alpha*self.alpha*knn_out), clf_out

    def _get_score(self, knn, clf, beta):
        return beta*(clf.sigmoid().log())+(1-beta)*(knn.sigmoid().log())
    
    def _reset(self):
        self.alpha.data.fill_(1.0)
        self.beta.data.fill_(1.0)
        self.gamma.data.fill_(0.0)

    def __repr__(self):
        return "Scale1(alpha={},beta={},gamma={})".format(self.alpha.data,self.beta.data,self.gamma.data)

class Scale2(nn.Module):
    def __init__(self, ):
        super(Scale2, self).__init__()
        pass

    def forward(self, knn_out, clf_out):
        return 1-knn_out, clf_out

    def _get_score(self, knn, clf, beta):
        return beta*(clf.sigmoid().log())+(1-beta)*(knn.sigmoid().log())
    
    def _reset(self):
        pass

    def __repr__(self):
        return "Scale2(alpha=1,beta=1)"

class return_identity(nn.Module):
    def __init__(self):
        super(return_identity, self).__init__()
        pass

    def forward(self, knn, clf):
        return 1-knn, clf

    def _get_score(self, knn, clf, beta):
        return beta*clf.sigmoid()+(1-beta)*(knn.sigmoid())

    def _reset(self):
        pass
    
    def __repr__(self):
        return "return_identity()"

class DeepXML(nn.Module):
    """
        DeepXML: A Scalable Deep learning approach for eXtreme Multi-label Learning
    """

    def __init__(self, params):
        super(DeepXML, self).__init__()
        self.vocabulary_dims = params.vocabulary_dims+1
        self.hidden_dims = params.embedding_dims
        self.embedding_dims = params.embedding_dims
        self.trans_method = params.trans_method
        self.dropout = params.dropout
        self.num_labels = params.num_labels
        self.use_hash_embeddings = params.use_hash_embeddings
        self.num_buckets = params.num_buckets
        self.num_hashes = params.num_hashes
        self.label_padding_index = params.label_padding_index
        self.use_shortlist = params.use_shortlist
        self.append_weight = True
        self.low_rank = params.low_rank if params.use_low_rank else -1
        assert params.use_low_rank == (
            self.low_rank != -1), "Sorry, can't train with negative rank! Go read about positive numbers."
        self.use_residual = params.use_residual
        # Hash embeddings append weights
        # TODO: will not work for aggregation_mode 'concat'
        self.pt_repr_dims, self.repr_dims = self._compute_rep_dims()
        self.logit_type = params.logit_type
        if self.logit_type == 1:
            self.rescale_logits = Scale1()
        elif self.logit_type == 2:
            self.rescale_logits = Scale2()
        elif self.logit_type == -1:
            self.rescale_logits = return_identity()
        else:
            raise NotImplementedError("Unknown logit type.")

        if self.use_hash_embeddings:
            assert self.num_buckets != -1, "#buckets must be positive"
            assert self.num_hashes != -1, "#hashes must be positive"
            self.embeddings = hash_embeddings.HashEmbedding(vocabulary_dims=self.vocabulary_dims,
                                                            embedding_dims=self.embedding_dims,
                                                            num_hashes=self.num_hashes,
                                                            num_buckets=self.num_buckets,
                                                            aggregation_mode='sum',
                                                            mask_zero=False,
                                                            append_weight=self.append_weight,
                                                            seed=None,
                                                            sparse=True)
        else:
            self.embeddings = custom_embeddings.CustomEmbedding(num_embeddings=self.vocabulary_dims,
                                                                embedding_dim=self.embedding_dims,
                                                                padding_idx=params.padding_idx,
                                                                scale_grad_by_freq=False,
                                                                sparse=True)

        self.transform = transform_layer.Transform(hidden_dims=params.hidden_dims,
                                                   embedding_dims=self.pt_repr_dims,
                                                   trans_method=params.trans_method,
                                                   dropout=params.dropout,
                                                   use_residual=params.use_residual,
                                                   res_init=params.res_init,
                                                   use_shortlist=self.use_shortlist
                                                   )
        if self.low_rank != -1:
            self.low_rank_layer = sparse_linear.SparseLinear(
                self.repr_dims, self.low_rank, sparse=False, bias=False)
        offset = 1 if self.label_padding_index is not None else 0
        self.classifier = sparse_linear.SparseLinear(self.repr_dims if self.low_rank == -1 else self.low_rank,
                                                     self.num_labels + offset, #last one is padding index
                                                     sparse=True if self.use_shortlist else False,
                                                     low_rank=self.low_rank,
                                                     padding_idx=self.label_padding_index)
        self.device_embeddings = None
        self.device_classifier = None

    def _compute_rep_dims(self):
        pt_repr_dims = self.embedding_dims
        pt_repr_dims += self.num_hashes if (
            self.use_hash_embeddings and self.append_weight == True) else 0
        rep_dims = pt_repr_dims
        if self.trans_method == 'deep_non_linear' or self.use_residual:
            rep_dims = self.hidden_dims
        return pt_repr_dims, rep_dims

    def forward(self, features, weights, shortlist=None, return_embeddings=False):
        """
            Forward pass
            Args:
                features: torch.LongTensor: feature indices
                weights: torch.Tensor: feature weights
                shortlist: torch.LongTensor: Relevant labels for each sample
                return_embeddings: boolean: Return embeddings or classify
            Returns:
                out: logits for each label
        """
        if weights.size()[0]==1:
            temp = features
        else:
            temp = self.embeddings(features, weights)
        
        embed = self.transform(temp)
        if return_embeddings:
            out = embed
        else:
            if self.low_rank != -1:
                embed = self.low_rank_layer(embed)
            # out = self.classifier(embed, shortlist)
            out = self.classifier(
                embed.to(self.device_classifier), shortlist)
            out = out.squeeze()
        return out

    def initialize_embeddings(self, word_embeddings):
        """
            Initialize embeddings from existing ones
            Args:
                word_embeddings: numpy array: existing embeddings
        """
        self.embeddings.weight.data.copy_(torch.from_numpy(word_embeddings))

    def initialize_classifier(self, clf_weights):
        """
            Initialize classifier from existing weights
            Args:
                clf_weights: numpy.ndarray: (num_labels, repr_dims+1) last dimension is bias
        """
        self.classifier.weight.data.copy_(torch.from_numpy(clf_weights[:, -1]))
        self.classifier.bias.data.copy_(
            torch.from_numpy(clf_weights[:, -1]).view(-1, 1))

    def _get_positive_logits(self, clf_out, knn_out):
        pos_labels_logits = clf_out.clamp(max=0) + knn_out.clamp(max=0) - \
            (
                (
                    -clf_out.clamp(min=0) + knn_out.clamp(max=0)
                ).exp() +
                (
                    -clf_out.abs() - knn_out.clamp(min=0)
                ).exp() +
                (
                    -knn_out.clamp(min=0)
                ).exp()
        ).log()
        return pos_labels_logits

    def _get_negative_logits(self, clf_out, knn_out):
        neg_labels_logits = clf_out.clamp(min=0) + knn_out.clamp(min=0) + \
            (
                (
                    knn_out.clamp(max=0)
                ).exp() +
                (
                    -clf_out.abs() + knn_out.clamp(max=0)
                ).exp() +
                (
                    -knn_out.clamp(min=0) + clf_out.clamp(max=0)
                ).exp()
        ).log()
        return neg_labels_logits

    def _get_logits_train(self, clf_out, knn_out, flag):
        """
        Logits_{EFF} = _0*SIG^{-1}(1-(1-P_{SVM})*(1-P_{KNN}))+_1*SIG^{-1}(P_{SVM}*P_{KNN})
        """
        if self.logit_type != -1:
            _1 = flag
            _0 = (1-flag)
            _knn_out, _clf_out = self.rescale_logits(knn_out, clf_out)

            if (clf_out != clf_out).any() or (knn_out != knn_out).any():
                print("Bhaai ye kyun nhi chal rha")
                print(clf_out[clf_out != clf_out], knn_out[knn_out != knn_out])
                exit(0)

            logits = (_0*self._get_negative_logits(_clf_out, _knn_out) +
                      _1*self._get_positive_logits(_clf_out, _knn_out))

            return logits
        return clf_out

    def _get_clf_wts(self):
        _wts = self.classifier.weight.cpu.numpy()
        _bias = self.classifier.bias.cpu.numpy()
        if self.label_padding_index is not None:
            _wts = _wts[:-1, :]
            _bias = _bias[:-1, :]
        return np.hstack([_wts, _bias])

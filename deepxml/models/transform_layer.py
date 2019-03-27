import torch
import torch.nn as nn
import models.residual_layer as residual_layer

__author__ = 'KD'


class Transform(nn.Module):
    """
        Transform document representation!
    """

    def __init__(self, hidden_dims, embedding_dims, trans_method, dropout, use_residual, res_init, use_shortlist):
        super(Transform, self).__init__()
        self.hidden_dims = hidden_dims
        self.embedding_dims = embedding_dims
        self.trans_method = trans_method
        self.dropout = dropout
        self.use_residual = use_residual
        self.res_init = res_init
        self.use_shortlist = use_shortlist
        modules = []
        if self.trans_method == 'linear':
            modules.append(nn.Dropout(self.dropout))
            self.hidden_dims = self.embedding_dims
        elif self.trans_method == 'non_linear':
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(self.dropout))
            self.hidden_dims = self.embedding_dims
        elif self.trans_method == 'deep_non_linear':
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(self.dropout))
            modules.append(nn.Linear(self.embedding_dims, self.hidden_dims))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(self.dropout))        
        elif self.trans_method == 'deep_non_linear_bn':
            modules.append(nn.BatchNorm1d(self.hidden_dims))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(self.dropout))
            modules.append(nn.Linear(self.embedding_dims, self.hidden_dims))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(self.dropout))               
        else:
            raise NotImplementedError("Unknown tranformation method!")

        if self.use_residual:
            modules.append(residual_layer.Residual(self.hidden_dims,
                                                         self.hidden_dims,
                                                         self.dropout,
                                                   use_shortlist=self.use_shortlist, init=self.res_init))
        self.transform = nn.Sequential(*modules)


    def forward(self, embed):
        """
            Forward pass for transform layer
            Args:
                embed: torch.Tensor: document representation
            Returns:
                embed: torch.Tensor: transformed document representation
        """
        return self.transform(embed)

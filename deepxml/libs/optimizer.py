import torch
import torch.nn as nn


class Optimizer(object):
    """Wrapper for pytorch optimizer class to handle
    mixture of sparse and dense parameters
    - Infers sparse/dense from 'sparse' attribute

    Parameters
    ----------
    opt_type: str, optional, default='Adam'
        optimizer to use
    learning_rate: float, optional, default=0.01
        learning rate for the optimizer
    momentum: float, optional, default=0.9
        momentum (valid for SGD only)
    weight_decay: float, optional, default=0.0
        l2-regularization cofficient
    nesterov: boolean, optional, default=True
        Use nesterov method (useful in SGD only)
    freeze_embeddings: boolean, optional, default=False
        Don't update embedding layer
    """

    def __init__(self, opt_type='Adam', learning_rate=0.01,
                 momentum=0.9, weight_decay=0.0, nesterov=True,
                 freeze_embeddings=False):
        self.opt_type = opt_type
        self.optimizer = []
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.freeze_embeddings = freeze_embeddings

    def _get_opt(self, params, is_sparse):
        if self.opt_type == 'SGD':
            if is_sparse:
                return torch.optim.SGD(
                    params,
                    lr=self.learning_rate,
                    momentum=self.momentum,
                )
            else:
                return torch.optim.SGD(
                    params,
                    lr=self.learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay
                )
        elif self.opt_type == 'Adam':
            if is_sparse:
                return torch.optim.SparseAdam(
                    params,
                    lr=self.learning_rate
                )
            else:
                return torch.optim.Adam(
                    params,
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay
                )
        else:
            raise NotImplementedError("Unknown optimizer!")

    def construct(self, model):
        """
        Get optimizer.
        Args:
            model: torch.nn.Module: network
            params: : parameters
        Returns:
            optimizer: torch.optim: optimizer as per given specifications
        """
        model_params, is_sparse = self.get_params(model)
        for _, item in enumerate(zip(model_params, is_sparse)):
            if item[0]:
                self.optimizer.append(self._get_opt(
                    params=item[0], is_sparse=item[1]))
            else:
                self.optimizer.append(None)

    def adjust_lr(self, dlr_factor):
        """
            Adjust learning rate
            Args:
                dlr_factor: float: dynamic learning rate factor
        """
        for opt in self.optimizer:
            if opt:
                for param_group in opt.param_groups:
                    param_group['lr'] *= dlr_factor

    def step(self):
        for opt in self.optimizer:
            if opt:
                opt.step()

    def load_state_dict(self, sd):
        for idx, item in enumerate(sd):
            if item:
                self.optimizer[idx].load_state_dict(item)

    def state_dict(self):
        out_states = []
        for item in self.optimizer:
            if item:
                out_states.append(item.state_dict())
            else:
                out_states.append(None)
        return out_states

    def get_params(self, net):
        self.net_params = {}
        self.net_params['sparse'] = list()
        self.net_params['dense'] = list()
        self.net_params['no_grad'] = list()
        if self.freeze_embeddings:
            for params in net.embeddings.parameters():
                params.requires_grad = False

        for key, val in net.__dict__['_modules'].items():
            is_sparse = val.__dict__.get("sparse", False)
            for params in val.parameters():
                if params.requires_grad:
                    if is_sparse:
                        self.net_params['sparse'].append(
                            {"params": params})
                    else:
                        self.net_params['dense'].append(
                            {"params": params})
                else:
                    self.net_params['no_grad'].append(params)
        return [self.net_params['sparse'], self.net_params['dense']], \
            [True, False]

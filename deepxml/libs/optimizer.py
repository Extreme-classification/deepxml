import torch
import torch.nn as nn
import models.transform_layer as transform_layer


class Optimizer(object):
    """Wrapper for pytorch optimizer class to handle
       mixture of sparse and dense parameters
    * Infers sparse/dense from 'sparse' attribute
    * Best results with Adam optimizer
    * Uses _modules() method by default; User may choose to define 
      modules_() to change the behaviour

    Arguments
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
                 momentum=0.9, weight_decay=0.0, nesterov=True):
        self.opt_type = opt_type
        self.optimizer = []
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov

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

        Arguments:
        ---------
        dlr_factor: float
            lr = lr * dlr_factor 
        """
        for opt in self.optimizer: # for each group
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

    def _modules(self, net):
        """
        Get modules (nn.Module) in the network

        * _modules() method by default
        * user may choose to define modules_() to change the behaviour
        """
        # Useful when parameters are tied etc.
        if hasattr(net, 'modules_'):
            return net.modules_.items()
        else:
            return net._modules.items()

    def _sparse(self, item):
        """
        * infer (sparse attribute) if parameter group is sparse or dense
        * assume dense of the sparse attribute is unavailable
        """
        try:
            return item.sparse
        except AttributeError:
            return False

    def _parameters(self, item):
        """
        Return the parameters and is_sparse for each group 
        * Uses parameters() method for a nn.Module object
        * traverse down the tree until a nn.Module object is found
          Unknown behaviour for infinite loop
        """
        if isinstance(item, transform_layer.Transform):
            return self._parameters(item.transform)
        elif isinstance(item, nn.Sequential):
            params = []
            is_sparse = []
            for _item in item:
                _p, _s = self._parameters(_item)
                params.append(_p)
                is_sparse.append(_s)
            return params, is_sparse
        elif isinstance(item, nn.Module):
            return item.parameters(), self._sparse(item)
        else:
            raise NotImplementedError("Unknown module class!")

    def _get_params(self, _params, _sparse, params):
        if isinstance(_params, list):
            for p, s in zip(_params, _sparse):
                self._get_params(p, s, params)
        else:  # Should be generator
            for p in _params:
                if p.requires_grad:
                    if _sparse:
                        params['sparse'].append({"params": p})
                    else:
                        params['dense'].append({"params": p})

    def get_params(self, net):
        self.net_params = {'sparse': [], 'dense': []}
        for _, val in self._modules(net):
            p, s = self._parameters(val)
            self._get_params(p, s, self.net_params)
        return [self.net_params['sparse'], self.net_params['dense']], \
            [True, False]

import torch
import torch.nn as nn

class Optimizer(object):
    def __init__(self, opt_type='Adam', learning_rate=0.01, momentum=0.9, weight_decay=0.0, nesterov=True, freeze_embeddings=False):
        self.opt_type = opt_type 
        self.optimizer = []
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.freeze_embeddings = freeze_embeddings

    def _get_opt(self, params, is_sparse):
        if self.opt_type == 'SGD':
            return torch.optim.SGD(params,
                                    lr=self.learning_rate,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay
                                )
        if self.opt_type == 'Adam':
            if is_sparse:
                return torch.optim.SparseAdam(params,
                                        lr=self.learning_rate
                                    )
            else:
                return torch.optim.Adam(params,
                                    lr=self.learning_rate,
                                    weight_decay=self.weight_decay
                                )


    def construct(self, model,args):
        """
            Get optimizer.
            Args:
                model: torch.nn.Module: network
                params: : parameters
                freeze_embeddings: boolean: specify if embeddings need to be trained
            Returns:
                optimizer: torch.optim: optimizer as per given specifications  
        """
        model_params, is_sparse = self.get_params(model,args)
        for _, item in enumerate(zip(model_params, is_sparse)):
            if item[0]:
                self.optimizer.append(self._get_opt(params=item[0], is_sparse=item[1]))
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

    def get_params(self, net, args):
        self.net_params = {}
        self.net_params['sparse']= list()
        self.net_params['dense']= list()
        self.net_params['no_grad'] = list()
        if self.freeze_embeddings:
            for params in net.embeddings.parameters():
                params.requires_grad = False
        lrs = args.lrs
        module_dict = net.__dict__['_modules']
        
        for key, val in module_dict.items():
            is_sparse = val.__dict__.get("sparse", False)
            lr = lrs.get(key,args.learning_rate)
            for params in val.parameters():
                if params.requires_grad:
                    if is_sparse:
                        self.net_params['sparse'].append({"params":params, "lr":lr})
                    else:
                        self.net_params['dense'].append({"params":params, "lr":lr})
                else:
                    self.net_params['no_grad'].append(params)
        return [self.net_params['sparse'], self.net_params['dense']], [True, False]
        
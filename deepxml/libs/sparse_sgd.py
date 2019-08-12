import torch


class SparseSGD(torch.optim.Optimizer):
    r"""Implements sparse version of SGD algorithm suitable for sparse tensors.

    In this variant, only moments that show up in the gradient get updated, and
    only those portions of the gradient get applied to the parameters.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        momentum (float, optional): coefficients used for computing
            running averages of gradients (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
    """

    def __init__(self, params, lr=1e-3, momentum=0.9):
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        defaults = dict(lr=lr, momentum=momentum)
        super(SparseSGD, self).__init__(params, defaults)


    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                momentum = group['momentum']
                if not grad.is_sparse:
                    raise RuntimeError('SparseSGD does not support dense gradients, please consider SGD instead')
                if momentum != 0:
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p.data)

                    state['step'] += 1

                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)

                    exp_avg = state['exp_avg']

                    # Decay the first moment running average coefficient
                    #      old <- b * old + (1 - b) * new
                    # <==> old += (1 - b) * (new - old)
                    old_exp_avg_values = exp_avg.sparse_mask(grad)._values()
                    exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(1 - momentum)
                    exp_avg.add_(make_sparse(exp_avg_update_values))
                    update = exp_avg_update_values.add_(old_exp_avg_values)
                    del exp_avg_update_values

                    step_size = group['lr'] 
                    p.data.add_(make_sparse(-step_size * update))
                else:
                    p.data.add_(-step_size * grad)

        return loss

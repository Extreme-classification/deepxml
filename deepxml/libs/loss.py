import torch
import torch.nn.functional as F


class _Loss(torch.nn.Module):
    def __init__(self, size_average=None, reduce=None,
                 reduction='mean'):
        super(_Loss, self).__init__()
        self.reduce = reduce
        if size_average is not None or reduce is not None:
            self.reduction = torch.nn._reduction.legacy_get_string(
                size_average, reduce)
        else:
            self.reduction = reduction

    def _reduce(self, loss):
        if not self.reduce:
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


def _convert_labels_for_svm(y):
    """
        Convert labels from {0, 1} to {-1, 1}
    """
    return 2.*y - 1.0


class HingeLoss(_Loss):
    """criterion for loss function
       y: 0/1 ground truth matrix of size: batch_size x output_size
       f: real number pred matrix of size: batch_size x output_size
    """

    def __init__(self, margin=1.0, size_average=None, reduce=True,
                 reduction='mean'):
        super(HingeLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input, target):
        loss = F.relu(self.margin - _convert_labels_for_svm(target)*input)
        return self._reduce(loss)


class SquaredHingeLoss(_Loss):
    """criterion for loss function
       y: 0/1 ground truth matrix of size: batch_size x output_size
       f: real number pred matrix of size: batch_size x output_size
    """

    def __init__(self, margin=1.0, size_average=None, reduce=True,
                 reduction='mean'):
        super(SquaredHingeLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input, target):
        loss = F.relu(self.margin - _convert_labels_for_svm(target)*input)
        loss = loss**2
        return self._reduce(loss)


class CosineEmbeddingLoss(_Loss):
    """criterion for loss function
       y: 0/1 ground truth matrix of size: batch_size x output_size
       f: real number pred matrix of size: batch_size x output_size
    """

    def __init__(self, margin=0.0, size_average=None, reduce=True,
                 reduction='mean'):
        super(CosineEmbeddingLoss, self).__init__(
            size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input, target):
        input_0, input_1 = input
        sim = F.cosine_similarity(input_0, input_1, dim=2)
        loss = torch.where(target == 1, 1-sim,
                           torch.max(
                               torch.zeros_like(sim), sim - self.margin))
        return self._reduce(loss)


class BCEWithLogitsLoss(_Loss):
    """ BCE loss (expects logits; numercial stable)
       target: 0/1 ground truth matrix of size: batch_size x output_size
       input: real number pred matrix of size: batch_size x output_size
    """
    __constants__ = ['weight', 'pos_weight', 'reduction']

    def __init__(self, weight=None, size_average=None, reduce=None,
                 reduction='mean', pos_weight=None):
        super(BCEWithLogitsLoss, self).__init__(
            size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input, target):
        return F.binary_cross_entropy_with_logits(input, target,
                                                  self.weight,
                                                  pos_weight=self.pos_weight,
                                                  reduction=self.reduction)

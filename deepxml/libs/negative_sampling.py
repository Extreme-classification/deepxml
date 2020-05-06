import numpy as np
import _pickle as pickle
from functools import partial


class NegativeSamplerBase(object):
    """Base class for negative sampling
    Parameters:
    ----------
    num_samples: int
        sample spce
    num_negatives: int
        #samples
    """
    def __init__(self, num_labels, num_negatives):
        self.num_labels = num_labels
        self.num_negatives = num_negatives
        self.index = None
        self._construct()

    def _construct(self):
        """Create a partial function with given parameters
        Index should take one argument i.e. size during querying
        """
        self.index = partial(np.random.randint, low=0, high=self.num_labels)

    def _query(self):
        """Query for one sample
        """
        return (self.index(size=self.num_negatives), [1.0]*self.num_negatives)

    def query(self, num_samples, *args, **kwargs):
        """Query shortlist for one or more samples
        """
        if num_samples == 1:
            return self._query()
        else:
            out = [self._query() for _ in range(num_samples)]
            return out

    def save(self, fname):
        """
            Save object
        """
        state = self.__dict__
        pickle.dump(state, open(fname, 'wb'))

    def load(self, fname):
        """ Load object
        """
        self = pickle.load(open(fname, 'rb'))

    @property
    def data_init(self):
        return True if self.index is not None else False


class NegativeSampler(NegativeSamplerBase):
    """Negative sampler with support for sampling from
        multinomial distribution
    Parameters:
    ----------
    num_samples: int
        sample spce
    num_negatives: int
        #samples
    probs: np.ndarray or None, optional, default=None
        probability of each item
    replace: boolean, optional, default=False
        with or without replacement
    """
    def __init__(self, num_labels, num_negatives, prob=None, replace=False):
        self.prob = prob
        self.replace = replace
        super().__init__(num_labels, num_negatives)

    def _construct(self):
        self.index = partial(
            np.random.default_rng().choice, a=self.num_labels,
            replace=self.replace, p=self.prob)

import numpy as np
from .dist_utils import Partitioner
import os
from .sampling import NegativeSampler
from scipy.sparse import load_npz
from xclib.utils import sparse as sp
from xclib.utils.matrix import SMatrix


def construct_handler(shortlist_type, num_instances, num_labels,
                      model_dir='', mode='train', size_shortlist=-1,
                      label_mapping=None, in_memory=True,
                      shorty=None, fname=None, corruption=200,
                      num_clf_partitions=1):
    if shortlist_type == 'static':
        return ShortlistHandlerStatic(
            num_instances, num_labels, model_dir, num_clf_partitions, mode,
            size_shortlist, in_memory, label_mapping, fname)
    elif shortlist_type == 'hybrid':
        return ShortlistHandlerHybrid(
            num_instances, num_labels, model_dir, num_clf_partitions, mode,
            size_shortlist, in_memory, label_mapping, corruption)
    elif shortlist_type == 'dynamic':
        return ShortlistHandlerDynamic(
            num_labels, shorty, model_dir,
            mode, num_clf_partitions, size_shortlist, label_mapping)
    else:
        raise NotImplementedError(
            "Unknown shortlist method: {}!".format(shortlist_type))


class ShortlistHandlerBase(object):
    """Base class for ShortlistHandler
    - support for partitioned classifier

    Arguments
    ----------
    num_labels: int
        number of labels
    shortlist:
        shortlist object
    model_dir: str, optional, default=''
        save the data in model_dir
    num_clf_partitions: int, optional, default=''
        #classifier splits
    mode: str: optional, default=''
        mode i.e. train or test or val
    size_shortlist:int, optional, default=-1
        get shortlist of this size
    label_mapping: None or dict: optional, default=None
        map labels as per this mapping
    """

    def __init__(self, num_labels, shortlist, model_dir='',
                 num_clf_partitions=1, mode='train', size_shortlist=-1,
                 label_mapping=None, max_pos=20):
        self.model_dir = model_dir
        self.num_clf_partitions = num_clf_partitions
        self.size_shortlist = size_shortlist
        self.mode = mode
        self.max_pos = max_pos
        self.num_labels = num_labels
        self.label_mapping = label_mapping
        # self._create_shortlist(shortlist)
        self._create_partitioner()
        self.label_padding_index = self.num_labels
        if self.num_clf_partitions > 1:
            self.label_padding_index = self.partitioner.get_padding_indices()

    def _create_shortlist(self, shortlist):
        """
            Create structure to hold shortlist
        """
        self.shortlist = shortlist

    def query(self, *args, **kwargs):
        return self.shortlist(*args, **kwargs)

    def _create_partitioner(self):
        """
            Create partiotionar to for splitted classifier
        """
        self.partitioner = None
        if self.num_clf_partitions > 1:
            if self.mode == 'train':
                self.partitioner = Partitioner(
                    self.num_labels, self.num_clf_partitions,
                    padding=False, contiguous=True)
                self.partitioner.save(os.path.join(
                    self.model_dir, 'partitionar.pkl'))
            else:
                self.partitioner = Partitioner(
                    self.num_labels, self.num_clf_partitions,
                    padding=False, contiguous=True)
                self.partitioner.load(os.path.join(
                    self.model_dir, 'partitionar.pkl'))

    def _adjust_shortlist(self, pos_labels, shortlist, sim):
        """
            Adjust shortlist for a instance
            Training: Add positive labels to the shortlist
            Inference: Return shortlist with label mask
        """
        if self.mode == 'train':
            _target = np.zeros(self.size_shortlist, dtype=np.float32)
            _sim = np.zeros(self.size_shortlist, dtype=np.float32)
            _shortlist = np.full(
                self.size_shortlist, fill_value=self.label_padding_index,
                dtype=np.int64)
            # TODO: Adjust sim as well
            if len(pos_labels) > self.max_pos:
                pos_labels = np.random.choice(
                    pos_labels, size=self.max_pos, replace=False)
            neg_labels = shortlist[~np.isin(shortlist, pos_labels)]
            _target[:len(pos_labels)] = 1.0
            #  #TODO not used during training; not perfect values
            _sim[:len(pos_labels)] = 1.0
            _short = np.concatenate([pos_labels, neg_labels])
            temp = min(len(_short), self.size_shortlist)
            _shortlist[:temp] = _short[:temp]
        else:
            _target = np.zeros(self.size_shortlist, dtype=np.float32)
            _shortlist = np.full(
                self.size_shortlist, fill_value=self.label_padding_index,
                dtype=np.int64)
            _shortlist[:len(shortlist)] = shortlist
            _target[np.isin(shortlist, pos_labels)] = 1.0
            _sim = np.zeros(self.size_shortlist, dtype=np.float32)
            _sim[:len(shortlist)] = sim
        return _shortlist, _target, _sim

    def _get_sl_one(self, index, pos_labels):
        shortlist, sim = self.query(index)
        shortlist, target, sim = self._adjust_shortlist(
            pos_labels, shortlist, sim)
        mask = shortlist != self.label_padding_index
        return shortlist, target, sim, mask

    def _get_sl_partitioned(self, index, pos_labels):
        # Partition labels
        pos_labels = self.partitioner.split_indices(pos_labels)
        if self.shortlist.data_init:  # Shortlist is initialized
            _shortlist, _sim = self.query(index)
            shortlist, target, sim, mask, rev_map = [], [], [], [], []
            # Get shortlist for each classifier
            for idx in range(self.num_clf_partitions):
                __shortlist, __target, __sim, __mask = self._adjust_shortlist(
                    pos_labels[idx],
                    _shortlist[idx],
                    _sim[idx])
                shortlist.append(__shortlist)
                target.append(__target)
                sim.append(__sim)
                mask.append(__mask)
                rev_map.append(
                    self.partitioner.map_to_original(__shortlist, idx))
            rev_map = np.concatenate(rev_map)
        else:  # Shortlist is un-initialized
            shortlist = [np.zeros(self.size_shortlist)]*self.num_clf_partitions
            target = [np.zeros(self.size_shortlist)]*self.num_clf_partitions
            sim = [np.zeros(self.size_shortlist)]*self.num_clf_partitions
            mask = [np.zeros(self.size_shortlist)]*self.num_clf_partitions
            rev_map = np.zeros(self.size_shortlist*self.num_clf_partitions)
        return shortlist, target, sim, mask, rev_map

    def get_shortlist(self, index, pos_labels=None):
        """
            Get data with shortlist for given data index
        """
        if self.num_clf_partitions > 1:
            return self._get_sl_partitioned(index, pos_labels)
        else:
            return self._get_sl_one(index, pos_labels)

    def get_partition_indices(self, index):
        return self.partitioner.get_indices(index)


class ShortlistHandlerStatic(ShortlistHandlerBase):
    """ShortlistHandler with static shortlist
    - save/load/update/process shortlist
    - support for partitioned classifier

    Arguments
    ----------
    num_labels: int
        number of labels
    model_dir: str, optional, default=''
        save the data in model_dir
    num_clf_partitions: int, optional, default=''
        #classifier splits
    mode: str: optional, default=''
        mode i.e. train or test or val
    size_shortlist:int, optional, default=-1
        get shortlist of this size
    in_memory: bool: optional, default=True
        Keep the shortlist in memory or on-disk
    label_mapping: None or dict: optional, default=None
        map labels as per this mapping
    """

    def __init__(self, num_instances, num_labels, model_dir='', 
                 num_clf_partitions=1, mode='train', size_shortlist=-1,
                 in_memory=True, label_mapping=None, fname=None):
        super().__init__(num_labels, None, model_dir, num_clf_partitions,
                         mode, size_shortlist, label_mapping)
        self.in_memory = in_memory
        self._create_shortlist(num_instances, num_labels, size_shortlist)
        if fname is not None:
            self.from_pretrained(fname)

    def from_pretrained(self, fname):
        """
            Load label shortlist and similarity for each instance
        """
        shortlist = load_npz(fname)
        _ind, _sim = sp.topk(shortlist,
                             self.size_shortlist, self.num_labels,
                             -1000, return_values=True)
        self.update_shortlist(_ind, _sim)

    def query(self, index):
        ind, sim = self.shortlist[index]
        return ind, sim

    def _create_shortlist(self, num_instances, num_labels, k):
        """
            Create structure to hold shortlist
        """
        _type = 'memory' if self.in_memory else 'memmap'
        if self.num_clf_partitions > 1:
            raise NotImplementedError()
        else:
            self.shortlist = SMatrix(num_instances, num_labels, k)

    def update_shortlist(self, ind, sim, fname='tmp'):
        """
            Update label shortlist for each instance
        """
        self.shortlist.update(ind, sim)
        del sim, ind

    def save_shortlist(self, fname):
        """
            Save label shortlist and similarity for each instance
        """
        raise NotImplementedError()

    def load_shortlist(self, fname):
        """
            Load label shortlist and similarity for each instance
        """
        raise NotImplementedError()


class ShortlistHandlerDynamic(ShortlistHandlerBase):
    """ShortlistHandler with dynamic shortlist

    Arguments
    ----------
    num_labels: int
        number of labels
    shortlist:
        shortlist object like negative sampler
    model_dir: str, optional, default=''
        save the data in model_dir
    mode: str: optional, default=''
        mode i.e. train or test or val
    size_shortlist:int, optional, default=-1
        get shortlist of this size
    label_mapping: None or dict: optional, default=None
        map labels as per this mapping
    """

    def __init__(self, num_labels, shortlist, model_dir='',
                 num_clf_partitions=1, mode='train',
                 size_shortlist=-1, label_mapping=None):
        super().__init__(
            num_labels, shortlist, model_dir, num_clf_partitions,
            mode, size_shortlist, label_mapping)
        self._create_shortlist(shortlist)

    def query(self, num_instances=1, ind=None):
        return self.shortlist.query(
            num_instances=num_instances, ind=ind)


class ShortlistHandlerHybrid(ShortlistHandlerBase):
    """ShortlistHandler with hybrid shortlist
    - save/load/update/process shortlist
    - support for partitioned classifier

    Arguments
    ----------
    num_labels: int
        number of labels
    model_dir: str, optional, default=''
        save the data in model_dir
    num_clf_partitions: int, optional, default=''
        #classifier splits
    mode: str: optional, default=''
        mode i.e. train or test or val
    size_shortlist:int, optional, default=-1
        get shortlist of this size
    in_memory: bool: optional, default=True
        Keep the shortlist in memory or on-disk
    label_mapping: None or dict: optional, default=None
        map labels as per this mapping
    _corruption: int, optional, default=None
        add these many random labels
    """

    def __init__(self, num_instances, num_labels, model_dir='',
                 num_clf_partitions=1, mode='train', size_shortlist=-1,
                 in_memory=True, label_mapping=None, _corruption=200):
        super().__init__(num_labels, None, model_dir, num_clf_partitions,
                         mode, size_shortlist, label_mapping)
        self.in_memory = in_memory
        self._create_shortlist(num_instances, num_labels, size_shortlist)
        self.shortlist_dynamic = NegativeSampler(num_labels, _corruption+20)
        self.size_shortlist = size_shortlist+_corruption  # Both

    def query(self, index):
        ind, sim = self.shortlist[index]
        _ind, _sim = self.shortlist_dynamic.query(1)
        ind = np.concatenate([ind, _ind])
        sim = np.concatenate([sim, _sim])
        return ind, sim

    def _create_shortlist(self, num_instances, num_labels, k):
        """
            Create structure to hold shortlist
        """
        _type = 'memory' if self.in_memory else 'memmap'
        if self.num_clf_partitions > 1:
            raise NotImplementedError()
        else:
            self.shortlist = SMatrix(num_instances, num_labels, k)

    def update_shortlist(self, ind, sim, fname='tmp'):
        """
            Update label shortlist for each instance
        """
        self.shortlist.update(ind, sim)
        del sim, ind

    def save_shortlist(self, fname):
        """
            Save label shortlist and similarity for each instance
        """
        raise NotImplementedError()

    def load_shortlist(self, fname):
        """
            Load label shortlist and similarity for each instance
        """
        raise NotImplementedError()

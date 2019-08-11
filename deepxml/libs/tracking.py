"""
    Tracking object; Maintain history of loss; accuracy etc.
"""

import _pickle as pickle


class Tracking(object):
    def __init__(self):
        self.checkpoint_history = 3
        self.mean_train_loss = []
        self.mean_val_loss = []
        self.saved_models = []
        self.val_precision = []
        self.val_ndcg = []
        self.train_time = 0
        self.validation_time = 0
        self.shortlist_time = 0
        self.saved_checkpoints = []
        self.last_saved_epoch = -1
        self.last_epoch = 0

    def save(self, fname):
        pickle.dump(self, open(fname, 'wb'))

    def load(self, fname):
        self = pickle.load(open(fname, 'rb'))

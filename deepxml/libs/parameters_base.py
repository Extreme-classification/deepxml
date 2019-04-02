import argparse
import json

__author__='X'

class ParametersBase():
    """
        Base class for parameters in XML
    """
    def __init__(self, description):
        self.parser = argparse.ArgumentParser(description)
        self.params = None

    def _construct(self):
        self.parser.add_argument(
            '--dataset',
            dest='dataset',
            action='store',
            type=str,
            help='dataset name')
        self.parser.add_argument(
            '--data_dir',
            dest='data_dir',
            action='store',
            type=str,
            help='path to main data directory')
        self.parser.add_argument(
            '--model_dir',
            dest='model_dir',
            action='store',
            type=str,
            help='directory to store models')
        self.parser.add_argument(
            '--result_dir',
            dest='result_dir',
            action='store',
            type=str,
            help='directory to store results')
        self.parser.add_argument(
            '--model_fname',
            dest='model_fname',
            default='model',
            action='store',
            type=str,
            help='model file name')
        self.parser.add_argument(
            '--pred_fname',
            dest='pred_fname',
            default='predictions.npy',
            action='store',
            type=str,
            help='prediction file name')
        self.parser.add_argument(
            '--tr_fname',
            dest='tr_fname',
            default='train.txt',
            action='store',
            type=str,
            help='training file name')
        self.parser.add_argument(
            '--val_fname',
            dest='val_fname',
            default='val.txt',
            action='store',
            type=str,
            help='validation file name')
        self.parser.add_argument(
            '--ts_fname',
            dest='ts_fname',
            default='test.txt',
            action='store',
            type=str,
            help='test file name')

    def parse_args(self):
        self.params = self.parser.parse_args()

    def load(self, fname):
        vars(self.params).update(json.load(open(fname)))

    def save(self, fname):
        print(vars(self.params))
        json.dump(vars(self.params), open(fname, 'w'), indent=4)


# Example to evaluate
import sys
import xctools.evaluation.xc_metrics as xc_metrics
import xctools.data.data_utils as data_utils
from scipy.io import loadmat
import numpy as np


def main(targets_file, train_file, predictions_file, A, B):
    # Load the dataset
    _, te_labels, _, _, te_num_labels = data_utils.read_data(targets_file)
    true_labels = data_utils.binarize_labels(te_labels, te_num_labels)
    _, tr_labels, _, _, tr_num_labels = data_utils.read_data(train_file)
    trn_labels = data_utils.binarize_labels(tr_labels, tr_num_labels)

    predicted_labels = data_utils.read_sparse_file(predictions_file)
    inv_propen = xc_metrics.compute_inv_propesity(trn_labels, A, B)
    acc = xc_metrics.Metrices(true_labels, inv_propensity_scores=inv_propen, remove_invalid=False)
    args = acc.eval(predicted_labels, 5)
    print(xc_metrics.format(*args))

if __name__ == '__main__':
    train_file = sys.argv[1]
    targets_file = sys.argv[2]  # Usually test data file
    predictions_file = sys.argv[3]  # In mat format
    A = float(sys.argv[4])
    B = float(sys.argv[5])
    main(targets_file, train_file, predictions_file, A, B)

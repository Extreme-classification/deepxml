from scipy.sparse import load_npz
from functools import reduce
import sys
import xclib.evaluation.xc_metrics as xc_metrics
import xclib.data.data_utils as data_utils
from scipy.sparse import load_npz, save_npz
import numpy as np
import os


def read_files(fnames):
    output = []
    for fname in fnames:
        output.append(load_npz(fname))
    return output


def merge(predictions):
    return reduce(lambda a, b: a+b, predictions)


def main(targets_label_file, train_label_file, prediction_files,
         A, B, save):
    prediction_files = prediction_files.rstrip(",").split(",")
    true_labels = data_utils.read_sparse_file(targets_label_file)
    trn_labels = data_utils.read_sparse_file(train_label_file)
    inv_propen = xc_metrics.compute_inv_propesity(trn_labels, A, B)
    acc = xc_metrics.Metrics(true_labels, inv_psp=inv_propen)
    root = os.path.dirname(prediction_files[0])
    predicted_labels = read_files(prediction_files)
    predicted_labels.append(merge(predicted_labels))
    for pred in predicted_labels:
        args = acc.eval(pred, 5)
        print(xc_metrics.format(*args)+"\n")
        if save:
            print("Saving predictions..")
            fname = os.path.join(root, "score.npz")
            save_npz(fname, pred, compressed=False)


if __name__ == '__main__':
    trn_label_file = sys.argv[1]
    target_file = sys.argv[2]
    pred_files = sys.argv[3]
    A = float(sys.argv[4])
    B = float(sys.argv[5])
    save = int(sys.argv[6])
    main(target_file, trn_label_file, pred_files, A, B, save)

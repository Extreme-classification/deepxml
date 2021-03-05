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


def main(tst_label_fname, trn_label_fname, pred_fname,
         A, B, save, *args, **kwargs):
    true_labels = data_utils.read_sparse_file(tst_label_fname)
    trn_labels = data_utils.read_sparse_file(trn_label_fname)
    inv_propen = xc_metrics.compute_inv_propesity(trn_labels, A, B)
    acc = xc_metrics.Metrics(true_labels, inv_psp=inv_propen)
    root = os.path.dirname(pred_fname[-1])
    predicted_labels = read_files(pred_fname)
    ens_predicted_labels = merge(predicted_labels)
    ans = ""
    for idx, pred in enumerate(predicted_labels):
        args = acc.eval(pred, 5)
        ans = ans + f"learner: {idx}\n{xc_metrics.format(*args)}\n"
    args = acc.eval(ens_predicted_labels, 5)
    ans = ans + f"Ensemble\n{xc_metrics.format(*args)}"
    if save:
        print("Saving predictions..")
        fname = os.path.join(root, "score.npz")
        save_npz(fname, ens_predicted_labels, compressed=False)
    line = "-"*30
    print(f"\n{line}\n{ans}\n{line}")
    return ans


if __name__ == '__main__':
    trn_label_fname = sys.argv[1]
    tst_label_fname = sys.argv[2]
    pred_fname = sys.argv[3].rstrip(",").split(",")
    A = float(sys.argv[4])
    B = float(sys.argv[5])
    save = int(sys.argv[6])
    main(tst_label_fname, trn_label_fname, pred_fname, A, B, save)

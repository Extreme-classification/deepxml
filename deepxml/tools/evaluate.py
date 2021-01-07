import sys
import xclib.evaluation.xc_metrics as xc_metrics
import xclib.data.data_utils as data_utils
from scipy.sparse import load_npz, save_npz
import numpy as np
import os
from xclib.utils.sparse import sigmoid, normalize


def main(tst_label_fname, trn_label_fname, pred_fname,
         A, B, betas, save):
    true_labels = data_utils.read_sparse_file(tst_label_fname)
    trn_labels = data_utils.read_sparse_file(trn_label_fname)
    inv_propen = xc_metrics.compute_inv_propesity(trn_labels, A, B)
    acc = xc_metrics.Metrics(true_labels, inv_psp=inv_propen)
    root = os.path.dirname(pred_fname)
    ans = ""
    if isinstance(betas, list) and betas[0] != -1:
        knn = load_npz(pred_fname+'_knn.npz')
        clf = load_npz(pred_fname+'_clf.npz')
        args = acc.eval(clf, 5)
        ans = f"classifier\n{xc_metrics.format(*args)}"
        args = acc.eval(knn, 5)
        ans = ans + f"\nshortlist\n{xc_metrics.format(*args)}"
        clf = normalize(sigmoid(clf), norm='max')
        knn = normalize(sigmoid(knn), norm='max')
        for beta in betas:
            predicted_labels = beta*clf + (1-beta)*knn
            args = acc.eval(predicted_labels, 5)
            ans = ans + f"\nbeta: {beta:.2f}\n{xc_metrics.format(*args)}"
            if save:
                fname = os.path.join(root, f"score_{beta:.2f}.npz")
                save_npz(fname, predicted_labels, compressed=False)
    else:
        predicted_labels = sigmoid(load_npz(pred_fname+'.npz'))
        args = acc.eval(predicted_labels, 5)
        ans = xc_metrics.format(*args)
        if save:
            print("Saving predictions..")
            fname = os.path.join(root, "score.npz")
            save_npz(fname, predicted_labels, compressed=False)
    line = "-"*30
    print(f"\n{line}\n{ans}\n{line}")
    return ans


if __name__ == '__main__':
    trn_label_file = sys.argv[1]
    targets_file = sys.argv[2]
    pred_fname = sys.argv[3]
    A = float(sys.argv[4])
    B = float(sys.argv[5])
    save = int(sys.argv[6])
    betas = list(map(float, sys.argv[7:]))
    main(targets_file, trn_label_file, pred_fname, A, B, betas, save)

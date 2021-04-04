import sys
import xclib.evaluation.xc_metrics as xc_metrics
import xclib.data.data_utils as data_utils
from scipy.sparse import load_npz, save_npz
import numpy as np
import os
from xclib.utils.sparse import sigmoid, normalize, retain_topk


def get_filter_map(fname):
    if fname is not None:
        mapping = np.loadtxt(fname).astype(np.int)
        if mapping.size != 0:
            return mapping
    return None


def filter_predictions(pred, mapping):
    if mapping is not None and len(mapping) > 0:
        print("Filtering labels.")
        pred[mapping[:, 0], mapping[:, 1]] = 0
        pred.eliminate_zeros()
    return pred


def main(tst_label_fname, trn_label_fname, filter_fname, pred_fname,
         A, B, betas, top_k, save):
    true_labels = data_utils.read_sparse_file(tst_label_fname)
    trn_labels = data_utils.read_sparse_file(trn_label_fname)
    inv_propen = xc_metrics.compute_inv_propesity(trn_labels, A, B)
    mapping = get_filter_map(filter_fname)
    acc = xc_metrics.Metrics(true_labels, inv_psp=inv_propen)
    root = os.path.dirname(pred_fname)
    ans = ""
    if isinstance(betas, list) and betas[0] != -1:
        knn = filter_predictions(
            load_npz(pred_fname+'_knn.npz'), mapping)
        clf = filter_predictions(
            load_npz(pred_fname+'_clf.npz'), mapping)
        args = acc.eval(clf, 5)
        ans = f"classifier\n{xc_metrics.format(*args)}"
        args = acc.eval(knn, 5)
        ans = ans + f"\nshortlist\n{xc_metrics.format(*args)}"
        clf = retain_topk(clf, k=top_k)
        knn = retain_topk(knn, k=top_k)
        clf = normalize(sigmoid(clf), norm='max')
        knn = normalize(sigmoid(knn), norm='max')
        for beta in betas:
            predicted_labels = beta*clf + (1-beta)*knn
            args = acc.eval(predicted_labels, 5)
            ans = ans + f"\nbeta: {beta:.2f}\n{xc_metrics.format(*args)}"
            if save:
                fname = os.path.join(root, f"score_{beta:.2f}.npz")
                save_npz(fname, retain_topk(predicted_labels, k=top_k),
                    compressed=False)
    else:
        predicted_labels = filter_predictions(
            sigmoid(load_npz(pred_fname+'.npz')), mapping)
        args = acc.eval(predicted_labels, 5)
        ans = xc_metrics.format(*args)
        if save:
            print("Saving predictions..")
            fname = os.path.join(root, "score.npz")
            save_npz(fname, retain_topk(predicted_labels, k=top_k),
                compressed=False)
    line = "-"*30
    print(f"\n{line}\n{ans}\n{line}")
    return ans


if __name__ == '__main__':
    trn_label_file = sys.argv[1]
    targets_file = sys.argv[2]
    filter_map = sys.argv[3]
    pred_fname = sys.argv[4]
    A = float(sys.argv[5])
    B = float(sys.argv[6])
    save = int(sys.argv[7])
    top_k = int(sys.argv[8])
    betas = list(map(float, sys.argv[9:]))
    main(targets_file, trn_label_file, filter_map, pred_fname, A, B, betas, top_k, save)

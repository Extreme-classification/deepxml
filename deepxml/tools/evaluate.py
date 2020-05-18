import sys
import xclib.evaluation.xc_metrics as xc_metrics
import xclib.data.data_utils as data_utils
from scipy.sparse import load_npz, save_npz
import numpy as np
import os
from xclib.utils.sparse import sigmoid, normalize


def main(targets_label_file, train_label_file, predictions_file,
         A, B, betas, save):
    true_labels = data_utils.read_sparse_file(targets_label_file)
    trn_labels = data_utils.read_sparse_file(train_label_file)
    inv_propen = xc_metrics.compute_inv_propesity(trn_labels, A, B)
    acc = xc_metrics.Metrics(true_labels, inv_psp=inv_propen)
    root = os.path.dirname(predictions_file)
    if betas[0] != -1:
        knn = load_npz(predictions_file+'_knn.npz')
        clf = load_npz(predictions_file+'_clf.npz')
        print("clf")
        args = acc.eval(clf, 5)
        print(xc_metrics.format(*args))
        print("knn")
        args = acc.eval(knn, 5)
        print(xc_metrics.format(*args))

        clf = normalize(sigmoid(clf), norm='max')
        knn = normalize(sigmoid(knn), norm='max')

        for beta in betas:
            predicted_labels = beta*clf + (1-beta)*knn
            args = acc.eval(predicted_labels, 5)
            print("beta: {0:.2f}".format(beta))
            print(xc_metrics.format(*args))
            if save:
                print("Saving predictions..")
                fname = os.path.join(
                    root, "score_beta_{0:.2f}.npz".format(beta))
                save_npz(fname, predicted_labels, compressed=False)
    else:
        predicted_labels = load_npz(predictions_file+'.npz')
        args = acc.eval(predicted_labels, 5)
        print(xc_metrics.format(*args))
        if save:
            print("Saving predictions..")
            fname = os.path.join(root, "score.npz")
            save_npz(fname, predicted_labels, compressed=False)


if __name__ == '__main__':
    trn_label_file = sys.argv[1]
    targets_file = sys.argv[2]
    pred_file = sys.argv[3]
    A = float(sys.argv[4])
    B = float(sys.argv[5])
    save = int(sys.argv[6])
    betas = list(map(float, sys.argv[7:]))
    main(targets_file, trn_label_file, pred_file, A, B, betas, save)

# Example to evaluate
import sys
import xclib.evaluation.xc_metrics as xc_metrics
import xclib.data.data_utils as data_utils
from scipy.sparse import lil_matrix, load_npz, csr_matrix, diags
import numpy as np
import copy
import os
import torch
from xclib.utils.sparse import sigmoid


def normalize(mat):
    mat.__dict__['data'] = np.exp(mat.__dict__['data'])
    _max = mat.max(axis=1).toarray().ravel()
    _max[_max == 0] = 1.0
    _norm = diags(1.0/_max)
    return _norm.dot(mat).tocsr()


def main(targets_label_file, train_label_file, predictions_file,
         A, B, betas, _type, save):
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

        if _type == 1:
            clf = sigmoid(clf)
            knn = sigmoid(knn)
        elif _type == 2:
            clf = normalize(clf)
            knn = normalize(knn)
        for beta in betas:
            predicted_labels = beta*clf + (1-beta)*knn
            args = acc.eval(predicted_labels, 5)
            print("beta %f, Method %d" % (beta, _type))
            print(xc_metrics.format(*args))
            if save:
                print("Saving predictions..")
                fname = os.path.join(
                    root, "score_beta_{0:.2f}_type_{1}.txt".format(beta, _type))
                data_utils.write_sparse_file(
                    predicted_labels, fname)
    else:
        predicted_labels = load_npz(predictions_file+'.npz')
        args = acc.eval(predicted_labels, 5)
        print(xc_metrics.format(*args))


if __name__ == '__main__':
    trn_label_file = sys.argv[1]
    targets_file = sys.argv[2]
    pred_file = sys.argv[3]
    A = float(sys.argv[4])
    B = float(sys.argv[5])
    _type = int(sys.argv[6])
    save = int(sys.argv[7])

    betas = list(map(float, sys.argv[8:]))
    main(targets_file, trn_label_file, pred_file, A, B, betas, _type, save)

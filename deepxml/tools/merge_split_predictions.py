from scipy.sparse import lil_matrix, csr_matrix, hstack, load_npz, save_npz
from xclib.data import data_utils
from scipy.io import loadmat, savemat
import pickle
import numpy as np
import os
import sys

class CombineResults(object):
    def __init__(self,ftype):
        self.label_mapping = [] # Mapping from new to original in each decile
        self.ftype = ftype

    def read_predictions(self, fname):
        if self.ftype == 'mat':
            return loadmat(fname)['predicted_labels']
        elif self.ftype == 'txt':
            return data_utils.read_sparse_file(fname)
        elif self.ftype == 'npz':
            return load_npz(fname)

    def write_predictions(self,file,fname):
        print("Saving at %s"%(fname))
        if self.ftype == 'mat':
            savemat(fname,{'predicted_labels':file})
        elif self.ftype == 'txt':
            data_utils.write_sparse_file(file,fname)
        elif self.ftype == 'npz':
            save_npz(fname, file)

    def read_mapping(self, fname, _set):
        self.label_mapping.append(np.loadtxt(fname, dtype=np.int32))

    def map_to_original(self, mat, label_map, n_cols):
        n_rows = mat.shape[0]
        row_idx, col_idx = mat.nonzero()
        vals = np.array(mat[row_idx, col_idx]).squeeze()
        col_indices = list(map(lambda x:label_map[x], col_idx))
        return csr_matrix((vals, (row_idx, np.array(col_indices))), shape=(n_rows, n_cols))
            
    def combine(self, num_labels, fname_predictions, fname_mapping, _set='test', ftype='npz'):
        combined_pred = None
        for idx, fname in enumerate(zip(fname_predictions, fname_mapping)):
            self.read_mapping(fname[1], _set)
            pred = self.read_predictions(fname[0])
            pred_mapped = self.map_to_original(pred, self.label_mapping[idx], num_labels)
            if combined_pred is None:
                combined_pred = pred_mapped
            else:
                combined_pred = combined_pred + pred_mapped
        return combined_pred



def main():
    result_dir = sys.argv[1]
    splits = sys.argv[2].split(',')
    suffix_predictions = sys.argv[3]
    mapping_dir = sys.argv[4]
    num_labels = int(sys.argv[5])
    cr = CombineResults('npz')
    fname_predictions = list(map(lambda x: os.path.join(result_dir, x, suffix_predictions), splits))
    fname_mapping = list(map(lambda x: os.path.join(mapping_dir, "labels_split_%s.txt"%(x)), splits))
    predicted_labels = cr.combine(num_labels, fname_predictions, fname_mapping)
    cr.write_predictions(predicted_labels, os.path.join(result_dir, suffix_predictions))
    print("Results combined!")



if __name__ == "__main__":
    main()

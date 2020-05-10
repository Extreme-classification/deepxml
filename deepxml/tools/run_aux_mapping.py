import xclib.data.data_utils as data_utils
import sys
import os
from aux_mapping import AuxMapping
import numpy as np
import _pickle as pickle
import json


def main():
    train_feat_fname = sys.argv[1]
    train_label_fname = sys.argv[2]
    method = int(sys.argv[3])
    split_threshold = int(sys.argv[4])
    tmp_dir = sys.argv[5]
    features = data_utils.read_sparse_file(train_feat_fname)
    labels = data_utils.read_sparse_file(train_label_fname)
    assert features.shape[0] == labels.shape[0], \
        "Number of instances must be same in features and labels"
    num_features = features.shape[1]
    num_labels = labels.shape[1]
    stats_obj = {}
    stats_obj['threshold'] = split_threshold
    stats_obj['method'] = method

    sd = AuxMapping(method=method, threshold=split_threshold)
    sd.fit(features, labels)
    stats_obj['aux'] = "{},{},{}".format(
        num_features, sd.num_aux_labels, sd.num_aux_labels)
    stats_obj['org'] = "{},{},{}".format(
        num_features, sd.num_labels, len(sd.valid_labels))

    json.dump(stats_obj, open(
        os.path.join(tmp_dir, "aux_stats.json"), 'w'), indent=4)

    np.savetxt(os.path.join(tmp_dir, "valid_labels.txt"),
               sd.valid_labels, fmt='%d')
    np.savetxt(os.path.join(tmp_dir, "aux_mapping.txt"),
               sd.mapping, fmt='%d')


if __name__ == '__main__':
    main()

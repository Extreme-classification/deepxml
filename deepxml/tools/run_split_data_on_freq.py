import xclib.data.data_utils as data_utils
import sys
import os
from split_data_on_labels import splitData
import numpy as np
import _pickle as pickle
import json

def main():
    data_dir = sys.argv[1]
    train_fname = sys.argv[2]
    split_threshold = list(map(int, sys.argv[3].split(",")))
    num_splits = len(split_threshold)+1
    tr_features, tr_labels, _, num_features, num_labels = data_utils.read_data(os.path.join(data_dir, train_fname))
    stats_obj = {'header': 'num_features,num_labels'}
    stats_obj['threshold'] = ",".join(map(str, split_threshold))
    sd = splitData(split_method=0, num_splits=num_splits, threshold=split_threshold)
    sd.fit(tr_features, tr_labels)
    total_n_valid_labels = 0
    for idx in range(num_splits):
        stats_obj[idx] = "{},{},{}".format(sd.features_split[idx].size, sd.labels_split[idx].size, sd.num_valid_labels[idx])
        total_n_valid_labels += sd.num_valid_labels[idx]
        np.savetxt(os.path.join(data_dir, 'features_split_{}.txt'.format(str(idx))), sd.features_split[idx], fmt='%d')
        np.savetxt(os.path.join(data_dir, 'labels_split_{}.txt'.format(str(idx))), sd.labels_split[idx], fmt='%d')        
    stats_obj['-1']="{},{},{}".format(num_features, num_labels, total_n_valid_labels)
    pickle.dump(sd, open(os.path.join(data_dir, "split_obj.pkl"), 'wb'))
    json.dump(stats_obj, open(os.path.join(data_dir, "split_stats.json"), 'w'), indent=4)

if __name__ == '__main__':
    main()
    
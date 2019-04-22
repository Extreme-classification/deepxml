import xclib.data.data_utils as data_utils
import sys
import numpy as np


def main():
    features = data_utils.read_sparse_file(sys.argv[1])
    print("Features loaded!")
    labels = data_utils.read_sparse_file(sys.argv[2])
    print("Labels loaded!")
    assert features.shape[0] == labels.shape[0]
    print("Features and labels shape: ", features.shape, labels.shape)
    data_utils.write_data(sys.argv[3], features.astype(np.float32), labels.astype(np.float32))


if __name__ == "__main__":
    main()

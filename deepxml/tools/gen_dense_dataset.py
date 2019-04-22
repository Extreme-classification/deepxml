import numpy as np
import sys
import xclib.data.data_utils as data_utils
import numpy as np

def _get_header(num_samples, embedding_dim, num_labels):
    """
        Header for XC format
    """
    return str(num_samples)+ " " + str(embedding_dim) + " " + str(num_labels)

def main():
    data_fname = sys.argv[1]
    embeddings_fname = sys.argv[2]
    out_file = sys.argv[3]
    _, labels, _, _, num_labels = data_utils.read_data(data_fname)
    labels = data_utils.binarize_labels(labels, num_labels)
    document_embeddings = np.load(embeddings_fname).astype(np.float32)
    data_utils.write_data(filename=out_file, features=document_embeddings, labels=labels)

if __name__ == '__main__':
    main()
